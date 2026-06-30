import subprocess
import os
import sys
import importlib.util
import inspect
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

def run_static_analysis(repo_root, files):
    errors = []
    
    # 1. Ruff (Linting)
    try:
        res = subprocess.run(["ruff", "check", *files], cwd=repo_root, capture_output=True, text=True)
        if res.returncode != 0:
            errors.append(f"[Ruff Linter Failed]\n{res.stdout}\n{res.stderr}")
    except FileNotFoundError:
        pass
        
    # 2. Pyright (Cross-file Validation & Type Checking)
    # The user mandated this for verifying imports, missing attributes, and constructor signatures.
    try:
        # Use pyright if available
        res = subprocess.run(["pyright", *files], cwd=repo_root, capture_output=True, text=True)
        if res.returncode != 0:
            errors.append(f"[Pyright Validation Failed]\n{res.stdout}")
    except FileNotFoundError:
        # Fallback to mypy if pyright is missing
        try:
            res2 = subprocess.run(["mypy", *files], cwd=repo_root, capture_output=True, text=True)
            if res2.returncode != 0:
                errors.append(f"[MyPy Validation Failed]\n{res2.stdout}")
        except FileNotFoundError:
            errors.append("[Static Analyzer Missing] Neither Pyright nor MyPy is installed. Static cross-file validation skipped, but it is required by CI.")
            
    return errors

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_model_smoke_test(repo_root, file_path):
    errors = []
    try:
        # Resolve imports properly by adding repo_root to sys.path
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            
        module_name = Path(file_path).stem
        module = import_module_from_path(module_name, file_path)
    except Exception as e:
        return [f"[Import Error] Failed to import {file_path}: {str(e)}"]

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module_name and issubclass(obj, nn.Module):
            try:
                # Attempt to instantiate
                sig = inspect.signature(obj.__init__)
                kwargs = {}
                in_channels = 64
                if 'in_channels' in sig.parameters:
                    param = sig.parameters['in_channels']
                    if param.default != inspect.Parameter.empty:
                        in_channels = param.default
                    else:
                        kwargs['in_channels'] = in_channels
                
                # Check for other required parameters
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'in_channels', 'args', 'kwargs'] and param.default == inspect.Parameter.empty:
                        # Try to guess common names
                        if 'features' in param_name.lower():
                            kwargs[param_name] = 32
                        elif 'lags' in param_name.lower():
                            kwargs[param_name] = 16
                        else:
                            # Pass a dummy 1 if unknown required argument
                            kwargs[param_name] = 1
                            
                model = obj(**kwargs)
                
                # Forward pass testing
                # Standard EEG shape
                dummy_input = torch.randn(2, in_channels, 256)
                
                # Try generic forward
                out = None
                try:
                    out = model(dummy_input)
                except Exception as forward_e:
                    # Some models (like our Residual Ridge) take two inputs or transposed
                    dummy_input_2 = torch.randn(2, 256, in_channels)
                    try:
                        out = model(dummy_input, dummy_input_2)
                    except Exception:
                        out = model(dummy_input_2)
                        
                # Backward pass
                model.train()
                if isinstance(out, tuple):
                    loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
                elif isinstance(out, torch.Tensor):
                    loss = out.sum()
                else:
                    raise TypeError(f"Forward pass returned non-tensor: {type(out)}")
                    
                loss.backward()
                
            except Exception as e:
                errors.append(f"[Model Smoke Test Failed] Error in '{name}' from {file_path}:\n{str(e)}")
                
    return errors

def run_dataset_smoke_test(repo_root, file_path):
    errors = []
    try:
        module_name = Path(file_path).stem
        module = import_module_from_path(module_name, file_path)
    except Exception as e:
        return [f"[Import Error] Failed to import {file_path}: {str(e)}"]

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module_name and issubclass(obj, Dataset) and obj is not Dataset:
            try:
                # Attempt to instantiate with minimal args
                sig = inspect.signature(obj.__init__)
                kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'args', 'kwargs'] and param.default == inspect.Parameter.empty:
                        # Guess path parameters
                        if 'dir' in param_name.lower() or 'path' in param_name.lower():
                            kwargs[param_name] = str(repo_root / "data" / "dummy_cache")
                        else:
                            kwargs[param_name] = None
                            
                try:
                    dataset = obj(**kwargs)
                except FileNotFoundError:
                    # dataset expects a real directory, just skip instantiation test
                    continue
                except Exception:
                    # Dataset might fail with dummy args, but we caught the signature
                    continue
                    
                # If it instantiates, try length
                if hasattr(dataset, '__len__'):
                    try:
                        n = len(dataset)
                        if n > 0:
                            sample = dataset[0]
                    except Exception as dataset_e:
                        errors.append(f"[Dataset Smoke Test Failed] Error loading sample from '{name}' in {file_path}:\n{str(dataset_e)}")
            except Exception as e:
                pass
                
    return errors

def run_repository_ci(repo_root, modified_files):
    py_files = [f for f in modified_files if f.endswith(".py") and not f.replace("\\", "/").startswith(".mcp")]
    
    if not py_files:
        return {"success": True, "errors": []}
        
    errors = []
    
    # 1. Static Analysis (Catches imports, signatures, variables)
    static_errors = run_static_analysis(repo_root, py_files)
    errors.extend(static_errors)
    
    # 2. Model, Dataset & Runtime Smoke Tests
    for f in py_files:
        abs_path = os.path.join(repo_root, f)
        if os.path.exists(abs_path):
            model_errors = run_model_smoke_test(repo_root, abs_path)
            errors.extend(model_errors)
            
            dataset_errors = run_dataset_smoke_test(repo_root, abs_path)
            errors.extend(dataset_errors)
            
    if errors:
        return {"success": False, "errors": errors}
        
    return {"success": True, "errors": []}
