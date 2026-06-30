import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory of core to path so we can import from server
core_dir = Path(__file__).resolve().parent
parent_dir = core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def gather_context(repo_root: Path, query: str) -> Dict:
    import server  # Dynamically import server to reuse functions
    
    # 1. Summary
    summary_str = server.project_summary()
    
    # 2. File tree
    file_tree = server.get_file_tree()
    
    # 3. Stats
    stats = server.repository_statistics()
    
    # 4. Search related files, datasets, training pipelines based on keywords in query
    related_files = []
    datasets = []
    pipelines = []
    
    # Parse query to look for specific datasets (e.g. kul, dtu)
    words = query.lower().split()
    for word in words:
        if len(word) > 2:
            files_matching = server.search_project(word)
            if "No matching files found" not in files_matching:
                related_files.extend(files_matching.split('\n'))
            
            # Dataset check
            dataset_info = server.find_dataset(word)
            if f"No files found for dataset" not in dataset_info:
                datasets.append(dataset_info)
                
            # Pipeline check
            pipeline_info = server.find_training_pipeline(word)
            if "No training pipelines found" not in pipeline_info:
                pipelines.append(pipeline_info)
                
    # Deduplicate related files
    related_files = list(set(related_files))
    
    return {
        "repository_summary": summary_str,
        "file_tree": file_tree,
        "statistics": stats,
        "related_files": related_files[:10],  # cap to top 10
        "datasets": datasets,
        "pipelines": pipelines
    }
