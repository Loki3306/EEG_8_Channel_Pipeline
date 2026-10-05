"""
NeuroSteer Clinical Suite — Standalone Brain-Steered Hearing Aid Software Launcher.

Usage:
    python run_dashboard.py [--port 8000] [--no-browser]
"""

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add repository root to Python path
REPO_ROOT = Path(__file__).parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn


import socket


def open_browser_when_ready(url: str, host: str, port: int, max_wait_sec: float = 20.0):
    """Waits until the server is actively accepting connections before opening browser."""
    start = time.time()
    while time.time() - start < max_wait_sec:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.3)
    time.sleep(0.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="NeuroSteer Clinical Brain-Steered Hearing Aid Suite")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for UI development")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    print("=" * 80)
    print("  NEUROSTEER CLINICAL SUITE — BRAIN-STEERED HEARING AID SOFTWARE")
    print("=" * 80)
    print(f"  • Architecture:     CA-TCN Neural Decoder + Spatial Matrix Adapter")
    print(f"  • Grand Cohort:     18 Subjects (1,026 Trials Validated)")
    print(f"  • Acoustic Engine:  Causal IIR Filter + AudioSteeringDSP (+9 dB / -18 dB)")
    print(f"  • Headphone Audio:  Continuous Web Audio API Streaming")
    print(f"  • Server URL:       {url}")
    print("=" * 80)
    print(f"  Starting clinical dashboard server on {url} ...")

    if not args.no_browser:
        threading.Thread(target=open_browser_when_ready, args=(url, args.host, args.port), daemon=True).start()

    uvicorn.run("src.ui.server:app", host=args.host, port=args.port, reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
