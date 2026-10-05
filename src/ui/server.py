"""
FastAPI Backend Server for Brain-Steered Hearing Aid Clinical Software Dashboard.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.ui.data_provider import StreamDataProvider
from src.ui.session_manager import StreamingSimulationSession

app = FastAPI(title="NeuroSteer Clinical Brain-Steered Hearing Aid Suite", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Shared Data Provider & Session
data_provider = StreamDataProvider()
session = StreamingSimulationSession(data_provider=data_provider)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Clinical Dashboard UI Initializing...</h1>")


@app.get("/api/subjects")
async def get_subjects():
    """Returns list of 18 subjects with empirical benchmark statistics."""
    return {"status": "ok", "subjects": data_provider.get_subjects_list()}


@app.get("/api/trials/{subject_id}")
async def get_trials(subject_id: str):
    """Returns list of 57 held-out trials for the specified subject."""
    return {"status": "ok", "subject_id": subject_id, "trials": data_provider.get_trials_for_subject(subject_id)}


@app.get("/api/device_status")
async def get_device_status():
    """Detects ReSpeaker USB circular mic array hardware connection."""
    respeaker_connected = data_provider.respeaker.is_hardware_connected
    return {
        "status": "ok",
        "respeaker_connected": respeaker_connected,
        "mic_channels": 4 if respeaker_connected else 0,
        "mode": "Physical Hardware (ReSpeaker)" if respeaker_connected else "Dataset Simulation (BioSemi ActiveTwo)"
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    High-frequency bidirectional streaming WebSocket.
    Dispatches 31.25 ms telemetry packets and handles interactive commands.
    """
    await websocket.accept()

    async def receive_loop():
        try:
            while True:
                raw_text = await websocket.receive_text()
                try:
                    data = json.loads(raw_text)
                    resp = session.handle_client_message(data)
                    await websocket.send_text(json.dumps({"type": "response", **resp}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except WebSocketDisconnect:
            session.is_playing = False
        except Exception:
            session.is_playing = False

    async def send_loop():
        try:
            while True:
                if session.is_playing:
                    t_start = asyncio.get_event_loop().time()
                    telemetry = session.step_simulation()
                    if telemetry is None:
                        await websocket.send_text(json.dumps({"type": "end_of_trial"}))
                        session.is_playing = False
                    else:
                        await websocket.send_text(json.dumps(telemetry))
                    
                    # 1.0x Real-Time Lockstep pacing (31.25 ms frame time)
                    elapsed = asyncio.get_event_loop().time() - t_start
                    sleep_time = max(0.001, session.block_sec - elapsed)
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            session.is_playing = False
        except Exception:
            session.is_playing = False

    recv_task = asyncio.create_task(receive_loop())
    send_task = asyncio.create_task(send_loop())
    
    done, pending = await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
