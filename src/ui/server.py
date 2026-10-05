"""
FastAPI Backend Server for Brain-Steered Hearing Aid Clinical Software Dashboard.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager
from src.ui.data_provider import StreamDataProvider
from src.ui.session_manager import StreamingSimulationSession

@asynccontextmanager
async def lifespan(app: FastAPI):
    global broadcast_task
    broadcast_task = asyncio.create_task(broadcast_loop())
    yield
    if broadcast_task:
        broadcast_task.cancel()

app = FastAPI(
    title="USCAPES Clinical Brain-Steered Hearing Aid Suite",
    version="2.4.0",
    lifespan=lifespan
)

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


# Enable 1ms high-precision multimedia timer on Windows
try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

# Active WebSocket Clients & Centralized Broadcaster
active_connections = set()
broadcast_task = None


async def broadcast_loop():
    """
    Centralized real-time clock and telemetry broadcaster with absolute deadline scheduling.
    Maintains exact 32.000 Hz real-time pacing with zero cumulative drift, eliminating
    audio starvation, buffer underruns, and hiccups.
    """
    start_time = None
    initial_tick = 0
    was_playing = False

    while True:
        try:
            if session.is_playing and active_connections:
                now = time.perf_counter()
                if not was_playing or start_time is None:
                    start_time = now
                    initial_tick = session.current_tick
                    was_playing = True

                telemetry = session.step_simulation()
                if telemetry is None:
                    msg = json.dumps({"type": "end_of_trial"})
                    session.is_playing = False
                    was_playing = False
                else:
                    msg = json.dumps(telemetry)

                dead = set()
                for ws in list(active_connections):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                for ws in dead:
                    active_connections.discard(ws)

                # Target timestamp relative to playback start
                target_time = start_time + (session.current_tick - initial_tick) * session.block_sec
                delay = target_time - time.perf_counter()
                if delay > 0.001:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)
            else:
                was_playing = False
                start_time = None
                await asyncio.sleep(0.02)
        except Exception as e:
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.02)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Bidirectional streaming WebSocket.
    Clients receive broadcasted telemetry frames and send control actions.
    """
    global broadcast_task
    if broadcast_task is None or broadcast_task.done():
        broadcast_task = asyncio.create_task(broadcast_loop())
    await websocket.accept()
    active_connections.add(websocket)
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
        pass
    except Exception:
        pass
    finally:
        active_connections.discard(websocket)
        if not active_connections:
            session.is_playing = False

