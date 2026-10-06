"""
USCAPES Clinical Suite — Backend Web & Streaming Server.

Provides FastAPI REST endpoints, WebSocket continuous telemetry streaming,
and real-time 16 kHz PCM audio transmission.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Set
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR
from .pipeline.session_manager import StreamingSimulationSession
from .pipeline.data_provider import StreamDataProvider
from .pipeline.calibrator import calibrate_subject

# Global instances
data_provider = StreamDataProvider()
session = StreamingSimulationSession(data_provider=data_provider)
connected_websockets: Set[WebSocket] = set()
broadcast_task: asyncio.Task = None


async def broadcast_loop():
    """Real-time 31.25 ms streaming broadcaster loop."""
    interval_sec = 0.03125
    next_tick_time = asyncio.get_event_loop().time()

    while True:
        try:
            if session.is_playing and connected_websockets:
                data = session.step()
                if data:
                    telemetry_json = json.dumps(data)
                    dead_sockets = set()

                    for ws in connected_websockets:
                        try:
                            await ws.send_text(telemetry_json)
                        except Exception:
                            dead_sockets.add(ws)

                    for dead in dead_sockets:
                        connected_websockets.discard(dead)

            next_tick_time += interval_sec
            sleep_duration = next_tick_time - asyncio.get_event_loop().time()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                next_tick_time = asyncio.get_event_loop().time()
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[USCAPES SERVER] Broadcast loop exception: {e}")
            await asyncio.sleep(0.01)


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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def get_index():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/api/subjects")
async def get_subjects():
    return data_provider.get_subjects_list()


@app.get("/api/trials/{subject_id}")
async def get_trials(subject_id: str):
    return data_provider.get_trials_for_subject(subject_id)


@app.get("/api/status")
async def get_status():
    return {
        "is_playing": session.is_playing,
        "subject": session.current_subject,
        "trial": session.current_trial,
        "is_calibrated": getattr(session, "is_calibrated", True),
        "listening_mode": session.listening_mode,
        "current_time": round(session.current_tick * (session.audio_block_smp / session.fs_audio), 2),
        "total_duration": session.total_duration_sec,
    }


@app.post("/api/calibrate/{subject_id}")
async def trigger_calibration(subject_id: str):
    """Triggers 3-trial fine-tuning calibration for target subject."""
    try:
        # Run calibration in thread pool so it does not block the asyncio event loop
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, calibrate_subject, subject_id)
        # Reload trial in session to apply new weights
        session.load_trial(session.current_subject, session.current_trial)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "init",
            "subject": session.current_subject,
            "trial": session.current_trial,
            "is_calibrated": getattr(session, "is_calibrated", True),
            "is_playing": session.is_playing,
            "listening_mode": session.listening_mode
        }))

        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)
            ack = session.handle_client_message(msg)

            # Acknowledge state change
            await websocket.send_text(json.dumps({
                "type": "state_ack",
                "is_playing": session.is_playing,
                "subject": session.current_subject,
                "trial": session.current_trial,
                "is_calibrated": getattr(session, "is_calibrated", True),
                "listening_mode": session.listening_mode,
                "ack": ack
            }))

    except WebSocketDisconnect:
        connected_websockets.discard(websocket)
    except Exception as e:
        print(f"[USCAPES WS] Error: {e}")
        connected_websockets.discard(websocket)
