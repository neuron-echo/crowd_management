"""FastAPI server — MJPEG streams + WebSocket push + static frontend."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from crowd_management.web.state import dashboard_state

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="Crowd Management Dashboard")

# ── Serve static frontend ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


# ── MJPEG helpers ─────────────────────────────────────────────────────────────

BLANK_FRAME: bytes = b""


def _blank_jpeg(w: int = 640, h: int = 360) -> bytes:
    import cv2, numpy as np
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "Initializing...", (w // 2 - 80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


async def _mjpeg_generator(camera_id: str, stream_type: str):
    """Yield MJPEG boundary frames at ~10fps."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    blank = _blank_jpeg()
    while True:
        cam = dashboard_state.cameras.get(camera_id)
        if cam is None:
            data = blank
        elif stream_type == "density":
            data = cam.density_jpeg if cam.density_jpeg else blank
        else:
            data = cam.frame_jpeg if cam.frame_jpeg else blank

        yield boundary + data + b"\r\n"
        await asyncio.sleep(0.1)  # 10 fps


@app.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    return StreamingResponse(
        _mjpeg_generator(camera_id, "frame"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/density/{camera_id}")
async def density_stream(camera_id: str):
    return StreamingResponse(
        _mjpeg_generator(camera_id, "density"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── REST snapshot ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return dashboard_state.snapshot()


# ── WebSocket broadcast ───────────────────────────────────────────────────────

_ws_clients: list[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(0.5)
            snap = dashboard_state.snapshot()
            # strip non-serialisable keys
            payload = json.dumps(snap, default=str)
            await ws.send_text(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
