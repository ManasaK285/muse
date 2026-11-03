import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aegis import scenarios
from aegis.graph.mission import Toolkit, run_mission
from aegis.recorder import Recorder

WEB = Path(__file__).resolve().parents[2] / "web"
log = logging.getLogger("aegis.server")
app = FastAPI(title="Aegis-VLA Mission Control")
app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/scenarios")
def list_scenarios():
    return [{"key": s.key, "title": s.title, "instruction": s.instruction, "flaw": s.naive_flaw}
            for s in scenarios.SCENARIOS.values()]


@app.get("/api/health")
def health():
    tk = Toolkit(lambda *_: None)
    return JSONResponse({"ok": True, "backends": tk.backends(), "stack": tk.model_stack()})


@app.post("/api/transcribe")
async def transcribe_audio(request: Request):
    from aegis.models import voice

    audio = await request.body()
    if not audio:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    name = request.headers.get("x-filename", "command.wav")
    try:
        text, backend = await asyncio.to_thread(voice.transcribe, audio, name)
    except Exception as e:
        return JSONResponse({"error": str(e)[:400]}, status_code=400)
    return {"text": text, "backend": backend}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    loop = asyncio.get_running_loop()
    busy = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()

    async def pump():
        while True:
            msg = await queue.get()
            await socket.send_text(msg)

    pump_task = asyncio.create_task(pump())

    def start(key, chaos, instruction, redteam):
        rec = Recorder(key)

        def emit(kind, data):
            rec(kind, data)
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": kind, "data": data}, default=str))

        try:
            run_mission(key, chaos=chaos, emit=emit, instruction=instruction, redteam=redteam)
        except Exception as e:  # surface failures in the UI instead of a dead socket
            log.exception("mission failed")
            emit("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            rec.close()
            busy.clear()

    try:
        while True:
            msg = json.loads(await socket.receive_text())
            if msg.get("action") != "run":
                continue
            if busy.is_set():
                await socket.send_text(json.dumps({"type": "error", "data": {"message": "a mission is already running"}}))
                continue
            key = msg.get("scenario", "collision")
            if key not in scenarios.SCENARIOS:
                continue
            busy.set()
            instruction = (msg.get("instruction") or "").strip() or None
            threading.Thread(target=start, args=(key, bool(msg.get("chaos", True)), instruction,
                                                 bool(msg.get("redteam", False))), daemon=True).start()
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
