"""Local server: streams captured frames over a WebSocket and serves the UI.

Run with:  uv run python -m galaxy.server
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .capture import Capture
from .generate import MAX_NEW_TOKENS, PromptError, generate
from .model import MAX_CONTEXT, load
from .protocol import encode_frame

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRONTEND_DIST = ROOT.parent / "frontend" / "dist"

state: dict = {}
run_lock = threading.Lock()  # one generation at a time; the hooks hold per-call state


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model, tokenizer = load()
    state["tokenizer"] = tokenizer
    state["capture"] = Capture(model)
    yield
    state["capture"].close()


app = FastAPI(lifespan=lifespan)


def _data_file(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(503, f"{name} is missing. Run: make precompute")
    return path


@app.get("/api/manifest")
def manifest():
    return json.loads(_data_file("manifest.json").read_text())


LAYOUT_FILES = {"writes": "layout.bin", "reads": "layout_reads.bin"}


@app.get("/api/layout")
def layout(by: str = "writes"):
    """Star positions. by=writes: UMAP of output weights; by=reads: of input weights."""
    if by not in LAYOUT_FILES:
        raise HTTPException(400, f"Unknown layout {by!r}. Use one of: {', '.join(LAYOUT_FILES)}")
    return FileResponse(_data_file(LAYOUT_FILES[by]), media_type="application/octet-stream")


def _clean_params(msg: dict) -> dict:
    return {
        "prompt": str(msg.get("prompt", "")),
        "max_new_tokens": max(0, min(int(msg.get("max_new_tokens", 40)), MAX_NEW_TOKENS)),
        "temperature": max(0.0, min(float(msg.get("temperature", 0.8)), 5.0)),
        "top_k": max(0, int(msg.get("top_k", 40))),
        "seed": int(msg.get("seed", 0)),
    }


async def _run(ws: WebSocket, params: dict, stop: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(item) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def worker() -> None:
        if not run_lock.acquire(blocking=False):
            put(("error", "Another run is in progress."))
            return
        try:
            count = 0
            for frame in generate(
                state["capture"], state["tokenizer"], should_stop=stop.is_set, **params
            ):
                put(("frame", encode_frame(frame)))
                count += 1
            put(("done", {"frames": count, "stopped": stop.is_set()}))
        except PromptError as e:
            put(("error", str(e)))
        except Exception as e:  # surface unexpected failures to the UI
            put(("error", f"{type(e).__name__}: {e}"))
        finally:
            run_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    # Lets the client lay out the token ring once instead of reflowing per token.
    n_prompt = len(state["tokenizer"](params["prompt"]).input_ids)
    await ws.send_json(
        {
            "type": "started",
            "params": params,
            "n_prompt": n_prompt,
            "expected_tokens": min(n_prompt + params["max_new_tokens"], MAX_CONTEXT),
        }
    )
    while True:
        kind, payload = await queue.get()
        if kind == "frame":
            await ws.send_bytes(payload)
        elif kind == "done":
            await ws.send_json({"type": "done", **payload})
            return
        else:
            await ws.send_json({"type": "error", "message": payload})
            return


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    stop = threading.Event()
    task: asyncio.Task | None = None
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "run":
                if task and not task.done():
                    await ws.send_json({"type": "error", "message": "A run is already in progress."})
                    continue
                try:
                    params = _clean_params(msg)
                except (TypeError, ValueError):
                    await ws.send_json({"type": "error", "message": "Invalid run settings."})
                    continue
                stop = threading.Event()
                task = asyncio.create_task(_run(ws, params, stop))
            elif kind == "stop":
                stop.set()
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        if task and not task.done():
            task.cancel()


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
