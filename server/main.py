"""
FastAPI backend. Two things a judge can do:
  1. Watch a session run live over the WebSocket (/ws/session).
  2. Type a question into the "ask the house" box (/ask) and get an answer
     straight from recall() — this is the highest-impact-per-hour stretch
     feature: it turns a demo into something judges can poke at themselves.

Run: uvicorn server.main:app --reload --port 8060
(port 8060, not 8000 — keep clear of anything else already bound locally)
"""
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from house_sim.scenarios import build_house, apply_standard_drift, TASKS
from agent.graph import run_session
from memory import memory_ops
from memory.cognee_config import configure
import json as _json

app = FastAPI(title="Amnesia")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# One in-memory house persists across sessions for a given server process —
# this IS the point: the sim resets, Cognee's on-disk store does not.
_house = None
_session_counter = 0


class AskRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup():
    configure()


@app.get("/")
def health():
    return {"status": "ok", "service": "amnesia"}


@app.get("/memory/graph")
async def memory_graph():
    """Returns the ACTUAL contents of the confidence store - real facts the
    agent has genuinely remembered, with real confidence scores, not mock
    data. This is what the frontend's memory graph panel visualizes."""
    if memory_ops.CONFIDENCE_STORE.exists():
        store = _json.loads(memory_ops.CONFIDENCE_STORE.read_text())
    else:
        store = {}
    nodes = [
        {"id": fid, "text": meta["text"][:120], "confidence": meta["confidence"],
         "session": meta.get("last_confirmed_session", 0)}
        for fid, meta in store.items()
    ]
    return {"nodes": nodes, "count": len(nodes)}


@app.post("/ask")
async def ask_the_house(req: AskRequest):
    """Live judge-facing query box — answers straight from recall(), no agent loop."""
    answer = await memory_ops.recall_context(req.query, current_session=_session_counter)
    return {"query": req.query, "answer": answer}


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket):
    """
    Client sends: {"task": "make_coffee", "mode": "cold" | "memory" | "drift", "use_llm": true}
    Server streams: one JSON event per agent step (perceive/recall/plan/act/memory_correction/session_end).
    """
    global _house, _session_counter
    await websocket.accept()
    try:
        params = await websocket.receive_json()
        task_name = params.get("task", "make_coffee")
        mode = params.get("mode", "memory")
        use_llm = params.get("use_llm", True)

        if mode == "cold" or _house is None:
            _house = build_house()
        if mode == "drift":
            apply_standard_drift(_house)

        _session_counter += 1

        def emit(event: dict):
            asyncio.create_task(websocket.send_json(event))

        result = await run_session(_house, task_name, _session_counter, use_llm, emit=emit)
        await websocket.send_json({"type": "final_result", **result})

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass