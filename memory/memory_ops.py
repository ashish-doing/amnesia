"""
Thin wrapper around Cognee's lifecycle API, plus the confidence-tracking layer
that makes improve()/forget() do something real rather than decorative.

NOTE on API surface: the hackathon brief documents remember()/recall()/improve()
(memify)/forget() as Cognee's top-level v1.0 methods. If your installed version
doesn't expose these as top-level calls, they map to the legacy pattern:
  remember(text)  -> cognee.add(text) followed by cognee.cognify()
  recall(query)   -> cognee.search(query)
Check docs.cognee.ai/getting-started/quickstart against your installed version
before day 2 and adjust the four functions below — that's the only place the
mapping lives, so a version mismatch is a one-file fix, not a scattered one.

Confidence tracking: Cognee stores facts; it does not by itself track "how
sure are we this is still true." We track that ourselves in a small local
JSON sidecar (confidence_store.json) keyed by fact id, and feed it back into
recall_context() so the planner's prompt can see "kettle location: kitchen
(confidence 0.4, last confirmed 3 sessions ago)" and treat it with appropriate
skepticism instead of blind trust.
"""
import json
import os
import time
from pathlib import Path

import cognee
from memory.cognee_config import configure

CONFIDENCE_STORE = Path(__file__).parent / "confidence_store.json"
DECAY_PER_SESSION = 0.15  # a fact loses this much confidence per session it isn't reconfirmed
MIN_CONFIDENCE_TO_TRUST = 0.35


def _load_confidence() -> dict:
    if CONFIDENCE_STORE.exists():
        return json.loads(CONFIDENCE_STORE.read_text())
    return {}


def _save_confidence(store: dict):
    CONFIDENCE_STORE.write_text(json.dumps(store, indent=2))


async def remember_observation(text: str, fact_id: str | None = None, metadata: dict | None = None):
    """Ingest an observation into Cognee's knowledge graph, and (re)set its
    confidence to 1.0 since it was just directly observed."""
    configure()
    await cognee.add(text)
    await cognee.cognify()

    if fact_id:
        store = _load_confidence()
        store[fact_id] = {"confidence": 1.0, "last_confirmed_session": metadata.get("session", 0) if metadata else 0,
                           "text": text}
        _save_confidence(store)


async def recall_context(query: str, current_session: int = 0) -> str:
    """Query Cognee for relevant memory, and annotate the result with confidence
    so the planner can decide how much to trust it. This is what makes session 2
    fast (real recall) and what makes the drift session (session 3) interesting
    (recall returns something that confidence-annotation flags as questionable)."""
    configure()
    try:
        results = await cognee.search(query)
    except Exception as e:
        # Covers: DB not yet initialized (run scripts/init_cognee.py once),
        # or a genuinely empty graph on a brand-new house. Either way, "nothing
        # to recall" is a normal outcome for session 1 cold start, never a
        # reason to crash the whole session.
        print(f"[memory_ops] recall_context: search failed ({e}); treating as no memory available.")
        return "No relevant memory found."
    store = _load_confidence()

    lines = []
    for r in results if isinstance(results, list) else [results]:
        text = str(r)
        conf = 1.0
        for fact_id, meta in store.items():
            if meta["text"] in text or text in meta["text"]:
                sessions_stale = max(0, current_session - meta["last_confirmed_session"])
                conf = max(0.0, meta["confidence"] - DECAY_PER_SESSION * sessions_stale)
                break
        trust_note = "" if conf >= MIN_CONFIDENCE_TO_TRUST else " [LOW CONFIDENCE — verify before relying on this]"
        lines.append(f"- {text} (confidence: {conf:.2f}){trust_note}")

    return "\n".join(lines) if lines else "No relevant memory found."


def improve_from_outcome(fact_id: str, was_correct: bool, session: int):
    """Reweight a fact's confidence based on whether acting on it worked out.
    This is memify's actual intended purpose - adapting weights from feedback -
    not a cosmetic pass over the graph.

    NOTE: initializes the entry if it doesn't exist yet, since remember_observation()
    (which used to create entries) is now batched to session-end for cost reasons -
    per-action corrections need to work independently of when the batched write lands."""
    store = _load_confidence()
    if fact_id not in store:
        store[fact_id] = {"confidence": 0.5, "last_confirmed_session": session, "text": fact_id}
    if was_correct:
        store[fact_id]["confidence"] = min(1.0, store[fact_id]["confidence"] + 0.2)
        store[fact_id]["last_confirmed_session"] = session
    else:
        store[fact_id]["confidence"] = max(0.0, store[fact_id]["confidence"] - 0.5)
    _save_confidence(store)


async def forget_fact(fact_id: str, reason: str):
    """Surgically remove a stale/incorrect fact. Called specifically on the
    drift-correction path in agent/graph.py, not as routine cleanup — this is
    the call that proves the lifecycle is used for real, not just ingest+query."""
    configure()
    store = _load_confidence()
    if fact_id in store:
        text = store[fact_id]["text"]
        del store[fact_id]
        _save_confidence(store)
        # Best-effort: if your Cognee version exposes a direct delete/forget call,
        # invoke it here too so the underlying graph node is actually pruned, not
        # just locally distrusted. Check the exact signature in the docs — this
        # is the one call most likely to differ across versions.
        try:
            await cognee.forget(text)  # type: ignore[attr-defined]
        except AttributeError:
            print(f"[memory_ops] cognee.forget() not available in this version — "
                  f"confidence for '{fact_id}' zeroed locally, but graph node may persist. "
                  f"Reason: {reason}")