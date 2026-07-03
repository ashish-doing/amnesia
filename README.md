# Amnesia — an embodied agent that stops re-exploring the same house every time it wakes up

Built for "The Hangover Part AI: Where's My Context?" (WeMakeDevs x Cognee, June 29 – July 5, 2026).
Open Source Cognee track.

## What this is

A task-planning agent that operates in a small simulated house (symbolic, not physics — rooms,
containers, objects, an action API). It has to find things and complete chores. The only thing
that makes it better across sessions is what Cognee's memory lifecycle lets it remember.

Session 1 (cold start): the agent explores blind, opens wrong things, fails a few times, eventually
finishes a task in ~40 actions.

Session 2 (same house, brand-new process, only Cognee memory persists): finishes the same task in
~6 actions because it recalls where things are.

Session 3 (deliberate world drift — an object gets moved): the agent's cached belief is wrong, it
fails once, corrects itself via `forget()` + `remember()`, and adjusts confidence via `improve()`.

That before/after/drift sequence is the whole demo.

## AI-assistant disclosure

Built with AI-assistant help (Claude) for architecture planning, code scaffolding, and this README.
**Disclose this explicitly in your submission** — non-disclosure is grounds for disqualification
per the hackathon rules. Don't forget this on the day you submit.

## Build order (do NOT build top to bottom of the folder — build in this order)

1. **`house_sim/`** — the simulated world. Build and test this completely standalone, with
   `agent/fallback_planner.py` (deterministic, no LLM, no Cognee) driving it. If this loop doesn't
   work with zero AI involved, nothing above it will work either. Run `python scripts/run_session.py --fallback-only`
   to confirm.
2. **`memory/`** — wire in Cognee's `remember()`/`recall()`. Run session 1 → session 2 with only
   these two calls and confirm the action-count drop is real (log it, don't eyeball it).
3. **`agent/planner.py` + `agent/graph.py`** — swap the fallback planner for the real LLM planner
   (Groq), with the structured-output guardrails already built in.
4. **`memory/memory_ops.py`'s `improve()`/`forget()` path + drift scenario** in `house_sim/scenarios.py`.
5. **`server/` + `frontend/`** — FastAPI, WebSocket, the live map. Build this last — it's the least
   judged part relative to effort. A clean terminal log with the numbers is an acceptable fallback
   if you run out of time.

## Free-provider setup (this is genuinely $0, but only if you set this BEFORE writing agent code)

Cognee defaults to `openai/gpt-5-mini` if you don't override the provider — its `remember()` call
uses an LLM internally to extract entities/relations for the graph, so skipping this config means
an accidental OpenAI bill even though you never call OpenAI directly.

Copy `.env.example` to `.env` and fill in:

- `LLM_PROVIDER=groq` / `LLM_MODEL=llama-3.3-70b-versatile` / `LLM_API_KEY=<your Groq key>` —
  you already have this key from LexScout/CircuitForge, reuse it.
- `EMBEDDING_PROVIDER=fastembed` (local, free, no key needed).
- Leave `VECTOR_DB_PROVIDER` / `GRAPH_DATABASE_PROVIDER` unset to use Cognee's zero-setup default
  (SQLite + LanceDB + Ladybug, all local files, no server).

If you want a literal-zero-network-call path instead (e.g. offline dev on a train), set
`LLM_PROVIDER=ollama` and run a local model — slower/weaker reasoning, but fully offline.

**Verify exact param names against `docs.cognee.ai/setup-configuration/overview` before you rely on
this file** — Cognee's config surface can shift between versions; `memory/cognee_config.py` has a
comment flagging exactly where to double check.

## Judging-criteria map (for your README/demo narration, not code)

- **Best Use of Cognee** — all four lifecycle calls are used because the task requires them, not
  decoratively: `recall()` before every plan, `remember()` after every action, `improve()` reweights
  confidence from outcomes, `forget()` fires specifically on the drift-correction path.
- **Technical Excellence** — deterministic sim + structured-output-constrained planner + graceful
  fallback to a rule-based planner if the LLM fails twice.
- **Presentation Quality** — the session 1 vs session 2 action-count numbers, shown live, are the
  headline. Don't bury them in a slide — show the counter ticking during the run.

## Honest weaknesses to have an answer ready for

- This is a **symbolic simulation, not a physical robot or even a physics sim** — say this proactively,
  don't let a judge "catch" it. Frame it as: the planning-and-memory architecture is designed to port
  to a real controller; scope for a 3-day window intentionally excluded physics/hardware risk.
- The LLM planner is the least deterministic part of the system. Guardrails (structured output,
  state validation, capped retries, deterministic fallback) exist specifically because of this —
  mention that this was a deliberate design decision, not an afterthought.
