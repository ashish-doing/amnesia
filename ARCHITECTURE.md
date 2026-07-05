# Amnesia — Architecture

## Overview

Amnesia is an embodied task-planning agent operating in a simulated house. A real, compiled
LangGraph `StateGraph` drives a six-node loop (perceive to recall to plan to act to correct to
finalize) that plans actions via a Groq-backed LLM (with a deterministic fallback), reads and
writes long-term memory through Cognee's lifecycle API, and streams every step live to a
FastAPI + WebSocket frontend with a real-time D3 force-directed memory graph.

---

## System Diagram

```mermaid
flowchart TD
    subgraph SIM["Simulation Layer"]
        HOUSE["house_sim/world.py<br/>House, Room, GameObject<br/>Partial observability<br/>move, open, close, pick, place, use"]
        SCEN["house_sim/scenarios.py<br/>Fixed 4-room layout<br/>4 tasks: make_coffee, make_tea,<br/>find_keys, tidy_kitchen<br/>apply_standard_drift()"]
        SCEN --> HOUSE
    end

    subgraph GRAPH["Agent Layer - real LangGraph StateGraph"]
        PER["perceive_node<br/>Reads house.perceive()<br/>Emits room, objects, inventory"]
        REC["recall_node<br/>Queries Cognee, cached per room<br/>Skipped entirely in fallback mode"]
        PLAN["plan_node<br/>Groq structured tool-call<br/>Falls back on 2 failed retries"]
        ACT["act_node<br/>Executes against House<br/>Emits real ActionResult"]
        COR["correct_node<br/>improve/forget logic<br/>Accumulates session_observations<br/>Always advances step counter"]
        FIN["finalize_node<br/>One batched remember() call<br/>Emits session_end"]
        PER --> REC --> PLAN --> ACT --> COR
        COR -.not done, step less than 60.-> PER
        COR -.done or max steps.-> FIN
    end

    subgraph PLANNING["Planning"]
        LLM["agent/planner.py<br/>Groq llama-3.3-70b-versatile<br/>Forced tool-calling schema<br/>State validation before execution<br/>2 retries, then raises PlannerExhausted"]
        FALL["agent/fallback_planner.py<br/>Deterministic explorer<br/>No LLM, no network<br/>Unit-tested against a real house"]
        PLAN --> LLM
        LLM -.exhausted.-> FALL
        FALL --> PLAN
    end

    subgraph MEMORY["Memory Layer"]
        COGNEE["Cognee 1.2.2<br/>SQLite + LanceDB + Ladybug<br/>remember, recall, improve, forget"]
        CONF["memory/memory_ops.py<br/>Local confidence store<br/>JSON sidecar, cognee-free import<br/>Unit-tested: 7 passing"]
        EXTRACT["Groq llama-3.1-8b-instant<br/>Cognee's internal extraction<br/>Separate 500K/day rate-limit pool"]
        REC <-->|search| COGNEE
        FIN -->|add + cognify, ONCE per session| COGNEE
        COR --> CONF
        COGNEE -.entity extraction.-> EXTRACT
    end

    subgraph BACKEND["Backend"]
        API["server/main.py - FastAPI<br/>GET /<br/>GET /memory/graph<br/>POST /ask<br/>WS /ws/session"]
        QUEUE["Ordered event queue<br/>Single-consumer coroutine<br/>Guarantees in-order delivery<br/>fixes earlier fire-and-forget race"]
        API --> QUEUE
        CONF -.read by.-> API
    end

    subgraph FRONTEND["Frontend - frontend/index.html"]
        MAP["Live map<br/>Real room + object state<br/>Synced from perceive events"]
        GRAPHVIZ["D3 memory graph<br/>Force-directed, from /memory/graph<br/>Color = real confidence<br/>Edges cluster by session"]
        CHART["Session comparison chart<br/>Real steps_taken per session<br/>Graph-size growth over time"]
        ASK["Ask the house<br/>Live recall() query box"]
    end

    HOUSE --> PER
    QUEUE -->|WebSocket| MAP
    QUEUE --> CHART
    API -->|GET /memory/graph| GRAPHVIZ
    API -->|POST /ask| ASK
```

---

## Sequence - A Full Session (cold to memory)

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant WS as WebSocket (ordered queue)
    participant G as LangGraph StateGraph
    participant H as House (sim)
    participant P as Planner (Groq 70b)
    participant C as Cognee (recall/remember)

    FE->>WS: task=make_coffee, mode=cold
    WS->>G: run_session(house, task, session_number)
    loop until done or 60 steps
        G->>H: perceive()
        G->>WS: emit perceive event (real objects, inventory)
        G->>C: recall(room facts) - cached per room
        C-->>G: real text response
        G->>WS: emit recall event (verbatim)
        G->>P: next_action(house, task, recall_ctx)
        P-->>G: structured AgentAction (validated)
        G->>H: execute(action)
        H-->>G: ActionResult
        G->>WS: emit act event
        G->>G: improve_from_outcome() - local confidence
    end
    G->>C: remember(all session observations) - ONCE
    G->>WS: emit session_end
    WS-->>FE: all events, strictly in order
```

---

## Sequence - Drift Correction (session 3)

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant G as LangGraph StateGraph
    participant H as House (drifted)
    participant Mem as memory_ops

    Note over H: apply_standard_drift moved keys before this session started
    G->>H: perceive() - agent is in the room it remembers keys being in
    G->>Mem: recall(keys location) - returns STALE cached belief
    G->>H: execute(move/open toward remembered location)
    H-->>G: ActionResult success=False - object isn't there
    G->>Mem: improve_from_outcome(fid, was_correct=False)
    G->>Mem: forget_fact(fid) - removes local confidence entry, best-effort prunes Cognee graph node
    G->>FE: emit memory_correction event
    Note over G: Planner re-plans with corrected context on the next loop iteration
```

---

## Sequence - "Ask the House" (live query)

```mermaid
sequenceDiagram
    participant User
    participant FE as Frontend
    participant API as POST /ask
    participant Mem as memory_ops.recall_context()
    participant C as Cognee

    User->>FE: types "where's the mug?"
    FE->>API: POST /ask query
    API->>Mem: recall_context(query, current_session)
    Mem->>C: cognee.search(query)
    C-->>Mem: real graph-completion answer
    Mem-->>API: confidence-annotated text
    API-->>FE: answer
    FE-->>User: displayed verbatim, no agent loop involved
```

---

## Component Map

| File | Responsibility |
|---|---|
| `house_sim/world.py` | `House`, `Room`, `GameObject` dataclasses; the action API (move/open/close/pick/place/use); partial observability via `perceive()` |
| `house_sim/scenarios.py` | Fixed 4-room house layout; `TASKS` dict (make_coffee, make_tea, find_keys, tidy_kitchen); `apply_standard_drift()` |
| `agent/schemas.py` | `AgentAction` Pydantic model; `AGENT_ACTION_TOOL` forced tool-calling schema for Groq |
| `agent/fallback_planner.py` | Deterministic, no-LLM explorer - the safety net when the LLM planner fails twice |
| `agent/planner.py` | `LLMPlanner` - Groq structured tool-calling, pre-execution state validation, capped retries, raises `PlannerExhausted` |
| `agent/graph.py` | The real `langgraph.graph.StateGraph` - 6 nodes, 1 conditional loop-back edge, compiled once and reused |
| `memory/cognee_config.py` | Explicit provider configuration; refuses to run unconfigured rather than silently defaulting to a paid provider |
| `memory/memory_ops.py` | `remember_observation` / `recall_context` / `improve_from_outcome` / `forget_fact`; local confidence-store sidecar; lazy `cognee` import so the pure logic is testable without Cognee installed |
| `server/main.py` | FastAPI app; `GET /memory/graph`, `POST /ask`, `WS /ws/session` with a single-consumer ordered event queue |
| `frontend/index.html` | Live map (real object/room state), D3 force-directed memory graph, session comparison chart, ask-the-house box, optional voice narration, auto-demo sequencer |
| `scripts/run_session.py` | CLI runner - the day-1/day-2 test harness, works with zero AI dependencies via `--fallback-only` |
| `scripts/init_cognee.py` | One-time Cognee database schema initialization |
| `tests/test_world.py` | 9 tests - simulation correctness + fallback planner verification |
| `tests/test_memory_ops.py` | 7 tests - confidence-tracking logic, zero Cognee dependency |

---

## Why Two Different Groq Models

| Model | Used by | Rate limit (free tier) | Why |
|---|---|---|---|
| `llama-3.3-70b-versatile` | `agent/planner.py` | 100K tokens/day | Needs real reasoning for action planning; makes few, short calls per session |
| `llama-3.1-8b-instant` | Cognee's internal entity/relation extraction | 500K tokens/day, separate pool | Extraction doesn't need frontier reasoning but calls more often per `cognify()` pass |

Splitting these fixed a real rate-limit exhaustion bug hit during development - both workloads
sharing the 70B model's 100K/day pool burned the entire daily budget in 2-3 sessions.

---

## Why remember() Is Batched to Session-End

The first implementation called `remember_observation()` after every single action. `cognify()`
resolves the entire accumulated graph to text and runs LLM extraction on every call - so this
meant every action was several sequential LLM round-trips, and cost scaled with graph size,
getting slower every step. Batching to one call per session (in `finalize_node`) is the
architecturally correct fix: real memory consolidation happens at natural checkpoints, not
continuously. Found and fixed during development by observing rate-limit exhaustion in production
logs, not designed in from the start.

---

## Data Flow - Confidence and Correction

```mermaid
flowchart LR
    A["Action executes<br/>act_node"] --> B{"Result?"}
    B -->|success| C["improve_from_outcome<br/>was_correct=True<br/>confidence += 0.2, capped at 1.0"]
    B -->|unexpected failure| D["improve_from_outcome<br/>was_correct=False<br/>confidence -= 0.5, floored at 0.0"]
    D --> E["forget_fact()<br/>Local removal: guaranteed<br/>Cognee graph prune: best-effort,<br/>outcome returned as auditable dict"]
    C --> F["confidence_store.json"]
    E --> F
    F -->|GET /memory/graph| G["D3 memory graph<br/>node color = live confidence"]
```

---

## Tech Stack

| Layer | Technology | Version/Detail |
|---|---|---|
| Simulation | Pure Python dataclasses | No external dependencies |
| Orchestration | LangGraph | `StateGraph`, compiled once, reused across sessions |
| Planning (primary) | Groq `llama-3.3-70b-versatile` | Forced tool-calling via the `groq` Python SDK |
| Planning (fallback) | Hand-rolled deterministic explorer | Zero dependencies, unit-tested |
| Memory | Cognee | `1.2.2`, SQLite + LanceDB + Ladybug (zero-setup local stack) |
| Memory extraction | Groq `llama-3.1-8b-instant` | Routed via LiteLLM's `groq/` prefix, `LLM_PROVIDER=custom` |
| Embeddings | Fastembed | `BAAI/bge-small-en-v1.5`, 384 dimensions, local |
| Backend | FastAPI + WebSocket | Single-consumer ordered event queue |
| Frontend | Vanilla JS + D3.js | Force-directed simulation via `d3.forceSimulation` |
| Testing | Zero-dependency assert-based runners | `python tests/test_world.py`, no pytest required |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Must be `custom` (not `groq`) - Cognee routes Groq via LiteLLM's `groq/` model prefix |
| `LLM_MODEL` | `groq/llama-3.1-8b-instant` - Cognee's internal extraction model |
| `LLM_API_KEY` | Groq API key, used by Cognee internally |
| `EMBEDDING_PROVIDER` | `fastembed` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIMENSIONS` | `384` - must be set explicitly or Cognee guesses wrong for this model |
| `GROQ_API_KEY` | Used directly by `agent/planner.py`, separate from Cognee's internal calls |
| `PLANNER_MODEL` | `llama-3.3-70b-versatile` - the planner's model |
| `ENABLE_BACKEND_ACCESS_CONTROL` | `false` - disables Cognee's multi-user auth for local solo use |
| `CACHING` | `false` - disables Cognee's session-memory caching during development |

---

*Built for WeMakeDevs x Cognee - "The Hangover Part AI: Where's My Context?", June 29 - July 5, 2026*