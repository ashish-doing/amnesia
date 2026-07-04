"""
The LangGraph loop. Six conceptual stages, run in a cycle until the task
completes or a step cap is hit:

  perceive -> recall -> plan -> act -> remember -> (improve | forget) -> loop

`emit` is a callback (sync function taking a dict) used to stream events out to
whatever's watching — a plain print() for the CLI script, or a WebSocket send in
server/session_runner.py. Keeping this as an injected callback rather than a
hard FastAPI dependency means agent/graph.py has zero server-framework coupling
and can be run standalone in scripts/run_session.py.
"""
import asyncio
from typing import Callable, TypedDict, Optional

from house_sim.world import House
from house_sim.scenarios import TASKS
from agent.schemas import AgentAction
from agent.fallback_planner import FallbackPlanner
# LLMPlanner and memory_ops are imported lazily inside run_session() when
# use_llm=True. This keeps `--fallback-only` runnable with ONLY pydantic
# installed - no groq, no cognee - which is the whole point of the day-1
# build-order step in the README: prove the sim loop works before any AI
# dependency enters the picture at all.

MAX_STEPS = 60


class SessionResult(TypedDict):
    task: str
    session_number: int
    steps_taken: int
    completed: bool
    used_fallback_count: int


def _fact_id(task: str, obj_or_room: str) -> str:
    return f"{task}:{obj_or_room}"


async def run_session(
    house: House,
    task_name: str,
    session_number: int,
    use_llm: bool,
    emit: Optional[Callable[[dict], None]] = None,
) -> SessionResult:
    emit = emit or (lambda e: None)
    task = TASKS[task_name]
    llm_planner = None
    memory_ops = None
    PlannerExhausted = None
    if use_llm:
        from agent.planner import LLMPlanner, PlannerExhausted as _PlannerExhausted
        from memory import memory_ops as _memory_ops
        llm_planner = LLMPlanner()
        memory_ops = _memory_ops
        PlannerExhausted = _PlannerExhausted
    fallback = FallbackPlanner()
    used_fallback_count = 0

    # PERFORMANCE/COST NOTE: remember() triggers cognee.add()+cognify(), which
    # re-resolves the WHOLE accumulated graph and costs more every step as the
    # graph grows. Calling it once per action (as an earlier version of this
    # file did) burns through Groq's free-tier token budget in 2-3 sessions and
    # makes every action several sequential LLM round-trips. Instead we
    # accumulate observations locally (free, no network call) and write them
    # to Cognee ONCE at session end. Real memory consolidation happens at
    # natural checkpoints, not continuously - this is a deliberate design
    # choice, not a shortcut.
    session_observations: list[str] = []
    room_recall_cache: dict[str, str] = {}

    emit({"type": "session_start", "task": task_name, "session": session_number})

    for step in range(MAX_STEPS):
        # --- perceive ---
        p = house.perceive()
        perception_text = house.perceive_text()
        emit({"type": "perceive", "room": house.agent_room, "step": step,
              "visible_objects": p["visible_objects"], "connects_to": p["connects_to"]})

        # --- recall (cached per room within this session - a room's facts don't
        # change between two consecutive actions taken inside it, so re-querying
        # every step is pure waste) ---
        recall_query = f"{task_name} {house.agent_room} object locations and past outcomes"
        recall_ctx = ""
        if use_llm:
            if house.agent_room in room_recall_cache:
                recall_ctx = room_recall_cache[house.agent_room]
            else:
                recall_ctx = await memory_ops.recall_context(recall_query, current_session=session_number)
                room_recall_cache[house.agent_room] = recall_ctx
            emit({"type": "recall", "query": recall_query, "result": recall_ctx})

        # --- plan (with capped LLM retries falling back to the deterministic planner) ---
        action: AgentAction
        if use_llm:
            try:
                action = llm_planner.next_action(house, task["description"], recall_ctx)
            except PlannerExhausted as e:
                used_fallback_count += 1
                emit({"type": "planner_fallback", "reason": str(e)})
                action = fallback.next_action(house, task["description"])
        else:
            action = fallback.next_action(house, task["description"])

        emit({"type": "plan", "action": action.model_dump(), "step": step})

        # --- act ---
        if action.type == "done":
            emit({"type": "agent_declared_done", "step": step})
            break
        result = _execute(house, action)
        emit({"type": "act", "action": action.model_dump(), "result": result.to_dict(), "step": step})

        # --- remember (locally accumulated - no network call here) + local confidence bookkeeping ---
        if use_llm:
            fid = _fact_id(task_name, action.target or house.agent_room)
            observation = f"Session {session_number}, step {step}: {perception_text} " \
                          f"Action taken: {action.type} {action.target or ''}. Result: {result.message}"
            session_observations.append(observation)

            if not result.success and "isn't visible" in result.message.lower() or \
               (not result.success and action.type in ("move", "open", "pick")):
                # The agent's belief was wrong - this IS the improve()/forget() path,
                # and it stays per-step (not batched) since a correction needs to
                # happen right when the contradiction is discovered, not at session
                # end. This only touches the local confidence store + one best-effort
                # network call in forget_fact(), not a full cognify() pass, so it's
                # cheap even at per-step frequency.
                memory_ops.improve_from_outcome(fid, was_correct=False, session=session_number)
                await memory_ops.forget_fact(fid, reason=f"Action failed unexpectedly: {result.message}")
                emit({"type": "memory_correction", "fact_id": fid, "reason": result.message})
            elif result.success:
                memory_ops.improve_from_outcome(fid, was_correct=True, session=session_number)

        if task["success"](house):
            emit({"type": "task_success", "step": step})
            break

    completed = task["success"](house)

    if use_llm and session_observations:
        # The one and only cognify() call for this whole session. This is what
        # cuts LLM calls from ~1 per action to 1 per session.
        session_fid = f"{task_name}:session{session_number}"
        await memory_ops.remember_observation(
            "\n".join(session_observations),
            fact_id=session_fid,
            metadata={"session": session_number},
        )
        emit({"type": "session_remembered", "observation_count": len(session_observations)})

    emit({"type": "session_end", "steps_taken": house.step_count, "completed": completed})

    return SessionResult(
        task=task_name,
        session_number=session_number,
        steps_taken=house.step_count,
        completed=completed,
        used_fallback_count=used_fallback_count,
    )


def _execute(house: House, action: AgentAction):
    if action.type == "move":
        return house.move(action.target)
    if action.type == "open":
        return house.open(action.target)
    if action.type == "close":
        return house.close(action.target)
    if action.type == "pick":
        return house.pick(action.target)
    if action.type == "place":
        return house.place(action.target, action.destination or house.agent_room)
    if action.type == "use":
        return house.use(action.target)
    raise ValueError(f"Unknown action type: {action.type}")
