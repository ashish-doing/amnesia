"""
The agent loop, as a REAL LangGraph StateGraph — not a plain for-loop pretending
to be one. This replaces an earlier version whose docstring called itself "the
LangGraph loop" while never importing langgraph at all; that was a false claim,
caught in review, and this is the actual fix, not a renamed comment.

Six nodes, one conditional loop-back edge:

  perceive -> recall -> plan -> act -> correct -> [loop back to perceive, or -> finalize -> END]

`emit` is a callback (sync function taking a dict) used to stream events out to
whatever's watching. It's threaded through state rather than closed over, since
LangGraph node functions only receive state.
"""
import asyncio
from typing import Callable, Optional, TypedDict, Any

from langgraph.graph import StateGraph, END

from house_sim.world import House
from house_sim.scenarios import TASKS
from agent.schemas import AgentAction
from agent.fallback_planner import FallbackPlanner

MAX_STEPS = 60


class SessionResult(TypedDict):
    task: str
    session_number: int
    steps_taken: int
    completed: bool
    used_fallback_count: int


class AgentState(TypedDict):
    house: House
    task_name: str
    session_number: int
    use_llm: bool
    step: int
    emit: Callable[[dict], None]
    fallback: FallbackPlanner
    llm_planner: Optional[Any]
    memory_ops: Optional[Any]
    planner_exhausted_cls: Optional[Any]
    session_observations: list
    room_recall_cache: dict
    used_fallback_count: int
    current_perception_text: str
    current_recall_ctx: str
    current_action: Optional[AgentAction]
    current_result: Optional[Any]
    done: bool


def _fact_id(task: str, obj_or_room: str) -> str:
    return f"{task}:{obj_or_room}"


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


# ---------------- nodes ----------------

async def perceive_node(state: AgentState) -> AgentState:
    house = state["house"]
    p = house.perceive()
    state["current_perception_text"] = house.perceive_text()
    state["emit"]({"type": "perceive", "room": house.agent_room, "step": state["step"],
                    "visible_objects": p["visible_objects"], "connects_to": p["connects_to"],
                    "inventory": p["inventory"]})
    return state


async def recall_node(state: AgentState) -> AgentState:
    house = state["house"]
    if not state["use_llm"]:
        state["current_recall_ctx"] = ""
        return state
    recall_query = f"{state['task_name']} {house.agent_room} object locations and past outcomes"
    cache = state["room_recall_cache"]
    if house.agent_room in cache:
        ctx = cache[house.agent_room]
    else:
        ctx = await state["memory_ops"].recall_context(recall_query, current_session=state["session_number"])
        cache[house.agent_room] = ctx
    state["current_recall_ctx"] = ctx
    state["emit"]({"type": "recall", "query": recall_query, "result": ctx})
    return state


async def plan_node(state: AgentState) -> AgentState:
    house = state["house"]
    task = TASKS[state["task_name"]]
    if state["use_llm"]:
        recent_history = "\n".join(state["session_observations"][-5:])
        try:
            action = state["llm_planner"].next_action(
                house, task["description"], state["current_recall_ctx"],
                recent_actions=recent_history,
            )
        except state["planner_exhausted_cls"] as e:
            state["used_fallback_count"] += 1
            state["emit"]({"type": "planner_fallback", "reason": str(e)})
            action = state["fallback"].next_action(house, task["description"])
    else:
        action = state["fallback"].next_action(house, task["description"])
    state["current_action"] = action
    state["emit"]({"type": "plan", "action": action.model_dump(), "step": state["step"]})
    return state


async def act_node(state: AgentState) -> AgentState:
    house = state["house"]
    action = state["current_action"]
    if action.type == "done":
        state["done"] = True
        state["emit"]({"type": "agent_declared_done", "step": state["step"]})
        return state
    result = _execute(house, action)
    state["current_result"] = result
    state["emit"]({"type": "act", "action": action.model_dump(), "result": result.to_dict(), "step": state["step"]})
    return state


async def correct_node(state: AgentState) -> AgentState:
    """improve()/forget() live here - reweighting confidence and correcting stale
    facts based on whether the just-taken action worked out. Named 'correct' rather
    than 'remember' because the actual Cognee write happens once at session end in
    finalize_node, not here - this node is purely the confidence-adjustment step."""
    if state["done"]:
        return state

    if state["use_llm"]:
        house = state["house"]
        action = state["current_action"]
        result = state["current_result"]
        memory_ops = state["memory_ops"]

        fid = _fact_id(state["task_name"], action.target or house.agent_room)
        observation = (f"Session {state['session_number']}, step {state['step']}: "
                       f"{state['current_perception_text']} Action taken: {action.type} "
                       f"{action.target or ''}. Result: {result.message}")
        state["session_observations"].append(observation)

        if (not result.success and "isn't visible" in result.message.lower()) or \
           (not result.success and action.type in ("move", "open", "pick")):
            memory_ops.improve_from_outcome(fid, was_correct=False, session=state["session_number"])
            await memory_ops.forget_fact(fid, reason=f"Action failed unexpectedly: {result.message}")
            state["emit"]({"type": "memory_correction", "fact_id": fid, "reason": result.message})
        elif result.success:
            memory_ops.improve_from_outcome(fid, was_correct=True, session=state["session_number"])

    if TASKS[state["task_name"]]["success"](state["house"]):
        state["emit"]({"type": "task_success", "step": state["step"]})
        state["done"] = True

    state["step"] += 1
    return state


async def finalize_node(state: AgentState) -> AgentState:
    if state["use_llm"] and state["session_observations"]:
        session_fid = f"{state['task_name']}:session{state['session_number']}"
        await state["memory_ops"].remember_observation(
            "\n".join(state["session_observations"]),
            fact_id=session_fid,
            metadata={"session": state["session_number"]},
        )
        state["emit"]({"type": "session_remembered", "observation_count": len(state["session_observations"])})

    completed = TASKS[state["task_name"]]["success"](state["house"])
    state["emit"]({"type": "session_end", "steps_taken": state["house"].step_count, "completed": completed})
    return state


def _should_continue(state: AgentState) -> str:
    if state["done"] or state["step"] >= MAX_STEPS:
        return "finalize"
    return "perceive"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("perceive", perceive_node)
    graph.add_node("recall", recall_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("correct", correct_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "recall")
    graph.add_edge("recall", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "correct")
    graph.add_conditional_edges("correct", _should_continue, {"perceive": "perceive", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_session(
    house: House,
    task_name: str,
    session_number: int,
    use_llm: bool,
    emit: Optional[Callable[[dict], None]] = None,
) -> SessionResult:
    emit = emit or (lambda e: None)
    llm_planner = None
    memory_ops = None
    planner_exhausted_cls = None
    if use_llm:
        from agent.planner import LLMPlanner, PlannerExhausted
        from memory import memory_ops as _memory_ops
        llm_planner = LLMPlanner()
        memory_ops = _memory_ops
        planner_exhausted_cls = PlannerExhausted

    initial_state: AgentState = {
        "house": house, "task_name": task_name, "session_number": session_number,
        "use_llm": use_llm, "step": 0, "emit": emit, "fallback": FallbackPlanner(),
        "llm_planner": llm_planner, "memory_ops": memory_ops,
        "planner_exhausted_cls": planner_exhausted_cls,
        "session_observations": [], "room_recall_cache": {}, "used_fallback_count": 0,
        "current_perception_text": "", "current_recall_ctx": "", "current_action": None,
        "current_result": None, "done": False,
    }

    emit({"type": "session_start", "task": task_name, "session": session_number})
    graph = get_compiled_graph()
    final_state = await graph.ainvoke(initial_state, config={"recursion_limit": MAX_STEPS * 6 + 10})

    return SessionResult(
        task=task_name,
        session_number=session_number,
        steps_taken=house.step_count,
        completed=TASKS[task_name]["success"](house),
        used_fallback_count=final_state["used_fallback_count"],
    )
