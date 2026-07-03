"""
The real LLM planner. Structured-output guardrails live here:

1. Forced tool-calling against AGENT_ACTION_TOOL — the model cannot emit a
   malformed action shape, only a wrong *target* within a valid shape.
2. Every proposed action is validated against house.perceive() before it's
   returned — if the target isn't visible/reachable, we ask the model to
   retry with that specific error, capped at MAX_RETRIES.
3. If MAX_RETRIES is exceeded, planner.py raises PlannerExhausted and
   agent/graph.py catches it to fall back to FallbackPlanner — see graph.py.
"""
import os
import json
from groq import Groq
from house_sim.world import House
from agent.schemas import AgentAction, AGENT_ACTION_TOOL

MAX_RETRIES = 2


class PlannerExhausted(Exception):
    pass


SYSTEM_PROMPT = """You are an embodied agent controlling a body in a house. You can only see the \
room you're currently in. You will be given: what you currently perceive, a task to complete, and \
(if available) relevant memories recalled from past visits to this house. Use the memories if they \
help, but trust your current perception over stale memory if they conflict — say so in your thought \
if that happens. Always call take_action with exactly one next action. Keep 'thought' to one short \
first-person sentence — this is shown to a person watching you work, make it genuine, not robotic."""


class LLMPlanner:
    def __init__(self, model: str | None = None):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model or os.environ.get("PLANNER_MODEL", "llama-3.3-70b-versatile")

    def next_action(self, house: House, task_description: str, recall_context: str = "") -> AgentAction:
        perception = house.perceive_text()
        retry_note = ""
        for attempt in range(MAX_RETRIES + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"TASK: {task_description}\n\n"
                    f"CURRENT PERCEPTION:\n{perception}\n\n"
                    f"RECALLED MEMORY (may be stale, verify against perception):\n"
                    f"{recall_context or 'Nothing recalled yet.'}\n\n"
                    f"{retry_note}"
                )},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[AGENT_ACTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "take_action"}},
                temperature=0,  # deterministic for demo reproducibility
            )
            tool_call = response.choices[0].message.tool_calls[0]
            try:
                raw = json.loads(tool_call.function.arguments)
                action = AgentAction(**raw)
            except Exception as e:
                retry_note = f"Your previous response was malformed ({e}). Try again with valid JSON."
                continue

            valid, reason = self._validate(house, action)
            if valid:
                return action
            retry_note = f"That action failed: {reason}. Pick a different, valid action."

        raise PlannerExhausted(f"LLM planner failed to produce a valid action after {MAX_RETRIES} retries.")

    @staticmethod
    def _validate(house: House, action: AgentAction) -> tuple[bool, str]:
        """Pre-flight check against ground-truth-adjacent but agent-visible state
        (i.e. what perceive() would show), so we never execute an invalid action
        and never crash the sim on a bad plan."""
        p = house.perceive()
        if action.type == "done":
            return True, ""
        if action.type == "move":
            if action.target not in p["connects_to"]:
                return False, f"{action.target} isn't reachable from {p['room']}"
            return True, ""
        if action.type in ("open", "close", "use"):
            visible_names = [line.strip("- ").split(" ")[0] for line in p["visible_objects"]]
            if action.target not in visible_names and action.target not in p["inventory"]:
                return False, f"{action.target} isn't visible here"
            return True, ""
        if action.type == "pick":
            visible_names = [line.strip("- ").split(" ")[0] for line in p["visible_objects"]]
            if action.target not in visible_names:
                return False, f"{action.target} isn't visible here"
            return True, ""
        if action.type == "place":
            if action.target not in p["inventory"]:
                return False, f"you aren't carrying {action.target}"
            return True, ""
        return False, "unknown action type"
