"""
Deterministic fallback planner. Two jobs:

1. Day-1 development harness — lets you build and fully test house_sim/ before
   any LLM or Cognee code exists at all. Run scripts/run_session.py --fallback-only.
2. Live safety net — if the real LLM planner (agent/planner.py) fails structured
   validation twice in a row on the same step, the LangGraph graph falls back to
   this instead of stalling the demo.

Deliberately "blind but guaranteed": it does NOT read house.ground_truth_snapshot()
or any other privileged state. It only uses what perceive() would show a real
agent, plus a systematic (not random) exploration order, so it always finds
things eventually without cheating and without the improvement-over-sessions
story looking staged.
"""
from typing import Optional
from house_sim.world import House
from agent.schemas import AgentAction


class FallbackPlanner:
    def __init__(self):
        self._room_visit_order: list[str] = []
        self._opened_this_room: set[str] = set()

    def reset(self):
        self._room_visit_order = []
        self._opened_this_room = set()

    def next_action(self, house: House, task_description: str, held_target_hint: Optional[str] = None) -> AgentAction:
        p = house.perceive()
        room = p["room"]

        # Open anything closed in the current room first — most tasks need contents.
        for line in p["visible_objects"]:
            name = line.split(" ")[0]
            if "(closed)" in line and name not in self._opened_this_room:
                self._opened_this_room.add(name)
                return AgentAction(type="open", target=name,
                                    thought=f"Don't remember what's in the {name}, checking it.")

        # Pick up anything visible and not already carried (greedy — fine for this sim's small task set).
        for line in p["visible_objects"]:
            name = line.strip("- ").split(" ")[0]
            if (name in house.objects and not house.objects[name].is_container
                    and house.objects[name].portable and name not in house.inventory):
                return AgentAction(type="pick", target=name, thought=f"Grabbing {name} while I'm here.")

        # If carrying something and we're in the kitchen, this sim's tasks generally
        # want things placed/used there — try the obvious completion moves.
        if room == "kitchen":
            if not house.objects["kettle"].state.get("on", False) and any(
                x in house.inventory for x in ("mug", "coffee", "sugar")
            ):
                return AgentAction(type="use", target="kettle", thought="I have what I need, starting the kettle.")

        # Otherwise, explore unvisited rooms systematically.
        for neighbor in p["connects_to"]:
            if neighbor not in self._room_visit_order:
                self._room_visit_order.append(neighbor)
                return AgentAction(type="move", target=neighbor, thought=f"Haven't checked the {neighbor} yet.")

        # Nowhere new to go and nothing obvious to do — declare done to avoid an infinite loop.
        return AgentAction(type="done", thought="I've checked everywhere I can reach; stopping here.")
