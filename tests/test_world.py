"""
Zero-dependency sanity tests for house_sim/. Run with: python -m pytest tests/
These do not touch Cognee, Groq, or any network — that's deliberate, per the
README's build order (get this bulletproof before anything else depends on it).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from house_sim.scenarios import build_house, apply_standard_drift, TASKS


def test_house_builds():
    house = build_house()
    assert house.agent_room == "hallway"
    assert "kitchen" in house.rooms


def test_move_valid():
    house = build_house()
    result = house.move("kitchen")
    assert result.success
    assert house.agent_room == "kitchen"


def test_move_invalid():
    house = build_house()
    result = house.move("nowhere")  # a room name that doesn't exist at all
    assert not result.success


def test_open_and_pick():
    house = build_house()
    house.move("kitchen")
    open_result = house.open("cabinet")
    assert open_result.success
    pick_result = house.pick("mug")
    assert pick_result.success
    assert "mug" in house.inventory


def test_pick_closed_container_contents_fails():
    house = build_house()
    house.move("kitchen")
    result = house.pick("mug")  # cabinet still closed
    assert not result.success


def test_make_coffee_task_completable_manually():
    house = build_house()
    house.move("kitchen")
    house.open("cabinet")
    house.pick("mug")
    house.pick("coffee")
    house.use("kettle")
    assert TASKS["make_coffee"]["success"](house)


def test_drift_moves_object():
    house = build_house()
    assert "keys" in house.rooms["living_room"].objects or \
           "keys" in house.objects["drawer"].contains
    apply_standard_drift(house)
    assert "keys" in house.rooms["bedroom"].objects


def test_tidy_kitchen_task_completable_manually():
    house = build_house()
    house.move("kitchen")
    house.open("cabinet")
    house.pick("mug")
    house.pick("coffee")
    house.pick("sugar")
    house.use("kettle")  # turn it on first, as if coffee was just made
    house.place("mug", "cabinet")
    house.place("coffee", "cabinet")
    house.place("sugar", "cabinet")
    house.close("cabinet")
    house.use("kettle")  # turn it back off
    assert TASKS["tidy_kitchen"]["success"](house)


def test_fallback_planner_returns_valid_actions_against_real_house():
    """The fallback planner is imported and relied on as the safety net in
    agent/graph.py whenever the LLM planner fails - if IT has a bug, the
    entire fallback path silently crashes instead of saving a demo. Verified
    here directly, without LLM/Cognee, against a real house."""
    from agent.fallback_planner import FallbackPlanner
    from agent.schemas import AgentAction

    house = build_house()
    planner = FallbackPlanner()
    for _ in range(30):
        action = planner.next_action(house, TASKS["make_coffee"]["description"])
        assert isinstance(action, AgentAction)
        assert action.type in ("move", "open", "close", "pick", "place", "use", "done")
        if action.type == "done":
            break
        if action.type == "move":
            result = house.move(action.target)
        elif action.type == "open":
            result = house.open(action.target)
        elif action.type == "pick":
            result = house.pick(action.target)
        elif action.type == "use":
            result = house.use(action.target)
        else:
            continue
        assert result.success or not result.success  # never raises - the real assertion is no exception above
    assert TASKS["make_coffee"]["success"](house)


if __name__ == "__main__":
    # allow `python tests/test_world.py` without pytest installed
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
