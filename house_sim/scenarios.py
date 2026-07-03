"""
The fixed house layout used across all sessions, plus task definitions and the
world-drift scenario. Keep this house layout IDENTICAL across sessions 1 and 2 —
the whole point of the demo is that it's the same house, only the agent's memory
changes. Session 3 deliberately calls apply_drift() to break that assumption on
purpose.
"""
from house_sim.world import House, Room, GameObject


def build_house() -> House:
    rooms = {
        "kitchen": Room(name="kitchen", connects_to=["hallway"], objects=["fridge", "cabinet", "kettle", "stove"]),
        "hallway": Room(name="hallway", connects_to=["kitchen", "living_room", "bedroom"], objects=[]),
        "living_room": Room(name="living_room", connects_to=["hallway"], objects=["drawer", "lamp", "remote"]),
        "bedroom": Room(name="bedroom", connects_to=["hallway"], objects=["closet", "desk"]),
    }

    objects = {
        "fridge": GameObject(name="fridge", location="kitchen", is_container=True, is_open=False,
                              contains=["milk"]),
        "cabinet": GameObject(name="cabinet", location="kitchen", is_container=True, is_open=False,
                               contains=["mug", "coffee", "sugar"]),
        "kettle": GameObject(name="kettle", location="kitchen", portable=False, state={"on": False}),
        "stove": GameObject(name="stove", location="kitchen", portable=False, state={"on": False}),
        "milk": GameObject(name="milk", location="fridge"),
        "mug": GameObject(name="mug", location="cabinet"),
        "coffee": GameObject(name="coffee", location="cabinet"),
        "sugar": GameObject(name="sugar", location="cabinet"),
        "drawer": GameObject(name="drawer", location="living_room", is_container=True, is_open=False,
                              contains=["keys"]),
        "lamp": GameObject(name="lamp", location="living_room", portable=False, state={"on": False}),
        "remote": GameObject(name="remote", location="living_room"),
        "keys": GameObject(name="keys", location="drawer"),
        "closet": GameObject(name="closet", location="bedroom", is_container=True, is_open=False,
                              contains=["book"]),
        "desk": GameObject(name="desk", location="bedroom"),
        "book": GameObject(name="book", location="closet"),
    }

    return House(rooms=rooms, objects=objects, start_room="hallway")


# Each task: goal description for the planner prompt, plus a success check the
# session runner uses to know when to stop (kept separate from the agent so the
# agent can't "cheat" by reading it).
TASKS = {
    "make_coffee": {
        "description": "Make a cup of coffee: get the mug, coffee, and sugar from wherever they are, "
                        "bring them to the kitchen, and turn the kettle on.",
        "success": lambda house: (
            "mug" in house.inventory + house.rooms["kitchen"].objects
            and "coffee" in house.inventory + house.rooms["kitchen"].objects
            and house.objects["kettle"].state.get("on", False)
        ),
    },
    "make_tea": {
        "description": "Make a cup of tea: get the mug and sugar, bring them to the kitchen, "
                        "and turn the kettle on. (Note: no separate 'tea' item exists — this task "
                        "is deliberately similar-but-not-identical to make_coffee, to test whether "
                        "recall() generalizes semantically rather than exact-matching a task name.)",
        "success": lambda house: (
            "mug" in house.inventory + house.rooms["kitchen"].objects
            and "sugar" in house.inventory + house.rooms["kitchen"].objects
            and house.objects["kettle"].state.get("on", False)
        ),
    },
    "find_keys": {
        "description": "Find the keys and bring them to the hallway.",
        "success": lambda house: "keys" in house.inventory + house.rooms["hallway"].objects,
    },
}


def apply_standard_drift(house: House):
    """The session-3 world-drift scenario: keys move from the living room drawer
    to the bedroom desk while the agent 'wasn't looking'. Call this once, between
    session 2 and session 3, on a fresh House built from build_house()."""
    house.apply_drift("keys", "bedroom")
