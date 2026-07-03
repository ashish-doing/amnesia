"""
Symbolic house simulation. Deterministic, no physics, no rendering here.

Design intent: this is the one component everything else depends on, so it is
built to be boring and bug-free rather than clever. Partial observability is
real — the agent only perceives the room it's currently in, which is what
makes "remembering the rest of the house" actually matter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class GameObject:
    name: str
    location: str          # room name, or a container name if inside one
    is_container: bool = False
    is_open: bool = False
    portable: bool = True  # False for fixed appliances (kettle, stove, lamp) - can't be pick()'d
    contains: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)  # e.g. {"on": False} for a kettle

    def to_text(self) -> str:
        bits = [self.name]
        if self.is_container:
            bits.append("(open)" if self.is_open else "(closed)")
        if self.state:
            bits.append(str(self.state))
        return " ".join(bits)


@dataclass
class Room:
    name: str
    connects_to: list[str]
    objects: list[str] = field(default_factory=list)  # top-level object names in this room


class ActionResult:
    def __init__(self, success: bool, message: str, observation: dict):
        self.success = success
        self.message = message
        self.observation = observation

    def to_dict(self) -> dict:
        return {"success": self.success, "message": self.message, "observation": self.observation}


class House:
    """
    Ground-truth world state. The agent never gets direct access to this object —
    it only gets what perceive() returns for its current room. Anything else the
    agent "knows" has to come from its own memory (Cognee), not from cheating and
    reading this object directly. Keep that boundary intact when you extend this.
    """

    def __init__(self, rooms: dict[str, Room], objects: dict[str, GameObject], start_room: str):
        self.rooms = rooms
        self.objects = objects
        self.agent_room = start_room
        self.inventory: list[str] = []
        self.action_log: list[dict] = []
        self.step_count = 0

    # ---------- perception ----------

    def perceive(self) -> dict:
        """What the agent can currently see: its room, connections, visible objects."""
        room = self.rooms[self.agent_room]
        visible = []
        for obj_name in room.objects:
            obj = self.objects[obj_name]
            visible.append(obj.to_text())
            if obj.is_container and obj.is_open:
                for inner_name in obj.contains:
                    visible.append(f"  - {self.objects[inner_name].to_text()} (inside {obj.name})")
        return {
            "room": room.name,
            "connects_to": room.connects_to,
            "visible_objects": visible,
            "inventory": list(self.inventory),
        }

    def perceive_text(self) -> str:
        p = self.perceive()
        lines = [f"You are in the {p['room']}."]
        lines.append(f"Doors lead to: {', '.join(p['connects_to']) if p['connects_to'] else 'nowhere else'}.")
        if p["visible_objects"]:
            lines.append("You can see: " + "; ".join(p["visible_objects"]))
        else:
            lines.append("You don't see anything notable here.")
        if p["inventory"]:
            lines.append("You are carrying: " + ", ".join(p["inventory"]))
        return "\n".join(lines)

    # ---------- action API ----------
    # Every action returns an ActionResult. Nothing raises on bad input — invalid
    # actions are a normal, expected outcome the planner has to handle, not an
    # exception path.

    def move(self, room_name: str) -> ActionResult:
        current = self.rooms[self.agent_room]
        if room_name not in current.connects_to:
            return self._log(ActionResult(False, f"Can't reach {room_name} from {current.name} directly.", {}))
        self.agent_room = room_name
        return self._log(ActionResult(True, f"Moved to {room_name}.", self.perceive()))

    def open(self, container_name: str) -> ActionResult:
        obj = self._find_in_current_room(container_name)
        if obj is None:
            return self._log(ActionResult(False, f"No {container_name} here.", {}))
        if not obj.is_container:
            return self._log(ActionResult(False, f"{container_name} isn't a container.", {}))
        if obj.is_open:
            return self._log(ActionResult(True, f"{container_name} was already open.", {}))
        obj.is_open = True
        return self._log(ActionResult(True, f"Opened {container_name}.", self.perceive()))

    def close(self, container_name: str) -> ActionResult:
        obj = self._find_in_current_room(container_name)
        if obj is None or not obj.is_container:
            return self._log(ActionResult(False, f"Can't close {container_name} here.", {}))
        obj.is_open = False
        return self._log(ActionResult(True, f"Closed {container_name}.", {}))

    def pick(self, object_name: str) -> ActionResult:
        obj = self._find_visible(object_name)
        if obj is None:
            return self._log(ActionResult(False, f"Can't see {object_name} here.", {}))
        if obj.is_container:
            return self._log(ActionResult(False, f"{object_name} is a container, can't pick it up.", {}))
        if not obj.portable:
            return self._log(ActionResult(False, f"{object_name} is fixed in place, can't pick it up.", {}))
        self._remove_from_wherever(object_name)
        self.inventory.append(object_name)
        return self._log(ActionResult(True, f"Picked up {object_name}.", {}))

    def place(self, object_name: str, target: str) -> ActionResult:
        if object_name not in self.inventory:
            return self._log(ActionResult(False, f"Not carrying {object_name}.", {}))
        target_obj = self._find_in_current_room(target) if target != self.agent_room else None
        self.inventory.remove(object_name)
        if target_obj is not None and target_obj.is_container:
            if not target_obj.is_open:
                self.inventory.append(object_name)
                return self._log(ActionResult(False, f"{target} is closed.", {}))
            target_obj.contains.append(object_name)
            self.objects[object_name].location = target_obj.name
        else:
            self.rooms[self.agent_room].objects.append(object_name)
            self.objects[object_name].location = self.agent_room
        return self._log(ActionResult(True, f"Placed {object_name} in/at {target}.", {}))

    def use(self, object_name: str) -> ActionResult:
        obj = self._find_visible(object_name) or (self.objects[object_name] if object_name in self.inventory else None)
        if obj is None:
            return self._log(ActionResult(False, f"Can't use {object_name} — not accessible.", {}))
        obj.state["on"] = not obj.state.get("on", False)
        return self._log(ActionResult(True, f"Toggled {object_name}: now {'on' if obj.state['on'] else 'off'}.", {}))

    # ---------- helpers ----------

    def _find_in_current_room(self, name: str) -> Optional[GameObject]:
        room = self.rooms[self.agent_room]
        if name in room.objects:
            return self.objects[name]
        return None

    def _find_visible(self, name: str) -> Optional[GameObject]:
        obj = self._find_in_current_room(name)
        if obj:
            return obj
        room = self.rooms[self.agent_room]
        for obj_name in room.objects:
            container = self.objects[obj_name]
            if container.is_container and container.is_open and name in container.contains:
                return self.objects[name]
        return None

    def _remove_from_wherever(self, name: str):
        room = self.rooms[self.agent_room]
        if name in room.objects:
            room.objects.remove(name)
            return
        for obj_name in room.objects:
            container = self.objects[obj_name]
            if container.is_container and name in container.contains:
                container.contains.remove(name)
                return

    def _log(self, result: ActionResult) -> ActionResult:
        self.step_count += 1
        self.action_log.append({"step": self.step_count, **result.to_dict()})
        return result

    # ---------- ground-truth snapshot (for drift injection / evaluation ONLY, never given to the agent) ----------

    def ground_truth_snapshot(self) -> dict:
        return {
            "objects": {
                name: {"location": obj.location, "is_open": obj.is_open, "state": obj.state}
                for name, obj in self.objects.items()
            }
        }

    def apply_drift(self, object_name: str, new_room: str):
        """Move an object to a new room between sessions, simulating the world changing
        while the agent wasn't watching. This is what makes forget()/improve() matter."""
        self._remove_from_wherever_global(object_name)
        self.rooms[new_room].objects.append(object_name)
        self.objects[object_name].location = new_room

    def _remove_from_wherever_global(self, name: str):
        for room in self.rooms.values():
            if name in room.objects:
                room.objects.remove(name)
        for obj in self.objects.values():
            if obj.is_container and name in obj.contains:
                obj.contains.remove(name)
