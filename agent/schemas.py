"""
Structured output schema for the planner. This is the primary guardrail: the
LLM is forced to emit one of these shapes via tool-calling, so "tries to open
something that isn't a container" becomes a content-validation problem the sim
rejects gracefully, never a parsing crash.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    type: Literal["move", "open", "close", "pick", "place", "use", "done"] = Field(
        ..., description="The action to take. 'done' means the agent believes the task is complete."
    )
    target: Optional[str] = Field(
        None, description="The object or room name this action applies to. Omit for 'done'."
    )
    destination: Optional[str] = Field(
        None, description="Only used for 'place' — where to put the object."
    )
    thought: str = Field(
        ..., description="One short sentence of the agent's own reasoning, spoken in first person. "
                          "This is what gets shown to the user as narration, e.g. "
                          "'This looks familiar, I remember the mug being in the cabinet.'"
    )


# JSON schema handed to Groq's tool-calling as a forced function call.
AGENT_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Choose exactly one next action in the house.",
        "parameters": AgentAction.model_json_schema(),
    },
}
