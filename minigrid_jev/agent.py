"""Jev-backed policy: choose the next MiniGrid action via a Choice question."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typesafe_sdk import Choice, ChoiceAnswer, TypeSafeClient

from minigrid_jev.tasks import Task

# Semantics of each action, shown to the model as Choice criteria.
ACTION_DESCRIPTIONS: dict[str, str] = {
    "turn_left": "Rotate 90 degrees counterclockwise in place (the agent stays in its cell).",
    "turn_right": "Rotate 90 degrees clockwise in place (the agent stays in its cell).",
    "move_forward": (
        "Move into the cell directly ahead. Works when it is empty floor, an open door, "
        "or the goal. Walls, closed doors, and objects block the move and waste the step."
    ),
    "pickup": "Pick up the object in the cell directly ahead.",
    "drop": "Drop the carried object into the cell directly ahead.",
    "toggle": (
        "Interact with the object directly ahead: open a closed door, or unlock a locked "
        "door while carrying a key of the matching color. Also opens boxes."
    ),
    "done": (
        "Declare the mission complete. Only succeeds when the mission's completion "
        "condition is already satisfied."
    ),
}

QUESTION_INSTRUCTIONS = (
    "You control an agent in a MiniGrid grid world. Based on the mission, the rules, "
    "the agent's egocentric view, and what happened on recent steps, choose the single "
    "best next action. Prefer actions that make progress toward the mission; do not "
    "repeat actions that were recently blocked or had no effect."
)


@dataclass
class Decision:
    action: str
    answer: ChoiceAnswer
    input_tokens: int
    output_tokens: int


def available_actions(env, task: Task) -> list[str]:
    """Actions worth offering in the current state.

    Turns and forward are always offered; pickup/drop/toggle appear only when the
    cell ahead supports them; `done` exists only in tasks that require it.
    """
    u = env.unwrapped
    front = u.grid.get(*u.front_pos)

    actions = ["turn_left", "turn_right", "move_forward"]
    if u.carrying is None and front is not None and front.can_pickup():
        actions.append("pickup")
    if u.carrying is not None and front is None:
        actions.append("drop")
    if front is not None and front.type in ("door", "box"):
        actions.append("toggle")
    if task.uses_done:
        actions.append("done")
    return actions


class JevAgent:
    """Queries Jev once per step with a Choice question over available actions."""

    def __init__(self, client: TypeSafeClient | None = None, model: str | None = None) -> None:
        self.client = client or TypeSafeClient()
        self.model = model

    def decide(self, state: dict[str, Any], actions: list[str]) -> Decision:
        response = self.client.system_one(
            state=state,
            questions={
                "action": Choice(
                    instructions=QUESTION_INSTRUCTIONS,
                    criteria={a: ACTION_DESCRIPTIONS[a] for a in actions},
                )
            },
            model=self.model,
        )
        answer = response.choices["action"]
        usage = response.usage
        return Decision(
            action=answer.choice,
            answer=answer,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
        )
