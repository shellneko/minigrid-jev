"""Encode MiniGrid environment state into JSON-serializable text for Jev."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any

from minigrid.core.world_object import WorldObj

from minigrid_jev.tasks import BASE_RULES

DIRECTION_NAMES = ("east", "south", "west", "north")

# Agent location within the 7x7 egocentric view returned by gen_obs_grid():
# bottom-center, facing "up" the grid (toward row 0).
AGENT_CELL = (3, 6)
FRONT_CELL = (3, 5)

VIEW_NOTE = (
    "7x7 egocentric view of the grid world. The agent occupies row 6, column 3 and "
    "faces toward row 0, so row 5 column 3 is the cell directly ahead, row 4 is two "
    "cells ahead, etc. Columns to the left/right of column 3 are to the agent's "
    "left/right. 'unseen' cells are currently out of view."
)


def describe_object(cell: WorldObj) -> str:
    """Human-readable description of a grid object, e.g. 'locked yellow door'."""
    color = getattr(cell, "color", None)
    desc = f"{color} {cell.type}" if color else cell.type
    if cell.type == "door":
        state = "open" if cell.is_open else "locked" if cell.is_locked else "closed"
        desc = f"{state} {desc}"
    return desc


def encode_state(
    env,
    *,
    env_rules: str | None,
    recent_actions: Iterable[str],
    visited: Collection[tuple[int, int]],
) -> dict[str, Any]:
    """Build the `state` payload for one decision step."""
    u = env.unwrapped
    view_grid, vis_mask = u.gen_obs_grid()

    cells: list[list[str]] = []
    for j in range(view_grid.height):
        row = []
        for i in range(view_grid.width):
            if (i, j) == AGENT_CELL:
                row.append("agent")
            elif not vis_mask[i, j]:
                row.append("unseen")
            else:
                obj = view_grid.get(i, j)
                row.append("empty floor" if obj is None else describe_object(obj))
        cells.append(row)

    state: dict[str, Any] = {
        "mission": u.mission,
        "rules": " ".join(r for r in (BASE_RULES, env_rules) if r),
        "agent": {
            "position": [int(u.agent_pos[0]), int(u.agent_pos[1])],
            "facing": DIRECTION_NAMES[u.agent_dir],
            "carrying": describe_object(u.carrying) if u.carrying else None,
        },
        "cell_directly_ahead": cells[FRONT_CELL[1]][FRONT_CELL[0]],
        "view": {"note": VIEW_NOTE, "cells": cells},
        "steps_used": u.step_count,
        "step_limit": u.max_steps,
        "positions_visited": sorted([list(p) for p in visited]),
        "recent_actions": list(recent_actions),
    }
    return state
