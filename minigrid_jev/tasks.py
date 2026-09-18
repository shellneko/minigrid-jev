"""Benchmark task registry: MiniGrid environments grouped by difficulty."""

from dataclasses import dataclass

# Universal mechanics the model is told about in every state.
BASE_RULES = (
    "Walls and closed or locked doors cannot be walked through. "
    "Stepping onto lava or colliding with a moving obstacle ends the episode with failure. "
    "A step is wasted when the chosen action has no effect."
)

# Environments where `done` must be taken adjacent to the target to succeed.
GOTO_RULES = (
    "The mission is completed by taking the 'done' action while standing on a cell "
    "directly adjacent (up/down/left/right) to the object named in the mission. "
    "Using 'toggle' ends the episode immediately with failure."
)

# Picking up any object ends the episode; it must be the correct one.
FETCH_RULES = (
    "The episode ends the moment the agent picks up an object: success only if it is "
    "exactly the object described by the mission; picking up anything else fails."
)

# Dropping the carried object ends the episode; it must land next to the target.
PUTNEAR_RULES = (
    "The episode ends the moment the agent drops the carried object: success only if "
    "it lands on a cell adjacent to the target described by the mission."
)

# Moving ball obstacles kill on contact.
DYNAMIC_RULES = (
    "Blue ball obstacles move one cell every step. Colliding with one ends the episode."
)


@dataclass(frozen=True)
class Task:
    env_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    note: str = ""
    env_rules: str | None = None  # extra rules shown to the model for this env
    uses_done: bool = False  # episode can only be completed via the `done` action


TASKS: list[Task] = [
    # ---- easy: pure navigation, no object interaction ----
    Task("MiniGrid-Empty-5x5-v0", "easy", "Empty room; reach the green goal."),
    Task("MiniGrid-Empty-Random-5x5-v0", "easy", "Empty room with random start; reach the goal."),
    Task("MiniGrid-DistShift1-v0", "easy", "Reach the goal while avoiding lava."),
    # ---- medium: small maps, interaction, or language grounding ----
    Task("MiniGrid-FourRooms-v0", "medium", "Navigate between four rooms to reach the goal."),
    Task("MiniGrid-DoorKey-5x5-v0", "medium", "Pick up the key, unlock the door, reach the goal."),
    Task("MiniGrid-DoorKey-6x6-v0", "medium", "Pick up the key, unlock the door, reach the goal."),
    Task("MiniGrid-SimpleCrossingS9N1-v0", "medium", "Cross a wall through a single gap."),
    Task("MiniGrid-RedBlueDoors-6x6-v0", "medium", "Open the red door, then the blue door, reach the goal."),
    Task("MiniGrid-GoToDoor-5x5-v0", "medium", "Identify and reach the correct door.",
         env_rules=GOTO_RULES, uses_done=True),
    Task("MiniGrid-Fetch-5x5-N2-v0", "medium", "Pick up the correct one of two objects.",
         env_rules=FETCH_RULES),
    # ---- hard: larger maps, multi-step reasoning, hazards ----
    Task("MiniGrid-DoorKey-8x8-v0", "hard", "Key and door in a larger room."),
    Task("MiniGrid-Unlock-v0", "hard", "Find the key in one room, unlock the door to the next."),
    Task("MiniGrid-LavaGapS7-v0", "hard", "Reach the goal without stepping into the lava gap."),
    Task("MiniGrid-Dynamic-Obstacles-6x6-v0", "hard", "Reach the goal while dodging moving obstacles.",
         env_rules=DYNAMIC_RULES),
    Task("MiniGrid-KeyCorridorS3R1-v0", "hard", "Explore locked rooms; find the key and the target."),
    Task("MiniGrid-ObstructedMaze-1Dl-v0", "hard", "Maze with keys and blocked doors."),
    Task("MiniGrid-MultiRoom-N2-S4-v0", "hard", "Navigate a sequence of locked rooms."),
    Task("MiniGrid-PutNear-6x6-N2-v0", "hard", "Carry the correct object next to the target.",
         env_rules=PUTNEAR_RULES),
]

TASKS_BY_ID = {t.env_id: t for t in TASKS}

DIFFICULTIES = ("easy", "medium", "hard")


def select_tasks(env_ids: list[str] | None = None, difficulty: str | None = None) -> list[Task]:
    """Filter the registry by env ids and/or difficulty tier."""
    tasks = TASKS
    if env_ids:
        unknown = sorted(set(env_ids) - set(TASKS_BY_ID))
        if unknown:
            raise ValueError(f"Unknown env ids: {', '.join(unknown)}")
        tasks = [TASKS_BY_ID[i] for i in env_ids]
    if difficulty:
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty: {difficulty} (expected one of {DIFFICULTIES})")
        tasks = [t for t in tasks if t.difficulty == difficulty]
    return tasks
