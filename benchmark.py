"""CLI entry point for the MiniGrid-Jev benchmark.

Examples:
    uv run python benchmark.py --list
    uv run python benchmark.py --difficulty easy --episodes 3
    uv run python benchmark.py --tasks MiniGrid-DoorKey-5x5-v0 --episodes 5 --verbose
    uv run python benchmark.py --episodes 5 --jobs 8 --out results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minigrid_jev import JevAgent, run_benchmark
from minigrid_jev.runner import results_to_json, summarize
from minigrid_jev.tasks import select_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniGrid benchmark driven by Jev (TypeSafe System One).")
    parser.add_argument("--tasks", help="Comma-separated env ids (default: all registered tasks).")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Restrict to one difficulty tier.")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task (default: 3).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; episode i uses seed+i (default: 0).")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel episodes (default: 4).")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps per episode (default: env limit).")
    parser.add_argument("--model", default=None, help="Model name override (default: jev-latest / TYPESAFE_DEFAULT_MODEL).")
    parser.add_argument("--render", action="store_true", help="Show a pygame window per running episode (use --jobs 1).")
    parser.add_argument("--verbose", action="store_true", help="Print every decision and keep per-step traces.")
    parser.add_argument("--out", help="Write per-episode results to this JSON file.")
    parser.add_argument("--list", action="store_true", help="List registered tasks and exit.")
    args = parser.parse_args()

    if args.list:
        for t in select_tasks():
            print(f"{t.difficulty:6s}  {t.env_id:38s} {t.note}")
        return 0

    env_ids = [s.strip() for s in args.tasks.split(",")] if args.tasks else None
    try:
        tasks = select_tasks(env_ids, args.difficulty)
    except ValueError as exc:
        parser.error(str(exc))

    if args.render and args.jobs > 1:
        print("note: --render with --jobs > 1 opens one window per parallel episode.")

    print(f"Running {len(tasks)} task(s) x {args.episodes} episode(s), jobs={args.jobs}")
    agent = JevAgent(model=args.model)
    results = run_benchmark(
        tasks,
        episodes=args.episodes,
        base_seed=args.seed,
        agent=agent,
        jobs=args.jobs,
        max_steps=args.max_steps,
        verbose=args.verbose,
        render=args.render,
    )

    print()
    print(summarize(results))

    if args.out:
        Path(args.out).write_text(results_to_json(results))
        print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
