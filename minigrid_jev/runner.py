"""Episode loop, outcome tracking, metrics, and reporting."""

from __future__ import annotations

import json
import statistics
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium
import minigrid  # noqa: F401  (registers MiniGrid-* envs with gymnasium)

from minigrid_jev.agent import JevAgent, available_actions
from minigrid_jev.encoding import describe_object, encode_state
from minigrid_jev.tasks import Task

ACTION_NAME_TO_ENUM = {
    "turn_left": "left",
    "turn_right": "right",
    "move_forward": "forward",
    "pickup": "pickup",
    "drop": "drop",
    "toggle": "toggle",
    "done": "done",
}


@dataclass
class StepRecord:
    step: int
    action: str
    confidence: float
    outcome: str


@dataclass
class EpisodeResult:
    env_id: str
    difficulty: str
    seed: int
    success: bool
    steps: int
    reward: float
    end_reason: str  # "success" | "failed" | "step_limit" | "error"
    api_calls: int
    input_tokens: int
    output_tokens: int
    mean_confidence: float
    action_counts: dict[str, int] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list, repr=False)


def _snapshot(env) -> dict[str, Any]:
    u = env.unwrapped
    front = u.grid.get(*u.front_pos)
    return {
        "pos": tuple(int(p) for p in u.agent_pos),
        "dir": u.agent_dir,
        "carrying": describe_object(u.carrying) if u.carrying else None,
        "front": describe_object(front) if front else None,
    }


def _describe_outcome(action: str, pre: dict[str, Any], post: dict[str, Any], reward: float, terminated: bool) -> str:
    if action == "move_forward":
        if terminated and reward > 0:
            return "reached the goal; mission complete"
        if terminated:
            return "moved into a lethal cell; episode over"
        return "moved forward" if post["pos"] != pre["pos"] else "blocked; stayed in place"
    if action in ("turn_left", "turn_right"):
        return f"turned; now facing {('east', 'south', 'west', 'north')[post['dir']]}"
    if action == "pickup":
        if terminated:
            return "picked up an object; episode over"
        return f"picked up {post['carrying']}" if post["carrying"] else "nothing picked up"
    if action == "drop":
        if terminated:
            return "dropped the object; episode over"
        return f"dropped {pre['carrying']}" if post["carrying"] is None else "drop failed"
    if action == "toggle":
        if terminated:
            return "toggled; episode over"
        return f"cell ahead is now {post['front']}" if post["front"] != pre["front"] else "nothing happened"
    if action == "done":
        return "declared done"
    return "no effect"


def run_episode(
    task: Task,
    seed: int,
    agent: JevAgent,
    *,
    max_steps: int | None = None,
    history_len: int = 12,
    verbose: bool = False,
    render: bool = False,
) -> EpisodeResult:
    env = gymnasium.make(task.env_id, render_mode="human" if render else None)
    try:
        env.reset(seed=seed)
        u = env.unwrapped
        limit = min(max_steps, u.max_steps) if max_steps else u.max_steps

        history: deque[str] = deque(maxlen=history_len)
        visited: set[tuple[int, int]] = {tuple(int(p) for p in u.agent_pos)}
        records: list[StepRecord] = []
        api_calls = input_tokens = output_tokens = 0
        reward, terminated, truncated = 0.0, False, False
        end_reason = "step_limit"

        while not (terminated or truncated) and u.step_count < limit:
            state = encode_state(env, env_rules=task.env_rules, recent_actions=history, visited=visited)
            actions = available_actions(env, task)
            decision = agent.decide(state, actions)
            api_calls += 1
            input_tokens += decision.input_tokens
            output_tokens += decision.output_tokens

            name = decision.action if decision.action in actions else "move_forward"
            pre = _snapshot(env)
            _, reward, terminated, truncated, _ = env.step(getattr(u.actions, ACTION_NAME_TO_ENUM[name]))
            post = _snapshot(env)
            outcome = _describe_outcome(name, pre, post, reward, terminated)
            history.append(f"{name} -> {outcome}")
            visited.add(post["pos"])
            records.append(StepRecord(u.step_count, name, decision.answer.confidence, outcome))

            if verbose:
                print(
                    f"  step {u.step_count:3d} | {name:13s} conf={decision.answer.confidence:.2f} "
                    f"| {outcome}",
                    flush=True,
                )

        success = bool(terminated and reward > 0)
        if success:
            end_reason = "success"
        elif terminated:
            end_reason = "failed"
        return EpisodeResult(
            env_id=task.env_id,
            difficulty=task.difficulty,
            seed=seed,
            success=success,
            steps=u.step_count,
            reward=float(reward),
            end_reason=end_reason,
            api_calls=api_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mean_confidence=statistics.mean(r.confidence for r in records) if records else 0.0,
            action_counts=dict(Counter(r.action for r in records)),
            trace=[asdict(r) for r in records] if verbose else [],
        )
    finally:
        env.close()


def run_benchmark(
    tasks: Sequence[Task],
    episodes: int,
    base_seed: int,
    agent: JevAgent,
    *,
    jobs: int = 4,
    max_steps: int | None = None,
    verbose: bool = False,
    render: bool = False,
) -> list[EpisodeResult]:
    plan = [(t, base_seed + ep) for t in tasks for ep in range(episodes)]
    results: list[EpisodeResult] = []

    def run_one(t: Task, seed: int) -> EpisodeResult:
        try:
            return run_episode(t, seed, agent, max_steps=max_steps, verbose=verbose, render=render)
        except Exception as exc:  # keep the benchmark running on transient failures
            return EpisodeResult(t.env_id, t.difficulty, seed, False, 0, 0.0, f"error: {exc}", 0, 0, 0, 0.0)

    if jobs <= 1:
        for t, seed in plan:
            r = run_one(t, seed)
            results.append(r)
            print(f"[done] {t.env_id} seed={seed} success={r.success} steps={r.steps} reason={r.end_reason}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_one, t, s): (t, s) for t, s in plan}
            for fut, (t, s) in futures.items():
                r = fut.result()
                results.append(r)
                print(f"[done] {t.env_id} seed={s} success={r.success} steps={r.steps} reason={r.end_reason}", flush=True)
    return results


def summarize(results: Sequence[EpisodeResult]) -> str:
    lines = []
    by_env: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_env.setdefault(r.env_id, []).append(r)

    header = f"{'env':40s} {'diff':6s} {'n':>3s} {'succ%':>6s} {'steps':>7s} {'reward':>7s} {'conf':>5s} {'tokens':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    diff_rank = {"easy": 0, "medium": 1, "hard": 2}
    for env_id, rs in sorted(by_env.items(), key=lambda kv: (diff_rank.get(kv[1][0].difficulty, 9), kv[0])):
        n = len(rs)
        succ = sum(r.success for r in rs) / n
        steps = statistics.mean(r.steps for r in rs)
        rew = statistics.mean(r.reward for r in rs)
        conf = statistics.mean(r.mean_confidence for r in rs if r.api_calls) if any(r.api_calls for r in rs) else 0.0
        toks = sum(r.input_tokens + r.output_tokens for r in rs)
        lines.append(f"{env_id:40s} {rs[0].difficulty:6s} {n:3d} {succ * 100:5.0f}% {steps:7.1f} {rew:7.3f} {conf:5.2f} {toks:8d}")

    if results:
        n = len(results)
        lines.append("-" * len(header))
        lines.append(
            f"{'OVERALL':40s} {'':6s} {n:3d} {sum(r.success for r in results) / n * 100:5.0f}% "
            f"{statistics.mean(r.steps for r in results):7.1f} {statistics.mean(r.reward for r in results):7.3f} "
            f"{'':5s} {sum(r.input_tokens + r.output_tokens for r in results):8d}"
        )
        by_diff: dict[str, list[EpisodeResult]] = {}
        for r in results:
            by_diff.setdefault(r.difficulty, []).append(r)
        lines.append("")
        lines.append("By difficulty:")
        for d in ("easy", "medium", "hard"):
            rs = by_diff.get(d)
            if rs:
                lines.append(f"  {d:6s}: {sum(r.success for r in rs)}/{len(rs)} episodes succeeded")
    return "\n".join(lines)


def results_to_json(results: Sequence[EpisodeResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
