# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""Aggregate sharded eval results (results_<suite>_shard{i}of{N}.json).

The run metadata (model / revision / method / task_suite / num_envs / date, ...) is
taken from the first shard file and copied into the aggregate JSON.
"""

import argparse
import datetime
import glob
import json
import os

# Metadata keys copied from the shard JSONs into the aggregate
_META_KEYS = (
    "model", "revision", "method", "task_suite",
    "num_envs", "action_horizon", "crop_img", "use_torch_compile", "early_stop", "seed",
)


def _fmt_hms(sec: float) -> str:
    """Always h:mm:ss (hours are never omitted)."""
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_object")
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--log-dir", default="eval/logs")
    args = p.parse_args()

    pattern = os.path.join(args.log_dir, f"results_{args.task_suite}_shard*of{args.num_shards}.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shard results found: {pattern}")
    if len(files) != args.num_shards:
        expected = {f"results_{args.task_suite}_shard{i}of{args.num_shards}.json" for i in range(args.num_shards)}
        found = {os.path.basename(f) for f in files}
        missing = sorted(expected - found)
        raise SystemExit(
            f"expected {args.num_shards} shard result files, found {len(files)}; missing: {missing} "
            "(a shard likely crashed: check its shard*.log, fix/re-run, then re-aggregate)"
        )

    meta: dict = {}
    total_s = total_n = 0
    per_task: dict[str, dict] = {}
    started_ats: list[str] = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        # Shards rewrite their JSON after every task; only finished ones may be aggregated.
        if d.get("finished") is not True:
            raise SystemExit(
                f"{os.path.basename(f)} is not finished ({d.get('tasks_done')}/{d.get('tasks_total')} "
                "tasks): the shard is still running or crashed (check its shard*.log)"
            )
        if not meta:  # take the metadata from the first shard
            meta = {k: d.get(k) for k in _META_KEYS}
        if d.get("started_at"):
            started_ats.append(d["started_at"])
        total_s += d["success"]
        total_n += d["total"]
        for task, v in d.get("per_task", {}).items():
            e = per_task.setdefault(task, {"success": 0, "total": 0, "instruction": v.get("instruction")})
            e["success"] += v["success"]
            e["total"] += v["total"]

    sr = total_s / max(1, total_n)
    print(f"=== {meta.get('model')} @ {meta.get('revision')} | {args.task_suite}: {len(files)} shards ===")
    for task, v in sorted(per_task.items()):
        print(f"  {task:<60} {v['success']}/{v['total']} = {v['success'] / max(1, v['total']):.1%}")
    print(f"OVERALL success_rate = {total_s}/{total_n} = {sr:.3%}")

    # Shards start in parallel: use the earliest started_at as the start of the run and
    # the aggregation time (right after the last shard finishes) as its end.
    now = datetime.datetime.now()
    started_at = min(started_ats) if started_ats else None
    duration = None
    if started_at:
        started_dt = datetime.datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        duration = _fmt_hms((now - started_dt).total_seconds())
        print(f"started_at={started_at}  duration={duration}")

    out = os.path.join(args.log_dir, f"results_{args.task_suite}_AGGREGATE.json")
    with open(out, "w") as fh:
        json.dump({
            **meta,
            "num_shards": len(files),
            "started_at": started_at,
            "aggregated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration,
            "success": total_s,
            "total": total_n,
            "success_rate": sr,
            "per_task": per_task,
        }, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
