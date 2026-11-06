# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""Export eval/logs/**/results_<suite>_AGGREGATE.json to a CSV with one row per result.

Usage:
    python eval/export_results_csv.py [-o eval/logs/results_summary.csv]
"""

import argparse
import csv
import glob
import json
import os
import re

FIELDNAMES = [
    "timestamp",
    "libero_hf_revision",
    "model",
    "benchmark",
    "ah",
    "success_rate",
    "success_count",
    "incomplete",
]

TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})$")

# Every LIBERO suite has 10 tasks. Fewer entries in per_task mean that a shard
# crashed and its tasks are missing from the aggregate.
EXPECTED_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
}


def load_rows(log_dir: str) -> list[dict]:
    rows = []
    pattern = os.path.join(log_dir, "**", "results_*_AGGREGATE.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            d = json.load(f)

        run_dir_name = os.path.basename(os.path.dirname(path))
        m = TIMESTAMP_RE.search(run_dir_name)
        timestamp = m.group(1) if m else ""

        task_suite = d.get("task_suite", "")

        n_tasks = len(d.get("per_task", {}))
        expected_tasks = EXPECTED_TASK_COUNTS.get(task_suite)
        incomplete = (
            f"missing shards ({n_tasks}/{expected_tasks} tasks)"
            if expected_tasks is not None and n_tasks < expected_tasks
            else ""
        )

        rows.append({
            "timestamp": timestamp,
            "libero_hf_revision": d.get("revision") or "",
            "model": d.get("model", ""),
            "benchmark": task_suite,
            "ah": d.get("action_horizon", ""),
            "success_rate": round(d["success_rate"] * 100, 1),
            "success_count": d.get("success", ""),
            "incomplete": incomplete,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_log_dir = os.path.join(repo_root, "eval", "logs")
    default_out = os.path.join(default_log_dir, "results_summary.csv")
    parser.add_argument("--log-dir", default=default_log_dir)
    parser.add_argument("-o", "--output", default=default_out)
    args = parser.parse_args()

    rows = load_rows(args.log_dir)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")

    flagged = [r for r in rows if r["incomplete"]]
    if flagged:
        print(f"WARNING: {len(flagged)} row(s) flagged as incomplete (crashed/missing shards):")
        for r in flagged:
            print(f"  {r['timestamp']} {r['model']} {r['libero_hf_revision']} "
                  f"{r['benchmark']}: {r['incomplete']}")


if __name__ == "__main__":
    main()
