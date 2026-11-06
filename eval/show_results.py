# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""Summarize LIBERO results under eval/logs/ as a markdown table.

Finished runs are read from results_<suite>_AGGREGATE.json. Runs that are still in
progress (or whose shards crashed) have no aggregate yet; their shard JSONs, which
eval.py rewrites after every task, are merged into a partial result. Partial values
are marked with "*" and listed below the table with the number of finished tasks.

If a (model, task_suite) pair has several results, the most recent one wins.
"""

import glob
import json
import os

SUITE_COLUMNS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long (10)",
}
COLUMN_ORDER = ["Spatial", "Object", "Goal", "Long (10)"]

TASKS_PER_SUITE = 10


def _load_partial(run_dir: str) -> dict | None:
    """Merge the shard JSONs of a run that has no aggregate yet."""
    shards = []
    for path in glob.glob(os.path.join(run_dir, "results_*_shard*of*.json")):
        try:
            with open(path) as f:
                shards.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue  # being written right now; picked up on the next call
    if not shards:
        return None
    d = dict(shards[0])
    d["success"] = sum(x["success"] for x in shards)
    d["total"] = sum(x["total"] for x in shards)
    d["success_rate"] = d["success"] / d["total"] if d["total"] else float("nan")
    d["per_task"] = {k: v for x in shards for k, v in x.get("per_task", {}).items()}
    d["updated_at"] = max(x.get("date", "") for x in shards)
    d["partial"] = f"{len(d['per_task'])}/{TASKS_PER_SUITE} tasks, {len(shards)}/{d.get('num_shards')} shards"
    if all("episodes_total" in x for x in shards):
        d["progress"] = {
            "done": sum(x["episodes_done"] for x in shards),
            # shards that have not written their JSON yet are counted as unknown
            "total": sum(x["episodes_total"] for x in shards),
            "complete_count": len(shards) == d.get("num_shards"),
            "started_at": min(x.get("started_at", "") for x in shards),
        }
    return d


def _progress_info(prog: dict) -> tuple[str, str]:
    """Elapsed time and a naive remaining-time estimate for a running evaluation."""
    import datetime

    try:
        start = datetime.datetime.strptime(prog["started_at"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError):
        return "-", "-"
    elapsed = (datetime.datetime.now() - start).total_seconds()
    fmt_t = lambda sec: f"{int(sec // 60)}:{int(sec % 60):02d}"  # noqa: E731
    if prog["done"] == 0 or not prog["complete_count"]:
        return fmt_t(elapsed), "-"
    left = elapsed * (prog["total"] - prog["done"]) / prog["done"]
    return fmt_t(elapsed), f"~{max(1, round(left / 60))} min"


def running_runs(latest: dict) -> list[tuple[str, str, dict]]:
    """(model, suite, progress) of the evaluations that are still running."""
    return [
        (model, suite, d["progress"])
        for (model, suite), d in sorted(latest.items())
        if d.get("partial") and d.get("progress")
    ]


def load_results(log_dir: str) -> dict:
    latest: dict[tuple[str, str], dict] = {}
    for run_dir in glob.glob(os.path.join(log_dir, "*")):
        aggregates = glob.glob(os.path.join(run_dir, "results_*_AGGREGATE.json"))
        if aggregates:
            with open(aggregates[0]) as f:
                d = json.load(f)
            d["updated_at"] = d.get("aggregated_at", "")
        else:
            d = _load_partial(run_dir)
            if d is None:
                continue
        key = (d["model"], d["task_suite"])
        if key not in latest or d["updated_at"] > latest[key]["updated_at"]:
            latest[key] = d
    return latest


def build_rows(latest: dict) -> dict:
    rows: dict[str, dict] = {}
    for (model, suite), d in latest.items():
        col = SUITE_COLUMNS.get(suite, suite)
        row = rows.setdefault(
            model, {"ah": d.get("action_horizon"), "revision": d.get("revision"), "partial": {}}
        )
        row[col] = d["success_rate"] * 100
        if d.get("partial"):
            row["partial"][col] = d["partial"]
    for row in rows.values():
        vals = [row[c] for c in COLUMN_ORDER if c in row]
        row["Avg"] = sum(vals) / len(vals) if vals else float("nan")
        if row["partial"]:
            row["partial"]["Avg"] = "includes partial results"
    return rows


def fmt(v, decimals=1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "nan"
    if decimals == 0:
        return str(int(v))
    return f"{v:.{decimals}f}"


def render_markdown(rows: dict, title: str, columns: list[str]) -> str:
    headers = ["params", "revision", "ah"] + columns + ["Avg"]
    lines_data = []
    for model in sorted(rows):
        row = rows[model]
        cells = [model, row.get("revision") or "nan", fmt(row.get("ah"), 0)]
        for c in columns + ["Avg"]:
            v = row.get(c, float("nan"))
            if c in row.get("partial", {}):
                cells.append((fmt(v) if v == v else "-") + "*")
            else:
                cells.append(fmt(v))
        lines_data.append(cells)

    widths = [len(h) for h in headers]
    for cells in lines_data:
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells, aligns):
        parts = []
        for c, w, a in zip(cells, widths, aligns):
            parts.append(c.rjust(w) if a == "r" else c.ljust(w))
        return "| " + " | ".join(parts) + " |"

    aligns = ["l", "l"] + ["r"] * (len(headers) - 2)
    sep = "|" + "|".join(
        (":" + "-" * (w - 1) if a == "l" else "-" * (w - 1) + ":") for w, a in zip(widths, aligns)
    ) + "|"

    bar = "=" * 72
    out = [bar, f"  {title}", bar, fmt_row(headers, aligns), sep]
    out += [fmt_row(c, aligns) for c in lines_data]
    notes = [
        f"  {model} / {col}: {info}"
        for model in sorted(rows)
        for col, info in rows[model].get("partial", {}).items()
        if col != "Avg"
    ]
    if notes:
        out += ["", "* in progress (or a shard crashed); partial results so far:"] + notes
    return "\n".join(out)


def render_progress_markdown(runs: list) -> str:
    lines = ["", "Running evaluations:"]
    for model, suite, prog in runs:
        elapsed, left = _progress_info(prog)
        pct = 100 * prog["done"] / max(1, prog["total"])
        lines.append(
            f"  {model} {suite}: {prog['done']}/{prog['total']} episodes ({pct:.0f}%), "
            f"elapsed {elapsed}, left {left}"
        )
    return "\n".join(lines)


def render_progress_rich(runs: list):
    from rich.progress_bar import ProgressBar
    from rich.table import Table

    table = Table(title="Running evaluations", box=None, title_style="bold", padding=(0, 1))
    table.add_column("model", style="cyan", overflow="fold", ratio=1)
    table.add_column("suite", no_wrap=True)
    table.add_column("progress", no_wrap=True, min_width=20)
    table.add_column("episodes", justify="right", no_wrap=True)
    table.add_column("elapsed", justify="right", no_wrap=True)
    table.add_column("left", justify="right", no_wrap=True)
    for model, suite, prog in runs:
        elapsed, left = _progress_info(prog)
        table.add_row(
            model,
            SUITE_COLUMNS.get(suite, suite),
            ProgressBar(total=max(1, prog["total"]), completed=prog["done"], width=20),
            f"{prog['done']}/{prog['total']}",
            elapsed,
            left,
        )
    return table


def render_rich(rows: dict, title: str, columns: list[str]):
    """Colored table for a terminal: best value per column in bold green,
    in-progress values in yellow with "*"."""
    from rich import box
    from rich.table import Table
    from rich.text import Text

    headers = columns + ["Avg"]
    best = {}
    for c in headers:
        vals = [r[c] for r in rows.values() if c in r and r[c] == r[c]]
        best[c] = max(vals) if vals else None

    show_revision = any(r.get("revision") for r in rows.values())
    table = Table(title=Text(title, style="bold"), box=box.SIMPLE_HEAVY)
    # Only the model name may wrap; numbers are never wrapped or truncated.
    table.add_column("model", style="cyan", overflow="fold", ratio=1)
    if show_revision:
        table.add_column("revision", no_wrap=True)
    table.add_column("ah", justify="right", no_wrap=True)
    for c in headers:
        table.add_column(c, justify="right", no_wrap=True, min_width=max(len(c), 6))
    for model in sorted(rows):
        row = rows[model]
        cells = [model] + ([row.get("revision") or "-"] if show_revision else []) + [fmt(row.get("ah"), 0)]
        for c in headers:
            v = row.get(c, float("nan"))
            if c in row.get("partial", {}):
                cells.append(Text((fmt(v) if v == v else "-") + "*", style="yellow"))
            elif best[c] is not None and v == best[c] and len(rows) > 1:
                cells.append(Text(fmt(v), style="bold green"))
            else:
                cells.append(Text(fmt(v) if v == v else "-", style="" if v == v else "dim"))
        table.add_row(*cells)
    notes = [
        f"  {model} / {col}: {info}"
        for model in sorted(rows)
        for col, info in rows[model].get("partial", {}).items()
        if col != "Avg"
    ]
    caption = None
    if notes:
        caption = Text(
            "* in progress (or a shard crashed); partial results so far:\n" + "\n".join(notes),
            style="yellow",
            justify="left",
        )
    return table, caption


def main() -> None:
    import argparse
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=os.path.join(repo_root, "eval", "logs"),
                   help="Directory with one sub-directory per evaluation run (default: eval/logs)")
    p.add_argument("--markdown", action="store_true",
                   help="Print a plain markdown table (default when the output is not a terminal)")
    args = p.parse_args()

    if not os.path.isdir(args.log_dir):
        raise SystemExit(f"log directory not found: {args.log_dir}")
    latest = load_results(args.log_dir)
    rows = build_rows(latest)
    runs = running_runs(latest)
    title = "LIBERO success rate (%)"
    rel = os.path.relpath(args.log_dir)
    where = f"results from {rel if not rel.startswith('..') else os.path.abspath(args.log_dir)}"
    if args.markdown or not sys.stdout.isatty():
        print(render_markdown(rows, f"{title}, {where}", COLUMN_ORDER))
        if runs:
            print(render_progress_markdown(runs))
        return

    from rich.console import Console
    from rich.text import Text

    console = Console()
    table, caption = render_rich(rows, title, COLUMN_ORDER)
    console.print(table)
    if caption is not None:
        console.print(caption)
    if runs:
        console.print()
        console.print(render_progress_rich(runs))
    console.print(Text(where, style="dim"))


if __name__ == "__main__":
    main()
