# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Collect per-epoch LIBERO results of outputs/clap_<size>/epoch-<k> into a CSV.

Reads eval/logs/*/results_<suite>_AGGREGATE.json (the latest run per checkpoint and
suite) and the checkpoint modification times (training hours since the start of the
run, i.e. since dataset_stats.pkl was written).

    pixi run --script docs/assets/collect_training_curve.py > docs/assets/training_curve.csv
    pixi run --script docs/assets/collect_training_curve.py --sizes 4b --max-epoch 10
"""

import argparse
import csv
import glob
import json
import os
import sys

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
STEPS_PER_EPOCH = 2136  # 273,465 frames / 128 samples per step


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["0.8b", "2b", "4b"], help="runs outputs/clap_<size>")
    p.add_argument("--max-epoch", type=int, default=10)
    args = p.parse_args()

    latest = {}
    for path in glob.glob("eval/logs/*/results_*_AGGREGATE.json"):
        d = json.load(open(path))
        model = d["model"].rstrip("/")
        if not model.startswith("outputs/clap_"):
            continue
        key = (model, d["task_suite"])
        if key not in latest or d.get("aggregated_at", "") > latest[key].get("aggregated_at", ""):
            latest[key] = d

    w = csv.writer(sys.stdout)
    w.writerow(["model", "epoch", "steps", "train_hours", "suite", "success_rate"])
    for size in args.sizes:
        run = f"outputs/clap_{size}"
        if not os.path.isdir(run):
            continue
        t0 = os.path.getmtime(f"{run}/dataset_stats.pkl")
        for epoch in range(1, args.max_epoch + 1):
            ckpt = f"{run}/epoch-{epoch}"
            if not os.path.isdir(ckpt):
                continue
            hours = (os.path.getmtime(f"{ckpt}/model.safetensors") - t0) / 3600
            for suite in SUITES:
                d = latest.get((ckpt, suite))
                if d is not None:
                    w.writerow([f"CLAP-{size.upper()}", epoch, epoch * STEPS_PER_EPOCH,
                                f"{hours:.2f}", suite, f"{100 * d['success_rate']:.1f}"])


if __name__ == "__main__":
    main()
