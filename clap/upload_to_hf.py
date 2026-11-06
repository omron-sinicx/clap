# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""Upload training outputs to HuggingFace Hub.

Epoch checkpoints are published as branches named epoch-N.
The final model is published to the main branch.

Usage:
    # Upload all epochs + final
    python -m clap.upload_to_hf --run outputs/<run> --repo <owner>/<model-name>

    # Upload a specific epoch only
    python -m clap.upload_to_hf --run outputs/<run> --repo <owner>/<model-name> --epoch 3

    # Dry run
    python -m clap.upload_to_hf --run outputs/<run> --repo <owner>/<model-name> --dry-run
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def _collect_files(src_dir: Path, prefix: str = "") -> list[tuple[Path, str]]:
    """Recursively collect (local_path, remote_path) pairs from a directory."""
    files = []
    for f in sorted(src_dir.rglob("*")):
        if f.is_file():
            remote = (
                f"{prefix}/{f.relative_to(src_dir)}"
                if prefix
                else str(f.relative_to(src_dir))
            )
            files.append((f, remote))
    return files


def upload_checkpoint(
    api: HfApi,
    repo_id: str,
    run_dir: Path,
    ckpt_name: str,
    revision: str | None,
    dry_run: bool,
) -> None:
    ckpt_dir = run_dir / ckpt_name
    if not ckpt_dir.exists():
        print(f"[skip] {ckpt_dir} not found")
        return

    # Without the action stats the uploaded model cannot be evaluated, so fail loudly.
    stats_file = run_dir / "dataset_stats.pkl"
    if not stats_file.exists():
        raise FileNotFoundError(
            f"{stats_file} not found; it is written by clap.train at the start of training"
        )
    files = _collect_files(ckpt_dir, prefix="model_final")
    files.append((stats_file, "dataset_stats.pkl"))

    rev_label = revision or "main"
    if dry_run:
        print(f"\n[dry-run] {ckpt_name} → {repo_id}@{rev_label}  ({len(files)} files)")
        for local, remote in files:
            size = local.stat().st_size
            size_str = f"{size/1e6:.1f}MB" if size > 1e6 else f"{size/1e3:.0f}KB"
            print(f"  {remote:60s} ({size_str})")
        return

    print(f"\n[upload] {ckpt_name} → {repo_id}@{rev_label}  ({len(files)} files)")
    if revision and revision != "main":
        api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for local_path, remote_path in files:
            dst = tmp / remote_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(local_path.resolve(), dst)

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message=f"Upload {ckpt_name} as {rev_label}",
        )
    print(f"  → https://huggingface.co/{repo_id}/tree/{rev_label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload checkpoints to HuggingFace Hub"
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Output directory (e.g. outputs/clap_2b)",
    )
    parser.add_argument(
        "--repo", required=True, help="HF repo id (e.g. org/CLAP-Qwen3.5-2B)"
    )
    parser.add_argument(
        "--epoch", default="all", help="Epoch to upload: 'all' or a specific number (e.g. 3)"
    )
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be uploaded"
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    api = HfApi()
    if not args.dry_run:
        print(f"Creating repo: {args.repo} (private={args.private})")
        create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    if args.epoch != "all":
        epoch_num = int(args.epoch)
        upload_checkpoint(
            api,
            args.repo,
            run_dir,
            ckpt_name=f"epoch-{epoch_num}",
            revision=f"epoch-{epoch_num}",
            dry_run=args.dry_run,
        )
    else:
        # All epochs as separate revisions
        epoch_dirs = sorted(
            run_dir.glob("epoch-*"), key=lambda p: int(p.name.split("-")[1])
        )
        for epoch_dir in epoch_dirs:
            revision = epoch_dir.name  # e.g. "epoch-1"
            upload_checkpoint(
                api,
                args.repo,
                run_dir,
                ckpt_name=epoch_dir.name,
                revision=revision,
                dry_run=args.dry_run,
            )

        # Final model → main branch
        upload_checkpoint(
            api,
            args.repo,
            run_dir,
            ckpt_name="final",
            revision=None,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
