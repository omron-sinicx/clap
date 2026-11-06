# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].
#
# Portions adapted from VLA-0 (NVIDIA, CC BY-NC 4.0); see NOTICE.

"""CLAP supervised fine-tuning trainer.

Usage:
    pixi run train clap_0.8b_4gpu            # configs/clap_0.8b_4gpu.yaml
    pixi run train clap_2b 0,1,2,3,4,5,6,7   # override the GPU IDs

    # or directly
    python -m clap.train --config configs/clap_2b.yaml --devices 0,1,2,3,4,5,6,7

Set MASTER_PORT to choose the DDP port (default: a random port in 27000-29999).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import roboverse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import trackio
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from clap.policy import ClapPolicy
from clap.utils import (
    actions_to_text,
    apply_action_mask_aug,
    build_clap_system_message,
    build_inputs_and_labels,
    collate_fn,
    log_samples_to_trackio,
    make_think_texts,
)


def setup_ddp(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # Default NCCL collective timeout (10 min) is too short for the barrier in
    # entry_train() that waits on rank 0's first-time dataset download, which
    # can take longer than that for large datasets.
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=60)
    )


def seed_everything(seed: int, rank: int) -> None:
    """Seed RNGs with a *per-rank* offset.

    CRITICAL: every rank must get a different seed. With
    distributed_sampler=False each GPU shuffles the full dataset on its own,
    so if all ranks shared one seed they would draw identical mini-batches and
    the effective batch would collapse from world_size*batch distinct samples
    to a single mini-batch replicated world_size times (destroying gradient
    diversity). Per-rank offset also decorrelates dropout / augmentation noise
    across GPUs.

    Note: we deliberately do NOT call torch.use_deterministic_algorithms(True)
    because flash-attention and several CUDA kernels have no deterministic variant,
    so runs are reproducible at the seed/data-order level but not bitwise.
    """
    s = seed + rank
    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random inside each DataLoader worker for reproducible aug.

    Must be module-level (not a closure) so it is picklable under the 'spawn'
    start method. torch already assigns each worker a unique base seed derived
    from the DataLoader's generator (seeded per-rank with seed+rank), so
    torch.initial_seed() here is distinct per (rank, worker); we mirror it into
    numpy/random which torch does not seed automatically.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _log_startup(cfg: dict, devices: list[int]) -> None:
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_attn_2_available,
            is_flash_linear_attention_available,
        )
    except ImportError:
        is_flash_attn_2_available = lambda: False
        is_flash_linear_attention_available = lambda: False
        is_causal_conv1d_available = lambda: False

    sep = "=" * 60
    print(sep)
    print("  CLAP Training")
    print(sep)
    print(f"  model        : {cfg['model']}")
    print(f"  output_dir   : {cfg['output_dir']}")
    print(f"  epochs       : {cfg['epochs']}")
    accum = cfg.get("gradient_accumulation_steps", 1)
    print(
        f"  batch_size   : {cfg['batch_size']}  (x{len(devices)} GPUs x{accum} accum"
        f" = {cfg['batch_size'] * len(devices) * accum} samples per step)"
    )
    print(f"  lr           : {cfg['lr']} -> {cfg['lr'] * len(devices):.2e} (scaled)")
    print(f"  grad_accum   : {accum}")
    print(f"  seed         : {cfg.get('seed', 42)}  (per-rank offset: seed+rank)")
    print(f"  dist_sampler : {cfg.get('distributed_sampler', False)}")
    print(f"  save_steps   : {cfg.get('save_steps', 0)}")
    print()
    print(f"  GPUs         : {devices}  (world_size={len(devices)})")
    for i, d in enumerate(devices):
        try:
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    cuda:{d}  {name}  ({mem:.0f} GB)")
        except AssertionError:
            print(f"    cuda:{d}  (not visible in current process)")
    print()
    flash_attn_enabled = cfg.get("flash_attn", True)
    print(
        f"  flash-attn2          : {'on' if flash_attn_enabled else 'off (sdpa)'}  [available={is_flash_attn_2_available()}]"
    )
    print(
        f"  flash-linear-attn    : [available={is_flash_linear_attention_available()}]"
    )
    print(f"  causal-conv1d        : [available={is_causal_conv1d_available()}]")
    print(sep)


def _generate_pred_actions(
    policy: ClapPolicy, rgb_batch, instrs: list[str], num_samples: int
) -> tuple[list[str], list[np.ndarray]]:
    """Greedy-decode the first ``num_samples`` examples for logging (rank 0 only)."""
    n = min(num_samples, len(instrs))
    model = policy.model
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            texts = policy.generate(rgb_batch[:n], instrs[:n])
    finally:
        if was_training:
            model.train()
    return texts, list(policy.decode_actions(texts))


def entry_train(rank: int, cfg: dict, devices: list[int], port: int) -> None:
    world_size = len(devices)
    gpu_id = devices[rank]
    device = torch.device(f"cuda:{gpu_id}")
    ddp = world_size > 1

    setup_ddp(rank, world_size, port)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed, rank)

    output_dir = Path(cfg["output_dir"])
    pix = cfg["img_size"] ** 2 * cfg["history"] * cfg["num_cam"]

    # ── Resume detection ─────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    resume_ckpt = None
    if output_dir.exists():
        epoch_dirs = sorted(
            [d for d in output_dir.iterdir()
             if d.is_dir() and d.name.startswith("epoch-") and d.name[6:].isdigit()],
            key=lambda d: int(d.name[6:]),
        )
        if epoch_dirs:
            resume_ckpt = epoch_dirs[-1]
            start_epoch = int(resume_ckpt.name[6:])
            resume_json = output_dir / "resume.json"
            if resume_json.exists():
                resume_state = json.loads(resume_json.read_text())
                global_step = resume_state.get("global_step", 0)
            if rank == 0:
                print(f"[resume] {resume_ckpt.name} → epoch {start_epoch + 1} / step {global_step}")

    # ── Model & processor ────────────────────────────────────────────────────
    model_path = str(resume_ckpt) if resume_ckpt else cfg["model"]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
        if cfg.get("flash_attn", True)
        else "sdpa",
    )
    model.to(device)
    model.train()
    # Left padding, as in evaluation: batched generation (pred-action logging here,
    # LIBERO rollouts in eval) needs it. Training itself runs with flash_attention_2,
    # which drops padding via the attention mask, so the padding side does not change
    # the loss; build_inputs_and_labels masks the left padding explicitly.
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=pix, max_pixels=pix, padding_side="left", use_fast=False
    )

    if ddp:
        model = DDP(model, device_ids=[gpu_id])

    # ── Dataset ───────────────────────────────────────────────────────────────
    # rank 0 downloads/builds first; other ranks wait so they don't race rank 0
    # for partially-downloaded HuggingFace cache files (dataset repo download
    # is not process-safe across concurrent DDP ranks).
    if rank == 0:
        print("[init] Loading dataset...")
        dataset = roboverse.get_unified_dataset(
            cfg_path=cfg["roboverse_cfg"],
            cfg_opts=cfg.get("roboverse_opts", ""),
        )
    if ddp:
        dist.barrier()
    if rank != 0:
        dataset = roboverse.get_unified_dataset(
            cfg_path=cfg["roboverse_cfg"],
            cfg_opts=cfg.get("roboverse_opts", ""),
        )
    act_stats = dataset.stats["out_ori_act"]

    # Write the action stats up front: evaluation reads <output_dir>/dataset_stats.pkl
    # for any checkpoint under output_dir (epoch-N/, step-latest/, final/), so it must
    # exist before the first checkpoint, not only after the last epoch. The stats are
    # derived from the dataset and do not change during training.
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "dataset_stats.pkl", "wb") as f:
            pickle.dump({"out_ori_act": act_stats}, f)
        print(f"[init] Saved action stats → {output_dir / 'dataset_stats.pkl'}")

    # distributed_sampler=True : partition dataset across GPUs (statistically correct,
    #   each sample seen once per epoch across all GPUs).
    # distributed_sampler=False (default): each GPU iterates the full dataset
    #   independently, matching vla0 repo behaviour (more gradient updates per epoch).
    use_dist_sampler = cfg.get("distributed_sampler", False)
    if use_dist_sampler:
        # DistributedSampler must receive the SAME base seed on every rank: it
        # builds one global permutation and slices it per-rank. set_epoch()
        # reshuffles deterministically as seed+epoch. Per-rank divergence comes
        # from the rank-based slicing, not from the seed.
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
    else:
        sampler = None

    # sampler=None path uses DataLoader's internal RandomSampler driven by this
    # generator. Seed it per-rank so each GPU shuffles the full dataset in a
    # DIFFERENT order (see seed_everything docstring).
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + rank)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=False,
        persistent_workers=(cfg.get("num_workers", 4) > 0),
        generator=loader_generator,
        worker_init_fn=_worker_init_fn,
    )

    system_msg = build_clap_system_message(cfg["horizon"], cfg["act_dim"], cfg["num_bins"])
    # Inference wrapper used only for pred-action logging; shares weights with the
    # model being trained (no extra copy).
    log_policy = ClapPolicy(
        model.module if ddp else model,
        processor,
        act_stats,
        horizon=cfg["horizon"],
        act_dim=cfg["act_dim"],
        num_bins=cfg["num_bins"],
        max_new_tokens=cfg.get("pred_log_max_new_tokens", 512),
        early_stop=False,
        device=device,
    )

    # ── Optimizer (LR scaled linearly with world_size) ────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]) * world_size,
        weight_decay=float(cfg.get("weight_decay", 1e-10)),
    )

    # ── Output dir & trackio (rank 0 only) ───────────────────────────────────
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get("trackio", True):
            # trackio.init() has no `dir` param (unlike wandb.init); it always
            # persists locally under its own cache dir.
            trackio.init(
                project=cfg.get("trackio_project", "clap"),
                name=output_dir.name,
                config=cfg,
            )
            _table_cols = ["step", "instruction", "image", "gt_action", "action_text", "think_text"]
            log_pred_action = cfg.get("log_pred_action", True)
            if log_pred_action:
                _table_cols += ["pred_action", "pred_text"]
            # trackio.Table has no add_data(): rows must be buffered and the
            # table rebuilt from scratch on every log call.
            sample_rows: list[list] = []
        else:
            sample_rows = None

    dist.barrier()

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    clip_norm = cfg.get("clip_grad_norm", 1.0)
    save_steps = cfg.get("save_steps", 0)
    log_sample_steps = cfg.get("log_sample_steps", 500)
    action_aug_per = cfg.get("action_mask_aug_per", 0.0)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = (
            tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}", dynamic_ncols=True)
            if rank == 0
            else loader
        )
        optimizer.zero_grad()
        epoch_losses: list[float] = []
        # Sum of micro-batch losses (each divided by grad_accum) since the last
        # optimizer step; kept on the GPU to avoid a host sync per micro-batch.
        accum_loss = torch.zeros((), device=device)

        for step, batch in enumerate(pbar):
            rgb = batch["rgb"]
            instrs: list[str] = batch["instr"]
            out_ori_act = batch["out_ori_act"]
            if isinstance(out_ori_act, torch.Tensor):
                out_ori_act = out_ori_act.numpy()

            action_txts = actions_to_text(out_ori_act, act_stats, cfg["num_bins"])

            think_txts = make_think_texts(
                out_ori_act,
                action_is_absolute=cfg.get("action_is_absolute", False),
                sum_decimal=str(cfg.get("think_sum_decimal", "0f")),
                rotation_precision=int(cfg.get("think_rotation_precision", 10)),
            )
            full_txts = [f"<think>{t}</think> {a}" for t, a in zip(think_txts, action_txts)]

            # CLAP_DEBUG=1: dump the full model input (system message included) and the
            # expected output for the first batch on rank 0, then continue training.
            debug_io = (
                os.environ.get("CLAP_DEBUG") == "1"
                and rank == 0
                and step == 0
                and epoch == start_epoch
            )

            model_inputs, labels = build_inputs_and_labels(
                processor=processor,
                system_msg=system_msg,
                rgb_batch=rgb,
                instrs=instrs,
                action_txts=full_txts,
                device=device,
                debug=debug_io,
            )

            if action_aug_per > 0:
                apply_action_mask_aug(
                    model_inputs=model_inputs,
                    labels=labels,
                    think_txts=think_txts,
                    action_aug_per=action_aug_per,
                    tok=processor.tokenizer,
                    action_txts=action_txts,
                )
            is_sync_step = (step + 1) % grad_accum == 0
            sync_ctx = nullcontext() if (is_sync_step or not ddp) else model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**model_inputs)
                shift_logits = outputs.logits[:, :-1].float().contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                (loss / grad_accum).backward()
            accum_loss += loss.detach() / grad_accum

            if is_sync_step:
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], clip_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log the loss of the whole optimizer step: mean over all
                # micro-batches (accum_loss) and all ranks, i.e. over
                # world_size * grad_accum * batch_size samples. Logging only
                # rank 0's last micro-batch made the curve ~4x noisier.
                step_loss = accum_loss
                if ddp:
                    dist.all_reduce(step_loss, op=dist.ReduceOp.SUM)
                    step_loss = step_loss / world_size
                step_loss = step_loss.item()
                accum_loss = torch.zeros((), device=device)
                epoch_losses.append(step_loss)

                if rank == 0:
                    if cfg.get("trackio", True):
                        log_dict = {
                            "train/loss": step_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch + 1,
                        }
                        if log_sample_steps > 0 and global_step % log_sample_steps == 0:
                            num_samples = cfg.get("log_num_samples", 4)
                            pred_actions = pred_texts = None
                            if cfg.get("log_pred_action", True):
                                # Optional logging must never crash training: a failure
                                # here (e.g. OOM) on rank 0 only would otherwise hang the
                                # other ranks at the next all-reduce until NCCL timeout.
                                try:
                                    pred_texts, pred_actions = _generate_pred_actions(
                                        log_policy, rgb, instrs, num_samples
                                    )
                                except Exception as e:
                                    print(
                                        f"[warn] pred-action logging skipped at step "
                                        f"{global_step}: {e!r}",
                                        flush=True,
                                    )
                                    pred_texts = pred_actions = None
                            log_dict.update(
                                log_samples_to_trackio(
                                    global_step=global_step,
                                    rgb_batch=rgb,
                                    instrs=instrs,
                                    out_ori_act=out_ori_act,
                                    action_txts=action_txts,
                                    think_txts=think_txts,
                                    sample_rows=sample_rows,
                                    columns=_table_cols,
                                    num_samples=num_samples,
                                    pred_actions=pred_actions,
                                    pred_texts=pred_texts,
                                )
                            )
                        trackio.log(log_dict, step=global_step)
                    pbar.set_postfix(
                        {"loss": f"{step_loss:.4f}", "avg": f"{sum(epoch_losses) / len(epoch_losses):.4f}", "step": global_step}
                    )

                if save_steps > 0 and global_step % save_steps == 0:
                    dist.barrier()
                    if rank == 0:
                        # Overwrite a single rolling checkpoint to limit disk usage.
                        step_dir = output_dir / "step-latest"
                        model_module = model.module if ddp else model
                        model_module.save_pretrained(str(step_dir))
                        processor.save_pretrained(str(step_dir))
                        print(
                            f"\n[ckpt] step-latest (epoch {epoch + 1} step {global_step}) → {step_dir}"
                        )


        # ── Save epoch checkpoint (rank 0 only) ───────────────────────────────
        if rank == 0:
            epoch_dir = output_dir / f"epoch-{epoch + 1}"
            model_module = model.module if ddp else model
            model_module.save_pretrained(str(epoch_dir))
            processor.save_pretrained(str(epoch_dir))
            (output_dir / "resume.json").write_text(
                json.dumps({"epoch": epoch + 1, "global_step": global_step})
            )
            print(f"\n[ckpt] Epoch {epoch + 1} saved → {epoch_dir}")
            if cfg.get("trackio", True) and epoch_losses:
                # One point per epoch: the mean loss over the epoch. (A running mean
                # logged every step resets at each epoch and looks like a staircase.)
                trackio.log(
                    {"train/epoch_loss": sum(epoch_losses) / len(epoch_losses)},
                    step=global_step,
                )
            step_latest = output_dir / "step-latest"
            if step_latest.exists():
                shutil.rmtree(step_latest)

        dist.barrier()

    # ── Final save ────────────────────────────────────────────────────────────
    dist.barrier()
    if rank == 0:
        final_dir = output_dir / "final"
        model_module = model.module if ddp else model
        model_module.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))

        print(f"[done] Final model → {final_dir}")
        if cfg.get("trackio", True):
            trackio.finish()

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAP supervised fine-tuning trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--devices", type=str, default=None, help="Override GPU IDs (e.g. '0,1,2,3')"
    )
    parser.add_argument(
        "--flash-attn",
        dest="flash_attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable flash attention 2 (overrides config)",
    )
    parser.add_argument(
        "--no-trackio", dest="no_trackio", action="store_true", help="Disable trackio logging"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["lr"] = float(cfg["lr"])
    # Older configs carry `method: clap`; that is still accepted. The VLA-0 baseline
    # is no longer supported here (use https://github.com/NVlabs/vla0).
    if cfg.pop("method", "clap") != "clap":
        parser.error("only `method: clap` is supported; train VLA-0 with https://github.com/NVlabs/vla0")

    if args.devices:
        cfg["devices"] = args.devices
    if args.flash_attn is not None:
        cfg["flash_attn"] = args.flash_attn
    if args.no_trackio:
        cfg["trackio"] = False

    devices = [int(x) for x in str(cfg.get("devices", "0")).split(",")]
    # A random port avoids clashes between jobs on the same node; MASTER_PORT overrides it.
    port = int(os.environ.get("MASTER_PORT") or random.randint(27000, 29999))

    _log_startup(cfg, devices)
    mp.spawn(entry_train, args=(cfg, devices, port), nprocs=len(devices), join=True)


if __name__ == "__main__":
    main()
