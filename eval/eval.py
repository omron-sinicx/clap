# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].
#
# Follows the LIBERO evaluation protocol of VLA-0 and openpi (settling steps, camera
# flip, per-suite episode lengths); the batched design is inspired by RLinf. See NOTICE.

"""Batched LIBERO evaluation for CLAP policies (standalone, no Ray).

  - **Batching**: N LIBERO envs run in parallel with SubprocVectorEnv, and one
    generate() call predicts actions for all N envs at once.
  - **Early stopping**: generation stops once horizon * act_dim action values have
    been produced (the parser only reads that many, so accuracy is unaffected).
  - **torch.compile** (--use-torch-compile): fuses kernels to cut the per-token
    decode overhead (mode=default, dynamic=True).
  - **Timing**: CLAP_TIMING=1 prints a preprocess / prefill / decode / parse breakdown.
  - **Task sharding**: --num-shards / --shard split the tasks of a suite across GPUs
    (one process per GPU).

The policy itself is clap.policy.ClapPolicy; this file only drives the LIBERO envs.
Normally this script is launched through eval/run_eval.sh (pixi run eval).

Usage:
    # Single GPU
    pixi run -e eval-libero python eval/eval.py \
        --model-path outputs/<run>/epoch-1 --task-suite libero_object --num-envs 25

    # 4 GPUs (task sharding): one process per GPU with a different --shard
    for i in 0 1 2 3; do
      pixi run -e eval-libero python eval/eval.py --model-path <...> \
        --task-suite libero_object --num-envs 25 --num-shards 4 --shard $i \
        --device cuda:$i &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from typing import TYPE_CHECKING

import numpy as np

# LIBERO
# Keep the module-level imports light: SubprocVectorEnv starts its simulator
# workers with "spawn", and every worker re-imports this module. torch /
# transformers (clap.policy) and libero.libero.benchmark (torch) are imported
# lazily inside the functions that run in the main process only; importing them
# in every worker made starting hundreds of simulators take minutes.
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv  # noqa: E402

if TYPE_CHECKING:
    from clap.policy import ClapPolicy


LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ─────────────────────────────────────────────────────────────────────────────
# LIBERO env (vectorized)
# ─────────────────────────────────────────────────────────────────────────────
def _get_task(task_suite_name: str, task_index: int):
    from libero.libero import benchmark

    ts = benchmark.get_benchmark_dict()[task_suite_name]()
    task = ts.tasks[task_index]
    init_states = ts.get_task_init_states(task_index)
    return task, init_states, task.language


def _make_env_fn(task):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    def _fn():
        return OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# Progress-log helpers (one line per update)
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    """Duplicate stdout/stderr to the terminal and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self._streams:
            st.flush()


def _bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _fmt_t(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_hms(sec: float) -> str:
    """Always h:mm:ss (hours are never omitted); used for durations in the summary JSON."""
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _lens_str(lens: list[int]) -> str:
    return (f"mean={int(np.mean(lens))} min={min(lens)} max={max(lens)}"
            if lens else "n/a")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop (one task)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_task(
    policy: ClapPolicy,
    task_suite_name: str,
    task_index: int,
    num_envs: int,
    action_horizon: int,
    frame_skip: int,
    max_steps: int,
    seed: int,
    shard: int = 0,
    crop_img: float = 0.875,
    img_size: int = 224,
    on_round_end=None,
) -> tuple[int, int]:
    """Evaluate one task on all its init states. ``on_round_end(episodes_done)``
    is called after every round of num_envs episodes (used for progress reporting)."""
    from clap.policy import libero_obs_to_images

    task, init_states, instruction = _get_task(task_suite_name, task_index)

    # Shard by task: one shard evaluates all init states of a task (states are not
    # split), so batches stay full and 50 states never need to divide by num_shards.

    tag = f"T{task_index:02d}"
    name = task.name[:34]
    print(f"[Eval] {task_suite_name} {tag} '{instruction}' | states={len(init_states)} "
          f"(shard {shard}) num_envs={num_envs}", flush=True)

    # Build num_envs envs once and reuse them for the whole task
    venv = SubprocVectorEnv([_make_env_fn(task) for _ in range(num_envs)])
    venv.seed(seed)

    n_success = 0
    n_eval = 0
    n_timeout = 0
    ep_lens: list[int] = []
    num_rounds = math.ceil(len(init_states) / num_envs)
    log_every = max(1, max_steps // 8)  # log progress about 8 times per episode

    for r in range(num_rounds):
        batch_states = init_states[r * num_envs:(r + 1) * num_envs]
        n = len(batch_states)  # active envs this round

        venv.reset()
        # Set one init state per env (the last state pads a partial round)
        pad_states = list(batch_states) + [batch_states[-1]] * (num_envs - n)
        obs = venv.set_init_state(np.asarray(pad_states))

        done = np.zeros(num_envs, dtype=bool)
        success = np.zeros(num_envs, dtype=bool)
        fin_step = np.full(num_envs, -1, dtype=int)  # step at which each env finished
        action_chunk = None
        action_i = 0
        t0 = time.perf_counter()
        step_real = 0  # policy steps, excluding the frame_skip warm-up

        for t in range(max_steps + frame_skip):
            # The first frame_skip steps send a dummy action to let the simulation settle
            if t < frame_skip:
                obs, _, _, _ = venv.step(np.asarray([LIBERO_DUMMY_ACTION] * num_envs))
                continue

            # Query the policy again once the action chunk is used up
            if action_chunk is None or action_i >= action_horizon:
                images = libero_obs_to_images(obs, crop_img=crop_img, img_size=img_size)  # [num_envs, 2, H, W, 3]
                action_chunk = policy.predict_batch(images[:n], [instruction] * n)  # [n, horizon, act_dim]
                if n < num_envs:  # pad the last, partial round
                    pad = np.repeat(action_chunk[-1:], num_envs - n, axis=0)
                    action_chunk = np.concatenate([action_chunk, pad], axis=0)
                action_i = 0

            act = action_chunk[:, action_i, :].copy()  # [num_envs, act_dim]
            act[:, -1] = np.where(act[:, -1] > 0, 1.0, -1.0)  # gripper to -1 / +1
            obs, _, dones, _ = venv.step(act)
            action_i += 1
            step_real += 1

            # Success check
            succ = np.asarray(venv.check_success())
            for i in range(n):
                if not done[i] and (succ[i] or dones[i]):
                    success[i] = bool(succ[i])
                    done[i] = True
                    fin_step[i] = step_real

            all_done = bool(done[:n].all())
            if step_real % log_every == 0 or all_done:
                nd = int(done[:n].sum())
                ns = int(success[:n].sum())
                frac = step_real / max_steps
                cum_s, cum_n = n_success + ns, n_eval + nd
                print(f"[Eval] {task_suite_name} {tag} {name} | round {r + 1}/{num_rounds} "
                      f"|{_bar(frac)}| step={step_real}/{max_steps} ({frac:.0%}) "
                      f"done={nd}/{n} round_sr={ns}/{max(1, nd)}={ns / max(1, nd):.1%} "
                      f"task_sr={cum_s}/{max(1, cum_n)}={cum_s / max(1, cum_n):.1%} "
                      f"elapsed={_fmt_t(time.perf_counter() - t0)}", flush=True)
            if all_done:
                break

        # Per-round summary
        r_succ = int(success[:n].sum())
        r_to = int(n - done[:n].sum())  # envs that did not finish within max_steps = timeout
        r_lens = [int(x) for x in fin_step[:n] if x > 0]
        n_success += r_succ
        n_eval += n
        n_timeout += r_to
        ep_lens.extend(r_lens)
        print(f"[Eval] {tag} {name} round {r + 1}/{num_rounds} done | ep {n}/{n} | "
              f"succ {r_succ / max(1, n):.1%} ({r_succ}/{n}) | ep_len {_lens_str(r_lens)} | "
              f"timeout={r_to}/{n} | elapsed {_fmt_t(time.perf_counter() - t0)}", flush=True)
        if on_round_end is not None:
            on_round_end(n_eval)

    venv.close()
    sr = n_success / max(1, n_eval)
    print(f"[Eval] {tag} {name} DONE | sr {sr:.1%} ({n_success}/{n_eval}) | "
          f"ep_len {_lens_str(ep_lens)} | timeout={n_timeout}/{n_eval}", flush=True)
    return n_success, n_eval


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    from clap.policy import DEFAULT_HF_SUBFOLDER, ClapPolicy
    from libero.libero import benchmark

    t_start = time.time()  # start time of this shard (for the duration field)

    # SubprocVectorEnv forks by default, but this process initializes CUDA when the
    # model is loaded; forked children would inherit a broken GPU/EGL state and
    # robosuite's eglCreateContext fails with EGL_BAD_ALLOC. Use spawn so children
    # start fresh. This must be set before the model is loaded.
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    p = argparse.ArgumentParser(description="Batched LIBERO eval for CLAP")
    p.add_argument("--model-path", required=True,
                   help="Local checkpoint (e.g. outputs/<run>/epoch-1) or HF repo id")
    p.add_argument("--stats-path", default=None, help="dataset_stats.pkl (default: the run directory for a local checkpoint, the repo root on HF)")
    p.add_argument("--hf-revision", default=None,
                   help="HF revision (branch / tag), e.g. epoch-1 or main. Ignored for local paths")
    p.add_argument("--hf-subfolder", default=DEFAULT_HF_SUBFOLDER,
                   help="Subfolder holding the model in the HF repo (upload_to_hf.py uses model_final). Ignored for local paths")
    p.add_argument("--task-suite", default="libero_object")
    # Kept so existing scripts that pass --method clap keep working; CLAP is the only method.
    p.add_argument("--method", default="clap", choices=["clap"], help=argparse.SUPPRESS)
    p.add_argument("--num-envs", type=int, default=25, help="Parallel envs per process (= inference batch size)")
    p.add_argument("--action-horizon", type=int, default=8, help="Steps executed from each predicted action chunk")
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--act-dim", type=int, default=7)
    p.add_argument("--num-bins", type=int, default=1000)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--crop-img", type=float, default=0.875,
                   help="Center-crop ratio; must match IMAGE.crop_img used in training (1.0 disables cropping)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=0, help="0 uses the task suite default")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    p.add_argument("--use-torch-compile", action="store_true")
    p.add_argument("--torch-compile-mode", default="default")
    p.add_argument("--num-shards", type=int, default=1, help="Split the tasks across processes (shard i runs tasks[i::num_shards])")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--log-dir", default="eval/logs")
    args = p.parse_args()
    if not 1 <= args.action_horizon <= args.horizon:
        p.error(f"--action-horizon must be between 1 and --horizon ({args.horizon}), got {args.action_horizon}")

    # Interactive runs (TTY) also tee the console output to a timestamped log file.
    # When stdout is already redirected to a file (e.g. by run_eval.sh) that file
    # is the log, so no tee is set up.
    os.makedirs(args.log_dir, exist_ok=True)
    if sys.stdout.isatty():
        _stamp = time.strftime("%Y%m%d_%H%M%S")
        _console_path = os.path.join(
            args.log_dir, f"eval_{args.task_suite}_shard{args.shard}of{args.num_shards}_{_stamp}.log")
        _console_fp = open(_console_path, "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _console_fp)
        sys.stderr = _Tee(sys.__stderr__, _console_fp)
        print(f"[Eval] console log -> {_console_path}", flush=True)

    is_local = os.path.exists(args.model_path)
    revision = None if is_local else args.hf_revision
    policy = ClapPolicy.from_pretrained(
        args.model_path,
        stats_path=args.stats_path,
        revision=revision,
        subfolder=args.hf_subfolder,
        device=args.device,
        img_size=args.img_size,
        use_torch_compile=args.use_torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        horizon=args.horizon,
        act_dim=args.act_dim,
        num_bins=args.num_bins,
        max_new_tokens=args.max_new_tokens,
        early_stop=args.early_stop,
    )
    method = "clap"  # recorded in the result JSON for compatibility with earlier runs
    print(f"[Eval] model={'HF:' + args.model_path if not is_local else args.model_path} "
          f"revision={revision} method={method}", flush=True)

    max_steps = args.max_steps or LIBERO_TASK_SUITE_MAX_STEPS.get(args.task_suite, 300)
    ts = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = len(ts.tasks)
    print(f"task_suite={args.task_suite} tasks={n_tasks} method={method} "
          f"early_stop={args.early_stop} compile={args.use_torch_compile}")

    # Shard by task: shard i runs tasks i, i+num_shards, ... (init states are not split).
    my_task_indices = list(range(args.shard, n_tasks, args.num_shards))
    print(f"[Eval] shard {args.shard}/{args.num_shards} handles {len(my_task_indices)} tasks: "
          f"{my_task_indices}", flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    out = os.path.join(args.log_dir, f"results_{args.task_suite}_shard{args.shard}of{args.num_shards}.json")
    total_s, total_n = 0, 0
    per_task = {}
    # Episodes of this shard, for progress reporting (every task has one episode per init state).
    episodes_total = sum(len(ts.get_task_init_states(ti)) for ti in my_task_indices)
    episodes_in_task = 0  # finished episodes of the task currently running

    def write_results(finished: bool) -> None:
        """(Re)write this shard's result JSON.

        Called after every round of episodes, so progress and partial results can
        be previewed while the evaluation runs (pixi run libero-table), and once
        more with finished=True at the end. Success counts only include finished
        tasks. aggregate.py only accepts finished shards. The file is replaced
        atomically so readers never see a half-written JSON.
        """
        sr = total_s / max(1, total_n)
        result = {
            "model": args.model_path,
            "revision": revision,
            "method": method,
            "task_suite": args.task_suite,
            "num_envs": args.num_envs,
            "action_horizon": args.action_horizon,
            "crop_img": args.crop_img,
            "use_torch_compile": args.use_torch_compile,
            "early_stop": args.early_stop,
            "seed": args.seed,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": _fmt_hms(time.time() - t_start),
            "num_shards": args.num_shards,
            "shard": args.shard,
            "finished": finished,
            "tasks_done": len(per_task),
            "tasks_total": len(my_task_indices),
            "episodes_done": total_n + episodes_in_task,
            "episodes_total": episodes_total,
            "success": total_s,
            "total": total_n,
            "success_rate": sr,
            "per_task": per_task,
        }
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, out)

    def on_round_end(episodes_done: int) -> None:
        nonlocal episodes_in_task
        episodes_in_task = episodes_done
        write_results(finished=False)

    write_results(finished=False)
    for ti in my_task_indices:
        s, nn = evaluate_task(
            policy, args.task_suite, ti, args.num_envs, args.action_horizon,
            args.frame_skip, max_steps, args.seed, args.shard,
            crop_img=args.crop_img, img_size=args.img_size,
            on_round_end=on_round_end,
        )
        episodes_in_task = 0
        total_s += s; total_n += nn
        per_task[ts.tasks[ti].name] = {"success": s, "total": nn,
                                       "instruction": ts.tasks[ti].language}
        write_results(finished=False)

    sr = total_s / max(1, total_n)
    # Benchmark table
    line = "=" * 72
    print(f"\n{line}")
    print(f"Benchmark: {args.task_suite}  (shard {args.shard}/{args.num_shards})")
    print(line)
    for v in per_task.values():
        instr = v.get("instruction") or ""
        tsr = v["success"] / max(1, v["total"])
        print(f"  {instr[:52]:<52} {tsr:6.1%}  ({v['success']}/{v['total']})")
    print("-" * 72)
    print(f"  {'OVERALL':<52} {sr:6.1%}  ({total_s}/{total_n})")
    print(line)

    write_results(finished=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
