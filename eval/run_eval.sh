#!/bin/bash
# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

# Sharded LIBERO evaluation (no Ray): the tasks of a suite are split across GPUs.
# Shard i runs tasks[i::NGPU], with inference and rendering pinned to GPU ${GPU_IDS[i]}.
#
# Usage:
#   bash eval/run_eval.sh <model_path> [task_suite] [num_gpus|gpu_id_list] [num_envs]
# The 3rd argument is either a GPU count (e.g. 4 -> GPUs 0,1,2,3) or a
# comma-separated list of GPU IDs (e.g. 2,3,4,5).
# Example (torch.compile + timing):
#   EXTRA="--use-torch-compile" CLAP_TIMING=1 \
#     bash eval/run_eval.sh outputs/<model>/final libero_object 4 25
# Example (GPUs 2,3,4,5):
#   bash eval/run_eval.sh outputs/<model>/final libero_object 2,3,4,5 25
#
# Note: some clusters set CUDA_VISIBLE_DEVICES to GPU UUIDs, which robosuite's EGL
# backend cannot parse. This script therefore leaves CUDA_VISIBLE_DEVICES alone and
# selects the GPU with --device cuda:${GPU_IDS[i]} (torch) and
# MUJOCO_EGL_DEVICE_ID=${GPU_IDS[i]} (rendering).
set -eu

MODEL="$1"
SUITE="${2:-libero_object}"
GPU_SPEC="${3:-4}"
NENV="${4:-25}"
EXTRA="${EXTRA:-}"

# A comma-separated GPU_SPEC is a list of GPU IDs; a plain number N means GPUs 0..N-1.
if [[ "$GPU_SPEC" == *,* ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$GPU_SPEC"
else
    GPU_IDS=()
    for ((i = 0; i < GPU_SPEC; i++)); do GPU_IDS+=("$i"); done
fi
NGPU="${#GPU_IDS[@]}"
LOG_DIR="${LOG_DIR:-eval/logs}"
# Optional flag (omitted when empty): HF revision.
HF_REVISION="${HF_REVISION:-}"
mkdir -p "$LOG_DIR"

OPT=""
[ -n "$HF_REVISION" ] && OPT="$OPT --hf-revision $HF_REVISION"

# A nested `pixi run` does not inherit the task's default-environment (it would
# pick up whatever environment is active in the calling shell), so pass -e explicitly.
ENV="eval-libero"

# One directory per evaluation run (model / revision / suite / timestamp) that
# collects the logs, results and aggregate of all shards.
STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')"
REV_TAG=""
[ -n "$HF_REVISION" ] && REV_TAG="_${HF_REVISION}"
RUN_DIR="${LOG_DIR}/${MODEL_TAG}${REV_TAG}_${SUITE}_${STAMP}"
mkdir -p "$RUN_DIR"

echo "model=$MODEL suite=$SUITE gpus=$NGPU num_envs=$NENV revision='$HF_REVISION' extra='$EXTRA'"
echo "run dir: $RUN_DIR"

pids=()
for i in $(seq 0 $((NGPU - 1))); do
    gpu="${GPU_IDS[$i]}"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" CLAP_TIMING="${CLAP_TIMING:-0}" \
        PYTHONFAULTHANDLER=1 \
        pixi run -e "$ENV" python eval/eval.py \
            --model-path "$MODEL" --task-suite "$SUITE" \
            --num-envs "$NENV" --num-shards "$NGPU" --shard "$i" \
            --device "cuda:$gpu" \
            $OPT --log-dir "$RUN_DIR" $EXTRA \
            > "${RUN_DIR}/shard${i}.log" 2>&1 &
    pids+=($!)
    echo "  launched shard $i on GPU $gpu (pid ${pids[-1]}) -> ${RUN_DIR}/shard${i}.log"
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
done

echo "=== all shards done (rc=$rc). aggregating ==="
pixi run -e "$ENV" python eval/aggregate.py --task-suite "$SUITE" --num-shards "$NGPU" --log-dir "$RUN_DIR"
pixi run -e "$ENV" python eval/export_results_csv.py --log-dir "$LOG_DIR" -o "${LOG_DIR}/results_summary.csv"
echo "=== results in: $RUN_DIR ==="
echo "  summary: ${RUN_DIR}/results_${SUITE}_AGGREGATE.json"
echo "  csv:     ${LOG_DIR}/results_summary.csv"
exit $rc
