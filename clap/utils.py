# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].
#
# Portions adapted from VLA-0 (NVIDIA, CC BY-NC 4.0); see NOTICE.

"""Shared utilities for CLAP training and inference."""

from __future__ import annotations

import random
import re

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from clap.models.clap_language import summarize_numeric_actions

# ── Image preprocessing ────────────────────────────────────────────────────────


def tile_and_pil(rgb: np.ndarray) -> list[Image.Image]:
    """Tile [history, num_cam, H, W, C] → single horizontally-concatenated PIL image."""
    history, num_cam, H, W, C = rgb.shape
    tiles = [rgb[t, c] for t in range(history) for c in range(num_cam)]
    tiled = np.concatenate(tiles, axis=1)
    return [Image.fromarray(tiled.astype(np.uint8))]


# ── Conversation formatting ────────────────────────────────────────────────────


def format_data(
    system_msg: str,
    images: list[Image.Image],
    instr: str,
    action_txt: str,
) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": instr}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": action_txt}]},
    ]


# ── System messages ────────────────────────────────────────────────────────────


def build_clap_system_message(horizon: int, act_dim: int, num_bins: int = 1000) -> str:
    return (
        f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
        f"Each action has {act_dim} dimensions. "
        f"First describe the motion in natural language inside <think>...</think>. "
        f"Then output exactly {horizon * act_dim} integers (0-{num_bins} each), "
        f"space-separated, for the discretized actions in timestep order. Nothing else."
    )


# ── Action text conversion ─────────────────────────────────────────────────────


def actions_to_text(
    actions: np.ndarray,
    stats: dict,
    num_bins: int = 1000,
) -> list[str]:
    """Normalize and bin [B, horizon, act_dim] actions → space-separated integer strings."""
    min_a = np.array(stats["min"])
    max_a = np.array(stats["max"])
    actions = actions.clip(min_a, max_a)
    normed = (actions - min_a) / (max_a - min_a + 1e-8)
    binned = np.round(normed * num_bins).astype(int).clip(0, num_bins)
    return [
        " ".join(map(str, binned[i].reshape(-1).tolist())) for i in range(len(actions))
    ]


def text_to_actions(
    text: str,
    stats: dict,
    num_bins: int,
    horizon: int,
    act_dim: int,
) -> np.ndarray:
    """Decode a model-generated integer string back to a [horizon, act_dim] action array.

    Inverse of ``actions_to_text``. Robust to extra tokens: for CLAP the
    ``<think>...</think>`` portion is stripped first so its numbers are not parsed
    as actions, and missing/extra integers are padded/truncated to ``horizon*act_dim``.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    nums = re.findall(r"-?\d+", text)
    expected = horizon * act_dim
    vals = [int(x) for x in nums[:expected]]
    if len(vals) < expected:
        vals = vals + [0] * (expected - len(vals))

    min_a = np.array(stats["min"])
    max_a = np.array(stats["max"])
    binned = np.array(vals, dtype=float).clip(0, num_bins).reshape(horizon, act_dim)
    normed = binned / num_bins
    return normed * (max_a - min_a) + min_a


def make_think_texts(
    actions: np.ndarray,
    action_is_absolute: bool = False,
    sum_decimal: str = "0f",
    rotation_precision: int = 10,
) -> list[str]:
    """Generate rule-based natural language descriptions for [B, horizon, act_dim] actions.

    ``action_is_absolute`` must match the action representation of the dataset:
    LIBERO actions are already per-step deltas, so it must be False there
    (True converts absolute poses to deltas first, as needed for the real robot).
    ``sum_decimal`` sets the decimals of the summed translation in cm ("0f" ->
    "338 cm", "1f" -> "338.0 cm"); ``rotation_precision`` rounds rotations to
    the nearest multiple of that many degrees.
    """
    results = []
    for i in range(len(actions)):
        arr = actions[i]
        if action_is_absolute and arr.ndim == 2 and arr.shape[-1] >= 7:
            arr_delta = np.zeros_like(arr)
            arr_delta[1:, :6] = np.diff(arr[:, :6], axis=0)
            arr_delta[:, 6] = arr[:, 6]
            arr = arr_delta
        results.append(
            summarize_numeric_actions(
                arr,
                sum_decimal=sum_decimal,
                include_rotation=True,
                rotation_precision=rotation_precision,
            )
            or ""
        )
    return results


# ── Action mask augmentation ──────────────────────────────────────────────────


def apply_action_mask_aug(
    model_inputs: dict,
    labels: torch.Tensor,
    think_txts: list[str] | None,
    action_aug_per: float,
    tok,
    action_txts: list[str] | None = None,
) -> None:
    """Apply random action-token masking in-place for regularization.

    Matches vla0's action_mask_aug_per behaviour exactly:
    - Uses the CHARACTER length of the integer string as the selection range.
    - Character indices are used directly as token offsets from the action start.
    - 10% of samples skip augmentation entirely.
    The think portion is skipped; only integer tokens are masked.
    """
    question_mark_id = tok.convert_tokens_to_ids("?")
    bs = labels.shape[0]
    seq_len = labels.shape[1]

    for i in range(bs):
        supervised = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(supervised) == 0:
            continue

        if think_txts is not None:
            # Supervised: <think>text</think> {integers}<|im_end|>
            # Find where the integer tokens begin (after think portion).
            think_tok_len = len(
                tok.encode(f"<think>{think_txts[i]}</think> ", add_special_tokens=False)
            )
            if think_tok_len >= len(supervised):
                continue
            action_start_idx = int(supervised[think_tok_len])
            maskable_txt = action_txts[i] if action_txts is not None else ""
        else:
            action_start_idx = int(supervised[0])
            maskable_txt = action_txts[i] if action_txts is not None else ""

        if not maskable_txt:
            continue

        # 10% chance of no augmentation (mirrors vla0)
        if random.random() < 0.1:
            continue

        # vla0 style: character count as range, char indices used as token offsets
        aug_per = random.uniform(0.0, action_aug_per)
        mask_len = int(len(maskable_txt) * aug_per)
        if mask_len == 0:
            continue

        char_indices = random.sample(range(len(maskable_txt)), mask_len)
        abs_indices = [action_start_idx + x for x in char_indices if action_start_idx + x < seq_len]

        if abs_indices:
            mask_pos = torch.tensor(abs_indices, dtype=torch.long, device=labels.device)
            labels[i, mask_pos] = -100
            model_inputs["input_ids"][i, mask_pos] = question_mark_id


# ── Label building ─────────────────────────────────────────────────────────────


def build_inputs_and_labels(
    processor,
    system_msg: str,
    rgb_batch: np.ndarray,
    instrs: list[str],
    action_txts: list[str],
    device: torch.device,
    debug: bool = False,
) -> tuple[dict, torch.Tensor]:
    """Tokenize conversations and build per-sample labels.

    Only the assistant reply is supervised, from ``<think>`` to the end (think
    text, action integers, ``<|im_end|>``); padding, the system and user turns
    and the assistant header are set to -100.

    The processor must pad on the left (needed for batched generation). The
    supervised span is located from the last ``<|im_start|>`` in the padded
    sequence, so it is correct for any amount of left padding.
    """
    bs = len(instrs)
    tok = processor.tokenizer

    imgs = [tile_and_pil(rgb_batch[i]) for i in range(bs)]
    examples = [
        format_data(system_msg, imgs[i], instrs[i], action_txts[i]) for i in range(bs)
    ]

    texts = [
        processor.apply_chat_template(
            ex,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        for ex in examples
    ]
    image_inputs = [process_vision_info(ex)[0] for ex in examples]

    model_inputs = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    model_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()
    }

    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    pad_id = tok.pad_token_id or tok.eos_token_id
    labels = model_inputs["input_ids"].clone()

    # Tokens between <|im_start|> and the first supervised token of the reply.
    # The chat template re-renders the think block as "<think>\n...\n</think>\n\n",
    # so the reply is located by this fixed header rather than by searching for
    # the raw target string.
    header = "assistant\n"
    header_ids = tok.encode(header, add_special_tokens=False)

    for i in range(bs):
        ids = model_inputs["input_ids"][i]
        positions = (ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise RuntimeError(f"sample {i}: no <|im_start|> token in the model input")
        reply_start = int(positions[-1]) + 1 + len(header_ids)
        if ids[int(positions[-1]) + 1 : reply_start].tolist() != header_ids:
            raise RuntimeError(
                f"sample {i}: the assistant turn does not start with {header!r}; "
                "the chat template may have changed"
            )
        labels[i, :reply_start] = -100

    labels[labels == pad_id] = -100

    if debug:
        sep = "=" * 100
        print(
            f"\n{sep}\n[CLAP_DEBUG] batch_size={bs}:  "
            f"dumping full model input and expected output for every sample in this batch"
        )
        for i in range(bs):
            sup_ids = labels[i][labels[i] != -100]
            supervised_text = tok.decode(sup_ids.tolist())
            print("-" * 100)
            print(f"[CLAP_DEBUG][sample {i}] instruction: {instrs[i]}")
            print(
                f"[CLAP_DEBUG][sample {i}] ===== FULL MODEL INPUT "
                f"(chat-templated, system + user + image + assistant) ====="
            )
            print(texts[i])
            print(
                f"[CLAP_DEBUG][sample {i}] ===== EXPECTED OUTPUT "
                f"(assistant target text) ====="
            )
            print(action_txts[i])
            print(
                f"[CLAP_DEBUG][sample {i}] ===== SUPERVISED TOKENS "
                f"(labels != -100, i.e. what the loss actually trains on) ====="
            )
            print(supervised_text)
        print(f"{sep}\n", flush=True)

    return model_inputs, labels.to(device)


# ── trackio sample logging ────────────────────────────────────────────────────


def log_samples_to_trackio(
    global_step: int,
    rgb_batch: np.ndarray,
    instrs: list[str],
    out_ori_act: np.ndarray,
    action_txts: list[str],
    think_txts: list[str] | None,
    sample_rows: list[list],
    columns: list[str],
    num_samples: int = 4,
    pred_actions: list[np.ndarray] | None = None,
    pred_texts: list[str] | None = None,
) -> dict:
    """Build trackio log dict with sample table rows and a preview image.

    Appends rows to *sample_rows* in-place and returns a log dict containing
    a table rebuilt from all rows so far, plus a real-time preview image.

    trackio.Table (unlike wandb.Table) has no add_data(): it only accepts a
    full data array at construction, so the table must be rebuilt from a
    buffered row list on every call rather than mutated incrementally.

    When *pred_actions*/*pred_texts* are provided (model predictions decoded via
    ``generate_pred_actions``), two extra columns are appended per row so the
    predicted action can be compared against the ground-truth ``gt_action``.
    """
    import trackio

    n = min(num_samples, len(instrs))
    for i in range(n):
        pil_img = tile_and_pil(rgb_batch[i])[0]
        gt_str = np.array2string(out_ori_act[i], precision=3, separator=", ")
        row = [
            global_step,
            instrs[i],
            trackio.Image(pil_img),
            gt_str,
            action_txts[i],
        ]
        if think_txts is not None:
            row.append(think_txts[i] if i < len(think_txts) else "")
        if pred_actions is not None:
            pred_str = (
                np.array2string(pred_actions[i], precision=3, separator=", ")
                if i < len(pred_actions)
                else ""
            )
            row.append(pred_str)
            row.append(
                pred_texts[i] if pred_texts is not None and i < len(pred_texts) else ""
            )
        sample_rows.append(row)

    pil_first = tile_and_pil(rgb_batch[0])[0]
    short_act = action_txts[0][:120] + ("…" if len(action_txts[0]) > 120 else "")
    caption = f"step={global_step} | {instrs[0]} | {short_act}"
    return {
        "samples/preview": trackio.Image(pil_first, caption=caption),
        "samples/table": trackio.Table(data=sample_rows, columns=columns),
    }


# ── DataLoader collate ─────────────────────────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        first = batch[0][k]
        if isinstance(first, np.ndarray):
            out[k] = np.stack([s[k] for s in batch])
        elif isinstance(first, torch.Tensor):
            out[k] = torch.stack([s[k] for s in batch])
        else:
            out[k] = [s[k] for s in batch]
    return out


