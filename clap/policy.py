# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].
#
# Portions adapted from VLA-0 (NVIDIA, CC BY-NC 4.0); see NOTICE.

"""CLAP inference policy: images + instructions -> action chunks.

Simulator-agnostic: depends only on torch / transformers / qwen-vl-utils.
The LIBERO-specific observation conversion lives in ``libero_obs_to_images``.

"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import center_crop
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

from clap.inference_timer import FirstTokenTimer, get_timer, timing_enabled
from clap.utils import (
    build_clap_system_message,
    format_data,
    text_to_actions,
    tile_and_pil,
)

# CLAP_EVAL_DEBUG=1: print instruction / raw model output / parsed actions per sample.
_DEBUG = os.environ.get("CLAP_EVAL_DEBUG", "0") == "1"

STATS_FILENAME = "dataset_stats.pkl"
DEFAULT_HF_SUBFOLDER = "model_final"  # where upload_to_hf.py puts the weights


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping (vectorized on GPU)
# ─────────────────────────────────────────────────────────────────────────────
class ActionCompleteStoppingCriteria(StoppingCriteria):
    """Stop each sequence once ``n_values`` integers have been generated.

    A value is a run of digit tokens; it is counted when the run ends. With
    ``reasoning=True`` (CLAP) counting starts after the first ``</think>``.
    All bookkeeping stays on the GPU (no ``.item()`` sync).
    """

    def __init__(self, tokenizer, n_values: int, reasoning: bool = False):
        self.n_values = int(n_values)
        self.reasoning = bool(reasoning)
        digit_ids = []
        for i in range(10):
            ids = tokenizer.encode(str(i), add_special_tokens=False)
            if ids:
                digit_ids.append(ids[0])
        self._digit_id_list = sorted(set(digit_ids))
        self._think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self._digit_ids_t = self._think_close_t = None
        self._value_count = self._prev_is_digit = self._passed_think = self._done = None

    def __call__(self, input_ids, scores, **kwargs):
        bs, cur_len = input_ids.shape
        device = input_ids.device
        if self._done is None:
            self._digit_ids_t = torch.tensor(self._digit_id_list, device=device, dtype=input_ids.dtype)
            self._think_close_t = (
                torch.tensor(self._think_close_ids, device=device, dtype=input_ids.dtype)
                if self._think_close_ids
                else None
            )
            self._value_count = torch.zeros(bs, dtype=torch.long, device=device)
            self._prev_is_digit = torch.zeros(bs, dtype=torch.bool, device=device)
            self._passed_think = torch.full((bs,), not self.reasoning, dtype=torch.bool, device=device)
            self._done = torch.zeros(bs, dtype=torch.bool, device=device)
        new_tok = input_ids[:, -1]
        is_digit = torch.isin(new_tok, self._digit_ids_t)
        was_passed = self._passed_think.clone()
        if self.reasoning and self._think_close_t is not None:
            k = int(self._think_close_t.numel())
            if cur_len >= k:
                matches = (input_ids[:, -k:] == self._think_close_t).all(dim=1)
                self._passed_think = self._passed_think | matches
        active = was_passed & ~self._done
        completed = active & self._prev_is_digit & ~is_digit
        self._value_count = self._value_count + completed.long()
        self._done = self._done | (self._value_count >= self.n_values)
        self._prev_is_digit = torch.where(was_passed, is_digit, torch.zeros_like(is_digit))
        return self._done


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def center_crop_resize(rgb: np.ndarray, crop_img: float = 0.875, img_size: int = 224) -> np.ndarray:
    """Center-crop by ratio ``crop_img`` then resize to ``img_size`` (bilinear).

    ``rgb``: [..., H, W, 3] uint8. Mirrors the training-time transform in
    ``roboverse.unifiers.image_unifier`` (IMAGE.crop_img / IMAGE.img_size).
    ``crop_img=1.0`` disables cropping.
    """
    if crop_img >= 1.0:
        return rgb
    lead = rgb.shape[:-3]
    h, w, c = rgb.shape[-3:]
    crop_h, crop_w = int(h * crop_img), int(w * crop_img)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).reshape(-1, h, w, c).permute(0, 3, 1, 2).float()
    t = center_crop(t, (crop_h, crop_w))
    if (crop_h, crop_w) != (img_size, img_size):
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.round().clamp(0, 255).to(torch.uint8)
    return t.permute(0, 2, 3, 1).reshape(*lead, img_size, img_size, c).contiguous().numpy()


def libero_obs_to_images(obs_list, crop_img: float = 0.875, img_size: int = 224) -> np.ndarray:
    """LIBERO observations (one dict per env) -> [n, num_cam=2, img_size, img_size, 3] uint8.

    Both cameras (agentview, eye-in-hand) are rotated 180 degrees to match the
    orientation of the ``physical-intelligence/libero`` training data.
    """
    imgs = []
    for o in obs_list:
        agent = o["agentview_image"][::-1, ::-1]
        wrist = o["robot0_eye_in_hand_image"][::-1, ::-1]
        imgs.append(np.stack([agent, wrist], axis=0))
    rgb = np.asarray(imgs, dtype=np.uint8)
    return center_crop_resize(rgb, crop_img=crop_img, img_size=img_size)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_action_stats(
    model_path: str,
    stats_path: str | None = None,
    revision: str | None = None,
) -> dict:
    """Load the action min/max stats (``dataset_stats.pkl['out_ori_act']``).

    Lookup order: ``stats_path`` if given; for a local checkpoint
    (``outputs/<run>/<epoch>``) the run directory; otherwise the root of the
    Hugging Face repo ``model_path``.
    """
    hint = (
        f"{STATS_FILENAME} holds the action normalization stats written by clap.train "
        "into the run directory; pass its location with --stats-path"
    )
    if stats_path is None:
        if os.path.exists(model_path):
            stats_path = str(Path(model_path).parent / STATS_FILENAME)
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError, HFValidationError

            try:
                stats_path = hf_hub_download(
                    repo_id=model_path, filename=STATS_FILENAME, revision=revision
                )
            except HFValidationError as e:
                raise FileNotFoundError(
                    f"{model_path!r} is neither an existing local checkpoint directory "
                    "nor a Hugging Face repo id"
                ) from e
            except EntryNotFoundError as e:
                raise FileNotFoundError(
                    f"{STATS_FILENAME} not found in the Hugging Face repo {model_path!r} "
                    f"(revision {revision or 'main'}). {hint}."
                ) from e
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path} not found. {hint}.")
    with open(stats_path, "rb") as f:
        return pickle.load(f)["out_ori_act"]


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────
class ClapPolicy:
    """Batched VLM policy: ``predict_batch(images, instructions) -> [B, horizon, act_dim]``.

    Construct from a checkpoint with :meth:`from_pretrained`, or wrap an
    already-loaded model/processor (e.g. during training) with the constructor.
    The processor must use left padding.
    """

    def __init__(
        self,
        model,
        processor,
        stats: dict,
        horizon: int = 8,
        act_dim: int = 7,
        num_bins: int = 1000,
        max_new_tokens: int = 512,
        early_stop: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.stats = stats
        self.horizon = horizon
        self.act_dim = act_dim
        self.num_bins = num_bins
        self.max_new_tokens = max_new_tokens
        self.early_stop = early_stop
        self.device = torch.device(device) if device is not None else model.device
        self.system_msg = build_clap_system_message(horizon, act_dim, num_bins)
        self._compiled = False
        self._orig_forward = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        stats_path: str | None = None,
        revision: str | None = None,
        subfolder: str | None = DEFAULT_HF_SUBFOLDER,
        device: torch.device | str = "cuda:0",
        img_size: int = 224,
        num_cam: int = 2,
        attn_implementation: str = "sdpa",
        use_torch_compile: bool = False,
        torch_compile_mode: str = "default",
        **kwargs,
    ) -> "ClapPolicy":
        """Load a policy from a local checkpoint dir or a Hugging Face repo id.

        For HF repos, ``revision`` selects the branch (e.g. ``epoch-8``; ``None`` = main)
        and ``subfolder`` the weights folder. Both are ignored for local paths.
        Remaining ``kwargs`` (horizon, act_dim, num_bins, max_new_tokens,
        early_stop) are forwarded to the constructor.
        """
        is_local = os.path.exists(model_path)
        hf_kw = {}
        if not is_local:
            if revision is not None:
                hf_kw["revision"] = revision
            if subfolder:
                hf_kw["subfolder"] = subfolder
        stats = load_action_stats(model_path, stats_path, revision=None if is_local else revision)

        device = torch.device(device)
        pix = img_size**2
        # Cameras are tiled side by side, so allow up to num_cam * img_size^2 pixels.
        # Left padding is required for batched generation (and matches training).
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=pix, max_pixels=pix * num_cam, padding_side="left", **hf_kw
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            # FA2 was slower than sdpa for this (short, batched) generation workload.
            attn_implementation=attn_implementation,
            device_map={"": device.index if device.index is not None else 0},
            **hf_kw,
        )
        model.eval()
        policy = cls(model, processor, stats, device=device, **kwargs)
        if use_torch_compile:
            policy.enable_torch_compile(torch_compile_mode)
        return policy

    # ── torch.compile ────────────────────────────────────────────────────────
    def enable_torch_compile(self, mode: str = "default") -> None:
        try:
            self._orig_forward = self.model.forward
            ckw = {"mode": mode, "fullgraph": False}
            if mode != "reduce-overhead":
                ckw["dynamic"] = True  # one graph for variable lengths; avoids recompiles
            self.model.forward = torch.compile(self.model.forward, **ckw)
            self._compiled = True
            print(f"[ClapPolicy] torch.compile enabled ({ckw}); first calls warm up for a few minutes.")
        except Exception as e:
            self._compiled = False
            print(f"[ClapPolicy] torch.compile setup skipped: {e}")

    def _disable_compile(self) -> None:
        if self._compiled and self._orig_forward is not None:
            self.model.forward = self._orig_forward
            self._compiled = False
            print("[ClapPolicy] torch.compile disabled -> eager fallback")

    # ── inference ────────────────────────────────────────────────────────────
    def _build_inputs(self, images: np.ndarray, instructions: list[str]) -> dict:
        """images: [B, num_cam, H, W, 3] or [B, history, num_cam, H, W, 3] uint8."""
        texts, image_inputs = [], []
        for i, instr in enumerate(instructions):
            rgb = images[i] if images[i].ndim == 5 else images[i][np.newaxis]
            conversation = format_data(self.system_msg, tile_and_pil(rgb), instr, "")[:-1]
            texts.append(
                self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )
            image_inputs.append(process_vision_info(conversation)[0])
        inputs = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    @torch.inference_mode()
    def generate(self, images: np.ndarray, instructions: list[str]) -> list[str]:
        """Return the raw decoded completions (language plan + integers for CLAP)."""
        timer = get_timer()
        timer.set_device(self.device)

        with timer.section("preprocess"):
            inputs = self._build_inputs(images, instructions)
        input_len = inputs["input_ids"].shape[1]

        stopping = []
        if self.early_stop:
            stopping.append(
                ActionCompleteStoppingCriteria(
                    self.processor.tokenizer,
                    n_values=self.horizon * self.act_dim,
                    reasoning=True,
                )
            )
        ft = gen_t0 = None
        if timing_enabled():
            ft = FirstTokenTimer()
            stopping.append(ft)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_t0 = time.perf_counter()

        gkw = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if stopping:
            gkw["stopping_criteria"] = StoppingCriteriaList(stopping)
        if self._compiled:
            gkw["cache_implementation"] = "static"
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass
        try:
            outputs = self.model.generate(**inputs, **gkw)
        except Exception as e:
            if not self._compiled:
                raise
            print(f"[ClapPolicy] generate failed under compile ({type(e).__name__}); fallback eager")
            self._disable_compile()
            gkw.pop("cache_implementation", None)
            outputs = self.model.generate(**inputs, **gkw)

        if timing_enabled():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_t1 = time.perf_counter()
            if ft is not None and ft.first_token_time is not None:
                timer.record_split("prefill", ft.first_token_time - gen_t0)
                timer.record_split("decode", gen_t1 - ft.first_token_time)
            timer.record_scalar("batch_size", len(instructions))
            timer.record_scalar("prefill_len", input_len)
            timer.record_scalar("n_gen_tokens", int(outputs[:, input_len:].shape[1]))

        return self.processor.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    def decode_actions(self, texts: list[str]) -> np.ndarray:
        """Raw completions -> [B, horizon, act_dim] float32 actions (un-normalized)."""
        return np.stack(
            [
                text_to_actions(t, self.stats, self.num_bins, self.horizon, self.act_dim).astype(np.float32)
                for t in texts
            ]
        )

    def predict_batch(self, images: np.ndarray, instructions: list[str]) -> np.ndarray:
        """images: [B, num_cam, H, W, 3] uint8 -> actions: [B, horizon, act_dim] float32."""
        texts = self.generate(images, instructions)
        timer = get_timer()
        with timer.section("parse"):
            actions = self.decode_actions(texts)
        if _DEBUG:
            for instr, txt, act in zip(instructions, texts, actions):
                print(
                    f"[CLAP_EVAL_DEBUG] instruction={instr!r}\n"
                    f"[CLAP_EVAL_DEBUG] raw_output={txt!r}\n"
                    f"[CLAP_EVAL_DEBUG] parsed_actions=\n{act}",
                    flush=True,
                )
        timer.step_and_maybe_log()
        return actions
