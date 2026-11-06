# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""Per-stage inference timing (enabled with CLAP_TIMING=1).

Each inference call is split into preprocess / prefill / decode / parse, and the
batch size, generated length and decode ms/token are reported. The cumulative
average is dominated by the one-time torch.compile warm-up for a long time, so the
average over the most recent calls is printed as well (use it for steady-state speed).
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

try:
    from transformers import StoppingCriteria
except Exception:  # pragma: no cover
    StoppingCriteria = object


_TIMING = os.environ.get("CLAP_TIMING", "0") == "1"
_TIMING_EVERY = int(os.environ.get("CLAP_TIMING_EVERY", "20"))


def timing_enabled() -> bool:
    return _TIMING


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FirstTokenTimer(StoppingCriteria):
    """Record when the first token is generated to split prefill from decode. Never stops generation."""

    def __init__(self):
        self.first_token_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_time is None:
            _sync()
            self.first_token_time = time.perf_counter()
        return False


class InferenceTimer:
    def __init__(self):
        self.enabled = _TIMING
        self._sum = defaultdict(float)
        self._scalar_sum = defaultdict(float)
        self.n_calls = 0
        self._device_str = ""
        self._last_n = 0
        self._last_sum = defaultdict(float)
        self._last_scalar = defaultdict(float)

    def set_device(self, device):
        try:
            self._device_str = str(device)
        except Exception:
            self._device_str = ""

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self._sum[name] += time.perf_counter() - t0

    def record_split(self, name, seconds):
        if self.enabled:
            self._sum[name] += seconds

    def record_scalar(self, name, value):
        if self.enabled:
            self._scalar_sum[name] += float(value)

    def step_and_maybe_log(self):
        if not self.enabled:
            return
        self.n_calls += 1
        if self.n_calls % _TIMING_EVERY == 0:
            self.log_summary()

    def log_summary(self):
        if not self.enabled or self.n_calls == 0:
            return
        n = self.n_calls
        keys = [k for k in ["preprocess", "prefill", "decode", "parse"] if k in self._sum]
        tot = sum(self._sum.values()) or 1.0
        cum = "\n".join(
            f"    {k:<11}: {self._sum[k] / n * 1000:8.2f} ms/call ({self._sum[k] / tot * 100:5.1f}%)"
            for k in keys
        )
        dn = n - self._last_n
        recent = ""
        if dn > 0:
            r_tot = (sum(self._sum.values()) - sum(self._last_sum.values())) / dn * 1000
            r_lines = "\n".join(
                f"    {k:<11}: {(self._sum[k] - self._last_sum[k]) / dn * 1000:8.2f} ms/call"
                for k in keys
            )
            r_dec = self._sum.get("decode", 0) - self._last_sum.get("decode", 0)
            r_gen = self._scalar_sum.get("n_gen_tokens", 0) - self._last_scalar.get("n_gen_tokens", 0)
            r_mstok = (r_dec / max(r_gen - dn, 1) * 1000) if r_gen > 0 else 0
            recent = (
                f"\n  ---- last {dn} calls (steady state, excluding warm-up) ----\n"
                f"    total      : {r_tot:8.2f} ms/call\n{r_lines}\n"
                f"    decode ms/token : {r_mstok:.3f}\n"
            )
        self._last_n = n
        self._last_sum = dict(self._sum)
        self._last_scalar = dict(self._scalar_sum)
        bs = self._scalar_sum.get("batch_size", 0) / n
        ng = self._scalar_sum.get("n_gen_tokens", 0) / n
        pl = self._scalar_sum.get("prefill_len", 0) / n
        print(
            f"\n{'=' * 60}\n[CLAP_TIMING] [{self._device_str}] calls={n} cumulative avg total={tot / n * 1000:.1f}ms\n"
            f"{cum}\n    batch={bs:.1f} prefill_len={pl:.1f} n_gen={ng:.1f}\n{recent}{'=' * 60}"
        )


_TIMER = InferenceTimer()


def get_timer() -> InferenceTimer:
    return _TIMER
