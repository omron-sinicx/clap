# Copyright (c) 2026 OMRON SINIC X Corporation.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].
#
# Language-action phrasing ported from LAP (Apache-2.0, see LICENSES/Apache-2.0.txt):
# https://github.com/lihzha/lap/blob/main/src/lap/policies/transforms/action_text.py

"""Rule-based conversion of continuous robot actions to natural language descriptions."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    return int(round(value / n) * n)


def _format_numeric(val: float, sum_decimal: str) -> str:
    if sum_decimal == "no_number":
        return ""
    if sum_decimal == "nearest_10":
        return str(int(round(val / 10) * 10))
    m = re.fullmatch(r"(\d+)f", sum_decimal)
    decimals = int(m.group(1)) if m else 0
    return f"{val:.{decimals}f}"


def summarize_numeric_actions(
    arr_like,
    sum_decimal: str = "0f",
    include_rotation: bool = True,
    rotation_precision: int = 10,
) -> Optional[str]:
    """Convert continuous actions (delta xyz/rotation, gripper) to a natural language string.

    Actions are expected as [horizon, act_dim] where dimensions are:
      0-2: delta xyz (meters), 3-5: delta roll/pitch/yaw (radians), 6: gripper (0/1)
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    if sum_decimal == "compact":
        return _summarize_compact(arr, include_rotation)

    if sum_decimal in {"no_number", "nearest_10"}:
        decimals = 0
    else:
        m = re.fullmatch(r"(\d+)f", sum_decimal)
        if m is None:
            raise ValueError(f"invalid sum_decimal={sum_decimal!r}")
        decimals = int(m.group(1))

    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    parts: list[str] = []
    if sum_decimal == "no_number":
        if dx_m > 0 and dx != 0:
            parts.append("move forward")
        elif dx_m < 0 and dx != 0:
            parts.append("move back")
        if dy_m > 0 and dy != 0:
            parts.append("move left")
        elif dy_m < 0 and dy != 0:
            parts.append("move right")
        if dz_m > 0 and dz != 0:
            parts.append("move up")
        elif dz_m < 0 and dz != 0:
            parts.append("move down")
        if include_rotation:
            droll_rad = float(arr[..., 3].sum())
            dpitch_rad = float(arr[..., 4].sum())
            dyaw_rad = float(arr[..., 5].sum())
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt back")
            elif dpitch_rad < 0:
                parts.append("tilt forward")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            droll_rad = float(arr[..., 3].sum())
            dpitch_rad = float(arr[..., 4].sum())
            dyaw_rad = float(arr[..., 5].sum())
            droll = _round_to_nearest_n(
                abs(droll_rad * 180.0 / np.pi), rotation_precision
            )
            dpitch = _round_to_nearest_n(
                abs(dpitch_rad * 180.0 / np.pi), rotation_precision
            )
            dyaw = _round_to_nearest_n(
                abs(dyaw_rad * 180.0 / np.pi), rotation_precision
            )
            if droll_rad > 0 and droll != 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0 and droll != 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0 and dpitch != 0:
                parts.append(f"tilt back {dpitch} degrees")
            elif dpitch_rad < 0 and dpitch != 0:
                parts.append(f"tilt forward {dpitch} degrees")
            if dyaw_rad > 0 and dyaw != 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0 and dyaw != 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    g_last = float(arr[-1, 6])
    parts.append("open gripper" if g_last >= 0.5 else "close gripper")
    return ", ".join(parts)


def _summarize_compact(arr: np.ndarray, include_rotation: bool) -> str:
    dx_cm = int(round(float(arr[..., 0].sum()) * 100.0))
    dy_cm = int(round(float(arr[..., 1].sum()) * 100.0))
    dz_cm = int(round(float(arr[..., 2].sum()) * 100.0))
    parts = [f"{dx_cm:+03d}", f"{dy_cm:+03d}", f"{dz_cm:+03d}"]
    if include_rotation:
        droll_deg = _round_to_nearest_n(float(arr[..., 3].sum()) * 180.0 / np.pi, 5)
        dpitch_deg = _round_to_nearest_n(float(arr[..., 4].sum()) * 180.0 / np.pi, 5)
        dyaw_deg = _round_to_nearest_n(float(arr[..., 5].sum()) * 180.0 / np.pi, 5)
        parts.extend([f"{droll_deg:+03d}", f"{dpitch_deg:+03d}", f"{dyaw_deg:+03d}"])
    g_last = float(arr[-1, 6])
    parts.append(str(1 if g_last >= 0.5 else 0))
    return "<" + " ".join(parts) + ">"
