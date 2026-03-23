#!/usr/bin/env python3
"""
BRDF → Gaussian Splat experiment generator
------------------------------------------
Creates experiment variants for studying how much Disney BRDF material
information survives different appearance encodings before export to a
standard degree-3 3DGS PLY.

Experiments implemented from the earlier recommendations:

  exp1_raw_linear
      Save RAW linear BRDF samples (best inverse-friendly baseline).
      Also exports a preview PLY using a full SH fit in linear space so you can
      still inspect the sphere in your renderer.

  exp2_sh_linear_fullfit
      Project LINEAR radiance directly into full degree-3 SH.
      No tone mapping. No forced DC/base-color hack.

  exp3_sh_linear_forced_dc
      Project LINEAR radiance into SH, but force the DC term so that
      f_dc * SH_C0 + 0.5 == base_color.
      This isolates the impact of the forced-DC decomposition.

  exp4_sh_tonemapped_forced_dc
      Your original display-oriented setup:
      apply Reinhard tone mapping first, then use forced DC and fit only the
      higher-order residual.

Lighting models:
  directional
      Discrete directional lights with configurable count, directions and RGB
      intensity. This preserves the old fixed-rig behavior by default.

  gaussian_splat
      Gaussian emitter splats with configurable position, scale, opacity,
      orientation and RGB emission. Incident illumination is estimated with the
      same cosine-weighted MC style as in your separate test script.

Examples:
    python brdf_splat_experiments.py --experiment exp4_sh_tonemapped_forced_dc
    python brdf_splat_experiments.py --experiment all

    # Original legacy rig, but redder lights
    python brdf_splat_experiments.py --light-model directional \
        --light-color 3.0 2.0 2.0

    # 6 generated directional lights over the upper hemisphere
    python brdf_splat_experiments.py --light-model directional \
        --directional-preset fibonacci_upper --light-count 6

    # One overhead Gaussian light splat written into the output PLY too
    python brdf_splat_experiments.py --light-model gaussian_splat \
        --light-layout overhead --light-color 3 3 3 --include-light-splats-in-ply

    # Two manually placed Gaussian lights
    python brdf_splat_experiments.py --light-model gaussian_splat \
        --light-pos  0  0 5 --light-pos  3  0 4 \
        --light-color 3 3 3 --light-color 1 1 3 \
        --light-scale-actual 3.0 --light-scale-actual 2.0 \
        --include-light-splats-in-ply
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# SH constants matching the target shader exactly (degree 3, 16 basis functions)
# -----------------------------------------------------------------------------
SH_C0   = 0.28209479177387814

SH_C1   = 0.4886025119029199

SH_C2_0 =  1.0925484305920792
SH_C2_1 = -1.0925484305920792
SH_C2_2 =  0.31539156525252005
SH_C2_3 = -1.0925484305920792
SH_C2_4 =  0.5462742152960396

SH_C3_0 = -0.5900435899266435
SH_C3_1 =  2.890611442640554
SH_C3_2 = -0.4570457994644658
SH_C3_3 =  0.3731763325901154
SH_C3_4 = -0.4570457994644658
SH_C3_5 =  1.445305721320277
SH_C3_6 = -0.5900435899266435


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    signal_space: str          # "linear" or "reinhard"
    fit_mode: str              # "full" or "forced_dc_basecolor"
    save_raw_linear: bool      # save raw linear samples in NPZ
    preview_note: str


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "exp1_raw_linear": ExperimentSpec(
        name="exp1_raw_linear",
        signal_space="linear",
        fit_mode="full",  # preview PLY only; raw dataset is the main output
        save_raw_linear=True,
        preview_note="Raw linear dataset + preview PLY from full linear SH fit",
    ),
    "exp2_sh_linear_fullfit": ExperimentSpec(
        name="exp2_sh_linear_fullfit",
        signal_space="linear",
        fit_mode="full",
        save_raw_linear=False,
        preview_note="Linear radiance projected to full SH",
    ),
    "exp3_sh_linear_forced_dc": ExperimentSpec(
        name="exp3_sh_linear_forced_dc",
        signal_space="linear",
        fit_mode="forced_dc_basecolor",
        save_raw_linear=False,
        preview_note="Linear radiance projected to SH with forced DC = base_color",
    ),
    "exp4_sh_tonemapped_forced_dc": ExperimentSpec(
        name="exp4_sh_tonemapped_forced_dc",
        signal_space="reinhard",
        fit_mode="forced_dc_basecolor",
        save_raw_linear=False,
        preview_note="Tone-mapped display signal projected to SH with forced DC",
    ),
}


# -----------------------------------------------------------------------------
# SH basis — order and signs match the shader exactly
# Returns (M, 16): col 0 = DC, cols 1..15 = higher-order basis values
# -----------------------------------------------------------------------------
def eval_sh_basis(dirs: np.ndarray) -> np.ndarray:
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z

    B = np.empty((len(dirs), 16), dtype=np.float64)
    B[:, 0]  = SH_C0
    B[:, 1]  = -SH_C1 * y
    B[:, 2]  =  SH_C1 * z
    B[:, 3]  = -SH_C1 * x
    B[:, 4]  =  SH_C2_0 * x * y
    B[:, 5]  =  SH_C2_1 * y * z
    B[:, 6]  =  SH_C2_2 * (2.0 * zz - xx - yy)
    B[:, 7]  =  SH_C2_3 * x * z
    B[:, 8]  =  SH_C2_4 * (xx - yy)
    B[:, 9]  =  SH_C3_0 * y * (3.0 * xx - yy)
    B[:, 10] =  SH_C3_1 * x * y * z
    B[:, 11] =  SH_C3_2 * y * (4.0 * zz - xx - yy)
    B[:, 12] =  SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    B[:, 13] =  SH_C3_4 * x * (4.0 * zz - xx - yy)
    B[:, 14] =  SH_C3_5 * z * (xx - yy)
    B[:, 15] =  SH_C3_6 * x * (xx - 3.0 * yy)
    return B


def eval_sh(coeffs: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    return eval_sh_basis(dirs) @ coeffs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fibonacci_sphere(n: int) -> np.ndarray:
    g = (1.0 + math.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=np.float64)
    th = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    ph = 2.0 * np.pi * i / g
    return np.stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1
    )


def fibonacci_hemisphere(n: int, pole: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    pole = np.asarray(pole, dtype=np.float64)
    pole /= np.linalg.norm(pole) + 1e-12
    dirs = fibonacci_sphere(2 * n)
    dirs = dirs[(dirs @ pole) >= 0.0]
    if len(dirs) < n:
        dirs = np.concatenate([dirs, -dirs], axis=0)
    return dirs[:n]


def sample_view_dirs(n: int, normal: np.ndarray, domain: str) -> np.ndarray:
    dirs = fibonacci_sphere(n)
    if domain == "sphere":
        return dirs
    if domain == "hemisphere":
        dirs = dirs.copy()
        mask = (dirs @ normal) < 0.0
        dirs[mask] *= -1.0
        return dirs
    raise ValueError(f"Unknown view domain: {domain}")


def quat_z_to_n(n: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    dot = float(np.dot(z, n))
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    axis = np.cross(z, n)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def build_tangent_frame(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, n)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v


def cosine_hemisphere_samples(n: int, normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u, v = build_tangent_frame(normal)
    r1 = rng.random(n)
    r2 = rng.random(n)
    cos_t = np.sqrt(r1)
    sin_t = np.sqrt(1.0 - r1)
    phi = 2.0 * np.pi * r2
    local = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    world = local[:, 0:1] * u + local[:, 1:2] * v + local[:, 2:3] * normal[None, :]
    world /= np.linalg.norm(world, axis=1, keepdims=True) + 1e-12
    return world


def pack_sh_for_ply(coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    coeffs: (16, 3) rows = basis 0..15, cols = RGB.
    Returns (f_dc(3,), f_rest(45,)).
    """
    return coeffs[0].astype(np.float32), coeffs[1:].astype(np.float32).reshape(-1)


def reinhard_tonemap(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + x)


def as_np_list(vals: Optional[Sequence[Sequence[float]]]) -> List[np.ndarray]:
    if not vals:
        return []
    return [np.asarray(v, dtype=np.float64) for v in vals]


def expand_vec3(items: Optional[Sequence[Sequence[float]]], count: int, default: Sequence[float]) -> np.ndarray:
    arrs = as_np_list(items)
    if len(arrs) == 0:
        return np.tile(np.asarray(default, dtype=np.float64)[None, :], (count, 1))
    if len(arrs) == 1:
        return np.tile(arrs[0][None, :], (count, 1))
    if len(arrs) != count:
        raise ValueError(f"Expected 1 or {count} vec3 values, got {len(arrs)}")
    return np.stack(arrs, axis=0)


def expand_scalar(items: Optional[Sequence[float]], count: int, default: float) -> np.ndarray:
    if not items:
        return np.full((count,), float(default), dtype=np.float64)
    vals = np.asarray(items, dtype=np.float64)
    if vals.size == 1:
        return np.full((count,), float(vals[0]), dtype=np.float64)
    if vals.size != count:
        raise ValueError(f"Expected 1 or {count} scalar values, got {vals.size}")
    return vals.astype(np.float64)


# -----------------------------------------------------------------------------
# Disney BRDF (reduced subset used in the original script)
# -----------------------------------------------------------------------------
def schlick(u: np.ndarray | float) -> np.ndarray | float:
    m = np.clip(1.0 - u, 0.0, 1.0)
    return (m * m) * (m * m) * m


def gtr2(h: np.ndarray, a: float) -> np.ndarray:
    a2 = a * a
    d = h * h * (a2 - 1.0) + 1.0
    return a2 / (np.pi * d * d + 1e-7)


def smith_g(v: np.ndarray | float, a: float) -> np.ndarray | float:
    a2 = a * a
    return 2.0 * v / (v + np.sqrt(a2 + (1.0 - a2) * v * v) + 1e-7)


def disney_batch_directional(
    N: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    light_rgb: np.ndarray,
    base_color: np.ndarray,
    metallic: float,
    specular: float,
    roughness: float,
) -> np.ndarray:
    """
    Returns (M,3) raw HDR radiance for M view directions under discrete
    directional lights with per-light RGB intensity.
    """
    alpha = max(roughness * roughness, 0.001)
    F0 = (1.0 - metallic) * 0.08 * specular + metallic * base_color
    NdotL = np.clip(L @ N, 0.0, 1.0)
    NdotV = np.clip(V @ N, 1e-4, 1.0)
    rad = np.zeros((len(V), 3), dtype=np.float64)

    for k in range(len(L)):
        if NdotL[k] < 1e-6:
            continue
        H = V + L[k]
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
        NdotH = np.clip(H @ N, 0.0, 1.0)
        LdotH = np.clip((H * L[k]).sum(axis=1), 0.0, 1.0)

        FL = schlick(float(NdotL[k]))
        FV = schlick(NdotV)
        Rr = 2.0 * alpha * LdotH * LdotH

        f_d = (base_color / np.pi) * (1.0 - metallic) * (
            (1.0 + (Rr - 1.0) * FL) * (1.0 + (Rr - 1.0) * FV)
        )[:, None]

        FH = schlick(LdotH)[:, None]
        F = F0 + (1.0 - F0) * FH
        D = gtr2(NdotH, alpha)
        G = smith_g(float(NdotL[k]), alpha) * smith_g(NdotV, alpha)
        f_s = F * (D * G / (4.0 * float(NdotL[k]) * NdotV + 1e-7))[:, None]
        rad += (f_d + f_s) * float(NdotL[k]) * light_rgb[k][None, :]
    return rad


# -----------------------------------------------------------------------------
# Gaussian light helpers (ported from your separate script, vectorized where useful)
# -----------------------------------------------------------------------------
def compute_alpha_exact_batch(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    g_pos: np.ndarray,
    g_scale_ply: np.ndarray,
    g_quat_wxyz: np.ndarray,
    g_opacity: float,
) -> np.ndarray:
    """
    Vectorized port of Gaussian::computeAlpha.

    ray_o      : (3,)
    ray_d      : (N,3)
    g_scale_ply: (3,) values stored in the PLY. Actual scale = exp(g_scale_ply).
    g_quat_wxyz: (4,) quaternion [w,x,y,z]
    Returns alpha per ray: (N,)
    """
    g_scale_ply = np.atleast_1d(np.asarray(g_scale_ply, dtype=np.float64))
    if g_scale_ply.size == 1:
        g_scale_ply = np.repeat(g_scale_ply, 3)
    scale = np.exp(g_scale_ply)

    w, x, y, z = g_quat_wxyz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
            [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
            [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

    S2 = np.diag(1.0 / scale)
    SinvR = R @ S2

    ray_o = np.asarray(ray_o, dtype=np.float64)
    ray_d = np.asarray(ray_d, dtype=np.float64)
    g_pos = np.asarray(g_pos, dtype=np.float64)

    o_g = (SinvR @ (ray_o - g_pos)).astype(np.float64)
    d_g = (SinvR @ ray_d.T).T
    d_dot = np.sum(d_g * d_g, axis=1)
    t_max = -np.sum(o_g[None, :] * d_g, axis=1) / np.maximum(1e-6, d_dot)
    closest = ray_o[None, :] + t_max[:, None] * ray_d
    p_g = (SinvR @ (g_pos[None, :] - closest).T).T
    alpha = float(g_opacity) * np.exp(-0.5 * np.sum(p_g * p_g, axis=1))
    return np.minimum(0.99, alpha)


def eval_light_sh(sh_coeffs: np.ndarray, ray_d: np.ndarray) -> np.ndarray:
    """SH colour + 0.5 offset, matching evaluateSphericalHarmonics in C++."""
    v = np.asarray(ray_d, dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    col = eval_sh(sh_coeffs, v)
    return np.maximum(col + 0.5, 0.0)


def disney_brdf_pairwise(
    N: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    base_color: np.ndarray,
    metallic: float,
    specular: float,
    roughness: float,
) -> np.ndarray:
    """
    Pairwise BRDF values for all combinations of view directions V and incident
    directions L, using the MC-friendly form from your separate script.

    V: (M,3)
    L: (K,3)
    Returns: (M,K,3)
    """
    alpha = max(roughness * roughness, 0.001)
    F0 = (1.0 - metallic) * 0.08 * specular + metallic * base_color

    NdotL = np.clip(L @ N, 0.0, 1.0)                     # (K,)
    NdotV = np.clip(V @ N, 1e-4, 1.0)                   # (M,)
    H = V[:, None, :] + L[None, :, :]                   # (M,K,3)
    H /= np.linalg.norm(H, axis=2, keepdims=True) + 1e-8
    NdotH = np.clip(np.sum(H * N[None, None, :], axis=2), 0.0, 1.0)
    LdotH = np.clip(np.sum(H * L[None, :, :], axis=2), 0.0, 1.0)

    Rr = 2.0 * alpha * LdotH * LdotH
    FL = schlick(NdotL)[None, :]
    FV = schlick(NdotV)[:, None]
    f_d_scale = (1.0 + (Rr - 1.0) * FL) * (1.0 + (Rr - 1.0) * FV)
    f_d = (base_color / np.pi)[None, None, :] * (1.0 - metallic) * f_d_scale[:, :, None]

    D = gtr2(NdotH, alpha)
    FH = schlick(LdotH)[:, :, None]
    F = F0[None, None, :] + (1.0 - F0)[None, None, :] * FH

    # Match the separate MC script's formulation for the specular factor.
    a2 = roughness ** 4
    G_L = 1.0 / (NdotL + np.sqrt(a2 + (1.0 - a2) * NdotL * NdotL) + 1e-7)
    G_V = 1.0 / (NdotV + np.sqrt(a2 + (1.0 - a2) * NdotV * NdotV) + 1e-7)
    G = G_V[:, None] * G_L[None, :]
    f_s = F * (D * G)[:, :, None]

    valid = (NdotL > 0.0)[None, :, None]
    return (f_d + f_s) * valid


# -----------------------------------------------------------------------------
# Projection / fitting helpers
# -----------------------------------------------------------------------------
def transform_signal(raw_linear: np.ndarray, signal_space: str) -> np.ndarray:
    if signal_space == "linear":
        return raw_linear
    if signal_space == "reinhard":
        return reinhard_tonemap(raw_linear)
    raise ValueError(f"Unknown signal space: {signal_space}")


def project_signal_to_sh(
    view_dirs: np.ndarray,
    signal: np.ndarray,
    fit_mode: str,
    base_color: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit signal into the shader's stored SH coefficient convention.

    Shader-side reconstruction is assumed to be:
        recon = B @ coeffs + 0.5
    so the least-squares target is (signal - 0.5).
    """
    B = eval_sh_basis(view_dirs)
    target = signal - 0.5

    if fit_mode == "full":
        coeffs, _, _, _ = np.linalg.lstsq(B, target, rcond=None)
    elif fit_mode == "forced_dc_basecolor":
        f_dc = (base_color - 0.5) / SH_C0
        dc_contribution = SH_C0 * f_dc
        residual = target - dc_contribution
        higher_coeffs, _, _, _ = np.linalg.lstsq(B[:, 1:], residual, rcond=None)
        coeffs = np.vstack([f_dc, higher_coeffs])
    else:
        raise ValueError(f"Unknown fit mode: {fit_mode}")

    recon = 0.5 + B @ coeffs
    err = recon - signal
    metrics = {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs": float(np.max(np.abs(err))),
        "mean_signal": float(np.mean(signal)),
    }
    return coeffs, metrics


# -----------------------------------------------------------------------------
# PLY writer
# -----------------------------------------------------------------------------
def write_ply(path: Path, rows: np.ndarray, comment: str) -> None:
    n = rows.shape[0]
    props = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    assert rows.shape[1] == len(props), f"{rows.shape[1]} vs {len(props)}"

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment {comment}",
        f"element vertex {n}",
    ]
    header += [f"property float {p}" for p in props]
    header += ["end_header\n"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(rows.astype(np.float32).tobytes())


# -----------------------------------------------------------------------------
# Light construction
# -----------------------------------------------------------------------------
def legacy_light_rig() -> np.ndarray:
    L_raw = np.array(
        [
            [0.000,  0.000,  1.000],
            [0.577,  0.577,  0.577],
            [-0.577,  0.577,  0.577],
            [0.577, -0.577,  0.577],
            [-0.577, -0.577,  0.577],
            [1.000,  0.000,  0.500],
            [-1.000,  0.000,  0.500],
            [0.000,  1.000,  0.500],
            [0.000, -1.000,  0.500],
            [0.000,  0.000, -1.000],
        ],
        dtype=np.float64,
    )
    return L_raw / np.linalg.norm(L_raw, axis=1, keepdims=True)


def build_directional_lights(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    custom_dirs = as_np_list(args.light_dir)
    if custom_dirs:
        dirs = np.stack(custom_dirs, axis=0)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    else:
        if args.directional_preset == "legacy":
            dirs = legacy_light_rig()
        elif args.directional_preset == "overhead":
            dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)SyntaxError: Non-ASCII character '\xe2' in file scripts/brdf_splat_experiments.py on line 4, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details
        elif args.directional_preset == "fibonacci_upper":
            dirs = fibonacci_hemisphere(args.light_count, pole=np.array([0.0, 0.0, 1.0]))
        else:
            raise ValueError(f"Unknown directional preset: {args.directional_preset}")

    colors = expand_vec3(args.light_color, len(dirs), default=[1.0, 1.0, 1.0])
    return {
        "light_model": np.array("directional"),
        "dirs": dirs.astype(np.float32),
        "colors": colors.astype(np.float32),
    }


def build_gaussian_light_positions(args: argparse.Namespace) -> np.ndarray:
    custom_pos = as_np_list(args.light_pos)
    if custom_pos:
        return np.stack(custom_pos, axis=0)

    count = args.light_count
    if args.light_layout == "overhead":
        if count != 1:
            raise ValueError("--light-layout overhead requires --light-count 1 unless you pass manual --light-pos values")
        return np.array([[0.0, 0.0, args.light_distance]], dtype=np.float64)

    if args.light_layout == "ring":
        ang = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return np.stack(
            [
                args.light_radius * np.cos(ang),
                args.light_radius * np.sin(ang),
                np.full_like(ang, args.light_distance),
            ],
            axis=1,
        )

    if args.light_layout == "fibonacci":
        dirs = fibonacci_hemisphere(count, pole=np.array([0.0, 0.0, 1.0]))
        return dirs * float(args.light_distance)

    raise ValueError(f"Unknown light layout: {args.light_layout}")SyntaxError: Non-ASCII character '\xe2' in file scripts/brdf_splat_experiments.py on line 4, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details


def build_gaussian_lights(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    positions = build_gaussian_light_positions(args)
    count = len(positions)

    colors = expand_vec3(args.light_color, count, default=[3.0, 3.0, 3.0])

    normals_raw = as_np_list(args.light_normal)
    if normals_raw:
        normals = expand_vec3(args.light_normal, count, default=[0.0, 0.0, -1.0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    else:
        normals = -positions.copy()
        bad = np.linalg.norm(normals, axis=1) < 1e-9
        normals[bad] = np.array([0.0, 0.0, -1.0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    if args.light_scale_ply and args.light_scale_actual:
        raise ValueError("Use either --light-scale-ply or --light-scale-actual, not both")
    if args.light_scale_ply:
        scale_ply = expand_scalar(args.light_scale_ply, count, default=math.log(3.0))
    else:
        actual = expand_scalar(args.light_scale_actual, count, default=3.0)
        scale_ply = np.log(np.maximum(actual, 1e-8))
    opacity = expand_scalar(args.light_opacity, count, default=0.99)

    quats = np.stack([quat_z_to_n(n) for n in normals], axis=0)
    sh_coeffs = np.zeros((count, 16, 3), dtype=np.float64)
    sh_coeffs[:, 0, :] = (colors - 0.5) / SH_C0

    return {
        "light_model": np.array("gaussian_splat"),
        "positions": positions.astype(np.float32),
        "normals": normals.astype(np.float32),
        "quats": quats.astype(np.float32),
        "scale_ply": np.repeat(scale_ply[:, None], 3, axis=1).astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "colors": colors.astype(np.float32),
        "sh_coeffs": sh_coeffs.astype(np.float32),
    }


def build_lighting(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    if args.light_model == "directional":
        return build_directional_lights(args)
    if args.light_model == "gaussian_splat":
        return build_gaussian_lights(args)
    raise ValueError(f"Unknown light model: {args.light_model}")


# -----------------------------------------------------------------------------
# Scene / export generation
# -----------------------------------------------------------------------------
def rows_to_dicts(rows: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "positions": rows[:, 0:3],
        "normals": rows[:, 3:6],
        "f_dc": rows[:, 6:9],
        "f_rest": rows[:, 9:54],
        "opacity": rows[:, 54:55],
        "scale": rows[:, 55:58],
        "quat": rows[:, 58:62],
    }


def sample_gaussian_light_Li(
    splat_pos: np.ndarray,
    N: np.ndarray,
    gaussian_lights: Dict[str, np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
    li_clamp_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    wi_world = cosine_hemisphere_samples(n_samples, N, rng)
    Li = np.zeros((n_samples, 3), dtype=np.float64)

    light_count = gaussian_lights["positions"].shape[0]
    for i in range(light_count):
        alpha = compute_alpha_exact_batch(
            ray_o=splat_pos,
            ray_d=wi_world,
            g_pos=gaussian_lights["positions"][i],
            g_scale_ply=gaussian_lights["scale_ply"][i],
            g_quat_wxyz=gaussian_lights["quats"][i],
            g_opacity=float(gaussian_lights["opacity"][i]),
        )
        if np.all(alpha < 1e-8):
            continue
        raw_col = eval_light_sh(gaussian_lights["sh_coeffs"][i], wi_world)
        Li += np.clip(raw_col * alpha[:, None], 0.0, li_clamp_max)

    return wi_world, Li


def render_raw_linear_signal(
    N: np.ndarray,
    view_dirs: np.ndarray,
    lighting: Dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
    splat_pos: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    base_color = np.array(args.base_color, dtype=np.float64)
    metallic = float(args.metallic)
    specular = float(args.specular)
    roughness = float(args.roughness)

    if str(lighting["light_model"]) == "directional":
        raw = disney_batch_directional(
            N=N,
            V=view_dirs,
            L=lighting["dirs"].astype(np.float64),
            light_rgb=lighting["colors"].astype(np.float64),
            base_color=base_color,
            metallic=metallic,
            specular=specular,
            roughness=roughness,
        )
        return raw, None, None

    wi, Li = sample_gaussian_light_Li(
        splat_pos=splat_pos,
        N=N,
        gaussian_lights=lighting,
        n_samples=args.n_light_samples,
        rng=rng,
        li_clamp_max=args.light_li_clamp_max,
    )

    brdf = disney_brdf_pairwise(
        N=N,
        V=view_dirs,
        L=wi,
        base_color=base_color,
        metallic=metallic,
        specular=specular,
        roughness=roughness,
    )
    valid = np.clip(wi @ N, 0.0, 1.0) > 0.0
    denom = max(1, int(valid.sum()))
    raw = (np.pi / denom) * np.sum(brdf[:, valid, :] * Li[None, valid, :], axis=1)
    return raw, wi.astype(np.float32), Li.astype(np.float32)



def build_light_rows_if_needed(args: argparse.Namespace, lighting: Dict[str, np.ndarray]) -> np.ndarray:
    if args.light_model != "gaussian_splat" or not args.include_light_splats_in_ply:
        return np.zeros((0, 62), dtype=np.float32)

    count = lighting["positions"].shape[0]
    rows = np.zeros((count, 62), dtype=np.float32)
    for i in range(count):
        f_dc, f_rest = pack_sh_for_ply(lighting["sh_coeffs"][i].astype(np.float64))
        rows[i] = np.concatenate(
            [
                lighting["positions"][i],
                lighting["normals"][i],
                f_dc,
                f_rest,
                np.array([lighting["opacity"][i]], dtype=np.float32),
                lighting["scale_ply"][i].astype(np.float32),
                lighting["quats"][i].astype(np.float32),
            ]
        )
    return rows



def build_rows_for_experiment(args: argparse.Namespace, spec: ExperimentSpec) -> Dict[str, np.ndarray]:
    n_splats = args.n_splats
    n_view_samples = args.n_view_samples

    base_color = np.array(args.base_color, dtype=np.float64)
    opacity = float(args.opacity)
    scale_t = float(args.scale_t)
    scale_n = float(args.scale_n)

    lighting = build_lighting(args)
    positions = fibonacci_sphere(n_splats)

    N_COLS = 3 + 3 + 3 + 45 + 1 + 3 + 4  # 62
    obj_rows = np.zeros((n_splats, N_COLS), dtype=np.float32)
    fit_rmse = np.zeros((n_splats,), dtype=np.float32)
    fit_max_abs = np.zeros((n_splats,), dtype=np.float32)

    all_view_dirs = np.zeros((n_splats, n_view_samples, 3), dtype=np.float32)
    save_raw = spec.save_raw_linear or args.save_samples
    raw_linear_samples = np.zeros((n_splats, n_view_samples, 3), dtype=np.float32) if save_raw else None
    transformed_samples = np.zeros((n_splats, n_view_samples, 3), dtype=np.float32) if save_raw else None

    incident_dirs = None
    incident_Li = None
    if args.light_model == "gaussian_splat":
        incident_dirs = np.zeros((n_splats, args.n_light_samples, 3), dtype=np.float32)
        incident_Li = np.zeros((n_splats, args.n_light_samples, 3), dtype=np.float32)

    seed_seq = np.random.SeedSequence(args.seed)
    child_seeds = seed_seq.spawn(n_splats)

    print(f"[{spec.name}] Building {n_splats} splats with {n_view_samples} view samples each ...")
    print(f"  light_model = {args.light_model}")
    if args.light_model == "gaussian_splat":
        print(f"  n_light_samples = {args.n_light_samples}")

    for i, pos in enumerate(positions):
        if i % max(1, args.progress_every) == 0:
            print(f"  splat {i:4d}/{n_splats}")

        rng = np.random.default_rng(child_seeds[i])
        N_vec = pos.copy()
        V = sample_view_dirs(n_view_samples, N_vec, args.view_domain)
        raw_linear, wi, Li = render_raw_linear_signal(
            N=N_vec,
            view_dirs=V,
            lighting=lighting,
            args=args,
            rng=rng,
            splat_pos=pos,
        )
        signal = transform_signal(raw_linear, spec.signal_space)
        coeffs, metrics = project_signal_to_sh(
            view_dirs=V,
            signal=signal,
            fit_mode=spec.fit_mode,
            base_color=base_color,
        )

        f_dc, f_rest = pack_sh_for_ply(coeffs)
        q = quat_z_to_n(N_vec)
        obj_rows[i] = np.concatenate(
            [
                pos.astype(np.float32),
                N_vec.astype(np.float32),
                f_dc,
                f_rest,
                np.float32([opacity]),
                np.float32([scale_t, scale_t, scale_n]),
                q.astype(np.float32),
            ]
        )
        fit_rmse[i] = metrics["rmse"]
        fit_max_abs[i] = metrics["max_abs"]
        all_view_dirs[i] = V.astype(np.float32)

        if raw_linear_samples is not None:
            raw_linear_samples[i] = raw_linear.astype(np.float32)
            transformed_samples[i] = signal.astype(np.float32)
        if incident_dirs is not None and wi is not None and Li is not None:
            incident_dirs[i] = wi
            incident_Li[i] = Li

    light_rows = build_light_rows_if_needed(args, lighting)
    rows = np.vstack([obj_rows, light_rows]) if len(light_rows) > 0 else obj_rows

    out: Dict[str, np.ndarray] = {
        "rows": rows,
        "object_rows": obj_rows,
        "fit_rmse": fit_rmse,
        "fit_max_abs": fit_max_abs,
        "view_dirs": all_view_dirs,
        "object_positions": positions.astype(np.float32),
        "object_normals": positions.astype(np.float32),
        "light_model": np.array(args.light_model),
    }
    if args.light_model == "directional":
        out["light_dirs"] = lighting["dirs"]
        out["light_colors"] = lighting["colors"]
    else:
        out["light_positions"] = lighting["positions"]
        out["light_normals"] = lighting["normals"]
        out["light_quats"] = lighting["quats"]
        out["light_scale_ply"] = lighting["scale_ply"]
        out["light_opacity"] = lighting["opacity"]
        out["light_colors"] = lighting["colors"]
        out["light_sh_coeffs"] = lighting["sh_coeffs"]
        out["light_rows"] = light_rows
        if incident_dirs is not None:
            out["incident_dirs"] = incident_dirs
            out["incident_Li"] = incident_Li
    if raw_linear_samples is not None:
        out["raw_linear_samples"] = raw_linear_samples
        out["signal_samples"] = transformed_samples
    return out


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BRDF→Gaussian-splat experiment variants and compatible PLY files."
    )
    parser.add_argument(
        "--experiment",
        default="exp4_sh_tonemapped_forced_dc",
        choices=[*EXPERIMENTS.keys(), "all"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/brdf_splat_experiments"),
        help="Directory for .ply, .npz and manifest files.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="shader_ball",
        help="Base filename stem for outputs.",
    )

    # Material parameters
    parser.add_argument("--base-color", type=float, nargs=3, default=[0.55, 0.90, 0.10], metavar=("R", "G", "B"))
    parser.add_argument("--metallic", type=float, default=0.12)
    parser.add_argument("--specular", type=float, default=0.60)
    parser.add_argument("--roughness", type=float, default=0.35)

    # Sphere / export parameters
    parser.add_argument("--n-splats", type=int, default=300)
    parser.add_argument("--n-view-samples", type=int, default=512)
    parser.add_argument("--view-domain", choices=["sphere", "hemisphere"], default="sphere")
    parser.add_argument("--opacity", type=float, default=10.0)
    parser.add_argument("--scale-t", type=float, default=0.055)
    parser.add_argument("--scale-n", type=float, default=0.002)
    parser.add_argument("--save-samples", action="store_true", help="For exp2/3/4, also save per-splat signal samples to NPZ.")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Lighting controls
    parser.add_argument("--light-model", choices=["directional", "gaussian_splat"], default="gaussian_splat")
    parser.add_argument("--light-count", type=int, default=1, help="How many generated lights to use when not specifying manual positions/directions.")
    parser.add_argument("--light-color", action="append", nargs=3, type=float, metavar=("R", "G", "B"),
                        help="Repeatable RGB intensity. One value broadcasts to all lights.")

    # Directional light controls
    parser.add_argument("--directional-preset", choices=["legacy", "overhead", "fibonacci_upper"], default="legacy")
    parser.add_argument("--light-dir", action="append", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Repeatable manual directional light direction. Overrides preset generation.")

    # Gaussian light controls
    parser.add_argument("--light-layout", choices=["overhead", "ring", "fibonacci"], default="overhead")
    parser.add_argument("--light-pos", action="append", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Repeatable manual Gaussian light position. Overrides layout generation.")
    parser.add_argument("--light-normal", action="append", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Repeatable manual Gaussian light normal. Defaults to facing the origin.")
    parser.add_argument("--light-distance", type=float, default=5.0, help="Generated Gaussian light height/radius distance.")
    parser.add_argument("--light-radius", type=float, default=3.0, help="Ring radius for generated Gaussian lights.")
    parser.add_argument("--light-scale-actual", action="append", type=float,
                        help="Repeatable actual Gaussian light scale. Internally converted to stored log(scale).")
    parser.add_argument("--light-scale-ply", action="append", type=float,
                        help="Repeatable stored Gaussian light scale value, where actual = exp(scale_ply).")
    parser.add_argument("--light-opacity", action="append", type=float,
                        help="Repeatable Gaussian light opacity.")
    parser.add_argument("--n-light-samples", type=int, default=256,
                        help="Cosine-weighted incident-direction samples for Gaussian light integration.")
    parser.add_argument("--light-li-clamp-max", type=float, default=2.0,
                        help="Clamp max for per-sample Gaussian light radiance, matching the separate test script.")
    parser.add_argument("--include-light-splats-in-ply", action="store_true",
                        help="Append Gaussian light splats to the output PLY.")

    return parser.parse_args()



def make_manifest(args: argparse.Namespace, spec: ExperimentSpec, built: Dict[str, np.ndarray], ply_path: Path, npz_path: Path) -> Dict[str, object]:
    fit_rmse = built["fit_rmse"]
    fit_max_abs = built["fit_max_abs"]

    lighting_manifest: Dict[str, object] = {
        "light_model": args.light_model,
        "include_light_splats_in_ply": bool(args.include_light_splats_in_ply),
    }
    if args.light_model == "directional":
        lighting_manifest.update(
            {
                "directional_preset": args.directional_preset,
                "light_count": int(built["light_dirs"].shape[0]),
                "light_dirs": built["light_dirs"].astype(float).tolist(),
                "light_colors": built["light_colors"].astype(float).tolist(),
            }
        )
    else:
        lighting_manifest.update(
            {
                "light_layout": args.light_layout,
                "light_count": int(built["light_positions"].shape[0]),
                "light_positions": built["light_positions"].astype(float).tolist(),
                "light_normals": built["light_normals"].astype(float).tolist(),
                "light_scale_ply": built["light_scale_ply"].astype(float).tolist(),
                "light_opacity": built["light_opacity"].astype(float).tolist(),
                "light_colors": built["light_colors"].astype(float).tolist(),
                "n_light_samples": int(args.n_light_samples),
                "li_clamp_max": float(args.light_li_clamp_max),
            }
        )

    manifest = {
        "experiment": spec.name,
        "preview_note": spec.preview_note,
        "signal_space": spec.signal_space,
        "fit_mode": spec.fit_mode,
        "view_domain": args.view_domain,
        "material": {
            "base_color": list(map(float, args.base_color)),
            "metallic": float(args.metallic),
            "specular": float(args.specular),
            "roughness": float(args.roughness),
        },
        "sphere": {
            "n_splats": int(args.n_splats),
            "n_view_samples": int(args.n_view_samples),
            "opacity": float(args.opacity),
            "scale_t": float(args.scale_t),
            "scale_n": float(args.scale_n),
        },
        "lighting": lighting_manifest,
        "fit_error": {
            "rmse_mean": float(np.mean(fit_rmse)),
            "rmse_std": float(np.std(fit_rmse)),
            "rmse_max": float(np.max(fit_rmse)),
            "max_abs_mean": float(np.mean(fit_max_abs)),
            "max_abs_max": float(np.max(fit_max_abs)),
        },
        "paths": {
            "ply": str(ply_path),
            "npz": str(npz_path),
        },
    }
    return manifest


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_one_experiment(args: argparse.Namespace, spec: ExperimentSpec) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_rows_for_experiment(args, spec)
    rows = built["rows"]

    stem = f"{args.base_name}_{spec.name}"
    ply_path = output_dir / f"{stem}.ply"
    npz_path = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"

    write_ply(
        ply_path,
        rows,
        comment=f"BRDF splat experiment: {spec.name} | {spec.preview_note} | light={args.light_model}",
    )

    npz_payload = {
        "experiment": np.array(spec.name),
        "signal_space": np.array(spec.signal_space),
        "fit_mode": np.array(spec.fit_mode),
        "view_domain": np.array(args.view_domain),
        "base_color": np.array(args.base_color, dtype=np.float32),
        "metallic": np.array(args.metallic, dtype=np.float32),
        "specular": np.array(args.specular, dtype=np.float32),
        "roughness": np.array(args.roughness, dtype=np.float32),
        "light_model": np.array(args.light_model),
        "fit_rmse": built["fit_rmse"],
        "fit_max_abs": built["fit_max_abs"],
        "view_dirs": built["view_dirs"],
        "object_positions": built["object_positions"],
        "object_normals": built["object_normals"],
    }
    npz_payload.update(rows_to_dicts(built["object_rows"]))
    if args.light_model == "directional":
        npz_payload["light_dirs"] = built["light_dirs"]
        npz_payload["light_colors"] = built["light_colors"]
    else:
        npz_payload["light_positions"] = built["light_positions"]
        npz_payload["light_normals"] = built["light_normals"]
        npz_payload["light_quats"] = built["light_quats"]
        npz_payload["light_scale_ply"] = built["light_scale_ply"]
        npz_payload["light_opacity"] = built["light_opacity"]
        npz_payload["light_colors"] = built["light_colors"]
        npz_payload["light_sh_coeffs"] = built["light_sh_coeffs"]
        if "light_rows" in built:
            npz_payload["light_rows"] = built["light_rows"]
        if "incident_dirs" in built:
            npz_payload["incident_dirs"] = built["incident_dirs"]
            npz_payload["incident_Li"] = built["incident_Li"]
    if "raw_linear_samples" in built:
        npz_payload["raw_linear_samples"] = built["raw_linear_samples"]
        npz_payload["signal_samples"] = built["signal_samples"]
    np.savez_compressed(npz_path, **npz_payload)

    manifest = make_manifest(args, spec, built, ply_path, npz_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"[{spec.name}] wrote:")
    print(f"  PLY : {ply_path}")
    print(f"  NPZ : {npz_path}")
    print(f"  JSON: {manifest_path}")
    print(f"  mean RMSE = {manifest['fit_error']['rmse_mean']:.6f}")
    print(f"  max  RMSE = {manifest['fit_error']['rmse_max']:.6f}")
    if spec.save_raw_linear:
        print("  note: NPZ contains RAW linear BRDF samples for inverse fitting.")



def main() -> None:
    args = parse_args()

    names: List[str]
    if args.experiment == "all":
        names = list(EXPERIMENTS.keys())
    else:
        names = [args.experiment]

    for name in names:
        print("=" * 80)
        print(f"Running {name}")
        run_one_experiment(args, EXPERIMENTS[name])


if __name__ == "__main__":
    main()