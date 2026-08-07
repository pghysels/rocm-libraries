#!/usr/bin/env python3
"""
mi455_predicted_speedup.py

Predicted FP64 GEMM emulation speedup on AMD MI455 using the roofline
performance model from `fp64_emulation.cpp` (Ozaki-II / OS2).

The figure reproduces the style of a panel from Figure 6 of arXiv:2511.13778,
but for the (unreleased) MI455 architecture using estimated hardware constants,
and shows a 2D heatmap of predicted speedup over native DGEMM.

Heatmap axes
------------
  X : m = n  (square output matrix size), log-spaced 512 → 32768
  Y : k      (inner / contraction dimension), log-spaced 512 → 32768
  Color : max(1,  t_native / t_emul)  for s = 16 moduli, tA=T, tB=N

Performance model
-----------------
  A faithful Python reimplementation of fp64EmulationPerfModelTimes()
  from projects/hipblaslt/library/src/amd_detail/rocblaslt/src/fp64_emulation.cpp.
  Uses the MONOLITHIC path (no recursive splitting, fused kernel disabled).

Hardware constants
------------------
  MI355X entries are taken verbatim from the C++ Oz2PerfModelParams table.
  MI455 entries are ESTIMATED:
    • HBM bandwidth  : ~23.3 TB/s
    • FP64 throughput: ~5 TFLOP/s   (SIMD only, no MFMA)
    • INT8 throughput: ~3300 TOPS effective  (5000 TOPS peak × 0.66)
  All kernel efficiency factors and launch latencies are inherited from the
  MI355X calibration (per the task specification).

Usage
-----
    python3 mi455_predicted_speedup.py [--out-dir plots] [--n-pts 40]
"""

import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm

# =============================================================================
# 1.  Constants  (verbatim from fp64_emulation.cpp)
# =============================================================================

LATENCY_KERNEL  = 5.0e-6    # s  – GPU kernel scheduling overhead
LATENCY_MATMUL  = 10.0e-6   # s  – hipBLASLt matmul launch overhead
LATENCY_MEMSET  = 2.0e-6    # s  – hipMemsetAsync overhead
LATENCY_SYNC    = 50.0e-6   # s  – CPU–GPU hipStreamSynchronize roundtrip

OZ2_ALIGN              = 128          # bytes
OZ2_CHUNK_TARGET_BYTES = 32 << 30    # 32 GiB
OZ2_HOST_OVERHEAD_S    = 0.100e-3    # 0.1 ms → s

# Kernel efficiency factors (tA, tB) → {prelim, scale}
# Calibrated on MI355X; reused for MI455 (monolithic path).
_EFF = {
    (True,  True ): {"prelim": 0.628, "scale": 0.401},
    (True,  False): {"prelim": 0.797, "scale": 0.461},
    (False, True ): {"prelim": 0.550, "scale": 0.354},
    (False, False): {"prelim": 0.622, "scale": 0.405},
}
EFF_REFINE = 0.511
EFF_ACCUM  = 0.775
EFF_FUSED  = 0.66    # fused kernel INT8 MFMA efficiency (applied to compute only, not BW)

# =============================================================================
# 2.  Hardware parameter sets
# =============================================================================

# ── From the C++ Oz2PerfModelParams table ────────────────────────────────────
# latency units: observed_bandwidth_bytes_per_sec × LATENCY_MATMUL
# So  c0 = latency / LATENCY_MATMUL  is observed HBM bandwidth in bytes/s.
#     c1 = c0 × ai                   is observed FP64 GEMM throughput  (ops/s)
#     c2 = c1 × ratio                is observed INT8 GEMM throughput  (ops/s)

HW_PARAMS = {
    "MI300X": {
        # pci_id 0x74a0 / 0x74a1 / 0x74a9  (from C++ table)
        "latency": 3.93e7,   # → c0 ≈ 3.93 TB/s
        "ai":       2.63868, # → c1 ≈  10.4 TFLOP/s FP64
        "ratio":  168.411,   # → c2 ≈ 1750 TOPS INT8
    },
    "MI350X": {
        # pci_id 0x75a0 / 0x75b0  (from C++ table)
        "latency": 6.08e7,   # → c0 ≈ 6.08 TB/s
        "ai":      11.2237,  # → c1 ≈  68.2 TFLOP/s FP64
        "ratio":   46.4678,  # → c2 ≈ 3169 TOPS INT8
    },
    "MI355X": {
        # pci_id 0x75a3 / 0x75b3  (from C++ table)
        "latency": 6.82e7,   # → c0 ≈ 6.82 TB/s
        "ai":      11.3284,  # → c1 ≈  77.2 TFLOP/s FP64
        "ratio":   42.7280,  # → c2 ≈ 3298 TOPS INT8
    },
    # ── MI455 (old estimate, superseded) ─────────────────────────────────────
    # "MI455": {
    #     "latency": 1.705e8,      # → c0 ≈ 17.05 TB/s  (2.5× MI355X BW)
    #     "ai":      10.0 / 17.05, # → c1 ≈ 10 TFLOP/s FP64 (SIMD only, no MFMA)
    #     "ratio":   211.3,         # → c2 ≈ 2113 TOPS INT8  (same fraction as MI355X)
    # },
    # ── MI455  —  ESTIMATED parameters ───────────────────────────────────────
    # Estimation rationale:
    #   HBM bandwidth  : 23.3 TB/s
    #     latency = 23.3e12 × LATENCY_MATMUL = 23.3e12 × 10e-6 = 2.33e8
    #   FP64 throughput: MI455 has no FP64 matrix (MFMA) instructions.
    #     Native DGEMM falls back to SIMD-only FP64 → ~5 TFLOP/s.
    #     ai = 5e12 / 23.3e12 ≈ 0.2146
    #   INT8 throughput: peak is 5000 TOPS (5 POPS), but the model uses the
    #     *effective* observed fraction.
    #     MI355X effective / peak = 3301 / 5000 ≈ 0.660.
    #     MI455 effective INT8 = 0.660 × 5000 ≈ 3300 TOPS.
    #     ratio = 3300e12 / 5e12 = 660.0
    #   All kernel efficiency factors and LATENCY_* constants are INHERITED
    #   from MI355X calibration (monolithic path, no fused kernel).
    "MI455": {
        "latency": 2.33e8,       # → c0 ≈ 23.3 TB/s
        "ai":      5.0 / 23.3,   # → c1 ≈ 5 TFLOP/s FP64 (SIMD only, no MFMA)
        "ratio":   660.0,         # → c2 ≈ 3300 TOPS INT8  (same fraction as MI355X)
    },
}

# =============================================================================
# 3.  Model helpers  (oz2_pad, oz2_compute_chunk_size)
# =============================================================================

def oz2_pad(n: int) -> int:
    """Pad n to the next multiple of OZ2_ALIGN (128 bytes)."""
    return (n + OZ2_ALIGN - 1) // OZ2_ALIGN * OZ2_ALIGN


def oz2_compute_chunk_size(m: int, n: int, k: int, s: int) -> int:
    """
    Compute chunk size: largest c ≤ s such that
    c × (mn×4 + lda8i×cola8i + lda8i×n) ≤ OZ2_CHUNK_TARGET_BYTES.

    Mirrors oz2_compute_chunk_size() in fp64_emulation.cpp exactly.
    """
    mn4    = m * n * 4
    lda8i  = oz2_pad(k)
    cola8i = oz2_pad(m)
    slc    = lda8i * cola8i + lda8i * n
    chunk  = s
    total  = mn4 + slc
    if total > 0:
        chunk = min(chunk, OZ2_CHUNK_TARGET_BYTES // total)
    return max(1, chunk)


def oz2_compute_chunk_size_fused(m: int, n: int, k: int, s: int) -> int:
    """
    Fused-kernel chunk size: excludes C32i from workspace budget (INT8 products
    stay in GPU registers).  Mirrors oz2_compute_chunk_size_fused() in
    fp64_emulation.cpp — typically returns s for all practical shapes, so the
    scale kernel processes all moduli in a single pass and binary-halving is
    rarely triggered.
    """
    lda8i  = oz2_pad(k)
    cola8i = oz2_pad(m)
    slc    = lda8i * cola8i + lda8i * n   # A8i + B8i per modulus (no C32i)
    chunk  = s
    if slc > 0:
        chunk = min(chunk, OZ2_CHUNK_TARGET_BYTES // slc)
    return max(1, chunk)


def oz2_effective_time(
    tA: bool,
    tB: bool,
    m: int,
    n: int,
    k: int,
    num_moduli: int,
    hw: dict,
    dynamic_mode: bool = False,
    use_fused: bool = False,
) -> float:
    """
    Python reimplementation of oz2_effective_time_ms() from fp64_emulation.cpp.

    Returns the minimum achievable emulation time in seconds, accounting for
    recursive binary-halving when the workspace budget forces n_chunks > 1.

    When n_chunks > 1 the implementation splits along the longer of m or n and
    runs two sequential sub-GEMMs.  The effective time is
        t_split = 2 × oz2_effective_time(half_m, half_n, k, ...)
    If t_split < t_mono the split is taken (recursively).

    This exactly mirrors the logic in fp64EmulatedGemmImpl / oz2_effective_time_ms.
    When use_fused=True the fused chunk formula is used for the splitting decision,
    mirroring the fused_forced branch in fp64EmulationWorkspaceSize.
    """
    t_mono = perf_model_times(tA, tB, m, n, k, num_moduli, hw, dynamic_mode,
                              use_fused=use_fused)["t_total"]

    if use_fused:
        chunk_sz = oz2_compute_chunk_size_fused(m, n, k, num_moduli)
    else:
        chunk_sz = oz2_compute_chunk_size(m, n, k, num_moduli)
    n_chunks = math.ceil(num_moduli / chunk_sz)

    if n_chunks > 1:
        split_m = (m >= n)
        half_m  = m // 2 if split_m else m
        half_n  = n       if split_m else n // 2

        t_split = 2.0 * oz2_effective_time(
            tA, tB, half_m, half_n, k, num_moduli, hw, dynamic_mode,
            use_fused=use_fused
        )
        if t_split < t_mono:
            return t_split

    return t_mono

# =============================================================================
# 4.  Core performance model  (fp64EmulationPerfModelTimes, monolithic path)
# =============================================================================

def perf_model_times(
    tA: bool,
    tB: bool,
    m: int,
    n: int,
    k: int,
    num_moduli: int,
    hw: dict,
    dynamic_mode: bool = False,
    use_fused: bool = False,
) -> dict:
    """
    Python reimplementation of fp64EmulationPerfModelTimes() from
    fp64_emulation.cpp.

    When use_fused=False (default): monolithic path with fused kernel DISABLED.
    When use_fused=True: models the fused MFMA+CRT kernel path.  The fused
    kernel reads pre-scaled INT8 A8i/B8i from workspace, performs MFMA + CRT
    accumulation in registers, and writes FP64 D directly.  The scale kernel
    still runs; only INT8 GEMMs + accum are replaced.

    Parameters
    ----------
    tA, tB        : transpose flags for A and B
    m, n, k       : GEMM dimensions
    num_moduli    : s  (number of Ozaki-II CRT moduli)
    hw            : hardware parameter dict with keys 'latency', 'ai', 'ratio'
    dynamic_mode  : if True, include ADP hipStreamSynchronize overhead
    use_fused     : if True, model the fused MFMA+CRT kernel

    Returns
    -------
    dict with keys:
        t_prelim_kern, t_prelim_gemm, t_refine_kern, t_adp,
        t_scale_kern, t_int8_gemms, t_accum_kern, t_fused_kern, t_host,
        t_total   (sum of all emulation components, seconds)
        t_native  (predicted native DGEMM time, seconds)
    All times are in seconds.
    """
    # ── Derived hardware capacities ──────────────────────────────────────────
    # c0: memory bandwidth  (bytes/s)
    # c1: FP64 GEMM throughput  (ops/s, where 1 op = 1 multiply-accumulate)
    # c2: INT8 GEMM throughput  (ops/s)
    c0 = hw["latency"] / LATENCY_MATMUL
    c1 = c0 * hw["ai"]
    c2 = c1 * hw["ratio"]

    s   = float(num_moduli)
    mn  = float(m) * float(n)
    mk  = float(m) * float(k)
    kn  = float(k) * float(n)
    mnk = mn * float(k)

    # Choose chunk formula: fused path excludes C32i, allowing a larger chunk
    if use_fused:
        chunk_sz = float(oz2_compute_chunk_size_fused(m, n, k, num_moduli))
    else:
        chunk_sz = float(oz2_compute_chunk_size(m, n, k, num_moduli))
    n_chunks       = math.ceil(s / chunk_sz)
    n_scale_chunks = n_chunks  # scale and GEMM share the same chunk

    eff_prelim = _EFF[(tA, tB)]["prelim"]
    eff_scale  = _EFF[(tA, tB)]["scale"]

    # ── Component-time formulas (transcribed verbatim from C++) ──────────────

    # Memory-bandwidth reference: read INT8 A (mk bytes), B (kn bytes),
    # write FP64 C/D (4·mn bytes = 32-bit output × 4 bytes).
    t_int8_bw = (mk + kn + 4.0 * mn) / c0

    # Preliminary shift + extraction kernel (reads fp64 A and B, writes INT8).
    # 17 bytes/elem of (mk+kn) combines: 8 fp64 read + 1 int8 write + overhead.
    t_prelim_kern = (
        (mk + kn) * 17.0 / c0 + 2.0 * LATENCY_KERNEL
    ) / eff_prelim

    # Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high.
    t_prelim_gemm = max(2.0 * mnk / c2, t_int8_bw) + LATENCY_MATMUL

    # Shift-refinement kernels (reads m×n INT32 C32i, writes sftA / sftB).
    # 8 bytes/elem accounts for the INT32 read + partial sft write.
    t_refine_kern = (
        mn * 8.0 / c0 + 3.0 * LATENCY_KERNEL + LATENCY_MEMSET
    ) / EFF_REFINE

    # Multi-modulus scale kernels (reads fp64 A/B, writes s×INT8 slices).
    # Per pass: (8·n_scale_chunks + s) bytes per (m,k) or (k,n) element.
    t_scale_kern = (
        (mk + kn) * (8.0 * n_scale_chunks + s) / c0
        + 2.0 * n_scale_chunks * LATENCY_KERNEL
    ) / eff_scale

    # All s batched INT8 GEMMs (one per modulus).
    t_int8_gemms = (
        s * max(2.0 * mnk / c2, t_int8_bw) + n_chunks * LATENCY_MATMUL
    )

    # CRT accumulation + finalize kernels.
    # (4s + 32·n_chunks − 16) bytes/elem of mn accounts for reading
    # s INT32 slices plus double-double Zhi/Zlo accumulation.
    t_accum_kern = (
        mn * (4.0 * s + 32.0 * n_chunks - 16.0) / c0
        + n_chunks * LATENCY_KERNEL
    ) / EFF_ACCUM

    t_host = OZ2_HOST_OVERHEAD_S

    # ADP (dynamic precision selection) overhead — includes two small
    # reduction kernels and one hipStreamSynchronize round-trip.
    if dynamic_mode:
        t_adp = (
            LATENCY_KERNEL                            # oz2_adp_reduce_A
            + max(4.0 * mn / c0, LATENCY_KERNEL)     # oz2_adp_reduce_B
            + LATENCY_SYNC                            # hipStreamSynchronize
        )
    else:
        t_adp = 0.0

    # ── Fused kernel: replaces INT8 GEMMs + accum when faster ────────────────
    # Reads:  s × INT8 A8i + s × INT8 B8i  (s*(mk+kn) bytes)
    #         FP64 C input (8*mn bytes, when beta≠0) + FP64 D output (8*mn bytes)
    # Compute: MFMA  s×2×mnk ops at INT8 throughput c2
    #          CRT   s×8×mn  FP64 ops at FP64 throughput c1
    # Fused kernel bandwidth: reads s×INT8 A8i/B8i, writes FP64 D (+ optional C read).
    t_fused_bw   = (s * (mk + kn) + 16.0 * mn) / c0
    # Fused kernel MFMA compute: s×2×mnk INT8 ops at effective throughput c2×EFF_FUSED.
    t_fused_int8 = s * 2.0 * mnk / (c2 * EFF_FUSED)
    # CRT FP64 accumulation is done in registers; on CDNA VALU FP64 dual-issues
    # alongside MFMA, so CRT compute is hidden.  The non-fused t_accum_kern model
    # also treats the CRT as BW-bound (ignoring its FP64 compute cost), so for
    # consistency we do the same here: t_fused_fp64 is excluded from t_fused_kern.
    t_fused_fp64 = s * 8.0 * mn / c1  # informational only — not added to t_fused_kern
    t_fused_kern = max(t_fused_bw, t_fused_int8) + LATENCY_KERNEL

    # Select GEMM+accum time: use fused when requested and faster
    if use_fused:
        t_gemm_accum = min(t_int8_gemms + t_accum_kern, t_fused_kern)
    else:
        t_gemm_accum = t_int8_gemms + t_accum_kern

    # ── Total emulation time ──────────────────────────────────────────────────
    t_total = (
        t_prelim_kern
        + t_prelim_gemm
        + t_refine_kern
        + t_scale_kern
        + t_gemm_accum
        + t_host
        + t_adp
    )

    # ── Native DGEMM time (roofline model) ───────────────────────────────────
    # FP64 GEMM: max of compute-bound (2·m·n·k / c1) and
    #            bandwidth-bound (8 bytes × (mk + kn + mn) / c0).
    t_native = (
        max(2.0 * mnk / c1, 8.0 * (mk + kn + mn) / c0)
        + LATENCY_MATMUL
    )

    return {
        "t_prelim_kern": t_prelim_kern,
        "t_prelim_gemm": t_prelim_gemm,
        "t_refine_kern": t_refine_kern,
        "t_adp":         t_adp,
        "t_scale_kern":  t_scale_kern,
        "t_int8_gemms":  t_int8_gemms,
        "t_accum_kern":  t_accum_kern,
        "t_fused_kern":  t_fused_kern,
        "t_host":        t_host,
        "t_total":       t_total,
        "t_native":      t_native,
    }

# =============================================================================
# 5.  Heatmap computation
# =============================================================================

def compute_time_ratio_grid(
    hw: dict,
    mn_vals: np.ndarray,
    k_vals:  np.ndarray,
    num_moduli: int = 16,
    tA: bool = True,
    tB: bool = False,
) -> np.ndarray:
    """
    Return a 2D array  ratio[i, j] = t_split_emul / t_fused_emul (unclamped).

    Values > 1 where the fused kernel is faster than the non-fused
    recursive-splitting path, < 1 where it is slower.  Unlike the clamped
    speedup grids (which both equal 1.0 wherever emulation loses to native
    DGEMM), this ratio varies over the full grid and exposes the fused
    kernel's C32i I/O savings even in regions where native DGEMM wins.
    """
    ratio = np.empty((len(k_vals), len(mn_vals)), dtype=float)
    for i, k in enumerate(k_vals):
        for j, mn in enumerate(mn_vals):
            t_split = oz2_effective_time(tA, tB, int(mn), int(mn), int(k),
                                         num_moduli, hw, use_fused=False)
            t_fused = oz2_effective_time(tA, tB, int(mn), int(mn), int(k),
                                         num_moduli, hw, use_fused=True)
            ratio[i, j] = t_split / max(t_fused, 1e-30)
    return ratio


def compute_speedup_grid(
    hw: dict,
    mn_vals: np.ndarray,   # m=n values (X-axis)
    k_vals:  np.ndarray,   # k values   (Y-axis)
    num_moduli: int = 16,
    tA: bool = True,
    tB: bool = False,
    use_splitting: bool = False,
    use_fused: bool = False,
) -> np.ndarray:
    """
    Return a 2D array  speedup[i, j] = max(1, t_native / t_emul)
    where i indexes k_vals (rows) and j indexes mn_vals (columns).

    When use_splitting=True the emulation time is computed with
    oz2_effective_time() which mirrors the recursive binary-halving in
    oz2_effective_time_ms() / fp64EmulatedGemmImpl from fp64_emulation.cpp.
    When use_splitting=False (default) the original monolithic model is used.
    When use_fused=True the fused MFMA+CRT kernel model is used instead of
    the separate INT8 GEMM + accumulation kernels.
    """
    speedup = np.empty((len(k_vals), len(mn_vals)), dtype=float)
    for i, k in enumerate(k_vals):
        for j, mn in enumerate(mn_vals):
            t = perf_model_times(tA, tB, int(mn), int(mn), int(k),
                                 num_moduli, hw, use_fused=use_fused)
            if use_splitting:
                t_emul = oz2_effective_time(tA, tB, int(mn), int(mn), int(k),
                                            num_moduli, hw, use_fused=use_fused)
            else:
                t_emul = t["t_total"]
            speedup[i, j] = max(1.0, t["t_native"] / t_emul)
    return speedup

# =============================================================================
# 6.  Figure helpers
# =============================================================================

_LOG_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0,
                   8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0,
                   30.0, 50.0, 70.0, 100.0, 150.0, 200.0]


def _build_levels(sp_max: float) -> list:
    """Discrete colour levels spanning [1, sp_max].

    For narrow speedup ranges (sp_max ≤ 3) use a 0.2-step linear grid so that
    chips like MI355X (max ≈ 2.2×) get ~6-8 distinct colour bands rather than
    the 3 bands that the sparse log candidates would produce.  For wider ranges
    fall back to the log-spaced _LOG_CANDIDATES list.
    """
    if sp_max <= 3.0:
        candidates = [round(1.0 + 0.1 * i, 1) for i in range(22)]  # 1.0, 1.1, …, 3.1
        return sorted(set(
            [l for l in candidates if 1.0 <= l <= sp_max] + [sp_max]
        ))
    return sorted(set(
        [l for l in _LOG_CANDIDATES if l <= sp_max * 1.05] + [1.0, sp_max]
    ))


def _select_ticks(vals: np.ndarray, n_ticks: int = 7) -> list:
    """Pick n_ticks indices evenly spread across vals (log-spaced grid)."""
    idx = np.round(np.linspace(0, len(vals) - 1, n_ticks)).astype(int)
    return [vals[i] for i in sorted(set(idx))]


def _fmt(v: int) -> str:
    """Format tick label: 'Xk' for values ≥ 1000."""
    return f"{v // 1000}k" if v >= 1000 else str(v)


def _draw_heatmap(ax, speedup: np.ndarray, mn_vals: np.ndarray,
                  k_vals: np.ndarray, cmap, norm, levels: list,
                  show_ylabel: bool = True) -> object:
    """
    Draw a single speedup heatmap panel onto ax.
    Returns the pcolormesh image handle for the caller to attach a colorbar.
    """
    speedup_T = speedup.T  # shape (len_mn, len_k)

    im = ax.pcolormesh(k_vals, mn_vals, speedup_T, cmap=cmap, norm=norm,
                       shading="nearest")


    ax.set_xscale("log")
    ax.set_yscale("log")

    xt = _select_ticks(k_vals)
    yt = _select_ticks(mn_vals)
    ax.set_xticks(xt)
    ax.set_xticklabels([_fmt(v) for v in xt], fontsize=9)
    ax.set_yticks(yt)
    ax.set_yticklabels([_fmt(v) for v in yt] if show_ylabel else [""] * len(yt),
                       fontsize=9)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel("Inner dimension  $k$", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Output matrix size  $m = n$", fontsize=11)

    return im


# =============================================================================
# 7.  Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predicted FP64 GEMM emulation speedup heatmap (all or one HW config)"
    )
    parser.add_argument("--out-dir", default="plots",
                        help="Output directory for PNG/PDF (default: plots/)")
    parser.add_argument("--n-pts", type=int, default=20,
                        help="Number of grid points per axis (default: 20)")
    parser.add_argument("--mn-max", type=int, default=524288,
                        help="Upper bound for M=N axis (default: 524288 = 2^19)")
    parser.add_argument("--num-moduli", type=int, default=16,
                        help="Number of Ozaki-II moduli s (default: 16)")
    parser.add_argument("--recursive-split", action="store_true", default=False,
                        help="Account for recursive binary-halving (mirrors "
                             "oz2_effective_time_ms in fp64_emulation.cpp)")
    parser.add_argument("--hw", default="all",
                        help="Hardware config to plot. 'all' (default) generates "
                             "plots for every entry in HW_PARAMS. Or specify one "
                             "of: " + ", ".join(HW_PARAMS))
    parser.add_argument("--fused", action="store_true", default=False,
                        help="Also generate fused-kernel speedup plots, modelling "
                             "the oz2_fused_TN_kernel that fuses INT8 MFMA + CRT "
                             "accumulation (HIPBLASLT_EMULATION_FUSED=on path).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Grid definition ───────────────────────────────────────────────────────
    # m=n: log-spaced 512 → mn_max
    mn_vals = np.unique(np.round(
        np.logspace(np.log10(512), np.log10(args.mn_max), args.n_pts)
    ).astype(int))
    # k: log-spaced 512 → 65536
    k_vals  = np.unique(np.round(
        np.logspace(np.log10(512), np.log10(65536), args.n_pts)
    ).astype(int))

    # ── Determine which HW configs to run ────────────────────────────────────
    if args.hw == "all":
        hw_configs = HW_PARAMS
    elif args.hw in HW_PARAMS:
        hw_configs = {args.hw: HW_PARAMS[args.hw]}
    else:
        raise SystemExit(
            f"Unknown --hw '{args.hw}'. Valid choices: all, "
            + ", ".join(HW_PARAMS)
        )

    s = args.num_moduli

    for hw_name, hw in hw_configs.items():
        hw_slug = hw_name.lower()   # e.g. "mi455", "mi300x"

        # ── Compute both grids ───────────────────────────────────────────────
        print(f"\n[{hw_name}] Computing {len(mn_vals)}×{len(k_vals)} monolithic "
              f"speedup grid, s={s} …")
        speedup_mono = compute_speedup_grid(hw, mn_vals, k_vals, num_moduli=s,
                                            tA=True, tB=False, use_splitting=False)
        idx_m = int(np.argmax(speedup_mono))
        print(f"  max speedup = {speedup_mono.max():.3f}×  "
              f"(at m=n={mn_vals[idx_m % len(mn_vals)]}, "
              f"k={k_vals[idx_m // len(mn_vals)]})")
        print(f"  cells with speedup > 1: "
              f"{(speedup_mono > 1.0).sum()} / {speedup_mono.size}")

        print(f"[{hw_name}] Computing {len(mn_vals)}×{len(k_vals)} recursive-"
              f"splitting speedup grid, s={s} …")
        speedup_split = compute_speedup_grid(hw, mn_vals, k_vals, num_moduli=s,
                                             tA=True, tB=False, use_splitting=True)
        idx_s = int(np.argmax(speedup_split))
        print(f"  max speedup = {speedup_split.max():.3f}×  "
              f"(at m=n={mn_vals[idx_s % len(mn_vals)]}, "
              f"k={k_vals[idx_s // len(mn_vals)]})")
        print(f"  cells with speedup > 1: "
              f"{(speedup_split > 1.0).sum()} / {speedup_split.size}")

        # Count cells that benefit from splitting
        n_split_cells = int(np.sum(speedup_split > speedup_mono + 1e-9))
        print(f"  cells improved by splitting: {n_split_cells} / {speedup_mono.size}")

        # ── Optional fused-kernel grid ───────────────────────────────────────
        speedup_fused = None
        if args.fused:
            print(f"[{hw_name}] Computing {len(mn_vals)}×{len(k_vals)} fused-"
                  f"kernel speedup grid, s={s} …")
            speedup_fused = compute_speedup_grid(hw, mn_vals, k_vals, num_moduli=s,
                                                 tA=True, tB=False,
                                                 use_splitting=True, use_fused=True)
            idx_f = int(np.argmax(speedup_fused))
            print(f"  max speedup = {speedup_fused.max():.3f}×  "
                  f"(at m=n={mn_vals[idx_f % len(mn_vals)]}, "
                  f"k={k_vals[idx_f // len(mn_vals)]})")
            print(f"  cells with speedup > 1: "
                  f"{(speedup_fused > 1.0).sum()} / {speedup_fused.size}")
            n_fused_better = int(np.sum(speedup_fused > speedup_split + 1e-9))
            print(f"  cells improved by fused kernel: {n_fused_better} / {speedup_mono.size}")

        # ── Shared colormap / levels ─────────────────────────────────────────
        all_grids = [speedup_mono, speedup_split]
        if speedup_fused is not None:
            all_grids.append(speedup_fused)
        sp_max_combined = max(g.max() for g in all_grids)
        levels = _build_levels(sp_max_combined)
        cmap   = plt.get_cmap("coolwarm")
        norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        title_base = (
            f"Predicted FP64 emulation (s={s}) speedup over native DGEMM"
            f" — {hw_name}"
        )

        def _save_individual(speedup, label, stem_suffix):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = _draw_heatmap(ax, speedup, mn_vals, k_vals, cmap, norm, levels,
                               show_ylabel=True)
            cbar = fig.colorbar(im, ax=ax, pad=0.02, extend="neither")
            cbar.set_label("Predicted speedup", fontsize=9)
            cbar.set_ticks(levels)
            cbar.set_ticklabels([f"{l:.1f}×" for l in levels])
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(title_base + (f", {label}" if label else ""), fontsize=11)
            fig.tight_layout()
            stem = f"{hw_slug}_predicted_speedup_s{s}{stem_suffix}"
            for ext in ("png", "pdf"):
                out = os.path.join(args.out_dir, f"{stem}.{ext}")
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved  {out}")
            plt.close(fig)

        # ── Individual figures ───────────────────────────────────────────────
        _save_individual(speedup_mono,  "",                    "")
        _save_individual(speedup_split, "recursive splitting", "_split")
        if speedup_fused is not None:
            _save_individual(speedup_fused, "fused kernel", "_fused")

        # ── Combined comparison figure ────────────────────────────────────────
        if speedup_fused is not None:
            # Ratio: raw emulation-time ratio t_split / t_fused (unclamped).
            # Uses actual emulation times so the ratio varies everywhere —
            # not just in cells where emulation beats native DGEMM.
            print(f"[{hw_name}] Computing {len(mn_vals)}×{len(k_vals)} "
                  f"fused/non-fused time ratio grid, s={s} …")
            ratio_grid   = compute_time_ratio_grid(hw, mn_vals, k_vals, num_moduli=s,
                                                   tA=True, tB=False)
            ratio_max    = max(ratio_grid.max(), 1.001)
            ratio_levels = _build_levels(ratio_max)
            ratio_cmap   = plt.get_cmap("YlOrRd")
            ratio_norm   = BoundaryNorm(ratio_levels, ncolors=ratio_cmap.N, clip=True)

            # 2×2 layout: (a) mono  | (b) split
            #              (c) fused | (d) fused/non-fused ratio
            fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            ax_a, ax_b = axes2[0]
            ax_c, ax_d = axes2[1]

            im_a = _draw_heatmap(ax_a, speedup_mono,  mn_vals, k_vals, cmap, norm,
                                 levels, show_ylabel=True)
            im_b = _draw_heatmap(ax_b, speedup_split, mn_vals, k_vals, cmap, norm,
                                 levels, show_ylabel=True)
            im_c = _draw_heatmap(ax_c, speedup_fused, mn_vals, k_vals, cmap, norm,
                                 levels, show_ylabel=True)
            im_d = _draw_heatmap(ax_d, ratio_grid, mn_vals, k_vals,
                                 ratio_cmap, ratio_norm, ratio_levels, show_ylabel=True)

            ax_a.set_title("(a) Monolithic", fontsize=11)
            ax_b.set_title("(b) Recursive splitting", fontsize=11)
            ax_c.set_title("(c) Fused kernel", fontsize=11)
            ax_d.set_title("(d) Fused / non-fused ratio", fontsize=11)

            # Individual colorbars per panel
            for ax_p, im_p, lev, clabel, fmt in [
                (ax_a, im_a, levels,       "Speedup over native DGEMM", "{:.1f}×"),
                (ax_b, im_b, levels,       "Speedup over native DGEMM", "{:.1f}×"),
                (ax_c, im_c, levels,       "Speedup over native DGEMM", "{:.1f}×"),
                (ax_d, im_d, ratio_levels, "Fused / non-fused ratio",   "{:.2f}×"),
            ]:
                cb = fig2.colorbar(im_p, ax=ax_p, pad=0.02, extend="neither")
                cb.set_label(clabel, fontsize=8)
                cb.set_ticks(lev)
                cb.set_ticklabels([fmt.format(l) for l in lev])
                cb.ax.tick_params(labelsize=7)

        else:
            # Two-panel: monolithic | recursive splitting
            fig2, (ax_mono2, ax_split2) = plt.subplots(
                1, 2, figsize=(8, 4), constrained_layout=True)
            _draw_heatmap(ax_mono2,  speedup_mono,  mn_vals, k_vals, cmap, norm,
                          levels, show_ylabel=True)
            im2 = _draw_heatmap(ax_split2, speedup_split, mn_vals, k_vals, cmap, norm,
                                levels, show_ylabel=False)
            ax_mono2.set_title("(a) Monolithic", fontsize=11)
            ax_split2.set_title("(b) Recursive splitting", fontsize=11)

            cbar2 = fig2.colorbar(im2, ax=ax_split2, pad=0.02, extend="neither")
            cbar2.set_label("Predicted speedup over native DGEMM", fontsize=9)
            cbar2.set_ticks(levels)
            cbar2.set_ticklabels([f"{l:.1f}×" for l in levels])
            cbar2.ax.tick_params(labelsize=8)

        fig2.suptitle(title_base, fontsize=12)

        stem2 = f"{hw_slug}_predicted_speedup_s{s}_comparison"
        for ext in ("png", "pdf"):
            out2 = os.path.join(args.out_dir, f"{stem2}.{ext}")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            print(f"Saved  {out2}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
