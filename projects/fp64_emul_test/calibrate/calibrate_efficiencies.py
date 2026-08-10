#!/usr/bin/env python3
"""
calibrate_efficiencies.py

Reads the profile CSV produced by run_calibrate.sh and computes updated
kernel efficiency constants for the fp64 emulation performance model in
fp64_emulation.cpp.

Usage:
    python3 calibrate_efficiencies.py [emul_calib.csv] [--cold-iters N]

The script computes efficiencies FROM SCRATCH using the raw model bandwidth
formulas and the hardware constants — it does NOT rely on the old efficiency
values already embedded in the pred_* columns of the CSV.

Calibration formula (per shape/combo, after cold-iteration removal):
    eff_X[tA][tB] = mean( raw_model_X(m,n,k,s,n_chunks) / t_X_ms )

where raw_model_X is the ideal bandwidth/compute time for component X
computed directly from the dimensions and the device's HBM bandwidth c0.

Device auto-detection uses pred_native_dgemm_ms to estimate c1 (FP64
throughput) and matches against the known hw_params table.

All rows are used (no num_sub_gemms filter) — component times from
binary-halving GEMMs are valid sums over sub-GEMMs.
"""

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Hardware parameters (verbatim from fp64_emulation.cpp) ─────────────────────
# Maps PCI device ID → (ai, ratio, latency)
# c0 = latency / LATENCY_MATMUL_S   (HBM bandwidth, bytes/s)
# c1 = c0 * ai                       (FP64 throughput effective)
# c2 = c1 * ratio                    (INT8 throughput effective)
HW_PARAMS = {
    0x74a0: ("MI300X",  2.63868, 168.411, 3.93e7),
    0x74a1: ("MI300X",  2.63868, 168.411, 3.93e7),
    0x74a9: ("MI300X",  2.63868, 168.411, 3.93e7),
    0x75a0: ("MI350X", 11.2237,  46.4678, 6.08e7),
    0x75b0: ("MI350X", 11.2237,  46.4678, 6.08e7),
    0x75a3: ("MI355X", 11.3284,  42.7280, 6.82e7),
    0x75b3: ("MI355X", 11.3284,  42.7280, 6.82e7),
}

# Latency constants (seconds) — must match fp64_emulation.cpp
LATENCY_MATMUL_S = 10e-6   # hipBLASLt matmul launch overhead
LATENCY_KERNEL_S =  5e-6   # GPU kernel scheduling overhead
LATENCY_MEMSET_S =  2e-6   # hipMemsetAsync overhead

# Cold-iteration rows to drop per (m,n,k,transA,transB) group.
# Set to 2 so 32K cubic shapes (cold_iters=2 in the YAML) keep all 10 warm rows.
# For regular shapes (cold_iters=10 in YAML), dropping 2 still leaves 108 warm rows.
COLD_ITERS = 2


# ── Raw model formulas (bandwidth-limited, no efficiency divisor) ───────────────

def raw_prelim_ms(m, n, k, c0):
    """Ideal prelim-kernel time (ms): FP64 A+B reads + INT8 writes."""
    return ((m * k + k * n) * 17.0 / c0 + 2.0 * LATENCY_KERNEL_S) * 1000.0


def raw_refine_ms(m, n, c0):
    """Ideal shift-refinement kernel time (ms)."""
    return (m * n * 8.0 / c0 + 3.0 * LATENCY_KERNEL_S + LATENCY_MEMSET_S) * 1000.0


def raw_scale_ms(m, n, k, s, n_scale_chunks, c0):
    """Ideal scale-kernel time (ms): read FP64, write s×INT8."""
    return ((m * k + k * n) * (8.0 * n_scale_chunks + s) / c0
            + 2.0 * n_scale_chunks * LATENCY_KERNEL_S) * 1000.0


def raw_accum_ms(m, n, s, n_chunks, c0):
    """Ideal CRT-accumulation kernel time (ms)."""
    return (m * n * (4.0 * s + 32.0 * n_chunks - 16.0) / c0
            + n_chunks * LATENCY_KERNEL_S) * 1000.0


# ── Device auto-detection ───────────────────────────────────────────────────────

def detect_device(df: pd.DataFrame) -> tuple:
    """
    Auto-detect the GPU from pred_native_dgemm_ms in the CSV.

    For compute-bound large GEMMs:  pred_native_ms ≈ 2·m·n·k / c1 · 1000
    → estimate c1, then find the closest hw_params entry.

    Returns (pci_id, name, c0).
    """
    if "pred_native_dgemm_ms" not in df.columns:
        raise ValueError("Column 'pred_native_dgemm_ms' not found — cannot auto-detect device.")

    # Use large square shapes where the compute term dominates
    sel = df[(df["m"] == df["n"]) & (df["m"].astype(float) >= 4096)
             & (df["k"].astype(float) >= 2048)].copy()
    if sel.empty:
        sel = df.copy()

    mnk  = sel["m"].values.astype(float) * sel["n"].values.astype(float) * sel["k"].values.astype(float)
    pred = sel["pred_native_dgemm_ms"].values.astype(float) / 1000.0  # → seconds
    valid = pred > 1e-9
    est_c1 = np.where(valid, 2.0 * mnk / pred, np.nan)
    est_c1 = float(np.nanmedian(est_c1))

    best_pci_id, best_name, best_diff = None, None, np.inf
    for pci_id, (name, ai, ratio, latency) in HW_PARAMS.items():
        c0 = latency / LATENCY_MATMUL_S
        c1 = c0 * ai
        diff = abs(c1 - est_c1) / max(est_c1, 1.0)
        if diff < best_diff:
            best_diff, best_pci_id, best_name = diff, pci_id, name

    ai, ratio, latency = HW_PARAMS[best_pci_id][1:]
    c0 = latency / LATENCY_MATMUL_S
    print(f"  Detected device : {best_name}  (PCI 0x{best_pci_id:04x})")
    print(f"  c0 = {c0:.3e} B/s   ai = {ai}   ratio = {ratio}"
          f"   (match error {best_diff:.1%})")
    return best_pci_id, best_name, c0


# ── Cold-row removal ────────────────────────────────────────────────────────────

def drop_cold_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the first COLD_ITERS rows per (m,n,k,transA,transB) group."""
    key = ["m", "n", "k", "transA", "transB"]
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))
    df_sorted = df.sort_values(key + ["_orig_idx"])
    df_sorted["_rank"] = df_sorted.groupby(key).cumcount()
    warm = df_sorted[df_sorted["_rank"] >= COLD_ITERS].drop(
        columns=["_rank", "_orig_idx"])
    n_shapes = df.groupby(key).ngroups
    dropped  = len(df) - len(warm)
    print(f"  Dropped {dropped} cold-iteration rows "
          f"({COLD_ITERS} per shape × {n_shapes} shapes)")
    return warm.reset_index(drop=True)


# ── Statistics helper ───────────────────────────────────────────────────────────

def eff_stats(raw_model: pd.Series, meas: pd.Series, label: str):
    """
    Compute mean/std of raw_model/meas (= the efficiency), excluding meas ≤ 0.
    Clips outliers beyond ±3σ before averaging.
    Returns (mean_eff, std_eff, n_valid).
    """
    valid = meas > 0
    if valid.sum() == 0:
        warnings.warn(f"No valid rows for {label} (all measured times are zero or negative).")
        return np.nan, np.nan, 0
    eff = raw_model[valid] / meas[valid]
    mu, sigma = eff.mean(), eff.std()
    if sigma > 0:
        inliers = eff[(eff >= mu - 3 * sigma) & (eff <= mu + 3 * sigma)]
        if len(inliers) >= len(eff) * 0.5:
            eff = inliers
    return float(eff.mean()), float(eff.std()), len(eff)


# ── Main calibration routine ────────────────────────────────────────────────────

def calibrate(csv_path: str) -> None:
    print(f"\nReading profile CSV : {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True)
    # Normalise column names to lowercase for robustness
    df.columns = [c.strip() for c in df.columns]
    print(f"  Total rows        : {len(df)}")
    print(f"  Columns           : {', '.join(df.columns[:15])}{'…' if len(df.columns) > 15 else ''}")

    # Detect hardware
    _, _dev_name, c0 = detect_device(df)

    # Validate required columns
    required = ["m", "n", "k", "transA", "transB",
                "t_prelim_ms", "t_refine_ms", "t_scale_ms", "t_accum_ms",
                "effective_s", "scale_chunk_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}\n"
                         f"Available: {df.columns.tolist()}")

    # Keep only monolithic GEMMs (num_sub_gemms==1).
    # Binary-halved shapes use sub-GEMMs with different chunk_sizes, so the
    # raw_model formula (which assumes a single monolithic computation) gives
    # incorrect results — typically eff >> 1 for large shapes.
    before = len(df)
    if "num_sub_gemms" in df.columns:
        df = df[df["num_sub_gemms"] == 1].copy()
        n_excl = before - len(df)
        if n_excl:
            print(f"  Excluded {n_excl} rows with num_sub_gemms>1 "
                  f"(binary-halved — raw model formula invalid for those shapes).")
    print(f"  Using {len(df)} rows (num_sub_gemms==1).")

    # Drop cold iterations
    df = drop_cold_rows(df)
    print(f"  Warm-iteration rows remaining: {len(df)}")

    # Compute auxiliary columns
    df["tA"]    = df["transA"].str.upper() == "T"
    df["tB"]    = df["transB"].str.upper() == "T"
    df["combo"] = df["transA"].str.upper() + df["transB"].str.upper()

    s_arr   = df["effective_s"].values.astype(float)
    chunk   = df["scale_chunk_size"].values.astype(float).clip(min=1)
    nc_arr  = np.ceil(s_arr / chunk).astype(int)  # n_chunks
    df["n_chunks"] = nc_arr

    m_arr = df["m"].values.astype(float)
    n_arr = df["n"].values.astype(float)
    k_arr = df["k"].values.astype(float)

    df["raw_prelim_ms"] = raw_prelim_ms(m_arr, n_arr, k_arr, c0)
    df["raw_refine_ms"] = raw_refine_ms(m_arr, n_arr, c0)
    df["raw_scale_ms"]  = raw_scale_ms(m_arr, n_arr, k_arr, s_arr, nc_arr, c0)
    df["raw_accum_ms"]  = raw_accum_ms(m_arr, n_arr, s_arr, nc_arr, c0)

    # ── Per-combo calibration ─────────────────────────────────────────────────
    print("\n── Per-(tA,tB) efficiency calibration ───────────────────────────")
    results = {}

    for combo in ["NN", "NT", "TN", "TT"]:
        sub = df[df["combo"] == combo]
        if sub.empty:
            print(f"  {combo}: NO DATA — skipping")
            continue
        n_shapes = sub.groupby(["m", "n", "k"]).ngroups
        print(f"\n  Combo {combo}  ({len(sub)} rows across {n_shapes} shapes)")

        mu, std, n = eff_stats(sub["raw_prelim_ms"], sub["t_prelim_ms"], f"prelim_{combo}")
        results[f"prelim_{combo}"] = mu
        print(f"    prelim : eff = {mu:.4f} ± {std:.4f}  (n={n})")

        mu, std, n = eff_stats(sub["raw_scale_ms"],  sub["t_scale_ms"],  f"scale_{combo}")
        results[f"scale_{combo}"] = mu
        print(f"    scale  : eff = {mu:.4f} ± {std:.4f}  (n={n})")

    # ── Scalar efficiencies ───────────────────────────────────────────────────
    print("\n── Scalar efficiency calibration ────────────────────────────────")

    mu, std, n = eff_stats(df["raw_refine_ms"], df["t_refine_ms"], "refine")
    results["refine"] = mu
    print(f"  refine : eff = {mu:.4f} ± {std:.4f}  (n={n})")

    mu, std, n = eff_stats(df["raw_accum_ms"],  df["t_accum_ms"],  "accum")
    results["accum"] = mu
    print(f"  accum  : eff = {mu:.4f} ± {std:.4f}  (n={n})")

    # ── Per-shape breakdown ───────────────────────────────────────────────────
    print("\n── Per-shape breakdown ───────────────────────────────────────────")
    for combo in ["NN", "NT", "TN", "TT"]:
        sub = df[df["combo"] == combo]
        if sub.empty:
            continue
        print(f"\n  [{combo}]  eff_prelim             eff_scale              eff_refine             eff_accum")
        for _, g in sub.groupby(["m", "n", "k"], sort=True):
            mv = int(g["m"].iloc[0])
            nv = int(g["n"].iloc[0])
            kv = int(g["k"].iloc[0])
            ep = g["raw_prelim_ms"] / g["t_prelim_ms"]
            es = g["raw_scale_ms"]  / g["t_scale_ms"]
            er = g["raw_refine_ms"] / g["t_refine_ms"]
            ea = g["raw_accum_ms"]  / g["t_accum_ms"]
            print(f"    {mv:6d}×{nv:6d}×{kv:6d} :  "
                  f"{ep.mean():.4f}±{ep.std():.4f}    "
                  f"{es.mean():.4f}±{es.std():.4f}    "
                  f"{er.mean():.4f}±{er.std():.4f}    "
                  f"{ea.mean():.4f}±{ea.std():.4f}")

    # ── C++ output ────────────────────────────────────────────────────────────
    def r(key):
        return results.get(key, float("nan"))

    cpp = f"""
/* ── Recalibrated efficiency constants ─────────────────────────────────────
 * Generated by calibrate_efficiencies.py from: {os.path.basename(csv_path)}
 * Dataset : {len(df)} warm iterations (cold_iters={COLD_ITERS} dropped per shape).
 * Method  : eff = mean(raw_model_bandwidth / t_measured)  — no old-eff dependency.
 *           raw_model computed from M,N,K and auto-detected hardware c0.
 * ──────────────────────────────────────────────────────────────────────────── */
static constexpr double eff_prelim[2][2] = {{
    {{{r("prelim_NN"):.3f}, {r("prelim_NT"):.3f}}},   /* [N][N], [N][T] */
    {{{r("prelim_TN"):.3f}, {r("prelim_TT"):.3f}}}}};  /* [T][N], [T][T] */
static constexpr double eff_scale[2][2]  = {{
    {{{r("scale_NN"):.3f},  {r("scale_NT"):.3f}}},   /* [N][N], [N][T] */
    {{{r("scale_TN"):.3f},  {r("scale_TT"):.3f}}}}};  /* [T][N], [T][T] */
static constexpr double eff_refine        = {r("refine"):.3f};
static constexpr double eff_accum         = {r("accum"):.3f};"""

    print("\n═══════════════════════════════════════════════════════════════════")
    print("  C++ snippet — paste into PerfModelKernelEffs in fp64_emulation.cpp")
    print("═══════════════════════════════════════════════════════════════════")
    print(cpp)
    print("═══════════════════════════════════════════════════════════════════\n")

    out_path = os.path.splitext(csv_path)[0] + "_new_effs.txt"
    with open(out_path, "w") as fout:
        fout.write(cpp + "\n")
    print(f"C++ snippet also written to: {out_path}\n")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    global COLD_ITERS
    parser = argparse.ArgumentParser(
        description="Calibrate fp64 emulation kernel efficiency constants from a profile CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "emul_calib.csv"),
        help="Path to the profile CSV (default: emul_calib.csv next to this script)",
    )
    parser.add_argument(
        "--cold-iters",
        type=int,
        default=COLD_ITERS,
        dest="cold_iters",
        help=f"Cold-iteration rows to drop per shape (default: {COLD_ITERS})",
    )
    args = parser.parse_args()
    COLD_ITERS = args.cold_iters

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: CSV file not found: {args.csv}\n"
                 "Run run_calibrate.sh first to generate it.")

    calibrate(args.csv)


if __name__ == "__main__":
    main()
