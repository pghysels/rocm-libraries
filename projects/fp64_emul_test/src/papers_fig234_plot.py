#!/usr/bin/env python3
"""
papers_fig234_plot.py

Plots Figures 2, 3, and 4 reproducing the numerical-accuracy experiments from:

  "Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores
   Through Extensions of the Ozaki Scheme"
  arXiv:2511.13778  (SCA/HPCAsia 2026)

Figure 2 (BLAS Test 2):
  X-axis : exponent-range parameter b
  Y-axis : max componentwise relative error  (log scale)
  Lines  : one per emulated-DGEMM moduli count s ∈ {4,6,8,10,12,16,18}

Figure 3 (Grade A criterion – maximum error):
  X-axis : matrix size N  (log scale)
  Y-axis : max componentwise relative error vs. DD-GEMM reference  (log scale)
  Lines  : native FP64 DGEMM + emulated DGEMM at each s value
  Refs   : Grade-A slope  O(n·ε),  Grade-B slope  O(n²·ε)

Figure 4 (Grade A criterion – median error):
  Same layout as Figure 3 but using median instead of max.

Usage:
  python3 papers_fig234_plot.py [--fig2  fig2_results.csv]
                                [--fig34 fig34_results.csv]
                                [--out-dir .]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib pandas numpy")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EPS_MACH = 2.0 ** -52  # FP64 unit round-off

# CRT mantissa bits for each num_moduli value (from CRT_BITS table)
CRT_BITS_MAP = {
    4:  31.945,
    6:  47.807,
    8:  63.572,
    10: 79.238,
    12: 94.801,
    14: 110.160,
    15: 117.782,
    16: 125.374,
    17: 132.949,
    18: 140.448,
}

# Colour palette: one colour per s value (consistent across all figures)
_MODULI_COLORS = {
    4:  "#d62728",   # red
    6:  "#ff7f0e",   # orange
    8:  "#bcbd22",   # yellow-green
    10: "#2ca02c",   # green
    12: "#17becf",   # cyan
    14: "#e377c2",   # pink
    15: "#7f7f7f",   # gray
    16: "#1f77b4",   # blue
    17: "#8c564b",   # brown
    18: "#9467bd",   # purple
}

_NATIVE_COLOR  = "black"
_NATIVE_MARKER = "^"
_NATIVE_LABEL  = "Native FP64 DGEMM"


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure2(df: pd.DataFrame, out_dir: str, n_fig2: int = 1024) -> None:
    """Figure 2: BLAS Test 2 error vs. exponent-range parameter b (log-log).

    Supports both the old format (columns: b, num_moduli, crt_bits, err_max)
    and the new format (adds algo column as first column).
    """
    import math

    fig, ax = plt.subplots(figsize=(8, 4))

    # Drop b=0 (undefined on log x-axis) and any non-positive b values
    df = df[df["b"] > 0].copy()

    has_algo = "algo" in df.columns

    # ── Fixed-s emulated DGEMM lines ─────────────────────────────────────────
    # Show the highest-precision fixed-s configs
    FIG2_SHOW = {14, 15, 16, 17, 18}

    if has_algo:
        fixed_df = df[df["algo"].str.startswith("OS2-fixed-s", na=False)].copy()
        # Extract s from algo string "OS2-fixed-sN"
        fixed_df = fixed_df.copy()
        fixed_df["s_val"] = fixed_df["algo"].str.replace("OS2-fixed-s", "", regex=False).astype(int)
        moduli_seen = sorted(s for s in fixed_df["s_val"].unique() if s in FIG2_SHOW)
        for s in moduli_seen:
            sub   = fixed_df[fixed_df["s_val"] == s].sort_values("b")
            bits  = CRT_BITS_MAP.get(s, float(s) * 7.8)
            color = _MODULI_COLORS.get(s, "gray")
            ax.loglog(
                sub["b"], sub["err_max"],
                marker="o", linewidth=1.8, markersize=4,
                color=color, label=f"OS2-accu  s = {s}  (~{bits:.0f} bits)",
            )
    else:
        # Old format: num_moduli column holds s
        moduli_seen = sorted(s for s in df["num_moduli"].unique() if int(s) in FIG2_SHOW)
        for s in moduli_seen:
            sub   = df[df["num_moduli"] == s].sort_values("b")
            bits  = CRT_BITS_MAP.get(int(s), float(s) * 7.8)
            color = _MODULI_COLORS.get(int(s), "gray")
            ax.loglog(
                sub["b"], sub["err_max"],
                marker="o", linewidth=1.8, markersize=4,
                color=color, label=f"OS2-accu  s = {s}  (~{bits:.0f} bits)",
            )

    # ── Native FP64 DGEMM (new format only) ───────────────────────────────────
    if has_algo:
        nat_df = df[df["algo"] == "DGEMM"].sort_values("b")
        if not nat_df.empty:
            ax.loglog(
                nat_df["b"], nat_df["err_max"],
                color="black", marker="^", linewidth=2.0, markersize=6,
                label="Native FP64 DGEMM",
                zorder=5,
            )

    # ── ADP (dynamic mode, new format only) ───────────────────────────────────
    if has_algo:
        adp_df = df[df["algo"] == "ADP"].sort_values("b")
        if not adp_df.empty:
            ax.loglog(
                adp_df["b"], adp_df["err_max"],
                color="#8c564b", marker="D", linewidth=2.0, markersize=5,
                linestyle="--", label="ADP (dynamic, max s=18)",
                zorder=4,
            )

    # ── Reference lines ───────────────────────────────────────────────────────
    sqrt_n = math.isqrt(n_fig2)   # exact integer sqrt when n is a perfect square

    ax.axhline(
        math.sqrt(n_fig2) * EPS_MACH, color="dimgray", linestyle="--", linewidth=1.2,
        label=f"√n·ε  (n={n_fig2})",
    )
    ax.axhline(
        EPS_MACH, color="silver", linestyle=":", linewidth=1.0,
        label="ε_mach  (2⁻⁵²)",
    )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel("Exponent-range parameter  $b$", fontsize=12)
    ax.set_ylabel(r"$\max_{i,j}\,e_{ij}$", fontsize=12)
    ax.set_title(
        r"diag: $e_{kk}=|C_{kk}-\mathbf{x}^\top\!\mathbf{x}|/|\mathbf{x}^\top\!\mathbf{x}|$;"
        r"  off-diag: $e_{ij}=|C_{ij}-C_{ij}^{\mathrm{ref}}|/|C_{ij}^{\mathrm{ref}}|$",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # Set x-axis ticks explicitly at each b value (powers of 2)
    b_vals = sorted(df["b"].unique())
    ax.set_xticks(b_vals)
    ax.set_xticklabels([str(b) for b in b_vals])
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # suppress minor ticks between
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    ax.set_ylim(1e-17, 1e-1)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"figure2.{ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figures 3 & 4  (shared helper)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_s(algo: str) -> int | None:
    """Parse num_moduli from algo string 'OS2-accu-sN' → N, or None."""
    if algo.startswith("OS2-accu-s"):
        try:
            return int(algo.split("-s")[-1])
        except ValueError:
            pass
    return None


def plot_figure_34(
    df: pd.DataFrame,
    fig_num: int,
    err_col: str,
    ylabel: str,
    title_suffix: str,
    out_dir: str,
    extra_ref_lines: list | None = None,
    show_grade_a: bool = True,
) -> None:
    """Generic plotter shared by Figures 3 and 4.

    extra_ref_lines: list of (label, fn, color, linestyle, linewidth) tuples
                     where fn(N_ref) → y values for the slope line.
    show_grade_a   : if False, suppress the n·ε Grade-A slope line.
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    df_native = df[df["algo"] == "DGEMM"].sort_values("N")
    df_adp    = df[df["algo"] == "ADP-dynamic"].sort_values("N")
    df_emul   = df[(df["algo"] != "DGEMM") & (df["algo"] != "ADP-dynamic")]

    # ── Native FP64 ───────────────────────────────────────────────────────────
    if not df_native.empty:
        ax.loglog(
            df_native["N"], df_native[err_col],
            color=_NATIVE_COLOR, marker=_NATIVE_MARKER,
            linewidth=2.0, markersize=6,
            label=_NATIVE_LABEL,
            zorder=5,
        )

    # ── ADP (dynamic mode) ───────────────────────────────────────────────────
    if not df_adp.empty:
        ax.loglog(
            df_adp["N"], df_adp[err_col],
            color="#8c564b", marker="D", linewidth=2.0, markersize=5,
            linestyle="--", label="ADP (dynamic, max s=18)",
            zorder=4,
        )

    # ── Emulated DGEMM per s ──────────────────────────────────────────────────
    algos_with_s = []
    for algo in df_emul["algo"].unique():
        s = _extract_s(algo)
        if s is not None:
            algos_with_s.append((s, algo))
    algos_with_s.sort()

    _MARKERS = ["o", "s", "^", "D", "v"]
    _LSTYLES = ["-", "--", "-.", ":"]
    for idx, (s, algo) in enumerate(algos_with_s):
        if s == 6:
            continue
        sub   = df_emul[df_emul["algo"] == algo].sort_values("N")
        bits  = CRT_BITS_MAP.get(s, float(s) * 7.8)
        color = _MODULI_COLORS.get(s, "gray")
        label = f"Emulated  s = {s}  (~{bits:.0f} bits)"
        ax.loglog(
            sub["N"], sub[err_col],
            marker=_MARKERS[idx % len(_MARKERS)],
            linestyle=_LSTYLES[idx % len(_LSTYLES)],
            linewidth=1.5, markersize=5,
            color=color, label=label,
        )

    # ── Reference lines — span the full visible x-axis range ──────────────
    N_ref = np.array([90.0, 40000.0])

    if not df_native.empty:
        nat_N   = df_native["N"].values.astype(float)
        nat_err = df_native[err_col].values

        if show_grade_a:
            # O(n) slope: c chosen so c*N is 50% above the tightest upper bound
            c_n = np.max(nat_err / nat_N) * 1.5
            ax.loglog(
                N_ref, c_n * N_ref, color="steelblue", linestyle="--",
                linewidth=1.3, label="O(n) slope",
            )
        if extra_ref_lines:
            for ref_label, fn, color, ls, lw in extra_ref_lines:
                # Pass native data so each lambda can fit its own constant
                y_ref = fn(N_ref, nat_N, nat_err)
                ax.loglog(N_ref, y_ref, color=color, linestyle=ls,
                          linewidth=lw, label=ref_label)
    else:
        if show_grade_a:
            ax.loglog(N_ref, N_ref * EPS_MACH, color="steelblue",
                      linestyle="--", linewidth=1.3, label="O(n) slope")

    # Machine epsilon horizontal reference
    ax.axhline(
        EPS_MACH, color="silver", linestyle="-.", linewidth=0.9,
        label="ε_mach  (2⁻⁵²)",
    )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xlabel(r"Matrix size  $m = n = k$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # x-axis: from just below 128 to just above 32K
    ax.set_xlim(90, 40000)
    n_vals = sorted(df["N"].unique())
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"figure{fig_num}.{ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_figures34(df: pd.DataFrame, out_dir: str) -> None:
    plot_figure_34(
        df,
        fig_num      = 3,
        err_col      = "err_max",
        ylabel       = r"$\max_{i,j}\,|\mathrm{fl}(AB)_{ij} - (AB)_{ij}|\,/\,|(AB)_{ij}|$",
        title_suffix = "Maximum",
        out_dir      = out_dir,
        extra_ref_lines=[
            ("O(√n) slope",
             lambda N_ref, nat_N, nat_err:
                 np.max(nat_err / np.sqrt(nat_N)) * 1.5 * np.sqrt(N_ref),
             "darkorange", "-.", 1.2),
        ],
    )
    plot_figure_34(
        df,
        fig_num      = 4,
        err_col      = "err_med",   # column holds arithmetic mean
        ylabel       = "Average componentwise relative error",
        title_suffix = "Average",
        out_dir      = out_dir,
        show_grade_a = False,
        extra_ref_lines=[
            # O(√n) slope fitted just above native DGEMM average errors.
            # For random U(0,1) matrices rounding errors partially cancel
            # (CLT), so the average error grows as √N·ε not N·ε.
            ("O(√n) slope",
             lambda N_ref, nat_N, nat_err:
                 np.max(nat_err / np.sqrt(nat_N)) * 1.5 * np.sqrt(N_ref),
             "darkorange", "-.", 1.2),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Figures 2, 3, 4 from arXiv:2511.13778"
    )
    parser.add_argument(
        "--fig2",
        default="fig2_results.csv",
        help="CSV produced by papers_fig234_bench --fig 2",
    )
    parser.add_argument(
        "--fig34",
        default="fig34_results.csv",
        help="CSV produced by papers_fig234_bench --fig 34",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Directory for output PNG files (default: plots/)",
    )
    parser.add_argument(
        "--n-fig2",
        type=int,
        default=1024,
        help="n used in the Fig-2 run (for the n·ε reference line, default: 1024)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    if os.path.exists(args.fig2):
        df2 = pd.read_csv(args.fig2)
        required = {"b", "err_max"}   # algo and num_moduli optional (old format compat)
        missing  = required - set(df2.columns)
        if missing:
            print(f"Warning: {args.fig2} is missing columns: {missing}")
        else:
            plot_figure2(df2, args.out_dir, n_fig2=args.n_fig2)
    else:
        print(f"Warning: {args.fig2} not found — skipping Figure 2")

    # ── Figures 3 & 4 ─────────────────────────────────────────────────────────
    if os.path.exists(args.fig34):
        df34 = pd.read_csv(args.fig34)
        required = {"algo", "N", "crt_bits", "err_max", "err_med"}
        missing  = required - set(df34.columns)
        if missing:
            print(f"Warning: {args.fig34} is missing columns: {missing}")
        else:
            plot_figures34(df34, args.out_dir)
    else:
        print(f"Warning: {args.fig34} not found — skipping Figures 3/4")


if __name__ == "__main__":
    main()
