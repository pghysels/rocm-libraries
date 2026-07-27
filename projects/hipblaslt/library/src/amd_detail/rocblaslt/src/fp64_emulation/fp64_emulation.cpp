// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/*
 * fp64_emulation.cpp
 *
 * FP64 GEMM emulation via Ozaki Scheme II (accurate mode) using INT8 Tensor Cores.
 *
 * Algorithm (paper: Ozaki, Uchino, Imamura, arXiv:2504.08009)
 * -----------------------------------------------------------
 * Given  D = alpha * op(A) * op(B) + beta * C   (A,B,C,D in FP64)
 *
 * Part 1 – Accurate scaling (OS II-accu):
 *   1a. Per-row 6-bit extraction of op(A) → A8i_high, with per-row shifts sftA[i].
 *   1b. Per-col 6-bit extraction of op(B) → B8i_high, with per-col shifts sftB[j].
 *   1c. Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high  (one extra GEMM).
 *   1d. Refine sftA[i] from per-row max of |C32i_prelim| (tighter bound than Cauchy-Schwarz).
 *   1e. Refine sftB[j] from per-col max of |C32i_prelim|.
 *   1f. Final scaling: A8i[t], B8i[t] for t=0..num_moduli-1 using refined sftA, sftB.
 *
 * Part 2 – For each of s coprime moduli m_t (including implicit m_0=256):
 *   2a.  A'_t = symmetric_mod(A', m_t)  → INT8  in [-m_t/2, m_t/2]
 *   2b.  B'_t = symmetric_mod(B', m_t)  → INT8
 *   2c.  C'_t = A'_t × B'_t             → INT32  (INT8 tensor cores)
 *   2d.  Z    += C'_t * qPi_t           (CRT accumulation, double-double)
 *
 * Part 3 – Range reduction: X = Z mod M  (unique because |X| < M/2)
 *
 * Part 4 – Per-element inverse scale: D[i,j] = alpha * X[i,j] * 2^-(sftA[i]+sftB[j])
 *                                              + beta * C[i,j]
 *
 * The number of moduli s (= number of INT8 GEMMs) is configurable at runtime via
 * HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT (default: s=16, ~125 bits of CRT
 * capacity, sufficient for guaranteed FP64-equivalent results on all inputs).
 *
 * Constants (tables) are taken verbatim from the open-source GEMMul8 implementation
 * (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8).
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 * Inner INT8 GEMMs use hipblasLtMatmul (INT8 tensor cores, INT32 accumulate).
 */

#include "fp64_emulation.hpp"
#include "fp64_emulation_fused.hpp"
#include "fp64_emulation_tables.hpp"
#include "handle.h"   /* _rocblaslt_handle */

#include "hipblaslt/hipblaslt.h"
#include <hip/hip_runtime.h>

#include <cstdlib>      // std::getenv
#include <cstdio>       // std::fopen / std::fprintf / std::fclose / std::ftell
#include <cstring>      // std::strcmp
#include <cmath>        // std::log2, std::floor, etc.
#include <cstdlib>      // std::getenv
#include <cstdio>       // std::fopen / std::fprintf / std::fclose / std::ftell
#include <cstring>      // std::strcmp
#include <cmath>        // std::log2, std::floor, etc.
#include <cerrno>
#include <climits>
#include <atomic>
#include <optional>     // std::optional
#include <unordered_map>// std::unordered_map
#include <optional>     // std::optional
#include <unordered_map>// std::unordered_map

/* =========================================================================
 * Tuning constants
 * ========================================================================= */
/* Workspace reserved for the INT8 GEMM algorithms (preliminary and batch).
 * Passing a non-null workspace allows hipBLASLt to select algorithms that
 * require workspace, which is necessary for very large k values where no
 * zero-workspace INT8 GEMM algorithm is available.
 * 128 MiB matches the default value of HIPBLASLT_TUNING_USER_MAX_WORKSPACE. */
static constexpr size_t OZ2_INT8_GEMM_WS_BYTES = 128ull << 20;  /* 128 MiB */

/* Total workspace budget per modulus (A8i + B8i + C32i simultaneously resident).
 * chunk × (mn4 + slc) ≤ OZ2_CHUNK_TARGET_BYTES constrains the combined allocation. */
static constexpr size_t OZ2_CHUNK_TARGET_BYTES = 32ull << 30;  /* 32 GiB */

/* Number of moduli processed per pass: the largest value ≤ s such that
 * the simultaneous allocation of A8i + B8i + C32i fits in the total budget.
 * Scale and GEMM share the same chunk — one scale launch followed by one
 * batched GEMM per pass.  The final pass handles the remainder naturally
 * via min(chunk, effective_s - chunk_start).                             */
static unsigned oz2_compute_chunk_size(int64_t m, int64_t n, int64_t k, unsigned s)
{
    const size_t mn4    = static_cast<size_t>(m) * static_cast<size_t>(n) * 4u;
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t slc    = lda8i * cola8i + lda8i * static_cast<size_t>(n);

    size_t chunk = s;
    if(mn4 > 0u || slc > 0u)
        chunk = std::min(chunk, OZ2_CHUNK_TARGET_BYTES / (mn4 + slc));
    return static_cast<unsigned>(std::max(size_t(1u), chunk));
}

/* Fused-kernel variant: the fused kernel accumulates INT32 products in GPU
 * registers (no C32i workspace required).  Only A8i + B8i per modulus need
 * to fit in the budget, so the INT32 output term (mn4) is excluded.
 * A larger chunk (up to s) is achievable, potentially fitting all S moduli
 * in a single scale pass and eliminating the need for binary M/N halving.  */
static unsigned oz2_compute_chunk_size_fused(int64_t m, int64_t n, int64_t k, unsigned s)
{
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t slc    = lda8i * cola8i + lda8i * static_cast<size_t>(n);  /* A8i + B8i per modulus */
    size_t chunk = s;
    if(slc > 0u)
        chunk = std::min(chunk, OZ2_CHUNK_TARGET_BYTES / slc);
    return static_cast<unsigned>(std::max(size_t(1u), chunk));
}

/* =========================================================================
 * Host-side tables (source: GEMMul8/GEMMul8/src/table.hpp)
 *
 * Moduli in order: 256 (implicit), 255, 253, 251, 247, 241, 239, 233,
 *                  229, 227, 223, 217, 211, 199, 197, 193, 191, 181
 *                  (OZ2_S_MAX = 18 total)
 *
 * All arrays indexed by table_idx = s - 2  (s = number of moduli, 2..18).
 * ========================================================================= */
/* log2P values for the shift-refinement formula:
 *   sft_delta = floor(-0.5 * log2(amax) + log2P)
 *
 * All entries use GEMMul8 'fast' values: fast::log2P = fld(log2(M-1)/2 - 1.5),
 * exactly 1.0 below the 'accu' values: accu::log2P = fld(log2(M-1)/2 - 0.5).
 *
 * Rationale: for every s, there exists a constructible FP64 input for which
 * the accu::log2P formula pushes a specific element's
 *   X_true = D × 2^{sftA[row]+sftB[col]}
 * above M_s/2 (the OZ2 CRT uniqueness bound), returning the wrong sign.
 * Using fast::log2P for all s guarantees a safety margin of ≥ 2 bits between
 * X_true_max and M_s/2 for any valid FP64 input, without any assumption on the
 * distribution or magnitude of A and B.
 *
 * The preliminary GEMM still provides full per-row/col adaptive scaling via the
 * data-dependent sft_delta = floor(-0.5 × log2(row_max) + log2P); only the
 * global constant log2P is reduced by 1 bit.
 *
 * Values match GEMMul8's table::fast::log2P (RIKEN GEMMul8 reference). */
static const float h_accu_log2P_all[OZ2_S_MAX - 1] = {
    6.49716520e+00F,   /* s=2  — fast */
    1.04886732e+01F,   /* s=3  — fast */
    1.44744443e+01F,   /* s=4  — fast */
    1.84486274e+01F,   /* s=5  — fast */
    2.24050731e+01F,   /* s=6  — fast */
    2.63555068e+01F,   /* s=7  — fast */
    3.02875995e+01F,   /* s=8  — fast */
    3.42071990e+01F,   /* s=9  — fast */
    3.81204757e+01F,   /* s=10 — fast */
    4.20209236e+01F,   /* s=11 — fast */
    4.59016990e+01F,   /* s=12 — fast */
    4.97622489e+01F,   /* s=13 — fast */
    5.35805625e+01F,   /* s=14 — fast */
    5.73915863e+01F,   /* s=15 — fast */
    6.11878166e+01F,   /* s=16 — fast */
    6.49765319e+01F,   /* s=17 — fast */
    6.87264480e+01F,   /* s=18 — fast */
};
/* =========================================================================
 * Host-side emulation control functions
 * ========================================================================= */
static constexpr unsigned FP64_EMULATION_DEFAULT_NUM_MODULI = 16u;
static constexpr unsigned FP64_EMULATION_MAX_MANTISSA_BITS  = 140u;

static bool parse_unsigned_env(const char* value, unsigned max_value, unsigned* out)
{
    if(value == nullptr || *value == '\0' || value[0] == '+' || value[0] == '-')
        return false;

    errno       = 0;
    char* end   = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 0);
    if(errno != 0 || end == value || *end != '\0' || parsed > max_value)
        return false;

    *out = static_cast<unsigned>(parsed);
    return true;
}

Fp64EmulationEnvValue fp64EmulationParseEnabledEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "0") == 0) return {FP64_EMULATION_ENV_VALID, 0u};
    if(std::strcmp(value, "1") == 0) return {FP64_EMULATION_ENV_VALID, 1u};
    return {FP64_EMULATION_ENV_INVALID, 0u};
}

Fp64EmulationEnvValue fp64EmulationParseStrategyEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "performant") == 0)
        return {FP64_EMULATION_ENV_VALID,
                static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT)};
    if(std::strcmp(value, "eager") == 0)
        return {FP64_EMULATION_ENV_VALID,
                static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER)};
    return {FP64_EMULATION_ENV_INVALID, 0u};
}

Fp64EmulationEnvValue fp64EmulationParseSpecialValuesMaskEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0x3u};

    unsigned parsed = 0u;
    if(!parse_unsigned_env(value, UINT_MAX, &parsed))
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, parsed};
}

Fp64EmulationEnvValue fp64EmulationParseMantissaBitCountEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};

    unsigned parsed = 0u;
    if(!parse_unsigned_env(value, FP64_EMULATION_MAX_MANTISSA_BITS, &parsed))
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, parsed};
}

bool fp64EmulationIsValidMantissaBitCount(int value)
{
    return value >= -1 && value <= static_cast<int>(FP64_EMULATION_MAX_MANTISSA_BITS);
}

static const Fp64EmulationEnvValue& cached_enabled_env()
{
    static const Fp64EmulationEnvValue parsed =
        fp64EmulationParseEnabledEnv(std::getenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION"));
    return parsed;
}

static const Fp64EmulationEnvValue& cached_strategy_env()
{
    static const Fp64EmulationEnvValue parsed =
        fp64EmulationParseStrategyEnv(std::getenv("HIPBLASLT_EMULATION_STRATEGY"));
    return parsed;
}

static const Fp64EmulationEnvValue& cached_special_values_mask_env()
{
    static const Fp64EmulationEnvValue parsed = fp64EmulationParseSpecialValuesMaskEnv(
        std::getenv("HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK"));
    return parsed;
}

static const Fp64EmulationEnvValue& cached_mantissa_bit_count_env()
{
    static const Fp64EmulationEnvValue parsed = fp64EmulationParseMantissaBitCountEnv(
        std::getenv("HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"));
    return parsed;
}

bool fp64EmulationIsEnabled()
{
    const auto& env = cached_enabled_env();
    return env.state == FP64_EMULATION_ENV_VALID && env.value != 0u;
}

bool fp64EmulationPerformanceCheck(int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    if(device < 0 || device >= 64) return std::nullopt;
    auto& entry = oz2_device_params_cache[device];
    if(!entry) {
        int chip_id = 0;
        const hipError_t attr_err =
            hipDeviceGetAttribute(&chip_id,
                                  hipDeviceAttributePciChipId, device);
        const uint32_t pci_device_id = static_cast<uint32_t>(chip_id) & 0xFFFFu;
        auto it = oz2_hw_params_by_pci_id.find(pci_device_id);
        entry = (it != oz2_hw_params_by_pci_id.end())
              ? std::optional<Oz2PerfModelParams>{it->second}
              : std::optional<Oz2PerfModelParams>{};
    }
    return *entry;
}

/* =========================================================================
 * Performance-model predicted times
 * Returns all sub-times in milliseconds.  Used both for the profiling CSV
 * and (via comparison of t_total_ms vs t_native_ms) for the performance
 * heuristic in fp64EmulationPerformanceCheck.
 * ========================================================================= */
struct Fp64PerfModelTimes {
    double t_prelim_ms;      /* prelim kernel (shift + extraction)  */
    double t_prelim_gemm_ms; /* preliminary INT8 GEMM               */
    double t_refine_ms;      /* sft-refinement kernels              */
    double t_adp_ms;         /* ADP reduce kernels + hipStreamSynchronize (dynamic mode only) */
    double t_scale_ms;       /* multi-modulus scaling kernels       */
    double t_int8_gemms_ms;  /* all INT8 GEMMs                      */
    double t_accum_ms;       /* CRT accumulation / finalize kernels */
    double t_host_ms;        /* per-call host overhead              */
    double t_launch_ms;      /* kernel-launch overhead              */
    double t_fused_ms;       /* fused TN kernel (replaces scale+GEMM+accum when better) */
    double t_total_ms;       /* total predicted emulation time      */
    double t_native_ms;      /* predicted native FP64 DGEMM time    */
};

static Fp64PerfModelTimes fp64EmulationPerfModelTimes(bool tA, bool tB,
                                                      int64_t m, int64_t n, int64_t k,
                                                      unsigned num_moduli, int device,
                                                      bool dynamic_mode)
{
    const auto hw_opt = oz2_get_perf_model_params(device);
    assert(hw_opt.has_value() &&
           "fp64EmulationPerfModelTimes called for a device not in oz2_hw_params_by_pci_id");
    const Oz2PerfModelParams& hw = *hw_opt;

    static constexpr double LATENCY_KERNEL = 5.0e-6;
    static constexpr double LATENCY_MATMUL = 10.0e-6;
    static constexpr double LATENCY_MEMSET = 2.0e-6;
    /* CPU-GPU roundtrip for hipStreamSynchronize: the host parks until the GPU
     * drains and signals, then re-queues the remaining kernels.  This is a
     * true OS/driver latency, distinct from the GPU-side LATENCY_KERNEL or
     * LATENCY_MATMUL scheduling overheads.                                  */
    static constexpr double LATENCY_SYNC   = 50.0e-6;

    const double c0 = hw.latency / LATENCY_MATMUL;
    const double c1 = c0 * hw.ai;
    const double c2 = c1 * hw.ratio;
    const double s   = static_cast<double>(num_moduli);
    const double mn  = static_cast<double>(m) * static_cast<double>(n);
    const double mk  = static_cast<double>(m) * static_cast<double>(k);
    const double kn  = static_cast<double>(k) * static_cast<double>(n);
    const double mnk = mn * static_cast<double>(k);

    const double chunk_sz       = static_cast<double>(oz2_compute_chunk_size(m, n, k, num_moduli));
    const double n_chunks       = std::ceil(s / chunk_sz);
    const double n_scale_chunks = n_chunks;  /* scale and GEMM share the same chunk */

    const double EFF_PRELIM_KERN = tA ? (tB ? 0.628 : 0.797) : (tB ? 0.550 : 0.622);
    const double EFF_SCALE_KERN  = tA ? (tB ? 0.401 : 0.461) : (tB ? 0.354 : 0.405);
    static constexpr double EFF_REFINE_KERN      = 0.511;
    static constexpr double EFF_ACCUM_KERN       = 0.775;
    static constexpr double OZ2_HOST_OVERHEAD_MS = 0.100;

    const double t_int8_bw     = (mk + kn + 4.0 * mn) / c0;
    const double t_prelim_kern = ((mk + kn) * 17.0 / c0 + 2.0 * LATENCY_KERNEL) / EFF_PRELIM_KERN;
    const double t_prelim_gemm = std::max(2.0 * mnk / c2, t_int8_bw) + LATENCY_MATMUL;
    const double t_refine_kern = (mn * 8.0 / c0 + 3.0 * LATENCY_KERNEL + LATENCY_MEMSET) / EFF_REFINE_KERN;
    const double t_scale_kern  = ((mk + kn) * (8.0 * n_scale_chunks + s) / c0 + 2.0 * n_scale_chunks * LATENCY_KERNEL) / EFF_SCALE_KERN;
    const double t_int8_gemms  = s * std::max(2.0 * mnk / c2, t_int8_bw) + n_chunks * LATENCY_MATMUL;
    const double t_accum_kern  = (mn * (4.0 * s + 32.0 * n_chunks - 16.0) / c0 + n_chunks * LATENCY_KERNEL) / EFF_ACCUM_KERN;
    const double t_host        = OZ2_HOST_OVERHEAD_MS * 1e-3;
    const double t_launch      = 0.0;   /* all launch overhead distributed into components above */

    /* Fused TN kernel: reads pre-computed INT8 A8i/B8i from workspace, performs
     * MFMA + CRT accumulation, writes FP64 D directly.  Scale runs separately.  */
    static constexpr double EFF_FUSED = 0.57;
    const double t_fused_bw   = (s * (mk + kn) + 16.0 * mn) / c0;  /* INT8 + FP64 C/D */
    const double t_fused_int8 = s * 2.0 * mnk / c2;                  /* MFMA              */
    const double t_fused_fp64 = s * 8.0 * mn / c1;                   /* CRT accum only    */
    const double t_fused_cmp  = t_fused_int8 + t_fused_fp64;
    const double t_fused = std::max(t_fused_bw, t_fused_cmp) / EFF_FUSED + LATENCY_KERNEL;

    /* Scale always runs.  Fused kernel replaces only GEMM + CRT accum.
     * Only consider the fused time when the fused kernel is not disabled:
     * HIPBLASLT_EMULATION_FUSED=off means the non-fused path is always used,
     * so the gate and split decisions must not assume the fused speedup.    */
    const double t_gemm_accum  = (oz2_fused_mode() != Oz2FusedMode::OFF)
                               ? std::min(t_int8_gemms + t_accum_kern, t_fused)
                               : t_int8_gemms + t_accum_kern;
    /* ADP (dynamic-mode) overhead: two small reduction kernels (oz2_adp_reduce_A
     * reads m ints; oz2_adp_reduce_B reads the full m×n C32i matrix), followed by
     * a hipStreamSynchronize that blocks the CPU until the GPU drains so the host
     * can inspect the results and choose effective_s.
     * LATENCY_SYNC is the CPU-GPU roundtrip latency (distinct from the GPU-side
     * LATENCY_KERNEL or LATENCY_MATMUL scheduling overheads).                  */
    const double t_adp = dynamic_mode
        ? LATENCY_KERNEL                            /* oz2_adp_reduce_A  */
        + std::max(4.0 * mn / c0, LATENCY_KERNEL)  /* oz2_adp_reduce_B  */
        + LATENCY_SYNC                              /* hipStreamSynchronize */
        : 0.0;
    const double t_total       = t_prelim_kern + t_prelim_gemm + t_refine_kern
                               + t_scale_kern  + t_gemm_accum  + t_host + t_adp;
    const double t_native      = std::max(2.0 * mnk / c1, 8.0 * (mk + kn + mn) / c0) + LATENCY_MATMUL;

    constexpr double s2ms = 1000.0;
    return { t_prelim_kern * s2ms, t_prelim_gemm * s2ms, t_refine_kern * s2ms,
             t_adp         * s2ms,
             t_scale_kern  * s2ms, t_int8_gemms  * s2ms, t_accum_kern  * s2ms,
             t_host        * s2ms, t_launch      * s2ms,
             t_fused       * s2ms,
             t_total       * s2ms, t_native      * s2ms };
}

/* Returns the minimum achievable emulation time in ms, accounting for the
 * recursive binary-halving that fp64EmulatedGemm applies when n_chunks > 1.
 * Both halves execute sequentially so the effective time is additive.
 *
 * The gate comparison uses the RECURSIVE effective times of each half
 * (not the flat perf-model times).  This correctly handles the case where
 * one split does not yet reduce n_chunks but further splitting would: the
 * recursive sub-call for the half discovers and accounts for those deeper
 * splits, returning the true best achievable time for that half.       */
static double oz2_effective_time_ms(bool tA, bool tB,
                                    int64_t m, int64_t n, int64_t k, unsigned s, int device,
                                    bool dynamic_mode)
{
    const double t_mono = fp64EmulationPerfModelTimes(tA, tB, m, n, k, s, device, dynamic_mode).t_total_ms;

    const unsigned chunk_sz = oz2_compute_chunk_size(m, n, k, s);
    const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
    if(n_chunks > 1u) {
        const bool    split_m = (m >= n);
        const int64_t half_m  = split_m ? m / 2 : m;
        const int64_t half_n  = split_m ? n     : n / 2;

        /* Both halves are nearly identical in size (differ by at most 1 when
         * m or n is odd), so approximate t_split = 2 × t_half.             */
        const double t_split = 2. * oz2_effective_time_ms(tA, tB, half_m, half_n, k, s, device, dynamic_mode);
        if(t_split < t_mono)
            return t_split;
    }
    return t_mono;
}

/* Returns per-component predicted times summed across ALL leaf sub-GEMMs,
 * mirroring the recursive binary-halving of oz2_effective_time_ms.
 * t_native_ms is always set to the top-level (m,n,k) native DGEMM time
 * because native DGEMM does not split.                                    */
static Fp64PerfModelTimes oz2_effective_perf_model_times(bool tA, bool tB,
                                                         int64_t m, int64_t n, int64_t k,
                                                         unsigned s, int device,
                                                         bool dynamic_mode)
{
    Fp64PerfModelTimes mono = fp64EmulationPerfModelTimes(tA, tB, m, n, k, s, device, dynamic_mode);

    const unsigned chunk_sz = oz2_compute_chunk_size(m, n, k, s);
    const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
    if(n_chunks > 1u) {
        const bool    split_m = (m >= n);
        const int64_t half_m  = split_m ? m / 2 : m;
        const int64_t half_n  = split_m ? n     : n / 2;

        const double t_split = 2. * oz2_effective_time_ms(tA, tB, half_m, half_n, k, s, device, dynamic_mode);
        if(t_split < mono.t_total_ms) {
            /* Recurse on one half, then double all components.
             * Both halves are ≈ equal in size so the approximation is exact
             * when m (or n) is even and negligible otherwise.               */
            Fp64PerfModelTimes half = oz2_effective_perf_model_times(tA, tB, half_m, half_n, k, s, device, dynamic_mode);
            half.t_prelim_ms      *= 2.0;
            half.t_prelim_gemm_ms *= 2.0;
            half.t_refine_ms      *= 2.0;
            half.t_adp_ms         *= 2.0;
            half.t_scale_ms       *= 2.0;
            half.t_int8_gemms_ms  *= 2.0;
            half.t_accum_ms       *= 2.0;
            half.t_host_ms        *= 2.0;
            half.t_launch_ms      *= 2.0;
            half.t_fused_ms       *= 2.0;
            half.t_total_ms       *= 2.0;
            /* Native DGEMM does not split: keep the top-level prediction. */
            half.t_native_ms = mono.t_native_ms;
            return half;
        }
    }
    return mono;
}

bool fp64EmulationPerformanceCheck(const _rocblaslt_handle* h,
                                   hipblasOperation_t opA, hipblasOperation_t opB,
                                   int64_t m, int64_t n, int64_t k)
{
    const int  device      = h->device;
    const bool tA          = (opA != HIPBLAS_OP_N);
    const bool tB          = (opB != HIPBLAS_OP_N);
    /* Include ADP overhead when the handle is configured for dynamic (ADP) mode,
     * so the performance gate correctly accounts for the hipStreamSynchronize cost. */
    const bool     dyn        = (h->emulation.mantissa_control != 1);
    const unsigned num_moduli = fp64EmulationEffectiveNumModuli(h);
    const double t_emul   = oz2_effective_time_ms(tA, tB, m, n, k, num_moduli, device, dyn);
    const double t_native = fp64EmulationPerfModelTimes(tA, tB, m, n, k, num_moduli, device, dyn).t_native_ms;
    return t_emul <= t_native;
}

bool fp64EmulationIsEager()
{
    const auto& env = cached_strategy_env();
    return env.state == FP64_EMULATION_ENV_VALID
           && env.value == static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER);
}

uint32_t fp64EmulationSpecialValuesMask()
{
    const auto& env = cached_special_values_mask_env();
    return env.state == FP64_EMULATION_ENV_VALID ? env.value : 0x3u;
}

static constexpr double oz2_cum_bits[OZ2_S_MAX - 1] = {
     15.994,  /* s=2  */  23.976,  /* s=3  */  31.945,  /* s=4  */
     39.894,  /* s=5  */  47.807,  /* s=6  */  55.708,  /* s=7  */
     63.572,  /* s=8  */  71.411,  /* s=9  */  79.238,  /* s=10 */
     87.040,  /* s=11 */  94.801,  /* s=12 */ 102.522,  /* s=13 */
    110.160,  /* s=14 */ 117.782,  /* s=15 */ 125.374,  /* s=16 */
    132.949,  /* s=17 */ 140.448,  /* s=18 */
};

static unsigned num_moduli_for_mantissa_bits(unsigned target)
{
    if(type_a != HIP_R_64F || batch_count != 1) return false;
    const bool emulEnabled = (h->emulation.enabled == 1)
                           || (h->emulation.enabled != 0 && fp64EmulationIsEnabled());
    if(!emulEnabled) return false;
    /* Disable emulation on devices not listed in the perf-model table to
     * avoid running with unvalidated performance predictions.              */
    const int dev = h->device;
    if(!oz2_get_perf_model_params(dev)) return false;
    const bool eager = (h->emulation.strategy == 2)
                     || (h->emulation.strategy != 1 && fp64EmulationIsEager());
    return eager || fp64EmulationPerformanceCheck(h, opA, opB, m, n, k);
}

static rocblaslt_status invalid_if_set(const Fp64EmulationEnvValue& env)
{
    return env.state == FP64_EMULATION_ENV_INVALID ? rocblaslt_status_invalid_value
                                                   : rocblaslt_status_success;
}

struct Fp64EmulationMantissaPolicy {
    rocblaslt_status status;
    unsigned int     num_moduli;
    bool             dynamic_mode;
};

static Fp64EmulationMantissaPolicy resolve_mantissa_policy(const _rocblaslt_handle* h)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0x3u};
    char*         endp = nullptr;
    const long    v    = std::strtol(value, &endp, 0);
    if(endp == value || *endp != '\0' || v < 0)
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, static_cast<unsigned>(v)};
}

Fp64EmulationEnvValue fp64EmulationParseMantissaBitCountEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};
    char*      endp = nullptr;
    const long v    = std::strtol(value, &endp, 10);
    if(endp == value || *endp != '\0' || v < 0 || v > 140)
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, static_cast<unsigned>(v)};
}

bool fp64EmulationIsValidMantissaBitCount(int value)
{
    return value >= -1 && value <= 140;
}

/* =========================================================================
 * Status-returning emulation gate (replaces the bool fp64EmulationWouldApply
 * for callers that need error propagation on bad env-var values).
 * ========================================================================= */

Fp64EmulationDecision fp64EmulationDecision(const _rocblaslt_handle* h,
                                            hipDataType              type_a,
                                            int64_t                  m,
                                            int64_t                  n,
                                            int64_t                  k,
                                            int                      batch_count)
{
    Fp64EmulationDecision result{rocblaslt_status_success, false, 0u, 0x3u, false};
    if(type_a != HIP_R_64F || batch_count != 1) return result;

    const auto& enabled_env = cached_enabled_env();
    bool emul_enabled = false;
    if(h->emulation.enabled == 0) {
        return result;
    } else if(h->emulation.enabled == 1) {
        emul_enabled = true;
    } else {
        result.status = invalid_if_set(enabled_env);
        if(result.status != rocblaslt_status_success) return result;
        emul_enabled = enabled_env.state == FP64_EMULATION_ENV_VALID && enabled_env.value;
    }
    if(!emul_enabled) return result;

    const auto& strategy_env = cached_strategy_env();
    result.status = invalid_if_set(strategy_env);
    if(result.status != rocblaslt_status_success) return result;

    const unsigned strategy = (strategy_env.state == FP64_EMULATION_ENV_VALID)
                                  ? strategy_env.value
                                  : static_cast<unsigned>(
                                        (h->emulation.strategy >= 0)
                                            ? h->emulation.strategy
                                            : HIPBLASLT_EMULATION_STRATEGY_PERFORMANT);

    const auto& mask_env = cached_special_values_mask_env();
    result.status = invalid_if_set(mask_env);
    if(result.status != rocblaslt_status_success) return result;
    result.sv_mask = (mask_env.state == FP64_EMULATION_ENV_VALID)
                         ? mask_env.value
                         : ((h->emulation.special_values_mask != ~0u)
                                ? h->emulation.special_values_mask
                                : 0x3u);

    /* Handle special_values_mask override. */
    if(h->emulation.special_values_mask != ~0u)
        result.sv_mask = h->emulation.special_values_mask;

    /* dynamic_mode: DYNAMIC (ADP) mantissa control — adaptively selects
     * the minimum s needed for FP64 precision on the given input data.   */
    result.dynamic_mode = (h->emulation.mantissa_control != 1);

    /* Strategy (eager vs performant). */
    const bool eager = (h->emulation.strategy == 2)
                     || (h->emulation.strategy != 1 && fp64EmulationIsEager());
    if(eager || fp64EmulationPerformanceCheck(h, opA, opB, m, n, k))
        result.apply = true;

    result.num_moduli   = mantissa.num_moduli;
    result.dynamic_mode = mantissa.dynamic_mode;
    result.apply = (strategy == static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER))
                   || fp64EmulationPerformanceCheck(m, n, k, result.num_moduli);
    return result;
}

void fp64EmulationWarnDynamicTemporary()
{
    static std::atomic<bool> warned{false};
    bool expected = false;
    if(warned.compare_exchange_strong(expected, true)) {
        std::fprintf(stderr,
                     "hipBLASLt FP64 emulation: dynamic mantissa control currently "
                     "uses the existing 16-moduli path; runtime ADP is not enabled yet.\n");
    }
}

/* =========================================================================
 * fp64EmulationWorkspaceSize
 * ========================================================================= */
size_t fp64EmulationWorkspaceSize(const _rocblaslt_handle*     h,
                                  hipblasOperation_t           opA,
                                  hipblasOperation_t           opB,
                                  int64_t m, int64_t n, int64_t k,
                                  const Fp64EmulationDecision& decision)
{
    assert(h != nullptr && "fp64EmulationWorkspaceSize requires a valid handle");
    /* In ADP mode the workspace must cover OZ2_S_MAX=18 moduli so that any
     * adaptive effective_s fits without reallocation.  In fixed mode use
     * the resolved moduli count directly.                                   */
    const unsigned num_moduli = decision.dynamic_mode ? OZ2_S_MAX : decision.num_moduli;
    const int  device = h->device;
    const bool tA     = (opA != HIPBLAS_OP_N);
    const bool tB     = (opB != HIPBLAS_OP_N);

    /* Mirror the splitting decision in fp64EmulatedGemmImpl exactly.
     *
     * fp64EmulatedGemmImpl uses settings.num_moduli (= fp64EmulationEffectiveNumModuli(h))
     * for the chunk-size / n_chunks check that decides whether to split, regardless of
     * whether dynamic mode is active (where num_moduli parameter = OZ2_S_MAX but
     * settings.num_moduli is the user-configured max, e.g. 16).
     *
     * Using num_moduli (= ws_moduli = OZ2_S_MAX in dynamic mode) here instead would
     * cause the workspace function to decide to split for problem sizes where the
     * implementation goes to the monolithic path.  The monolithic path needs the full
     * (m, n) workspace, which is larger than max(WS(half), WS(half)), resulting in
     * the workspace being under-allocated and a buffer overflow at runtime.           */
    const unsigned split_num_moduli = fp64EmulationEffectiveNumModuli(h);
    const unsigned chunk_sz = oz2_compute_chunk_size(m, n, k, split_num_moduli);
    const unsigned n_chunks = (split_num_moduli + chunk_sz - 1u) / chunk_sz;

    if(n_chunks > 1u) {
        const bool    split_m = (m >= n);
        const int64_t half_m  = split_m ? m / 2 : m;
        const int64_t half_n  = split_m ? n     : n / 2;
        const int64_t m2      = split_m ? (m - m / 2) : m;
        const int64_t n2      = split_m ? n            : (n - n / 2);

        const double t_mono  = fp64EmulationPerfModelTimes(tA, tB, m, n, k, split_num_moduli, device, decision.dynamic_mode).t_total_ms;
        const double t_split = 2. * oz2_effective_time_ms(tA, tB, half_m, half_n, k, split_num_moduli, device, decision.dynamic_mode);

        if(t_split < t_mono) {
            /* num_moduli (= ws_moduli) is forwarded to the recursive calls so that
             * the monolithic workspace at each leaf is sized for the correct layout
             * (e.g. OZ2_S_MAX in dynamic mode, matching layout_moduli in the impl). */
            return std::max(fp64EmulationWorkspaceSize(h, opA, opB, half_m, half_n, k, decision),
                            fp64EmulationWorkspaceSize(h, opA, opB, m2,     n2,     k, decision));
        }
    }

    /* Monolithic path workspace. */
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);
    const unsigned chunk_ws = oz2_compute_chunk_size(m, n, k, num_moduli);

    /* Zhi/Zlo accumulators are only needed when there are multiple passes
     * (chunk_ws < num_moduli).  Single-pass finalize writes D directly.  */
    const size_t szZhi_ws = (chunk_ws < num_moduli) ? szC32i : 0u;

    return   chunk_ws * lda8i * cola8i * sizeof(int8_t)
           + chunk_ws * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)
           + chunk_ws * szC32i * sizeof(int32_t)
           + szZhi_ws * sizeof(double) * 2
           + cola8i * sizeof(int16_t)
           + padn   * sizeof(int16_t)
           + sizeof(uint32_t)
           + cola8i * sizeof(int32_t)
           + 2 * sizeof(float)      /* ADP float buffer: adp_buf[0..1] (bias ±200) */
           + OZ2_INT8_GEMM_WS_BYTES; /* INT8 GEMM workspace (preliminary + batch)  */
}

unsigned fp64EmulationNumModuli()
{
    static const unsigned num_moduli = []() -> unsigned {
        const auto& env = cached_mantissa_bit_count_env();
        if(env.state != FP64_EMULATION_ENV_VALID) return FP64_EMULATION_DEFAULT_NUM_MODULI;
        const unsigned target = env.value;
        for(unsigned s = 2u; s <= OZ2_S_MAX; ++s)
            if(oz2_cum_bits[s - 2u] >= static_cast<double>(target)) return s;
        return OZ2_S_MAX;
    }();
    return num_moduli;
}

/* =========================================================================
 * Device helper: warp-level reductions
 * ========================================================================= */
static __device__ __forceinline__ double warp_reduce_max_abs_d(double val)
{
    val = fabs(val);
    unsigned long long bits; __builtin_memcpy(&bits, &val, 8);
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        unsigned long long other = __shfl_down(bits, off);
        if(other > bits) bits = other;
    }
    double res; __builtin_memcpy(&res, &bits, 8);
    return res;
}

static __device__ __forceinline__ int32_t warp_reduce_max_abs_i32(int32_t val)
{
    if(val < 0) val = -val;
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        int32_t other = __shfl_down(val, off);
        if(other > val) val = other;
    }
    return val;
}

static __device__ __forceinline__ double
block_reduce_max_d(double warp_max, double* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    double result = 0.0;
    if(threadIdx.x == 0) {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < nw; ++w) if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

static __device__ __forceinline__ int32_t
block_reduce_max_i32(int32_t warp_max, int32_t* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    int32_t result = 0;
    if(threadIdx.x == 0) {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < nw; ++w) if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

/* =========================================================================
 * GPU kernel — fused preliminary shift computation + INT8 extraction
 *
 * oz2_accu_prelim_kernel<TRANS_A, TRANS_B, CHECK_NAN>
 *
 * Fuses per-row/col shift computation (sftA, sftB) and preliminary INT8
 * extraction (A8i_high, B8i_high) into a SINGLE kernel dispatch, eliminating
 * the extra launch overhead of the previous two-kernel approach and achieving
 * coalesced HBM reads AND writes for all four (TRANS_A, TRANS_B) combinations.
 *
 *   TRANS_A=true  (op(A)=A^T, A stored k×m col-major): k-fast (threadIdx.x=j).
 *     Loop 1: read A[row*lda+j] (COALESCED) → block-reduce → sft.
 *     Loop 2: read A[row*lda+j] (COALESCED) → scale → write A8i_high[j+row*lda8i] (COALESCED).
 *
 *   TRANS_A=false (op(A)=A, A stored m×k col-major): tiled SHMEM.
 *     One block per TILE-row tile; two passes over k:
 *       Pass 1: load m-fast tiles into SHMEM (COALESCED), accumulate per-row max → sft.
 *       Pass 2: load m-fast tiles into SHMEM (COALESCED), scale, write A8i_high k-fast
 *               from SHMEM via SHMEM transposition (COALESCED).
 *
 *   TRANS_B=false / TRANS_B=true: symmetric to TRANS_A=true / TRANS_A=false.
 *
 * Grid  = dim3(m_blks + n_blks, 1)
 *   m_blks = TRANS_A ? m       : ceil(m / OZ2_PRELIM_TILE)
 *   n_blks = TRANS_B ? ceil(n / OZ2_PRELIM_TILE) : n
 * Block = dim3(OZ2_PRELIM_TILE * OZ2_PRELIM_TILE, 1) = dim3(256, 1)
 * ========================================================================= */
static constexpr int OZ2_PRELIM_TILE_K = 64;  /* k-tile size (reduces k-tile loop count) */
static constexpr int OZ2_PRELIM_TILE_M = 4;   /* rows/cols per tile (= blockDim.x / TILE_K) */
/* blockDim.x = TILE_K × TILE_M = 256 threads */

/* Returns floor(log2(x)) for a positive normalized FP64 value x by extracting
 * the IEEE 754 biased exponent field (bits 52-62) via integer bit ops.
 * Replaces the quarter-rate transcendental log2() + floor() sequence (~50-100
 * cycles on CDNA) with 2-3 full-rate integer instructions (~4-5 cycles).
 * Precondition: x > 0 and x is a normalized FP64 (guaranteed by the
 * < DBL_MIN guard that replaces zero/subnormal row/col maxima with DBL_MIN). */
static __device__ __forceinline__ int oz2_floor_log2_d(double x)
{
    unsigned long long bits;
    __builtin_memcpy(&bits, &x, 8);
    return static_cast<int>((bits >> 52) & 0x7FFull) - 1023;
}

/* ── A_T: TRANS_A=true, k-fast coalesced, blockDim=256, one block per row ── */
template <bool CHECK_NAN>
__global__ static void
oz2_accu_prelim_kernel(const double* __restrict__ A,
                        int64_t m, int64_t k, int64_t lda,
                        int8_t*  __restrict__  A8i_high, size_t lda8i,
                        int16_t* __restrict__  sftA,
                        const double* __restrict__ B,
                        int64_t n, int64_t ldb,
                        int8_t*  __restrict__  B8i_high, size_t ldb8i,
                        int16_t* __restrict__  sftB,
                        uint32_t* __restrict__ nan_flag,
                        unsigned m_blks)
{
    static constexpr int TILE_K = OZ2_PRELIM_TILE_K;  /* k-tile size (256/TILE_M iterations) */
    static constexpr int TILE_M = OZ2_PRELIM_TILE_M;  /* rows/cols per block */

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    double local_max = 0.0;
    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val = A[row * lda + j];                             /* COALESCED */
        if constexpr (CHECK_NAN)
            if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
        double av = fabs(val);
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_d(local_max);
    local_max = block_reduce_max_d(local_max, s_wmax);
    if(threadIdx.x == 0) {
        if(local_max < std::numeric_limits<double>::min()) local_max = std::numeric_limits<double>::min();
        s_sft[0] = static_cast<int16_t>(6 - oz2_floor_log2_d(local_max));
        sftA[row] = s_sft[0];
    }
    __syncthreads();
    const int sft = static_cast<int>(s_sft[0]);
    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val    = A[row * lda + j];                          /* COALESCED */
        double scaled = ceil(ldexp(fabs(val), sft));
        A8i_high[static_cast<size_t>(j) + static_cast<size_t>(row) * lda8i] =
            static_cast<int8_t>(static_cast<int32_t>(scaled));    /* COALESCED */
    }
}

/* ── A_N: TRANS_A=false, SHMEM transposition, blockDim=1024, TILE_M=16 rows/block ── */
template <bool CHECK_NAN>
__global__ static void
oz2_accu_prelim_A_N_kernel(const double* __restrict__ A,
                             int64_t m, int64_t k, int64_t lda,
                             int8_t*  __restrict__ A8i_high, size_t lda8i,
                             int16_t* __restrict__ sftA,
                             uint32_t* __restrict__ nan_flag)
{
    static constexpr int TILE_K = OZ2_PRELIM_TILE_K;
    static constexpr int TILE_M = OZ2_PRELIM_SHMEM_TILE_M;
    __shared__ double  shmem[TILE_K][TILE_M + 1];  /* +1 avoids bank conflicts */
    __shared__ int16_t s_sft[TILE_M];

    const int64_t m_base = static_cast<int64_t>(blockIdx.x) * TILE_M;
    const int t       = static_cast<int>(threadIdx.x);
    const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
    const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
    const int64_t i   = m_base + m_local;

    /* Pass 1: each thread accumulates its partial per-row max over all k-tiles */
    double thr_max = 0.0;
    for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
        const int64_t j = k_base + k_local;
        if(i < m && j < k) {
            double val = A[i + j * lda];                           /* COALESCED */
            if constexpr (CHECK_NAN)
                if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
            double av = fabs(val);
            if(av > thr_max) thr_max = av;
        }
    }
    shmem[k_local][m_local] = thr_max;
    __syncthreads();
    if(k_local == 0) {
        double row_max = 0.0;
        for(int kl = 0; kl < TILE_K; ++kl)
            if(shmem[kl][m_local] > row_max) row_max = shmem[kl][m_local];
        if(row_max < std::numeric_limits<double>::min()) row_max = std::numeric_limits<double>::min();
        s_sft[m_local] = static_cast<int16_t>(6 - oz2_floor_log2_d(row_max));
        if(i < m) sftA[i] = s_sft[m_local];
    }
    __syncthreads();
    const int sft = static_cast<int>(s_sft[m_local]);

    /* Pass 2: coalesced loads (m-fast) → SHMEM → coalesced writes (k-fast) */
    for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
        const int64_t j = k_base + k_local;
        double scaled = 0.0;
        if(i < m && j < k)
            scaled = ceil(ldexp(fabs(A[i + j * lda]), sft));      /* COALESCED */
        shmem[k_local][m_local] = scaled;
        __syncthreads();
        const int k_write = t % TILE_K;
        const int m_write = t / TILE_K;
        const int64_t j_out = k_base + k_write;
        const int64_t i_out = m_base + m_write;
        if(i_out < m && j_out < k)
            A8i_high[static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i] =
                static_cast<int8_t>(static_cast<int32_t>(shmem[k_write][m_write]));
        __syncthreads();
    }
}

/* ── B_N: TRANS_B=false, j-fast coalesced, blockDim=256, one block per col ── */
template <bool CHECK_NAN>
__global__ static void
oz2_accu_prelim_B_N_kernel(const double* __restrict__ B,
                             int64_t n, int64_t k, int64_t ldb,
                             int8_t*  __restrict__ B8i_high, size_t ldb8i,
                             int16_t* __restrict__ sftB,
                             uint32_t* __restrict__ nan_flag)
{
    __shared__ double  s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE]; /* 8 slots */
    __shared__ int16_t s_sft[1];

    const int64_t col = static_cast<int64_t>(blockIdx.x);
    double local_max = 0.0;
    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val = B[j + col * ldb];                             /* COALESCED */
        if constexpr (CHECK_NAN)
            if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
        double av = fabs(val);
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_d(local_max);
    local_max = block_reduce_max_d(local_max, s_wmax);
    if(threadIdx.x == 0) {
        if(local_max < std::numeric_limits<double>::min()) local_max = std::numeric_limits<double>::min();
        s_sft[0] = static_cast<int16_t>(6 - oz2_floor_log2_d(local_max));
        sftB[col] = s_sft[0];
    }
    __syncthreads();
    const int sft = static_cast<int>(s_sft[0]);
    for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
        double val    = B[j + col * ldb];                          /* COALESCED */
        double scaled = ceil(ldexp(fabs(val), sft));
        B8i_high[static_cast<size_t>(j) + static_cast<size_t>(col) * ldb8i] =
            static_cast<int8_t>(static_cast<int32_t>(scaled));    /* COALESCED */
    }
}

/* ── B_T: TRANS_B=true, SHMEM transposition, blockDim=1024, TILE_M=16 cols/block ── */
template <bool CHECK_NAN>
__global__ static void
oz2_accu_prelim_B_T_kernel(const double* __restrict__ B,
                             int64_t n, int64_t k, int64_t ldb,
                             int8_t*  __restrict__ B8i_high, size_t ldb8i,
                             int16_t* __restrict__ sftB,
                             uint32_t* __restrict__ nan_flag)
{
    static constexpr int TILE_K = OZ2_PRELIM_TILE_K;
    static constexpr int TILE_M = OZ2_PRELIM_SHMEM_TILE_M;
    __shared__ double  shmem[TILE_K][TILE_M + 1];
    __shared__ int16_t s_sft[TILE_M];
    __shared__ double  s_wmax[4];

    if(blockIdx.x < m_blks) {
        /* ── A block ────────────────────────────────────────────────────── */
        if constexpr (TRANS_A) {
            /* k-fast: one block per op(A) row; two k-loops.               */
            const int64_t row = static_cast<int64_t>(blockIdx.x);

            /* Loop 1: compute per-row max */
            double local_max = 0.0;
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val = A[row * lda + j];                     /* COALESCED */
                if constexpr (CHECK_NAN)
                    if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > local_max) local_max = av;
            }
            local_max = warp_reduce_max_abs_d(local_max);
            local_max = block_reduce_max_d(local_max, s_wmax);
            if(threadIdx.x == 0) {
                if(local_max < 1e-300) local_max = 1.0;
                s_sft[0] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(local_max))));
                sftA[row] = s_sft[0];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[0]);

            /* Loop 2: scale and write A8i_high */
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val    = A[row * lda + j];                  /* COALESCED */
                double scaled = ceil(ldexp(fabs(val), sft));
                A8i_high[static_cast<size_t>(j)
                         + static_cast<size_t>(row) * lda8i] =
                    static_cast<int8_t>(static_cast<int32_t>(scaled));  /* COALESCED */
            }

        } else {
            /* Tiled SHMEM: one block per TILE_M-row tile.
             * Thread t: k_local = t/TILE_M (k-index in tile, 0..TILE_K-1),
             *           m_local = t%TILE_M (row-index in tile, 0..TILE_M-1).
             * Adjacent threads (same k_local, consecutive m_local) access
             * consecutive rows of A → COALESCED loads.  k-tile iterations =
             * k/TILE_K (e.g. 16 for k=1024, TILE_K=64).                    */
            const int64_t m_base = static_cast<int64_t>(blockIdx.x) * TILE_M;
            const int t       = static_cast<int>(threadIdx.x);
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
            const int64_t i   = m_base + m_local;

            /* Pass 1: accumulate per-row max across all k-tiles */
            double thr_max = 0.0;
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;
                if(i < m && j < k) {
                    /* A[i + j*lda]: adjacent i (= m_local, varies) → COALESCED */
                    double val = A[i + j * lda];
                    if constexpr (CHECK_NAN)
                        if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                    double av = fabs(val);
                    if(av > thr_max) thr_max = av;
                }
            }
            /* Reduce across k_local dimension (TILE_K=64 values per row) */
            shmem[k_local][m_local] = thr_max;
            __syncthreads();
            if(k_local == 0) {   /* TILE_M=4 threads finalise, one per row */
                double row_max = 0.0;
                for(int kl = 0; kl < TILE_K; ++kl)
                    if(shmem[kl][m_local] > row_max) row_max = shmem[kl][m_local];
                if(row_max < 1e-300) row_max = 1.0;
                s_sft[m_local] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(row_max))));
                if(i < m) sftA[i] = s_sft[m_local];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[m_local]);  /* row-specific */

            /* Pass 2: coalesced loads (m-fast) → SHMEM → coalesced writes (k-fast)
             * k_tile iterations = k/TILE_K (32 syncs for k=2048 vs 256 with TILE=16) */
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;

                /* Load: A[i + j*lda], i=m_base+t%TILE_M varies → COALESCED */
                double scaled = 0.0;
                if(i < m && j < k)
                    scaled = ceil(ldexp(fabs(A[i + j * lda]), sft));
                shmem[k_local][m_local] = scaled;
                __syncthreads();

                /* Write: k_write = t%TILE_K varies fast → COALESCED */
                const int k_write = t % TILE_K;
                const int m_write = t / TILE_K;
                const int64_t j_out = k_base  + k_write;
                const int64_t i_out = m_base  + m_write;
                if(i_out < m && j_out < k)
                    A8i_high[static_cast<size_t>(j_out)
                             + static_cast<size_t>(i_out) * lda8i] =
                        static_cast<int8_t>(static_cast<int32_t>(
                            shmem[k_write][m_write]));
                __syncthreads();
            }
        }
    }
    shmem[k_local][l_local] = thr_max;
    __syncthreads();
    if(k_local == 0) {
        double col_max = 0.0;
        for(int kl = 0; kl < TILE_K; ++kl)
            if(shmem[kl][l_local] > col_max) col_max = shmem[kl][l_local];
        if(col_max < std::numeric_limits<double>::min()) col_max = std::numeric_limits<double>::min();
        s_sft[l_local] = static_cast<int16_t>(6 - oz2_floor_log2_d(col_max));
        if(col < n) sftB[col] = s_sft[l_local];
    }
    __syncthreads();
    const int sft = static_cast<int>(s_sft[l_local]);

    } else {
        /* ── B block (symmetric to A, with TRANS_B) ─────────────────── */
        if constexpr (!TRANS_B) {
            /* j-fast: one block per op(B) col, threadIdx.x = j */
            const int64_t col = static_cast<int64_t>(blockIdx.x - m_blks);

            /* Loop 1: compute per-col max */
            double local_max = 0.0;
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val = B[j + col * ldb];                     /* COALESCED */
                if constexpr (CHECK_NAN)
                    if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > local_max) local_max = av;
            }
            local_max = warp_reduce_max_abs_d(local_max);
            local_max = block_reduce_max_d(local_max, s_wmax);
            if(threadIdx.x == 0) {
                if(local_max < 1e-300) local_max = 1.0;
                s_sft[0] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(local_max))));
                sftB[col] = s_sft[0];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[0]);

            /* Loop 2: scale and write B8i_high */
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val    = B[j + col * ldb];                  /* COALESCED */
                double scaled = ceil(ldexp(fabs(val), sft));
                B8i_high[static_cast<size_t>(j)
                         + static_cast<size_t>(col) * ldb8i] =
                    static_cast<int8_t>(static_cast<int32_t>(scaled));  /* COALESCED */
            }

        } else {
            /* Tiled SHMEM for TRANS_B=T (B stored n×k, B[col,j]=B[col+j*ldb]).
             * One block per TILE_M-col tile.                                */
            const int64_t n_base = static_cast<int64_t>(blockIdx.x - m_blks) * TILE_M;
            const int t       = static_cast<int>(threadIdx.x);
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int l_local = t % TILE_M;   /* col-index within tile */
            const int64_t col = n_base + l_local;

            /* Pass 1: accumulate per-col max */
            double thr_max = 0.0;
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;
                if(col < n && j < k) {
                    /* B[col + j*ldb]: adjacent col (= l_local, varies) → COALESCED */
                    double val = B[col + j * ldb];
                    if constexpr (CHECK_NAN)
                        if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                    double av = fabs(val);
                    if(av > thr_max) thr_max = av;
                }
            }
            shmem[k_local][l_local] = thr_max;
            __syncthreads();
            if(k_local == 0) {
                double col_max = 0.0;
                for(int kl = 0; kl < TILE_K; ++kl)
                    if(shmem[kl][l_local] > col_max) col_max = shmem[kl][l_local];
                if(col_max < 1e-300) col_max = 1.0;
                s_sft[l_local] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(col_max))));
                if(col < n) sftB[col] = s_sft[l_local];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[l_local]);

            /* Pass 2: coalesced loads (col-fast) → SHMEM → coalesced writes (k-fast) */
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;

                double scaled = 0.0;
                if(col < n && j < k)
                    scaled = ceil(ldexp(fabs(B[col + j * ldb]), sft));  /* COALESCED */
                shmem[k_local][l_local] = scaled;
                __syncthreads();

                const int k_write = t % TILE_K;
                const int l_write = t / TILE_K;
                const int64_t j_out   = k_base  + k_write;
                const int64_t col_out = n_base  + l_write;
                if(col_out < n && j_out < k)
                    B8i_high[static_cast<size_t>(j_out)
                             + static_cast<size_t>(col_out) * ldb8i] =
                        static_cast<int8_t>(static_cast<int32_t>(
                            shmem[k_write][l_write]));
                __syncthreads();
            }
        }
    }
}

/* =========================================================================
 * GPU kernels — accu mode Part 1: shift refinement from preliminary GEMM
 * ========================================================================= */
__global__ static void
oz2_refine_sftA_partial_kernel(const int32_t* __restrict__ C32i,
                                int64_t m, int64_t n, size_t ldc32i,
                                int32_t* __restrict__ row_max)
{
    const int64_t row      = static_cast<int64_t>(blockIdx.x) * 64
                           + static_cast<int64_t>(threadIdx.x);
    const int64_t col_base = static_cast<int64_t>(blockIdx.y) * 64;
    if(row >= m) return;
    int32_t local_max = 0;
    const int64_t col_end = (col_base + 64 < n) ? col_base + 64 : n;
    for(int64_t col = col_base; col < col_end; ++col) {
        int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    if(local_max > 0) atomicMax(row_max + static_cast<size_t>(row), local_max);
}

/* ── ADP (Adaptive Precision) helpers ────────────────────────────────────── */
/* atomicMax for non-negative floats: IEEE 754 positive floats are totally ordered
 * by their integer bit representation, so int-based atomicMax is correct.
 * The caller must ensure val ≥ 0 (we bias log2P_req values by +200 to guarantee this). */
static __device__ __forceinline__ void oz2_adp_atomicMaxF(float* addr, float val)
{
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
}

/* Computes global max of  (52 − sftA_init[i]) + 0.5·log2(row_max[i]) + 200
 * over all rows i ∈ [0, m).  The +200 bias guarantees a non-negative result
 * (log2P_req_unbiased is in ≈ [−50, 170] for any valid FP64 input).
 * Subtract 200 on the host to recover the actual log2P requirement.
 *
 * Must be launched AFTER oz2_refine_sftA_partial_kernel (which fills row_max[])
 * and BEFORE oz2_refine_sftA_apply_kernel (which overwrites sftA[]).           */
__global__ static void
oz2_adp_reduce_A_kernel(const int32_t* __restrict__ row_max,
                         const int16_t* __restrict__ sftA_init,
                         int64_t m,
                         float* __restrict__ adp_A_out)
{
    __shared__ float s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE]; /* 8 slots */
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x
                      + static_cast<int64_t>(threadIdx.x);

    /* Biased log2P requirement for this row.
     * Skip rows where row_max == 0: those rows have all-zero preliminary inner
     * products (e.g. zero-matrix inputs) and need zero CRT precision (s=2).
     * Using max(row_max,1) would incorrectly return (52 − sftA_init) bits for
     * zero rows because sftA_init=6 is a dummy value set for zero inputs.    */
    float local_val = 0.0f;  /* 0.0f = biased −200 = needs no precision */
    if(row < m && row_max[row] > 0) {
        const int32_t rm   = row_max[row];
        const float sftA_f = static_cast<float>(sftA_init[row]);
        local_val = (52.0f - sftA_f) + 0.5f * log2f(static_cast<float>(rm)) + 200.0f;
    }

    /* Warp-level max reduction. */
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        float other = __shfl_down(local_val, off);
        if(other > local_val) local_val = other;
    }

    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = local_val;
    __syncthreads();

    if(threadIdx.x == 0) {
        float block_max = 0.0f;
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        for(int w = 0; w < nw; ++w)
            if(s_wmax[w] > block_max) block_max = s_wmax[w];
        oz2_adp_atomicMaxF(adp_A_out, block_max);
    }
}

/* log2P is passed as a host-side float constant (from h_accu_log2P_all[s-2]).
 * For s=13,14,15 this is fast::log2P (1 bit below accu) to prevent the OZ2
 * CRT invariant |X_true| < M_s/2 from being violated by floor discretisation.
 * For all other s the full accu::log2P is used, preserving maximum precision. */
__global__ static void
oz2_refine_sftA_apply_kernel(const int32_t* __restrict__ row_max,
                              int16_t* __restrict__ sftA, int64_t m,
                              float log2P)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 64
                      + static_cast<int64_t>(threadIdx.x);
    if(row >= m) return;
    int32_t max_val = row_max[row];
    if(max_val < 1) max_val = 1;
    sftA[row] += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(max_val)) + log2P));
}

/* Computes global max of  (52 − sftB_init[j]) + 0.5·log2(col_max[j]) + 200
 * over all columns j ∈ [0, n).  col_max[j] is the per-column max of |C32i[i,j]|.
 * Does NOT modify sftB[].  Must be launched BEFORE oz2_refine_sftB_kernel
 * so that sftB[] still holds sftB_init (the initial values from step 1b).     */
__global__ static void
oz2_adp_reduce_B_kernel(const int32_t* __restrict__ C32i,
                         int64_t m, int64_t n, size_t ldc32i,
                         const int16_t* __restrict__ sftB_init,
                         float* __restrict__ adp_B_out)
{
    __shared__ int32_t s_wmax[8];
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if(col >= n) return;
    int32_t local_max = 0;
    for(int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_i32(local_max);
    local_max = block_reduce_max_i32(local_max, s_wmax);
    /* Skip zero columns for the same reason as zero rows in the A-side kernel:
     * col_max == 0 means all preliminary products for this column are zero,
     * so no CRT precision is needed (leaves adp_B_out at its init value 0.0f). */
    if(threadIdx.x == 0 && local_max > 0) {
        const float sftB_f = static_cast<float>(sftB_init[col]);
        const float req_biased = (52.0f - sftB_f)
                               + 0.5f * log2f(static_cast<float>(local_max))
                               + 200.0f;
        oz2_adp_atomicMaxF(adp_B_out, req_biased);
    }
}

/* oz2_refine_sftB_kernel — computes per-column max of |C32i_prelim| and applies
 * the shift-refinement delta to sftB[col].  When used in dynamic (ADP) mode, the
 * log2P argument is already the correct value for effective_s (chosen after the
 * ADP reduction above); no separate ADP output is needed here.                */
__global__ static void
oz2_refine_sftB_kernel(const int32_t* __restrict__ C32i,
                       int64_t m, int64_t n, size_t ldc32i,
                       int16_t* __restrict__ sftB, float log2P)
{
    __shared__ int32_t s_wmax[8];
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if(col >= n) return;
    int32_t local_max = 0;
    for(int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_i32(local_max);
    local_max = block_reduce_max_i32(local_max, s_wmax);
    if(threadIdx.x == 0) {
        if(local_max < 1) local_max = 1;
        sftB[col] += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P));
    }
}

/* =========================================================================
 * GPU kernels — Part 1f: full multi-modulus scaling (A and B fused)
 *
 * oz2_scaleAB_kernel<T_COUNT, TRANS_A, TRANS_B>: compile-time template.
 *
 * Both A and B branches apply the same 2-pass symmetric modular reduction
 * (FP64 pass + 1 FP32 refinement pass), matching the GEMMul8 reference.
 * With OZ2_S_MAX=18 there is no need for a 3rd FP32 pass or a runtime-
 * fallback kernel.
 *
 * For TRANS_A=true / TRANS_B=false (coalesced reads):
 *   j = t%TILE_K varies fast within warp → stride-1 HBM reads (COALESCED).
 *
 * For TRANS_A=false / TRANS_B=true (non-coalesced reads):
 *   Tiled SHMEM transposition (same structure as oz2_accu_prelim_kernel):
 *     Load:  k_local=t/TILE_M, m_local=t%TILE_M → A[i+j*lda] (m_local fast → COALESCED)
 *     Store raw val in shmem[k_local][m_local]; store per-row sft in s_sft[m_local].
 *     __syncthreads()
 *     Write: k_write=t%TILE_K, m_write=t/TILE_K → shmem transposed read
 *            → A8i[j_out + i_out*lda8i + t_local*stride]  (k_write fast → COALESCED)
 *
 * Grid  = dim3(ceil(k/TILE_K), ceil(m/TILE_M) + ceil(n/TILE_M))  [unchanged]
 * Block = dim3(TILE_K × TILE_M) = dim3(256)                      [was dim3(64,4)]
 *   blockIdx.y <  m_y_blocks → A scaling
 *   blockIdx.y >= m_y_blocks → B scaling
 * ========================================================================= */
static constexpr unsigned OZ2_SCALE_TILE_K = 64;
static constexpr unsigned OZ2_SCALE_TILE_M = 4;

template <unsigned T_COUNT, bool TRANS_A, bool TRANS_B>
__global__ static void
oz2_scale_A_T_kernel(const double* __restrict__ A,
                     int64_t m, int64_t lda,
                     int8_t* __restrict__ A8i, size_t lda8i, size_t cola8i,
                     const int16_t* __restrict__ sftA,
                     int64_t k, unsigned t_start)
{
    static constexpr int TILE_K = static_cast<int>(OZ2_SCALE_TILE_K);   /* 64 */
    static constexpr int TILE_M = static_cast<int>(OZ2_SCALE_COALESC_TILE_M); /* 8 */
    const int t = static_cast<int>(threadIdx.x);
    const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
    /* j0 is always even → the double2 load is 16-byte aligned when lda is even. */
    const int64_t j0 = static_cast<int64_t>(blockIdx.x) * (TILE_K * 2)
                       + static_cast<int64_t>(t % TILE_K) * 2;
    const int64_t j1 = j0 + 1;          /* adjacent element */
    const int64_t i  = m_base + (t / TILE_K);
    if(i >= m || j0 >= k) return;
    const int  sft    = static_cast<int>(sftA[i]);
    const bool valid1 = (j1 < k);
    double ival0, ival1;
    if(valid1) {
        /* 128-bit load: reads j0 and j0+1 in one instruction (j0 is even → aligned). */
        const double2 vv = *reinterpret_cast<const double2*>(A + i * lda + j0);
        ival0 = trunc(ldexp(vv.x, sft));
        ival1 = trunc(ldexp(vv.y, sft));
    } else {
        ival0 = trunc(ldexp(A[i * lda + j0], sft));
        ival1 = 0.0;
    }
    const size_t stride   = lda8i * cola8i;
    const size_t off_base = static_cast<size_t>(i) * lda8i;
    const size_t off0 = static_cast<size_t>(j0) + off_base; /* always even → uint16_t aligned */
    #pragma unroll
    for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
        const unsigned tidx    = t_start + t_local;
        const double  neg_mod  = oz2_neg_mod(tidx);
        const double  inv_mod  = oz2_inv_mod(tidx);
        const float   inv_modf = oz2_inv_mod_f(tidx);
        const double  r0   = fma(neg_mod, rint(ival0 * inv_mod), ival0);
        const float   rf0  = static_cast<float>(r0);
        const auto    b0   = static_cast<int8_t>(static_cast<int32_t>(
                                 fmaf(rintf(rf0 * inv_modf), static_cast<float>(neg_mod), rf0)));
        if(valid1) {
            /* Pack INT8[j0] and INT8[j0+1] into one 16-bit NT store. */
            const double  r1   = fma(neg_mod, rint(ival1 * inv_mod), ival1);
            const float   rf1  = static_cast<float>(r1);
            const auto    b1   = static_cast<int8_t>(static_cast<int32_t>(
                                     fmaf(rintf(rf1 * inv_modf), static_cast<float>(neg_mod), rf1)));
            const uint16_t packed = static_cast<uint8_t>(b0)
                                  | (static_cast<uint16_t>(static_cast<uint8_t>(b1)) << 8);
            __builtin_nontemporal_store(packed,
                reinterpret_cast<uint16_t*>(A8i + t_local * stride + off0));
        } else {
            __builtin_nontemporal_store(b0, A8i + t_local * stride + off0);
        }
    }
}

/* ── A_N: TRANS_A=false, SHMEM transposition, blockDim=1024, TILE_M=16 ── */
template <unsigned T_COUNT>
__global__ static void
oz2_scale_A_N_kernel(const double* __restrict__ A,
                     int64_t m, int64_t lda,
                     int8_t* __restrict__ A8i, size_t lda8i, size_t cola8i,
                     const int16_t* __restrict__ sftA,
                     int64_t k, unsigned t_start)
{
    static constexpr int TILE_K = static_cast<int>(OZ2_SCALE_TILE_K);
    static constexpr int TILE_M = static_cast<int>(OZ2_SCALE_TILE_M);

    /* SHMEM used only in non-coalesced paths (TRANS_A=false / TRANS_B=true).
     * Declared unconditionally; coalesced paths skip it via if constexpr.   */
    __shared__ double  shmem[TILE_K][TILE_M + 1];  /* +1 avoids bank conflicts */
    __shared__ int16_t s_sft[TILE_M];

    const int t = static_cast<int>(threadIdx.x);

    if(static_cast<unsigned>(blockIdx.y) < m_y_blocks) {
        /* ── A block ─────────────────────────────────────────────────────── */
        const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;

    const int k_write = t % TILE_K;
    const int m_write = t / TILE_K;
    const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
    const int64_t i_out = m_base + m_write;
    if(i_out < m && j_out < k) {
        const double val  = shmem[k_write][m_write];
        const double ival = trunc(ldexp(val, static_cast<int>(s_sft[m_write])));
        const size_t stride = lda8i * cola8i;
        const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i;
        #pragma unroll
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
            const unsigned tidx = t_start + t_local;
            const double  r  = fma(oz2_neg_mod(tidx), rint(ival * oz2_inv_mod(tidx)), ival);
            const float   rf = static_cast<float>(r);
            const float  rf2 = fmaf(rintf(rf * oz2_inv_mod_f(tidx)), static_cast<float>(oz2_neg_mod(tidx)), rf);
            __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                        A8i + t_local * stride + offset);
        }
    }
}

/* ── B_N: TRANS_B=false, j-fast coalesced, blockDim=512, TILE_M=8, K_UNROLL=2 ──
 * Mirrors oz2_scale_A_T_kernel: double2 load + uint16_t packed store. */
template <unsigned T_COUNT>
__global__ static void
oz2_scale_B_N_kernel(const double* __restrict__ B,
                     int64_t n, int64_t ldb,
                     int8_t* __restrict__ B8i, size_t ldb8i,
                     const int16_t* __restrict__ sftB,
                     int64_t k, unsigned t_start)
{
    static constexpr int TILE_K = static_cast<int>(OZ2_SCALE_TILE_K);   /* 64 */
    static constexpr int TILE_M = static_cast<int>(OZ2_SCALE_COALESC_TILE_M); /* 8 */
    const int t = static_cast<int>(threadIdx.x);
    const int64_t n_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int64_t j0  = static_cast<int64_t>(blockIdx.x) * (TILE_K * 2)
                        + static_cast<int64_t>(t % TILE_K) * 2;
    const int64_t j1  = j0 + 1;         /* adjacent element */
    const int64_t col = n_base + (t / TILE_K);
    if(col >= n || j0 >= k) return;
    const int  sft    = static_cast<int>(sftB[col]);
    const bool valid1 = (j1 < k);
    double ival0, ival1;
    if(valid1) {
        const double2 vv = *reinterpret_cast<const double2*>(B + col * ldb + j0);
        ival0 = trunc(ldexp(vv.x, sft));
        ival1 = trunc(ldexp(vv.y, sft));
    } else {
        ival0 = trunc(ldexp(B[col * ldb + j0], sft));
        ival1 = 0.0;
    }
    const size_t stride   = ldb8i * static_cast<size_t>(n);
    const size_t off_base = static_cast<size_t>(col) * ldb8i;
    const size_t off0 = static_cast<size_t>(j0) + off_base; /* always even → uint16_t aligned */
    #pragma unroll
    for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
        const unsigned tidx    = t_start + t_local;
        const double  neg_mod  = oz2_neg_mod(tidx);
        const double  inv_mod  = oz2_inv_mod(tidx);
        const float   inv_modf = oz2_inv_mod_f(tidx);
        const double  r0   = fma(neg_mod, rint(ival0 * inv_mod), ival0);
        const float   rf0  = static_cast<float>(r0);
        const auto    b0   = static_cast<int8_t>(static_cast<int32_t>(
                                 fmaf(rintf(rf0 * inv_modf), static_cast<float>(neg_mod), rf0)));
        if(valid1) {
            const double  r1   = fma(neg_mod, rint(ival1 * inv_mod), ival1);
            const float   rf1  = static_cast<float>(r1);
            const auto    b1   = static_cast<int8_t>(static_cast<int32_t>(
                                     fmaf(rintf(rf1 * inv_modf), static_cast<float>(neg_mod), rf1)));
            const uint16_t packed = static_cast<uint8_t>(b0)
                                  | (static_cast<uint16_t>(static_cast<uint8_t>(b1)) << 8);
            __builtin_nontemporal_store(packed,
                reinterpret_cast<uint16_t*>(B8i + t_local * stride + off0));
        } else {
            /* Non-coalesced load: A stored m×k, A[i,j] = A[i + j*lda].
             * Use SHMEM transposition for coalesced reads AND writes.       */
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_local;
            const int64_t i = m_base + m_local;

            /* Load raw val (m_local varies fast → COALESCED) */
            shmem[k_local][m_local] = (i < m && j < k) ? A[i + j * lda] : 0.0;
            if(k_local == 0 && i < m) s_sft[m_local] = sftA[i];
            __syncthreads();

            /* Write: k_write varies fast → COALESCED writes */
            const int k_write = t % TILE_K;
            const int m_write = t / TILE_K;
            const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
            const int64_t i_out = m_base + m_write;
            if(i_out < m && j_out < k) {
                const double val  = shmem[k_write][m_write];
                const double ival = trunc(ldexp(val, static_cast<int>(s_sft[m_write])));
                const size_t stride = lda8i * cola8i;
                const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i;
                #pragma unroll
                for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                    const unsigned tidx = t_start + t_local;
                    const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                    const float   rf = static_cast<float>(r);
                    const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                            static_cast<float>(cNegMod[tidx]), rf);
                    __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                                A8i + t_local * stride + offset);
                }
            }
        }
    } else {
        /* ── B block (symmetric to A, with TRANS_B) ─────────────────────── */
        const int64_t n_base = static_cast<int64_t>(blockIdx.y - m_y_blocks) * TILE_M;

        if constexpr (!TRANS_B) {
            /* Coalesced: B stored k×n, B[l,j] = B[j + l*ldb].
             * j = t%TILE_K varies fast → stride-1 reads.                   */
            const int64_t j   = static_cast<int64_t>(blockIdx.x) * TILE_K + (t % TILE_K);
            const int64_t col = n_base + (t / TILE_K);
            if(col >= n || j >= k) return;
            const double val  = B[col * ldb + j];                       /* COALESCED */
            const double ival = trunc(ldexp(val, static_cast<int>(sftB[col])));
            const size_t stride = ldb8i * static_cast<size_t>(n);
            const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(col) * ldb8i;
            #pragma unroll
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                const unsigned tidx = t_start + t_local;
                const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                const float   rf = static_cast<float>(r);
                const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                        static_cast<float>(cNegMod[tidx]), rf);
                __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                            B8i + t_local * stride + offset);
            }
        } else {
            /* Non-coalesced: B stored n×k, B[l,j] = B[l + j*ldb].
             * Use SHMEM transposition.                                       */
            const int k_local = t / TILE_M;
            const int l_local = t % TILE_M;
            const int64_t j   = static_cast<int64_t>(blockIdx.x) * TILE_K + k_local;
            const int64_t col = n_base + l_local;

            /* Load raw val (l_local varies fast → COALESCED) */
            shmem[k_local][l_local] = (col < n && j < k) ? B[col + j * ldb] : 0.0;
            if(k_local == 0 && col < n) s_sft[l_local] = sftB[col];
            __syncthreads();

    shmem[k_local][l_local] = (col < n && j < k) ? B[col + j * ldb] : 0.0; /* COALESCED */
    if(k_local == 0 && col < n) s_sft[l_local] = sftB[col];
    __syncthreads();

    const int k_write = t % TILE_K;
    const int l_write = t / TILE_K;
    const int64_t j_out   = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
    const int64_t col_out = n_base + l_write;
    if(col_out < n && j_out < k) {
        const double val  = shmem[k_write][l_write];
        const double ival = trunc(ldexp(val, static_cast<int>(s_sft[l_write])));
        const size_t stride = ldb8i * static_cast<size_t>(n);
        const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(col_out) * ldb8i;
        #pragma unroll
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
            const unsigned tidx = t_start + t_local;
            const double  r  = fma(oz2_neg_mod(tidx), rint(ival * oz2_inv_mod(tidx)), ival);
            const float   rf = static_cast<float>(r);
            const float  rf2 = fmaf(rintf(rf * oz2_inv_mod_f(tidx)), static_cast<float>(oz2_neg_mod(tidx)), rf);
            __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                        B8i + t_local * stride + offset);
        }
    }
}

/* =========================================================================
 * GPU kernels — Part 2d: chunked CRT accumulation
 * ========================================================================= */
template <bool HAS_LO>
__global__ static void
oz2_chunk_accum_kernel_rt(const int32_t* __restrict__ C32i_batch,
                           double* __restrict__ Zhi, double* __restrict__ Zlo,
                           int64_t m, int64_t n, size_t ldc32i,
                           unsigned chunk_start, unsigned chunk_size, bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    if(is_first_chunk) { Zhi[idx] = local_hi; Zlo[idx] = local_lo; }
    else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi; Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO>
__global__ static void
oz2_accum_finalize_kernel_rt(const int32_t* __restrict__ C32i_batch,
                              const double* __restrict__ Zhi_in, const double* __restrict__ Zlo_in,
                              const double* __restrict__ C, double* __restrict__ D,
                              int64_t m, int64_t n, size_t ldc32i, int64_t ldc, int64_t ldd,
                              double alpha, double beta,
                              const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                              unsigned chunk_start, unsigned chunk_size, bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    double Zh, Zl;
    if(is_first_chunk) { Zh = local_hi; Zl = local_lo; }
    else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi; Zl = Zlo_in[idx] + err + local_lo;
    }
    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);
    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_chunk_accum_kernel(const int32_t* __restrict__ C32i_batch,
                       double* __restrict__ Zhi, double* __restrict__ Zlo,
                       int64_t m, int64_t n, size_t ldc32i, unsigned chunk_start, unsigned effective_s)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(oz2_neg_mod(t), rint(dc_raw * oz2_inv_mod(t)), dc_raw);
        const double hi     = dc * oz2_qpi_hi(effective_s - 2, t);
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, oz2_qpi_lo(effective_s - 2, t), local_lo + err);
        else                   local_lo += err;
    }
    if constexpr (IS_FIRST_CHUNK) {
        __builtin_nontemporal_store(local_hi, Zhi + idx);
        __builtin_nontemporal_store(local_lo, Zlo + idx);
    } else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi; Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_accum_finalize_kernel(const int32_t* __restrict__ C32i_batch,
                          const double* __restrict__ Zhi_in, const double* __restrict__ Zlo_in,
                          const double* __restrict__ C, double* __restrict__ D,
                          int64_t m, int64_t n, size_t ldc32i, int64_t ldc, int64_t ldd,
                          double alpha, double beta,
                          const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                          unsigned chunk_start, unsigned effective_s)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(oz2_neg_mod(t), rint(dc_raw * oz2_inv_mod(t)), dc_raw);
        const double hi     = dc * oz2_qpi_hi(effective_s - 2, t);
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, oz2_qpi_lo(effective_s - 2, t), local_lo + err);
        else                   local_lo += err;
    }
    double Zh, Zl;
    if constexpr (IS_FIRST_CHUNK) { Zh = local_hi; Zl = local_lo; }
    else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi; Zl = Zlo_in[idx] + err + local_lo;
    }
    const double q = rint((Zh + Zl) * oz2_inv_P(effective_s - 2));
    const double X = fma(oz2_P_lo(effective_s - 2), q, fma(oz2_P_hi(effective_s - 2), q, Zh) + Zl);
    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    double d_val = alpha * ldexp(X, inv_sft);
    if (beta != 0.0) {
        const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
        d_val += beta * C[c_idx];
    }
    __builtin_nontemporal_store(d_val, D + d_idx);
}

static const char* oz2_profile_file()
{
    static const char* const fn = std::getenv("HIPBLASLT_EMULATION_PROFILE");
    return fn;
}

/* =========================================================================
 * oz2_native_dgemm_fallback — run a plain FP64 hipblasLtMatmul for a
 * sub-block.  Used by the split path in fp64EmulatedGemmImpl when the
 * second half's emulation fails after the first half has already written D,
 * so that D is fully correct without corrupting the first half's output.
 *
 * A fresh hipblasLtHandle is created with emulation explicitly disabled
 * (emulation.enabled=0) so that HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 in
 * the environment does not cause this call to re-enter emulation recursively.
 * The default enabled=-1 means "check env var", which would re-trigger it.
 * ========================================================================= */
static rocblaslt_status
oz2_native_dgemm_fallback(const _rocblaslt_handle* h,
                           hipblasOperation_t opA, hipblasOperation_t opB,
                           int64_t m, int64_t n, int64_t k,
                           const double* alpha, const double* A, int64_t lda,
                           const double* B,     int64_t ldb,
                           const double* beta,  const double* C, int64_t ldc,
                           double* D, int64_t ldd,
                           hipStream_t stream)
{
    hipblasLtHandle_t       fp64_handle = nullptr;
    hipblasLtMatrixLayout_t layoutA     = nullptr;
    hipblasLtMatrixLayout_t layoutB     = nullptr;
    hipblasLtMatrixLayout_t layoutC     = nullptr;
    hipblasLtMatrixLayout_t layoutD     = nullptr;
    hipblasLtMatmulDesc_t   desc        = nullptr;

    auto cleanup = [&]() noexcept {
        if(desc)        (void)hipblasLtMatmulDescDestroy(desc);
        if(layoutD)     (void)hipblasLtMatrixLayoutDestroy(layoutD);
        if(layoutC)     (void)hipblasLtMatrixLayoutDestroy(layoutC);
        if(layoutB)     (void)hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)     (void)hipblasLtMatrixLayoutDestroy(layoutA);
        if(fp64_handle) (void)hipblasLtDestroy(fp64_handle);
    };

    if(hipblasLtCreate(&fp64_handle) != HIPBLAS_STATUS_SUCCESS)
        return rocblaslt_status_internal_error;

    /* Explicitly disable emulation on this fresh handle.  The default
     * enabled=-1 means "check env var"; 0 means "force off" regardless of
     * HIPBLASLT_EMULATE_DOUBLE_PRECISION, preventing recursive re-entry. */
    reinterpret_cast<_rocblaslt_handle*>(fp64_handle)->emulation.enabled = 0;

    /* Physical (stored) matrix dimensions for column-major layout:
     *   opA=N → A is m×k; opA=T → A is k×m (transposed in matmulDesc).
     *   opB=N → B is k×n; opB=T → B is n×k.                             */
    const uint64_t rows_A = (opA == HIPBLAS_OP_N) ? static_cast<uint64_t>(m) : static_cast<uint64_t>(k);
    const uint64_t cols_A = (opA == HIPBLAS_OP_N) ? static_cast<uint64_t>(k) : static_cast<uint64_t>(m);
    const uint64_t rows_B = (opB == HIPBLAS_OP_N) ? static_cast<uint64_t>(k) : static_cast<uint64_t>(n);
    const uint64_t cols_B = (opB == HIPBLAS_OP_N) ? static_cast<uint64_t>(n) : static_cast<uint64_t>(k);

    hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_64F, rows_A, cols_A, lda);
    hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_64F, rows_B, cols_B, ldb);
    hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_64F, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldc);
    hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_64F, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldd);
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    const hipblasStatus_t st =
        hipblasLtMatmul(fp64_handle, desc,
                        alpha, A, layoutA,
                        B,        layoutB,
                        beta,  C, layoutC,
                        D,        layoutD,
                        nullptr, nullptr, 0, stream);
    cleanup();
    return (st == HIPBLAS_STATUS_SUCCESS) ? rocblaslt_status_success
                                          : rocblaslt_status_internal_error;
}

/* =========================================================================
 * fp64EmulatedGemm — profiling accumulator + implementation
 * ========================================================================= */

/* Aggregates per-component GPU times across all leaf sub-GEMMs and counts
 * how many leaf (monolithic) sub-GEMMs were executed.  Owned by the public
 * fp64EmulatedGemm wrapper; passed by pointer through recursive calls.     */
struct Fp64ProfileAccum {
    float    t_prelim      = 0.f;
    float    t_prelim_gemm = 0.f;
    float    t_extract     = 0.f;
    float    t_refine      = 0.f;
    float    t_adp         = 0.f;   /* ADP reduce kernels + hipStreamSynchronize (dynamic mode only) */
    float    t_fused       = 0.f;   /* fused TN kernel (non-zero when fused path taken) */
    float    t_scale       = 0.f;
    float    t_int8        = 0.f;
    float    t_accum       = 0.f;
    float    t_finalize    = 0.f;
    unsigned effective_s_used = 0u;   /* ADP: actual s chosen (= num_moduli in fixed mode) */
    unsigned n_sub_gemms   = 0u;
};

/* Internal implementation — called recursively during binary-halving.
 * prof != nullptr enables per-component accumulation across all leaves.   */
static rocblaslt_status
fp64EmulatedGemmImpl(const _rocblaslt_handle*     h,
                     hipblasOperation_t           opA,
                     hipblasOperation_t           opB,
                     int64_t                      m,
                     int64_t                      n,
                     int64_t                      k,
                     const double*                alpha,
                     const double*                A,
                     int64_t                      lda,
                     const double*                B,
                     int64_t                      ldb,
                     const double*                beta,
                     const double*                C,
                     int64_t                      ldc,
                     double*                      D,
                     int64_t                      ldd,
                     hipStream_t                  stream,
                     const Fp64EmulationSettings& settings,
                     Fp64ProfileAccum*            prof)
{
    if(settings.num_moduli == 0u
       && cached_mantissa_bit_count_env().state == FP64_EMULATION_ENV_INVALID)
        return rocblaslt_status_invalid_value;
    const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= OZ2_S_MAX)
                                    ? settings.num_moduli : fp64EmulationNumModuli();
    {
        const unsigned chunk_sz = oz2_compute_chunk_size(m, n, k, num_moduli);
        const unsigned n_chunks = (num_moduli + chunk_sz - 1u) / chunk_sz;

        if(n_chunks > 1u) {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n     : n / 2;
            const int64_t m2      = split_m ? (m - m / 2) : m;
            const int64_t n2      = split_m ? n            : (n - n / 2);

            const int device = h->device;
            const bool tA = (opA != HIPBLAS_OP_N);
            const bool tB = (opB != HIPBLAS_OP_N);
            const double t_mono  = fp64EmulationPerfModelTimes(tA, tB, m, n, k, num_moduli, device, settings.dynamic_mode).t_total_ms;
            const double t_split = 2. * oz2_effective_time_ms(tA, tB, half_m, half_n, k, num_moduli, device, settings.dynamic_mode);

            if(t_split < t_mono) {
                /* First half: rows 0..half_m-1 or cols 0..half_n-1. */
                {
                    rocblaslt_status st =
                        fp64EmulatedGemmImpl(h, opA, opB, half_m, half_n, k, alpha,
                                             A, lda, B, ldb, beta, C, ldc, D, ldd,
                                             stream, settings, prof);
                    if(st != rocblaslt_status_success) return st;
                }

                /* Second half: rows half_m..m-1 or cols half_n..n-1.
                 * CORRECTNESS NOTE: the first half has already written D[0..half_m-1]
                 * (or the first half_n columns of D) with emulation output.  If we
                 * propagate a second-half failure to the caller, it would run whole-matrix
                 * native DGEMM which reads D as C (corrupting beta*C when C==D in-place).
                 * Instead, on second-half failure recover with a targeted native DGEMM for
                 * that sub-block only, keeping the full D output correct.             */
                const double* const A2 = split_m ? (tA ? A + half_m * lda : A + half_m) : A;
                const double* const B2 = split_m ? B : (tB ? B + half_n : B + half_n * ldb);
                const double* const C2 = split_m ? C + half_m : C + half_n * ldc;
                double*       const D2 = split_m ? D + half_m : D + half_n * ldd;
                const rocblaslt_status st2 =
                    fp64EmulatedGemmImpl(h, opA, opB, m2, n2, k, alpha,
                                         A2, lda, B2, ldb, beta, C2, ldc, D2, ldd,
                                         stream, settings, prof);
                if(st2 != rocblaslt_status_success) {
                    /* First half already committed emulation output to D; fall back to
                     * native DGEMM for the second half only so D remains fully correct. */
                    std::fprintf(stderr,
                        "[hipBLASLt FP64 emulation] WARNING: second-half emulation failed "
                        "(m=%lld, n=%lld, k=%lld, st=%d). "
                        "Running native DGEMM for second half to preserve first-half output.\n",
                        (long long)m2, (long long)n2, (long long)k, (int)st2);
                    return oz2_native_dgemm_fallback(h, opA, opB, m2, n2, k,
                                                     alpha, A2, lda, B2, ldb,
                                                     beta, C2, ldc, D2, ldd, stream);
                }
                return rocblaslt_status_success;
            }
        }
    }
    /* ── Existing monolithic path ──────────────────────────────────────────── */

    const bool   _prof = (prof != nullptr);
    hipEvent_t _ev0{}, _ev1{};
    float _t_prelim = 0, _t_prelim_gemm = 0, _t_extract = 0, _t_refine = 0,
          _t_adp = 0, _t_fused = 0, _t_scale = 0, _t_int8 = 0, _t_accum = 0, _t_finalize = 0;
    if(_prof) { (void)hipEventCreate(&_ev0); (void)hipEventCreate(&_ev1); }
    auto _pstart = [&]() noexcept { if(_prof) (void)hipEventRecord(_ev0, stream); };
    auto _pstop  = [&](float& t) noexcept {
        if(_prof) {
            (void)hipEventRecord(_ev1, stream); (void)hipStreamSynchronize(stream);
            float ms = 0.f; (void)hipEventElapsedTime(&ms, _ev0, _ev1); t += ms;
        }
    };

    /* In dynamic (ADP) mode the workspace was allocated for OZ2_S_MAX=18 moduli,
     * so use OZ2_S_MAX for the chunk-size calculation.  This ensures:
     *  - layout_moduli ≥ effective_s for any ADP-chosen effective_s
     *  - chunk_size ≥ effective_s, so all moduli are processed in a single
     *    scale pass + single GEMM pass regardless of effective_s (no 16+2 split)
     *  - buffer layout is consistent with the workspace allocation             */
    const unsigned layout_moduli = settings.dynamic_mode ? OZ2_S_MAX : num_moduli;
    const unsigned chunk_size    = oz2_compute_chunk_size(m, n, k, layout_moduli);

    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);

    const size_t szA8i   = chunk_size * lda8i * cola8i;
    const size_t szB8i   = chunk_size * ldb8i * static_cast<size_t>(n);
    /* Zhi/Zlo needed when there are multiple passes (chunk_size < layout_moduli). */
    const size_t szZhi   = (chunk_size < layout_moduli) ? szC32i : 0u;
    const size_t szZlo   = szZhi;
    const size_t szSftA  = cola8i;
    const size_t szSftB  = padn;
    const size_t szNanFlag = 1;
    const size_t szRowMax  = cola8i;

    const size_t wsBytes =
          szA8i    * sizeof(int8_t)
        + szB8i    * sizeof(int8_t)
        + chunk_size * szC32i * sizeof(int32_t)
        + szZhi    * sizeof(double)
        + szZlo    * sizeof(double)
        + szSftA   * sizeof(int16_t)
        + szSftB   * sizeof(int16_t)
        + szNanFlag * sizeof(uint32_t)
        + szRowMax  * sizeof(int32_t);

    bool   ws_owned = false;
    char*  ws       = nullptr;
    if(settings.workspace != nullptr && settings.workspace_bytes >= wsBytes) {
        ws = static_cast<char*>(settings.workspace);
    } else {
        ws_owned = true;
        if(hipMallocAsync(&ws, wsBytes, stream) != hipSuccess)
            return rocblaslt_status_memory_error;
    }

    int8_t*   const A8i        = reinterpret_cast<int8_t*>(ws);
    int8_t*   const B8i        = A8i + szA8i;
    int32_t*  const C32i_batch = reinterpret_cast<int32_t*>(B8i + szB8i);
    double*   const Zhi        = reinterpret_cast<double*>(C32i_batch + chunk_size * szC32i);
    double*   const Zlo        = Zhi + szZhi;
    int16_t*  const sftA       = reinterpret_cast<int16_t*>(Zlo + szZlo);
    int16_t*  const sftB       = sftA + szSftA;
    uint32_t* const nan_flag   = reinterpret_cast<uint32_t*>(sftB + szSftB);
    int32_t*  const row_max    = reinterpret_cast<int32_t*>(nan_flag + szNanFlag);
    /* adp_buf and int8_ws follow row_max in the workspace layout.
     * Declared here so they are in scope for both the preliminary GEMM
     * and the ADP kernels that come later.                               */
    float*    const adp_buf    = reinterpret_cast<float*>(row_max + szRowMax);
    void*     const int8_ws    = static_cast<void*>(adp_buf + 2);
    constexpr size_t int8_ws_size = OZ2_INT8_GEMM_WS_BYTES;
    int32_t*  const C32i       = C32i_batch;

    if(_prof) (void)hipEventRecord(_ev_tot, stream);

    int8_t* const A8i_high = A8i;
    int8_t* const B8i_high = B8i;

    const bool tA = (opA != HIPBLAS_OP_N);
    const bool tB = (opB != HIPBLAS_OP_N);

    const uint32_t svmask = (settings.sv_mask != ~0u)
                                ? settings.sv_mask : fp64EmulationSpecialValuesMask();

    hipblasLtHandle_t       int8_handle = nullptr; /* dedicated handle for INT8 GEMMs —
                                                    * keeps INT8 Tensile state isolated from
                                                    * the caller's handle so sequential calls
                                                    * with different shapes don't corrupt it. */
    hipblasLtMatrixLayout_t layoutA  = nullptr;
    hipblasLtMatrixLayout_t layoutB  = nullptr;
    hipblasLtMatrixLayout_t layoutCD = nullptr;
    hipblasLtMatrixLayout_t layoutA_b  = nullptr;
    hipblasLtMatrixLayout_t layoutB_b  = nullptr;
    hipblasLtMatrixLayout_t layoutCD_b = nullptr;
    hipblasLtMatmulDesc_t   matmulDesc = nullptr;

    auto oz2_cleanup = [&]() noexcept {
        if(matmulDesc)   (void)hipblasLtMatmulDescDestroy(matmulDesc);
        if(layoutCD)     (void)hipblasLtMatrixLayoutDestroy(layoutCD);
        if(layoutB)      (void)hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)      (void)hipblasLtMatrixLayoutDestroy(layoutA);
        if(int8_handle)  (void)hipblasLtDestroy(int8_handle);
        if(_prof) { (void)hipEventDestroy(_ev1); (void)hipEventDestroy(_ev0); }
    };

    if(svmask != 0u) {
        if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess) {
            oz2_cleanup();
            return rocblaslt_status_internal_error;
        }
    }

    if(hipblasLtCreate(&int8_handle) != HIPBLAS_STATUS_SUCCESS) {
        oz2_cleanup();
        return rocblaslt_status_internal_error;
    }
    hipblasLtMatrixLayoutCreate(&layoutA,  HIP_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(m), static_cast<int64_t>(lda8i));
    hipblasLtMatrixLayoutCreate(&layoutB,  HIP_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(n), static_cast<int64_t>(ldb8i));
    hipblasLtMatrixLayoutCreate(&layoutCD, HIP_R_32I, static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<int64_t>(ldc32i));
    hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
    {
        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        if(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)) != HIPBLAS_STATUS_SUCCESS)
            return fail_internal();
        if(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) != HIPBLAS_STATUS_SUCCESS)
            return fail_internal();
    }

    const int32_t one_i = 1, zero_i = 0;

    /* Fused preliminary shift + extraction (oz2_accu_prelim_kernel).
     * Grid adapts to TRANS_A/TRANS_B:
     *   m_blks = TRANS_A ? m           : ceil(m / OZ2_PRELIM_TILE_M)
     *   n_blks = TRANS_B ? ceil(n/TM)  : n
     * Block = OZ2_PRELIM_TILE_K × OZ2_PRELIM_TILE_M = 256 threads.    */
    const unsigned m_blks_prelim = tA
        ? static_cast<unsigned>(m)
        : static_cast<unsigned>((m + OZ2_PRELIM_TILE_M - 1) / OZ2_PRELIM_TILE_M);
    const unsigned n_blks_prelim = tB
        ? static_cast<unsigned>((n + OZ2_PRELIM_TILE_M - 1) / OZ2_PRELIM_TILE_M)
        : static_cast<unsigned>(n);
    _pstart();
#define OZ2_PRELIM(TA, TB, CN) \
    hipLaunchKernelGGL((oz2_accu_prelim_kernel<(TA),(TB),(CN)>), \
                       dim3(m_blks_prelim + n_blks_prelim), \
                       dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_TILE_M), 0, stream, \
                       A, m, k, lda, A8i_high, lda8i, sftA, \
                       B, n, ldb, B8i_high, ldb8i, sftB, nan_flag, m_blks_prelim)
    if(svmask == 0u) {
        if(tA && !tB)       OZ2_PRELIM(true,  false, false);
        else if(!tA && !tB) OZ2_PRELIM(false, false, false);
        else if(!tA && tB)  OZ2_PRELIM(false, true,  false);
        else                OZ2_PRELIM(true,  true,  false);
    } else {
        if(tA && !tB)       OZ2_PRELIM(true,  false, true);
        else if(!tA && !tB) OZ2_PRELIM(false, false, true);
        else if(!tA && tB)  OZ2_PRELIM(false, true,  true);
        else                OZ2_PRELIM(true,  true,  true);
    }
#undef OZ2_PRELIM
    _pstop(_t_prelim);
    /* _t_extract remains 0: extraction is now fused into _t_prelim */

    if(svmask != 0u) {
        if(hipStreamSynchronize(stream) != hipSuccess) {
            oz2_cleanup(); return rocblaslt_status_internal_error;
        }
        uint32_t detected = 0u;
        if(hipMemcpy(&detected, nan_flag, sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
            oz2_cleanup(); return rocblaslt_status_internal_error;
        }
        if(detected & svmask) {
            oz2_cleanup(); return rocblaslt_status_invalid_value;
        }
    }

    /* Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high */
    _pstart();
    {
        const hipblasStatus_t prelim_st =
            hipblasLtMatmul(int8_handle, matmulDesc,
                            &one_i, A8i_high, layoutA, B8i_high, layoutB,
                            &zero_i, C32i, layoutCD, C32i, layoutCD, nullptr, int8_ws, int8_ws_size, stream);
        _pstop(_t_prelim_gemm);
        if(prelim_st != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr,
                    "[hipBLASLt FP64 emulation] WARNING: preliminary INT8 GEMM failed "
                    "(m=%lld, n=%lld, k=%lld, status=%d). "
                    "Falling back to native DGEMM.\n",
                    (long long)m, (long long)n, (long long)k, (int)prelim_st);
            /* Drain the stream before clearing the error so that the stream is in
             * a clean state for future operations on this stream/context.       */
            (void)hipStreamSynchronize(stream);
            (void)hipGetLastError();
            oz2_cleanup();
            return rocblaslt_status_internal_error;
        }
    }

    /* log2P from the mixed table: fast::log2P for s=13,14,15 (prevents CRT overflow);
     * accu::log2P for all other s (preserves maximum precision).               */
    const float accu_log2P = h_accu_log2P_all[num_moduli - 2];
    _pstart();
    const unsigned sftA_m_blks = static_cast<unsigned>((m + 63) / 64);
    const unsigned sftA_n_blks = static_cast<unsigned>((n + 63) / 64);
    /* adp_buf: 2 floats used by ADP (Adaptive Precision) kernels (see above).
     * Initialised to 0.0f (= biased −200 → effective_s=2 if all rows/cols zero). */

    /* effective_s: ADP may reduce this below num_moduli; determined BEFORE the
     * shift-refinement delta is applied so we can use the correct log2P.       */
    unsigned effective_s = num_moduli;

    (void)hipMemsetAsync(row_max, 0, szRowMax * sizeof(int32_t), stream);
    hipLaunchKernelGGL(oz2_refine_sftA_partial_kernel, dim3(sftA_m_blks, sftA_n_blks), dim3(64), 0, stream,
                       C32i, m, n, ldc32i, row_max);
    _pstop(_t_refine);  /* partial: refine_sftA_partial only */

    /* ADP (dynamic mode): determine effective_s from the preliminary GEMM result
     * BEFORE applying any shift delta.  Both ADP kernels read sftA_init and
     * sftB_init (the values set in step 1a/1b, before any += delta).
     * The refine delta is then applied using log2P_{effective_s} so that
     * X_true is sized to fit within M_{effective_s}/2, not M_{num_moduli}/2.  */
    if(settings.dynamic_mode) {
        _pstart();
        (void)hipMemsetAsync(adp_buf, 0, 2 * sizeof(float), stream);
        /* A-side: reads row_max[] and sftA[] before apply_kernel modifies sftA. */
        hipLaunchKernelGGL(oz2_adp_reduce_A_kernel, dim3(sftA_m_blks), dim3(OZ2_PRELIM_COALESC_THRS), 0, stream,
                           row_max, sftA, m, adp_buf + 0);
        /* B-side: reads C32i and sftB[] before refine_sftB_kernel modifies sftB. */
        hipLaunchKernelGGL(oz2_adp_reduce_B_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, stream,
                           C32i, m, n, ldc32i, sftB, adp_buf + 1);

        /* Sync, copy 2 floats, compute effective_s on host.
         * hipGetLastError() clears any sticky thread-level error that may have
         * been set by the ADP kernels (e.g. from a previous GPU fault on the
         * same stream).  Without this, a subsequent hipMallocAsync on the same
         * stream may fail even though the device has plenty of free memory.   */
        (void)hipStreamSynchronize(stream);
        (void)hipGetLastError();
        _pstop(_t_adp);  /* ADP reduce kernels + hipStreamSynchronize */
        float h_adp[2] = {0.0f, 0.0f};
        (void)hipMemcpy(h_adp, adp_buf, 2 * sizeof(float), hipMemcpyDeviceToHost);

        const float log2P_needed = std::max(h_adp[0], h_adp[1]) - 200.0f;

        if(log2P_needed > h_accu_log2P_all[OZ2_S_MAX - 2u]) {
            /* ADP determined s=OZ2_S_MAX is still insufficient for this input.
             * This occurs for matrices with extremely large dynamic range within
             * a single row/column (e.g. condition number ≫ 2^{2×log2P_18}).
             * The Ozaki shift-refinement would set A8i_final or B8i_final to
             * near-zero for the elements providing cancellation, giving wrong
             * results regardless of s.  Fall back to native DGEMM.
             * Rate-limited warning (≤5 per process).                         */
            std::fprintf(stderr,
                    "[hipBLASLt FP64 emulation] WARNING: ADP overflow for GEMM "
                    "(m=%lld, n=%lld, k=%lld): "
                    "A-side log2P_req=%.1f bits, B-side=%.1f bits, "
                    "max required=%.1f > supported max=%.1f "
                    "(s=%u moduli, ~%.0f cumulative bits). "
                    "Falling back to native DGEMM.\n",
                    (long long)m, (long long)n, (long long)k,
                    h_adp[0] - 200.0f, h_adp[1] - 200.0f,
                    log2P_needed, h_accu_log2P_all[OZ2_S_MAX - 2u],
                    OZ2_S_MAX, oz2_cum_bits[OZ2_S_MAX - 2u]);
            oz2_cleanup();
            return rocblaslt_status_invalid_value;
        }

        for(unsigned s = 2u; s <= OZ2_S_MAX; ++s) {
            if(h_accu_log2P_all[s - 2u] >= log2P_needed) {
                effective_s = s;
                break;
            }
        }

        if(effective_s != num_moduli) {
            }
    }

    /* Apply shift-refinement delta using the correct log2P for effective_s.
     * This ensures X_true ≤ M_{effective_s}/4 < M_{effective_s}/2 (CRT safe). */
    const float refine_log2P = h_accu_log2P_all[effective_s - 2u];

    _pstart();
    hipLaunchKernelGGL(oz2_refine_sftA_apply_kernel, dim3(sftA_m_blks), dim3(64), 0, stream,
                       row_max, sftA, m, refine_log2P);
    hipLaunchKernelGGL(oz2_refine_sftB_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, stream,
                       C32i, m, n, ldc32i, sftB, refine_log2P);
    _pstop(_t_refine);  /* partial: refine_sftA_apply + refine_sftB */

    /* ── Scale + Fused/non-fused dispatch ────────────────────────────────────
     * Scale grid/block configuration shared by both fused and non-fused paths.
     * Gate: fused kernel is used when pm.t_fused_ms < pm.t_int8_gemms_ms + pm.t_accum_ms.
     * (Scale always runs — its time is excluded from the gate comparison.)
     *
     * Fused path:  scale → oz2_fused_TN_kernel (reads INT8 A8i/B8i → writes FP64 D)
     * Non-fused:   scale → hipblasLtMatmul (INT8 GEMM) → accum/finalize kernels        */
    const dim3 blk_scale_T(OZ2_SCALE_TILE_K * OZ2_SCALE_COALESC_TILE_M);  /* 512  */
    const dim3 blk_scale_N(OZ2_SCALE_TILE_K * OZ2_SCALE_SHMEM_TILE_M);    /* 1024 */
    const unsigned k_x_blks   = static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K);
    const unsigned k_x_blks_c = static_cast<unsigned>((k + 2u * OZ2_SCALE_TILE_K - 1u) / (2u * OZ2_SCALE_TILE_K));
    const dim3 g_scale_A_T(k_x_blks_c, static_cast<unsigned>((m + OZ2_SCALE_COALESC_TILE_M - 1) / OZ2_SCALE_COALESC_TILE_M));
    const dim3 g_scale_A_N(k_x_blks,   static_cast<unsigned>((m + OZ2_SCALE_SHMEM_TILE_M    - 1) / OZ2_SCALE_SHMEM_TILE_M));
    const dim3 g_scale_B_N(k_x_blks_c, static_cast<unsigned>((n + OZ2_SCALE_COALESC_TILE_M - 1) / OZ2_SCALE_COALESC_TILE_M));
    const dim3 g_scale_B_T(k_x_blks,   static_cast<unsigned>((n + OZ2_SCALE_SHMEM_TILE_M    - 1) / OZ2_SCALE_SHMEM_TILE_M));
    const size_t strideA8i = lda8i * cola8i;
    const size_t strideB8i = ldb8i * static_cast<size_t>(n);

#define OZ2_SCALE_LAUNCH_A(TC, SS) \
    do { if(tA) \
        hipLaunchKernelGGL((oz2_scale_A_T_kernel<(TC)>), g_scale_A_T, blk_scale_T, 0, stream, \
                           A, m, lda, A8i, lda8i, cola8i, sftA, k, (SS)); \
    else \
        hipLaunchKernelGGL((oz2_scale_A_N_kernel<(TC)>), g_scale_A_N, blk_scale_N, 0, stream, \
                           A, m, lda, A8i, lda8i, cola8i, sftA, k, (SS)); \
    } while(0)
#define OZ2_SCALE_LAUNCH_B(TC, SS) \
    do { if(!tB) \
        hipLaunchKernelGGL((oz2_scale_B_N_kernel<(TC)>), g_scale_B_N, blk_scale_T, 0, stream, \
                           B, n, ldb, B8i, ldb8i, sftB, k, (SS)); \
    else \
        hipLaunchKernelGGL((oz2_scale_B_T_kernel<(TC)>), g_scale_B_T, blk_scale_N, 0, stream, \
                           B, n, ldb, B8i, ldb8i, sftB, k, (SS)); \
    } while(0)
    /* ── Scale dispatch lambda ─────────────────────────────────────────────── */
    /* Shared by both fused and non-fused paths to avoid duplicating the
     * 18-case switch.  Launches A and B scale kernels for one chunk and
     * accumulates elapsed time into _t_scale.                                   */
    auto launch_scale_chunk = [&](unsigned sc_start, unsigned sc_count) {
        _pstart();
        switch(sc_count) {
            case  1: OZ2_SCALE_LAUNCH_A( 1, sc_start); OZ2_SCALE_LAUNCH_B( 1, sc_start); break;
            case  2: OZ2_SCALE_LAUNCH_A( 2, sc_start); OZ2_SCALE_LAUNCH_B( 2, sc_start); break;
            case  3: OZ2_SCALE_LAUNCH_A( 3, sc_start); OZ2_SCALE_LAUNCH_B( 3, sc_start); break;
            case  4: OZ2_SCALE_LAUNCH_A( 4, sc_start); OZ2_SCALE_LAUNCH_B( 4, sc_start); break;
            case  5: OZ2_SCALE_LAUNCH_A( 5, sc_start); OZ2_SCALE_LAUNCH_B( 5, sc_start); break;
            case  6: OZ2_SCALE_LAUNCH_A( 6, sc_start); OZ2_SCALE_LAUNCH_B( 6, sc_start); break;
            case  7: OZ2_SCALE_LAUNCH_A( 7, sc_start); OZ2_SCALE_LAUNCH_B( 7, sc_start); break;
            case  8: OZ2_SCALE_LAUNCH_A( 8, sc_start); OZ2_SCALE_LAUNCH_B( 8, sc_start); break;
            case  9: OZ2_SCALE_LAUNCH_A( 9, sc_start); OZ2_SCALE_LAUNCH_B( 9, sc_start); break;
            case 10: OZ2_SCALE_LAUNCH_A(10, sc_start); OZ2_SCALE_LAUNCH_B(10, sc_start); break;
            case 11: OZ2_SCALE_LAUNCH_A(11, sc_start); OZ2_SCALE_LAUNCH_B(11, sc_start); break;
            case 12: OZ2_SCALE_LAUNCH_A(12, sc_start); OZ2_SCALE_LAUNCH_B(12, sc_start); break;
            case 13: OZ2_SCALE_LAUNCH_A(13, sc_start); OZ2_SCALE_LAUNCH_B(13, sc_start); break;
            case 14: OZ2_SCALE_LAUNCH_A(14, sc_start); OZ2_SCALE_LAUNCH_B(14, sc_start); break;
            case 15: OZ2_SCALE_LAUNCH_A(15, sc_start); OZ2_SCALE_LAUNCH_B(15, sc_start); break;
            case 16: OZ2_SCALE_LAUNCH_A(16, sc_start); OZ2_SCALE_LAUNCH_B(16, sc_start); break;
            case 17: OZ2_SCALE_LAUNCH_A(17, sc_start); OZ2_SCALE_LAUNCH_B(17, sc_start); break;
            case 18: OZ2_SCALE_LAUNCH_A(18, sc_start); OZ2_SCALE_LAUNCH_B(18, sc_start); break;
            default: break;
        }
        _pstop(_t_scale);
    };

    bool took_fused_path = false;
    rocblaslt_status fused_st = rocblaslt_status_success;

    {
        const int dev = h->device;
        if (oz2_get_perf_model_params(dev).has_value()) {
            const Fp64PerfModelTimes pm =
                fp64EmulationPerfModelTimes(tA, tB, m, n, k, num_moduli, dev, settings.dynamic_mode);
            /* Gate: fused replaces only INT8 GEMM + accum; scale always runs.
             * Works for all transpose combinations (A8i/B8i always in canonical format).
             * HIPBLASLT_EMULATION_FUSED=off disables the fused path entirely;
             * HIPBLASLT_EMULATION_FUSED=on/force forces it regardless of perf model.
             *
             * fused_single_pass: the fused kernel reads all effective_s moduli from A8i/B8i
             * in one pass.  When scale_chunk_size < effective_s the scale loop ran in
             * multiple passes, each overwriting the beginning of A8i/B8i with a new chunk,
             * so only the last chunk is valid at the point the fused kernel would run.
             * Only use the fused path when all moduli were written in a single scale pass. */
            const bool fused_single_pass = (chunk_size >= effective_s);
            const Oz2FusedMode fused_mode = oz2_fused_mode();
            if (fused_single_pass && fused_mode != Oz2FusedMode::OFF &&
                (fused_mode == Oz2FusedMode::ON ||
                 (pm.t_fused_ms > 0.0 &&
                  pm.t_fused_ms < pm.t_int8_gemms_ms + pm.t_accum_ms))) {
                /* Zero A8i/B8i workspace padding so the fused kernel's double-buffer
                 * prefetch reads zeros beyond k_int. */
                /* Zero workspace padding so the fused kernel's double-buffer prefetch
                 * reads zeros beyond k.  OZ2_FUSED_KBLK_LOAD_MAX is the maximum
                 * KBLK_LOAD across all kernel variants (TILE=16 and TILE=32).      */
                if (static_cast<int>(k) % static_cast<int>(OZ2_FUSED_KBLK_LOAD_MAX) != 0) {
                    (void)hipMemsetAsync(A8i, 0, szA8i, stream);
                    (void)hipMemsetAsync(B8i, 0, szB8i, stream);
                }
                /* Fused path: scale all moduli in chunks, then MFMA+CRT fused kernel. */
                for (unsigned chunk_start = 0; chunk_start < effective_s; chunk_start += chunk_size)
                    launch_scale_chunk(chunk_start, std::min(chunk_size, effective_s - chunk_start));
                /* Fused MFMA+CRT kernel: reads A8i/B8i, writes D directly. */
                _pstart();
                fused_st = oz2_launch_fused_TN(A8i, B8i, lda8i, cola8i, ldb8i,
                                               C, D, m, n, k, ldc, ldd,
                                               *alpha, *beta, sftA, sftB, effective_s, stream);
                _pstop(_t_fused);
                took_fused_path = true;
            }
        }
    }

    if (!took_fused_path) {
    /* Non-fused path: scale + hipblasLtMatmul (INT8 GEMM) + accum/finalize. */
    const dim3 blk_acc(64, 8);
    const dim3 grid_acc((m + 63) / 64, (n + 7) / 8);

    /* Flat single loop: each pass scales `actual` moduli into A8i[0..actual-1]
     * then runs one batched GEMM of batch_count=actual.                       */
    int32_t       batch_cur  = 0;   /* set on first iteration */
    const int64_t stride_A_b = static_cast<int64_t>(strideA8i);
    const int64_t stride_B_b = static_cast<int64_t>(strideB8i);
    const int64_t stride_C_b = static_cast<int64_t>(szC32i);
    hipblasLtMatrixLayoutSetAttribute(layoutA,  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A_b, sizeof(stride_A_b));
    hipblasLtMatrixLayoutSetAttribute(layoutB,  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B_b, sizeof(stride_B_b));
    hipblasLtMatrixLayoutSetAttribute(layoutCD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C_b, sizeof(stride_C_b));

    for(unsigned chunk_start = 0; chunk_start < effective_s; chunk_start += chunk_size) {
        const unsigned actual = std::min(chunk_size, effective_s - chunk_start);

        /* Scale: write actual moduli into A8i[0..actual-1] / B8i[0..actual-1] */
        launch_scale_chunk(chunk_start, actual);

        /* Update batch_count when it changes (normally constant; may differ on
         * the final pass when effective_s is not a multiple of chunk_size).    */
        if(static_cast<int32_t>(actual) != batch_cur) {
            batch_cur = static_cast<int32_t>(actual);
            hipblasLtMatrixLayoutSetAttribute(layoutA,  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
            hipblasLtMatrixLayoutSetAttribute(layoutB,  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
            hipblasLtMatrixLayoutSetAttribute(layoutCD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur));
        }

        _pstart();
        {
            /* A8i and B8i both start at position 0: the scale pass wrote actual
             * moduli into A8i[0..actual-1] / B8i[0..actual-1].                */
            const hipblasStatus_t batch_st =
                hipblasLtMatmul(int8_handle, matmulDesc,
                                &one_i, A8i, layoutA, B8i, layoutB,
                                &zero_i, C32i_batch, layoutCD, C32i_batch, layoutCD,
                                nullptr, int8_ws, int8_ws_size, stream);
            _pstop(_t_int8);
            if(batch_st != HIPBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr,
                        "[hipBLASLt FP64 emulation] WARNING: INT8 batch GEMM failed "
                        "(m=%lld, n=%lld, k=%lld, batch_count=%d, moduli_offset=%u, status=%d). "
                        "Falling back to native DGEMM.\n",
                        (long long)m, (long long)n, (long long)k,
                        batch_cur, chunk_start, (int)batch_st);
                (void)hipStreamSynchronize(stream);
                (void)hipGetLastError();
                oz2_cleanup();
                return rocblaslt_status_internal_error;
            }
        }

        const bool is_first = (chunk_start == 0);
        const bool is_last  = (chunk_start + actual >= effective_s);
        const bool has_lo   = (effective_s > 7u);
        _pstart();
#define OZ2_FARGS C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd, *alpha, *beta, sftA, sftB, chunk_start, effective_s
#define OZ2_AARGS C32i_batch, Zhi, Zlo, m, n, ldc32i, chunk_start, effective_s
#define OZ2_FINALIZE(HL, CS) \
        do { if(is_first) hipLaunchKernelGGL((oz2_accum_finalize_kernel<(HL),(CS),true>),  grid_acc, blk_acc, 0, stream, OZ2_FARGS); \
             else         hipLaunchKernelGGL((oz2_accum_finalize_kernel<(HL),(CS),false>), grid_acc, blk_acc, 0, stream, OZ2_FARGS); } while(0)
#define OZ2_ACCUM(HL, CS) \
        do { if(is_first) hipLaunchKernelGGL((oz2_chunk_accum_kernel<(HL),(CS),true>),  grid_acc, blk_acc, 0, stream, OZ2_AARGS); \
             else         hipLaunchKernelGGL((oz2_chunk_accum_kernel<(HL),(CS),false>), grid_acc, blk_acc, 0, stream, OZ2_AARGS); } while(0)
#define OZ2_DISPATCH(CS) \
        do { \
            if(is_last) { if(has_lo) OZ2_FINALIZE(true,(CS)); else OZ2_FINALIZE(false,(CS)); } \
            else        { if(has_lo) OZ2_ACCUM(true,(CS));    else OZ2_ACCUM(false,(CS));    } \
        } while(0)
        switch(actual) {
            case  1: OZ2_DISPATCH( 1); break; case  2: OZ2_DISPATCH( 2); break;
            case  3: OZ2_DISPATCH( 3); break; case  4: OZ2_DISPATCH( 4); break;
            case  5: OZ2_DISPATCH( 5); break; case  6: OZ2_DISPATCH( 6); break;
            case  7: OZ2_DISPATCH( 7); break; case  8: OZ2_DISPATCH( 8); break;
            case  9: OZ2_DISPATCH( 9); break; case 10: OZ2_DISPATCH(10); break;
            case 11: OZ2_DISPATCH(11); break; case 12: OZ2_DISPATCH(12); break;
            case 13: OZ2_DISPATCH(13); break; case 14: OZ2_DISPATCH(14); break;
            case 15: OZ2_DISPATCH(15); break; case 16: OZ2_DISPATCH(16); break;
            case 17: OZ2_DISPATCH(17); break; case 18: OZ2_DISPATCH(18); break;
        }
#undef OZ2_DISPATCH
#undef OZ2_FARGS
#undef OZ2_AARGS
#undef OZ2_FINALIZE
#undef OZ2_ACCUM
        _pstop(_t_accum);
    }

    } /* end if (!took_fused_path) */

#undef OZ2_SCALE_LAUNCH_B
#undef OZ2_SCALE_LAUNCH_A

    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(layoutCD);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtDestroy(int8_handle);

    if(_prof) {
        /* Accumulate component times into the caller's accumulator. */
        prof->t_prelim      += _t_prelim;
        prof->t_prelim_gemm += _t_prelim_gemm;
        prof->t_extract     += _t_extract;
        prof->t_refine      += _t_refine;
        prof->t_adp         += _t_adp;
        prof->t_fused       += _t_fused;
        prof->t_scale       += _t_scale;
        prof->t_int8        += _t_int8;
        prof->t_accum       += _t_accum;
        prof->t_finalize    += _t_finalize;
        prof->effective_s_used = std::max(prof->effective_s_used, effective_s);
        prof->n_sub_gemms   += 1u;
        (void)hipEventDestroy(_ev1);
        (void)hipEventDestroy(_ev0);
    }
    return took_fused_path ? fused_st : rocblaslt_status_success;
}

/* =========================================================================
 * fp64EmulatedGemm — public wrapper
 *
 * Owns the profiling accumulator.  Records a single HIP event pair around
 * the entire call (including all recursive sub-GEMMs) to measure the true
 * GPU wall-clock time, then writes one summary CSV row with:
 *   – summed component times across all leaf sub-GEMMs
 *   – the measured t_total_ms for the full call
 *   – num_sub_gemms (number of monolithic leaf calls executed)
 * ========================================================================= */
rocblaslt_status fp64EmulatedGemm(hipblasLtHandle_t            handle,
                                  hipblasOperation_t           opA,
                                  hipblasOperation_t           opB,
                                  int64_t                      m,
                                  int64_t                      n,
                                  int64_t                      k,
                                  const double*                alpha,
                                  const double*                A,
                                  int64_t                      lda,
                                  const double*                B,
                                  int64_t                      ldb,
                                  const double*                beta,
                                  const double*                C,
                                  int64_t                      ldc,
                                  double*                      D,
                                  int64_t                      ldd,
                                  hipStream_t                  stream,
                                  const Fp64EmulationSettings& settings)
{
    /* ── Pre-allocate a single workspace for the entire call ──────────────────
     * fp64EmulatedGemmImpl recurses for split shapes; without a pre-allocated
     * buffer each leaf sub-GEMM would do its own hipMallocAsync/hipFreeAsync.
     * Allocating once here and passing it through settings eliminates that
     * overhead (e.g. 16 redundant alloc/free pairs for the 65K square case).
     * If the caller already provided a sufficient workspace we use it as-is.  */
    const _rocblaslt_handle* h = reinterpret_cast<const _rocblaslt_handle*>(handle);
    const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= OZ2_S_MAX)
                                    ? settings.num_moduli : fp64EmulationNumModuli();
    const int device = h->device;
    /* Build a lightweight decision just to communicate dynamic_mode and
     * num_moduli to fp64EmulationWorkspaceSize — only those two fields are
     * consulted by the workspace function.                                  */
    Fp64EmulationDecision ws_decision{};
    ws_decision.dynamic_mode = settings.dynamic_mode;
    ws_decision.num_moduli   = num_moduli;
    const size_t wsNeeded = fp64EmulationWorkspaceSize(h, opA, opB, m, n, k, ws_decision);

    Fp64EmulationSettings effectiveSettings = settings;
    void* ws_toplevel = nullptr;

    if(wsNeeded > 0 &&
       (effectiveSettings.workspace == nullptr ||
        effectiveSettings.workspace_bytes < wsNeeded))
    {
        /* Use hipMalloc (not hipMallocAsync) so that:
         *  (1) The allocation is not affected by any HIP stream error state from
         *      a previous GPU fault on the same stream.  hipMallocAsync fails
         *      immediately when the stream has a sticky error, even with plenty
         *      of free device memory.
         *  (2) The allocation draws from the same device memory pool as the
         *      benchmark's memory_pool<d_memory> (which also uses hipMalloc),
         *      so pool contention is resolved naturally: if device memory is
         *      scarce, the benchmark pool's existing retry logic (pool.clear()
         *      on hipMalloc failure) frees idle matrix buffers.
         *  The synchronous overhead of hipMalloc is negligible compared to the
         *  hundreds-of-millisecond GEMM that follows.                           */
        if(hipMalloc(&ws_toplevel, wsNeeded) != hipSuccess)
            return rocblaslt_status_memory_error;
        effectiveSettings.workspace       = ws_toplevel;
        effectiveSettings.workspace_bytes = wsNeeded;
    }
    /* ────────────────────────────────────────────────────────────────────── */

    const char* const _pf   = oz2_profile_file();
    const bool        _prof = (_pf != nullptr);

    Fp64ProfileAccum accum{};
    hipEvent_t ev_start{}, ev_end{};
    if(_prof) {
        (void)hipEventCreate(&ev_start);
        (void)hipEventCreate(&ev_end);
        (void)hipEventRecord(ev_start, stream);
    }

    const rocblaslt_status st =
        fp64EmulatedGemmImpl(h, opA, opB, m, n, k, alpha, A, lda, B, ldb,
                             beta, C, ldc, D, ldd, stream, effectiveSettings,
                             _prof ? &accum : nullptr);

    /* Release the top-level workspace now that all leaves have finished.    */
    if(ws_toplevel != nullptr)
        (void)hipFree(ws_toplevel);

    if(_prof) {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        /* Use effective_s_used for profiling chunk sizes so the CSV reflects
         * what was actually computed per pass (not the configured maximum).  */
        const unsigned prof_s = accum.effective_s_used ? accum.effective_s_used : num_moduli;
        const unsigned chunk_size       = oz2_compute_chunk_size(m, n, k, prof_s);
        const unsigned scale_chunk_size = chunk_size;
        const bool tA = (opA != HIPBLAS_OP_N);
        const bool tB = (opB != HIPBLAS_OP_N);
        /* Use the split-aware model: each component is the sum across all leaves.
         * t_native_ms remains for the original (m,n,k) problem.           */
        const Fp64PerfModelTimes pm = oz2_effective_perf_model_times(tA, tB, m, n, k, num_moduli, device, settings.dynamic_mode);

        std::FILE* _f = std::fopen(_pf, "a");
        if(_f) {
            if(std::ftell(_f) == 0)
                std::fprintf(_f,
                    "m,n,k,transA,transB,num_moduli,effective_s,scale_chunk_size,gemm_chunk_size,"
                    "workspace_bytes,num_sub_gemms,"
                    "t_prelim_ms,t_prelim_gemm_ms,t_extract_ms,t_refine_ms,"
                    "t_fused_ms,t_adp_ms,t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                    "t_finalize_ms,t_total_ms,"
                    "pred_prelim_ms,pred_prelim_gemm_ms,pred_refine_ms,pred_adp_ms,"
                    "pred_scale_ms,pred_int8_gemm_ms,pred_accum_ms,"
                    "pred_host_ms,pred_launch_ms,pred_fused_ms,pred_total_ms,pred_native_dgemm_ms\n");
            std::fprintf(_f,
                "%lld,%lld,%lld,%c,%c,%u,%u,%u,%u,"
                "%llu,%u,"
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                (long long)m, (long long)n, (long long)k,
                tA ? 'T' : 'N', tB ? 'T' : 'N',
                num_moduli, accum.effective_s_used, scale_chunk_size, chunk_size,
                (unsigned long long)wsNeeded, accum.n_sub_gemms,
                accum.t_prelim, accum.t_prelim_gemm, accum.t_extract, accum.t_refine,
                accum.t_fused, accum.t_adp, accum.t_scale, accum.t_int8, accum.t_accum, accum.t_finalize, t_total,
                pm.t_prelim_ms, pm.t_prelim_gemm_ms, pm.t_refine_ms, pm.t_adp_ms,
                pm.t_scale_ms, pm.t_int8_gemms_ms, pm.t_accum_ms,
                pm.t_host_ms, pm.t_launch_ms, pm.t_fused_ms, pm.t_total_ms, pm.t_native_ms);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}
