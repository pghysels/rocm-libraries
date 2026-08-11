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
#include "handle.h" /* _rocblaslt_handle */
#include "hipblaslt_ostream.hpp" /* hipblaslt_cerr */

#include "hipblaslt/hipblaslt.h"
#include <hip/hip_runtime.h>

#include <atomic> // std::atomic (ADP overflow warning rate-limiter)
#include <cassert> // assert
#include <cmath> // std::log2, std::floor, etc.
#include <cstdio> // std::fopen / std::fprintf / std::fclose / std::ftell
#include <cstdlib> // std::getenv
#include <cstring> // std::strcmp
#include <limits> // std::numeric_limits (for fused kernel model)
#include <optional> // std::optional
#include <unordered_map> // std::unordered_map
#include <utility> // std::index_sequence, std::make_index_sequence

namespace FP64Emulation
{
    /* =========================================================================
     * Tuning constants
     * ========================================================================= */
    /* Workspace reserved for the INT8 GEMM algorithms (preliminary and batch).
     * Passing a non-null workspace allows hipBLASLt to select algorithms that
     * require workspace, which is necessary for very large k values where no
     * zero-workspace INT8 GEMM algorithm is available.
     * 128 MiB matches the default value of HIPBLASLT_TUNING_USER_MAX_WORKSPACE. */
    static constexpr size_t OZ2_INT8_GEMM_WS_BYTES = 128ull << 20; /* 128 MiB */

    /* Total workspace budget per modulus (A8i + B8i + C32i simultaneously resident).
     * chunk × (mn4 + slc) ≤ OZ2_CHUNK_TARGET_BYTES constrains the combined allocation. */
    static constexpr size_t OZ2_CHUNK_TARGET_BYTES = 32ull << 30; /* 32 GiB */

    /* Number of moduli processed per pass: the largest value ≤ s such that
     * the simultaneous allocation of A8i + B8i + C32i fits in the total budget.
     * Scale and GEMM share the same chunk — one scale launch followed by one
     * batched GEMM per pass.  The final pass handles the remainder naturally
     * via min(chunk, effective_s - chunk_start).                             */
    static unsigned compute_chunk_size(int64_t m, int64_t n, int64_t k, unsigned s)
    {
        const size_t mn4    = static_cast<size_t>(m) * static_cast<size_t>(n) * 4u;
        const size_t lda8i  = pad(static_cast<size_t>(k));
        const size_t cola8i = pad(static_cast<size_t>(m));
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
    static unsigned compute_chunk_size_fused(int64_t m, int64_t n, int64_t k, unsigned s)
    {
        const size_t lda8i  = pad(static_cast<size_t>(k));
        const size_t cola8i = pad(static_cast<size_t>(m));
        const size_t slc
            = lda8i * cola8i + lda8i * static_cast<size_t>(n); /* A8i + B8i per modulus */
        size_t chunk = s;
        if(slc > 0u)
            chunk = std::min(chunk, OZ2_CHUNK_TARGET_BYTES / slc);
        return static_cast<unsigned>(std::max(size_t(1u), chunk));
    }

    /* Kernel efficiency factors and latency constants calibrated on MI355X. */
    struct PerfModelKernelEffs
    {
        static constexpr double eff_prelim[2][2] = {{0.697, 0.561}, /* [N][N], [N][T] */
                                                    {0.902, 0.682}}; /* [T][N], [T][T] */
        static constexpr double eff_scale[2][2]  = {{0.595, 0.520}, /* [N][N], [N][T] */
                                                    {0.683, 0.583}}; /* [T][N], [T][T] */
        static constexpr double eff_refine       = 0.586;
        static constexpr double eff_accum        = 0.938;
        static constexpr double eff_fused        = 0.333;
        static constexpr double host_overhead_s = 1.0e-4; /* host overhead (s) = 0.1 ms           */
        static constexpr double latency_kernel_s
            = 5.0e-6; /* GPU kernel scheduling overhead (s)   */
        static constexpr double latency_matmul_s
            = 10.0e-6; /* hipBLASLt matmul launch overhead (s) */
        static constexpr double latency_memset_s
            = 2.0e-6; /* hipMemsetAsync overhead (s)          */
        static constexpr double latency_sync_s = 50.0e-6; /* hipStreamSynchronize cost (s)        */
    };

    struct PerfModelDeviceParams
    {
        double ai;
        double ratio;
        double latency;
    };

    /* Returns the perf-model parameters for the given HIP device, or nullopt if
     * the device is not in the table (in which case emulation should not run). */
    static std::optional<PerfModelDeviceParams> get_perf_model_params(int device)
    {
        static const std::unordered_map<uint32_t, PerfModelDeviceParams> hw_params_by_pci_id = {
            {0x74a0u, {2.63868, 168.411, 3.93e7}},
            {0x74a1u, {2.63868, 168.411, 3.93e7}},
            {0x74a9u, {2.63868, 168.411, 3.93e7}},

            {0x75a0u, {11.2237, 46.4678, 6.08e7}},
            {0x75b0u, {11.2237, 46.4678, 6.08e7}},

            {0x75a3u, {11.3284, 42.7280, 6.82e7}},
            {0x75b3u, {11.3284, 42.7280, 6.82e7}},
        };
        static std::optional<std::optional<PerfModelDeviceParams>> device_params_cache[64];

        if(device < 0 || device >= 64)
            return std::nullopt;
        auto& entry = device_params_cache[device];
        if(!entry)
        {
            int              chip_id = 0;
            const hipError_t attr_err
                = hipDeviceGetAttribute(&chip_id, hipDeviceAttributePciChipId, device);
            const uint32_t pci_device_id = static_cast<uint32_t>(chip_id) & 0xFFFFu;
            auto           it            = hw_params_by_pci_id.find(pci_device_id);
            entry                        = (it != hw_params_by_pci_id.end())
                                               ? std::optional<PerfModelDeviceParams>{it->second}
                                               : std::optional<PerfModelDeviceParams>{};
        }
        return *entry;
    }

    /* =========================================================================
     * Performance-model predicted times
     * Returns all sub-times in milliseconds.  Used both for the profiling CSV
     * and (via comparison of t_total_ms vs t_native_ms) for the performance
     * heuristic in fp64EmulationPerformanceCheck.
     * ========================================================================= */
    struct PerfModelTimes
    {
        double t_prelim_ms; /* prelim kernel (shift + extraction)  */
        double t_prelim_gemm_ms; /* preliminary INT8 GEMM               */
        double t_refine_ms; /* sft-refinement kernels              */
        double t_adp_ms; /* ADP reduce kernels + hipStreamSynchronize (dynamic mode only) */
        double t_scale_ms; /* multi-modulus scaling kernels       */
        double t_int8_gemms_ms; /* all INT8 GEMMs                      */
        double t_accum_ms; /* CRT accumulation / finalize kernels */
        double t_host_ms; /* per-call host overhead              */
        double t_fused_ms; /* fused TN kernel (replaces scale+GEMM+accum when better) */
        double t_total_ms; /* total predicted emulation time      */
        double t_native_ms; /* predicted native FP64 DGEMM time    */
    };

    static PerfModelTimes perf_model_times(bool     tA,
                                           bool     tB,
                                           int64_t  m,
                                           int64_t  n,
                                           int64_t  k,
                                           unsigned num_moduli,
                                           int      device,
                                           bool     dynamic_mode)
    {
        using K           = PerfModelKernelEffs;
        const auto hw_opt = get_perf_model_params(device);
        assert(hw_opt.has_value()
               && "perf_model_times called for a device not in hw_params_by_pci_id");
        const PerfModelDeviceParams& hw = *hw_opt;

        const double c0  = hw.latency / K::latency_matmul_s;
        const double c1  = c0 * hw.ai;
        const double c2  = c1 * hw.ratio;
        const double s   = static_cast<double>(num_moduli);
        const double mn  = static_cast<double>(m) * static_cast<double>(n);
        const double mk  = static_cast<double>(m) * static_cast<double>(k);
        const double kn  = static_cast<double>(k) * static_cast<double>(n);
        const double mnk = mn * static_cast<double>(k);

        const double chunk_sz       = static_cast<double>(compute_chunk_size(m, n, k, num_moduli));
        const double n_chunks       = std::ceil(s / chunk_sz);
        const double n_scale_chunks = n_chunks; /* scale and GEMM share the same chunk */

        const double t_int8_bw = (mk + kn + 4.0 * mn) / c0;
        const double t_prelim_kern
            = ((mk + kn) * 17.0 / c0 + 2.0 * K::latency_kernel_s) / K::eff_prelim[tA][tB];
        const double t_prelim_gemm = std::max(2.0 * mnk / c2, t_int8_bw) + K::latency_matmul_s;
        const double t_refine_kern
            = (mn * 8.0 / c0 + 3.0 * K::latency_kernel_s + K::latency_memset_s) / K::eff_refine;
        const double t_scale_kern = ((mk + kn) * (8.0 * n_scale_chunks + s) / c0
                                     + 2.0 * n_scale_chunks * K::latency_kernel_s)
                                    / K::eff_scale[tA][tB];
        const double t_int8_gemms
            = s * std::max(2.0 * mnk / c2, t_int8_bw) + n_chunks * K::latency_matmul_s;
        const double t_accum_kern
            = (mn * (4.0 * s + 32.0 * n_chunks - 16.0) / c0 + n_chunks * K::latency_kernel_s)
              / K::eff_accum;
        const double t_host = K::host_overhead_s;

        /* Fused TN kernel: reads pre-computed INT8 A8i/B8i from workspace, performs
         * MFMA + CRT accumulation, writes FP64 D directly.  Scale runs separately.  */
        const double t_fused_bw   = (s * (mk + kn) + 16.0 * mn) / c0; /* INT8 + FP64 C/D */
        const double t_fused_int8 = s * 2.0 * mnk / c2; /* MFMA              */
        const double t_fused_fp64 = s * 8.0 * mn / c1; /* CRT accum only    */
        const double t_fused_cmp  = t_fused_int8 + t_fused_fp64;
        const double t_fused
            = std::max(t_fused_bw, t_fused_cmp) / K::eff_fused + K::latency_kernel_s;

        /* Scale always runs.  Fused kernel replaces only GEMM + CRT accum.
         * Only consider the fused time when the fused kernel is not disabled:
         * HIPBLASLT_EMULATION_FUSED=off means the non-fused path is always used,
         * so the gate and split decisions must not assume the fused speedup.    */
        const double t_gemm_accum = (oz2_fused_mode() != Oz2FusedMode::OFF)
                                        ? std::min(t_int8_gemms + t_accum_kern, t_fused)
                                        : t_int8_gemms + t_accum_kern;
        /* ADP (dynamic-mode) overhead: two tiny reduction kernels followed by a
         * hipStreamSynchronize that blocks the CPU until the GPU drains.
         *   adp_reduce_A_kernel reads row_max[m]  (~m × 4 bytes, negligible)
         *   adp_reduce_B_kernel reads col_max[n]  (~n × 4 bytes, negligible)
         *     — col_max[] was precomputed by col_max_kernel (charged to t_refine);
         *       adp_reduce_B_kernel no longer reads the full m×n C32i matrix.
         * Bottleneck is entirely the CPU-GPU hipStreamSynchronize roundtrip.       */
        const double t_adp
            = dynamic_mode
                  ? 2.0 * K::latency_kernel_s /* adp_reduce_A_kernel + adp_reduce_B_kernel */
                        + K::latency_sync_s /* hipStreamSynchronize — dominant cost */
                  : 0.0;
        const double t_total = t_prelim_kern + t_prelim_gemm + t_refine_kern + t_scale_kern
                               + t_gemm_accum + t_host + t_adp;
        const double t_native
            = std::max(2.0 * mnk / c1, 8.0 * (mk + kn + mn) / c0) + K::latency_matmul_s;

        constexpr double s2ms = 1000.0;
        return {t_prelim_kern * s2ms,
                t_prelim_gemm * s2ms,
                t_refine_kern * s2ms,
                t_adp * s2ms,
                t_scale_kern * s2ms,
                t_int8_gemms * s2ms,
                t_accum_kern * s2ms,
                t_host * s2ms,
                t_fused * s2ms,
                t_total * s2ms,
                t_native * s2ms};
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
    static double effective_time_ms(bool     tA,
                                    bool     tB,
                                    int64_t  m,
                                    int64_t  n,
                                    int64_t  k,
                                    unsigned s,
                                    int      device,
                                    bool     dynamic_mode)
    {
        const double t_mono = perf_model_times(tA, tB, m, n, k, s, device, dynamic_mode).t_total_ms;

        const unsigned chunk_sz = compute_chunk_size(m, n, k, s);
        const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
        if(n_chunks > 1u)
        {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n : n / 2;

            /* Both halves are nearly identical in size (differ by at most 1 when
             * m or n is odd), so approximate t_split = 2 × t_half.             */
            const double t_split
                = 2. * effective_time_ms(tA, tB, half_m, half_n, k, s, device, dynamic_mode);
            if(t_split < t_mono)
                return t_split;
        }
        return t_mono;
    }

    /* Returns per-component predicted times summed across ALL leaf sub-GEMMs,
     * mirroring the recursive binary-halving of effective_time_ms.
     * t_native_ms is always set to the top-level (m,n,k) native DGEMM time
     * because native DGEMM does not split.                                    */
    static PerfModelTimes effective_perf_model_times(bool     tA,
                                                     bool     tB,
                                                     int64_t  m,
                                                     int64_t  n,
                                                     int64_t  k,
                                                     unsigned s,
                                                     int      device,
                                                     bool     dynamic_mode)
    {
        PerfModelTimes mono = perf_model_times(tA, tB, m, n, k, s, device, dynamic_mode);

        const unsigned chunk_sz = compute_chunk_size(m, n, k, s);
        const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
        if(n_chunks > 1u)
        {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n : n / 2;

            const double t_split
                = 2. * effective_time_ms(tA, tB, half_m, half_n, k, s, device, dynamic_mode);
            if(t_split < mono.t_total_ms)
            {
                /* Recurse on one half, then double all components.
                 * Both halves are ≈ equal in size so the approximation is exact
                 * when m (or n) is even and negligible otherwise.               */
                PerfModelTimes half = effective_perf_model_times(
                    tA, tB, half_m, half_n, k, s, device, dynamic_mode);
                half.t_prelim_ms *= 2.0;
                half.t_prelim_gemm_ms *= 2.0;
                half.t_refine_ms *= 2.0;
                half.t_adp_ms *= 2.0;
                half.t_scale_ms *= 2.0;
                half.t_int8_gemms_ms *= 2.0;
                half.t_accum_ms *= 2.0;
                half.t_host_ms *= 2.0;
                half.t_fused_ms *= 2.0;
                half.t_total_ms *= 2.0;
                /* Native DGEMM does not split: keep the top-level prediction. */
                half.t_native_ms = mono.t_native_ms;
                return half;
            }
        }
        return mono;
    }

    /* =========================================================================
     * Device helper: warp-level reductions
     * ========================================================================= */
    static __device__ __forceinline__ double warp_reduce_max_abs_d(double val)
    {
        val = fabs(val);
        unsigned long long bits;
        __builtin_memcpy(&bits, &val, 8);
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            unsigned long long other = __shfl_down(bits, off);
            if(other > bits)
                bits = other;
        }
        double res;
        __builtin_memcpy(&res, &bits, 8);
        return res;
    }

    static __device__ __forceinline__ int32_t warp_reduce_max_abs_i32(int32_t val)
    {
        if(val < 0)
            val = -val;
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            int32_t other = __shfl_down(val, off);
            if(other > val)
                val = other;
        }
        return val;
    }

    static __device__ __forceinline__ double block_reduce_max_d(double warp_max,
                                                                double* __restrict__ s_wmax)
    {
        if(threadIdx.x % warpSize == 0)
            s_wmax[threadIdx.x / warpSize] = warp_max;
        __syncthreads();
        double result = 0.0;
        if(threadIdx.x == 0)
        {
            const int nw = (blockDim.x + warpSize - 1) / warpSize;
            result       = s_wmax[0];
            for(int w = 1; w < nw; ++w)
                if(s_wmax[w] > result)
                    result = s_wmax[w];
        }
        return result;
    }

    static __device__ __forceinline__ int32_t block_reduce_max_i32(int32_t warp_max,
                                                                   int32_t* __restrict__ s_wmax)
    {
        if(threadIdx.x % warpSize == 0)
            s_wmax[threadIdx.x / warpSize] = warp_max;
        __syncthreads();
        int32_t result = 0;
        if(threadIdx.x == 0)
        {
            const int nw = (blockDim.x + warpSize - 1) / warpSize;
            result       = s_wmax[0];
            for(int w = 1; w < nw; ++w)
                if(s_wmax[w] > result)
                    result = s_wmax[w];
        }
        return result;
    }

    /* =========================================================================
     * GPU kernels — preliminary shift + INT8 extraction (separate per path)
     *
     * Four kernels, named by the transpose value they handle (_T = transposed,
     * _N = non-transposed), each with optimal blockDim and SHMEM for its path:
     *
     *   accu_prelim_A_T_kernel  TRANS_A=true  (k-fast double2 coalesced, blockDim=256)
     *   accu_prelim_A_N_kernel  TRANS_A=false (SHMEM transposition, blockDim=1024)
     *   accu_prelim_B_N_kernel  TRANS_B=false (j-fast double2 coalesced, blockDim=256)
     *   accu_prelim_B_T_kernel  TRANS_B=true  (SHMEM transposition, blockDim=1024)
     *
     * Coalesced kernels use double2 loads (128-bit), tiny LDS (s_wmax only).
     * SHMEM kernels use TILE_M=16 → 4x fewer blocks/syncs.
     * ========================================================================= */
    static constexpr int OZ2_PRELIM_TILE_K = 64; /* k-tile size for SHMEM paths */
    static constexpr int OZ2_PRELIM_SHMEM_TILE_M
        = 16; /* rows/cols per SHMEM block (blockDim=1024) */
    static constexpr int OZ2_PRELIM_COALESC_THRS = 256; /* threads for coalesced paths */
    static constexpr int OZ2_MIN_WARP_SIZE = 32; /* minimum warpSize across supported devices */

    /* Returns floor(log2(x)) for a positive normalized FP64 value x by extracting
     * the IEEE 754 biased exponent field (bits 52-62) via integer bit ops.
     * Replaces the quarter-rate transcendental log2() + floor() sequence (~50-100
     * cycles on CDNA) with 2-3 full-rate integer instructions (~4-5 cycles).
     * Precondition: x > 0 and x is a normalized FP64 (guaranteed by the
     * < DBL_MIN guard that replaces zero/subnormal row/col maxima with DBL_MIN). */
    static __device__ __forceinline__ int floor_log2_d(double x)
    {
        unsigned long long bits;
        __builtin_memcpy(&bits, &x, 8);
        return static_cast<int>((bits >> 52) & 0x7FFull) - 1023;
    }

    /* ── A_T: TRANS_A=true, k-fast double2 coalesced, blockDim=256, one block per row ──
     * Both passes use double2 loads (128-bit) to process two k-positions per iteration,
     * halving the number of memory transactions versus scalar 64-bit loads.
     * Alignment: j = 2*threadIdx.x is always even; the base A+row*lda is assumed even
     * in double-units (lda is even for any aligned allocation), so double2 is safe.
     * Odd-k tail (at most one element) is handled by thread 0 with a scalar load.  */
    template <bool CHECK_NAN>
    __global__ static void accu_prelim_A_T_kernel(const double* __restrict__ A,
                                                  int64_t m,
                                                  int64_t k,
                                                  int64_t lda,
                                                  int8_t* __restrict__ A8i_high,
                                                  size_t lda8i,
                                                  int16_t* __restrict__ sftA,
                                                  uint32_t* __restrict__ nan_flag)
    {
        __shared__ double  s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE]; /* 8 slots */
        __shared__ int16_t s_sft[1];

        const int64_t       row      = static_cast<int64_t>(blockIdx.x);
        const int64_t       k_even   = k & ~int64_t{1}; /* floor(k/2)*2 — double2 main range */
        const double* const row_base = A + static_cast<size_t>(row) * static_cast<size_t>(lda);

        /* Pass 1: reduce per-row max using double2 loads (2 elements per memory txn). */
        double local_max = 0.0;
        for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
        {
            const double2 vv = *reinterpret_cast<const double2*>(row_base + j); /* COALESCED */
            if constexpr(CHECK_NAN)
            {
                if(!isfinite(vv.x))
                    (void)atomicOr(nan_flag, isinf(vv.x) ? 1u : 2u);
                if(!isfinite(vv.y))
                    (void)atomicOr(nan_flag, isinf(vv.y) ? 1u : 2u);
            }
            const double av0 = fabs(vv.x), av1 = fabs(vv.y);
            if(av0 > local_max)
                local_max = av0;
            if(av1 > local_max)
                local_max = av1;
        }
        /* Scalar tail: last element when k is odd (thread 0 only). */
        if((k & 1) && threadIdx.x == 0)
        {
            const double val = row_base[k - 1];
            if constexpr(CHECK_NAN)
                if(!isfinite(val))
                    (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
            const double av = fabs(val);
            if(av > local_max)
                local_max = av;
        }

        local_max = warp_reduce_max_abs_d(local_max);
        local_max = block_reduce_max_d(local_max, s_wmax);
        if(threadIdx.x == 0)
        {
            if(local_max < std::numeric_limits<double>::min())
                local_max = std::numeric_limits<double>::min();
            s_sft[0]  = static_cast<int16_t>(6 - floor_log2_d(local_max));
            sftA[row] = s_sft[0];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[0]);

        /* Pass 2: scale and extract INT8 using double2 loads. */
        const size_t row_out = static_cast<size_t>(row) * lda8i;
        for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
        {
            const double2 vv = *reinterpret_cast<const double2*>(row_base + j); /* COALESCED */
            A8i_high[row_out + static_cast<size_t>(j)]
                = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.x), sft))));
            A8i_high[row_out + static_cast<size_t>(j) + 1]
                = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.y), sft))));
        }
        /* Scalar tail. */
        if((k & 1) && threadIdx.x == 0)
        {
            const double scaled = ceil(ldexp(fabs(row_base[k - 1]), sft));
            A8i_high[row_out + static_cast<size_t>(k - 1)]
                = static_cast<int8_t>(static_cast<int32_t>(scaled));
        }
    }

    /* ── A_N: TRANS_A=false, SHMEM transposition, blockDim=1024, TILE_M=16 rows/block ── */
    template <bool CHECK_NAN>
    __global__ static void accu_prelim_A_N_kernel(const double* __restrict__ A,
                                                  int64_t m,
                                                  int64_t k,
                                                  int64_t lda,
                                                  int8_t* __restrict__ A8i_high,
                                                  size_t lda8i,
                                                  int16_t* __restrict__ sftA,
                                                  uint32_t* __restrict__ nan_flag)
    {
        static constexpr int TILE_K = OZ2_PRELIM_TILE_K;
        static constexpr int TILE_M = OZ2_PRELIM_SHMEM_TILE_M;
        __shared__ double    shmem[TILE_K][TILE_M + 1]; /* +1 avoids bank conflicts */
        __shared__ int16_t   s_sft[TILE_M];

        const int64_t m_base  = static_cast<int64_t>(blockIdx.x) * TILE_M;
        const int     t       = static_cast<int>(threadIdx.x);
        const int     k_local = t / TILE_M; /* 0..TILE_K-1 */
        const int     m_local = t % TILE_M; /* 0..TILE_M-1 */
        const int64_t i       = m_base + m_local;

        /* Pass 1: each thread accumulates its partial per-row max over all k-tiles */
        double thr_max = 0.0;
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j = k_base + k_local;
            if(i < m && j < k)
            {
                double val = A[i + j * lda]; /* COALESCED */
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > thr_max)
                    thr_max = av;
            }
        }
        shmem[k_local][m_local] = thr_max;
        __syncthreads();
        if(k_local == 0)
        {
            double row_max = 0.0;
            for(int kl = 0; kl < TILE_K; ++kl)
                if(shmem[kl][m_local] > row_max)
                    row_max = shmem[kl][m_local];
            if(row_max < std::numeric_limits<double>::min())
                row_max = std::numeric_limits<double>::min();
            s_sft[m_local] = static_cast<int16_t>(6 - floor_log2_d(row_max));
            if(i < m)
                sftA[i] = s_sft[m_local];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[m_local]);

        /* Pass 2: coalesced loads (m-fast) → SHMEM → coalesced writes (k-fast) */
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j      = k_base + k_local;
            double        scaled = 0.0;
            if(i < m && j < k)
                scaled = ceil(ldexp(fabs(A[i + j * lda]), sft)); /* COALESCED */
            shmem[k_local][m_local] = scaled;
            __syncthreads();
            const int     k_write = t % TILE_K;
            const int     m_write = t / TILE_K;
            const int64_t j_out   = k_base + k_write;
            const int64_t i_out   = m_base + m_write;
            if(i_out < m && j_out < k)
                A8i_high[static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i]
                    = static_cast<int8_t>(static_cast<int32_t>(shmem[k_write][m_write]));
            __syncthreads();
        }
    }

    /* ── B_N: TRANS_B=false, j-fast double2 coalesced, blockDim=256, one block per col ──
     * Mirrors accu_prelim_A_T_kernel: double2 loads halve memory transactions in both
     * passes.  Alignment: j = 2*threadIdx.x is always even; the base B+col*ldb is
     * assumed even in double-units (ldb even for aligned allocations).              */
    template <bool CHECK_NAN>
    __global__ static void accu_prelim_B_N_kernel(const double* __restrict__ B,
                                                  int64_t n,
                                                  int64_t k,
                                                  int64_t ldb,
                                                  int8_t* __restrict__ B8i_high,
                                                  size_t ldb8i,
                                                  int16_t* __restrict__ sftB,
                                                  uint32_t* __restrict__ nan_flag)
    {
        __shared__ double  s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE]; /* 8 slots */
        __shared__ int16_t s_sft[1];

        const int64_t       col      = static_cast<int64_t>(blockIdx.x);
        const int64_t       k_even   = k & ~int64_t{1};
        const double* const col_base = B + static_cast<size_t>(col) * static_cast<size_t>(ldb);

        /* Pass 1: reduce per-col max using double2 loads. */
        double local_max = 0.0;
        for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
        {
            const double2 vv = *reinterpret_cast<const double2*>(col_base + j); /* COALESCED */
            if constexpr(CHECK_NAN)
            {
                if(!isfinite(vv.x))
                    (void)atomicOr(nan_flag, isinf(vv.x) ? 1u : 2u);
                if(!isfinite(vv.y))
                    (void)atomicOr(nan_flag, isinf(vv.y) ? 1u : 2u);
            }
            const double av0 = fabs(vv.x), av1 = fabs(vv.y);
            if(av0 > local_max)
                local_max = av0;
            if(av1 > local_max)
                local_max = av1;
        }
        /* Scalar tail when k is odd (thread 0 only). */
        if((k & 1) && threadIdx.x == 0)
        {
            const double val = col_base[k - 1];
            if constexpr(CHECK_NAN)
                if(!isfinite(val))
                    (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
            const double av = fabs(val);
            if(av > local_max)
                local_max = av;
        }

        local_max = warp_reduce_max_abs_d(local_max);
        local_max = block_reduce_max_d(local_max, s_wmax);
        if(threadIdx.x == 0)
        {
            if(local_max < std::numeric_limits<double>::min())
                local_max = std::numeric_limits<double>::min();
            s_sft[0]  = static_cast<int16_t>(6 - floor_log2_d(local_max));
            sftB[col] = s_sft[0];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[0]);

        /* Pass 2: scale and extract INT8 using double2 loads. */
        const size_t col_out = static_cast<size_t>(col) * ldb8i;
        for(int64_t j = 2LL * threadIdx.x; j < k_even; j += 2LL * blockDim.x)
        {
            const double2 vv = *reinterpret_cast<const double2*>(col_base + j); /* COALESCED */
            B8i_high[col_out + static_cast<size_t>(j)]
                = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.x), sft))));
            B8i_high[col_out + static_cast<size_t>(j) + 1]
                = static_cast<int8_t>(static_cast<int32_t>(ceil(ldexp(fabs(vv.y), sft))));
        }
        /* Scalar tail. */
        if((k & 1) && threadIdx.x == 0)
        {
            const double scaled = ceil(ldexp(fabs(col_base[k - 1]), sft));
            B8i_high[col_out + static_cast<size_t>(k - 1)]
                = static_cast<int8_t>(static_cast<int32_t>(scaled));
        }
    }

    /* ── B_T: TRANS_B=true, SHMEM transposition, blockDim=1024, TILE_M=16 cols/block ── */
    template <bool CHECK_NAN>
    __global__ static void accu_prelim_B_T_kernel(const double* __restrict__ B,
                                                  int64_t n,
                                                  int64_t k,
                                                  int64_t ldb,
                                                  int8_t* __restrict__ B8i_high,
                                                  size_t ldb8i,
                                                  int16_t* __restrict__ sftB,
                                                  uint32_t* __restrict__ nan_flag)
    {
        static constexpr int TILE_K = OZ2_PRELIM_TILE_K;
        static constexpr int TILE_M = OZ2_PRELIM_SHMEM_TILE_M;
        __shared__ double    shmem[TILE_K][TILE_M + 1];
        __shared__ int16_t   s_sft[TILE_M];

        const int64_t n_base  = static_cast<int64_t>(blockIdx.x) * TILE_M;
        const int     t       = static_cast<int>(threadIdx.x);
        const int     k_local = t / TILE_M;
        const int     l_local = t % TILE_M;
        const int64_t col     = n_base + l_local;

        /* Pass 1: accumulate per-col max */
        double thr_max = 0.0;
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j = k_base + k_local;
            if(col < n && j < k)
            {
                double val = B[col + j * ldb]; /* COALESCED */
                if constexpr(CHECK_NAN)
                    if(!isfinite(val))
                        (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > thr_max)
                    thr_max = av;
            }
        }
        shmem[k_local][l_local] = thr_max;
        __syncthreads();
        if(k_local == 0)
        {
            double col_max = 0.0;
            for(int kl = 0; kl < TILE_K; ++kl)
                if(shmem[kl][l_local] > col_max)
                    col_max = shmem[kl][l_local];
            if(col_max < std::numeric_limits<double>::min())
                col_max = std::numeric_limits<double>::min();
            s_sft[l_local] = static_cast<int16_t>(6 - floor_log2_d(col_max));
            if(col < n)
                sftB[col] = s_sft[l_local];
        }
        __syncthreads();
        const int sft = static_cast<int>(s_sft[l_local]);

        /* Pass 2: coalesced loads (col-fast) → SHMEM → coalesced writes (k-fast) */
        for(int64_t k_base = 0; k_base < k; k_base += TILE_K)
        {
            const int64_t j      = k_base + k_local;
            double        scaled = 0.0;
            if(col < n && j < k)
                scaled = ceil(ldexp(fabs(B[col + j * ldb]), sft)); /* COALESCED */
            shmem[k_local][l_local] = scaled;
            __syncthreads();
            const int     k_write = t % TILE_K;
            const int     l_write = t / TILE_K;
            const int64_t j_out   = k_base + k_write;
            const int64_t col_out = n_base + l_write;
            if(col_out < n && j_out < k)
                B8i_high[static_cast<size_t>(j_out) + static_cast<size_t>(col_out) * ldb8i]
                    = static_cast<int8_t>(static_cast<int32_t>(shmem[k_write][l_write]));
            __syncthreads();
        }
    }

    /* =========================================================================
     * GPU kernels — accu mode Part 1: shift refinement from preliminary GEMM
     * ========================================================================= */
    __global__ static void refine_sftA_partial_kernel(const int32_t* __restrict__ C32i,
                                                      int64_t m,
                                                      int64_t n,
                                                      size_t  ldc32i,
                                                      int32_t* __restrict__ row_max)
    {
        const int64_t row
            = static_cast<int64_t>(blockIdx.x) * 64 + static_cast<int64_t>(threadIdx.x);
        const int64_t col_base = static_cast<int64_t>(blockIdx.y) * 64;
        if(row >= m)
            return;
        int32_t       local_max = 0;
        const int64_t col_end   = (col_base + 64 < n) ? col_base + 64 : n;
        for(int64_t col = col_base; col < col_end; ++col)
        {
            int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc32i];
            int32_t av = v < 0 ? -v : v;
            if(av > local_max)
                local_max = av;
        }
        if(local_max > 0)
            atomicMax(row_max + static_cast<size_t>(row), local_max);
    }

    /* col_max_kernel — computes per-column max of |C32i_prelim| and writes col_max[n].
     * Reads C32i in column-major order (one block per column, 256 threads reduce over rows).
     * Avoids LDS atomic contention by using a standard block reduction.
     * Runs in the same stream as refine_sftA_partial_kernel so both C32i reads
     * are pipelined; downstream adp_reduce_B and refine_sftB read only the tiny col_max[n]
     * array (n×4 bytes) instead of the full m×n C32i matrix.
     * Launch: dim3(n) blocks × dim3(256) threads.                                        */
    __global__ static void col_max_kernel(const int32_t* __restrict__ C32i,
                                          int64_t m,
                                          size_t  ldc32i,
                                          int32_t* __restrict__ col_max)
    {
        __shared__ int32_t s_wmax[8];
        const int64_t      col       = static_cast<int64_t>(blockIdx.x);
        int32_t            local_max = 0;
        for(int64_t i = threadIdx.x; i < m; i += blockDim.x)
        {
            int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
            int32_t av = v < 0 ? -v : v;
            if(av > local_max)
                local_max = av;
        }
        local_max = warp_reduce_max_abs_i32(local_max);
        local_max = block_reduce_max_i32(local_max, s_wmax);
        if(threadIdx.x == 0)
            col_max[col] = local_max; /* direct write, no atomicMax needed */
    }

    /* ── ADP (Adaptive Precision) helpers ────────────────────────────────────── */
    /* atomicMax for non-negative floats: IEEE 754 positive floats are totally ordered
     * by their integer bit representation, so int-based atomicMax is correct.
     * The caller must ensure val ≥ 0 (we bias log2P_req values by +200 to guarantee this). */
    static __device__ __forceinline__ void adp_atomicMaxF(float* addr, float val)
    {
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
    }

    /* Computes global max of  (52 − sftA_init[i]) + 0.5·log2(row_max[i]) + 200
     * over all rows i ∈ [0, m).  The +200 bias guarantees a non-negative result
     * (log2P_req_unbiased is in ≈ [−50, 170] for any valid FP64 input).
     * Subtract 200 on the host to recover the actual log2P requirement.
     *
     * Must be launched AFTER refine_sftA_partial_kernel (which fills row_max[])
     * and BEFORE refine_sftA_apply_kernel (which overwrites sftA[]).           */
    __global__ static void adp_reduce_A_kernel(const int32_t* __restrict__ row_max,
                                               const int16_t* __restrict__ sftA_init,
                                               int64_t m,
                                               float* __restrict__ adp_A_out)
    {
        __shared__ float s_wmax[OZ2_PRELIM_COALESC_THRS / OZ2_MIN_WARP_SIZE]; /* 8 slots */
        const int64_t    row
            = static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);

        /* Biased log2P requirement for this row.
         * Skip rows where row_max == 0: those rows have all-zero preliminary inner
         * products (e.g. zero-matrix inputs) and need zero CRT precision (s=2).
         * Using max(row_max,1) would incorrectly return (52 − sftA_init) bits for
         * zero rows because sftA_init=6 is a dummy value set for zero inputs.    */
        float local_val = 0.0f; /* 0.0f = biased −200 = needs no precision */
        if(row < m && row_max[row] > 0)
        {
            const int32_t rm     = row_max[row];
            const float   sftA_f = static_cast<float>(sftA_init[row]);
            local_val            = (52.0f - sftA_f) + 0.5f * log2f(static_cast<float>(rm)) + 200.0f;
        }

        /* Warp-level max reduction. */
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            float other = __shfl_down(local_val, off);
            if(other > local_val)
                local_val = other;
        }

        if(threadIdx.x % warpSize == 0)
            s_wmax[threadIdx.x / warpSize] = local_val;
        __syncthreads();

        if(threadIdx.x == 0)
        {
            float     block_max = 0.0f;
            const int nw        = (blockDim.x + warpSize - 1) / warpSize;
            for(int w = 0; w < nw; ++w)
                if(s_wmax[w] > block_max)
                    block_max = s_wmax[w];
            adp_atomicMaxF(adp_A_out, block_max);
        }
    }

    __global__ static void refine_sftA_apply_kernel(const int32_t* __restrict__ row_max,
                                                    int16_t* __restrict__ sftA,
                                                    int64_t m,
                                                    float   log2P)
    {
        const int64_t row
            = static_cast<int64_t>(blockIdx.x) * 64 + static_cast<int64_t>(threadIdx.x);
        if(row >= m)
            return;
        int32_t max_val = row_max[row];
        if(max_val < 1)
            max_val = 1;
        sftA[row]
            += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(max_val)) + log2P));
    }

    /* adp_reduce_B_kernel — reads the precomputed col_max[j] (NOT the full C32i
     * matrix) to compute the B-side ADP log2P requirement.  col_max[] is filled by
     * refine_sftA_partial_kernel in the same pass that computes row_max[], so
     * C32i is read only ONCE total instead of three times.
     *
     * Each thread handles one column independently — no inter-thread reduction needed.
     * Launch config: ceil(n/256) blocks × 256 threads.                              */
    __global__ static void adp_reduce_B_kernel(const int32_t* __restrict__ col_max,
                                               int64_t n,
                                               const int16_t* __restrict__ sftB_init,
                                               float* __restrict__ adp_B_out)
    {
        const int64_t col
            = static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
        if(col >= n)
            return;
        const int32_t local_max = col_max[col];
        /* Skip zero columns: col_max==0 means all preliminary products for that
         * column are zero, so no CRT precision is needed for it.                 */
        if(local_max > 0)
        {
            const float sftB_f = static_cast<float>(sftB_init[col]);
            const float req_biased
                = (52.0f - sftB_f) + 0.5f * log2f(static_cast<float>(local_max)) + 200.0f;
            adp_atomicMaxF(adp_B_out, req_biased);
        }
    }

    /* refine_sftB_kernel — reads the precomputed col_max[j] (NOT the full C32i
     * matrix) to apply the shift-refinement delta to sftB[col].
     * Launch config: ceil(n/256) blocks × 256 threads.                              */
    __global__ static void refine_sftB_kernel(const int32_t* __restrict__ col_max,
                                              int64_t n,
                                              int16_t* __restrict__ sftB,
                                              float log2P)
    {
        const int64_t col
            = static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
        if(col >= n)
            return;
        int32_t local_max = col_max[col];
        if(local_max < 1)
            local_max = 1;
        sftB[col]
            += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P));
    }

    /* =========================================================================
     * GPU kernels — Part 1f: multi-modulus scaling (separate per path)
     *
     * Four kernels named by the transpose value they handle:
     *   scale_A_T_kernel  TRANS_A=true  (k-fast coalesced, blockDim=512, TILE_M=8)
     *   scale_A_N_kernel  TRANS_A=false (SHMEM transposition, blockDim=256, TILE_M=16)
     *   scale_B_N_kernel  TRANS_B=false (j-fast coalesced, blockDim=512, TILE_M=8)
     *   scale_B_T_kernel  TRANS_B=true  (SHMEM transposition, blockDim=256, TILE_M=16)
     *
     * Coalesced kernels: no SHMEM → low LDS → good latency hiding.
     * SHMEM kernels: TILE_M=16, K_UNROLL=4 → blockDim=256; 8.7 KB LDS per block.
     * ========================================================================= */
    static constexpr int OZ2_SCALE_TILE_K = 64;
    /* Coalesced kernels (A_T, B_N) process 4 k-positions per thread (K_UNROLL=4):
     * effective k-tile per block = OZ2_SCALE_TILE_K * 4 = 256.
     * 32 threads × 4 bytes = 128 bytes = one full HBM cache line per warp.      */
    static constexpr int OZ2_SCALE_COALESC_TILE_M = 8; /* blockDim=512, no SHMEM */
    static constexpr int OZ2_SCALE_SHMEM_TILE_M   = 16; /* TILE_M for SHMEM kernels */
    /* SHMEM kernels (A_N, B_T) also use K_UNROLL=4: each thread loads/stores 4
     * consecutive k-positions, packs 4 int8 bytes as uint32_t per store.
     * blockDim = (TILE_K / 4) * TILE_M = 16 * 16 = 256 threads.
     * LDS bank conflicts drop from 4-way to 2-way.    */
    static constexpr int OZ2_SCALE_SHMEM_BLOCK_DIM
        = (OZ2_SCALE_TILE_K / 4) * OZ2_SCALE_SHMEM_TILE_M; /* 256 */

    /* ── A_T: TRANS_A=true, k-fast coalesced, blockDim=512, TILE_M=8, K_UNROLL=4 ──
     * Each thread processes FOUR adjacent k-positions: j0..j0+3.
     * j0 = blockIdx.x * 4*TILE_K + (t%TILE_K)*4  →  always 4-aligned.
     * Loads:  two double2 reads (j0..j0+1 and j0+2..j0+3), each 128-bit, each aligned.
     * Stores: one uint32_t NT store per modulus (packs INT8[j0..j0+3]) = 128 bytes/warp.
     * Moduli loop is OUTER so nm/im/imf are loaded once per modulus, and only
     * TILE_M=8 NT write-combine buffers are needed simultaneously.               */
    template <unsigned T_COUNT>
    __global__ static void scale_A_T_kernel(const double* __restrict__ A,
                                            int64_t m,
                                            int64_t lda,
                                            int8_t* __restrict__ A8i,
                                            size_t lda8i,
                                            size_t cola8i,
                                            const int16_t* __restrict__ sftA,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M = OZ2_SCALE_COALESC_TILE_M; /* 8 */
        const int            t      = static_cast<int>(threadIdx.x);
        const int64_t        m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        /* j0 is always 4-aligned → both double2 loads and the uint32_t store are aligned. */
        const int64_t j0 = static_cast<int64_t>(blockIdx.x) * (TILE_K * 4)
                           + static_cast<int64_t>(t % TILE_K) * 4;
        const int64_t i = m_base + (t / TILE_K);
        if(i >= m || j0 >= k)
            return;
        const int sft = static_cast<int>(sftA[i]);
        /* Load j0/j0+1 as double2 (j0 is 4-aligned → even → 16-byte aligned). */
        double ival[4];
        {
            const double2 vv = *reinterpret_cast<const double2*>(A + i * lda + j0);
            ival[0]          = trunc(ldexp(vv.x, sft));
            ival[1]          = (j0 + 1 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
        }
        /* Load j0+2/j0+3 as double2 (j0+2 is even → aligned) only when valid. */
        if(j0 + 2 < k)
        {
            const double2 vv = *reinterpret_cast<const double2*>(A + i * lda + j0 + 2);
            ival[2]          = trunc(ldexp(vv.x, sft));
            ival[3]          = (j0 + 3 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
        }
        else
        {
            ival[2] = 0.0;
            ival[3] = 0.0;
        }
        const size_t stride   = lda8i * cola8i;
        const size_t off_base = static_cast<size_t>(i) * lda8i;
        const size_t off0 = static_cast<size_t>(j0) + off_base; /* 4-aligned → uint32_t aligned */
        /* Outer loop over moduli — sequential (no unroll) so nm/im/imf loaded once per modulus,
         * keeping register pressure low and WCB usage at TILE_M=8 entries simultaneously.   */
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
        {
            const unsigned tidx = t_start + t_local;
            const double   nm   = neg_mod(tidx);
            const double   im   = inv_mod(tidx);
            const float    imf  = inv_mod_f(tidx);
            /* Compute INT8 residues for all four k-positions. */
            int8_t b[4];
#pragma unroll
            for(int p = 0; p < 4; ++p)
            {
                const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                const float  rfp = static_cast<float>(rp);
                b[p]             = static_cast<int8_t>(
                    static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
            }
            /* Pack and store: always write 4 bytes as uint32_t.
             * Out-of-bounds b[p] are zero (ival[p] was pre-zeroed for p >= valid count),
             * so the extra bytes land harmlessly in the [k, lda8i) padding region.
             * j0 is always 4-aligned, so dst is always uint32_t-aligned.
             * This eliminates branch divergence in tail k-tiles.              */
            int8_t* const  dst    = A8i + t_local * stride + off0;
            const uint32_t packed = static_cast<uint8_t>(b[0])
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
            __builtin_nontemporal_store(packed, reinterpret_cast<uint32_t*>(dst));
        }
    }

    /* ── A_N: TRANS_A=false, SHMEM transposition, blockDim=256, TILE_M=16, K_UNROLL=4 ──
     * Each thread loads K_UNROLL=4 consecutive k-positions into SHMEM, then in the
     * write phase reads 4 SHMEM entries, computes 4 int8 residues, and stores them as a
     * single uint32_t NT write (4 bytes per store instead of 1).
     * blockDim = (TILE_K / K_UNROLL) * TILE_M = 16 * 16 = 256.
     *
     * Load phase: k_grp = t/TILE_M (0..15), m_loc = t%TILE_M (0..15).
     *   Each thread fills shmem[k_grp*4+p][m_loc] for p=0..3 (4 coalesced column loads).
     * Write phase: k_wb = (t%16)*4 (always 4-aligned), m_wr = t/16 (0..15).
     *   Thread reads 4 shmem entries for the same row, packs 4 int8 → uint32_t.
     *   j_out = blockIdx.x*TILE_K + k_wb is always 4-aligned → uint32_t aligned.
     *   Out-of-bounds shmem entries are pre-zeroed in the load phase (stored 0.0 for j≥k),
     *   so the extra bytes in the padding region [k, lda8i) are harmlessly zero. */
    template <unsigned T_COUNT>
    __global__ static void scale_A_N_kernel(const double* __restrict__ A,
                                            int64_t m,
                                            int64_t lda,
                                            int8_t* __restrict__ A8i,
                                            size_t lda8i,
                                            size_t cola8i,
                                            const int16_t* __restrict__ sftA,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K   = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M   = OZ2_SCALE_SHMEM_TILE_M; /* 16 */
        static constexpr int K_UNROLL = 4;
        /* blockDim = OZ2_SCALE_SHMEM_BLOCK_DIM = (TILE_K/K_UNROLL)*TILE_M = 256 */
        __shared__ double  shmem[TILE_K][TILE_M + 1];
        __shared__ int16_t s_sft[TILE_M];

        const int     t      = static_cast<int>(threadIdx.x);
        const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        const int     k_grp  = t / TILE_M; /* 0..TILE_K/K_UNROLL-1 = 0..15 */
        const int     m_loc  = t % TILE_M; /* 0..TILE_M-1 = 0..15           */
        const int64_t i      = m_base + m_loc;

        /* Load phase: each thread fills K_UNROLL consecutive SHMEM entries.
         * Access A[i + j*lda] (m-fast across m_loc) is coalesced within each
         * group of TILE_M threads sharing the same k_grp.                   */
#pragma unroll
        for(int p = 0; p < K_UNROLL; p++)
        {
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_grp * K_UNROLL + p;
            shmem[k_grp * K_UNROLL + p][m_loc] = (i < m && j < k) ? A[i + j * lda] : 0.0;
        }
        if(k_grp == 0 && i < m)
            s_sft[m_loc] = sftA[i];
        __syncthreads();

        /* Write phase: k_wb is always K_UNROLL-aligned so j_out is always 4-aligned.
         * lda8i = pad(k) is 128-aligned → offset is 4-aligned → uint32_t aligned.
         * Out-of-bounds positions have shmem=0.0 → ival=0.0 → b[p]=0.            */
        const int     k_wb  = (t % (TILE_K / K_UNROLL)) * K_UNROLL; /* 0,4,8,...,60 */
        const int     m_wr  = t / (TILE_K / K_UNROLL); /* 0..15         */
        const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_wb;
        const int64_t i_out = m_base + m_wr;
        if(i_out < m && j_out < k)
        {
            const int sft = static_cast<int>(s_sft[m_wr]);
            /* Read K_UNROLL scaled values from SHMEM; shmem already contains 0.0 for
             * out-of-bounds j≥k positions set during the load phase above.         */
            double ival[K_UNROLL];
#pragma unroll
            for(int p = 0; p < K_UNROLL; p++)
                ival[p] = trunc(ldexp(shmem[k_wb + p][m_wr], sft));

            const size_t stride = lda8i * cola8i;
            const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i;
            /* Outer loop over moduli: nm/im/imf loaded once per modulus, reused for
             * all K_UNROLL=4 k-positions.  Pack 4 int8 bytes as one uint32_t NT store. */
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
            {
                const unsigned tidx = t_start + t_local;
                const double   nm   = neg_mod(tidx);
                const double   im   = inv_mod(tidx);
                const float    imf  = inv_mod_f(tidx);
                int8_t         b[K_UNROLL];
#pragma unroll
                for(int p = 0; p < K_UNROLL; p++)
                {
                    const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                    const float  rfp = static_cast<float>(rp);
                    b[p]             = static_cast<int8_t>(
                        static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
                }
                /* Pack and store: always write 4 bytes as uint32_t.
                 * Out-of-bounds b[p] are zero (ival[p]=0.0 from pre-zeroed shmem),
                 * so extra bytes land harmlessly in the [k, lda8i) padding region. */
                const uint32_t packed = static_cast<uint8_t>(b[0])
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
                __builtin_nontemporal_store(
                    packed, reinterpret_cast<uint32_t*>(A8i + t_local * stride + offset));
            }
        }
    }

    /* ── B_N: TRANS_B=false, j-fast coalesced, blockDim=512, TILE_M=8, K_UNROLL=4 ──
     * Mirrors scale_A_T_kernel: two double2 loads (j0..j0+1 and j0+2..j0+3) + uint32_t packed store. */
    template <unsigned T_COUNT>
    __global__ static void scale_B_N_kernel(const double* __restrict__ B,
                                            int64_t n,
                                            int64_t ldb,
                                            int8_t* __restrict__ B8i,
                                            size_t ldb8i,
                                            const int16_t* __restrict__ sftB,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M = OZ2_SCALE_COALESC_TILE_M; /* 8 */
        const int            t      = static_cast<int>(threadIdx.x);
        const int64_t        n_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        const int64_t        j0     = static_cast<int64_t>(blockIdx.x) * (TILE_K * 4)
                           + static_cast<int64_t>(t % TILE_K) * 4;
        const int64_t col = n_base + (t / TILE_K);
        if(col >= n || j0 >= k)
            return;
        const int sft = static_cast<int>(sftB[col]);
        double    ival[4];
        {
            const double2 vv = *reinterpret_cast<const double2*>(B + col * ldb + j0);
            ival[0]          = trunc(ldexp(vv.x, sft));
            ival[1]          = (j0 + 1 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
        }
        if(j0 + 2 < k)
        {
            const double2 vv = *reinterpret_cast<const double2*>(B + col * ldb + j0 + 2);
            ival[2]          = trunc(ldexp(vv.x, sft));
            ival[3]          = (j0 + 3 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
        }
        else
        {
            ival[2] = 0.0;
            ival[3] = 0.0;
        }
        const size_t stride   = ldb8i * static_cast<size_t>(n);
        const size_t off_base = static_cast<size_t>(col) * ldb8i;
        const size_t off0 = static_cast<size_t>(j0) + off_base; /* 4-aligned → uint32_t aligned */
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
        {
            const unsigned tidx = t_start + t_local;
            const double   nm   = neg_mod(tidx);
            const double   im   = inv_mod(tidx);
            const float    imf  = inv_mod_f(tidx);
            int8_t         b[4];
#pragma unroll
            for(int p = 0; p < 4; ++p)
            {
                const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                const float  rfp = static_cast<float>(rp);
                b[p]             = static_cast<int8_t>(
                    static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
            }
            /* Pack and store: always write 4 bytes as uint32_t.
             * Out-of-bounds b[p] are zero (ival[p] was pre-zeroed for p >= valid count),
             * so the extra bytes land harmlessly in the [k, ldb8i) padding region.
             * j0 is always 4-aligned, so dst is always uint32_t-aligned.
             * This eliminates branch divergence in tail k-tiles.              */
            int8_t* const  dst    = B8i + t_local * stride + off0;
            const uint32_t packed = static_cast<uint8_t>(b[0])
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
            __builtin_nontemporal_store(packed, reinterpret_cast<uint32_t*>(dst));
        }
    }

    /* ── B_T: TRANS_B=true, SHMEM transposition, blockDim=256, TILE_M=16, K_UNROLL=4 ──
     * Mirrors scale_A_N_kernel exactly, but operates on B (col-fast layout).
     * Each thread loads K_UNROLL=4 consecutive k-positions for one column into SHMEM,
     * then in the write phase reads 4 SHMEM entries and packs them as a uint32_t NT store.
     * blockDim = OZ2_SCALE_SHMEM_BLOCK_DIM = (TILE_K/K_UNROLL)*TILE_M = 256.
     *
     * Load phase: k_grp = t/TILE_M (0..15), l_loc = t%TILE_M (0..15).
     *   Each thread fills shmem[k_grp*4+p][l_loc] for p=0..3 (coalesced col-fast loads).
     * Write phase: k_wb = (t%16)*4 (always 4-aligned), l_wr = t/16 (0..15).
     *   Thread reads 4 shmem entries for the same column, packs 4 int8 → uint32_t.
     *   j_out = blockIdx.x*TILE_K + k_wb is always 4-aligned → uint32_t aligned.
     *   Out-of-bounds shmem entries are pre-zeroed in the load phase (stored 0.0 for j≥k),
     *   so extra bytes in the padding region [k, ldb8i) are harmlessly zero.           */
    template <unsigned T_COUNT>
    __global__ static void scale_B_T_kernel(const double* __restrict__ B,
                                            int64_t n,
                                            int64_t ldb,
                                            int8_t* __restrict__ B8i,
                                            size_t ldb8i,
                                            const int16_t* __restrict__ sftB,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K   = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M   = OZ2_SCALE_SHMEM_TILE_M; /* 16 */
        static constexpr int K_UNROLL = 4;
        /* blockDim = OZ2_SCALE_SHMEM_BLOCK_DIM = (TILE_K/K_UNROLL)*TILE_M = 256 */
        __shared__ double  shmem[TILE_K][TILE_M + 1];
        __shared__ int16_t s_sft[TILE_M];

        const int     t      = static_cast<int>(threadIdx.x);
        const int64_t n_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        const int     k_grp  = t / TILE_M; /* 0..TILE_K/K_UNROLL-1 = 0..15 */
        const int     l_loc  = t % TILE_M; /* 0..TILE_M-1 = 0..15           */
        const int64_t col    = n_base + l_loc;

        /* Load phase: each thread fills K_UNROLL consecutive SHMEM entries.
         * Access B[col + j*ldb] (col-fast across l_loc) is coalesced within each
         * group of TILE_M threads sharing the same k_grp.                   */
#pragma unroll
        for(int p = 0; p < K_UNROLL; p++)
        {
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_grp * K_UNROLL + p;
            shmem[k_grp * K_UNROLL + p][l_loc] = (col < n && j < k) ? B[col + j * ldb] : 0.0;
        }
        if(k_grp == 0 && col < n)
            s_sft[l_loc] = sftB[col];
        __syncthreads();

        /* Write phase: k_wb is always K_UNROLL-aligned so j_out is always 4-aligned.
         * ldb8i = pad(k) is 128-aligned → offset is 4-aligned → uint32_t aligned.
         * Out-of-bounds positions have shmem=0.0 → ival=0.0 → b[p]=0.            */
        const int     k_wb  = (t % (TILE_K / K_UNROLL)) * K_UNROLL; /* 0,4,8,...,60 */
        const int     l_wr  = t / (TILE_K / K_UNROLL); /* 0..15         */
        const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_wb;
        const int64_t c_out = n_base + l_wr;
        if(c_out < n && j_out < k)
        {
            const int sft = static_cast<int>(s_sft[l_wr]);
            /* Read K_UNROLL scaled values from SHMEM; shmem already contains 0.0 for
             * out-of-bounds j≥k positions set during the load phase above.         */
            double ival[K_UNROLL];
#pragma unroll
            for(int p = 0; p < K_UNROLL; p++)
                ival[p] = trunc(ldexp(shmem[k_wb + p][l_wr], sft));

            const size_t stride = ldb8i * static_cast<size_t>(n);
            const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(c_out) * ldb8i;
            /* Outer loop over moduli: nm/im/imf loaded once per modulus, reused for
             * all K_UNROLL=4 k-positions.  Pack 4 int8 bytes as one uint32_t NT store. */
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
            {
                const unsigned tidx = t_start + t_local;
                const double   nm   = neg_mod(tidx);
                const double   im   = inv_mod(tidx);
                const float    imf  = inv_mod_f(tidx);
                int8_t         b[K_UNROLL];
#pragma unroll
                for(int p = 0; p < K_UNROLL; p++)
                {
                    const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                    const float  rfp = static_cast<float>(rp);
                    b[p]             = static_cast<int8_t>(
                        static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
                }
                /* Pack and store: always write 4 bytes as uint32_t.
                 * Out-of-bounds b[p] are zero (ival[p]=0.0 from pre-zeroed shmem),
                 * so extra bytes land harmlessly in the [k, ldb8i) padding region. */
                const uint32_t packed = static_cast<uint8_t>(b[0])
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
                __builtin_nontemporal_store(
                    packed, reinterpret_cast<uint32_t*>(B8i + t_local * stride + offset));
            }
        }
    }

    /* =========================================================================
     * GPU kernels — Part 2d: chunked CRT accumulation
     * ========================================================================= */

    template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
    __global__ static void chunk_accum_kernel(const int32_t* __restrict__ C32i_batch,
                                              double* __restrict__ Zhi_out,
                                              double* __restrict__ Zlo_out,
                                              int64_t  m,
                                              int64_t  n,
                                              size_t   ldc32i,
                                              unsigned chunk_start,
                                              unsigned effective_s)
    {
        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        if(i >= m || l >= n)
            return;
        const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
        const size_t slice_stride = ldc32i * static_cast<size_t>(n);
        /* Two-pass: dc_s[CS] caches balanced residues from independent HBM loads.
         * Pass 1: all CS loads + rint+fma issued simultaneously (MLP, ~200cy latency);
         *   FP64 compute overlaps HBM latency. Uses CS×2 VGPRs (~52 for CS=16).
         * Pass 2: pure register TwoSum, loop-carried dep through Zhi only.
         * Note: storing dc as int32 (dc_int_s) was tested — same performance at all
         * sizes (kernel is HBM-bandwidth-limited; higher occupancy does not help). */
        double dc_s[CHUNK_SIZE];
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
            dc_s[t_local]         = fma(neg_mod(t), rint(dc_raw * inv_mod(t)), dc_raw);
        }
        double Zhi = 0.0, Zlo = 0.0;
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc     = dc_s[t_local];
            const double   hi     = dc * qpi_hi(effective_s - 2, t);
            const double   new_hi = Zhi + hi;
            const double   err    = hi - (new_hi - Zhi);
            Zhi                   = new_hi;
            if constexpr(HAS_LO)
                Zlo = fma(dc, qpi_lo(effective_s - 2, t), Zlo + err);
            else
                Zlo += err;
        }
        if constexpr(IS_FIRST_CHUNK)
        {
            __builtin_nontemporal_store(Zhi, Zhi_out + idx);
            __builtin_nontemporal_store(Zlo, Zlo_out + idx);
        }
        else
        {
            const double old_hi = Zhi_out[idx];
            const double s_hi   = old_hi + Zhi;
            const double err    = Zhi - (s_hi - old_hi);
            Zhi_out[idx]        = s_hi;
            Zlo_out[idx] += err + Zlo;
        }
    }

    template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
    __global__ static void accum_finalize_kernel(const int32_t* __restrict__ C32i_batch,
                                                 const double* __restrict__ Zhi_in,
                                                 const double* __restrict__ Zlo_in,
                                                 const double* __restrict__ C,
                                                 double* __restrict__ D,
                                                 int64_t m,
                                                 int64_t n,
                                                 size_t  ldc32i,
                                                 int64_t ldc,
                                                 int64_t ldd,
                                                 double  alpha,
                                                 double  beta,
                                                 const int16_t* __restrict__ sftA,
                                                 const int16_t* __restrict__ sftB,
                                                 unsigned chunk_start,
                                                 unsigned effective_s)
    {
        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        if(i >= m || l >= n)
            return;
        const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
        const size_t slice_stride = ldc32i * static_cast<size_t>(n);
        double       dc_s[CHUNK_SIZE];
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
            dc_s[t_local]         = fma(neg_mod(t), rint(dc_raw * inv_mod(t)), dc_raw);
        }
        double Zhi = 0.0, Zlo = 0.0;
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc     = dc_s[t_local];
            const double   hi     = dc * qpi_hi(effective_s - 2, t);
            const double   new_hi = Zhi + hi;
            const double   err    = hi - (new_hi - Zhi);
            Zhi                   = new_hi;
            if constexpr(HAS_LO)
                Zlo = fma(dc, qpi_lo(effective_s - 2, t), Zlo + err);
            else
                Zlo += err;
        }
        /* IS_FIRST_CHUNK=true: Zhi/Zlo already hold the final value (no prior chunk).
         * IS_FIRST_CHUNK=false: fold the prior Zhi_in/Zlo_in into Zhi/Zlo in-place. */
        if constexpr(!IS_FIRST_CHUNK)
        {
            const double old_hi = Zhi_in[idx];
            const double s_hi   = old_hi + Zhi;
            const double err    = Zhi - (s_hi - old_hi);
            Zhi                 = s_hi;
            Zlo                 = Zlo_in[idx] + err + Zlo;
        }
        const double q = rint((Zhi + Zlo) * inv_P(effective_s - 2));
        const double X = fma(P_lo(effective_s - 2), q, fma(P_hi(effective_s - 2), q, Zhi) + Zlo);
        const int    inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
        const size_t d_idx
            = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
        double d_val = alpha * ldexp(X, inv_sft);
        if(beta != 0.0)
        {
            const size_t c_idx
                = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
            d_val += beta * C[c_idx];
        }
        __builtin_nontemporal_store(d_val, D + d_idx);
    }

    static const char* profile_file()
    {
        static const char* const fn = std::getenv("HIPBLASLT_EMULATION_PROFILE");
        return fn;
    }

    /* =========================================================================
     * native_dgemm_fallback — run a plain FP64 hipblasLtMatmul for a
     * sub-block.  Used by the split path in emulated_gemm_impl when the
     * second half's emulation fails after the first half has already written D,
     * so that D is fully correct without corrupting the first half's output.
     *
     * A fresh hipblasLtHandle is created with emulation explicitly disabled
     * (emulation.enabled=0) so that HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 in
     * the environment does not cause this call to re-enter emulation recursively.
     * The default enabled=-1 means "check env var", which would re-trigger it.
     * ========================================================================= */
    static rocblaslt_status native_dgemm_fallback(const _rocblaslt_handle* h,
                                                  hipblasOperation_t       opA,
                                                  hipblasOperation_t       opB,
                                                  int64_t                  m,
                                                  int64_t                  n,
                                                  int64_t                  k,
                                                  const double*            alpha,
                                                  const double*            A,
                                                  int64_t                  lda,
                                                  const double*            B,
                                                  int64_t                  ldb,
                                                  const double*            beta,
                                                  const double*            C,
                                                  int64_t                  ldc,
                                                  double*                  D,
                                                  int64_t                  ldd,
                                                  hipStream_t              stream)
    {
        hipblasLtHandle_t       fp64_handle = nullptr;
        hipblasLtMatrixLayout_t layoutA     = nullptr;
        hipblasLtMatrixLayout_t layoutB     = nullptr;
        hipblasLtMatrixLayout_t layoutC     = nullptr;
        hipblasLtMatrixLayout_t layoutD     = nullptr;
        hipblasLtMatmulDesc_t   desc        = nullptr;

        auto cleanup = [&]() noexcept {
            if(desc)
                (void)hipblasLtMatmulDescDestroy(desc);
            if(layoutD)
                (void)hipblasLtMatrixLayoutDestroy(layoutD);
            if(layoutC)
                (void)hipblasLtMatrixLayoutDestroy(layoutC);
            if(layoutB)
                (void)hipblasLtMatrixLayoutDestroy(layoutB);
            if(layoutA)
                (void)hipblasLtMatrixLayoutDestroy(layoutA);
            if(fp64_handle)
                (void)hipblasLtDestroy(fp64_handle);
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
        const uint64_t rows_A
            = (opA == HIPBLAS_OP_N) ? static_cast<uint64_t>(m) : static_cast<uint64_t>(k);
        const uint64_t cols_A
            = (opA == HIPBLAS_OP_N) ? static_cast<uint64_t>(k) : static_cast<uint64_t>(m);
        const uint64_t rows_B
            = (opB == HIPBLAS_OP_N) ? static_cast<uint64_t>(k) : static_cast<uint64_t>(n);
        const uint64_t cols_B
            = (opB == HIPBLAS_OP_N) ? static_cast<uint64_t>(n) : static_cast<uint64_t>(k);

        hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_64F, rows_A, cols_A, lda);
        hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_64F, rows_B, cols_B, ldb);
        hipblasLtMatrixLayoutCreate(
            &layoutC, HIP_R_64F, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldc);
        hipblasLtMatrixLayoutCreate(
            &layoutD, HIP_R_64F, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldd);
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F);
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

        const hipblasStatus_t st = hipblasLtMatmul(fp64_handle,
                                                   desc,
                                                   alpha,
                                                   A,
                                                   layoutA,
                                                   B,
                                                   layoutB,
                                                   beta,
                                                   C,
                                                   layoutC,
                                                   D,
                                                   layoutD,
                                                   nullptr,
                                                   nullptr,
                                                   0,
                                                   stream);
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
    struct ProfileAccum
    {
        float    t_prelim      = 0.f;
        float    t_prelim_gemm = 0.f;
        float    t_refine      = 0.f;
        float    t_adp   = 0.f; /* ADP reduce kernels + hipStreamSynchronize (dynamic mode only) */
        float    t_fused = 0.f; /* fused TN kernel (non-zero when fused path taken) */
        float    t_scale = 0.f;
        float    t_int8  = 0.f;
        float    t_accum = 0.f;
        unsigned effective_s_used = 0u; /* ADP: actual s chosen (= num_moduli in fixed mode) */
        unsigned n_sub_gemms      = 0u;
    };

    /* ── Generic index dispatcher ──────────────────────────────────────────────────────── */
    /* Calls fn(integral_constant<unsigned, N>{}) for the unique N in [1, N_MAX]     */
    /* such that actual == N.  Uses a C++17 fold expression to avoid a switch.       */
    template <unsigned N_MAX, typename Fn, std::size_t... Ns>
    static void dispatch_by_index_impl(unsigned actual, Fn&& fn, std::index_sequence<Ns...>)
    {
        (...,
         (void)((actual == Ns + 1u) && (fn(std::integral_constant<unsigned, Ns + 1u>{}), true)));
    }

    template <unsigned N_MAX, typename Fn>
    static void dispatch_by_index(unsigned actual, Fn&& fn)
    {
        dispatch_by_index_impl<N_MAX>(
            actual, std::forward<Fn>(fn), std::make_index_sequence<N_MAX>{});
    }

    /* ── Preliminary shift+extraction kernel launch helper ────────────────────── */
    /* Launches the four accu_prelim kernels for A and B (transpose-aware).  The  */
    /* CHECK_NAN template parameter is true when NaN/Inf detection is active.      */
    template <bool CHECK_NAN>
    static void launch_prelim_kernels(bool          tA,
                                      bool          tB,
                                      const double* A,
                                      int64_t       m,
                                      int64_t       k,
                                      int64_t       lda,
                                      int8_t*       A8i_high,
                                      size_t        lda8i,
                                      int16_t*      sftA,
                                      uint32_t*     nan_flag,
                                      const double* B,
                                      int64_t       n,
                                      int64_t       ldb,
                                      int8_t*       B8i_high,
                                      size_t        ldb8i,
                                      int16_t*      sftB,
                                      hipStream_t   stream)
    {
        const unsigned m_blks_A_T = static_cast<unsigned>(m);
        const unsigned m_blks_A_N
            = static_cast<unsigned>((m + OZ2_PRELIM_SHMEM_TILE_M - 1) / OZ2_PRELIM_SHMEM_TILE_M);
        const unsigned n_blks_B_N = static_cast<unsigned>(n);
        const unsigned n_blks_B_T
            = static_cast<unsigned>((n + OZ2_PRELIM_SHMEM_TILE_M - 1) / OZ2_PRELIM_SHMEM_TILE_M);
        if(tA)
            hipLaunchKernelGGL((accu_prelim_A_T_kernel<CHECK_NAN>),
                               dim3(m_blks_A_T),
                               dim3(OZ2_PRELIM_COALESC_THRS),
                               0,
                               stream,
                               A,
                               m,
                               k,
                               lda,
                               A8i_high,
                               lda8i,
                               sftA,
                               nan_flag);
        else
            hipLaunchKernelGGL((accu_prelim_A_N_kernel<CHECK_NAN>),
                               dim3(m_blks_A_N),
                               dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_SHMEM_TILE_M),
                               0,
                               stream,
                               A,
                               m,
                               k,
                               lda,
                               A8i_high,
                               lda8i,
                               sftA,
                               nan_flag);
        if(!tB)
            hipLaunchKernelGGL((accu_prelim_B_N_kernel<CHECK_NAN>),
                               dim3(n_blks_B_N),
                               dim3(OZ2_PRELIM_COALESC_THRS),
                               0,
                               stream,
                               B,
                               n,
                               k,
                               ldb,
                               B8i_high,
                               ldb8i,
                               sftB,
                               nan_flag);
        else
            hipLaunchKernelGGL((accu_prelim_B_T_kernel<CHECK_NAN>),
                               dim3(n_blks_B_T),
                               dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_SHMEM_TILE_M),
                               0,
                               stream,
                               B,
                               n,
                               k,
                               ldb,
                               B8i_high,
                               ldb8i,
                               sftB,
                               nan_flag);
    }

    /* ── Scale kernel launch helpers ──────────────────────────────────────────── */
    /* Encapsulate the A-matrix and B-matrix scale kernel launches.  The grid    */
    /* is computed from the problem shape and compile-time tile constants; it is  */
    /* not pre-computed in the caller.                                            */
    template <unsigned TC>
    static void launch_scale_A(bool          tA,
                               const double* A,
                               int64_t       m,
                               int64_t       lda,
                               int8_t*       A8i,
                               size_t        lda8i,
                               size_t        cola8i,
                               int16_t*      sftA,
                               int64_t       k,
                               unsigned      sc_start,
                               hipStream_t   stream)
    {
        const unsigned k_c
            = static_cast<unsigned>((k + 4u * OZ2_SCALE_TILE_K - 1u) / (4u * OZ2_SCALE_TILE_K));
        const unsigned k_x = static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K);
        if(tA)
            hipLaunchKernelGGL((scale_A_T_kernel<TC>),
                               dim3(k_c,
                                    static_cast<unsigned>((m + OZ2_SCALE_COALESC_TILE_M - 1)
                                                          / OZ2_SCALE_COALESC_TILE_M)),
                               dim3(OZ2_SCALE_TILE_K * OZ2_SCALE_COALESC_TILE_M),
                               0,
                               stream,
                               A,
                               m,
                               lda,
                               A8i,
                               lda8i,
                               cola8i,
                               sftA,
                               k,
                               sc_start);
        else
            hipLaunchKernelGGL((scale_A_N_kernel<TC>),
                               dim3(k_x,
                                    static_cast<unsigned>((m + OZ2_SCALE_SHMEM_TILE_M - 1)
                                                          / OZ2_SCALE_SHMEM_TILE_M)),
                               dim3(OZ2_SCALE_SHMEM_BLOCK_DIM),
                               0,
                               stream,
                               A,
                               m,
                               lda,
                               A8i,
                               lda8i,
                               cola8i,
                               sftA,
                               k,
                               sc_start);
    }

    template <unsigned TC>
    static void launch_scale_B(bool          tB,
                               const double* B,
                               int64_t       n,
                               int64_t       ldb,
                               int8_t*       B8i,
                               size_t        ldb8i,
                               int16_t*      sftB,
                               int64_t       k,
                               unsigned      sc_start,
                               hipStream_t   stream)
    {
        const unsigned k_c
            = static_cast<unsigned>((k + 4u * OZ2_SCALE_TILE_K - 1u) / (4u * OZ2_SCALE_TILE_K));
        const unsigned k_x = static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K);
        if(!tB)
            hipLaunchKernelGGL((scale_B_N_kernel<TC>),
                               dim3(k_c,
                                    static_cast<unsigned>((n + OZ2_SCALE_COALESC_TILE_M - 1)
                                                          / OZ2_SCALE_COALESC_TILE_M)),
                               dim3(OZ2_SCALE_TILE_K * OZ2_SCALE_COALESC_TILE_M),
                               0,
                               stream,
                               B,
                               n,
                               ldb,
                               B8i,
                               ldb8i,
                               sftB,
                               k,
                               sc_start);
        else
            hipLaunchKernelGGL((scale_B_T_kernel<TC>),
                               dim3(k_x,
                                    static_cast<unsigned>((n + OZ2_SCALE_SHMEM_TILE_M - 1)
                                                          / OZ2_SCALE_SHMEM_TILE_M)),
                               dim3(OZ2_SCALE_SHMEM_BLOCK_DIM),
                               0,
                               stream,
                               B,
                               n,
                               ldb,
                               B8i,
                               ldb8i,
                               sftB,
                               k,
                               sc_start);
    }

    /* ── Accumulate/finalize dispatch helper ─────────────────────────────────── */
    /* Encapsulates the CRT accumulation and finalization kernel dispatch.  The   */
    /* grid is computed from m and n; CS is the compile-time chunk size (1..18). */
    template <unsigned CS>
    static void dispatch_accum_chunk(bool           is_first,
                                     bool           is_last,
                                     bool           has_lo,
                                     const int32_t* C32i_batch,
                                     double*        Zhi,
                                     double*        Zlo,
                                     int64_t        m,
                                     int64_t        n,
                                     size_t         ldc32i,
                                     unsigned       chunk_start,
                                     unsigned       effective_s,
                                     const double*  C,
                                     double*        D,
                                     int64_t        ldc,
                                     int64_t        ldd,
                                     double         alpha,
                                     double         beta,
                                     const int16_t* sftA,
                                     const int16_t* sftB,
                                     hipStream_t    stream)
    {
        const dim3 blk_acc(64, 8);
        const dim3 grid_acc(static_cast<unsigned>((m + 63) / 64),
                            static_cast<unsigned>((n + 7) / 8));
        if(is_last)
        {
            if(has_lo)
            {
                if(is_first)
                    hipLaunchKernelGGL((accum_finalize_kernel<true, CS, true>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((accum_finalize_kernel<true, CS, false>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
            }
            else
            {
                if(is_first)
                    hipLaunchKernelGGL((accum_finalize_kernel<false, CS, true>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((accum_finalize_kernel<false, CS, false>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
            }
        }
        else
        {
            if(has_lo)
            {
                if(is_first)
                    hipLaunchKernelGGL((chunk_accum_kernel<true, CS, true>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((chunk_accum_kernel<true, CS, false>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
            }
            else
            {
                if(is_first)
                    hipLaunchKernelGGL((chunk_accum_kernel<false, CS, true>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((chunk_accum_kernel<false, CS, false>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
            }
        }
    }

    /* ── RAII guard for INT8 GEMM handles and profiling events ───────────────── */
    /* Destroyed automatically on scope exit — covers both normal return and  */
    /* all early-return error paths, eliminating explicit oz2_cleanup() calls. */
    struct Ozaki2Context
    {
        hipblasLtHandle_t       int8_handle = nullptr;
        hipblasLtMatrixLayout_t layoutA     = nullptr;
        hipblasLtMatrixLayout_t layoutB     = nullptr;
        hipblasLtMatrixLayout_t layoutCD    = nullptr;
        hipblasLtMatmulDesc_t   matmulDesc  = nullptr;
        hipEvent_t              ev0         = nullptr;
        hipEvent_t              ev1         = nullptr;

        Ozaki2Context()                                = default;
        Ozaki2Context(const Ozaki2Context&)            = delete;
        Ozaki2Context& operator=(const Ozaki2Context&) = delete;

        ~Ozaki2Context() noexcept
        {
            if(matmulDesc)
                (void)hipblasLtMatmulDescDestroy(matmulDesc);
            if(layoutCD)
                (void)hipblasLtMatrixLayoutDestroy(layoutCD);
            if(layoutB)
                (void)hipblasLtMatrixLayoutDestroy(layoutB);
            if(layoutA)
                (void)hipblasLtMatrixLayoutDestroy(layoutA);
            if(int8_handle)
                (void)hipblasLtDestroy(int8_handle);
            if(ev1)
                (void)hipEventDestroy(ev1);
            if(ev0)
                (void)hipEventDestroy(ev0);
        }
    };

    /* Internal implementation — called recursively during binary-halving.
     * prof != nullptr enables per-component accumulation across all leaves.   */
    static rocblaslt_status emulated_gemm_impl(const _rocblaslt_handle*     h,
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
                                               ProfileAccum*                prof)
    {
        const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= S_MAX)
                                        ? settings.num_moduli
                                        : fp64EmulationNumModuli();
        {
            /* When fused mode is active (HIPBLASLT_EMULATION_FUSED=on/force), use the
             * fused chunk formula (no C32i in workspace budget): chunk_size = S for all
             * practical shapes → n_chunks = 1 → no binary-halving split.
             */
            const bool     fused_forced = (oz2_fused_mode() == Oz2FusedMode::ON);
            const unsigned chunk_sz = fused_forced ? compute_chunk_size_fused(m, n, k, num_moduli)
                                                   : compute_chunk_size(m, n, k, num_moduli);
            const unsigned n_chunks = (num_moduli + chunk_sz - 1u) / chunk_sz;

            if(n_chunks > 1u)
            {
                const bool    split_m = (m >= n);
                const int64_t half_m  = split_m ? m / 2 : m;
                const int64_t half_n  = split_m ? n : n / 2;
                const int64_t m2      = split_m ? (m - m / 2) : m;
                const int64_t n2      = split_m ? n : (n - n / 2);

                const int    device = h->device;
                const bool   tA     = (opA != HIPBLAS_OP_N);
                const bool   tB     = (opB != HIPBLAS_OP_N);
                const double t_mono
                    = perf_model_times(tA, tB, m, n, k, num_moduli, device, settings.dynamic_mode)
                          .t_total_ms;
                const double t_split
                    = 2.
                      * effective_time_ms(
                          tA, tB, half_m, half_n, k, num_moduli, device, settings.dynamic_mode);

                if(t_split < t_mono)
                {
                    {
                        rocblaslt_status st = emulated_gemm_impl(h,
                                                                 opA,
                                                                 opB,
                                                                 half_m,
                                                                 half_n,
                                                                 k,
                                                                 alpha,
                                                                 A,
                                                                 lda,
                                                                 B,
                                                                 ldb,
                                                                 beta,
                                                                 C,
                                                                 ldc,
                                                                 D,
                                                                 ldd,
                                                                 stream,
                                                                 settings,
                                                                 prof);
                        if(st != rocblaslt_status_success)
                            return st;
                    }

                    const double* const    A2  = split_m ? (tA ? A + half_m * lda : A + half_m) : A;
                    const double* const    B2  = split_m ? B : (tB ? B + half_n : B + half_n * ldb);
                    const double* const    C2  = split_m ? C + half_m : C + half_n * ldc;
                    double* const          D2  = split_m ? D + half_m : D + half_n * ldd;
                    const rocblaslt_status st2 = emulated_gemm_impl(h,
                                                                    opA,
                                                                    opB,
                                                                    m2,
                                                                    n2,
                                                                    k,
                                                                    alpha,
                                                                    A2,
                                                                    lda,
                                                                    B2,
                                                                    ldb,
                                                                    beta,
                                                                    C2,
                                                                    ldc,
                                                                    D2,
                                                                    ldd,
                                                                    stream,
                                                                    settings,
                                                                    prof);
                    if(st2 != rocblaslt_status_success)
                    {
                        hipblaslt_cerr
                            << "[hipBLASLt FP64 emulation] WARNING: second-half emulation failed "
                            << "(m=" << m2 << ", n=" << n2 << ", k=" << k << ", st=" << (int)st2
                            << "). "
                            << "Running native DGEMM for second half to preserve first-half output."
                            << std::endl;
                        return native_dgemm_fallback(h,
                                                     opA,
                                                     opB,
                                                     m2,
                                                     n2,
                                                     k,
                                                     alpha,
                                                     A2,
                                                     lda,
                                                     B2,
                                                     ldb,
                                                     beta,
                                                     C2,
                                                     ldc,
                                                     D2,
                                                     ldd,
                                                     stream);
                    }
                    return rocblaslt_status_success;
                }
            }
        }
        /* ── Existing monolithic path ────────────────────────────────────── */

        const bool    _prof = (prof != nullptr);
        Ozaki2Context context;
        float         _t_prelim = 0, _t_prelim_gemm = 0, _t_refine = 0, _t_adp = 0, _t_fused = 0,
              _t_scale = 0, _t_int8 = 0, _t_accum = 0;
        if(_prof)
        {
            (void)hipEventCreate(&context.ev0);
            (void)hipEventCreate(&context.ev1);
        }
        auto _pstart = [&]() noexcept {
            if(_prof)
                (void)hipEventRecord(context.ev0, stream);
        };
        auto _pstop = [&](float& t) noexcept {
            if(_prof)
            {
                (void)hipEventRecord(context.ev1, stream);
                (void)hipStreamSynchronize(stream);
                float ms = 0.f;
                (void)hipEventElapsedTime(&ms, context.ev0, context.ev1);
                t += ms;
            }
        };

        const bool     fused_forced = (oz2_fused_mode() == Oz2FusedMode::ON);
        const unsigned layout_moduli
            = fused_forced ? num_moduli : (settings.dynamic_mode ? S_MAX : num_moduli);
        const unsigned chunk_size = fused_forced ? compute_chunk_size_fused(m, n, k, layout_moduli)
                                                 : compute_chunk_size(m, n, k, layout_moduli);

        const size_t lda8i  = pad(static_cast<size_t>(k));
        const size_t cola8i = pad(static_cast<size_t>(m));
        const size_t ldb8i  = lda8i;
        const size_t ldc32i = cola8i;
        const size_t padn   = pad(static_cast<size_t>(n));
        const size_t szC32i = ldc32i * static_cast<size_t>(n);

        const size_t szA8i        = chunk_size * lda8i * cola8i;
        const size_t szB8i        = chunk_size * ldb8i * static_cast<size_t>(n);
        const size_t szZhi        = (chunk_size < layout_moduli) ? szC32i : 0u;
        const size_t szZlo        = szZhi;
        const size_t n_c32i_slots = fused_forced ? 1u : static_cast<size_t>(chunk_size);
        const size_t szSftA       = cola8i;
        const size_t szSftB       = padn;
        const size_t szNanFlag    = 1;
        const size_t szRowMax     = cola8i;
        const size_t szColMax     = padn;

        char* const ws = static_cast<char*>(settings.workspace);

        int8_t* const    A8i        = reinterpret_cast<int8_t*>(ws);
        int8_t* const    B8i        = A8i + szA8i;
        int32_t* const   C32i_batch = reinterpret_cast<int32_t*>(B8i + szB8i);
        double* const    Zhi        = reinterpret_cast<double*>(C32i_batch + n_c32i_slots * szC32i);
        double* const    Zlo        = Zhi + szZhi;
        int16_t* const   sftA       = reinterpret_cast<int16_t*>(Zlo + szZlo);
        int16_t* const   sftB       = sftA + szSftA;
        uint32_t* const  nan_flag   = reinterpret_cast<uint32_t*>(sftB + szSftB);
        int32_t* const   row_max    = reinterpret_cast<int32_t*>(nan_flag + szNanFlag);
        int32_t* const   col_max    = row_max + szRowMax;
        float* const     adp_buf    = reinterpret_cast<float*>(col_max + szColMax);
        void* const      int8_ws    = static_cast<void*>(adp_buf + 2);
        constexpr size_t int8_ws_size = OZ2_INT8_GEMM_WS_BYTES;
        int32_t* const   C32i         = C32i_batch;

        int8_t* const A8i_high = A8i;
        int8_t* const B8i_high = B8i;

        const bool tA = (opA != HIPBLAS_OP_N);
        const bool tB = (opB != HIPBLAS_OP_N);

        const uint32_t svmask
            = (settings.sv_mask != ~0u) ? settings.sv_mask : fp64EmulationSpecialValuesMask();

        if(svmask != 0u)
        {
            if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess)
            {
                return rocblaslt_status_internal_error;
            }
        }

        if(hipblasLtCreate(&context.int8_handle) != HIPBLAS_STATUS_SUCCESS)
        {
            return rocblaslt_status_internal_error;
        }
        hipblasLtMatrixLayoutCreate(&context.layoutA,
                                    HIP_R_8I,
                                    static_cast<uint64_t>(k),
                                    static_cast<uint64_t>(m),
                                    static_cast<int64_t>(lda8i));
        hipblasLtMatrixLayoutCreate(&context.layoutB,
                                    HIP_R_8I,
                                    static_cast<uint64_t>(k),
                                    static_cast<uint64_t>(n),
                                    static_cast<int64_t>(ldb8i));
        hipblasLtMatrixLayoutCreate(&context.layoutCD,
                                    HIP_R_32I,
                                    static_cast<uint64_t>(m),
                                    static_cast<uint64_t>(n),
                                    static_cast<int64_t>(ldc32i));
        hipblasLtMatmulDescCreate(&context.matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
        {
            hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(
                context.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
            hipblasLtMatmulDescSetAttribute(
                context.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        }

        const int32_t one_i = 1, zero_i = 0;

        /* Preliminary shift + extraction: separate A and B kernels.
         * Coalesced paths (TRANS_A=T → A_T, TRANS_B=N → B_N): blockDim=256, 1 block/row or col.
         * SHMEM paths    (TRANS_A=N → A_N, TRANS_B=T → B_T): blockDim=1024, TILE_M=16.    */
        _pstart();
        if(svmask == 0u)
            launch_prelim_kernels<false>(tA,
                                         tB,
                                         A,
                                         m,
                                         k,
                                         lda,
                                         A8i_high,
                                         lda8i,
                                         sftA,
                                         nan_flag,
                                         B,
                                         n,
                                         ldb,
                                         B8i_high,
                                         ldb8i,
                                         sftB,
                                         stream);
        else
            launch_prelim_kernels<true>(tA,
                                        tB,
                                        A,
                                        m,
                                        k,
                                        lda,
                                        A8i_high,
                                        lda8i,
                                        sftA,
                                        nan_flag,
                                        B,
                                        n,
                                        ldb,
                                        B8i_high,
                                        ldb8i,
                                        sftB,
                                        stream);
        _pstop(_t_prelim);

        if(svmask != 0u)
        {
            if(hipStreamSynchronize(stream) != hipSuccess)
            {
                return rocblaslt_status_internal_error;
            }
            uint32_t detected = 0u;
            if(hipMemcpy(&detected, nan_flag, sizeof(uint32_t), hipMemcpyDeviceToHost)
               != hipSuccess)
            {
                return rocblaslt_status_internal_error;
            }
            if(detected & svmask)
            {
                return rocblaslt_status_invalid_value;
            }
        }

        /* Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high */
        _pstart();
        {
            const hipblasStatus_t prelim_st = hipblasLtMatmul(context.int8_handle,
                                                              context.matmulDesc,
                                                              &one_i,
                                                              A8i_high,
                                                              context.layoutA,
                                                              B8i_high,
                                                              context.layoutB,
                                                              &zero_i,
                                                              C32i,
                                                              context.layoutCD,
                                                              C32i,
                                                              context.layoutCD,
                                                              nullptr,
                                                              int8_ws,
                                                              int8_ws_size,
                                                              stream);
            _pstop(_t_prelim_gemm);
            if(prelim_st != HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cerr
                    << "[hipBLASLt FP64 emulation] WARNING: preliminary INT8 GEMM failed "
                    << "(m=" << m << ", n=" << n << ", k=" << k << ", status=" << (int)prelim_st
                    << "). " << "Falling back to native DGEMM." << std::endl;
                /* Drain the stream before clearing the error so that the stream is in
                 * a clean state for future operations on this stream/context.       */
                (void)hipStreamSynchronize(stream);
                (void)hipGetLastError();
                return rocblaslt_status_internal_error;
            }
        }

        const float accu_log2P = log2P[num_moduli - 2];
        _pstart();
        const unsigned sftA_m_blks = static_cast<unsigned>((m + 63) / 64);
        const unsigned sftA_n_blks = static_cast<unsigned>((n + 63) / 64);
        /* adp_buf: 2 floats used by ADP (Adaptive Precision) kernels (see above).
         * Initialised to 0.0f (= biased −200 → effective_s=2 if all rows/cols zero). */

        /* effective_s: ADP may reduce this below num_moduli; determined BEFORE the
         * shift-refinement delta is applied so we can use the correct log2P.       */
        unsigned effective_s = num_moduli;

        (void)hipMemsetAsync(row_max, 0, szRowMax * sizeof(int32_t), stream);
        /* col_max is written directly by col_max_kernel (no atomicMax → no memset needed). */
        hipLaunchKernelGGL(refine_sftA_partial_kernel,
                           dim3(sftA_m_blks, sftA_n_blks),
                           dim3(64),
                           0,
                           stream,
                           C32i,
                           m,
                           n,
                           ldc32i,
                           row_max);
        /* Compute col_max[n] from C32i in a separate kernel (column-major read, no LDS contention).
         * Runs concurrently with the above in the same stream; both read the same C32i buffer.
         * adp_reduce_B and refine_sftB then read col_max instead of C32i. */
        hipLaunchKernelGGL(col_max_kernel,
                           dim3(static_cast<unsigned>(n)),
                           dim3(256),
                           0,
                           stream,
                           C32i,
                           m,
                           ldc32i,
                           col_max);
        _pstop(_t_refine); /* partial: refine_sftA_partial only */

        /* ADP (dynamic mode): determine effective_s from the preliminary GEMM result
         * BEFORE applying any shift delta.  Both ADP kernels read sftA_init and
         * sftB_init (the values set in step 1a/1b, before any += delta).
         * The refine delta is then applied using log2P_{effective_s} so that
         * X_true is sized to fit within M_{effective_s}/2, not M_{num_moduli}/2.  */
        if(settings.dynamic_mode)
        {
            _pstart();
            (void)hipMemsetAsync(adp_buf, 0, 2 * sizeof(float), stream);
            /* A-side: reads row_max[] and sftA[] before apply_kernel modifies sftA. */
            hipLaunchKernelGGL(adp_reduce_A_kernel,
                               dim3(sftA_m_blks),
                               dim3(OZ2_PRELIM_COALESC_THRS),
                               0,
                               stream,
                               row_max,
                               sftA,
                               m,
                               adp_buf + 0);
            /* B-side: reads precomputed col_max[] (not C32i) before refine_sftB_kernel modifies sftB. */
            hipLaunchKernelGGL(adp_reduce_B_kernel,
                               dim3((static_cast<unsigned>(n) + 255u) / 256u),
                               dim3(256),
                               0,
                               stream,
                               col_max,
                               n,
                               sftB,
                               adp_buf + 1);

            /* Sync, copy 2 floats, compute effective_s on host.
             * hipGetLastError() clears any sticky thread-level error that may have
             * been set by the ADP kernels (e.g. from a previous GPU fault on the
             * same stream).  Without this, a subsequent hipMallocAsync on the same
             * stream may fail even though the device has plenty of free memory.   */
            (void)hipStreamSynchronize(stream);
            (void)hipGetLastError();
            _pstop(_t_adp); /* ADP reduce kernels + hipStreamSynchronize */
            float h_adp[2] = {0.0f, 0.0f};
            (void)hipMemcpy(h_adp, adp_buf, 2 * sizeof(float), hipMemcpyDeviceToHost);

            const float log2P_needed = std::max(h_adp[0], h_adp[1]) - 200.0f;

            if(log2P_needed > log2P[S_MAX - 2u])
            {
                /* ADP determined s=S_MAX is still insufficient for this input.
                 * This occurs for matrices with extremely large dynamic range within
                 * a single row/column (e.g. condition number ≫ 2^{2×log2P_18}).
                 * The Ozaki shift-refinement would set A8i_final or B8i_final to
                 * near-zero for the elements providing cancellation, giving wrong
                 * results regardless of s.  Fall back to native DGEMM.
                 * Rate-limited warning (≤5 per process).                         */
                hipblaslt_cerr << "[hipBLASLt FP64 emulation] WARNING: ADP overflow for GEMM "
                               << "(m=" << m << ", n=" << n << ", k=" << k
                               << "): " << "A-side log2P_req=" << (h_adp[0] - 200.0f) << " bits, "
                               << "B-side=" << (h_adp[1] - 200.0f) << " bits, "
                               << "max required=" << log2P_needed
                               << " > supported max=" << log2P[S_MAX - 2u] << " (s=" << S_MAX
                               << " moduli, ~" << cum_bits[S_MAX - 2u]
                               << " cumulative bits). Falling back to native DGEMM." << std::endl;
                return rocblaslt_status_invalid_value;
            }

            for(unsigned s = 2u; s <= S_MAX; ++s)
            {
                if(log2P[s - 2u] >= log2P_needed)
                {
                    effective_s = s;
                    break;
                }
            }
        }

        /* Apply shift-refinement delta using the correct log2P for effective_s.
         * This ensures X_true ≤ M_{effective_s}/4 < M_{effective_s}/2 (CRT safe). */
        const float refine_log2P = log2P[effective_s - 2u];

        _pstart();
        hipLaunchKernelGGL(refine_sftA_apply_kernel,
                           dim3(sftA_m_blks),
                           dim3(64),
                           0,
                           stream,
                           row_max,
                           sftA,
                           m,
                           refine_log2P);
        hipLaunchKernelGGL(refine_sftB_kernel,
                           dim3((static_cast<unsigned>(n) + 255u) / 256u),
                           dim3(256),
                           0,
                           stream,
                           col_max,
                           n,
                           sftB,
                           refine_log2P);
        _pstop(_t_refine); /* partial: refine_sftA_apply + refine_sftB */

        /* ── Scale + Fused/non-fused dispatch ────────────────────────────────────────────
         * Gate: fused kernel is used when pm.t_fused_ms < pm.t_int8_gemms_ms + pm.t_accum_ms.
         * (Scale always runs — its time is excluded from the gate comparison.)
         *
         * Fused path:  scale → oz2_fused_TN_kernel (reads INT8 A8i/B8i → writes FP64 D)
         * Non-fused:   scale → hipblasLtMatmul (INT8 GEMM) → accum/finalize kernels        */
        const size_t strideA8i = lda8i * cola8i;
        const size_t strideB8i = ldb8i * static_cast<size_t>(n);

        /* ── Scale dispatch lambda ─────────────────────────────────────────────── */
        /* Shared by both fused and non-fused paths to avoid duplicating the
         * 18-case switch.  Launches A and B scale kernels for one chunk and
         * accumulates elapsed time into _t_scale.                                   */
        auto launch_scale_chunk = [&](unsigned sc_start, unsigned sc_count) {
            _pstart();
            dispatch_by_index<S_MAX>(sc_count, [&](auto Count) {
                constexpr unsigned TC = Count.value;
                launch_scale_A<TC>(tA, A, m, lda, A8i, lda8i, cola8i, sftA, k, sc_start, stream);
                launch_scale_B<TC>(tB, B, n, ldb, B8i, ldb8i, sftB, k, sc_start, stream);
            });

            _pstop(_t_scale);
        };

        bool             took_fused_path = false;
        rocblaslt_status fused_st        = rocblaslt_status_success;

        {
            const int dev = h->device;
            if(get_perf_model_params(dev).has_value())
            {
                const PerfModelTimes pm
                    = perf_model_times(tA, tB, m, n, k, num_moduli, dev, settings.dynamic_mode);
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
                const bool         fused_single_pass = (chunk_size >= effective_s);
                const Oz2FusedMode fused_mode        = oz2_fused_mode();
                if(fused_single_pass && fused_mode != Oz2FusedMode::OFF
                   && (fused_mode == Oz2FusedMode::ON
                       || (pm.t_fused_ms > 0.0
                           && pm.t_fused_ms < pm.t_int8_gemms_ms + pm.t_accum_ms)))
                {
                    /* Zero ONLY the K-padding tail [k, lda8i) of each row so the fused
                     * kernel's double-buffer prefetch reads zeros beyond k_int.  The
                     * scale kernels write columns [0, k); the padding columns [k, lda8i)
                     * (lda8i = pad(k), a 128-byte multiple) are what the DTL prefetch
                     * over-reads and must be zero.
                     *
                     * Previously this zeroed the ENTIRE A8i/B8i workspace
                     * (chunk_size × lda8i × cola8i and chunk_size × ldb8i × n) whenever
                     * k was not a multiple of OZ2_FUSED_KBLK_LOAD_MAX.  Since the INT8
                     * arrays are 128-byte aligned/padded (ALIGN=128) with a
                     * contiguous per-modulus slab layout, all moduli's rows form one
                     * contiguous (rows × pitch) block, so a single 2D memset of just the
                     * tail band (width = lda8i - k, height = rows) zeros every modulus's
                     * padding in one launch — moving far less bandwidth for non-256-K
                     * (e.g. k=1000 → 24/1024 bytes/row instead of the full row).
                     * Guard unchanged: when k is a multiple of KBLK_LOAD_MAX the DTL
                     * prefetch never over-reads, so no zeroing is needed.               */
                    if(static_cast<int>(k) % static_cast<int>(OZ2_FUSED_KBLK_LOAD_MAX) != 0)
                    {
                        const size_t tail_w = lda8i - static_cast<size_t>(k); /* == ldb8i - k */
                        if(tail_w > 0u)
                        {
                            /* A8i: rows = cola8i × chunk_size (contiguous slabs), pitch lda8i,
                             * zero columns [k, lda8i). */
                            (void)hipMemset2DAsync(A8i + static_cast<size_t>(k),
                                                   lda8i,
                                                   0,
                                                   tail_w,
                                                   cola8i * static_cast<size_t>(chunk_size),
                                                   stream);
                            /* B8i: rows = n × chunk_size, pitch ldb8i, zero columns [k, ldb8i). */
                            (void)hipMemset2DAsync(B8i + static_cast<size_t>(k),
                                                   ldb8i,
                                                   0,
                                                   tail_w,
                                                   static_cast<size_t>(n)
                                                       * static_cast<size_t>(chunk_size),
                                                   stream);
                        }
                    }
                    /* Fused path: scale all moduli in chunks, then MFMA+CRT fused kernel. */
                    for(unsigned chunk_start = 0; chunk_start < effective_s;
                        chunk_start += chunk_size)
                        launch_scale_chunk(chunk_start,
                                           std::min(chunk_size, effective_s - chunk_start));
                    /* Fused MFMA+CRT kernel: reads A8i/B8i, writes D directly. */
                    _pstart();
                    fused_st = oz2_launch_fused_TN(A8i,
                                                   B8i,
                                                   lda8i,
                                                   cola8i,
                                                   ldb8i,
                                                   C,
                                                   D,
                                                   m,
                                                   n,
                                                   k,
                                                   ldc,
                                                   ldd,
                                                   *alpha,
                                                   *beta,
                                                   sftA,
                                                   sftB,
                                                   effective_s,
                                                   stream);
                    _pstop(_t_fused);
                    took_fused_path = true;
                }
            }
        }

        if(!took_fused_path)
        {
            /* Non-fused path: scale + hipblasLtMatmul (INT8 GEMM) + accum/finalize. */
            /* Flat single loop: each pass scales `actual` moduli into A8i[0..actual-1]
             * then runs one batched GEMM of batch_count=actual.                       */
            int32_t       batch_cur  = 0; /* set on first iteration */
            const int64_t stride_A_b = static_cast<int64_t>(strideA8i);
            const int64_t stride_B_b = static_cast<int64_t>(strideB8i);
            const int64_t stride_C_b = static_cast<int64_t>(szC32i);
            hipblasLtMatrixLayoutSetAttribute(context.layoutA,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_A_b,
                                              sizeof(stride_A_b));
            hipblasLtMatrixLayoutSetAttribute(context.layoutB,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_B_b,
                                              sizeof(stride_B_b));
            hipblasLtMatrixLayoutSetAttribute(context.layoutCD,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_C_b,
                                              sizeof(stride_C_b));

            for(unsigned chunk_start = 0; chunk_start < effective_s; chunk_start += chunk_size)
            {
                const unsigned actual = std::min(chunk_size, effective_s - chunk_start);

                /* Scale: write actual moduli into A8i[0..actual-1] / B8i[0..actual-1] */
                launch_scale_chunk(chunk_start, actual);

                /* Update batch_count when it changes (normally constant; may differ on
                 * the final pass when effective_s is not a multiple of chunk_size).    */
                if(static_cast<int32_t>(actual) != batch_cur)
                {
                    batch_cur = static_cast<int32_t>(actual);
                    hipblasLtMatrixLayoutSetAttribute(context.layoutA,
                                                      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                      &batch_cur,
                                                      sizeof(batch_cur));
                    hipblasLtMatrixLayoutSetAttribute(context.layoutB,
                                                      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                      &batch_cur,
                                                      sizeof(batch_cur));
                    hipblasLtMatrixLayoutSetAttribute(context.layoutCD,
                                                      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                      &batch_cur,
                                                      sizeof(batch_cur));
                }

                _pstart();
                {
                    /* A8i and B8i both start at position 0: the scale pass wrote actual
                     * moduli into A8i[0..actual-1] / B8i[0..actual-1].                */
                    const hipblasStatus_t batch_st = hipblasLtMatmul(context.int8_handle,
                                                                     context.matmulDesc,
                                                                     &one_i,
                                                                     A8i,
                                                                     context.layoutA,
                                                                     B8i,
                                                                     context.layoutB,
                                                                     &zero_i,
                                                                     C32i_batch,
                                                                     context.layoutCD,
                                                                     C32i_batch,
                                                                     context.layoutCD,
                                                                     nullptr,
                                                                     int8_ws,
                                                                     int8_ws_size,
                                                                     stream);
                    _pstop(_t_int8);
                    if(batch_st != HIPBLAS_STATUS_SUCCESS)
                    {
                        hipblaslt_cerr
                            << "[hipBLASLt FP64 emulation] WARNING: INT8 batch GEMM failed "
                            << "(m=" << m << ", n=" << n << ", k=" << k
                            << ", batch_count=" << batch_cur << ", moduli_offset=" << chunk_start
                            << ", status=" << (int)batch_st << "). "
                            << "Falling back to native DGEMM." << std::endl;
                        (void)hipStreamSynchronize(stream);
                        (void)hipGetLastError();
                        return rocblaslt_status_internal_error;
                    }
                }

                const bool is_first = (chunk_start == 0);
                const bool is_last  = (chunk_start + actual >= effective_s);
                const bool has_lo   = (effective_s > 7u);
                _pstart();
                dispatch_by_index<S_MAX>(actual, [&](auto Count) {
                    dispatch_accum_chunk<Count.value>(is_first,
                                                      is_last,
                                                      has_lo,
                                                      C32i_batch,
                                                      Zhi,
                                                      Zlo,
                                                      m,
                                                      n,
                                                      ldc32i,
                                                      chunk_start,
                                                      effective_s,
                                                      C,
                                                      D,
                                                      ldc,
                                                      ldd,
                                                      *alpha,
                                                      *beta,
                                                      sftA,
                                                      sftB,
                                                      stream);
                });

                _pstop(_t_accum);
            }

        } /* end if (!took_fused_path) */

        if(_prof)
        {
            /* Accumulate component times into the caller's accumulator. */
            prof->t_prelim += _t_prelim;
            prof->t_prelim_gemm += _t_prelim_gemm;
            prof->t_refine += _t_refine;
            prof->t_adp += _t_adp;
            prof->t_fused += _t_fused;
            prof->t_scale += _t_scale;
            prof->t_int8 += _t_int8;
            prof->t_accum += _t_accum;
            prof->effective_s_used = std::max(prof->effective_s_used, effective_s);
            prof->n_sub_gemms += 1u;
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

} // namespace FP64Emulation

bool fp64EmulationIsEnabled()
{
    using namespace FP64Emulation;
    static const bool enabled = []() -> bool {
        const char* v = std::getenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION");
        return (v != nullptr && std::strcmp(v, "1") == 0);
    }();
    return enabled;
}

bool fp64EmulationPerformanceCheck(const _rocblaslt_handle* h,
                                   hipblasOperation_t       opA,
                                   hipblasOperation_t       opB,
                                   int64_t                  m,
                                   int64_t                  n,
                                   int64_t                  k)
{
    using namespace FP64Emulation;
    const int  device = h->device;
    const bool tA     = (opA != HIPBLAS_OP_N);
    const bool tB     = (opB != HIPBLAS_OP_N);
    /* Include ADP overhead when the handle is configured for dynamic (ADP) mode,
     * so the performance gate correctly accounts for the hipStreamSynchronize cost. */
    const bool     dyn        = (h->emulation.mantissa_control != 1);
    const unsigned num_moduli = fp64EmulationEffectiveNumModuli(h);
    const double   t_emul     = effective_time_ms(tA, tB, m, n, k, num_moduli, device, dyn);
    const double t_native = perf_model_times(tA, tB, m, n, k, num_moduli, device, dyn).t_native_ms;
    return t_emul <= t_native;
}

bool fp64EmulationIsEager()
{
    using namespace FP64Emulation;
    static const bool eager = []() -> bool {
        const char* v = std::getenv("HIPBLASLT_EMULATION_STRATEGY");
        return (v != nullptr && std::strcmp(v, "eager") == 0);
    }();
    return eager;
}

uint32_t fp64EmulationSpecialValuesMask()
{
    using namespace FP64Emulation;
    static const uint32_t mask = []() -> uint32_t {
        const char* v = std::getenv("HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK");
        if(v == nullptr)
            return 0x3u;
        return static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
    }();
    return mask;
}

bool fp64EmulationWouldApply(const _rocblaslt_handle* h,
                             hipDataType              type_a,
                             hipblasOperation_t       opA,
                             hipblasOperation_t       opB,
                             int64_t                  m,
                             int64_t                  n,
                             int64_t                  k,
                             int                      batch_count)
{
    using namespace FP64Emulation;
    if(type_a != HIP_R_64F || batch_count != 1)
        return false;
    const bool emulEnabled
        = (h->emulation.enabled == 1) || (h->emulation.enabled != 0 && fp64EmulationIsEnabled());
    if(!emulEnabled)
        return false;
    /* Disable emulation on devices not listed in the perf-model table to
     * avoid running with unvalidated performance predictions.              */
    const int dev = h->device;
    if(!get_perf_model_params(dev))
        return false;
    const bool eager
        = (h->emulation.strategy == 2) || (h->emulation.strategy != 1 && fp64EmulationIsEager());
    return eager || fp64EmulationPerformanceCheck(h, opA, opB, m, n, k);
}

Fp64EmulationEnvValue fp64EmulationParseEnabledEnv(const char* value)
{
    using namespace FP64Emulation;
    if(value == nullptr)
        return {FP64_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "1") == 0)
        return {FP64_EMULATION_ENV_VALID, 1u};
    if(std::strcmp(value, "0") == 0)
        return {FP64_EMULATION_ENV_VALID, 0u};
    return {FP64_EMULATION_ENV_INVALID, 0u};
}

Fp64EmulationEnvValue fp64EmulationParseStrategyEnv(const char* value)
{
    using namespace FP64Emulation;
    if(value == nullptr)
        return {FP64_EMULATION_ENV_UNSET, 0u};
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
    using namespace FP64Emulation;
    if(value == nullptr)
        return {FP64_EMULATION_ENV_UNSET, 0x3u};
    char*      endp = nullptr;
    const long v    = std::strtol(value, &endp, 0);
    if(endp == value || *endp != '\0' || v < 0)
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, static_cast<unsigned>(v)};
}

Fp64EmulationEnvValue fp64EmulationParseMantissaBitCountEnv(const char* value)
{
    using namespace FP64Emulation;
    if(value == nullptr)
        return {FP64_EMULATION_ENV_UNSET, 0u};
    char*      endp = nullptr;
    const long v    = std::strtol(value, &endp, 10);
    if(endp == value || *endp != '\0' || v < 0 || v > 140)
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, static_cast<unsigned>(v)};
}

bool fp64EmulationIsValidMantissaBitCount(int value)
{
    using namespace FP64Emulation;
    return value >= -1 && value <= 140;
}

Fp64EmulationDecision fp64EmulationDecision(const _rocblaslt_handle* h,
                                            hipDataType              type_a,
                                            hipblasOperation_t       opA,
                                            hipblasOperation_t       opB,
                                            int64_t                  m,
                                            int64_t                  n,
                                            int64_t                  k,
                                            int                      batch_count)
{
    using namespace FP64Emulation;
    Fp64EmulationDecision result{};
    result.status       = rocblaslt_status_success;
    result.apply        = false;
    result.num_moduli   = fp64EmulationNumModuli();
    result.sv_mask      = fp64EmulationSpecialValuesMask();
    result.dynamic_mode = false;

    /* Type and batch-count pre-checks (no env-var validation needed). */
    if(type_a != HIP_R_64F || batch_count != 1)
        return result; /* apply=false, success */

    /* Emulation enabled check. */
    const bool emulEnabled
        = (h->emulation.enabled == 1) || (h->emulation.enabled != 0 && fp64EmulationIsEnabled());
    if(!emulEnabled)
        return result;

    /* Device must be in the supported table. */
    const int dev = h->device;
    if(!get_perf_model_params(dev))
        return result;

    /* Resolve num_moduli from handle settings. */
    result.num_moduli = fp64EmulationEffectiveNumModuli(h);

    /* Handle special_values_mask override. */
    if(h->emulation.special_values_mask != ~0u)
        result.sv_mask = h->emulation.special_values_mask;

    /* dynamic_mode: DYNAMIC (ADP) mantissa control — adaptively selects
     * the minimum s needed for FP64 precision on the given input data.   */
    result.dynamic_mode = (h->emulation.mantissa_control != 1);

    /* Strategy (eager vs performant). */
    const bool eager
        = (h->emulation.strategy == 2) || (h->emulation.strategy != 1 && fp64EmulationIsEager());
    if(eager || fp64EmulationPerformanceCheck(h, opA, opB, m, n, k))
        result.apply = true;

    return result;
}

unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h)
{
    using namespace FP64Emulation;
    if(h->emulation.mantissa_control == 1 && h->emulation.max_mantissa_bits >= 0)
    {
        const unsigned target = static_cast<unsigned>(h->emulation.max_mantissa_bits);
        for(unsigned s = 2u; s <= S_MAX; ++s)
            if(cum_bits[s - 2u] >= static_cast<double>(target))
                return s;
        return S_MAX;
    }
    return fp64EmulationNumModuli();
}

size_t fp64EmulationWorkspaceSize(const _rocblaslt_handle*     h,
                                  hipblasOperation_t           opA,
                                  hipblasOperation_t           opB,
                                  int64_t                      m,
                                  int64_t                      n,
                                  int64_t                      k,
                                  const Fp64EmulationDecision& decision)
{
    using namespace FP64Emulation;
    assert(h != nullptr && "fp64EmulationWorkspaceSize requires a valid handle");
    /* In ADP mode the workspace must cover S_MAX=18 moduli so that any
     * adaptive effective_s fits without reallocation.  In fixed mode use
     * the resolved moduli count directly.                                   */
    const unsigned num_moduli = decision.dynamic_mode ? S_MAX : decision.num_moduli;
    const int      device     = h->device;
    const bool     tA         = (opA != HIPBLAS_OP_N);
    const bool     tB         = (opB != HIPBLAS_OP_N);

    /* Mirror the splitting decision in emulated_gemm_impl exactly.
     *
     * emulated_gemm_impl uses settings.num_moduli (= fp64EmulationEffectiveNumModuli(h))
     * for the chunk-size / n_chunks check that decides whether to split, regardless of
     * whether dynamic mode is active (where num_moduli parameter = S_MAX but
     * settings.num_moduli is the user-configured max, e.g. 16).
     *
     * Using num_moduli (= ws_moduli = S_MAX in dynamic mode) here instead would
     * cause the workspace function to decide to split for problem sizes where the
     * implementation goes to the monolithic path.  The monolithic path needs the full
     * (m, n) workspace, which is larger than max(WS(half), WS(half)), resulting in
     * the workspace being under-allocated and a buffer overflow at runtime.           */
    const unsigned split_num_moduli = fp64EmulationEffectiveNumModuli(h);
    /* When the fused kernel is forced (HIPBLASLT_EMULATION_FUSED=on/force), use the
     * fused chunk formula (excludes C32i from workspace budget).  This eliminates
     * binary-halving for all practical shapes, avoiding per-leaf pipeline overhead.
     * Consistent with the split decision in emulated_gemm_impl.              */
    const bool     fused_on_ws = (oz2_fused_mode() == Oz2FusedMode::ON);
    const unsigned chunk_sz    = fused_on_ws ? compute_chunk_size_fused(m, n, k, split_num_moduli)
                                             : compute_chunk_size(m, n, k, split_num_moduli);
    const unsigned n_chunks    = (split_num_moduli + chunk_sz - 1u) / chunk_sz;

    if(n_chunks > 1u)
    {
        const bool    split_m = (m >= n);
        const int64_t half_m  = split_m ? m / 2 : m;
        const int64_t half_n  = split_m ? n : n / 2;
        const int64_t m2      = split_m ? (m - m / 2) : m;
        const int64_t n2      = split_m ? n : (n - n / 2);

        const double t_mono
            = perf_model_times(tA, tB, m, n, k, split_num_moduli, device, decision.dynamic_mode)
                  .t_total_ms;
        const double t_split
            = 2.
              * effective_time_ms(
                  tA, tB, half_m, half_n, k, split_num_moduli, device, decision.dynamic_mode);

        if(t_split < t_mono)
        {
            /* num_moduli (= ws_moduli) is forwarded to the recursive calls so that
             * the monolithic workspace at each leaf is sized for the correct layout
             * (e.g. S_MAX in dynamic mode, matching layout_moduli in the impl). */
            return std::max(fp64EmulationWorkspaceSize(h, opA, opB, half_m, half_n, k, decision),
                            fp64EmulationWorkspaceSize(h, opA, opB, m2, n2, k, decision));
        }
    }

    /* Monolithic path workspace. */
    const size_t lda8i  = pad(static_cast<size_t>(k));
    const size_t cola8i = pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);
    /* Fused path: no C32i workspace needed for production (registers hold it).
     * Non-fused path: standard chunk formula (includes C32i in budget).        */
    const unsigned chunk_ws = fused_on_ws ? compute_chunk_size_fused(m, n, k, num_moduli)
                                          : compute_chunk_size(m, n, k, num_moduli);

    /* Zhi/Zlo accumulators are only needed when there are multiple passes
     * (chunk_ws < num_moduli).  Single-pass finalize writes D directly.  */
    const size_t szZhi_ws = (chunk_ws < num_moduli) ? szC32i : 0u;

    /* C32i production slots:
     *   Fused path: C32i is accumulated in registers — only 1 slot needed for
     *               the preliminary GEMM result.
     *   Non-fused:  chunk_ws slots for the batched INT8 GEMMs.                */
    const size_t n_c32i_ws = fused_on_ws ? 1u : static_cast<size_t>(chunk_ws);

    return chunk_ws * lda8i * cola8i * sizeof(int8_t)
           + chunk_ws * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)
           + n_c32i_ws * szC32i * sizeof(int32_t) + szZhi_ws * sizeof(double) * 2
           + cola8i * sizeof(int16_t) + padn * sizeof(int16_t) + sizeof(uint32_t)
           + cola8i * sizeof(int32_t) /* row_max[m] */
           + padn * sizeof(int32_t) /* col_max[n] — precomputed by partial kernel */
           + 2 * sizeof(float) /* ADP float buffer: adp_buf[0..1] (bias ±200) */
           + OZ2_INT8_GEMM_WS_BYTES; /* INT8 GEMM workspace (preliminary + batch)  */
}

unsigned fp64EmulationNumModuli()
{
    using namespace FP64Emulation;
    static const unsigned num_moduli = []() -> unsigned {
        const char* v = std::getenv("HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT");
        if(v == nullptr)
            return 16u;
        const unsigned target = static_cast<unsigned>(std::strtoul(v, nullptr, 0));
        if(target == 0u)
            return S_MAX;
        for(unsigned s = 2u; s <= S_MAX; ++s)
            if(cum_bits[s - 2u] >= static_cast<double>(target))
                return s;
        return S_MAX;
    }();
    return num_moduli;
}

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
    using namespace FP64Emulation;
    /* ── Pre-allocate a single workspace for the entire call ──────────────────
     * emulated_gemm_impl recurses for split shapes; without a pre-allocated
     * buffer each leaf sub-GEMM would do its own hipMallocAsync/hipFreeAsync.
     * Allocating once here and passing it through settings eliminates that
     * overhead (e.g. 16 redundant alloc/free pairs for the 65K square case).
     * If the caller already provided a sufficient workspace we use it as-is.  */
    const _rocblaslt_handle* h = reinterpret_cast<const _rocblaslt_handle*>(handle);
    const unsigned num_moduli  = (settings.num_moduli >= 2u && settings.num_moduli <= S_MAX)
                                     ? settings.num_moduli
                                     : fp64EmulationNumModuli();
    const int      device      = h->device;
    /* Build a lightweight decision just to communicate dynamic_mode and
     * num_moduli to fp64EmulationWorkspaceSize — only those two fields are
     * consulted by the workspace function.                                  */
    Fp64EmulationDecision ws_decision{};
    ws_decision.dynamic_mode = settings.dynamic_mode;
    ws_decision.num_moduli   = num_moduli;
    const size_t wsNeeded    = fp64EmulationWorkspaceSize(h, opA, opB, m, n, k, ws_decision);

    Fp64EmulationSettings effectiveSettings = settings;
    void*                 ws_toplevel       = nullptr;

    if(wsNeeded > 0
       && (effectiveSettings.workspace == nullptr || effectiveSettings.workspace_bytes < wsNeeded))
    {
        /* Use hipMalloc (not hipMallocAsync) so that:
         *  (1) The allocation is not affected by any HIP stream error state from
         *      a previous GPU fault on the same stream.  hipMallocAsync fails
         *      immediately when the stream has a sticky error, even with plenty
         *      of free device memory.
         */
        if(hipMalloc(&ws_toplevel, wsNeeded) != hipSuccess)
            return rocblaslt_status_memory_error;
        effectiveSettings.workspace       = ws_toplevel;
        effectiveSettings.workspace_bytes = wsNeeded;
    }
    /* ────────────────────────────────────────────────────────────────────── */

    const char* const _pf   = profile_file();
    const bool        _prof = (_pf != nullptr);

    ProfileAccum accum{};
    hipEvent_t   ev_start{}, ev_end{};
    if(_prof)
    {
        (void)hipEventCreate(&ev_start);
        (void)hipEventCreate(&ev_end);
        (void)hipEventRecord(ev_start, stream);
    }

    const rocblaslt_status st = emulated_gemm_impl(h,
                                                   opA,
                                                   opB,
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   A,
                                                   lda,
                                                   B,
                                                   ldb,
                                                   beta,
                                                   C,
                                                   ldc,
                                                   D,
                                                   ldd,
                                                   stream,
                                                   effectiveSettings,
                                                   _prof ? &accum : nullptr);

    /* Release the top-level workspace now that all leaves have finished.    */
    if(ws_toplevel != nullptr)
        (void)hipFree(ws_toplevel);

    if(_prof)
    {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        /* Use effective_s_used for profiling chunk sizes so the CSV reflects
         * what was actually computed per pass (not the configured maximum).  */
        const unsigned prof_s     = accum.effective_s_used ? accum.effective_s_used : num_moduli;
        const unsigned chunk_size = compute_chunk_size(m, n, k, prof_s);
        const unsigned scale_chunk_size = chunk_size;
        const bool     tA               = (opA != HIPBLAS_OP_N);
        const bool     tB               = (opB != HIPBLAS_OP_N);
        /* Use the split-aware model: each component is the sum across all leaves.
         * t_native_ms remains for the original (m,n,k) problem.           */
        const PerfModelTimes pm = effective_perf_model_times(
            tA, tB, m, n, k, num_moduli, device, settings.dynamic_mode);

        std::FILE* _f = std::fopen(_pf, "a");
        if(_f)
        {
            if(std::ftell(_f) == 0)
                std::fprintf(
                    _f,
                    "m,n,k,transA,transB,num_moduli,effective_s,scale_chunk_size,gemm_chunk_size,"
                    "workspace_bytes,num_sub_gemms,"
                    "t_prelim_ms,t_prelim_gemm_ms,t_refine_ms,"
                    "t_fused_ms,t_adp_ms,t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                    "t_total_ms,"
                    "pred_prelim_ms,pred_prelim_gemm_ms,pred_refine_ms,pred_adp_ms,"
                    "pred_scale_ms,pred_int8_gemm_ms,pred_accum_ms,"
                    "pred_host_ms,pred_fused_ms,pred_total_ms,pred_native_dgemm_"
                    "ms\n");
            std::fprintf(_f,
                         "%lld,%lld,%lld,%c,%c,%u,%u,%u,%u,"
                         "%llu,%u,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                         "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                         (long long)m,
                         (long long)n,
                         (long long)k,
                         tA ? 'T' : 'N',
                         tB ? 'T' : 'N',
                         num_moduli,
                         accum.effective_s_used,
                         scale_chunk_size,
                         chunk_size,
                         (unsigned long long)wsNeeded,
                         accum.n_sub_gemms,
                         accum.t_prelim,
                         accum.t_prelim_gemm,
                         accum.t_refine,
                         accum.t_fused,
                         accum.t_adp,
                         accum.t_scale,
                         accum.t_int8,
                         accum.t_accum,
                         t_total,
                         pm.t_prelim_ms,
                         pm.t_prelim_gemm_ms,
                         pm.t_refine_ms,
                         pm.t_adp_ms,
                         pm.t_scale_ms,
                         pm.t_int8_gemms_ms,
                         pm.t_accum_ms,
                         pm.t_host_ms,
                         pm.t_fused_ms,
                         pm.t_total_ms,
                         pm.t_native_ms);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}
