// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/*
 * fp64_emulation_fused.cpp
 *
 * Self-contained implementation of the fused MFMA+CRT kernel for the
 * Ozaki Scheme II FP64 GEMM emulation.  Contains:
 *
 *   — GPU constant-memory extern declarations (symbols defined in
 *     fp64_emulation.cpp, shared via -fgpu-rdc device linking)
 *   — Per-architecture MFMA constants (OZ2_KBLK_16 / OZ2_KBLK_32)
 *   — MFMA vector types, instruction wrappers, LDS load helpers
 *   — oz2_fused_TN_kernel template definition (all instantiations are
 *     in oz2_launch_fused_TN, so the template stays in this file)
 *   — oz2_fused_mode()       — env-var gate
 *   — oz2_launch_fused_TN()  — macrotile-dispatch launcher
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 */

#include "fp64_emulation_fused.hpp"
#include "fp64_emulation_tables.hpp"

#include <cstdio>    /* std::fprintf */
#include <cstdlib>   /* std::getenv, std::abort */
#include <cstring>   /* std::strcmp  */


/* =========================================================================
 * TILE-templated MFMA traits
 *
 * Collects all per-TILE, per-architecture MFMA constants in one place:
 *   kblk  – K-block size consumed per MFMA instruction
 *   src_t – MFMA source operand type (ext_vector_type(2) on gfx95x, int64_t on gfx94x)
 *   acc_t – MFMA accumulator output type (ext_vector_type(4) or (16))
 *
 * Unified oz2_load_mfma_src<TILE> / oz2_do_mfma<TILE> below use these traits
 * so that apply_mfma can be written once without if constexpr on TILE.
 * All resolved at compile time — zero runtime overhead.
 * ========================================================================= */
template <int TILE> struct oz2_mfma_traits;
template<> struct oz2_mfma_traits<16> {
#if defined(__gfx950__)
    static constexpr int    kblk       = 64;
    static constexpr size_t lds_budget = 160 * 1024;
    static constexpr int    ku_max     = 4;
    typedef long src_t __attribute__((ext_vector_type(2)));
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
    static constexpr int    kblk       = 32;
    static constexpr size_t lds_budget = 64 * 1024;
    static constexpr int    ku_max     = 4;
    using src_t = int64_t;
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
    typedef int acc_t __attribute__((ext_vector_type(4)));
};
template<> struct oz2_mfma_traits<32> {
#if defined(__gfx950__)
    static constexpr int    kblk       = 32;
    static constexpr size_t lds_budget = 160 * 1024;
    static constexpr int    ku_max     = 2;   /* KU=4 → scratch=424B on gfx950 T32 */
    typedef long src_t __attribute__((ext_vector_type(2)));
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
    static constexpr int    kblk       = 16;
    static constexpr size_t lds_budget = 64 * 1024;
    static constexpr int    ku_max     = 4;
    using src_t = int64_t;
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
    typedef int acc_t __attribute__((ext_vector_type(16)));
};

/* ── Architecture-specific MFMA source load helper ─────────────────────────
 * On gfx95x (K_A_BYTES=16): two 8-byte loads give 2-way LDS bank conflicts
 * instead of 4-way from a single 16-byte load.
 * On gfx94x (K_A_BYTES=8): single 8-byte load is always conflict-free.
 * src16 == src32 on all architectures, so one template covers both tiles.   */
template <int TILE>
__device__ __forceinline__ typename oz2_mfma_traits<TILE>::src_t
oz2_load_mfma_src(const int8_t* __restrict__ ptr) noexcept {
#if defined(__gfx950__)
    typename oz2_mfma_traits<TILE>::src_t v;
    v[0] = *reinterpret_cast<const long*>(ptr);
    v[1] = *reinterpret_cast<const long*>(ptr + 8);
    return v;
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
    return *reinterpret_cast<const typename oz2_mfma_traits<TILE>::src_t*>(ptr);
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
}

/* ── Architecture-specific MFMA instruction wrapper ────────────────────────
 * Dispatches to the correct v_mfma_i32 builtin based on TILE and arch.
 * TILE=16: 16x16x32 (gfx94x) / 16x16x64 (gfx95x)
 * TILE=32: 32x32x16 (gfx94x) / 32x32x32 (gfx95x)                         */
template <int TILE>
__device__ __forceinline__ typename oz2_mfma_traits<TILE>::acc_t
oz2_do_mfma(typename oz2_mfma_traits<TILE>::src_t a,
            typename oz2_mfma_traits<TILE>::src_t b,
            typename oz2_mfma_traits<TILE>::acc_t c) noexcept {
#if defined(__gfx950__)
    if constexpr (TILE == 16) return __builtin_amdgcn_mfma_i32_16x16x64_i8(a, b, c, 0, 0, 0);
    else                      return __builtin_amdgcn_mfma_i32_32x32x32_i8(a, b, c, 0, 0, 0);
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
    if constexpr (TILE == 16) return __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, 0, 0);
    else                      return __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, 0, 0);
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
}

/* ── Load-width type trait ─────────────────────────────────────────────────
 * Maps a compile-time LOAD_BYTES (4, 8, or 16) to the corresponding C++
 * register type.  The compiler lowers these to global_load_dword,
 * global_load_dwordx2, or global_load_dwordx4 respectively.                */
template <int LB> struct oz2_load_type;
template<> struct oz2_load_type<4>  { using type = int32_t; };
template<> struct oz2_load_type<8>  { using type = int64_t; };
template<> struct oz2_load_type<16> { using type = longlong2; };

/* =========================================================================
 * oz2_fused_TN_kernel — unified template supporting three macrotile sizes:
 *
 *   TILE=16, WM=4, WN=4, WaveM=1, WaveN=1 → 64×64   macrotile, 1024 threads (default)
 *   TILE=32, WM=4, WN=2, WaveM=1, WaveN=1 → 128×64  macrotile,  512 threads (m ≥ n preferred)
 *   TILE=32, WM=2, WN=4, WaveM=1, WaveN=1 → 64×128  macrotile,  512 threads (m < n preferred)
 *
 * Macrotile dimensions: (WM × WaveM × TILE) × (WN × WaveN × TILE)
 * Thread block size:    WM × WN × 64 threads
 *   WaveM/WaveN do NOT add threads; instead each wavefront computes
 *   WaveM×WaveN MFMA output tiles, improving LDS data reuse.
 *
 * CRT accumulators (Zhi/Zlo):
 *   When LDS budget permits (USE_LDS_ACCUM=true, e.g., MI350 with large
 *   WaveM×WaveN), Zhi/Zlo are stored in LDS rather than registers.  This
 *   keeps only NREG_single doubles in registers at any time regardless of
 *   WaveM×WaveN, enabling configurations like WM=1,WN=1,WaveM=4,WaveN=4
 *   that would otherwise spill.  The 2-way LDS bank conflict for 64-lane
 *   wavefronts (unavoidable with 32 banks and double-precision values) is
 *   hidden by the surrounding MFMA pipeline.
 *
 * MFMA instructions used:
 *   TILE=16: v_mfma_i32_16x16x32_i8 (gfx94x) / v_mfma_i32_16x16x64_i8 (gfx95x)
 *   TILE=32: v_mfma_i32_32x32x16_i8 (gfx94x) / v_mfma_i32_32x32x32_i8 (gfx95x)
 *
 * Template parameters:
 *   S      – number of moduli (2..OZ2_S_MAX)
 *   HAS_LO – true when S > 7 (double-double CRT accumulation)
 *   WM     – wavefronts in M direction (default 4)
 *   WN     – wavefronts in N direction (default 4)
 *   TILE   – MFMA tile: 16 (16×16 MFMA) or 32 (32×32 MFMA)
 *   WaveM  – MFMA tiles per wavefront in M direction (default 1)
 *   WaveN  – MFMA tiles per wavefront in N direction (default 1)
 *
 * All instantiations are in oz2_launch_fused_TN() below, so the template
 * definition lives here rather than in the header.
 * ========================================================================= */
/* KU_PARAM=0 means "auto-select K_UNROLL from LDS budget" (the default).
 * Set KU_PARAM=1,2,4 via OZ2_FUSED_SHAPE_OVERRIDE to override for tuning. */
template <int S, bool HAS_LO, int WM = 4, int WN = 4, int TILE = 16,
          int WaveM = 1, int WaveN = 1, int KU_PARAM = 0,
          bool FORCE_VGPR_ACCUM = false, int PGR = 1, int LB_PARAM = 0>
__global__ static void
oz2_fused_TN_kernel(
    const int8_t*  __restrict__ A8i,   /* [S × lda8i × cola8i] INT8 A  */
    size_t         stride_A_s,          /* lda8i × cola8i               */
    size_t         lda8i,               /* padded k                      */
    const int8_t*  __restrict__ B8i,   /* [S × ldb8i × n]  INT8 B      */
    size_t         stride_B_s,          /* ldb8i × n                    */
    size_t         ldb8i,               /* padded k                      */
    const double*  __restrict__ C,
    double*        __restrict__ D,
    int64_t m, int64_t n, int64_t k,
    int64_t ldc, int64_t ldd,
    double alpha, double beta,
    const int16_t* __restrict__ sftA,
    const int16_t* __restrict__ sftB,
    int            num_xccs)            /* number of XCCs on this device */
{
    /* ── Derived compile-time constants ────────────────────────────────────── */
    using mfma_acc_t = typename oz2_mfma_traits<TILE>::acc_t;           /* v4i32 (TILE=16) or v16i32 (TILE=32)  */
    static constexpr int NREG_single = TILE * TILE / 64;                /* accumulators per thread per MFMA tile */
    static constexpr int NREG        = WaveM * WaveN * NREG_single;    /* total accumulators per thread         */
    static constexpr int KBLK        = oz2_mfma_traits<TILE>::kblk;
    /* K-loop unroll factor: auto-selected from LDS budget and per-arch
     * register-pressure ceiling (ku_max), or overridden via KU_PARAM.
     * All arch-specific values come from oz2_mfma_traits — no #ifdef here. */
    static constexpr size_t   LDS_BUDGET    = oz2_mfma_traits<TILE>::lds_budget;
    static constexpr size_t   LDS_MFMA_K4   = 2 * (WM * WaveM * TILE * (KBLK * 4)
                                                   + WN * WaveN * TILE * (KBLK * 4));
    static constexpr int K_UNROLL_LDS   = (LDS_MFMA_K4 <= LDS_BUDGET) ? 4 : 2;
    static constexpr int K_UNROLL_AUTO  = (K_UNROLL_LDS < oz2_mfma_traits<TILE>::ku_max)
                                        ? K_UNROLL_LDS : oz2_mfma_traits<TILE>::ku_max;
    static constexpr int K_UNROLL = (KU_PARAM > 0) ? KU_PARAM : K_UNROLL_AUTO;
    static constexpr int KBLK_LOAD   = KBLK * K_UNROLL;
    static constexpr int K_A_BYTES   = KBLK * TILE / 64;
    static constexpr int KBLK_PAD_UNIT = (K_A_BYTES == 16) ? 8 : K_A_BYTES;
    static constexpr size_t   LDS_BYTES_OLD = 2 * (WM * WaveM * TILE * KBLK_LOAD
                                                  + WN * WaveN * TILE * KBLK_LOAD);
    /* Pad LDS stride so that apply_mfma's 64-lane reads at m_col*KBLK_STRIDE
     * land on distinct LDS banks (conflict-free).                             */
    static constexpr int KBLK_PAD    = (LDS_BYTES_OLD + KBLK_PAD_UNIT * TILE * (WM * WaveM + WN * WaveN) * 2
                                         <= LDS_BUDGET) ? KBLK_PAD_UNIT : 0;
    static constexpr int KBLK_STRIDE = KBLK_LOAD + KBLK_PAD;
    static constexpr int BLK_THR    = WM * WN * 64;

    /* ── LDS accumulator decision (computed early — needed by LOAD_BYTES) ───
     * Store Zhi/Zlo in LDS when the combined budget (MFMA tiles + accumulators
     * + scale shifts) fits within LDS_BUDGET.  This enables large WaveM×WaveN
     * configurations (e.g., WM=1,WN=1,WaveM=4,WaveN=4 on MI350) that would
     * otherwise spill registers.  When false (e.g., all current MI300 configs),
     * accumulators stay in VGPRs with zero overhead.                          */
    static constexpr int  N_BUF       = PGR + 1;   /* 2 for PGR=1, 3 for PGR=2 */
    static constexpr size_t LDS_MFMA      = N_BUF * (WM * WaveM * TILE * KBLK_STRIDE
                                                    + WN * WaveN * TILE * KBLK_STRIDE);
    static constexpr size_t LDS_ZHI_ZLO   = 2 * BLK_THR * NREG * sizeof(double);
    static constexpr bool   FITS_IN_LDS   = (LDS_MFMA + LDS_ZHI_ZLO <= LDS_BUDGET);
    /* USE_LDS_ACCUM: use LDS for Zhi/Zlo only when LDS has room AND FORCE_VGPR_ACCUM is not set.
     * FORCE_VGPR_ACCUM=true keeps accumulators in registers even when LDS has space,
     * which can be faster when NREG is small enough to avoid VGPR spilling.             */
    static constexpr bool   USE_LDS_ACCUM = FITS_IN_LDS && !FORCE_VGPR_ACCUM;

    /* Auto-select the widest global load (4/8/16 bytes = dword/dwordx2/dwordx4)
     * that keeps both A_STEPS and B_STEPS ≥ 1.  Wider loads reduce instruction
     * count; the limit is the smaller of the A and B tile data per thread.
     *
     * HIGH_VGPR_PRESSURE guard: when NREG is large (e.g., 256×256 with
     * NREG=64), wider loads reduce instruction count but also reduce
     * scheduling flexibility.  Fall back to 4-byte loads to preserve
     * the original instruction schedule for these high-NREG configs.
     * Applied unconditionally regardless of USE_LDS_ACCUM — benchmarked
     * regression on both gfx942 and gfx950 when wider loads are used
     * with NREG=64.
     *
     * gfx942 (KBLK=32): 8 B for KU≥2 symmetric with NREG≤16, 4 B otherwise.
     * gfx950 (KBLK=64): 16 B for NREG≤16, 4 B for NREG>16.                */
    static constexpr int MIN_WAVE_DIM    = ((WM * WaveM) < (WN * WaveN))
                                         ? (WM * WaveM) : (WN * WaveN);
    static constexpr int MIN_BYTES_PER_THR = MIN_WAVE_DIM * TILE * KBLK_LOAD / BLK_THR;
    static constexpr bool HIGH_VGPR_PRESSURE = (NREG > 16);
    static constexpr int LOAD_BYTES_AUTO = (!HIGH_VGPR_PRESSURE && MIN_BYTES_PER_THR >= 16) ? 16
                                         : (!HIGH_VGPR_PRESSURE && MIN_BYTES_PER_THR >=  8) ?  8
                                         :                                                      4;
    static constexpr int LOAD_BYTES  = (LB_PARAM > 0) ? LB_PARAM : LOAD_BYTES_AUTO;
    using load_t = typename oz2_load_type<LOAD_BYTES>::type;
    static constexpr int KDIM       = KBLK_LOAD / LOAD_BYTES;
    static constexpr int A_STEPS    = (WM * WaveM * TILE * KDIM) / BLK_THR;
    static constexpr int B_STEPS    = (WN * WaveN * TILE * KDIM) / BLK_THR;
    /* VALID_CONFIG: false when A_STEPS or B_STEPS < 1 (gfx950-only KU=1 on gfx942). */
    static constexpr int  A_STEPS_SAFE = (A_STEPS > 0) ? A_STEPS : 1;
    static constexpr int  B_STEPS_SAFE = (B_STEPS > 0) ? B_STEPS : 1;
    static constexpr bool VALID_CONFIG  = (A_STEPS >= 1 && B_STEPS >= 1);
    if constexpr (!VALID_CONFIG) return; /* bail early for invalid arch/config combos */

    /* ── Static LDS ────────────────────────────────────────────────────────── */
    static constexpr int A_BUF_BYTES = WM * WaveM * TILE * KBLK_STRIDE;
    static constexpr int B_BUF_BYTES = WN * WaveN * TILE * KBLK_STRIDE;
    /* A8i_lds and B8i_lds are declared as a single contiguous flat buffer so
     * that the finalize section can safely reinterpret the combined region as
     * a double* tile_out for the coalesced output transpose.                  */
    __shared__ int8_t  mfma_lds_flat[N_BUF * (A_BUF_BYTES + B_BUF_BYTES)];
    auto (&A8i_lds)[N_BUF][A_BUF_BYTES] =
        *reinterpret_cast<int8_t(*)[N_BUF][A_BUF_BYTES]>(mfma_lds_flat);
    auto (&B8i_lds)[N_BUF][B_BUF_BYTES] =
        *reinterpret_cast<int8_t(*)[N_BUF][B_BUF_BYTES]>(mfma_lds_flat + N_BUF * A_BUF_BYTES);
    /* CRT accumulator LDS — flat layout [e * BLK_THR + tid].
     * Thread tid owns column tid for accumulator row e, giving consecutive
     * intra-wavefront access (2-way bank conflict; unavoidable for 64-lane
     * waves with 32 LDS banks and 8-byte doubles).
     * When USE_LDS_ACCUM=false, arrays are size 1 (placeholder, never read). */
    __shared__ double Zhi_lds[USE_LDS_ACCUM ? NREG * BLK_THR : 1];
    __shared__ double Zlo_lds[USE_LDS_ACCUM ? NREG * BLK_THR : 1];

    /* ── Thread decomposition ───────────────────────────────────────────────── */
    const int tid  = static_cast<int>(threadIdx.x);
    const int wid  = tid / 64;
    const int wm   = wid / WN;
    const int wn   = wid % WN;
    const int lane = tid % 64;
    const int k_int = static_cast<int>(k);

    /* ── XCC-aware block tile mapping ───────────────────────────────────────── */
    static constexpr int SWIZZLE_B8i_PREF = 4;

    const int m_tiles_total   = (static_cast<int>(m) + WM * WaveM * TILE - 1)
                                / (WM * WaveM * TILE);
    const int n_tiles_total   = (static_cast<int>(n) + WN * WaveN * TILE - 1)
                                / (WN * WaveN * TILE);
    uint32_t hw_xcc_id = 0u;
    asm volatile("s_getreg_b32 %0, hwreg(20)" : "=s"(hw_xcc_id));

    const int linear_block = static_cast<int>(blockIdx.x);
    const int xcc_id       = static_cast<int>(hw_xcc_id);
    const int intra_xcc    = linear_block / num_xccs;

    int m_tile, n_tile;
    if (n <= static_cast<int64_t>(m)) {
        /* Tall / square: swizzle M — B8i L2 reuse (groups of swizzle_eff M-tiles
         * share the same N-range, keeping B8i hot in L2).                      */
        const int m_tiles_per_xcc = (m_tiles_total + num_xccs - 1) / num_xccs;
        const int swizzle_eff     = (SWIZZLE_B8i_PREF < m_tiles_per_xcc) ? SWIZZLE_B8i_PREF : m_tiles_per_xcc;
        const int tiles_per_xcc   = m_tiles_per_xcc * n_tiles_total;
        if (intra_xcc >= tiles_per_xcc) return;
        const int swizzle_group = intra_xcc / swizzle_eff;
        const int m_local       = intra_xcc % swizzle_eff;
        n_tile = swizzle_group % n_tiles_total;
        m_tile = xcc_id * m_tiles_per_xcc + (swizzle_group / n_tiles_total) * swizzle_eff + m_local;
        if (m_tile >= m_tiles_total) return;
    } else {
        /* Wide: swizzle N — A8i L2 reuse (groups of swizzle_eff N-tiles
         * share the same M-range, keeping A8i hot in L2).                      */
        const int n_tiles_per_xcc = (n_tiles_total + num_xccs - 1) / num_xccs;
        const int swizzle_eff     = (SWIZZLE_B8i_PREF < n_tiles_per_xcc) ? SWIZZLE_B8i_PREF : n_tiles_per_xcc;
        const int tiles_per_xcc   = m_tiles_total * n_tiles_per_xcc;
        if (intra_xcc >= tiles_per_xcc) return;
        const int swizzle_group = intra_xcc / swizzle_eff;
        const int n_local       = intra_xcc % swizzle_eff;
        m_tile = swizzle_group % m_tiles_total;
        n_tile = xcc_id * n_tiles_per_xcc + (swizzle_group / m_tiles_total) * swizzle_eff + n_local;
        if (n_tile >= n_tiles_total) return;
    }

    const int block_m_base = m_tile * (WM * WaveM * TILE);
    const int block_n_base = n_tile * (WN * WaveN * TILE);
    const int m_base = block_m_base + wm * (WaveM * TILE);
    const int n_base = block_n_base + wn * (WaveN * TILE);

    /* ── CRT accumulators ───────────────────────────────────────────────────
     * Size-1 placeholder when USE_LDS_ACCUM=true (never accessed).
     * Each thread owns its own Zhi/Zlo elements — no barrier needed for init. */
    double Zhi_reg[USE_LDS_ACCUM ? 1 : NREG] = {};
    double Zlo_reg[USE_LDS_ACCUM ? 1 : NREG] = {};
    if constexpr (USE_LDS_ACCUM) {
        for (int e = 0; e < NREG; ++e) {
            Zhi_lds[e * BLK_THR + tid] = 0.0;
            Zlo_lds[e * BLK_THR + tid] = 0.0;
        }
    }

    /* ── Flat LDS/HBM offsets ──────────────────────────────────────────────── */
    /* Each thread precomputes its LDS write offset and HBM read offset once.  */
    int    lds_off_A[A_STEPS_SAFE];  /* flat LDS byte offset within one buffer */
    size_t hbm_off_A[A_STEPS_SAFE];  /* flat HBM byte offset from A8i base   */
    int    lds_off_B[B_STEPS_SAFE];
    size_t hbm_off_B[B_STEPS_SAFE];
    /* ── Pre-fetch modulus s=0 prologue into registers ─────────────────────── */
    /* rA/rB serve dual purpose: K-loop double-buffer AND cross-modulus prologue.
     * Declaring at kernel scope (outside both S-loop and K-loop) lets the compiler
     * allocate them once, saving A_STEPS+B_STEPS VGPRs vs the old rA_pro/rB_pro
     * design where both were simultaneously live during the K-loop.            */
    load_t rA[A_STEPS_SAFE];
    load_t rB[B_STEPS_SAFE];
    {
        /* Compute flat LDS + HBM offsets, then prefetch first K-block. */
        #pragma unroll
        for (int ls = 0; ls < A_STEPS; ++ls) {
            const int flat = static_cast<int>(threadIdx.x) + ls * BLK_THR;
            const int tile_dim = TILE * KDIM;
            const int wm  = flat / tile_dim;
            const int ml  = (flat % tile_dim) / KDIM;
            const int kd  = (flat % tile_dim) % KDIM;
            lds_off_A[ls] = (wm * TILE + ml) * KBLK_STRIDE + kd * LOAD_BYTES;
            const int mi = block_m_base + wm * TILE + ml;
            hbm_off_A[ls] = static_cast<size_t>(mi) * lda8i + static_cast<size_t>(kd) * LOAD_BYTES;
        }
        #pragma unroll
        for (int ls = 0; ls < B_STEPS; ++ls) {
            const int flat = static_cast<int>(threadIdx.x) + ls * BLK_THR;
            const int tile_dim = TILE * KDIM;
            const int wn  = flat / tile_dim;
            const int nl  = (flat % tile_dim) / KDIM;
            const int kd  = (flat % tile_dim) % KDIM;
            lds_off_B[ls] = (wn * TILE + nl) * KBLK_STRIDE + kd * LOAD_BYTES;
            const int ni = block_n_base + wn * TILE + nl;
            hbm_off_B[ls] = static_cast<size_t>(ni) * ldb8i + static_cast<size_t>(kd) * LOAD_BYTES;
        }
        if (k_int > 0) {
            #pragma unroll
            for (int ls = 0; ls < A_STEPS; ++ls)
                rA[ls] = *reinterpret_cast<const load_t*>(A8i + hbm_off_A[ls]);
            #pragma unroll
            for (int ls = 0; ls < B_STEPS; ++ls)
                rB[ls] = *reinterpret_cast<const load_t*>(B8i + hbm_off_B[ls]);
        }
    }
    if constexpr (USE_LDS_ACCUM) __syncthreads(); /* sync Zhi_lds/Zlo_lds init */

    /* ── MFMA helper ──────────────────────────────────────────────────────────
     * A_wm_base: pointer to A8i_lds[cur][wm*WaveM][0][0]
     * B_wn_base: pointer to B8i_lds[cur][wn*WaveN][0][0]
     * Loops over WaveM×WaveN MFMA tiles; each tile i's accumulators live in
     * C32[i*NREG_single .. (i+1)*NREG_single-1].
     * With WaveM=WaveN=1 the loop bodies execute once, identical to the
     * single-tile case.                                                        */
    static constexpr int N_TILES = WaveM * WaveN;
    auto apply_mfma = [&](const int8_t* A_wm_base, const int8_t* B_wn_base,
                          mfma_acc_t (&C32)[N_TILES]) __attribute__((always_inline)) {
        using src_t = typename oz2_mfma_traits<TILE>::src_t;
        const int m_col   = lane % TILE;
        const int k_base  = K_A_BYTES * (lane / TILE);
        const int base_off = m_col * KBLK_STRIDE + k_base;
        static constexpr int A_ROW_STRIDE = TILE * KBLK_STRIDE;
        static constexpr int B_ROW_STRIDE = TILE * KBLK_STRIDE;

        #pragma unroll
        for (int ku = 0; ku < K_UNROLL; ++ku) {
            const int off = base_off + ku * KBLK;

            if constexpr (WaveM < WaveN) {
                /* No pre-load array: wm is smaller → wm outer, hoist sa.
                 * Total LDS reads: WaveM(1+WaveN)×KU — minimized by making
                 * the smaller dimension outer.  Only 1 src_t in register.   */
                #pragma unroll
                for (int wm_w = 0; wm_w < WaveM; ++wm_w) {
                    const src_t sa = oz2_load_mfma_src<TILE>(A_wm_base + wm_w * A_ROW_STRIDE + off);
                    #pragma unroll
                    for (int wn_w = 0; wn_w < WaveN; ++wn_w) {
                        const src_t sb = oz2_load_mfma_src<TILE>(B_wn_base + wn_w * B_ROW_STRIDE + off);
                        C32[wm_w * WaveN + wn_w] = oz2_do_mfma<TILE>(sa, sb,
                                                                       C32[wm_w * WaveN + wn_w]);
                    }
                }
            } else {
                /* WaveM >= WaveN: wn is smaller or equal → wn outer, hoist sb. */
                #pragma unroll
                for (int wn_w = 0; wn_w < WaveN; ++wn_w) {
                    const src_t sb = oz2_load_mfma_src<TILE>(B_wn_base + wn_w * B_ROW_STRIDE + off);
                    #pragma unroll
                    for (int wm_w = 0; wm_w < WaveM; ++wm_w) {
                        const src_t sa = oz2_load_mfma_src<TILE>(A_wm_base + wm_w * A_ROW_STRIDE + off);
                        C32[wm_w * WaveN + wn_w] = oz2_do_mfma<TILE>(sa, sb,
                                                                       C32[wm_w * WaveN + wn_w]);
                    }
                }
            }
        }
    };

    /* ── CRT update helper ─────────────────────────────────────────────────────
     * Factored out so both the sequential and pipelined S-loops can reuse it.
     * Performs the TwoSum CRT accumulation for one modulus, given the MFMA
     * results in C32[] and the per-modulus CRT coefficients.                   */
    auto do_crt = [&](mfma_acc_t (&C32)[N_TILES], int s_idx)
                  __attribute__((always_inline)) {
        const double nm  = oz2_neg_mod(s_idx), im = oz2_inv_mod(s_idx);
        const double qhi = oz2_qpi_hi(S - 2, s_idx);
        const double qlo = HAS_LO ? oz2_qpi_lo(S - 2, s_idx) : 0.0;
        if constexpr (!USE_LDS_ACCUM) {
            for (int r = 0; r < NREG; ++r) {
                const double dc_raw = static_cast<double>(
                    C32[r / NREG_single][r % NREG_single]);
                const double dc     = fma(nm, rint(dc_raw * im), dc_raw);
                const double hi     = dc * qhi;
                const double new_hi = Zhi_reg[r] + hi;
                const double err    = hi - (new_hi - Zhi_reg[r]);
                Zhi_reg[r] = new_hi;
                if constexpr (HAS_LO) Zlo_reg[r] = fma(dc, qlo, Zlo_reg[r] + err);
                else                  Zlo_reg[r] += err;
            }
        } else {
            for (int r = 0; r < NREG; ++r) {
                const int idx = r * BLK_THR + tid;
                double Zhi = Zhi_lds[idx];
                double Zlo = Zlo_lds[idx];
                const double dc_raw = static_cast<double>(
                    C32[r / NREG_single][r % NREG_single]);
                const double dc     = fma(nm, rint(dc_raw * im), dc_raw);
                const double hi     = dc * qhi;
                const double new_hi = Zhi + hi;
                const double err    = hi - (new_hi - Zhi);
                Zhi = new_hi;
                if constexpr (HAS_LO) Zlo = fma(dc, qlo, Zlo + err);
                else                  Zlo += err;
                Zhi_lds[idx] = Zhi;
                Zlo_lds[idx] = Zlo;
            }
        }
    };

    /* ── K-loop body helper ────────────────────────────────────────────────────
     * Runs the K-loop (LDS fill, global prefetch, MFMA, drain) for one modulus.
     * Used by both sequential and pipelined S-loops.                           */
    auto run_kloop = [&](int s, mfma_acc_t (&C32)[N_TILES])
                     __attribute__((always_inline)) {
        const int8_t* A8i_s = A8i + static_cast<size_t>(s) * stride_A_s;
        const int8_t* B8i_s = B8i + static_cast<size_t>(s) * stride_B_s;

        if (k_int > 0) {
            __syncthreads();

            /* Store K-block 0 (already in rA/rB) to LDS[0].
             * For LOAD_BYTES=16 (dwordx4), split into two 8-byte LDS stores
             * because KBLK_STRIDE may not be 16-byte aligned (KBLK_PAD=8). */
            #pragma unroll
            for (int ls = 0; ls < A_STEPS; ++ls) {
                if constexpr (LOAD_BYTES == 16) {
                    *reinterpret_cast<int64_t*>(&A8i_lds[0][lds_off_A[ls]])     = rA[ls].x;
                    *reinterpret_cast<int64_t*>(&A8i_lds[0][lds_off_A[ls] + 8]) = rA[ls].y;
                } else {
                    *reinterpret_cast<load_t*>(&A8i_lds[0][lds_off_A[ls]]) = rA[ls];
                }
            }
            #pragma unroll
            for (int ls = 0; ls < B_STEPS; ++ls) {
                if constexpr (LOAD_BYTES == 16) {
                    *reinterpret_cast<int64_t*>(&B8i_lds[0][lds_off_B[ls]])     = rB[ls].x;
                    *reinterpret_cast<int64_t*>(&B8i_lds[0][lds_off_B[ls] + 8]) = rB[ls].y;
                } else {
                    *reinterpret_cast<load_t*>(&B8i_lds[0][lds_off_B[ls]]) = rB[ls];
                }
            }

            /* K-loop global-prefetch cursors.  Advance in place by compile-
             * time-constant KBLK_LOAD to keep per-iteration address math to
             * a single add per pointer.                                       */
            const int8_t* A_cur[A_STEPS_SAFE];
            const int8_t* B_cur[B_STEPS_SAFE];
            #pragma unroll
            for (int ls = 0; ls < A_STEPS; ++ls)
                A_cur[ls] = A8i_s + hbm_off_A[ls];
            #pragma unroll
            for (int ls = 0; ls < B_STEPS; ++ls)
                B_cur[ls] = B8i_s + hbm_off_B[ls];

            /* PGR=2 prologue: also load and store K-block 1 to LDS[1]. */
            if constexpr (PGR >= 2) {
                if (KBLK_LOAD < k_int) {
                    #pragma unroll
                    for (int ls = 0; ls < A_STEPS; ++ls) {
                        A_cur[ls] += KBLK_LOAD;
                        rA[ls] = *reinterpret_cast<const load_t*>(A_cur[ls]);
                    }
                    #pragma unroll
                    for (int ls = 0; ls < B_STEPS; ++ls) {
                        B_cur[ls] += KBLK_LOAD;
                        rB[ls] = *reinterpret_cast<const load_t*>(B_cur[ls]);
                    }
                    #pragma unroll
                    for (int ls = 0; ls < A_STEPS; ++ls) {
                        if constexpr (LOAD_BYTES == 16) {
                            *reinterpret_cast<int64_t*>(&A8i_lds[1][lds_off_A[ls]])     = rA[ls].x;
                            *reinterpret_cast<int64_t*>(&A8i_lds[1][lds_off_A[ls] + 8]) = rA[ls].y;
                        } else {
                            *reinterpret_cast<load_t*>(&A8i_lds[1][lds_off_A[ls]]) = rA[ls];
                        }
                    }
                    #pragma unroll
                    for (int ls = 0; ls < B_STEPS; ++ls) {
                        if constexpr (LOAD_BYTES == 16) {
                            *reinterpret_cast<int64_t*>(&B8i_lds[1][lds_off_B[ls]])     = rB[ls].x;
                            *reinterpret_cast<int64_t*>(&B8i_lds[1][lds_off_B[ls] + 8]) = rB[ls].y;
                        } else {
                            *reinterpret_cast<load_t*>(&B8i_lds[1][lds_off_B[ls]]) = rB[ls];
                        }
                    }
                }
            }

            __syncthreads();

            int cur = 0;

            /* Main K-loop: prefetch PGR K-blocks ahead, compute from LDS[cur].
             * For PGR=1 (N_BUF=2): identical to the original double-buffer loop.
             * For PGR=2 (N_BUF=3): triple-buffer with 2 K-blocks of read-ahead,
             *   giving global reads two full MFMA cycles to complete.           */
            for (int k_off = 0;
                 k_off + PGR * KBLK_LOAD < k_int;
                 k_off += KBLK_LOAD) {
                const int buf_store = (cur + PGR >= N_BUF)
                                    ? cur + PGR - N_BUF
                                    : cur + PGR;
                #pragma unroll
                for (int ls = 0; ls < A_STEPS; ++ls) {
                    A_cur[ls] += KBLK_LOAD;
                    rA[ls] = *reinterpret_cast<const load_t*>(A_cur[ls]);
                }
                #pragma unroll
                for (int ls = 0; ls < B_STEPS; ++ls) {
                    B_cur[ls] += KBLK_LOAD;
                    rB[ls] = *reinterpret_cast<const load_t*>(B_cur[ls]);
                }
                apply_mfma(&A8i_lds[cur][wm * (WaveM * TILE * KBLK_STRIDE)], &B8i_lds[cur][wn * (WaveN * TILE * KBLK_STRIDE)], C32);
                #pragma unroll
                for (int ls = 0; ls < A_STEPS; ++ls) {
                    if constexpr (LOAD_BYTES == 16) {
                        *reinterpret_cast<int64_t*>(&A8i_lds[buf_store][lds_off_A[ls]])     = rA[ls].x;
                        *reinterpret_cast<int64_t*>(&A8i_lds[buf_store][lds_off_A[ls] + 8]) = rA[ls].y;
                    } else {
                        *reinterpret_cast<load_t*>(&A8i_lds[buf_store][lds_off_A[ls]]) = rA[ls];
                    }
                }
                #pragma unroll
                for (int ls = 0; ls < B_STEPS; ++ls) {
                    if constexpr (LOAD_BYTES == 16) {
                        *reinterpret_cast<int64_t*>(&B8i_lds[buf_store][lds_off_B[ls]])     = rB[ls].x;
                        *reinterpret_cast<int64_t*>(&B8i_lds[buf_store][lds_off_B[ls] + 8]) = rB[ls].y;
                    } else {
                        *reinterpret_cast<load_t*>(&B8i_lds[buf_store][lds_off_B[ls]]) = rB[ls];
                    }
                }
                __syncthreads();
                cur = (cur + 1 >= N_BUF) ? 0 : cur + 1;
            }
            /* Drain remaining K-blocks (PGR=1: always 1, PGR=2: 1 or 2). */
            apply_mfma(&A8i_lds[cur][wm * (WaveM * TILE * KBLK_STRIDE)], &B8i_lds[cur][wn * (WaveN * TILE * KBLK_STRIDE)], C32);
            if constexpr (PGR >= 2) {
                /* Second drain only if K was large enough to fill LDS[1]. */
                if (KBLK_LOAD < k_int) {
                    __syncthreads();
                    cur = (cur + 1 >= N_BUF) ? 0 : cur + 1;
                    apply_mfma(&A8i_lds[cur][wm * (WaveM * TILE * KBLK_STRIDE)], &B8i_lds[cur][wn * (WaveN * TILE * KBLK_STRIDE)], C32);
                }
            }

            /* After the last MFMA, rA/rB are dead (last K-block written to LDS).
             * Reuse them to preload the next modulus's first K-block from HBM.
             * The global loads overlap with the CRT FP64 work below, hiding
             * KBLK_LOAD = K_UNROLL × KBLK bytes of latency per A and B.      */
            if (s + 1 < S) {
                #pragma unroll
                for (int ls = 0; ls < A_STEPS; ++ls)
                    rA[ls] = *reinterpret_cast<const load_t*>(A8i + static_cast<size_t>(s + 1) * stride_A_s + hbm_off_A[ls]);
                #pragma unroll
                for (int ls = 0; ls < B_STEPS; ++ls)
                    rB[ls] = *reinterpret_cast<const load_t*>(B8i + static_cast<size_t>(s + 1) * stride_B_s + hbm_off_B[ls]);
            }
        }
    };  /* end run_kloop */

    /* ── S-loop: iterate over moduli ───────────────────────────────────────────
     * Sequential: K-loop then CRT for each modulus.  The compiler's instruction
     * scheduler naturally overlaps FP64 CRT ops with the MFMA pipeline drain
     * (confirmed by ISA analysis — see PLR investigation notes).     */
    for (int s = 0; s < S; ++s) {
        mfma_acc_t C32[N_TILES] = {};
        run_kloop(s, C32);
        do_crt(C32, s);
    }

    /* ── Finalize: CRT range-reduction + inverse scale + write D ─────────────
     * Coalesced output path: per-wave LDS transpose for row-consecutive D writes.
     *
     * MFMA output layout: col = lane%TILE, row = 4*(lane/TILE) + 8*(e/4) + (e%4).
     * Problem: each lane writes to a different column → stride-ldd → no coalescing.
     * Fix: store d_val to LDS in column-major layout, read back in row-major order
     * so that consecutive threads write consecutive rows → stride-1 → coalesced.
     * Also coalesces C reads (when beta != 0).
     *
     * LDS reuse: A8i_lds/B8i_lds are dead after the S-loop; reinterpret as double*.
     * Each wave gets TILE*(TILE+1) doubles; the +1 padding ensures that 4 groups
     * of 16 lanes (different cols, same rows) hit different banks (no conflict).
     *
     * Falls back to scattered stores when LDS is too small (e.g., large TILE=32). */
    static constexpr int TILE_OUT_STRIDE = TILE + 1;
    static constexpr size_t   TILE_OUT_BYTES  = static_cast<size_t>(WM * WN)
                                              * TILE * TILE_OUT_STRIDE * sizeof(double);
    static constexpr bool USE_COALESCED_OUT   = (TILE_OUT_BYTES
                                              <= static_cast<size_t>(N_BUF) * (A_BUF_BYTES + B_BUF_BYTES));

    if constexpr (USE_COALESCED_OUT) {
        __syncthreads();  /* ensure all waves done with CRT / K-loop LDS before reuse */
        /* Reinterpret MFMA LDS (mfma_lds_flat, dead after S-loop) as output tile. */
        double* tile_out = reinterpret_cast<double*>(mfma_lds_flat);

        for (int wm_w = 0; wm_w < WaveM; ++wm_w) {
            for (int wn_w = 0; wn_w < WaveN; ++wn_w) {
                const int m_wave_base   = m_base + wm_w * TILE;
                const int n_wave_base   = n_base + wn_w * TILE;
                const int mfma_col      = lane % TILE;
                const int mfma_row_base = 4 * (lane / TILE);
                const int reg_off       = (wm_w * WaveN + wn_w) * NREG_single;
                double* my_tile = tile_out + wid * TILE * TILE_OUT_STRIDE;

                /* Phase 1: compute d_val in MFMA register order → LDS column-major.
                 * Each lane stores its NREG_single elements to
                 * my_tile[mfma_col * TILE_OUT_STRIDE + tile_row].                  */
                for (int e = 0; e < NREG_single; ++e) {
                    const int tile_row = mfma_row_base + 8 * (e / 4) + (e % 4);
                    const int gi = m_wave_base + tile_row;
                    const int gj = n_wave_base + mfma_col;
                    double Zhi, Zlo;
                    if constexpr (USE_LDS_ACCUM) {
                        Zhi = Zhi_lds[(reg_off + e) * BLK_THR + tid];
                        Zlo = Zlo_lds[(reg_off + e) * BLK_THR + tid];
                    } else {
                        Zhi = Zhi_reg[reg_off + e];
                        Zlo = Zlo_reg[reg_off + e];
                    }
                    const double q = rint((Zhi + Zlo) * oz2_inv_P(S - 2));
                    const double X = fma(oz2_P_lo(S - 2), q,
                                         fma(oz2_P_hi(S - 2), q, Zhi) + Zlo);
                    const int inv_sft = (gi < static_cast<int>(m) && gj < static_cast<int>(n))
                                      ? -(static_cast<int>(sftA[gi]) + static_cast<int>(sftB[gj]))
                                      : 0;
                    my_tile[mfma_col * TILE_OUT_STRIDE + tile_row]
                        = alpha * ldexp(X, inv_sft);
                }
                /* No __syncthreads() needed: each wave reads only its own LDS partition. */

                /* Phase 2: read LDS in row-consecutive order → coalesced D write.
                 * 64 threads, TILE×TILE elements → NREG_single passes.
                 * Pass j: thread t handles linear element j*64 + lane.
                 * row = lin % TILE, col = lin / TILE.
                 * Threads 0..15 → rows 0..15 of the same column → coalesced!     */
                for (int j = 0; j < NREG_single; ++j) {
                    const int lin = j * 64 + lane;
                    const int r = lin % TILE;
                    const int c = lin / TILE;
                    const int gi = m_wave_base + r;
                    const int gj = n_wave_base + c;
                    if (gi < static_cast<int>(m) && gj < static_cast<int>(n)) {
                        double val = my_tile[c * TILE_OUT_STRIDE + r];
                        if (beta != 0.0) {
                            val += beta * C[static_cast<size_t>(gi) + static_cast<size_t>(gj) * ldc];
                        }
                        __builtin_nontemporal_store(val,
                            D + static_cast<size_t>(gi) + static_cast<size_t>(gj) * ldd);
                    }
                }
            }
        }
    } else {
        /* Fallback: scattered stores (original path, used when LDS is too small). */
        for (int wm_w = 0; wm_w < WaveM; ++wm_w) {
            for (int wn_w = 0; wn_w < WaveN; ++wn_w) {
                const int m_wave_base   = m_base + wm_w * TILE;
                const int n_wave_base   = n_base + wn_w * TILE;
                const int col_val       = n_wave_base + (lane % TILE);
                const int lane_row_base = m_wave_base + 4 * (lane / TILE);
                const int reg_off       = (wm_w * WaveN + wn_w) * NREG_single;
                for (int e = 0; e < NREG_single; ++e) {
                    const int ri = lane_row_base + 8 * (e / 4) + (e % 4);
                    const int ci = col_val;
                    if (ri >= static_cast<int>(m) || ci >= static_cast<int>(n)) continue;
                    double Zhi, Zlo;
                    if constexpr (USE_LDS_ACCUM) {
                        Zhi = Zhi_lds[(reg_off + e) * BLK_THR + tid];
                        Zlo = Zlo_lds[(reg_off + e) * BLK_THR + tid];
                    } else {
                        Zhi = Zhi_reg[reg_off + e];
                        Zlo = Zlo_reg[reg_off + e];
                    }
                    const double q = rint((Zhi + Zlo) * oz2_inv_P(S - 2));
                    const double X = fma(oz2_P_lo(S - 2), q,
                                         fma(oz2_P_hi(S - 2), q, Zhi) + Zlo);
                    const int inv_sft = -(static_cast<int>(sftA[ri])
                                         + static_cast<int>(sftB[ci]));
                    const size_t d_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldd;
                    double d_val = alpha * ldexp(X, inv_sft);
                    if (beta != 0.0) {
                        const size_t c_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldc;
                        d_val += beta * C[c_idx];
                    }
                    __builtin_nontemporal_store(d_val, D + d_idx);
                }
            }
        }
    }
}

/* =========================================================================
 * oz2_fused_mode — env-var reader (re-read each call, not cached)
 *
 * Not cached so that callers can change HIPBLASLT_EMULATION_FUSED at
 * runtime (e.g. between the split-decision check and the monolithic GEMM
 * launch) and still observe the correct mode.  The overhead is a single
 * getenv() call per GEMM invocation — negligible vs hundreds-of-ms GEMMs.
 * ========================================================================= */
Oz2FusedMode oz2_fused_mode()
{
    const char* e = std::getenv("HIPBLASLT_EMULATION_FUSED");
    if(e == nullptr) return Oz2FusedMode::OFF;   /* disabled by default */
    if(std::strcmp(e, "on") == 0 || std::strcmp(e, "force") == 0)
        return Oz2FusedMode::ON;
    if(std::strcmp(e, "off") == 0 || std::strcmp(e, "never") == 0)
        return Oz2FusedMode::OFF;
    return Oz2FusedMode::AUTO;   /* "auto", "performant", or unrecognized */
}

/* =========================================================================
 * oz2_launch_fused_TN — host-side launcher for the fused TN kernel
 *
 * Shape heuristic (tuned on MI300X/gfx942, S=16 moduli; global-warmup to drain
 *   burst clock + full workspace pre-allocated to eliminate hipMalloc overhead;
 *   shapes M=32768 N=256 K=32768 / M=256 N=32768 K=32768 / M=N=32768 K=1024):
 *
 *   Tall  (m >= 4n): WM4WN4Wm4Wn2T16, KU=1 — 256×128 macrotile, 1024 threads.
 *     Measured 16219 GFLOP/s for tall (M=32768, N=256, K=32768).
 *     Large WaveM=4 amortises scale/MFMA startup; KU=1 reduces per-barrier
 *     KBLK_LOAD for very large K.
 *
 *   Wide  (n >= 4m): WM2WN4Wm1Wn2T16, KU=2 — 32×128 macrotile (TILE=16).
 *     Measured 10734 GFLOP/s for wide (M=256, N=32768, K=32768).
 *     32×128 with WaveN=2 and KU=2 outperforms TILE=32 by 3.3%; NREG_single=4
 *     for TILE=16 avoids the VGPR pressure that slows TILE=32 on gfx942.
 *
 *   Square / near-square: WM4WN4Wm1Wn2T16, KU=4 — 64×128 macrotile,
 *     1024 threads.  Measured 24771 GFLOP/s vs 24233 for 128×64 (+2%).
 *     WaveN=2 processes 2 N-MFMA-tiles per block; slightly better than
 *     WaveM=2 (128×64) for square shapes with small K.
 *
 * TILE=32 instantiations are compiled for OZ2_FUSED_SHAPE_OVERRIDE tuning
 * and for future MI350 re-evaluation where USE_LDS_ACCUM=true may remove
 * the VGPR spilling penalty that makes TILE=32 ~10× slower on gfx942.
 * ========================================================================= */
rocblaslt_status oz2_launch_fused_TN(
    const int8_t*  A8i,
    const int8_t*  B8i,
    size_t         lda8i,
    size_t         cola8i,
    size_t         ldb8i,
    const double*  C,
    double*        D,
    int64_t m, int64_t n, int64_t k,
    int64_t ldc, int64_t ldd,
    double alpha, double beta,
    const int16_t* sftA, const int16_t* sftB,
    unsigned num_moduli, hipStream_t stream)
{
    const size_t stride_A_s = lda8i * cola8i;
    const size_t stride_B_s = ldb8i * static_cast<size_t>(n);
    const bool   has_lo     = (num_moduli > 7u);

    /* Query current device once; used for both XCC count and gfx950 detection. */
    int cur_dev = 0;
    (void)hipGetDevice(&cur_dev);

    /* TILE=16 is used for all shapes and architectures in the fused kernel.
     *
     * Empirical tuning on MI300X (gfx942) shows TILE=32 is ~10× slower than
     * TILE=16 for all tested shapes.  The root cause is register spilling:
     * TILE=32 → NREG_single=16 doubles per wave tile → Zhi_reg[]+Zlo_reg[]
     * consume ~128 VGPRs (when USE_LDS_ACCUM=false, which is always the case
     * on MI300X with its 63 KB LDS budget), causing heavy scratch traffic.
     * TILE=16 → NREG_single=4 → only ~32 VGPRs for accumulators, fits cleanly.
     *
     * TILE=32 instantiations are still compiled (used by OZ2_FUSED_SHAPE_OVERRIDE
     * for tuning experiments) and remain available for future re-evaluation on
     * MI350 where USE_LDS_ACCUM=true may eliminate the VGPR pressure.           */
    int is_gfx950 = 0;
    {
        int gfx950_chip_id = 0;
        (void)hipDeviceGetAttribute(&gfx950_chip_id, hipDeviceAttributePciChipId, cur_dev);
        const uint32_t pci_id = static_cast<uint32_t>(gfx950_chip_id) & 0xFFFFu;
        is_gfx950 = (pci_id == 0x75a3u || pci_id == 0x75b3u     /* MI350X */
                  || pci_id == 0x75a0u || pci_id == 0x75b0u) ? 1 : 0;  /* MI355X */
    }
    (void)is_gfx950;                  /* reserved for future gfx950-specific tuning */
    /* Query number of XCCs (Graphics Compute Dies) on the current device. */
    int num_xccs = 1;
    {
        (void)hipDeviceGetAttribute(&num_xccs, hipDeviceAttributeNumberOfXccs, cur_dev);
        if(num_xccs <= 0) num_xccs = 1;
    }

    /* Compute 1D grid for a given macrotile configuration.
     * Macrotile M dimension = wm_v * wm_wave_v * tile_v (similarly for N).
     * Thread block size = wm_v * wn_v * 64 (WaveM/WaveN don't add threads).
     * With XCC mapping, total_blocks = m_tiles_per_xcc × n_tiles × num_xccs. */
    auto make_grid = [&](int wm_v, int wn_v, int tile_v,
                         int wm_wave_v = 1, int wn_wave_v = 1) -> dim3 {
        const int mt = static_cast<int>(
            (m + wm_v * wm_wave_v * tile_v - 1) / (wm_v * wm_wave_v * tile_v));
        const int nt = static_cast<int>(
            (n + wn_v * wn_wave_v * tile_v - 1) / (wn_v * wn_wave_v * tile_v));
        /* Match the kernel's swizzle dimension: M for tall/square, N for wide.
         * Uses raw matrix dimensions (not tile counts) because L2 reuse depends
         * on which matrix operand is smaller, not the macrotile shape.          */
        int total_blocks;
        if (n <= m) {
            const int m_per_xcc = (mt + num_xccs - 1) / num_xccs;
            total_blocks = m_per_xcc * nt * num_xccs;
        } else {
            const int n_per_xcc = (nt + num_xccs - 1) / num_xccs;
            total_blocks = mt * n_per_xcc * num_xccs;
        }
        return dim3(static_cast<unsigned>(total_blocks), 1u);
    };

/* OZ2_FUSED_LAUNCH: dispatch oz2_fused_TN_kernel with explicit KU_PARAM and FVA_V.
 * KU_V=0 → auto-select K_UNROLL from LDS budget (same as omitting the param).
 * KU_V=2,4 → force a specific K_UNROLL for tuning.
 * FVA_V=false → use LDS accumulators when they fit (default).
 * FVA_V=true  → force VGPR-based accumulators even when LDS has space.        */
/* LB_V=0 → auto-select LOAD_BYTES; LB_V=4/8/16 → override for tuning. */
#define OZ2_FUSED_LAUNCH(S_V, HL, WM_V, WN_V, TILE_V, WM_WAVE_V, WN_WAVE_V, KU_V, FVA_V, PGR_V, ...) \
    hipLaunchKernelGGL((oz2_fused_TN_kernel<(S_V),(HL),(WM_V),(WN_V),(TILE_V),(WM_WAVE_V),(WN_WAVE_V),(KU_V),(FVA_V),(PGR_V),##__VA_ARGS__>), \
                       make_grid((WM_V),(WN_V),(TILE_V),(WM_WAVE_V),(WN_WAVE_V)), \
                       dim3((WM_V)*(WN_V)*64), 0, stream, \
                       A8i, stride_A_s, lda8i, B8i, stride_B_s, ldb8i, \
                       C, D, m, n, k, ldc, ldd, alpha, beta, sftA, sftB, \
                       num_xccs)


/* ── OZ2_DISPATCH_SHAPE: normal shape selection macro ──────────────────────────
 * Used for the per-num_moduli switch below when OZ2_FUSED_SHAPE_OVERRIDE is not
 * set (or not matched).  Selects the default TILE=16 or TILE=32 config.
 *
 * OZ2_DISPATCH_SHAPE_OVERRIDE is called ONCE before the switch, fixed at S=16.
 * If the override env var is set and matched, it dispatches
 * oz2_fused_TN_kernel<16,...> and returns immediately, bypassing the switch.
 * The tuning driver must fix num_moduli=16 (maxBits=118) so that the 16 sets of
 * INT8 data prepared by the library match the kernel's S=16 loop.
 *
 * OZ2_FUSED_SHAPE_OVERRIDE format: "WM WN WaveM WaveN TILE KU"  (6 params).
 * KU=0 → auto-select K_UNROLL from LDS budget.
 * KU=2 or KU=4 → explicit K_UNROLL override (useful for MI350X tuning).       */

/* Helper: dispatch one config.  FVA_V is a literal 0 or 1 (converted to
 * false/true for the bool FORCE_VGPR_ACCUM template parameter).
 * The dispatch table encodes _fva==0 / _fva==1 in the condition, so each call
 * site always passes a compile-time constant — no runtime branch needed here.  */
/* _lb is defined in the enclosing scope: parsed from override string, or 0
 * (auto) in OZ2_DISPATCH_SHAPE.  Passed as the variadic 11th template arg
 * (LB_PARAM) to OZ2_FUSED_LAUNCH — when 0, LOAD_BYTES auto-selects.       */
#define _OV_DISPATCH(S_V,WM_V,WN_V,T_V,WMW_V,WNW_V,KU_V,FVA_V,PGR_V,LB_V) \
    do { \
        if(has_lo) OZ2_FUSED_LAUNCH((S_V),true, WM_V,WN_V,T_V,WMW_V,WNW_V,KU_V,FVA_V,PGR_V,(LB_V)); \
        else       OZ2_FUSED_LAUNCH((S_V),false,WM_V,WN_V,T_V,WMW_V,WNW_V,KU_V,FVA_V,PGR_V,(LB_V)); \
    } while(0)

#define OZ2_DISPATCH_SHAPE_OVERRIDE(S_V) \
    do { \
        const char* _ov = std::getenv("OZ2_FUSED_SHAPE_OVERRIDE"); \
        if (_ov) { \
            int _wm=0,_wn=0,_wm_w=0,_wn_w=0,_t=0,_ku=0,_fva=0,_pgr=1,_lb=0; \
            { std::sscanf(_ov,"%d %d %d %d %d %d %d %d %d",&_wm,&_wn,&_wm_w,&_wn_w,&_t,&_ku,&_fva,&_pgr,&_lb); } \
            if (_wm && _wn && (_t==16||_t==32)) { \
                /* OZ2_DISPATCH_SHAPE_OVERRIDE always dispatches oz2_fused_TN_kernel<16,...>. \
                 * The caller MUST fix num_moduli=16 via the emulation handle so that the \
                 * library prepares exactly 16 INT8 data sets to match the kernel's S=16 loop. \
                 * Abort loudly rather than silently reading out-of-bounds GPU memory. */ \
                if (num_moduli != 16u) { \
                    std::fprintf(stderr, \
                        "[oz2_launch_fused_TN] FATAL: OZ2_FUSED_SHAPE_OVERRIDE is set but " \
                        "num_moduli=%u (expected 16). Fix the emulation handle: call " \
                        "hipblasLtSetFixedPointEmulationMantissaControl(FIXED) and " \
                        "hipblasLtSetFixedPointEmulationMaxMantissaBitCount(118).\n", \
                        num_moduli); \
                    std::abort(); \
                } \
                if      (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,2,32,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,4,32,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,2,16,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,1,16,1,2,4,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,2,16,2,1,4,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,1,16,2,2,4,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,1,32,1,1,4,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,1,16,1,1,4,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,1,16,1,2,4,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,2,16,1,1,4,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,1,16,2,1,4,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,1,16,1,1,4,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,4,16,2,1,4,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,2,16,1,2,4,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,2,16,2,2,4,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,4,16,4,1,4,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,1,16,1,4,4,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,2,16,4,2,4,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,1,16,2,4,4,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,1,16,4,4,4,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,4,16,2,1,4,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,2,4,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,2,16,2,1,4,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,4,16,1,2,4,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,2,16,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,4,16,1,1,4,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,1,16,1,1,4,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,4,16,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,2,32,1,1,4,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2,1,32,1,1,4,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,2,32,1,1,4,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4,1,32,1,1,4,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1,4,32,1,1,4,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,4,16,2,1,2,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,4,16,4,1,2,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,4,2,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,4,16,2,2,2,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,4,16,2,1,2,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,2,16,1,2,2,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,2,16,2,2,2,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,4,16,4,1,2,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,1,16,1,4,2,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,2,16,4,2,2,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,1,16,2,4,2,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,1,16,4,4,2,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,2,2,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,2,16,2,1,2,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,4,16,1,2,2,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,2,32,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,4,32,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,2,16,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,1,16,1,2,2,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,2,16,2,1,2,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,1,16,2,2,2,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,1,32,1,1,2,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,1,16,1,1,2,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,1,16,1,2,2,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,2,16,1,1,2,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,1,16,2,1,2,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,1,16,1,1,2,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,2,16,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,4,16,1,1,2,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,1,16,1,1,2,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,4,16,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,2,32,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2,1,32,1,1,2,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,2,32,1,1,2,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4,1,32,1,1,2,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1,4,32,1,1,2,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,4,16,2,1,1,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4,2,16,1,2,1,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,2,16,2,2,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4,4,16,2,2,1,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,1,16,4,4,1,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,4,16,4,1,1,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4,1,16,1,4,1,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,2,16,4,2,1,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,1,16,2,4,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4,4,16,4,2,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4,4,16,2,4,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4,4,16,4,4,1,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,2,16,1,1,1,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,1,16,1,2,1,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,2,16,2,1,1,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,1,16,2,2,1,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,1,32,1,1,1,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,1,16,1,1,1,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,1,16,1,2,1,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,2,16,1,1,1,false,1,0); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,1,16,2,1,1,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,1,16,1,1,1,false,1,0); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,2,32,1,1,1,false,1,0); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2,1,32,1,1,1,false,1,0); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1,2,32,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,2,1,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,2,16,2,1,1,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),2,4,16,1,2,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,4,16,2,1,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,4,16,4,1,1,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,2,32,1,1,1,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),2,4,32,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,4,16,1,4,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,4,32,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,2,16,1,1,1,false,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),2,4,16,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,1,16,1,1,1,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),1,4,16,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4,1,32,1,1,1,false,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),1,4,32,1,1,1,false,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,2,1,true,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,2,16,2,1,1,true,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2,4,16,1,2,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,1,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,2,1,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,4,1,1,true,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,2,32,1,1,1,true,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2,4,32,1,1,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,4,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,32,1,1,1,true,1,0); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,2,16,1,1,1,true,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2,4,16,1,1,1,true,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,1,16,1,1,1,true,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),1,4,16,1,1,1,true,1,0); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,1,32,1,1,1,true,1,0); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),1,4,32,1,1,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,1,4,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,2,4,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,2,1,4,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,4,2,1,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,2,2,2,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,4,1,2,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,4,2,true,1,0); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4,4,16,1,1,2,true,1,0); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2,4,32,1,1,4,true,1,0); \
                /* ── PGR=2 (triple-buffered K-loop) — benefits larger K ──── */ \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,2,2,2,false,2,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&_fva==0&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,4,2,1,false,2,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,4,4,1,false,2,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,2,2,1,false,2,0); \
                /* ── FVA=1 + PGR=2 (gfx950 only) ──────────────────────── */ \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,2,2,1,true,2,0); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&is_gfx950&&_fva==1&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,4,4,1,true,2,0); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&is_gfx950&&_fva==1&&_pgr==2) _OV_DISPATCH((S_V),4,4,16,2,2,2,true,2,0); \
                /* ── LB_PARAM override (LOAD_BYTES tuning) ─────────────────── */ \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0&&_lb==4)  _OV_DISPATCH((S_V),4,4,16,2,2,2,false,1,4); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0&&_lb==8)  _OV_DISPATCH((S_V),4,4,16,2,2,2,false,1,8); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0&&_lb==16) _OV_DISPATCH((S_V),4,4,16,2,2,2,false,1,16); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0&&_lb==4)  _OV_DISPATCH((S_V),4,4,16,4,4,1,false,1,4); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0&&_lb==8)  _OV_DISPATCH((S_V),4,4,16,4,4,1,false,1,8); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0&&_lb==16) _OV_DISPATCH((S_V),4,4,16,4,4,1,false,1,16); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0&&_lb==4)  _OV_DISPATCH((S_V),4,4,16,2,2,1,false,1,4); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0&&_lb==8)  _OV_DISPATCH((S_V),4,4,16,2,2,1,false,1,8); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0&&_lb==16) _OV_DISPATCH((S_V),4,4,16,2,2,1,false,1,16); \
                else { return rocblaslt_status_invalid_value; } \
                return rocblaslt_status_success; \
            } \
        } \
    } while(0)

#define OZ2_DISPATCH_SHAPE(S_V) \
        do { /* Shape heuristic — two architectures, one macro.                        \
         * gfx942 (MI300X):                                                       \
         *   DEFAULT: WM4WN4Wm2Wn2T16ku2 (128×128, KU=2) wins for all square     \
         *     and near-square shapes across the full K range (K=64..32768).      \
         *     Measured on gfx942 MI300X (32768×32768, S=16, global-warmup):      \
         *       K=64:  6300   K=128: 12446   K=256: 18716   K=512: 24707        \
         *       K=1024: 28943  K=2048: 31165  K=32768: 32149 GFLOP/s            \
         *   ELONGATED (tall M≫N or wide N≫M) with large K: the 256×128         \
         *     macrotile WM4WN4Wm4Wn2T16ku1 is faster:                           \
         *       tall M=32768,N=256,K=32768: 16354 GFLOP/s (+5% vs 128×128)      \
         *       wide M=256,N=32768,K=32768: 15460 GFLOP/s (+4% vs 128×128)      \
         *     256×128 wins BOTH directions (beats 128×256 on wide too), because  \
         *     the longer M-macrotile amortises scale/MFMA startup over the       \
         *     skinny dimension while KU=1 keeps per-barrier KBLK_LOAD small for  \
         *     very large K.  Gated on large K (≥4096) so small-K elongated       \
         *     shapes keep the 128×128 KU=2 default.                              \
         *                                                                        \
         * gfx950 (MI355X): 128×128 KU=2 for small-K / tall / wide; 256×256 KU=1 \
         *   for large symmetric shapes (M,N,K ≥ 8192, +23%).  The gfx942        \
         *   elongated 256×128 branch is gfx942-only (unmeasured on gfx950).      \
         * Threshold M,N,K ≥ 8192 ensures ≥1024 output tiles for 256×256.        */ \
        if (is_gfx950) { \
            /* ── gfx950 (MI350X) tile selection ────────────────────────── \
             * Tuned on MI350X (Aug 2, 2026 run, 32768² shapes, S=16):     \
             *   256×256 KU=1: 86412 GFLOP/s for M,N,K ≥ 8192             \
             *   128×128 KU=2: 48037 GFLOP/s for K ≥ 2048 (beats KU=1     \
             *     46407 at K=2048, 66354 vs 49817 at K=32768)             \
             *   128×128 KU=1: 11100-42091 GFLOP/s for K < 2048           \
             *     (beats KU=2 at K≤1024: 42091 vs 36897)                  */ \
            if (m >= 8192 && n >= 8192 && k >= 8192) { \
                _OV_DISPATCH((S_V),4,4,16,4,4,1,false,1,0);  /* 256×256, KU=1 */ \
            } else if (k >= 2048) { \
                _OV_DISPATCH((S_V),4,4,16,2,2,2,false,1,0);  /* 128×128, KU=2 */ \
            } else { \
                _OV_DISPATCH((S_V),4,4,16,2,2,1,false,1,0);  /* 128×128, KU=1 */ \
            } \
        } else { \
            /* ── gfx942 (MI300X) tile selection ────────────────────────── */ \
            if (m >= 8192 && n >= 8192 && k >= 8192) { \
                _OV_DISPATCH((S_V),4,4,16,4,4,1,false,1,0);  /* 256×256, KU=1 (37237 GFLOP/s vs PGR2 36572) */ \
            } else { \
                _OV_DISPATCH((S_V),4,4,16,2,2,2,false,1,0);  /* 128×128, KU=2 */ \
            } \
        } \
    } while(0)

    /* Shape override: always dispatches oz2_fused_TN_kernel<16,...>.
     * If OZ2_FUSED_SHAPE_OVERRIDE is set, returns here before the switch so
     * the normal per-S dispatch is skipped.  The caller (driver) must fix
     * num_moduli=16 so the kernel's S=16 loop matches the allocated INT8 data. */
    OZ2_DISPATCH_SHAPE_OVERRIDE(16);

    switch(num_moduli) {
        case  2: OZ2_DISPATCH_SHAPE( 2); break;
        case  3: OZ2_DISPATCH_SHAPE( 3); break;
        case  4: OZ2_DISPATCH_SHAPE( 4); break;
        case  5: OZ2_DISPATCH_SHAPE( 5); break;
        case  6: OZ2_DISPATCH_SHAPE( 6); break;
        case  7: OZ2_DISPATCH_SHAPE( 7); break;
        case  8: OZ2_DISPATCH_SHAPE( 8); break;
        case  9: OZ2_DISPATCH_SHAPE( 9); break;
        case 10: OZ2_DISPATCH_SHAPE(10); break;
        case 11: OZ2_DISPATCH_SHAPE(11); break;
        case 12: OZ2_DISPATCH_SHAPE(12); break;
        case 13: OZ2_DISPATCH_SHAPE(13); break;
        case 14: OZ2_DISPATCH_SHAPE(14); break;
        case 15: OZ2_DISPATCH_SHAPE(15); break;
        case 16: OZ2_DISPATCH_SHAPE(16); break;
        case 17: OZ2_DISPATCH_SHAPE(17); break;
        case 18: OZ2_DISPATCH_SHAPE(18); break;
        default: return rocblaslt_status_invalid_value;
    }

#undef _OV_DISPATCH
#undef OZ2_DISPATCH_SHAPE_OVERRIDE
#undef OZ2_DISPATCH_SHAPE
#undef OZ2_FUSED_LAUNCH

    return rocblaslt_status_success;
}
