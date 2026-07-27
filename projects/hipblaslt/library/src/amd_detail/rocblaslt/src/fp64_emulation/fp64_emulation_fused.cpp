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
 * Per-architecture MFMA K-block sizes.
 * TILE=16: K_BLOCK = 32 (gfx94x) or 64 (gfx95x)
 * TILE=32: K_BLOCK = 16 (gfx94x) or 32 (gfx95x)
 * ========================================================================= */
#if defined(__gfx950__)
static constexpr unsigned OZ2_KBLK_16 = 64u;
static constexpr unsigned OZ2_KBLK_32 = 32u;
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
static constexpr unsigned OZ2_KBLK_16 = 32u;
static constexpr unsigned OZ2_KBLK_32 = 16u;
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif

/* =========================================================================
 * MFMA vector output types and instruction wrappers
 * ========================================================================= */

typedef int v4i32  __attribute__((ext_vector_type(4)));
typedef int v16i32 __attribute__((ext_vector_type(16)));

/* TILE=16 MFMA source type and instruction wrapper */
#if defined(__gfx950__)
typedef long oz2_mfma_src16_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ v4i32
oz2_do_mfma_16(oz2_mfma_src16_t a, oz2_mfma_src16_t b, v4i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_16x16x64_i8(a, b, c, 0, 0, 0); }
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
typedef int64_t oz2_mfma_src16_t;
__device__ __forceinline__ v4i32
oz2_do_mfma_16(int64_t a, int64_t b, v4i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, 0, 0); }
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif

/* TILE=32 MFMA source type and instruction wrapper.
 * oz2_mfma_src32_t == oz2_mfma_src16_t on all supported architectures. */
typedef oz2_mfma_src16_t oz2_mfma_src32_t;
#if defined(__gfx950__)
__device__ __forceinline__ v16i32
oz2_do_mfma_32(oz2_mfma_src32_t a, oz2_mfma_src32_t b, v16i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_32x32x32_i8(a, b, c, 0, 0, 0); }
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
__device__ __forceinline__ v16i32
oz2_do_mfma_32(oz2_mfma_src32_t a, oz2_mfma_src32_t b, v16i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, 0, 0); }
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif

/* ── Architecture-specific MFMA source load helper ─────────────────────────
 * On gfx95x (K_A_BYTES=16): two 8-byte loads give 2-way LDS bank conflicts
 * instead of 4-way from a single 16-byte load.
 * On gfx94x (K_A_BYTES=8): single 8-byte load is always conflict-free.       */
#if defined(__gfx950__)
__device__ __forceinline__ oz2_mfma_src16_t
oz2_load_mfma_src16(const int8_t* __restrict__ ptr) noexcept {
    oz2_mfma_src16_t v;
    v[0] = *reinterpret_cast<const long*>(ptr);
    v[1] = *reinterpret_cast<const long*>(ptr + 8);
    return v;
}
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
__device__ __forceinline__ oz2_mfma_src16_t
oz2_load_mfma_src16(const int8_t* __restrict__ ptr) noexcept {
    return *reinterpret_cast<const oz2_mfma_src16_t*>(ptr);
}
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
/* Same helper for TILE=32 source (oz2_mfma_src32_t == oz2_mfma_src16_t). */
__device__ __forceinline__ oz2_mfma_src32_t
oz2_load_mfma_src32(const int8_t* __restrict__ ptr) noexcept {
    return oz2_load_mfma_src16(ptr);
}

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
/* NO_CRT=true skips both the CRT accumulation and the finalize/D-write sections.
 * Results are garbage but kernel time isolates pure MFMA+prefetch overhead.   */
template <unsigned S, bool HAS_LO, unsigned WM = 4u, unsigned WN = 4u, unsigned TILE = 16u,
          unsigned WaveM = 1u, unsigned WaveN = 1u, unsigned KU_PARAM = 0u,
          bool FORCE_VGPR_ACCUM = false, bool NO_CRT = false>
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
    static constexpr unsigned NREG_single = TILE * TILE / 64u;          /* accumulators per thread per MFMA tile */
    static constexpr unsigned NREG        = WaveM * WaveN * NREG_single; /* total accumulators per thread         */
    /* NREG_sub: CRT update sub-tile size.  Keeps Zhi_t/Zlo_t to ≤4 doubles
     * (16 VGPRs) regardless of NREG_single, reducing peak register pressure
     * for TILE=32 when USE_LDS_ACCUM=true (e.g., MI350X TILE=32 NREG_single=16
     * → 64-VGPR Zhi_t/Zlo_t reduced to 16 VGPRs, hiding behind MFMAs).
     * For TILE=16: NREG_single=4 → NREG_sub=4 → trivial 1-iter loop (no-op). */
    static constexpr unsigned NREG_sub = (NREG_single > 4u) ? 4u : NREG_single;
    static constexpr unsigned NREG_sub_iters = NREG_single / NREG_sub;
    static constexpr unsigned KBLK        = (TILE == 16u) ? OZ2_KBLK_16 : OZ2_KBLK_32;
    /* K-loop unroll factor:
     *   gfx94x (MI300): K_UNROLL=4 — measured scratch=108B ✓
     *   gfx95x (MI350): K_UNROLL=2 — KBLK is doubled vs gfx94x; K_UNROLL=4
     *     would double A_STEPS/B_STEPS → same register pressure as gfx94x
     *     K_UNROLL=8 (scratch=388B, 2.5× slower). Confirmed on hardware:
     *     K_UNROLL=4 → scratch=424B ✗  K_UNROLL=2 → scratch=104B ✓          */
#if defined(__gfx950__)
    static constexpr size_t   LDS_BUDGET  = 159u * 1024u;
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
    static constexpr size_t   LDS_BUDGET  = 63u * 1024u;
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
    static constexpr size_t   LDS_MFMA_K4 = 2u * (WM * WaveM * TILE * (KBLK * 4u)
                                                  + WN * WaveN * TILE * (KBLK * 4u));
    static constexpr size_t   LDS_SFT     = static_cast<size_t>((WM * WaveM + WN * WaveN) * TILE * 2u);
    /* K_UNROLL: auto-selected from LDS budget, or overridden via KU_PARAM. */
#if defined(__gfx950__)
    static constexpr unsigned K_UNROLL_AUTO = (TILE == 32u) ? 2u : 4u;
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
      || !defined(__HIP_DEVICE_COMPILE__)  /* host compilation pass */
    static constexpr unsigned K_UNROLL_AUTO = (LDS_MFMA_K4 + LDS_SFT <= LDS_BUDGET) ? 4u : 2u;
#else
#  error "fp64_emulation_fused: unsupported GPU architecture (gfx940/941/942 or gfx950 required)"
#endif
    static constexpr unsigned K_UNROLL = (KU_PARAM > 0u) ? KU_PARAM : K_UNROLL_AUTO;
    static constexpr unsigned KBLK_LOAD   = KBLK * K_UNROLL;
    static constexpr unsigned K_A_BYTES   = KBLK * TILE / 64u;
    static constexpr unsigned KBLK_PAD_UNIT = (K_A_BYTES == 16u) ? 8u : K_A_BYTES;
    static constexpr size_t   LDS_BYTES_OLD = 2u * (WM * WaveM * TILE * KBLK_LOAD
                                                   + WN * WaveN * TILE * KBLK_LOAD);
    static constexpr unsigned KBLK_PAD    = (LDS_BYTES_OLD + KBLK_PAD_UNIT * TILE * (WM * WaveM + WN * WaveN) * 2u
                                             <= LDS_BUDGET) ? KBLK_PAD_UNIT : 0u;
    static constexpr unsigned KBLK_STRIDE = KBLK_LOAD + KBLK_PAD;
    static constexpr unsigned BLK_THR   = WM * WN * 64u;
    static constexpr unsigned K4DIM     = KBLK_LOAD / 4u;
    static constexpr unsigned A_STEPS   = (WM * WaveM * TILE * K4DIM) / BLK_THR;
    static constexpr unsigned B_STEPS   = (WN * WaveN * TILE * K4DIM) / BLK_THR;
    /* VALID_CONFIG: false when A_STEPS or B_STEPS < 1 (gfx950-only KU=1 on gfx942). */
    static constexpr unsigned A_STEPS_SAFE = (A_STEPS > 0u) ? A_STEPS : 1u;
    static constexpr unsigned B_STEPS_SAFE = (B_STEPS > 0u) ? B_STEPS : 1u;
    static constexpr bool     VALID_CONFIG  = (A_STEPS >= 1u && B_STEPS >= 1u);

    /* ── LDS accumulator decision ───────────────────────────────────────────
     * Store Zhi/Zlo in LDS when the combined budget (MFMA tiles + accumulators
     * + scale shifts) fits within LDS_BUDGET.  This enables large WaveM×WaveN
     * configurations (e.g., WM=1,WN=1,WaveM=4,WaveN=4 on MI350) that would
     * otherwise spill registers.  When false (e.g., all current MI300 configs),
     * accumulators stay in VGPRs with zero overhead.                          */
    static constexpr size_t LDS_MFMA      = 2u * (WM * WaveM * TILE * KBLK_STRIDE
                                                 + WN * WaveN * TILE * KBLK_STRIDE);
    static constexpr size_t LDS_ZHI_ZLO   = 2u * BLK_THR * NREG * sizeof(double);
    static constexpr bool   FITS_IN_LDS   = (LDS_MFMA + LDS_SFT + LDS_ZHI_ZLO <= LDS_BUDGET);
    /* USE_LDS_ACCUM: use LDS for Zhi/Zlo only when LDS has room AND FORCE_VGPR_ACCUM is not set.
     * FORCE_VGPR_ACCUM=true keeps accumulators in registers even when LDS has space,
     * which can be faster when NREG is small enough to avoid VGPR spilling.             */
    static constexpr bool   USE_LDS_ACCUM = FITS_IN_LDS && !FORCE_VGPR_ACCUM;

    /* ── Static LDS ────────────────────────────────────────────────────────── */
    __shared__ int8_t  A8i_lds[2][WM * WaveM][TILE][KBLK_STRIDE];
    __shared__ int8_t  B8i_lds[2][WN * WaveN][TILE][KBLK_STRIDE];
    __shared__ int16_t sftA_lds[WM * WaveM * TILE];
    __shared__ int16_t sftB_lds[WN * WaveN * TILE];
    /* CRT accumulator LDS — flat layout [e * BLK_THR + tid].
     * Thread tid owns column tid for accumulator row e, giving consecutive
     * intra-wavefront access (2-way bank conflict; unavoidable for 64-lane
     * waves with 32 LDS banks and 8-byte doubles).
     * When USE_LDS_ACCUM=false, arrays are size 1 (placeholder, never read). */
    __shared__ double Zhi_lds[USE_LDS_ACCUM ? NREG * BLK_THR : 1u];
    __shared__ double Zlo_lds[USE_LDS_ACCUM ? NREG * BLK_THR : 1u];

    /* ── Thread decomposition ───────────────────────────────────────────────── */
    const int tid  = static_cast<int>(threadIdx.x);
    const int wid  = tid / 64;
    const int wm   = wid / static_cast<int>(WN);
    const int wn   = wid % static_cast<int>(WN);
    const int lane = tid % 64;
    const int k_int = static_cast<int>(k);

    /* ── XCC-aware block tile mapping ───────────────────────────────────────── */
    static constexpr int SWIZZLE_B8i_PREF = 4;

    const int m_tiles_total   = (static_cast<int>(m) + static_cast<int>(WM * WaveM * TILE) - 1)
                                / static_cast<int>(WM * WaveM * TILE);
    const int n_tiles_total   = (static_cast<int>(n) + static_cast<int>(WN * WaveN * TILE) - 1)
                                / static_cast<int>(WN * WaveN * TILE);
    const int m_tiles_per_xcc = (m_tiles_total + num_xccs - 1) / num_xccs;
    const int tiles_per_xcc   = m_tiles_per_xcc * n_tiles_total;

    const int swizzle_eff = (SWIZZLE_B8i_PREF > m_tiles_per_xcc) ? m_tiles_per_xcc : SWIZZLE_B8i_PREF;

    uint32_t hw_xcc_id = 0u;
    asm volatile("s_getreg_b32 %0, hwreg(20)" : "=s"(hw_xcc_id));

    const int linear_block = static_cast<int>(blockIdx.x);
    const int xcc_id       = static_cast<int>(hw_xcc_id);
    const int intra_xcc    = linear_block / num_xccs;

    if (intra_xcc >= tiles_per_xcc) return;

    const int swizzle_group = intra_xcc / swizzle_eff;
    const int m_local       = intra_xcc % swizzle_eff;
    const int n_tile        = swizzle_group % n_tiles_total;
    const int m_group       = swizzle_group / n_tiles_total;
    const int m_tile        = xcc_id * m_tiles_per_xcc + m_group * swizzle_eff + m_local;

    if (m_tile >= m_tiles_total) return;

    const int block_m_base = m_tile * static_cast<int>(WM * WaveM * TILE);
    const int block_n_base = n_tile * static_cast<int>(WN * WaveN * TILE);
    const int m_base = block_m_base + wm * static_cast<int>(WaveM * TILE);
    const int n_base = block_n_base + wn * static_cast<int>(WaveN * TILE);

    /* ── CRT accumulators ───────────────────────────────────────────────────
     * Register fallback (size 1 placeholder when USE_LDS_ACCUM=true).
     * Each thread owns its own Zhi/Zlo elements — no barrier needed for init. */
    double Zhi_reg[USE_LDS_ACCUM ? 1u : NREG] = {};
    double Zlo_reg[USE_LDS_ACCUM ? 1u : NREG] = {};
    if constexpr (USE_LDS_ACCUM) {
        #pragma unroll
        for (unsigned e = 0; e < NREG; ++e) {
            Zhi_lds[e * BLK_THR + static_cast<unsigned>(tid)] = 0.0;
            Zlo_lds[e * BLK_THR + static_cast<unsigned>(tid)] = 0.0;
        }
    }

    /* ── Cooperative-load position tables ───────────────────────────────────── */
    /* wm_A ranges 0..WM*WaveM-1, covering all M rows in the macrotile. */
    unsigned wm_A[A_STEPS_SAFE]; int ml_A[A_STEPS_SAFE], k4_A[A_STEPS_SAFE], mi_A[A_STEPS_SAFE];
    #pragma unroll
    for (unsigned ls = 0; ls < A_STEPS; ++ls) {
        const int flat     = static_cast<int>(threadIdx.x + ls * BLK_THR);
        const int tile_dim = static_cast<int>(TILE * K4DIM);
        wm_A[ls] = static_cast<unsigned>(flat / tile_dim);
        ml_A[ls] = (flat % tile_dim) / static_cast<int>(K4DIM);
        k4_A[ls] = (flat % tile_dim) % static_cast<int>(K4DIM);
        mi_A[ls] = block_m_base + static_cast<int>(wm_A[ls]) * static_cast<int>(TILE) + ml_A[ls];
    }
    unsigned wn_B[B_STEPS_SAFE]; int nl_B[B_STEPS_SAFE], k4_B[B_STEPS_SAFE], ni_B[B_STEPS_SAFE];
    #pragma unroll
    for (unsigned ls = 0; ls < B_STEPS; ++ls) {
        const int flat     = static_cast<int>(threadIdx.x + ls * BLK_THR);
        const int tile_dim = static_cast<int>(TILE * K4DIM);
        wn_B[ls] = static_cast<unsigned>(flat / tile_dim);
        nl_B[ls] = (flat % tile_dim) / static_cast<int>(K4DIM);
        k4_B[ls] = (flat % tile_dim) % static_cast<int>(K4DIM);
        ni_B[ls] = block_n_base + static_cast<int>(wn_B[ls]) * static_cast<int>(TILE) + nl_B[ls];
    }

    /* ── Boundary-safe 4-byte load helpers (prologue only) ─────────────────── */
    auto cond_load_a = [&](const int8_t* As, unsigned ls, int ki) -> int32_t {
        int32_t v = 0;
        const int mi = mi_A[ls];
        if (mi < static_cast<int>(m)) {
            if (ki + 3 < k_int)
                v = *reinterpret_cast<const int32_t*>(As + static_cast<size_t>(mi) * lda8i + ki);
            else if (ki < k_int) {
                const int8_t* p = As + static_cast<size_t>(mi) * lda8i + ki;
                for (int b = 0; b < 4 && ki + b < k_int; ++b)
                    reinterpret_cast<int8_t*>(&v)[b] = p[b];
            }
        }
        return v;
    };
    auto cond_load_b = [&](const int8_t* Bs, unsigned ls, int ki) -> int32_t {
        int32_t v = 0;
        const int ni = ni_B[ls];
        if (ni < static_cast<int>(n)) {
            if (ki + 3 < k_int)
                v = *reinterpret_cast<const int32_t*>(Bs + static_cast<size_t>(ni) * ldb8i + ki);
            else if (ki < k_int) {
                const int8_t* p = Bs + static_cast<size_t>(ni) * ldb8i + ki;
                for (int b = 0; b < 4 && ki + b < k_int; ++b)
                    reinterpret_cast<int8_t*>(&v)[b] = p[b];
            }
        }
        return v;
    };

    /* ── Cooperative load of sftA/sftB into LDS ────────────────────────────── */
    /* Loop-stride pattern so any BLK_THR covers all entries, even when
     * BLK_THR < (WM*WaveM + WN*WaveN)*TILE (e.g., WM=WN=1, WaveM=WaveN=4). */
    for (unsigned si = static_cast<unsigned>(threadIdx.x);
         si < WM * WaveM * TILE; si += BLK_THR) {
        const int mi = block_m_base + static_cast<int>(si);
        sftA_lds[si] = (mi < static_cast<int>(m)) ? sftA[mi] : 0;
    }
    for (unsigned si = static_cast<unsigned>(threadIdx.x);
         si < WN * WaveN * TILE; si += BLK_THR) {
        const int ni = block_n_base + static_cast<int>(si);
        sftB_lds[si] = (ni < static_cast<int>(n)) ? sftB[ni] : 0;
    }

    /* ── Pre-fetch modulus s=0 prologue into registers ─────────────────────── */
    int32_t rA_pro[A_STEPS_SAFE] = {};
    int32_t rB_pro[B_STEPS_SAFE] = {};
    if (k_int > 0) {
        #pragma unroll
        for (unsigned ls = 0; ls < A_STEPS; ++ls)
            rA_pro[ls] = cond_load_a(A8i, ls, k4_A[ls] * 4);
        #pragma unroll
        for (unsigned ls = 0; ls < B_STEPS; ++ls)
            rB_pro[ls] = cond_load_b(B8i, ls, k4_B[ls] * 4);
    }
    if constexpr (!VALID_CONFIG) return; /* no-op on wrong arch */
    __syncthreads();

    /* ── MFMA helper ──────────────────────────────────────────────────────────
     * A_wm_base: pointer to A8i_lds[cur][wm*WaveM][0][0]
     * B_wn_base: pointer to B8i_lds[cur][wn*WaveN][0][0]
     * Loops over WaveM×WaveN MFMA tiles; each tile i's accumulators live in
     * C32[i*NREG_single .. (i+1)*NREG_single-1].
     * With WaveM=WaveN=1 the loop bodies execute once, identical to the
     * single-tile case.                                                        */
    auto apply_mfma = [&](const int8_t* A_wm_base, const int8_t* B_wn_base,
                          int32_t (&C32)[NREG]) __attribute__((always_inline)) {
        const int m_col  = lane % static_cast<int>(TILE);
        const int k_base = static_cast<int>(K_A_BYTES) * (lane / static_cast<int>(TILE));
        /* Stride (in bytes) between consecutive M-rows of the LDS A tile. */
        static constexpr int A_ROW_STRIDE = static_cast<int>(TILE * KBLK_STRIDE);
        static constexpr int B_ROW_STRIDE = static_cast<int>(TILE * KBLK_STRIDE);
        #pragma unroll
        for (unsigned wm_w = 0; wm_w < WaveM; ++wm_w) {
            const int8_t* A_wm_slot = A_wm_base + wm_w * A_ROW_STRIDE;
            #pragma unroll
            for (unsigned wn_w = 0; wn_w < WaveN; ++wn_w) {
                const int8_t* B_wn_slot = B_wn_base + wn_w * B_ROW_STRIDE;
                const unsigned reg_off = (wm_w * WaveN + wn_w) * NREG_single;
                if constexpr (TILE == 16u) {
                    v4i32 vx;
                    for (unsigned e = 0; e < NREG_single; ++e) vx[static_cast<int>(e)] = C32[reg_off + e];
                    for (unsigned ku = 0; ku < K_UNROLL; ++ku) {
                        const int kab   = k_base + static_cast<int>(ku * KBLK);
                        const int off_A = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                        const int off_B = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                        const auto sa = oz2_load_mfma_src16(A_wm_slot + off_A);
                        const auto sb = oz2_load_mfma_src16(B_wn_slot + off_B);
                        vx = oz2_do_mfma_16(sa, sb, vx);
                    }
                    for (unsigned e = 0; e < NREG_single; ++e) C32[reg_off + e] = vx[static_cast<int>(e)];
                } else {
                    v16i32 vx;
                    for (unsigned e = 0; e < NREG_single; ++e) vx[static_cast<int>(e)] = C32[reg_off + e];
                    for (unsigned ku = 0; ku < K_UNROLL; ++ku) {
                        const int kab   = k_base + static_cast<int>(ku * KBLK);
                        const int off_A = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                        const int off_B = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                        const auto sa = oz2_load_mfma_src32(A_wm_slot + off_A);
                        const auto sb = oz2_load_mfma_src32(B_wn_slot + off_B);
                        vx = oz2_do_mfma_32(sa, sb, vx);
                    }
                    for (unsigned e = 0; e < NREG_single; ++e) C32[reg_off + e] = vx[static_cast<int>(e)];
                }
            }
        }
    };

    /* ── Main loop: one modulus per iteration ───────────────────────────────── */
    for (unsigned s = 0; s < S; ++s) {
        const int8_t* A8i_s = A8i + static_cast<size_t>(s) * stride_A_s;
        const int8_t* B8i_s = B8i + static_cast<size_t>(s) * stride_B_s;
        int32_t C32[NREG] = {};

        if (k_int > 0) {
            __syncthreads();

            #pragma unroll
            for (unsigned ls = 0; ls < A_STEPS; ++ls)
                *reinterpret_cast<int32_t*>(&A8i_lds[0][wm_A[ls]][ml_A[ls]][k4_A[ls] * 4u]) = rA_pro[ls];
            #pragma unroll
            for (unsigned ls = 0; ls < B_STEPS; ++ls)
                *reinterpret_cast<int32_t*>(&B8i_lds[0][wn_B[ls]][nl_B[ls]][k4_B[ls] * 4u]) = rB_pro[ls];

            if (s + 1 < S) {
                #pragma unroll
                for (unsigned ls = 0; ls < A_STEPS; ++ls)
                    rA_pro[ls] = cond_load_a(A8i + static_cast<size_t>(s + 1) * stride_A_s, ls, k4_A[ls] * 4);
                #pragma unroll
                for (unsigned ls = 0; ls < B_STEPS; ++ls)
                    rB_pro[ls] = cond_load_b(B8i + static_cast<size_t>(s + 1) * stride_B_s, ls, k4_B[ls] * 4);
            }

            __syncthreads();

            const int8_t* A_base[A_STEPS_SAFE];
            const int8_t* B_base[B_STEPS_SAFE];
            #pragma unroll
            for (unsigned ls = 0; ls < A_STEPS; ++ls)
                A_base[ls] = A8i_s + static_cast<size_t>(mi_A[ls]) * lda8i
                           + static_cast<size_t>(k4_A[ls]) * 4u;
            #pragma unroll
            for (unsigned ls = 0; ls < B_STEPS; ++ls)
                B_base[ls] = B8i_s + static_cast<size_t>(ni_B[ls]) * ldb8i
                           + static_cast<size_t>(k4_B[ls]) * 4u;

            int32_t rA[A_STEPS_SAFE], rB[B_STEPS_SAFE];
            int cur = 0;

            for (int k_off = 0; k_off + static_cast<int>(KBLK_LOAD) < k_int;
                 k_off += static_cast<int>(KBLK_LOAD)) {
                const int nxt   = 1 - cur;
                const int nxt_k = k_off + static_cast<int>(KBLK_LOAD);
                #pragma unroll
                for (unsigned ls = 0; ls < A_STEPS; ++ls)
                    rA[ls] = *reinterpret_cast<const int32_t*>(A_base[ls] + nxt_k);
                #pragma unroll
                for (unsigned ls = 0; ls < B_STEPS; ++ls)
                    rB[ls] = *reinterpret_cast<const int32_t*>(B_base[ls] + nxt_k);
                apply_mfma(&A8i_lds[cur][wm * WaveM][0][0], &B8i_lds[cur][wn * WaveN][0][0], C32);
                #pragma unroll
                for (unsigned ls = 0; ls < A_STEPS; ++ls)
                    *reinterpret_cast<int32_t*>(
                        &A8i_lds[nxt][wm_A[ls]][ml_A[ls]][k4_A[ls] * 4u]) = rA[ls];
                #pragma unroll
                for (unsigned ls = 0; ls < B_STEPS; ++ls)
                    *reinterpret_cast<int32_t*>(
                        &B8i_lds[nxt][wn_B[ls]][nl_B[ls]][k4_B[ls] * 4u]) = rB[ls];
                __syncthreads();
                cur = nxt;
            }
            apply_mfma(&A8i_lds[cur][wm * WaveM][0][0], &B8i_lds[cur][wn * WaveN][0][0], C32);
        }

        /* ── CRT update ── (skipped when NO_CRT=true for MFMA-isolation experiment) */
        if constexpr (!NO_CRT) {
        /* ── CRT update: one (wm_w, wn_w) tile × one NREG_sub chunk at a time ─
         * The outer (wm_w, wn_w) loop covers WaveM×WaveN MFMA tiles.
         * The inner sub loop splits NREG_single into NREG_sub_iters passes of
         * NREG_sub elements each, keeping only NREG_sub doubles live (4 for
         * TILE=16 already; 4-of-16 for TILE=32, saving 48 VGPRs when
         * USE_LDS_ACCUM=true).  For TILE=16: NREG_sub=NREG_single=4,
         * NREG_sub_iters=1 → single-iteration no-op, identical to before.
         * No inter-thread barrier needed (each thread owns its own column).
         *
         * Note: Phase A/B split (dc_v/hi_v temp arrays) was tested and found
         * to be -2% due to extra VGPR pressure; compiler already schedules
         * the independent chains optimally from the unrolled inner loop.      */
        #pragma unroll
        for (unsigned wm_w = 0; wm_w < WaveM; ++wm_w) {
            #pragma unroll
            for (unsigned wn_w = 0; wn_w < WaveN; ++wn_w) {
                const unsigned reg_off = (wm_w * WaveN + wn_w) * NREG_single;
                const double nm = oz2_neg_mod(s), im = oz2_inv_mod(s);
                if constexpr (!USE_LDS_ACCUM && NREG_sub_iters == 1u) {
                    /* Fast path: compute directly on Zhi_reg/Zlo_reg.
                     * For TILE=16 (NREG_sub=NREG_single=4, NREG_sub_iters=1) the
                     * subtiling is a no-op. Eliminating Zhi_t/Zlo_t saves 16 arch
                     * VGPRs, increasing occupancy from 3 to 4-5 waves per SIMD.  */
                    #pragma unroll
                    for (unsigned e = 0; e < NREG_sub; ++e) {
                        const double dc_raw = static_cast<double>(C32[reg_off + e]);
                        const double dc     = fma(nm, rint(dc_raw * im), dc_raw);
                        const double hi     = dc * oz2_qpi_hi(S - 2, s);
                        const double new_hi = Zhi_reg[reg_off + e] + hi;
                        const double err    = hi - (new_hi - Zhi_reg[reg_off + e]);
                        Zhi_reg[reg_off + e] = new_hi;
                        if constexpr (HAS_LO)
                            Zlo_reg[reg_off + e] = fma(dc, oz2_qpi_lo(S - 2, s),
                                                       Zlo_reg[reg_off + e] + err);
                        else Zlo_reg[reg_off + e] += err;
                    }
                } else {
                    /* General subtiling path: for USE_LDS_ACCUM (gfx950 large
                     * WaveM/WaveN) or TILE=32 where NREG_sub_iters > 1.        */
                    #pragma unroll
                    for (unsigned sub = 0; sub < NREG_sub_iters; ++sub) {
                        const unsigned sub_e0 = sub * NREG_sub;
                        double Zhi_t[NREG_sub], Zlo_t[NREG_sub];
                        if constexpr (USE_LDS_ACCUM) {
                            #pragma unroll
                            for (unsigned e = 0; e < NREG_sub; ++e) {
                                Zhi_t[e] = Zhi_lds[(reg_off + sub_e0 + e) * BLK_THR + static_cast<unsigned>(tid)];
                                Zlo_t[e] = Zlo_lds[(reg_off + sub_e0 + e) * BLK_THR + static_cast<unsigned>(tid)];
                            }
                        } else {
                            #pragma unroll
                            for (unsigned e = 0; e < NREG_sub; ++e) {
                                Zhi_t[e] = Zhi_reg[reg_off + sub_e0 + e];
                                Zlo_t[e] = Zlo_reg[reg_off + sub_e0 + e];
                            }
                        }
                        #pragma unroll
                        for (unsigned e = 0; e < NREG_sub; ++e) {
                            const double dc_raw = static_cast<double>(C32[reg_off + sub_e0 + e]);
                            const double dc     = fma(nm, rint(dc_raw * im), dc_raw);
                            const double hi     = dc * oz2_qpi_hi(S - 2, s);
                            const double new_hi = Zhi_t[e] + hi;
                            const double err    = hi - (new_hi - Zhi_t[e]);
                            Zhi_t[e] = new_hi;
                            if constexpr (HAS_LO) Zlo_t[e] = fma(dc, oz2_qpi_lo(S - 2, s), Zlo_t[e] + err);
                            else                  Zlo_t[e] += err;
                        }
                        if constexpr (USE_LDS_ACCUM) {
                            #pragma unroll
                            for (unsigned e = 0; e < NREG_sub; ++e) {
                                Zhi_lds[(reg_off + sub_e0 + e) * BLK_THR + static_cast<unsigned>(tid)] = Zhi_t[e];
                                Zlo_lds[(reg_off + sub_e0 + e) * BLK_THR + static_cast<unsigned>(tid)] = Zlo_t[e];
                            }
                        } else {
                            #pragma unroll
                            for (unsigned e = 0; e < NREG_sub; ++e) {
                                Zhi_reg[reg_off + sub_e0 + e] = Zhi_t[e];
                                Zlo_reg[reg_off + sub_e0 + e] = Zlo_t[e];
                            }
                        }
                    }
                }
            }
        }  /* end CRT update for this (wm_w, wn_w) */
        }  /* end NO_CRT constexpr guard */
    }

    /* ── Finalize ── (skipped when NO_CRT=true for MFMA-isolation experiment) */
    if constexpr (!NO_CRT) {
    /* ── Finalize: CRT range-reduction + inverse scale + write D ─────────────
     * ISA-verified output layout (AMD CDNA3, Sec. 7.1.4.2) per MFMA tile:
     *   col = lane % TILE
     *   row = 4*(lane/TILE) + 8*(e/4) + (e%4)   for e = 0..NREG_single-1
     *
     * Outer WaveM×WaveN loops iterate over the wavefront's MFMA tile grid.
     * Zhi/Zlo for each tile are loaded into NREG_single temporaries from
     * either LDS or registers depending on USE_LDS_ACCUM.                    */
    #pragma unroll
    for (unsigned wm_w = 0; wm_w < WaveM; ++wm_w) {
        #pragma unroll
        for (unsigned wn_w = 0; wn_w < WaveN; ++wn_w) {
            const int m_wave_base    = m_base + static_cast<int>(wm_w * TILE);
            const int n_wave_base    = n_base + static_cast<int>(wn_w * TILE);
            const int col_val        = n_wave_base + (lane % static_cast<int>(TILE));
            const int lane_row_base  = m_wave_base + 4 * (lane / static_cast<int>(TILE));
            const unsigned reg_off   = (wm_w * WaveN + wn_w) * NREG_single;

            /* Load accumulator tile and finalize.
             * For !USE_LDS_ACCUM (gfx942): use Zhi_reg/Zlo_reg directly — no
             * Zhi_t/Zlo_t temporaries needed, saving another 16 arch VGPRs.     */
            if constexpr (!USE_LDS_ACCUM) {
            #pragma unroll
            for (unsigned e = 0; e < NREG_single; ++e) {
                const int ri = lane_row_base + static_cast<int>(8u * (e / 4u) + (e % 4u));
                const int ci = col_val;
                if (ri >= static_cast<int>(m) || ci >= static_cast<int>(n)) continue;
                const double q = rint((Zhi_reg[reg_off + e] + Zlo_reg[reg_off + e]) * oz2_inv_P(S - 2));
                const double X = fma(oz2_P_lo(S - 2), q,
                                     fma(oz2_P_hi(S - 2), q, Zhi_reg[reg_off + e]) + Zlo_reg[reg_off + e]);
                const int inv_sft = -(static_cast<int>(sftA_lds[ri - block_m_base])
                                     + static_cast<int>(sftB_lds[ci - block_n_base]));
                const size_t d_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldd;
                double d_val = alpha * ldexp(X, inv_sft);
                if (beta != 0.0) {
                    const size_t c_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldc;
                    d_val += beta * C[c_idx];
                }
                __builtin_nontemporal_store(d_val, D + d_idx);
            }
            } else {
            /* LDS path (gfx950 USE_LDS_ACCUM): load into temporaries first. */
            double Zhi_t[NREG_single], Zlo_t[NREG_single];
            #pragma unroll
            for (unsigned e = 0; e < NREG_single; ++e) {
                Zhi_t[e] = Zhi_lds[(reg_off + e) * BLK_THR + static_cast<unsigned>(tid)];
                Zlo_t[e] = Zlo_lds[(reg_off + e) * BLK_THR + static_cast<unsigned>(tid)];
            }
            #pragma unroll
            for (unsigned e = 0; e < NREG_single; ++e) {
                const int ri = lane_row_base + static_cast<int>(8u * (e / 4u) + (e % 4u));
                const int ci = col_val;
                if (ri >= static_cast<int>(m) || ci >= static_cast<int>(n)) continue;
                const double q = rint((Zhi_t[e] + Zlo_t[e]) * oz2_inv_P(S - 2));
                const double X = fma(oz2_P_lo(S - 2), q,
                                     fma(oz2_P_hi(S - 2), q, Zhi_t[e]) + Zlo_t[e]);
                const int inv_sft = -(static_cast<int>(sftA_lds[ri - block_m_base])
                                     + static_cast<int>(sftB_lds[ci - block_n_base]));
                const size_t d_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldd;
                double d_val = alpha * ldexp(X, inv_sft);
                if (beta != 0.0) {
                    const size_t c_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldc;
                    d_val += beta * C[c_idx];
                }
                __builtin_nontemporal_store(d_val, D + d_idx);
            }
            } /* end USE_LDS_ACCUM else */
        }
    }
    } /* end NO_CRT constexpr guard for finalize */
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
        is_gfx950 = (pci_id == 0x75a3u || pci_id == 0x75b3u) ? 1 : 0;
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
    auto make_grid = [&](unsigned wm_v, unsigned wn_v, unsigned tile_v,
                         unsigned wm_wave_v = 1u, unsigned wn_wave_v = 1u) -> dim3 {
        const int mt = static_cast<int>(
            (m + static_cast<int64_t>(wm_v * wm_wave_v * tile_v) - 1)
            / static_cast<int64_t>(wm_v * wm_wave_v * tile_v));
        const int nt = static_cast<int>(
            (n + static_cast<int64_t>(wn_v * wn_wave_v * tile_v) - 1)
            / static_cast<int64_t>(wn_v * wn_wave_v * tile_v));
        const int m_per_xcc    = (mt + num_xccs - 1) / num_xccs;
        const int total_blocks = m_per_xcc * nt * num_xccs;
        return dim3(static_cast<unsigned>(total_blocks), 1u);
    };

/* OZ2_FUSED_LAUNCH: dispatch oz2_fused_TN_kernel with explicit KU_PARAM and FVA_V.
 * KU_V=0 → auto-select K_UNROLL from LDS budget (same as omitting the param).
 * KU_V=2,4 → force a specific K_UNROLL for tuning.
 * FVA_V=false → use LDS accumulators when they fit (default).
 * FVA_V=true  → force VGPR-based accumulators even when LDS has space.        */
#define OZ2_FUSED_LAUNCH(S_V, HL, WM_V, WN_V, TILE_V, WM_WAVE_V, WN_WAVE_V, KU_V, FVA_V) \
    hipLaunchKernelGGL((oz2_fused_TN_kernel<(S_V),(HL),(WM_V),(WN_V),(TILE_V),(WM_WAVE_V),(WN_WAVE_V),(KU_V),(FVA_V)>), \
                       make_grid((WM_V),(WN_V),(TILE_V),(WM_WAVE_V),(WN_WAVE_V)), \
                       dim3((WM_V)*(WN_V)*64u), 0, stream, \
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

/* Helper: dispatch one config.  FVA_V is a literal 0u or 1u (converted to
 * false/true for the bool FORCE_VGPR_ACCUM template parameter).
 * The dispatch table encodes _fva==0 / _fva==1 in the condition, so each call
 * site always passes a compile-time constant — no runtime branch needed here.  */
#define _OV_DISPATCH(S_V,WM_V,WN_V,T_V,WMW_V,WNW_V,KU_V,FVA_V) \
    do { \
        if(has_lo) OZ2_FUSED_LAUNCH((S_V),true, WM_V,WN_V,T_V,WMW_V,WNW_V,KU_V,FVA_V); \
        else       OZ2_FUSED_LAUNCH((S_V),false,WM_V,WN_V,T_V,WMW_V,WNW_V,KU_V,FVA_V); \
    } while(0)

#define OZ2_DISPATCH_SHAPE_OVERRIDE(S_V) \
    do { \
        const char* _ov = std::getenv("OZ2_FUSED_SHAPE_OVERRIDE"); \
        if (_ov) { \
            unsigned _wm=0,_wn=0,_wm_w=0,_wn_w=0,_t=0,_ku=0,_fva=0; \
            { std::sscanf(_ov,"%u %u %u %u %u %u %u",&_wm,&_wn,&_wm_w,&_wn_w,&_t,&_ku,&_fva); } \
            if (_wm && _wn && (_t==16u||_t==32u)) { \
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
                if      (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,2u,32u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,4u,32u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,2u,16u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,1u,2u,4u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,2u,1u,4u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,2u,2u,4u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,1u,32u,1u,1u,4u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,1u,1u,4u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,1u,2u,4u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,1u,1u,4u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,2u,1u,4u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,1u,1u,4u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,2u,1u,4u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,1u,2u,4u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,2u,16u,2u,2u,4u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,4u,16u,4u,1u,4u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,1u,16u,1u,4u,4u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,4u,2u,4u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,2u,4u,4u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,4u,4u,4u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,2u,1u,4u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,2u,4u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,2u,1u,4u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,1u,2u,4u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,1u,1u,4u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,1u,16u,1u,1u,4u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,4u,16u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,2u,32u,1u,1u,4u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),2u,1u,32u,1u,1u,4u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,2u,32u,1u,1u,4u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),4u,1u,32u,1u,1u,4u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&_fva==0) _OV_DISPATCH((S_V),1u,4u,32u,1u,1u,4u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,2u,1u,2u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,4u,1u,2u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,4u,2u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,2u,2u,2u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,2u,1u,2u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,1u,2u,2u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,2u,16u,2u,2u,2u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,4u,16u,4u,1u,2u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,1u,16u,1u,4u,2u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,4u,2u,2u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,2u,4u,2u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,4u,4u,2u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,2u,2u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,2u,1u,2u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,1u,2u,2u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,2u,32u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,4u,32u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,2u,16u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,1u,2u,2u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,2u,1u,2u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,2u,2u,2u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,1u,32u,1u,1u,2u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,1u,1u,2u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,1u,2u,2u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,1u,1u,2u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,2u,1u,2u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,1u,1u,2u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,1u,1u,2u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,1u,16u,1u,1u,2u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,4u,16u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,2u,32u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),2u,1u,32u,1u,1u,2u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,2u,32u,1u,1u,2u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),4u,1u,32u,1u,1u,2u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==2&&_fva==0) _OV_DISPATCH((S_V),1u,4u,32u,1u,1u,2u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,2u,1u,1u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,1u,2u,1u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,2u,16u,2u,2u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,2u,2u,1u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,4u,4u,1u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,4u,16u,4u,1u,1u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4u,1u,16u,1u,4u,1u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,4u,2u,1u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,2u,4u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,4u,2u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,2u,4u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==4&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,4u,4u,1u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,2u,16u,1u,1u,1u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,1u,2u,1u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,2u,1u,1u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,2u,2u,1u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,1u,32u,1u,1u,1u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,1u,1u,1u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,1u,2u,1u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,2u,16u,1u,1u,1u,0u); \
                else if (_wm==1&&_wn==1&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,1u,16u,2u,1u,1u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,1u,16u,1u,1u,1u,0u); \
                else if (_wm==2&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,2u,32u,1u,1u,1u,0u); \
                else if (_wm==2&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),2u,1u,32u,1u,1u,1u,0u); \
                else if (_wm==1&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&_fva==0) _OV_DISPATCH((S_V),1u,2u,32u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,2u,1u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,2u,1u,1u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,1u,2u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,2u,1u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,4u,1u,1u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,2u,32u,1u,1u,1u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),2u,4u,32u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,4u,16u,1u,4u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,4u,32u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,2u,16u,1u,1u,1u,0u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),2u,4u,16u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,1u,16u,1u,1u,1u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),1u,4u,16u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),4u,1u,32u,1u,1u,1u,0u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==0) _OV_DISPATCH((S_V),1u,4u,32u,1u,1u,1u,0u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,2u,1u,1u); \
                else if (_wm==4&&_wn==2&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,2u,16u,2u,1u,1u,1u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2u,4u,16u,1u,2u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,1u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,2u,1u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,4u,1u,1u,1u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,2u,32u,1u,1u,1u,1u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2u,4u,32u,1u,1u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,4u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,32u,1u,1u,1u,1u); \
                else if (_wm==4&&_wn==2&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,2u,16u,1u,1u,1u,1u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2u,4u,16u,1u,1u,1u,1u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,1u,16u,1u,1u,1u,1u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),1u,4u,16u,1u,1u,1u,1u); \
                else if (_wm==4&&_wn==1&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,1u,32u,1u,1u,1u,1u); \
                else if (_wm==1&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),1u,4u,32u,1u,1u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,1u,4u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==2&&_t==16&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,2u,4u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==1&&_t==16&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,2u,1u,4u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==2&&_t==16&&_ku==1&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,4u,2u,1u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==2&&_wn_w==2&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,2u,2u,2u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==4&&_wn_w==1&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,4u,1u,2u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==4&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,4u,2u,1u); \
                else if (_wm==4&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==16&&_ku==2&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),4u,4u,16u,1u,1u,2u,1u); \
                else if (_wm==2&&_wn==4&&_wm_w==1&&_wn_w==1&&_t==32&&_ku==4&&is_gfx950&&_fva==1) _OV_DISPATCH((S_V),2u,4u,32u,1u,1u,4u,1u); \
                else { return rocblaslt_status_invalid_value; } \
                return rocblaslt_status_success; \
            } \
        } \
    } while(0)

#define OZ2_DISPATCH_SHAPE(S_V) \
    do { \
        /* Shape heuristic (MI300X gfx942, S=16; measurements with full workspace \
         *   pre-allocated — eliminating hipMalloc overhead from timing):         \
         *   Tall (m >= 4n): WM4WN4Wm4Wn2T16 KU=1 — 256×128 tile, 16219 GFLOP/s.\
         *     256×128 gives n_tiles=2 for N=256; large WaveM amortizes per-block \
         *     overhead; KU=1 reduces per-barrier KBLK_LOAD for large K.          \
         *   Wide (n >= 4m): WM2WN4Wm1Wn2T16 KU=2 — 32×128 (TILE=16), 10734 G/s.\
         *     WaveN=2, KU=2 with TILE=16 beats TILE=32 by 3.3% on gfx942.       \
         *   Square: WM4WN4Wm1Wn2T16 KU=4 — 64×128 tile, 24771 GFLOP/s.         \
         *     Slightly better than 128×64 (24233) for square with small K.       */ \
        if (m >= n * static_cast<int64_t>(4)) { \
            _OV_DISPATCH((S_V),4u,4u,16u,4u,2u,1u,false);  /* WM4WN4Wm4Wn2T16: 256×128 macrotile, K_UNROLL=1 */ \
        } else if (n >= m * static_cast<int64_t>(4)) { \
            _OV_DISPATCH((S_V),2u,4u,16u,1u,2u,2u,false);  /* WM2WN4Wm1Wn2T16: 32×128 (TILE=16), K_UNROLL=2 */ \
        } else { \
            _OV_DISPATCH((S_V),4u,4u,16u,1u,2u,4u,false);  /* WM4WN4Wm1Wn2T16: 64×128 macrotile, K_UNROLL=4 */ \
        } \
    } while(0)

    /* ── OZ2_NO_CRT=1 experiment: dispatch MFMA-only kernel (no CRT/finalize) ──
     * Results are garbage but the kernel duration isolates pure MFMA+prefetch
     * overhead and reveals the true CRT cost (28ms_full - X_mfma_only = CRT). */
    {
        const char* _no_crt_env = std::getenv("OZ2_NO_CRT");
        if (_no_crt_env && std::strcmp(_no_crt_env, "1") == 0) {
            if (num_moduli != 16u) {
                std::fprintf(stderr,
                    "[oz2_launch_fused_TN] OZ2_NO_CRT=1 requires num_moduli=16 (got %u)\n",
                    num_moduli);
                std::abort();
            }
            /* Dispatch WM4WN4Wm1Wn2T16ku4 (square heuristic) with NO_CRT=true.
             * WaveN=2, KU=4 matches the production kernel for large square shapes
             * so that OZ2_NO_CRT=1 isolates MFMA overhead for exactly that config. */
            if (has_lo)
                hipLaunchKernelGGL(
                    (oz2_fused_TN_kernel<16u, true,  4u, 4u, 16u, 1u, 2u, 4u, false, true>),
                    make_grid(4u, 4u, 16u, 1u, 2u), dim3(4u * 4u * 64u), 0, stream,
                    A8i, stride_A_s, lda8i, B8i, stride_B_s, ldb8i,
                    C, D, m, n, k, ldc, ldd, alpha, beta, sftA, sftB, num_xccs);
            else
                hipLaunchKernelGGL(
                    (oz2_fused_TN_kernel<16u, false, 4u, 4u, 16u, 1u, 2u, 4u, false, true>),
                    make_grid(4u, 4u, 16u, 1u, 2u), dim3(4u * 4u * 64u), 0, stream,
                    A8i, stride_A_s, lda8i, B8i, stride_B_s, ldb8i,
                    C, D, m, n, k, ldc, ldd, alpha, beta, sftA, sftB, num_xccs);
            return rocblaslt_status_success;
        }
    }

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
