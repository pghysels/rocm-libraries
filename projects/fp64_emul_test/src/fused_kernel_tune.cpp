// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * fused_kernel_tune.cpp
 *
 * Tuning benchmark for the fused MFMA+CRT kernel (oz2_fused_TN_kernel).
 * Sweeps 11 candidate (WM, WN, WaveM, WaveN, TILE) configurations, checks
 * correctness for each, and reports GFLOP/s across three representative
 * GEMM shapes on the current GPU (gfx942 or gfx950).
 *
 * Shapes (all TN, S fixed at library-selected maximum):
 *   large_sq : M=4096, N=4096, K= 256  — M,N >> K (HPL-like)
 *   tall     : M=4096, N= 256, K=4096  — M,K >> N
 *   wide     : M= 256, N=4096, K=4096  — N,K >> M
 *
 * The fused kernel is selected via the OZ2_FUSED_SHAPE_OVERRIDE env var.
 * Correctness is checked by comparing each emulated result against a
 * reference native DGEMM (emulation disabled on a second handle).
 *
 * Build (from projects/fp64_emul_test/):
 *   cmake -B build -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx950" \
 *         -DHIPBLASLT_BUILD_DIR=../hipblaslt/build/release
 *   cmake --build build -j32 --target fused_kernel_tune
 *
 * Run:
 *   ./build/fused_kernel_tune [--warmup N] [--iters N]
 */

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* =========================================================================
 * Error-check helpers
 * ========================================================================= */
#define HIP_CHECK(expr) \
    do { \
        hipError_t _e = (expr); \
        if (_e != hipSuccess) { \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", \
                         __FILE__, __LINE__, hipGetErrorString(_e)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HLT_CHECK(expr) \
    do { \
        hipblasStatus_t _s = (expr); \
        if (_s != HIPBLAS_STATUS_SUCCESS) { \
            std::fprintf(stderr, "hipBLASLt error %s:%d: status=%d\n", \
                         __FILE__, __LINE__, static_cast<int>(_s)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

/* =========================================================================
 * Candidate kernel configurations
 * (WM, WN, WaveM, WaveN, TILE)  — all 64×64 macrotile unless noted
 * ========================================================================= */
struct KernelConfig {
    unsigned WM, WN, WaveM, WaveN, TILE;
    unsigned KU;           /* K_UNROLL override: 0=auto, 2=force-2, 4=force-4 */
    bool     gfx950_only;      /* skip on gfx942 */
    bool     force_vgpr_accum; /* true → FORCE_VGPR_ACCUM=1 (7th OZ2_FUSED_SHAPE_OVERRIDE param) */
    const char* label;         /* short human-readable name */
};

static const KernelConfig CONFIGS[] = {
    /* WM, WN, WaveM, WaveN, TILE, KU, label */
    /* KU=0/auto removed throughout — all entries use explicit KU=1/2/4.
     * KU_AUTO on gfx942: =4 when WM×WaveM+WN×WaveN ≤ 15, else =2.
     * KU_AUTO on gfx950: TILE=16→4, TILE=32→2 (hardcoded). */
    /* ── TILE=32, K_UNROLL=4 override (MI350X: LDS_MFMA_K4=49KB<159KB) ── */
    /* On MI350X: USE_LDS_ACCUM=true + sub-tile CRT + K_UNROLL=4 may win   */
    { 4, 2, 1, 1, 32, 4, false, false, "WM4WN2Wm1Wn1T32ku4" },
    { 2, 4, 1, 1, 32, 4, false, false, "WM2WN4Wm1Wn1T32ku4" },
    /* ── KU=4: 32×32 macrotile configs ── */
    { 2, 2, 1, 1, 16, 4, false, false, "WM2WN2Wm1Wn1T16ku4" },
    { 2, 1, 1, 2, 16, 4, false, false, "WM2WN1Wm1Wn2T16ku4" },
    { 1, 2, 2, 1, 16, 4, false, false, "WM1WN2Wm2Wn1T16ku4" },
    { 1, 1, 2, 2, 16, 4, false, false, "WM1WN1Wm2Wn2T16ku4" },
    { 1, 1, 1, 1, 32, 4, false, false, "WM1WN1Wm1Wn1T32ku4" },
    /* ── KU=4: sub-32×32 macrotile configs (16×16 / 16×32 / 32×16, TILE=16) ── */
    { 1, 1, 1, 1, 16, 4, false, false, "WM1WN1Wm1Wn1T16ku4" },  /* 16×16, 64 thr, NREG=4 */
    { 1, 1, 1, 2, 16, 4, false, false, "WM1WN1Wm1Wn2T16ku4" },  /* 16×32, 64 thr, NREG=8 */
    { 1, 2, 1, 1, 16, 4, false, false, "WM1WN2Wm1Wn1T16ku4" },  /* 16×32, 128 thr, NREG=4 */
    { 1, 1, 2, 1, 16, 4, false, false, "WM1WN1Wm2Wn1T16ku4" },  /* 32×16, 64 thr, NREG=8 */
    { 2, 1, 1, 1, 16, 4, false, false, "WM2WN1Wm1Wn1T16ku4" },  /* 32×16, 128 thr, NREG=4 */
    /* ── KU=4: all TILE=16 standard 64x64 configs ── */
    { 4, 4, 1, 1, 16, 4, false, false, "WM4WN4Wm1Wn1T16ku4" },
    { 2, 4, 2, 1, 16, 4, false, false, "WM2WN4Wm2Wn1T16ku4" },
    { 4, 2, 1, 2, 16, 4, false, false, "WM4WN2Wm1Wn2T16ku4" },
    { 2, 2, 2, 2, 16, 4, false, false, "WM2WN2Wm2Wn2T16ku4" },
    { 1, 4, 4, 1, 16, 4, false, false, "WM1WN4Wm4Wn1T16ku4" },
    { 4, 1, 1, 4, 16, 4, false, false, "WM4WN1Wm1Wn4T16ku4" },
    { 1, 2, 4, 2, 16, 4, false, false, "WM1WN2Wm4Wn2T16ku4" },
    { 2, 1, 2, 4, 16, 4, false, false, "WM2WN1Wm2Wn4T16ku4" },
    { 1, 1, 4, 4, 16, 4, false, false, "WM1WN1Wm4Wn4T16ku4" },
    /* ── KU=4: TILE=16 larger macrotiles (KU_AUTO=4 on gfx942 for all; KU_AUTO=4 on gfx950) ── */
    { 4, 4, 2, 1, 16, 4, false, false, "WM4WN4Wm2Wn1T16ku4" },  /* 128×64 macrotile */
    { 4, 4, 1, 2, 16, 4, false, false, "WM4WN4Wm1Wn2T16ku4" },  /* 64×128 macrotile */
    { 4, 2, 2, 1, 16, 4, false, false, "WM4WN2Wm2Wn1T16ku4" },
    { 2, 4, 1, 2, 16, 4, false, false, "WM2WN4Wm1Wn2T16ku4" },
    /* ── KU=4: TILE=16 asymmetric macrotiles (64×32, 32×64, 64×16, 16×64) ── */
    { 4, 2, 1, 1, 16, 4, false, false, "WM4WN2Wm1Wn1T16ku4" },  /* 64×32, 512 thr, NREG=4 */
    { 2, 4, 1, 1, 16, 4, false, false, "WM2WN4Wm1Wn1T16ku4" },  /* 32×64, 512 thr, NREG=4 */
    { 4, 1, 1, 1, 16, 4, false, false, "WM4WN1Wm1Wn1T16ku4" },  /* 64×16, 256 thr, NREG=4 */
    { 1, 4, 1, 1, 16, 4, false, false, "WM1WN4Wm1Wn1T16ku4" },  /* 16×64, 256 thr, NREG=4 */
    /* ── KU=4: TILE=32 additional WM×WN configs ── */
    { 2, 2, 1, 1, 32, 4, false, false, "WM2WN2Wm1Wn1T32ku4" },  /* 64×64 (T32), 256 thr */
    { 2, 1, 1, 1, 32, 4, false, false, "WM2WN1Wm1Wn1T32ku4" },  /* 64×32 (T32), 128 thr */
    { 1, 2, 1, 1, 32, 4, false, false, "WM1WN2Wm1Wn1T32ku4" },  /* 32×64 (T32), 128 thr */
    { 4, 1, 1, 1, 32, 4, false, false, "WM4WN1Wm1Wn1T32ku4" },  /* 128×32 (T32), 256 thr */
    { 1, 4, 1, 1, 32, 4, false, false, "WM1WN4Wm1Wn1T32ku4" },  /* 32×128 (T32), 256 thr */
    /* ── TILE=16 128×64, K_UNROLL=2 (test vs default 4) ─────────────────── */
    { 4, 4, 2, 1, 16, 2, false, false, "WM4WN4Wm2Wn1T16ku2" },
    /* ── KU=2: 256×64, 64×256, 128×128 (KU_AUTO=2 on gfx942; LDS overflow at KU=4) ── */
    { 4, 4, 4, 1, 16, 2, false, false, "WM4WN4Wm4Wn1T16ku2" },
    { 4, 4, 1, 4, 16, 2, false, false, "WM4WN4Wm1Wn4T16ku2" },  /* 64×256 macrotile */
    { 4, 4, 2, 2, 16, 2, false, false, "WM4WN4Wm2Wn2T16ku2" },
    /* ── KU=2: all standard TILE=16 64×64 configs ──────────────────────────── */
    { 4, 4, 1, 1, 16, 2, false, false, "WM4WN4Wm1Wn1T16ku2" },
    { 2, 4, 2, 1, 16, 2, false, false, "WM2WN4Wm2Wn1T16ku2" },
    { 4, 2, 1, 2, 16, 2, false, false, "WM4WN2Wm1Wn2T16ku2" },
    { 2, 2, 2, 2, 16, 2, false, false, "WM2WN2Wm2Wn2T16ku2" },
    { 1, 4, 4, 1, 16, 2, false, false, "WM1WN4Wm4Wn1T16ku2" },
    { 4, 1, 1, 4, 16, 2, false, false, "WM4WN1Wm1Wn4T16ku2" },
    { 1, 2, 4, 2, 16, 2, false, false, "WM1WN2Wm4Wn2T16ku2" },
    { 2, 1, 2, 4, 16, 2, false, false, "WM2WN1Wm2Wn4T16ku2" },
    { 1, 1, 4, 4, 16, 2, false, false, "WM1WN1Wm4Wn4T16ku2" },
    /* ── KU=2: TILE=16 larger macrotiles (missing from 128x64 section) ── */
    { 4, 4, 1, 2, 16, 2, false, false, "WM4WN4Wm1Wn2T16ku2" },  /* 64x128 */
    { 4, 2, 2, 1, 16, 2, false, false, "WM4WN2Wm2Wn1T16ku2" },  /* 128x32 */
    { 2, 4, 1, 2, 16, 2, false, false, "WM2WN4Wm1Wn2T16ku2" },  /* 32x128 */
    /* ── KU=2: TILE=32 ── */
    { 4, 2, 1, 1, 32, 2, false, false, "WM4WN2Wm1Wn1T32ku2" },
    { 2, 4, 1, 1, 32, 2, false, false, "WM2WN4Wm1Wn1T32ku2" },
    /* ── KU=2: 32×32 macrotile configs ── */
    { 2, 2, 1, 1, 16, 2, false, false, "WM2WN2Wm1Wn1T16ku2" },
    { 2, 1, 1, 2, 16, 2, false, false, "WM2WN1Wm1Wn2T16ku2" },
    { 1, 2, 2, 1, 16, 2, false, false, "WM1WN2Wm2Wn1T16ku2" },
    { 1, 1, 2, 2, 16, 2, false, false, "WM1WN1Wm2Wn2T16ku2" },
    { 1, 1, 1, 1, 32, 2, false, false, "WM1WN1Wm1Wn1T32ku2" },
    /* ── KU=2: sub-32×32 macrotile configs (16×16 / 16×32 / 32×16, TILE=16) ── */
    { 1, 1, 1, 1, 16, 2, false, false, "WM1WN1Wm1Wn1T16ku2" },  /* 16×16, 64 thr, NREG=4 */
    { 1, 1, 1, 2, 16, 2, false, false, "WM1WN1Wm1Wn2T16ku2" },  /* 16×32, 64 thr, NREG=8 */
    { 1, 2, 1, 1, 16, 2, false, false, "WM1WN2Wm1Wn1T16ku2" },  /* 16×32, 128 thr, NREG=4 */
    { 1, 1, 2, 1, 16, 2, false, false, "WM1WN1Wm2Wn1T16ku2" },  /* 32×16, 64 thr, NREG=8 */
    { 2, 1, 1, 1, 16, 2, false, false, "WM2WN1Wm1Wn1T16ku2" },  /* 32×16, 128 thr, NREG=4 */
    /* ── KU=2: TILE=16 asymmetric macrotiles (64×32, 32×64, 64×16, 16×64) ── */
    { 4, 2, 1, 1, 16, 2, false, false, "WM4WN2Wm1Wn1T16ku2" },  /* 64×32, 512 thr, NREG=4 */
    { 2, 4, 1, 1, 16, 2, false, false, "WM2WN4Wm1Wn1T16ku2" },  /* 32×64, 512 thr, NREG=4 */
    { 4, 1, 1, 1, 16, 2, false, false, "WM4WN1Wm1Wn1T16ku2" },  /* 64×16, 256 thr, NREG=4 */
    { 1, 4, 1, 1, 16, 2, false, false, "WM1WN4Wm1Wn1T16ku2" },  /* 16×64, 256 thr, NREG=4 */
    /* ── KU=2: TILE=32 additional WM×WN configs ── */
    { 2, 2, 1, 1, 32, 2, false, false, "WM2WN2Wm1Wn1T32ku2" },  /* 64×64 (T32), 256 thr */
    { 2, 1, 1, 1, 32, 2, false, false, "WM2WN1Wm1Wn1T32ku2" },  /* 64×32 (T32), 128 thr */
    { 1, 2, 1, 1, 32, 2, false, false, "WM1WN2Wm1Wn1T32ku2" },  /* 32×64 (T32), 128 thr */
    { 4, 1, 1, 1, 32, 2, false, false, "WM4WN1Wm1Wn1T32ku2" },  /* 128×32 (T32), 256 thr */
    { 1, 4, 1, 1, 32, 2, false, false, "WM1WN4Wm1Wn1T32ku2" },  /* 32×128 (T32), 256 thr */
    /* ── KU=1: small-K tuning — valid on both gfx942 and gfx950 ───────────── */
    { 2, 4, 2, 1, 16, 1, false, false, "WM2WN4Wm2Wn1T16ku1" },  /* A=B=1 on gfx942 */
    { 4, 2, 1, 2, 16, 1, false, false, "WM4WN2Wm1Wn2T16ku1" },  /* A=B=1 on gfx942 */
    { 2, 2, 2, 2, 16, 1, false, false, "WM2WN2Wm2Wn2T16ku1" },  /* A=B=2 on gfx942 */
    { 4, 4, 2, 2, 16, 1, false, false, "WM4WN4Wm2Wn2T16ku1" },  /* A=B=1 on gfx942 */
    { 1, 1, 4, 4, 16, 1, false, false, "WM1WN1Wm4Wn4T16ku1" },  /* A=B=8 on gfx942 */
    { 1, 4, 4, 1, 16, 1, false, false, "WM1WN4Wm4Wn1T16ku1" },  /* A=B=2 on gfx942 */
    { 4, 1, 1, 4, 16, 1, false, false, "WM4WN1Wm1Wn4T16ku1" },  /* A=B=2 on gfx942 */
    { 1, 2, 4, 2, 16, 1, false, false, "WM1WN2Wm4Wn2T16ku1" },  /* A=B=4 on gfx942 */
    { 2, 1, 2, 4, 16, 1, false, false, "WM2WN1Wm2Wn4T16ku1" },  /* A=B=4 on gfx942 */
    /* ── KU=1: TILE=16 larger WaveM x WaveN (both-arch valid) ── */
    { 4, 4, 4, 2, 16, 1, false, false, "WM4WN4Wm4Wn2T16ku1" },  /* A=2,B=1 on gfx942 */
    { 4, 4, 2, 4, 16, 1, false, false, "WM4WN4Wm2Wn4T16ku1" },  /* A=1,B=2 on gfx942 */
    { 4, 4, 4, 4, 16, 1, false, false, "WM4WN4Wm4Wn4T16ku1" },  /* A=2,B=2 on gfx942 */
    /* ── KU=1: 32×32 macrotile configs (all valid on both gfx942 and gfx950) ── */
    { 2, 2, 1, 1, 16, 1, false, false, "WM2WN2Wm1Wn1T16ku1" },  /* 32×32 macrotile, 256 thr, NREG=4 */
    { 2, 1, 1, 2, 16, 1, false, false, "WM2WN1Wm1Wn2T16ku1" },  /* 32×32 macrotile, 128 thr, NREG=8 */
    { 1, 2, 2, 1, 16, 1, false, false, "WM1WN2Wm2Wn1T16ku1" },  /* 32×32 macrotile, 128 thr, NREG=8 */
    { 1, 1, 2, 2, 16, 1, false, false, "WM1WN1Wm2Wn2T16ku1" },  /* 32×32 macrotile, 64 thr, NREG=16 */
    { 1, 1, 1, 1, 32, 1, false, false, "WM1WN1Wm1Wn1T32ku1" },  /* 32×32 macrotile (TILE=32), 64 thr, NREG=16 */
    /* ── KU=1: sub-32×32 macrotile configs (both-arch valid) ── */
    { 1, 1, 1, 1, 16, 1, false, false, "WM1WN1Wm1Wn1T16ku1" },  /* 16×16, 64 thr, NREG=4 */
    { 1, 1, 1, 2, 16, 1, false, false, "WM1WN1Wm1Wn2T16ku1" },  /* 16×32, 64 thr, NREG=8 */
    { 1, 2, 1, 1, 16, 1, false, false, "WM1WN2Wm1Wn1T16ku1" },  /* 16×32, 128 thr, NREG=4 */
    { 1, 1, 2, 1, 16, 1, false, false, "WM1WN1Wm2Wn1T16ku1" },  /* 32×16, 64 thr, NREG=8 */
    { 2, 1, 1, 1, 16, 1, false, false, "WM2WN1Wm1Wn1T16ku1" },  /* 32×16, 128 thr, NREG=4 */
    /* ── KU=1: TILE=32 additional WM×WN (both-arch valid: B_STEPS≥1 on gfx942) ── */
    { 2, 2, 1, 1, 32, 1, false, false, "WM2WN2Wm1Wn1T32ku1" },  /* 64×64 (T32), 256 thr */
    { 2, 1, 1, 1, 32, 1, false, false, "WM2WN1Wm1Wn1T32ku1" },  /* 64×32 (T32), 128 thr */
    { 1, 2, 1, 1, 32, 1, false, false, "WM1WN2Wm1Wn1T32ku1" },  /* 32×64 (T32), 128 thr */
    /* ── gfx950-only KU=1: guarded by is_gfx950 at dispatch, VALID_CONFIG=false on gfx942 ── */
    { 4, 4, 1, 2, 16, 1, true, false, "WM4WN4Wm1Wn2T16ku1" },  /* 64x128 KU=1, gfx950 only */
    { 4, 2, 2, 1, 16, 1, true, false, "WM4WN2Wm2Wn1T16ku1" },  /* 128x32 KU=1, gfx950 only */
    { 2, 4, 1, 2, 16, 1, true, false, "WM2WN4Wm1Wn2T16ku1" },  /* 32x128 KU=1, gfx950 only */
    { 4, 4, 1, 1, 16, 1, true, false, "WM4WN4Wm1Wn1T16ku1" },
    { 4, 4, 2, 1, 16, 1, true, false, "WM4WN4Wm2Wn1T16ku1" },
    { 4, 4, 4, 1, 16, 1, true, false, "WM4WN4Wm4Wn1T16ku1" },
    { 4, 2, 1, 1, 32, 1, true, false, "WM4WN2Wm1Wn1T32ku1" },
    { 2, 4, 1, 1, 32, 1, true, false, "WM2WN4Wm1Wn1T32ku1" },
    { 4, 4, 1, 4, 16, 1, true, false, "WM4WN4Wm1Wn4T16ku1" },  /* 64x256 KU=1, gfx950 only */
    { 4, 4, 1, 1, 32, 1, true, false, "WM4WN4Wm1Wn1T32ku1" },  /* T32 WM4WN4 KU=1, gfx950 only */
    /* ── KU=1 gfx950-only: TILE=16 asymmetric + TILE=32 larger WM×WN ── */
    { 4, 2, 1, 1, 16, 1, true, false, "WM4WN2Wm1Wn1T16ku1" },  /* 64×32 (T16), 512 thr, gfx950 only */
    { 2, 4, 1, 1, 16, 1, true, false, "WM2WN4Wm1Wn1T16ku1" },  /* 32×64 (T16), 512 thr, gfx950 only */
    { 4, 1, 1, 1, 16, 1, true, false, "WM4WN1Wm1Wn1T16ku1" },  /* 64×16 (T16), 256 thr, gfx950 only */
    { 1, 4, 1, 1, 16, 1, true, false, "WM1WN4Wm1Wn1T16ku1" },  /* 16×64 (T16), 256 thr, gfx950 only */
    { 4, 1, 1, 1, 32, 1, true, false, "WM4WN1Wm1Wn1T32ku1" },  /* 128×32 (T32), 256 thr, gfx950 only */
    { 1, 4, 1, 1, 32, 1, true, false, "WM1WN4Wm1Wn1T32ku1" },  /* 32×128 (T32), 256 thr, gfx950 only */
    /* ── FORCE_VGPR_ACCUM=true variants (gfx950 only) ─────────────────────
     * These duplicates test register-based Zhi/Zlo for configs where
     * FITS_IN_LDS=true on gfx950 (159 KB LDS).  Comparing fva0 vs fva1
     * reveals whether the 2-way LDS bank conflicts for FP64 doubles cost
     * more than the occupancy benefit from freed VGPRs.
     * All entries are gfx950_only=true; on gfx942 FITS_IN_LDS=false so
     * FORCE_VGPR_ACCUM has no effect.                                       */
    /* ── fva1: gfx950-specific configs (KU=1 large macrotiles) ── */
    { 4, 4, 1, 2, 16, 1, true, true, "WM4WN4Wm1Wn2T16ku1fva1" },
    { 4, 2, 2, 1, 16, 1, true, true, "WM4WN2Wm2Wn1T16ku1fva1" },
    { 2, 4, 1, 2, 16, 1, true, true, "WM2WN4Wm1Wn2T16ku1fva1" },
    { 4, 4, 1, 1, 16, 1, true, true, "WM4WN4Wm1Wn1T16ku1fva1" },
    { 4, 4, 2, 1, 16, 1, true, true, "WM4WN4Wm2Wn1T16ku1fva1" },
    { 4, 4, 4, 1, 16, 1, true, true, "WM4WN4Wm4Wn1T16ku1fva1" },
    { 4, 2, 1, 1, 32, 1, true, true, "WM4WN2Wm1Wn1T32ku1fva1" },
    { 2, 4, 1, 1, 32, 1, true, true, "WM2WN4Wm1Wn1T32ku1fva1" },
    { 4, 4, 1, 4, 16, 1, true, true, "WM4WN4Wm1Wn4T16ku1fva1" },
    { 4, 4, 1, 1, 32, 1, true, true, "WM4WN4Wm1Wn1T32ku1fva1" },
    { 4, 2, 1, 1, 16, 1, true, true, "WM4WN2Wm1Wn1T16ku1fva1" },
    { 2, 4, 1, 1, 16, 1, true, true, "WM2WN4Wm1Wn1T16ku1fva1" },
    { 4, 1, 1, 1, 16, 1, true, true, "WM4WN1Wm1Wn1T16ku1fva1" },
    { 1, 4, 1, 1, 16, 1, true, true, "WM1WN4Wm1Wn1T16ku1fva1" },
    { 4, 1, 1, 1, 32, 1, true, true, "WM4WN1Wm1Wn1T32ku1fva1" },
    { 1, 4, 1, 1, 32, 1, true, true, "WM1WN4Wm1Wn1T32ku1fva1" },
    /* ── fva1: key standard configs — compare LDS vs VGPR Zhi/Zlo on gfx950 ── */
    { 4, 4, 1, 1, 16, 4, true, true, "WM4WN4Wm1Wn1T16ku4fva1" },  /* 64×64,   NREG=4  */
    { 4, 4, 1, 2, 16, 4, true, true, "WM4WN4Wm1Wn2T16ku4fva1" },  /* 64×128,  NREG=8  — square heuristic */
    { 4, 4, 2, 1, 16, 4, true, true, "WM4WN4Wm2Wn1T16ku4fva1" },  /* 128×64,  NREG=8  */
    { 4, 4, 4, 2, 16, 1, true, true, "WM4WN4Wm4Wn2T16ku1fva1" },  /* 256×128, NREG=32 — tall heuristic */
    { 4, 4, 2, 2, 16, 2, true, true, "WM4WN4Wm2Wn2T16ku2fva1" },  /* 128×128, NREG=16 */
    { 4, 4, 4, 1, 16, 2, true, true, "WM4WN4Wm4Wn1T16ku2fva1" },  /* 256×64,  NREG=16 */
    { 4, 4, 1, 4, 16, 2, true, true, "WM4WN4Wm1Wn4T16ku2fva1" },  /* 64×256,  NREG=16 */
    { 4, 4, 1, 1, 16, 2, true, true, "WM4WN4Wm1Wn1T16ku2fva1" },  /* 64×64,   NREG=4,  KU=2 */
    { 2, 4, 1, 1, 32, 4, true, true, "WM2WN4Wm1Wn1T32ku4fva1" },  /* 64×128 T32, NREG=16 — wide heuristic */
};
static const int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

/* =========================================================================
 * Test shapes
 * ========================================================================= */
struct Shape {
    int64_t m, n, k;
    const char* name;
};

static const Shape SHAPES[] = {
    // { 4096, 4096,  256, "large_sq (M,N>>K)" },
    // { 4096,  256, 4096, "tall     (M,K>>N)" },
    // {  256, 4096, 4096, "wide     (N,K>>M)" },

    { 32768, 32768,  32768, "large_sq (M,N>>K)" },
    //{ 32768,  256, 32768, "tall     (M,K>>N)" },
    //{  256, 32768, 32768, "wide     (N,K>>M)" },
};
static const int NUM_SHAPES = static_cast<int>(sizeof(SHAPES) / sizeof(SHAPES[0]));

/* =========================================================================
 * Small correctness shape
 * ========================================================================= */
static const int64_t CORR_N = 256;

/* =========================================================================
 * Utility: detect GPU name from PCI ID
 * ========================================================================= */
static std::string detect_gpu_name(int dev)
{
    int chip_id = 0;
    (void)hipDeviceGetAttribute(&chip_id, hipDeviceAttributePciChipId, dev);
    uint32_t pci = static_cast<uint32_t>(chip_id) & 0xFFFFu;
    if (pci == 0x75a3u || pci == 0x75b3u) return "gfx950 (MI350)";
    if (pci == 0x74a1u) return "gfx942 (MI300X)";
    char name[128] = {};
    (void)hipDeviceGetName(name, sizeof(name), dev);
    return std::string(name);
}

/* =========================================================================
 * GEMM helper: allocates workspace and runs one DGEMM via hipBLASLt.
 * Returns elapsed time in milliseconds (hipEvent timing).
 * workspace_d must be pre-allocated to >= workspace_bytes.
 * ========================================================================= */
static float run_gemm(
    hipblasLtHandle_t   handle,
    int64_t m, int64_t n, int64_t k,
    double alpha, double beta,
    const double* A_d, int64_t lda,
    const double* B_d, int64_t ldb,
    const double* C_d, int64_t ldc,
    double*       D_d, int64_t ldd,
    void*         ws_d, size_t ws_bytes,
    hipStream_t   stream)
{
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    /* TN: A is m×k (col-major), B is k×n (col-major) → opA=T means A^T */
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_64F, k, m, lda)); /* cols×rows for T */
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_64F, k, n, ldb));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_64F, m, n, ldc));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_64F, m, n, ldd));

    hipblasLtMatmulDesc_t matmul;
    HLT_CHECK(hipblasLtMatmulDescCreate(&matmul,
                                        HIPBLAS_COMPUTE_64F,
                                        HIP_R_64F));
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    HLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
        HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    HLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul,
        HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    hipblasLtMatmulPreference_t pref;
    HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ws_bytes, sizeof(ws_bytes)));

    hipblasLtMatmulHeuristicResult_t heurResult;
    int returnedAlgoCount = 0;
    HLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, matmul,
        layoutA, layoutB, layoutC, layoutD,
        pref, 1, &heurResult, &returnedAlgoCount));

    /* Timed launch */
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0, stream));

    HLT_CHECK(hipblasLtMatmul(handle, matmul,
        &alpha, A_d, layoutA,
        B_d, layoutB,
        &beta,  C_d, layoutC,
        D_d, layoutD,
        (returnedAlgoCount > 0) ? &heurResult.algo : nullptr,
        ws_d, ws_bytes, stream));

    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));

    HLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    HLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(layoutA));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(layoutB));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(layoutC));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(layoutD));

    return ms;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char** argv)
{
    /* ── Parse simple CLI args ──────────────────────────────────────────── */
    int warmup_runs        = 3;
    int timed_runs         = 10;
    int global_warmup_runs = 20;   /* pre-sweep GPU saturation pass */
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--warmup")        == 0 && i+1 < argc) warmup_runs        = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--iters")         == 0 && i+1 < argc) timed_runs         = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--global-warmup") == 0 && i+1 < argc) global_warmup_runs = std::atoi(argv[++i]);
    }

    /* ── Set mandatory env vars before any HIP/hipBLASLt calls ─────────── */
    setenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION", "1", 1);
    setenv("HIPBLASLT_EMULATION_FUSED",          "force", 1);

    /* ── Device info ────────────────────────────────────────────────────── */
    int cur_dev = 0;
    HIP_CHECK(hipGetDevice(&cur_dev));
    std::string gpu_name = detect_gpu_name(cur_dev);
    int num_xccs = 0;
    (void)hipDeviceGetAttribute(&num_xccs, hipDeviceAttributeNumberOfXccs, cur_dev);
    std::fprintf(stderr, "Device: %s  XCCs: %d\n", gpu_name.c_str(), num_xccs);
    const bool is_gfx950 = (gpu_name.find("MI350") != std::string::npos);

    /* ── Create handles ─────────────────────────────────────────────────── */
    /* Emulated handle: eager mode, S fixed at 16.
     * OZ2_DISPATCH_SHAPE_OVERRIDE now always dispatches oz2_fused_TN_kernel<16,...>,
     * so the library must also prepare exactly 16 sets of INT8 data.
     * maxBits=118 selects s=16 moduli (verified in fp64_emul_scale_bench.cpp). */
    hipblasLtHandle_t handle_emul;
    HLT_CHECK(hipblasLtCreate(&handle_emul));
    HLT_CHECK(hipblasLtSetEmulationEnabled(handle_emul, true));
    HLT_CHECK(hipblasLtSetEmulationStrategy(handle_emul,
                                             HIPBLASLT_EMULATION_STRATEGY_EAGER));
    HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
        handle_emul, HIPBLASLT_EMULATION_MANTISSA_CONTROL_FIXED));
    HLT_CHECK(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(handle_emul, 118)); /* s=16 */

    /* Reference handle: native DGEMM */
    hipblasLtHandle_t handle_ref;
    HLT_CHECK(hipblasLtCreate(&handle_ref));
    HLT_CHECK(hipblasLtSetEmulationEnabled(handle_ref, false));

    /* ── HIP stream ─────────────────────────────────────────────────────── */
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* ── Compute buffer sizes from actual shapes ────────────────────────── */
    /* TN GEMM layout: A is k*m elements (lda=k), B is k*n (ldb=k),
     * C and D are m*n (ldc=ldd=m).  Compute the maximum needed per slot
     * across all shapes plus the correctness shape.                         */
    size_t max_A  = static_cast<size_t>(CORR_N) * CORR_N;
    size_t max_B  = static_cast<size_t>(CORR_N) * CORR_N;
    size_t max_CD = static_cast<size_t>(CORR_N) * CORR_N;
    for (int i = 0; i < NUM_SHAPES; ++i) {
        max_A  = std::max(max_A,  static_cast<size_t>(SHAPES[i].k) * SHAPES[i].m);
        max_B  = std::max(max_B,  static_cast<size_t>(SHAPES[i].k) * SHAPES[i].n);
        max_CD = std::max(max_CD, static_cast<size_t>(SHAPES[i].m) * SHAPES[i].n);
    }
    std::fprintf(stderr, "# Buffer sizes: A=%.1f GiB  B=%.1f GiB  C/D=%.1f GiB\n",
                 max_A * 8.0 / (1<<30), max_B * 8.0 / (1<<30), max_CD * 8.0 / (1<<30));

    double *A_d, *B_d, *C_d, *D_emul_d, *D_ref_d;
    HIP_CHECK(hipMalloc(&A_d,      max_A  * sizeof(double)));
    HIP_CHECK(hipMalloc(&B_d,      max_B  * sizeof(double)));
    HIP_CHECK(hipMalloc(&C_d,      max_CD * sizeof(double)));
    HIP_CHECK(hipMalloc(&D_emul_d, max_CD * sizeof(double)));
    HIP_CHECK(hipMalloc(&D_ref_d,  max_CD * sizeof(double)));

    /* ── Fill correctness-check buffers on device ───────────────────────── */
    /* For performance runs, hipMemset initializes device memory without a
     * large host allocation (exact values don't affect timing).
     * For the correctness check (CORR_N×CORR_N only), copy deterministic
     * host data so element-wise comparison is meaningful.                   */
    const size_t corr_sz = static_cast<size_t>(CORR_N) * CORR_N;
    std::vector<double> h_A(corr_sz), h_B(corr_sz), h_C(corr_sz);
    for (size_t i = 0; i < corr_sz; ++i) h_A[i] = std::sin(static_cast<double>(i + 1));
    for (size_t i = 0; i < corr_sz; ++i) h_B[i] = std::cos(static_cast<double>(i + 1));
    for (size_t i = 0; i < corr_sz; ++i) h_C[i] = static_cast<double>(i % 97) * 0.01;

    /* Initialize full performance buffers with non-NaN values via memset. */
    HIP_CHECK(hipMemset(A_d,  0, max_A  * sizeof(double)));
    HIP_CHECK(hipMemset(B_d,  0, max_B  * sizeof(double)));
    HIP_CHECK(hipMemset(C_d,  0, max_CD * sizeof(double)));

    /* Overwrite just the CORR_N×CORR_N region with deterministic data. */
    HIP_CHECK(hipMemcpy(A_d, h_A.data(), corr_sz * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, h_B.data(), corr_sz * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(C_d, h_C.data(), corr_sz * sizeof(double), hipMemcpyHostToDevice));

    /* ── Workspace: query required size from hipBLASLt before allocating ───
     * The emulation INT8 arrays A8i and B8i scale as S×K×M and S×K×N
     * respectively, so workspace ∝ S×K×(M+N) — not M×N×K.  A tall shape
     * with large K can need far more than a square shape with small K.
     * We query every shape and take the maximum workspaceSize reported by
     * hipblasLtMatmulAlgoGetHeuristic (which accounts for the full pipeline).
     * The query upper-bound is capped at 80 % of currently free GPU memory. */
    size_t WS_BYTES = 256ull * 1024 * 1024;  /* fallback minimum */
    {
        size_t free_mem = 0, total_mem = 0;
        (void)hipMemGetInfo(&free_mem, &total_mem);
        const size_t max_query_ws = (free_mem * 4u) / 5u;

        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;

        for (int si = 0; si < NUM_SHAPES; ++si) {
            const Shape& sh = SHAPES[si];

            hipblasLtMatrixLayout_t qA, qB, qC, qD;
            HLT_CHECK(hipblasLtMatrixLayoutCreate(&qA, HIP_R_64F, sh.k, sh.m, sh.k));
            HLT_CHECK(hipblasLtMatrixLayoutCreate(&qB, HIP_R_64F, sh.k, sh.n, sh.k));
            HLT_CHECK(hipblasLtMatrixLayoutCreate(&qC, HIP_R_64F, sh.m, sh.n, sh.m));
            HLT_CHECK(hipblasLtMatrixLayoutCreate(&qD, HIP_R_64F, sh.m, sh.n, sh.m));

            hipblasLtMatmulDesc_t qMatmul;
            HLT_CHECK(hipblasLtMatmulDescCreate(&qMatmul, HIPBLAS_COMPUTE_64F, HIP_R_64F));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(qMatmul,
                HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(qMatmul,
                HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

            hipblasLtMatmulPreference_t qPref;
            HLT_CHECK(hipblasLtMatmulPreferenceCreate(&qPref));
            HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(qPref,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &max_query_ws, sizeof(max_query_ws)));

            hipblasLtMatmulHeuristicResult_t qResult;
            int qCount = 0;
            HLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_emul, qMatmul,
                qA, qB, qC, qD, qPref, 1, &qResult, &qCount));
            if (qCount > 0 && qResult.workspaceSize > WS_BYTES)
                WS_BYTES = qResult.workspaceSize;

            std::fprintf(stderr, "# Workspace query shape[%d] (%ldx%ldx%ld): %.3f GiB\n",
                         si, static_cast<long>(sh.m), static_cast<long>(sh.n),
                         static_cast<long>(sh.k),
                         (qCount > 0 ? qResult.workspaceSize : 0u) / static_cast<double>(1ull << 30));

            HLT_CHECK(hipblasLtMatmulPreferenceDestroy(qPref));
            HLT_CHECK(hipblasLtMatmulDescDestroy(qMatmul));
            HLT_CHECK(hipblasLtMatrixLayoutDestroy(qA));
            HLT_CHECK(hipblasLtMatrixLayoutDestroy(qB));
            HLT_CHECK(hipblasLtMatrixLayoutDestroy(qC));
            HLT_CHECK(hipblasLtMatrixLayoutDestroy(qD));
        }
    }
    std::fprintf(stderr, "# Workspace: %.3f GiB (queried from hipBLASLt)\n",
                 WS_BYTES / static_cast<double>(1ull << 30));
    void* ws_d;
    HIP_CHECK(hipMalloc(&ws_d, WS_BYTES));

    /* ── Host buffer for error checking (correctness shape only) ────────── */
    std::vector<double> h_emul(corr_sz), h_ref(corr_sz);

    /* ── Result accumulator ─────────────────────────────────────────────── */
    struct Result {
        const char* label;
        int         shape_idx;
        bool        correct;
        double      mean_ms;
        double      gflops;
        unsigned    macro_m;   /* WM * WaveM * TILE */
        unsigned    macro_n;   /* WN * WaveN * TILE */
    };
    std::vector<Result> results;
    results.reserve(NUM_CONFIGS * NUM_SHAPES);

    /* ── Global GPU warm-up: drain burst-clock before the sweep ────────────
     * The MI300X (and similar multi-XCC GPUs) boost to a burst frequency
     * when idle.  Whichever config runs first captures this elevated clock
     * and appears artificially fast (~10000 GFLOP/s vs ~4000-9000 sustained).
     * Running the baseline GEMM global_warmup_runs times before the sweep
     * saturates the GPU to its sustained frequency, so all configs are
     * measured on an equal footing.  Default: 20 iters × ~220ms ≈ 4.4 s.  */
    if (global_warmup_runs > 0) {
        std::fprintf(stderr, "# Global GPU warm-up: %d iters (draining burst clock)...\n",
                     global_warmup_runs);
        setenv("OZ2_FUSED_SHAPE_OVERRIDE", "4 4 1 1 16 4", 1);  /* WM4WN4Wm1Wn1T16 KU=4 */
        const Shape& sh0 = SHAPES[0];
        for (int r = 0; r < global_warmup_runs; ++r)
            run_gemm(handle_emul, sh0.m, sh0.n, sh0.k, 1.0, 0.0,
                     A_d, sh0.k, B_d, sh0.k, C_d, sh0.m, D_emul_d, sh0.m,
                     ws_d, WS_BYTES, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        unsetenv("OZ2_FUSED_SHAPE_OVERRIDE");
        std::fprintf(stderr, "# Global warm-up complete.\n");
    }

    std::fprintf(stderr, "# Sweeping %d configs × %d shapes...\n",
                 NUM_CONFIGS, NUM_SHAPES);

    /* ── Sweep configs ──────────────────────────────────────────────────── */
    for (int ci = 0; ci < NUM_CONFIGS; ++ci) {
        const KernelConfig& cfg = CONFIGS[ci];

        /* Skip gfx950-only configs when running on gfx942. */
        if (cfg.gfx950_only && !is_gfx950) {
            std::fprintf(stderr, "  %-24s  (skipped: gfx950 only)\n", cfg.label);
            continue;
        }

        /* Set override env var for this config.
         * Always emit all 6 params including KU (0 = auto-select K_UNROLL). */
        char ov_str[64];
        std::snprintf(ov_str, sizeof(ov_str), "%u %u %u %u %u %u %u",
                      cfg.WM, cfg.WN, cfg.WaveM, cfg.WaveN, cfg.TILE, cfg.KU,
                      cfg.force_vgpr_accum ? 1u : 0u);
        setenv("OZ2_FUSED_SHAPE_OVERRIDE", ov_str, 1);

        /* ── Correctness check ─────────────────────────────────────────── */
        bool correct = true;
        {
            const int64_t cm = CORR_N, cn = CORR_N, ck = CORR_N;

            HIP_CHECK(hipMemset(D_emul_d, 0, cm * cn * sizeof(double)));
            HIP_CHECK(hipMemset(D_ref_d,  0, cm * cn * sizeof(double)));

            run_gemm(handle_emul, cm, cn, ck,
                     1.0, 0.0,
                     A_d, ck, B_d, ck, C_d, cm, D_emul_d, cm,
                     ws_d, WS_BYTES, stream);
            run_gemm(handle_ref, cm, cn, ck,
                     1.0, 0.0,
                     A_d, ck, B_d, ck, C_d, cm, D_ref_d, cm,
                     ws_d, WS_BYTES, stream);

            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipMemcpy(h_emul.data(), D_emul_d, cm * cn * sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_ref.data(),  D_ref_d,  cm * cn * sizeof(double), hipMemcpyDeviceToHost));

            double max_relerr = 0.0;
            for (int64_t i = 0; i < cm * cn; ++i) {
                double err = std::fabs(h_emul[i] - h_ref[i]);
                double den = std::max(1.0, std::fabs(h_ref[i]));
                max_relerr = std::max(max_relerr, err / den);
            }
            correct = (max_relerr < 1e-10);
            if (!correct)
                std::fprintf(stderr, "  [FAIL] %s: max_relerr=%.3e\n",
                             cfg.label, max_relerr);
        }

        /* ── Performance timing for each shape ─────────────────────────── */
        for (int si = 0; si < NUM_SHAPES; ++si) {
            const Shape& sh = SHAPES[si];

            for (int r = 0; r < warmup_runs; ++r)
                run_gemm(handle_emul, sh.m, sh.n, sh.k, 1.0, 0.0,
                         A_d, sh.k, B_d, sh.k, C_d, sh.m, D_emul_d, sh.m,
                         ws_d, WS_BYTES, stream);
            HIP_CHECK(hipStreamSynchronize(stream));

            double total_ms = 0.0;
            for (int r = 0; r < timed_runs; ++r)
                total_ms += run_gemm(handle_emul, sh.m, sh.n, sh.k, 1.0, 0.0,
                                     A_d, sh.k, B_d, sh.k, C_d, sh.m, D_emul_d, sh.m,
                                     ws_d, WS_BYTES, stream);
            HIP_CHECK(hipStreamSynchronize(stream));

            double mean_ms = total_ms / timed_runs;
            double gflops  = 2.0 * static_cast<double>(sh.m) *
                             static_cast<double>(sh.n) *
                             static_cast<double>(sh.k) / (mean_ms * 1e6);

            const unsigned macro_m = cfg.WM * cfg.WaveM * cfg.TILE;
            const unsigned macro_n = cfg.WN * cfg.WaveN * cfg.TILE;
            results.push_back({ cfg.label, si, correct, mean_ms, gflops, macro_m, macro_n });
            std::fprintf(stderr, "  %6ux%-6u %-28s  shape %d  %.1f GFLOP/s\n",
                         macro_m, macro_n, cfg.label, si, gflops);
        }
    }

    /* ── Print results sorted by GFLOP/s (descending) per shape ──────────── */
    std::printf("# Fused kernel tuning — GPU: %s\n", gpu_name.c_str());
    std::printf("# Warmup=%d  Iters=%d\n", warmup_runs, timed_runs);

    for (int si = 0; si < NUM_SHAPES; ++si) {
        /* Gather results for this shape */
        std::vector<Result*> row;
        for (auto& r : results)
            if (r.shape_idx == si) row.push_back(&r);

        /* Sort descending by GFLOP/s */
        std::sort(row.begin(), row.end(),
                  [](const Result* a, const Result* b){
                      return a->gflops > b->gflops;
                  });

        std::printf("\n## Shape: %s  (M=%ld N=%ld K=%ld)\n",
                    SHAPES[si].name,
                    static_cast<long>(SHAPES[si].m),
                    static_cast<long>(SHAPES[si].n),
                    static_cast<long>(SHAPES[si].k));
        std::printf("%-12s  %-28s  %7s  %9s  %10s\n",
                    "macrotile", "config", "correct", "mean_ms", "gflops");
        std::printf("%s\n", std::string(72, '-').c_str());
        for (const Result* r : row) {
            char tile_str[16];
            std::snprintf(tile_str, sizeof(tile_str), "%ux%u", r->macro_m, r->macro_n);
            std::printf("%-12s  %-28s  %7s  %9.3f  %10.1f\n",
                        tile_str,
                        r->label,
                        r->correct ? "PASS" : "FAIL",
                        r->mean_ms,
                        r->gflops);
        }
    }

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK(hipFree(D_emul_d));
    HIP_CHECK(hipFree(D_ref_d));
    HIP_CHECK(hipFree(ws_d));
    HIP_CHECK(hipStreamDestroy(stream));
    HLT_CHECK(hipblasLtDestroy(handle_emul));
    HLT_CHECK(hipblasLtDestroy(handle_ref));

    return 0;
}
