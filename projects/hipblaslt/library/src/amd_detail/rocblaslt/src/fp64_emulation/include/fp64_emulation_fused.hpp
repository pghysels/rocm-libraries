// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/*
 * fp64_emulation_fused.hpp
 *
 * Public interface for the fused MFMA+CRT kernel.
 *
 * This header is included by fp64_emulation.cpp for:
 *   - Oz2FusedMode enum and oz2_fused_mode() declaration
 *   - OZ2_FUSED_KBLK_LOAD_MAX constant (workspace-zero guard)
 *   - oz2_launch_fused_TN() declaration
 *
 * The kernel template (oz2_fused_TN_kernel), all MFMA types/helpers,
 * and the GPU constant-memory extern declarations are defined/declared
 * in fp64_emulation_fused.cpp, where the only instantiations occur.
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 */

#include "fp64_emulation.hpp"  /* rocblaslt_status, OZ2_S_MAX */
#include <hip/hip_runtime.h>

/* =========================================================================
 * Fused mode control
 * ========================================================================= */

/* HIPBLASLT_EMULATION_FUSED controls whether the fused MFMA+CRT kernel is used:
 *   "on"  / "force"      → always use fused (bypasses performance model)
 *   "off" / "never"      → never  use fused (forces non-fused path)
 *   "auto"/ "performant" → performance model decides
 *   unset (default)      → OFF — fused kernel disabled until production-ready */
enum class Oz2FusedMode { AUTO, ON, OFF };
Oz2FusedMode oz2_fused_mode();

/* Maximum KBLK_LOAD across all kernel variants AND all supported architectures —
 * used for the workspace-zero guard in fp64EmulatedGemmImpl (HOST-side).
 * gfx942: TILE=16, K_UNROLL=4, KBLK_16=32 → KBLK_LOAD = 128
 * gfx950: TILE=16, K_UNROLL=4, KBLK_16=64 → KBLK_LOAD = 256
 * Must be the architecture-independent maximum (256) to ensure the workspace is
 * correctly zeroed on gfx950, where host code does not see __gfx950__.           */
inline constexpr unsigned OZ2_FUSED_KBLK_LOAD_MAX = 256u;

/* =========================================================================
 * Fused TN kernel launcher (implemented in fp64_emulation_fused.cpp)
 * ========================================================================= */
rocblaslt_status oz2_launch_fused_TN(
    const int8_t*  A8i,    /* workspace INT8 A (all S moduli stacked) */
    const int8_t*  B8i,    /* workspace INT8 B (all S moduli stacked) */
    size_t         lda8i,  /* padded k, leading dim of A8i per modulus */
    size_t         cola8i, /* padded m                                 */
    size_t         ldb8i,  /* padded k, leading dim of B8i per modulus */
    const double*  C,
    double*        D,
    int64_t m, int64_t n, int64_t k,
    int64_t ldc, int64_t ldd,
    double alpha, double beta,
    const int16_t* sftA, const int16_t* sftB,
    unsigned num_moduli, hipStream_t stream);
