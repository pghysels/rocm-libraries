// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/*
 * FP64 GEMM emulation via Ozaki Scheme II.
 *
 * This header declares the host-side interface used by rocblaslt_mat.cpp.
 * It is intentionally free of any HIP device types so it can be included
 * from plain C++ translation units.
 *
 * Environment variable:
 *   HIPBLASLT_EMULATE_DOUBLE_PRECISION=1   enables emulation
 *
 * Performance model (in fp64_emulation.cpp — fp64EmulationPerformanceCheck):
 *   A Roofline-based estimate comparing t_emulation vs t_native_DGEMM.
 *   Calibrated for MI350 (HBM_BW=6.4 TB/s, INT8_PEAK=3050 TOPS,
 *   FP64_EFF=70 TFLOPS effective).  Update the hardware constants for
 *   other architectures.
 */

#include "rocblaslt.h"
#include <hip/hip_runtime_api.h>


enum Fp64EmulationEnvState {
    FP64_EMULATION_ENV_UNSET = 0,
    FP64_EMULATION_ENV_VALID = 1,
    FP64_EMULATION_ENV_INVALID = 2,
};

struct Fp64EmulationEnvValue {
    Fp64EmulationEnvState state;
    unsigned int          value;
};

/* Pure parsers used by tests and by the cached runtime readers below. */
Fp64EmulationEnvValue fp64EmulationParseEnabledEnv(const char* value);
Fp64EmulationEnvValue fp64EmulationParseStrategyEnv(const char* value);
Fp64EmulationEnvValue fp64EmulationParseSpecialValuesMaskEnv(const char* value);
Fp64EmulationEnvValue fp64EmulationParseMantissaBitCountEnv(const char* value);
bool                 fp64EmulationIsValidMantissaBitCount(int value);

/* Returns true when HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 is set.
 * The environment variable is read once and cached. Invalid values are reported
 * through fp64EmulationDecision(), not through this legacy bool helper. */
bool fp64EmulationIsEnabled();

/* Returns true when the emulation is estimated to be at least as fast as
 * native FP64 DGEMM for the given problem size and number of moduli.
 * Uses a Roofline performance model calibrated for the target hardware. */
bool fp64EmulationPerformanceCheck(int64_t m, int64_t n, int64_t k, unsigned num_moduli);

/* Returns true when HIPBLASLT_EMULATION_STRATEGY=eager is set.
 * In eager mode emulation is used regardless of arithmetic intensity.
 * The environment variable is read once and cached. */
bool fp64EmulationIsEager();

/* Returns the special-values support mask from
 * HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK (default: 0x3).
 *   bit 0 = Inf detection enabled
 *   bit 1 = NaN detection enabled
 * A return value of 0 means no Inf/NaN checking is performed. */
uint32_t fp64EmulationSpecialValuesMask();

/* Returns the number of INT8 GEMMs (moduli) to use, in the range [2, 18].
 * Reads HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT; maps the
 * requested precision in bits to the minimum number of moduli required.
 * Default (env var absent): 16 moduli (~125 bits of CRT capacity).
 * A set env var is validated strictly; 0 or 1 selects the minimum of 2 moduli.
 * Maximum supported: 18 moduli (~140 bits of CRT capacity).
 * Notable values: 55 bits → 7 GEMMs, 79 bits → 10 GEMMs, 110 bits → 14 GEMMs. */
unsigned fp64EmulationNumModuli();

/* Returns the byte count of the emulation workspace for the given problem.
 * Use this to check whether a caller-provided workspace is sufficient. */
size_t fp64EmulationWorkspaceSize(int64_t m, int64_t n, int64_t k, unsigned num_moduli);

/* Forward declaration — callers already include handle.h which provides the full
 * definition.  Declared here so the two helpers below can use the type.     */
struct _rocblaslt_handle;

struct Fp64EmulationDecision {
    rocblaslt_status status;       /* rocblaslt_status_invalid_value on bad env   */
    bool             apply;        /* true → use emulation; false → native path  */
    unsigned int     num_moduli;   /* resolved moduli count to pass to settings  */
    unsigned int     sv_mask;      /* resolved special-values mask               */
    bool             dynamic_mode; /* true when DYNAMIC (ADP) mode selected      */
};

/* Status-returning FP64 emulation gate. Invalid env-var values return
 * rocblaslt_status_invalid_value so callers do not silently fall back to
 * native FP64. On success, apply=false means the native path should be used
 * without error. */
Fp64EmulationDecision fp64EmulationDecision(const _rocblaslt_handle* h,
                                            hipDataType              type_a,
                                            int64_t                  m,
                                            int64_t                  n,
                                            int64_t                  k,
                                            int                      batch_count);

/* Returns the effective number of CRT moduli (2..18) given the handle's emulation
 * settings.
 *   FIXED mode (mantissa_control=1, max_mantissa_bits≥0): maps the bit count to
 *     the minimum s whose CRT capacity ≥ max_mantissa_bits.
 *   Default/env mode with HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT:
 *     forces FIXED at the requested bit count.
 *   DYNAMIC mode or no precision hint: uses the current non-ADP 16-moduli path. */
unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h);

/* Per-call emulation settings.
 * Fields with sentinel values (0 for num_moduli, ~0u for sv_mask) cause the
 * function to fall back to the process-wide env var defaults.              */
struct Fp64EmulationSettings {
    unsigned int      num_moduli;      /* 2..18; 0 = derive from env var          */
    unsigned int      sv_mask;         /* special-values mask; ~0u = env var      */
    bool              dynamic_mode;    /* true when ADP (Adaptive Precision) mode */
    void*             workspace;       /* caller workspace; nullptr = allocate     */
    size_t            workspace_bytes; /* size of caller workspace                */
    hipblasLtHandle_t handle;          /* caller handle for INT8 GEMMs             */
};

/* Run an emulated FP64 GEMM using Ozaki Scheme II (accurate mode).
 *
 * Computes D = alpha * op(A) * op(B) + beta * C  where A, B, C, D are
 * device pointers to double arrays. C and D may be the same pointer.
 * Only non-batched (batch count == 1) FP64 GEMM is supported.
 *
 * Subnormal inputs (flush-to-zero semantics)
 * -------------------------------------------
 * Subnormal FP64 values in A or B (magnitudes in the range [DBL_TRUE_MIN,
 * DBL_MIN)) are silently treated as zero during the INT8 extraction step.
 * Any dot-product element whose true value would be subnormal may therefore
 * be returned as 0.0 instead of the correct subnormal.  The absolute error
 * is at most DBL_MIN ≈ 2.2e-308.
 * If subnormal correctness is required, disable emulation for the affected
 * GEMMs or fall back to native FP64 DGEMM.
 *
 * Returns rocblaslt_status_success on success,
 *         rocblaslt_status_memory_error if workspace allocation fails,
 *         rocblaslt_status_invalid_value if Inf/NaN is detected in the inputs
 *             (controlled by the sv_mask field in settings), or if dynamic
 *             mode (ADP) determines that even s=18 is insufficient for the
 *             given input's dynamic range.
 *             In both cases the caller should fall back to native FP64. */
rocblaslt_status fp64EmulatedGemm(hipblasOperation_t           opA,
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
                                  const Fp64EmulationSettings& settings);
