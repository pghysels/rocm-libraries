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
 * Environment variables:
 *   HIPBLASLT_EMULATE_DOUBLE_PRECISION=1   enables emulation
 *   HIPBLASLT_EMULATION_NUM_MODULI=N        direct moduli count [2..18] (preferred)
 *
 * Setting precedence: handle setters > env vars > built-in defaults.
 * Handle setters (hipblasLtSet* functions) permanently override env vars
 * for the lifetime of that handle. See fp64EmulationDecision() for details.
 *
 * See docs/how-to/fp64-emulation.rst
 */

#include "rocblaslt.h"
#include <hip/hip_runtime_api.h>

/* =========================================================================
 * Environment variable parsing infrastructure
 * ========================================================================= */
enum Fp64EmulationEnvState
{
    FP64_EMULATION_ENV_UNSET   = 0,
    FP64_EMULATION_ENV_VALID   = 1,
    FP64_EMULATION_ENV_INVALID = 2,
};

struct Fp64EmulationEnvValue
{
    Fp64EmulationEnvState state;
    unsigned int          value;
};

/* Pure parsers used by tests and by the cached runtime readers below. */
Fp64EmulationEnvValue fp64EmulationParseEnabledEnv(const char* value);
Fp64EmulationEnvValue fp64EmulationParseStrategyEnv(const char* value);
Fp64EmulationEnvValue fp64EmulationParseSpecialValuesMaskEnv(const char* value);

/* Returns true when HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 is set.
 * The environment variable is read once and cached. Invalid values are reported
 * through fp64EmulationDecision(), not through this legacy bool helper. */
bool fp64EmulationIsEnabled();

/* Forward declaration — callers already include handle.h which provides the full
 * definition.  Declared here so the functions below can use the type.       */
struct _rocblaslt_handle;

/* Returns true when the emulation is estimated to be at least as fast as
 * native FP64 DGEMM for the given problem size.
 * The number of moduli is derived from the handle's emulation settings via
 * fp64EmulationEffectiveNumModuli(h) so the check always uses the same
 * moduli count that would be used at run time.
 * Uses a Roofline performance model calibrated for the target hardware.
 * The handle is used to obtain the target device for the performance model.
 * opA/opB select per-transpose efficiency factors in the BW-kernel model. */
bool fp64EmulationPerformanceCheck(const _rocblaslt_handle* h,
                                   hipblasOperation_t       opA,
                                   hipblasOperation_t       opB,
                                   int64_t                  m,
                                   int64_t                  n,
                                   int64_t                  k);

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
 * Reads HIPBLASLT_EMULATION_NUM_MODULI (direct count [2..18]).
 * Maps the
 * requested precision in bits to the minimum number of moduli required.
 * Default (env var absent): 16 moduli (~125 bits of CRT capacity).
 * A set env var is validated strictly; 0 or 1 selects the minimum of 2 moduli.
 * Maximum supported: 18 moduli (~140 bits of CRT capacity).
 * Notable values: 55 bits → 7 GEMMs, 79 bits → 10 GEMMs, 110 bits → 14 GEMMs. */
unsigned fp64EmulationNumModuli();

/* =========================================================================
 * Decision struct and gate function — status-propagating
 * ========================================================================= */
struct Fp64EmulationDecision
{
    rocblaslt_status status; /* rocblaslt_status_invalid_value on bad env   */
    bool             apply; /* true → use emulation; false → native path  */
    unsigned int     num_moduli; /* resolved moduli count to pass to settings  */
    unsigned int     sv_mask; /* resolved special-values mask               */
    bool             dynamic_mode; /* true when DYNAMIC (ADP) mode selected      */
};

/* Status-returning FP64 emulation gate. Invalid env-var values return
 * rocblaslt_status_invalid_value so callers do not silently fall back to
 * native FP64. On success, apply=false means the native path should be used
 * without error.
 * opA/opB select per-transpose efficiency factors in the performance model. */
Fp64EmulationDecision fp64EmulationDecision(const _rocblaslt_handle* h,
                                            hipDataType              type_a,
                                            hipblasOperation_t       opA,
                                            hipblasOperation_t       opB,
                                            int64_t                  m,
                                            int64_t                  n,
                                            int64_t                  k,
                                            int                      batch_count);

/* Returns the effective number of CRT moduli (2..18) given the handle's emulation
 * settings.
 *   FIXED mode (num_moduli ∈ [2..18]): uses that count directly.
 *   Default/env mode with HIPBLASLT_EMULATION_NUM_MODULI:
 *     forces FIXED at the requested bit count.
 *   DYNAMIC mode or no precision hint: uses the current non-ADP 16-moduli path. */
unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h);

/* Returns the byte count of the emulation workspace for the given problem.
 * The decision is used to derive the layout moduli (18 in ADP mode,
 * decision.num_moduli otherwise) and the handle to obtain the device for the
 * performance model.
 * Use this to check whether a caller-provided workspace is sufficient. */
size_t fp64EmulationWorkspaceSize(const _rocblaslt_handle*     h,
                                  hipblasOperation_t           opA,
                                  hipblasOperation_t           opB,
                                  int64_t                      m,
                                  int64_t                      n,
                                  int64_t                      k,
                                  const Fp64EmulationDecision& decision);

/* Per-call emulation settings.
 * Fields with sentinel values (0 for num_moduli, ~0u for sv_mask) cause the
 * function to fall back to the process-wide env var defaults.              */
struct Fp64EmulationSettings
{
    unsigned int num_moduli; /* 2..18; 0 = derive from env var          */
    unsigned int sv_mask; /* special-values mask; ~0u = env var      */
    bool         dynamic_mode; /* true when ADP (Adaptive Precision) mode */
    void*        workspace; /* caller workspace; nullptr = allocate     */
    size_t       workspace_bytes; /* size of caller workspace                */
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
                                  const Fp64EmulationSettings& settings);
