// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/*
 * FP64 GEMM emulation via Ozaki Scheme II.
 *
 * This header declares the host-side interface used by the library and tests.
 * It is intentionally free of any HIP device types so it can be included
 * from plain C++ translation units.
 *
 * Environment variables:
 *   HIPBLASLT_EMULATE_DOUBLE_PRECISION=1   enables emulation
 *   HIPBLASLT_EMULATION_NUM_MODULI=N        force fixed moduli count [2..20].
 *     WARNING: bypasses the adaptive (ADP) precision check; accuracy is not
 *     guaranteed for all inputs.  The recommended usage is NOT to set this
 *     variable and rely on the default ADP mode instead.
 *   HIPBLASLT_EMULATION_STRATEGY=performant|eager
 *     performant (default): use emulation only when predicted to outperform native DGEMM.
 *     eager: always emulate regardless of problem size or arithmetic intensity.
 *   HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=<hex>
 *     bitmask controlling NaN/Inf detection: bit 0 = Inf, bit 1 = NaN.  Default 0x3.
 *     Set to 0 to skip special-value detection entirely.
 *   HIPBLASLT_EMULATION_TOLERANCE=<value>
 *     target relative accuracy for ADP (dynamic) mode, expressed as a positive
 *     floating-point value (e.g. 1e-8, 1e-16).  ADP selects the minimum number
 *     of CRT moduli needed to achieve this accuracy; fewer moduli means faster
 *     GEMMs at the cost of lower precision.
 *     Default (absent): full IEEE 754 double precision (~1.11e-16, 52 mantissa bits).
 *     Only affects dynamic (ADP) mode; has no effect in fixed-s mode.
 *   HIPBLASLT_EMULATION_PROFILE=<path>
 *     append per-call profiling CSV rows to the given file path.
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
Fp64EmulationEnvValue fp64EmulationParseNumModuliEnv(const char* value);
Fp64EmulationEnvValue fp64EmulationParseToleranceEnv(const char* value);

/* Returns true when HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 is set.
 * The environment variable is read once and cached.
 * Any value other than "1" (including absent) returns false. */
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
 * opA/opB select per-transpose efficiency factors in the BW-kernel model.
 * workspace_bytes — caller workspace size; ~size_t{0} (SIZE_MAX) means no
 *   budget constraint (optimal, single-pass performance model). */
bool fp64EmulationPerformanceCheck(const _rocblaslt_handle* h,
                                   hipblasOperation_t       opA,
                                   hipblasOperation_t       opB,
                                   int64_t                  m,
                                   int64_t                  n,
                                   int64_t                  k,
                                   size_t                   workspace_bytes);

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

/* Returns the fixed moduli count from HIPBLASLT_EMULATION_NUM_MODULI [2..20],
 * or 0 if the env var is absent or invalid (= ADP/dynamic mode).
 * Cached on first call.  Use fp64EmulationEffectiveNumModuli(h) to get the
 * fully-resolved count (handle override → env var → S_MAX for ADP).
 * Maximum supported: 20 moduli (~155 bits of CRT capacity). */
unsigned fp64EmulationNumModuli();

/* Returns the ADP target mantissa-bit count from
 * HIPBLASLT_EMULATION_TOLERANCE, or 52 if the env var is absent or
 * invalid (= full FP64 precision).  Cached on first call.
 * Value in [1..52].                                                         */
int fp64EmulationAdpMantissaBits();

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
    size_t           workspace_cap; /* caller workspace limit; ~size_t{0} = none */
    /* ADP target precision in mantissa bits.  52 = full IEEE 754 FP64.
     * Derived from HIPBLASLT_EMULATION_TOLERANCE via floor(-log2(tol)).
     * A lower value allows ADP to choose fewer moduli, trading accuracy for speed.
     * Only consulted when dynamic_mode = true.                              */
    int              adp_mantissa_bits; /* [1..52]; 52 = full FP64 precision */
};

/* Status-returning FP64 emulation gate. Invalid env-var values return
 * rocblaslt_status_invalid_value so callers do not silently fall back to
 * native FP64. On success, apply=false means the native path should be used
 * without error.
 * opA/opB select per-transpose efficiency factors in the performance model.
 * workspace_bytes — caller workspace size; ~size_t{0} (SIZE_MAX) means
 *   no budget constraint (optimal performance assumed). */
Fp64EmulationDecision fp64EmulationDecision(const _rocblaslt_handle* h,
                                            hipDataType              type_a,
                                            hipblasOperation_t       opA,
                                            hipblasOperation_t       opB,
                                            int64_t                  m,
                                            int64_t                  n,
                                            int64_t                  k,
                                            int32_t                  batch_count,
                                            size_t                   workspace_bytes);

/* Returns the upper bound on CRT moduli (2..20) for the given handle.
 * Used as the workspace layout count and the ADP upper bound.
 *   FIXED (num_moduli ∈ [2..20]): returns that count directly.
 *   ADP  (sentinel/unset, env var absent): returns S_MAX (= 20), the upper
 *     bound used for workspace pre-allocation.  ADP selects the actual per-call
 *     s from the input data at run time; there is no fixed default. */
unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h);

/* Returns the workspace size in bytes for the given problem.
 * The decision selects the layout moduli (S_MAX=20 in ADP mode,
 * decision.num_moduli in FIXED mode).
 * When decision.workspace_cap != ~size_t{0} (set by fp64EmulationDecision
 * from the caller's workspace preference), the returned size is capped at
 * min(optimal, decision.workspace_cap).  This allows the library to report a
 * workspace size that never exceeds the user's allocation, while the
 * performance model correctly accounts for the constrained (split) execution.
 * Providing less workspace than optimal will cause multi-pass or spatial-split
 * execution; the performance model accounts for this correctly. */
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
    unsigned int num_moduli; /* 2..20; 0 = derive from env var          */
    unsigned int sv_mask; /* special-values mask; ~0u = env var      */
    bool         dynamic_mode; /* true when ADP (Adaptive Precision) mode */
    void*        workspace; /* caller workspace; nullptr = no workspace */
    size_t       workspace_bytes; /* size of caller workspace                */
    /* ADP target precision in mantissa bits [1..52]; 52 = full FP64.
     * 0 = sentinel: derive from HIPBLASLT_EMULATION_TOLERANCE (or 52).
     * Only consulted when dynamic_mode = true.                              */
    int          adp_mantissa_bits; /* 0 = env var default                 */
};

/* Run an emulated FP64 GEMM using Ozaki Scheme II (accurate mode).
 *
 * Computes D = alpha * op(A) * op(B) + beta * C  where A, B, C, D are
 * device pointers to double arrays. C and D may be the same pointer.
 * Only non-batched (batch count == 1) FP64 GEMM is supported.
 *
 * Returns rocblaslt_status_success on success,
 *         rocblaslt_status_memory_error if no workspace is provided (W==0)
 *             or the workspace is too small for emulation to outperform native DGEMM,
 *         rocblaslt_status_invalid_value if Inf/NaN is detected in the inputs
 *             (controlled by the sv_mask field in settings), or if dynamic
 *             mode (ADP) determines that even s=20 is insufficient for the
 *             given input's dynamic range.
 *             In both invalid_value cases the caller should fall back to native FP64. */
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
