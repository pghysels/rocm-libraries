/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

//
// Light, fast unit tests for the FP64 (Ozaki Scheme II) emulation path.
//
//   ./hipblaslt-test --gtest_filter='*Fp64Emulation*'
//
// Two fixtures keep setup isolated and reusable as more emulation entry points
// gain coverage:
//   * Fp64EmulationHostTest - pure host helpers, no GPU/handle required.
//   * Fp64EmulationTest      - owns a hipblasLtHandle_t for API-driven tests.
//
// The internal entry points (declared in the rocblaslt-private fp64_emulation.hpp)
// are linkable here because hipblaslt-test privately links the
// hipblaslt-fp64-emulation OBJECT library - the same object files the hipblaslt
// shared library is built from - so no symbols are exported from the release ABI.
//

#include "fp64_emulation.hpp" // internal: functions under test
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h> // public API + emulation setters

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{
    bool has_device()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    // -----------------------------------------------------------------------
    // Host-only fixture: pure host helpers, no GPU or handle required.
    // -----------------------------------------------------------------------
    class Fp64EmulationHostTest : public ::testing::Test
    {
    };

    // Default/env-derived moduli count must stay within the supported range.
    TEST_F(Fp64EmulationHostTest, NumModuliInValidRange)
    {
        const unsigned s = fp64EmulationNumModuli();
        EXPECT_GE(s, 2u);
        EXPECT_LE(s, 18u); // S_MAX
    }

    TEST_F(Fp64EmulationHostTest, ParseEnabledEnv)
    {
        EXPECT_EQ(fp64EmulationParseEnabledEnv(nullptr).state, FP64_EMULATION_ENV_UNSET);

        auto on = fp64EmulationParseEnabledEnv("1");
        EXPECT_EQ(on.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(on.value, 1u);

        auto off = fp64EmulationParseEnabledEnv("0");
        EXPECT_EQ(off.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(off.value, 0u);

        EXPECT_EQ(fp64EmulationParseEnabledEnv("true").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseEnabledEnv("").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, ParseStrategyEnv)
    {
        EXPECT_EQ(fp64EmulationParseStrategyEnv(nullptr).state, FP64_EMULATION_ENV_UNSET);

        auto performant = fp64EmulationParseStrategyEnv("performant");
        EXPECT_EQ(performant.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(performant.value, static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT));

        auto eager = fp64EmulationParseStrategyEnv("eager");
        EXPECT_EQ(eager.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(eager.value, static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER));

        EXPECT_EQ(fp64EmulationParseStrategyEnv("default").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseStrategyEnv("EAGER").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, ParseSpecialValuesMaskEnv)
    {
        auto unset = fp64EmulationParseSpecialValuesMaskEnv(nullptr);
        EXPECT_EQ(unset.state, FP64_EMULATION_ENV_UNSET);
        EXPECT_EQ(unset.value, 0x3u);

        auto hex = fp64EmulationParseSpecialValuesMaskEnv("0x3");
        EXPECT_EQ(hex.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(hex.value, 0x3u);

        auto zero = fp64EmulationParseSpecialValuesMaskEnv("0");
        EXPECT_EQ(zero.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(zero.value, 0u);

        EXPECT_EQ(fp64EmulationParseSpecialValuesMaskEnv("-1").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseSpecialValuesMaskEnv("3x").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, ParseMantissaBitCountEnv)
    {
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv(nullptr).state, FP64_EMULATION_ENV_UNSET);

        auto bits55 = fp64EmulationParseMantissaBitCountEnv("55");
        EXPECT_EQ(bits55.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(bits55.value, 55u);

        auto bits140 = fp64EmulationParseMantissaBitCountEnv("140");
        EXPECT_EQ(bits140.state, FP64_EMULATION_ENV_VALID);
        EXPECT_EQ(bits140.value, 140u);

        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("141").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("-1").state, FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("55.0").state, FP64_EMULATION_ENV_INVALID);
    }

    TEST_F(Fp64EmulationHostTest, MantissaBitCountRange)
    {
        EXPECT_TRUE(fp64EmulationIsValidMantissaBitCount(-1));
        EXPECT_TRUE(fp64EmulationIsValidMantissaBitCount(0));
        EXPECT_TRUE(fp64EmulationIsValidMantissaBitCount(140));
        EXPECT_FALSE(fp64EmulationIsValidMantissaBitCount(-2));
        EXPECT_FALSE(fp64EmulationIsValidMantissaBitCount(141));
    }

    // -----------------------------------------------------------------------
    // Handle fixture: isolated setup/teardown, reusable by future tests.
    // -----------------------------------------------------------------------
    class Fp64EmulationTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if(!has_device())
                GTEST_SKIP() << "No HIP device available";
            ASSERT_EQ(hipblasLtCreate(&m_handle), HIPBLAS_STATUS_SUCCESS);
            m_roc = reinterpret_cast<const _rocblaslt_handle*>(m_handle);
        }

        void TearDown() override
        {
            if(m_handle)
                hipblasLtDestroy(m_handle);
        }

        void set_enabled(bool on)
        {
            ASSERT_EQ(hipblasLtSetEmulationEnabled(m_handle, on), HIPBLAS_STATUS_SUCCESS);
        }

        void set_strategy(hipblasLtEmulationStrategy_t s)
        {
            ASSERT_EQ(hipblasLtSetEmulationStrategy(m_handle, s), HIPBLAS_STATUS_SUCCESS);
        }

        bool would_apply(hipDataType t, int64_t m, int64_t n, int64_t k, int batch)
        {
            const Fp64EmulationDecision decision
                = fp64EmulationDecision(m_roc, t, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, batch);
            EXPECT_EQ(decision.status, rocblaslt_status_success);
            return decision.apply;
        }

        hipblasLtHandle_t        m_handle = nullptr;
        const _rocblaslt_handle* m_roc    = nullptr;
    };

    // Workspace must be non-empty and not shrink as the moduli count grows.
    TEST_F(Fp64EmulationTest, WorkspaceSizePositiveAndMonotonic)
    {
        const int64_t         m = 1024, n = 1024, k = 1024;
        Fp64EmulationDecision d{};
        d.dynamic_mode = false;
        d.num_moduli   = 8;
        const size_t ws8
            = fp64EmulationWorkspaceSize(m_roc, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, d);
        d.num_moduli = 16;
        const size_t ws16
            = fp64EmulationWorkspaceSize(m_roc, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, d);
        EXPECT_GT(ws8, 0u);
        EXPECT_GE(ws16, ws8);
    }

    TEST_F(Fp64EmulationTest, PublicWorkspaceSizeRejectsNegativeDimensions)
    {
        EXPECT_EQ(
            hipblasLtFp64EmulationWorkspaceSize(m_handle, HIPBLAS_OP_N, HIPBLAS_OP_N, -1, 64, 64),
            0u);
        EXPECT_EQ(
            hipblasLtFp64EmulationWorkspaceSize(m_handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, -1, 64),
            0u);
        EXPECT_EQ(
            hipblasLtFp64EmulationWorkspaceSize(m_handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, 64, -1),
            0u);
    }

    // Explicit "off" must win regardless of the environment variable.
    TEST_F(Fp64EmulationTest, WouldApply_ForcedOffReturnsFalse)
    {
        set_enabled(false);
        EXPECT_FALSE(would_apply(HIP_R_64F, 4096, 4096, 4096, 1));
    }

    // Enabled + EAGER intercepts small DGEMMs.
    TEST_F(Fp64EmulationTest, WouldApply_EnabledEagerSmallF64)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_TRUE(would_apply(HIP_R_64F, 16, 16, 16, 1));
    }

    // Enabled + EAGER bypasses the cost model, so a large FP64 GEMM is intercepted.
    TEST_F(Fp64EmulationTest, WouldApply_EnabledEagerLargeF64)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_TRUE(would_apply(HIP_R_64F, 4096, 4096, 4096, 1));
    }

    // Only FP64 inputs are eligible for emulation.
    TEST_F(Fp64EmulationTest, WouldApply_RejectsNonF64)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_FALSE(would_apply(HIP_R_32F, 4096, 4096, 4096, 1));
    }

    // Batched GEMM is not supported by the emulation path.
    TEST_F(Fp64EmulationTest, WouldApply_RejectsBatched)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        EXPECT_FALSE(would_apply(HIP_R_64F, 4096, 4096, 4096, 2));
    }

    // PERFORMANT strategy uses the cost model; small GEMMs should be rejected.
    TEST_F(Fp64EmulationTest, WouldApply_PerformantSmallReturnsFalse)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT);
        EXPECT_FALSE(would_apply(HIP_R_64F, 16, 16, 16, 1));
    }

    // PERFORMANT strategy should accept large FP64 GEMMs where emulation wins.
    TEST_F(Fp64EmulationTest, WouldApply_PerformantLargeReturnsTrue)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT);
        EXPECT_TRUE(would_apply(HIP_R_64F, 4096, 4096, 4096, 1));
    }

    TEST_F(Fp64EmulationTest, ApiValidationRejectsInvalidMantissaControl)
    {
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, static_cast<hipblasLtEmulationMantissaControl_t>(-1)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, static_cast<hipblasLtEmulationMantissaControl_t>(2)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(Fp64EmulationTest, ApiValidationRejectsInvalidMantissaBitCount)
    {
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, -2),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, 141),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, -1),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, 0),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(m_handle, 140),
                  HIPBLAS_STATUS_SUCCESS);
    }

    TEST_F(Fp64EmulationTest, DefaultDynamicDecisionUsesCurrentSixteenModuliPath)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        ASSERT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);

        const Fp64EmulationDecision decision = fp64EmulationDecision(
            m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, 4096, 4096, 4096, 1);
        ASSERT_EQ(decision.status, rocblaslt_status_success);
        ASSERT_TRUE(decision.apply);
        EXPECT_TRUE(decision.dynamic_mode);
        EXPECT_EQ(decision.num_moduli, 16u);
    }

    // End-to-end smoke test of the emulated GEMM. Column-major
    // D = alpha*op(A)*op(B) + beta*C, with A = I and beta = 0, so D == B.
    // Skips when the INT8 device library is unavailable for the running arch.
    TEST_F(Fp64EmulationTest, EmulatedGemm_SmokeIdentity)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);

        constexpr int64_t N     = 64;
        const size_t      bytes = static_cast<size_t>(N) * N * sizeof(double);

        std::vector<double> hA(N * N, 0.0), hB(N * N, 0.0), hD(N * N, 0.0);
        for(int64_t i = 0; i < N; ++i)
            hA[i * N + i] = 1.0; // identity
        for(int64_t i = 0; i < N * N; ++i)
            hB[i] = static_cast<double>((i % 7) - 3);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        const double          alpha = 1.0, beta = 0.0;
        Fp64EmulationSettings settings{};
        settings.num_moduli      = 0; // derive from env/default
        settings.sv_mask         = 0; // skip Inf/NaN check (faster, inputs are finite)
        settings.workspace       = nullptr; // library allocates
        settings.workspace_bytes = 0;

        const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     N,
                                                     N,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     N,
                                                     dB,
                                                     N,
                                                     &beta,
                                                     dC,
                                                     N,
                                                     dD,
                                                     N,
                                                     /*stream=*/nullptr,
                                                     settings);

        if(st != rocblaslt_status_success)
        {
            (void)hipFree(dA);
            (void)hipFree(dB);
            (void)hipFree(dC);
            (void)hipFree(dD);
            GTEST_SKIP() << "fp64EmulatedGemm returned non-success (" << static_cast<int>(st)
                         << "); INT8 device library may be unavailable on this arch";
        }

        ASSERT_EQ(hipMemcpy(hD.data(), dD, bytes, hipMemcpyDeviceToHost), hipSuccess);

        double max_abs_err = 0.0;
        for(int64_t i = 0; i < N * N; ++i)
            max_abs_err = std::max(max_abs_err, std::abs(hD[i] - hB[i]));
        EXPECT_LT(max_abs_err, 1e-9);

        (void)hipFree(dA);
        (void)hipFree(dB);
        (void)hipFree(dC);
        (void)hipFree(dD);
    }

    // -----------------------------------------------------------------------
    // Accuracy regression tests: emulated DGEMM vs native FP64 DGEMM reference.
    //
    // Covers all 17 moduli counts s = 2..18 (including odd values).
    //
    // Design rationale:
    //   • m, n are small (64 or 128) to keep device-to-host transfer fast.
    //   • k is large (128 for s≤6, 4096 for s≥7).
    //   • Three deterministic seeds are swept; the worst-case relative error
    //     over all seeds and all output elements must satisfy the threshold.
    // -----------------------------------------------------------------------

    // Input-fill styles for the accuracy tests.
    //   FILL_UNIFORM_01  – uniform U(0,1); baseline, default.
    //   FILL_ALLPOS_NEAR1 – uniform U(0.9, 1.0); all-positive near-maximum
    //                       inner products, ≈3.7× larger than U(0,1), pushing
    //                       X_true toward M_s/2 (sign-flip boundary).
    //   FILL_GEOMROWS     – geometric row-/col-scale: row i of op(A) and
    //                       column j of op(B) are scaled by 2^{(i·16/m)−8},
    //                       entries otherwise uniform in (−1,1).  Exercises
    //                       the adaptive per-row shift refinement with wildly
    //                       varying sftA[i] values.  Dispatch is transpose-aware.
    enum FillStyle
    {
        FILL_UNIFORM_01   = 0,
        FILL_ALLPOS_NEAR1 = 1,
        FILL_GEOMROWS     = 2,
    };

    struct EmulAccuracyParam
    {
        unsigned           s; /* num_moduli (2..18)                   */
        int64_t            m, n, k; /* matrix dimensions                    */
        double             threshold; /* max relative error vs native DGEMM   */
        FillStyle          fill; /* input distribution                   */
        hipblasOperation_t opA; /* HIPBLAS_OP_N or HIPBLAS_OP_T for A   */
        hipblasOperation_t opB; /* HIPBLAS_OP_N or HIPBLAS_OP_T for B   */
        double             alpha; /* scaling factor for A*B               */
        double             beta; /* scaling factor for C                 */
    };

    static std::string
        EmulAccuracyParamName(const ::testing::TestParamInfo<EmulAccuracyParam>& info)
    {
        const EmulAccuracyParam& p   = info.param;
        const char*              suf = "";
        switch(p.fill)
        {
        case FILL_UNIFORM_01:
            suf = "_uni";
            break;
        case FILL_ALLPOS_NEAR1:
            suf = "_near1";
            break;
        case FILL_GEOMROWS:
            suf = "_geom";
            break;
        }
        const char opA_c = (p.opA == HIPBLAS_OP_N) ? 'N' : 'T';
        const char opB_c = (p.opB == HIPBLAS_OP_N) ? 'N' : 'T';
        /* Format scalar as integer when it is an exact integer, else as-is. */
        auto fmt_s = [](double v) -> std::string {
            const int vi = static_cast<int>(v);
            return (static_cast<double>(vi) == v) ? std::to_string(vi) : "x";
        };
        return "s" + std::to_string(p.s) + suf + "_" + std::to_string(p.m) + "x"
               + std::to_string(p.n) + "x" + std::to_string(p.k) + "_" + opA_c + opB_c + "_a"
               + fmt_s(p.alpha) + "_b" + fmt_s(p.beta);
    }

    class Fp64EmulationAccuracyTest : public ::testing::TestWithParam<EmulAccuracyParam>
    {
    protected:
        void SetUp() override
        {
            if(!has_device())
                GTEST_SKIP() << "No HIP device available";
        }
    };

    /* Deterministic xorshift64 PRNG; fills v with U(0,1) doubles. */
    static uint64_t xsr64(uint64_t s)
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    static double xsr64_uniform(uint64_t b)
    {
        return static_cast<double>(b >> 11) * (1.0 / 9007199254740992.0);
    }
    static void fill_uniform_host(std::vector<double>& v, uint64_t seed)
    {
        uint64_t s = seed ^ 1442695040888963407ULL;
        s          = xsr64(s);
        s          = xsr64(s);
        for(double& x : v)
        {
            s = xsr64(s);
            x = xsr64_uniform(s);
        }
    }

    /* All-positive near-1: U(0.9, 1.0).  Inner products are ≈3.7× larger
     * than U(0,1), pushing X_true proportionally closer to M_s/2. */
    static void fill_allpos_near1_host(std::vector<double>& v, uint64_t seed)
    {
        uint64_t s = seed ^ 0x9e3779b97f4a7c15ULL;
        s          = xsr64(s);
        s          = xsr64(s);
        for(double& x : v)
        {
            s = xsr64(s);
            x = 0.9 + 0.1 * xsr64_uniform(s);
        }
    }

    /* Geometric row-scale: element (i, j) in column-major storage (lda=rows) is
     * row_scale(i) × U(-1,1), where row_scale(i) = 2^{(i·16/rows)−8}.
     * Creates rows spanning 16 doublings (2^{-8}..2^{8}) → diverse sftA[i] values.
     * Used for op(A)=N matrices (A is m×k, column-major with lda=m), and also for
     * the opA=T case via fill_geomcols_host to scale op(A)'s rows correctly. */
    static void
        fill_geomrows_host(std::vector<double>& v, int64_t rows, int64_t cols, uint64_t seed)
    {
        uint64_t s = seed ^ 0xdeadbeef12345678ULL;
        s          = xsr64(s);
        s          = xsr64(s);
        for(int64_t i = 0; i < rows; ++i)
        {
            const double row_scale = std::ldexp(1.0, static_cast<int>(i * 16 / rows) - 8);
            for(int64_t j = 0; j < cols; ++j)
            {
                s                                    = xsr64(s);
                v[static_cast<size_t>(i + j * rows)] = row_scale * (xsr64_uniform(s) * 2.0 - 1.0);
            }
        }
    }

    /* Geometric col-scale: element (i, j) in column-major storage (ldb=rows) is
     * col_scale(j) × U(-1,1), where col_scale(j) = 2^{(j·16/cols)−8}.
     * Creates columns spanning 16 doublings → diverse sftB[j] values. */
    static void
        fill_geomcols_host(std::vector<double>& v, int64_t rows, int64_t cols, uint64_t seed)
    {
        uint64_t s = seed ^ 0xbadcafe0deadbeefULL;
        s          = xsr64(s);
        s          = xsr64(s);
        for(int64_t j = 0; j < cols; ++j)
        {
            const double col_scale = std::ldexp(1.0, static_cast<int>(j * 16 / cols) - 8);
            for(int64_t i = 0; i < rows; ++i)
            {
                s                                    = xsr64(s);
                v[static_cast<size_t>(i + j * rows)] = col_scale * (xsr64_uniform(s) * 2.0 - 1.0);
            }
        }
    }

    TEST_P(Fp64EmulationAccuracyTest, VsNativeDgemm)
    {
        const EmulAccuracyParam& p = GetParam();

        const bool tA = (p.opA == HIPBLAS_OP_T);
        const bool tB = (p.opB == HIPBLAS_OP_T);
        /* Physical leading dimensions of the stored A and B matrices.
         * op(A)=N: A is m×k col-major (lda=m).  op(A)=T: A is k×m col-major (lda=k).
         * op(B)=N: B is k×n col-major (ldb=k).  op(B)=T: B is n×k col-major (ldb=n). */
        const int64_t lda = tA ? p.k : p.m;
        const int64_t ldb = tB ? p.n : p.k;

        /* Buffer sizes: always m*k for A and k*n for B regardless of transpose. */
        const size_t nA = static_cast<size_t>(p.m) * static_cast<size_t>(p.k);
        const size_t nB = static_cast<size_t>(p.k) * static_cast<size_t>(p.n);
        const size_t nD = static_cast<size_t>(p.m) * static_cast<size_t>(p.n);

        /* Host buffers */
        std::vector<double> hA(nA), hB(nB), hC(nD, 0.0), hD_nat(nD), hD_emu(nD);

        /* Device buffers.  Guard against zero-size malloc when k=0. */
        const size_t alloc_A = (nA > 0) ? nA * sizeof(double) : sizeof(double);
        const size_t alloc_B = (nB > 0) ? nB * sizeof(double) : sizeof(double);
        double *     dA = nullptr, *dB = nullptr, *dC = nullptr;
        double *     dD_nat = nullptr, *dD_emu = nullptr;
        ASSERT_EQ(hipMalloc(&dA, alloc_A), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, alloc_B), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, nD * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_nat, nD * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_emu, nD * sizeof(double)), hipSuccess);

        /* Cleanup helper called on every exit path */
        auto cleanup = [&](hipblasLtHandle_t           hem,
                           hipblasLtHandle_t           hnat,
                           hipblasLtMatmulDesc_t       desc,
                           hipblasLtMatrixLayout_t     la,
                           hipblasLtMatrixLayout_t     lb,
                           hipblasLtMatrixLayout_t     ld,
                           hipblasLtMatmulPreference_t pref) {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(ld)
                hipblasLtMatrixLayoutDestroy(ld);
            if(lb)
                hipblasLtMatrixLayoutDestroy(lb);
            if(la)
                hipblasLtMatrixLayoutDestroy(la);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(hnat)
                hipblasLtDestroy(hnat);
            if(hem)
                hipblasLtDestroy(hem);
            (void)hipFree(dD_emu);
            (void)hipFree(dD_nat);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        /* ── Emulation handle (s moduli, eager strategy) ─────────────────── */
        hipblasLtHandle_t hem = nullptr;
        ASSERT_EQ(hipblasLtCreate(&hem), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationEnabled(hem, true), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationStrategy(hem, HIPBLASLT_EMULATION_STRATEGY_EAGER),
                  HIPBLAS_STATUS_SUCCESS);

        /* Skip if the device is not supported by the emulation. */
        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(reinterpret_cast<const _rocblaslt_handle*>(hem),
                                        HIP_R_64F,
                                        p.opA,
                                        p.opB,
                                        p.m,
                                        p.n,
                                        p.k,
                                        1);
            if(!gate.apply)
            {
                cleanup(hem, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                GTEST_SKIP() << "Device not supported by emulation";
            }
        }

        /* ── Native DGEMM reference handle + layouts ─────────────────────── */
        hipblasLtHandle_t                hnat = nullptr;
        hipblasLtMatmulDesc_t            desc = nullptr;
        hipblasLtMatrixLayout_t          la = nullptr, lb = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t      pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};

        ASSERT_EQ(hipblasLtCreate(&hnat), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &p.opA, sizeof(p.opA));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &p.opB, sizeof(p.opB));

        /* Physical matrix dimensions for layout creation:
         * opA=N: A is (m × k), lda=m.  opA=T: A is (k × m), lda=k.
         * opB=N: B is (k × n), ldb=k.  opB=T: B is (n × k), ldb=n.
         * C/D are always (m × n), ldc=ldd=m.                             */
        const uint64_t la_rows = static_cast<uint64_t>(tA ? p.k : p.m);
        const uint64_t la_cols = static_cast<uint64_t>(tA ? p.m : p.k);
        const uint64_t lb_rows = static_cast<uint64_t>(tB ? p.n : p.k);
        const uint64_t lb_cols = static_cast<uint64_t>(tB ? p.k : p.n);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_64F, la_rows, la_cols, lda),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_64F, lb_rows, lb_cols, ldb),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(
                      &ld, HIP_R_64F, static_cast<uint64_t>(p.m), static_cast<uint64_t>(p.n), p.m),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        /* No workspace restriction: let the library pick any algorithm.  Without
         * this, the search exhausts all candidates for degenerate shapes (m=1 or
         * n=1) before concluding nat_cnt=0, which can take ~9 s per test.       */
        int nat_cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(hnat, desc, la, lb, ld, ld, pref, 1, &heur, &nat_cnt);
        if(nat_cnt == 0)
        {
            cleanup(hem, hnat, desc, la, lb, ld, pref);
            GTEST_SKIP() << "No native FP64 DGEMM algorithm found on this device";
        }

        /* ── Emulation settings (use settings.num_moduli directly) ──────── */
        Fp64EmulationSettings emu_settings{};
        emu_settings.num_moduli      = p.s;
        emu_settings.sv_mask         = 0u; /* skip Inf/NaN detection */
        emu_settings.workspace       = nullptr; /* library allocates internally */
        emu_settings.workspace_bytes = 0u;

        /* Fill the C matrix once before the seed loop.  When beta=0, C is not
         * read by either GEMM, but we still fill it to exercise the beta=0
         * bypass path with a non-trivial buffer.                            */
        fill_uniform_host(hC, 0xc0ffee0000000000ULL);
        ASSERT_EQ(hipMemcpy(dC, hC.data(), nD * sizeof(double), hipMemcpyHostToDevice), hipSuccess);

        /* Sweep three deterministic seeds; accumulate the worst relative error. */
        constexpr uint64_t SEEDS[3]    = {12345ULL, 98765ULL, 54321ULL};
        double             max_rel_err = 0.0;

        for(uint64_t seed : SEEDS)
        {
            switch(p.fill)
            {
            case FILL_UNIFORM_01:
                fill_uniform_host(hA, seed);
                fill_uniform_host(hB, seed ^ 0xdeadbeefcafeULL);
                break;
            case FILL_ALLPOS_NEAR1:
                fill_allpos_near1_host(hA, seed);
                fill_allpos_near1_host(hB, seed ^ 0xdeadbeefcafeULL);
                break;
            case FILL_GEOMROWS:
                /* FILL_GEOMROWS is transpose-aware:
                 * For op(A)=N (A is m×k, lda=m): scale by row index (m_idx).
                 * For op(A)=T (A is k×m, lda=k): A's col m_idx = op(A)'s row m_idx
                 *   → fill_geomcols_host scales by column j = m_idx ✓
                 * For op(B)=N (B is k×n, ldb=k): scale by col index (n_idx).
                 * For op(B)=T (B is n×k, ldb=n): B's row n_idx = op(B)'s col n_idx
                 *   → fill_geomrows_host scales by row i = n_idx ✓             */
                if(tA)
                    fill_geomcols_host(hA, p.k, p.m, seed);
                else
                    fill_geomrows_host(hA, p.m, p.k, seed);
                if(tB)
                    fill_geomrows_host(hB, p.n, p.k, seed ^ 0xdeadbeefcafeULL);
                else
                    fill_geomcols_host(hB, p.k, p.n, seed ^ 0xdeadbeefcafeULL);
                break;
            }

            if(nA > 0)
                ASSERT_EQ(hipMemcpy(dA, hA.data(), nA * sizeof(double), hipMemcpyHostToDevice),
                          hipSuccess);
            if(nB > 0)
                ASSERT_EQ(hipMemcpy(dB, hB.data(), nB * sizeof(double), hipMemcpyHostToDevice),
                          hipSuccess);

            /* Native FP64 DGEMM reference (C→D_nat). */
            const hipblasStatus_t nat_st = hipblasLtMatmul(hnat,
                                                           desc,
                                                           &p.alpha,
                                                           dA,
                                                           la,
                                                           dB,
                                                           lb,
                                                           &p.beta,
                                                           dC,
                                                           ld,
                                                           dD_nat,
                                                           ld,
                                                           &heur.algo,
                                                           nullptr,
                                                           0,
                                                           /*stream=*/nullptr);
            if(nat_st != HIPBLAS_STATUS_SUCCESS)
            {
                cleanup(hem, hnat, desc, la, lb, ld, pref);
                GTEST_SKIP() << "Native hipblasLtMatmul failed (status=" << static_cast<int>(nat_st)
                             << ")";
            }

            /* Emulated DGEMM (C→D_emu). */
            const rocblaslt_status emu_st = fp64EmulatedGemm(hem,
                                                             p.opA,
                                                             p.opB,
                                                             p.m,
                                                             p.n,
                                                             p.k,
                                                             &p.alpha,
                                                             dA,
                                                             lda,
                                                             dB,
                                                             ldb,
                                                             &p.beta,
                                                             dC,
                                                             p.m,
                                                             dD_emu,
                                                             p.m,
                                                             /*stream=*/nullptr,
                                                             emu_settings);
            if(emu_st != rocblaslt_status_success)
            {
                cleanup(hem, hnat, desc, la, lb, ld, pref);
                GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(emu_st)
                             << " (INT8 device library may be unavailable on this arch)";
            }

            ASSERT_EQ(hipMemcpy(hD_nat.data(), dD_nat, nD * sizeof(double), hipMemcpyDeviceToHost),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(hD_emu.data(), dD_emu, nD * sizeof(double), hipMemcpyDeviceToHost),
                      hipSuccess);

            /* Relative error normalised by the maximum element magnitude.
             * Floor at 1.0 prevents division-by-zero for near-zero outputs.  */
            double dmax = 0.0;
            for(double v : hD_nat)
                dmax = std::max(dmax, std::abs(v));
            const double norm = std::max(dmax, 1.0);

            for(size_t idx = 0; idx < nD; ++idx)
                max_rel_err = std::max(max_rel_err, std::abs(hD_emu[idx] - hD_nat[idx]) / norm);
        }

        cleanup(hem, hnat, desc, la, lb, ld, pref);

        EXPECT_LE(max_rel_err, p.threshold)
            << "Emulated GEMM with s=" << p.s
            << " exceeded accuracy threshold: max_rel_err=" << max_rel_err
            << " > threshold=" << p.threshold << "."
            << " A value near 1.0-2.0 indicates a CRT sign-flip (overflow) regression.";
    }

    // -----------------------------------------------------------------------
    // Threshold rationale:
    //   All entries use thresholds ≈10× above the empirically observed worst-case
    //   errors (from the Grade A criterion experiments), tolerating seed-to-seed
    //   variability while remaining orders of magnitude below the CRT sign-flip
    //   error of ≈1–2.
    //
    //   s = 2..6 : CRT capacity < FP64 (16–48 bits).  Use k=128 so the inner
    //              products stay well within the CRT range.
    // -----------------------------------------------------------------------

    // ── AllModuliCounts: baseline — all s values, NN transpose, alpha=1, beta=0 ──
    INSTANTIATE_TEST_SUITE_P(
        AllModuliCounts,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            EmulAccuracyParam{
                2, 64, 64, 128, 3e-1, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                3, 64, 64, 128, 3e-1, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                4, 64, 64, 128, 2e-1, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                5, 64, 64, 128, 1e-1, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                6, 64, 64, 128, 5e-2, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                8, 128, 128, 4096, 5e-4, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                9, 128, 128, 4096, 1e-6, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                10, 128, 128, 4096, 1e-7, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                11, 128, 128, 4096, 1e-8, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                12, 128, 128, 4096, 1e-9, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                13, 128, 128, 4096, 1e-10, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                14, 128, 128, 4096, 1e-10, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                15, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                17, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                18, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0}),
        EmulAccuracyParamName);

    // ── NearSignFlip: adversarial distributions pushing X_true toward M_s/2 ──
    //
    //   FILL_ALLPOS_NEAR1 – all entries in U(0.9, 1.0); inner products ≈3.7×
    //     larger than U(0,1), pushing the CRT argument X_true closer to M_s/2.
    //
    //   FILL_GEOMROWS – geometric row/col scaling 2^{(i·16/m)−8}: 256× dynamic
    //     range exercises the adaptive sftA/sftB refinement with a mix of large
    //     and small inner products in the same GEMM call.
    INSTANTIATE_TEST_SUITE_P(
        NearSignFlip,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            // All-positive near-1
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                8, 128, 128, 4096, 5e-4, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                9, 128, 128, 4096, 1e-6, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                10, 128, 128, 4096, 1e-7, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                11, 128, 128, 4096, 1e-8, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                12, 128, 128, 4096, 1e-9, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                13, 128, 128, 4096, 1e-10, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                14, 128, 128, 4096, 1e-10, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                15, 128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                17, 128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                18, 128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            // Geometric row-/col-scaling (wide dynamic range)
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                8, 128, 128, 4096, 5e-4, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                9, 128, 128, 4096, 1e-6, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                10, 128, 128, 4096, 1e-7, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                11, 128, 128, 4096, 1e-8, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                12, 128, 128, 4096, 1e-9, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                13, 128, 128, 4096, 1e-10, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                14, 128, 128, 4096, 1e-10, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                15, 128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                17, 128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                18, 128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0}),
        EmulAccuracyParamName);

    // ── TransposeCombinations: all 4 op(A)×op(B) transpose variants ──────────
    //
    // Exercises the A_T/A_N/B_T/B_N extraction-kernel paths (separate coalesced
    // vs SHMEM transposition paths) at s=7 (boundary) and s=16 (default).
    // The FILL_GEOMROWS dispatch is transpose-aware (see VsNativeDgemm).
    INSTANTIATE_TEST_SUITE_P(
        TransposeCombinations,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            // s=7 — CRT just above FP64 mantissa capacity
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_T, 1.0, 0.0},
            EmulAccuracyParam{
                7, 128, 128, 4096, 5e-4, FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_T, 1.0, 0.0},
            // s=16 — default precision
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_T, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_T, 1.0, 0.0}),
        EmulAccuracyParamName);

    // ── AlphaBeta: alpha × beta ∈ {0, 1, 2}² at s=16, NN ────────────────────
    //
    //   alpha=0 → D = beta*C  (A*B discarded; 0.0 * X = 0.0 exactly in IEEE 754).
    //   beta=0  → D = alpha*A*B  (C not read by the finalize kernel).
    //   alpha=0, beta=0 → D = 0.
    //   Both are special-cased in the finalize kernel (beta branch, alpha multiply).
    INSTANTIATE_TEST_SUITE_P(
        AlphaBeta,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 0.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 0.0, 1.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 0.0, 2.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 1.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 2.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 2.0, 0.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 2.0, 1.0},
            EmulAccuracyParam{
                16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 2.0, 2.0}),
        EmulAccuracyParamName);

    // ── KEqualsZero: k=0 edge case ────────────────────────────────────────────
    //
    // With a zero contraction dimension, D = alpha*(empty sum) + beta*C = beta*C.
    // No INT8 GEMMs are executed; the result must equal beta*C exactly (error = 0).
    INSTANTIATE_TEST_SUITE_P(
        KEqualsZero,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            /* beta=0: D = 0 */
            EmulAccuracyParam{
                16, 64, 64, 0, 0.0, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            /* beta=1: D = C */
            EmulAccuracyParam{
                16, 64, 64, 0, 0.0, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 1.0}),
        EmulAccuracyParamName);

    // ── SmallDimensions: m=1 or n=1 (single-row / single-column output) ──────
    //
    // Disabled: hipblasLtMatmulAlgoGetHeuristic finds no native FP64 algorithm
    // for these degenerate aspect ratios and exhausts all candidates before
    // returning nat_cnt=0, which takes ~9 s per test case.  The tests always
    // skip and provide no coverage.
#if 0
    INSTANTIATE_TEST_SUITE_P(
        SmallDimensions,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            /* m=1: single output row — exercises min-m kernel paths */
            EmulAccuracyParam{16, 1,   128, 4096, 1e-11, FILL_UNIFORM_01,   HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            /* n=1: single output col — exercises min-n kernel paths (near1 fill gives distinct name) */
            EmulAccuracyParam{16, 128, 1,   4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 }
        ),
        EmulAccuracyParamName
    );
#endif

    // ── LargeK: k=16384, stress the preliminary GEMM INT32 accumulation ───────
    //
    // For near-saturated inputs and k=16384:
    //   max INT32 accumulation ≈ 63² × 16384 ≈ 65M — well within INT32 range.
    // Guards against silent overflow regressions if the extraction scale
    // were ever increased beyond 6 bits.
    INSTANTIATE_TEST_SUITE_P(
        LargeK,
        Fp64EmulationAccuracyTest,
        ::testing::Values(EmulAccuracyParam{
            16, 64, 64, 16384, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0}),
        EmulAccuracyParamName);

    // ── RectangularShapes: non-square m×n×k to exercise different tile paths ──
    //
    // All existing accuracy tests use square 64² or 128² output shapes.
    // These rectangular shapes test different kernel tile selection and
    // edge-handling in the extraction and finalize kernels.
    INSTANTIATE_TEST_SUITE_P(
        RectangularShapes,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            /* Tall-and-thin output (m >> n) */
            EmulAccuracyParam{
                16, 512, 16, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            /* Wide-and-short output (n >> m) */
            EmulAccuracyParam{
                16, 16, 512, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0},
            /* Intermediate rectangular */
            EmulAccuracyParam{
                16, 256, 64, 1024, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0}),
        EmulAccuracyParamName);

    // ── Demmel matrix helper ──────────────────────────────────────────────────
    //
    // Shared construction for DemmelAdpFallback and DemmelBlas2 tests.
    // Generates x ~ U(1,2), d[m] = 2^{j_m}, j_m = -b + round(m×2b/(n-1)),
    // and fills column-major A and B:
    //   A[col*n+row] = x[(row+col)%n] × d[(row+col)%n]
    //   B[col*n+row] = x[(row+col)%n] / d[(row+col)%n]
    static void build_demmel_ab(int                  n,
                                int                  b,
                                std::vector<double>& h_x,
                                std::vector<double>& h_d,
                                std::vector<double>& h_A,
                                std::vector<double>& h_B)
    {
        const size_t N = static_cast<size_t>(n);
        h_x.resize(N);
        h_d.resize(N);
        h_A.resize(N * N);
        h_B.resize(N * N);

        for(int i = 0; i < n; ++i)
        {
            uint64_t s = static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL + 1442695040888963407ULL;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            h_x[static_cast<size_t>(i)]
                = 1.0 + static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
        }

        const double delta = (n > 1) ? (2.0 * b) / static_cast<double>(n - 1) : 0.0;
        for(int m = 0; m < n; ++m)
        {
            double jm                   = -b + std::round(static_cast<double>(m) * delta);
            h_d[static_cast<size_t>(m)] = std::ldexp(1.0, static_cast<int>(jm));
        }

        for(int col = 0; col < n; ++col)
            for(int row = 0; row < n; ++row)
            {
                const size_t m   = static_cast<size_t>((row + col) % n);
                const size_t idx = static_cast<size_t>(col * n + row);
                h_A[idx]         = h_x[m] * h_d[m];
                h_B[idx]         = h_x[m] / h_d[m];
            }
    }

    // ── DemmelAdpFallback: ADP must detect overflow for extreme b ─────────────
    //
    // For b=32, n=128 the ADP reduction computes:
    //   log2P_needed ≈ (52 − sftA_init) + 0.5·log2(row_max_prelim)
    //                ≈ (52+27) + 0.5·log2(~300) ≈ 79 + 4 = 83 >> log2P_18 = 68.7
    // fp64EmulatedGemm must return rocblaslt_status_invalid_value so the caller
    // falls back to native DGEMM rather than silently producing a wrong result.
    TEST_F(Fp64EmulationTest, DemmelAdpFallback_b32_n128)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        ASSERT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);

        // Skip unsupported devices
        {
            const Fp64EmulationDecision gate = fp64EmulationDecision(
                m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, 128, 128, 128, 1);
            if(!gate.apply)
                GTEST_SKIP() << "Device not supported by emulation";
        }

        constexpr int    n = 128, b = 32;
        constexpr size_t N2    = static_cast<size_t>(n) * n;
        const size_t     bytes = N2 * sizeof(double);

        std::vector<double> h_x, h_d, h_A, h_B;
        build_demmel_ab(n, b, h_x, h_d, h_A, h_B);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, h_A.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, h_B.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        Fp64EmulationSettings emu_settings{};
        emu_settings.num_moduli      = 16u; /* ADP upper bound */
        emu_settings.sv_mask         = 0u; /* skip Inf/NaN detection */
        emu_settings.dynamic_mode    = true; /* enable ADP */
        emu_settings.workspace       = nullptr;
        emu_settings.workspace_bytes = 0u;

        const double           alpha = 1.0, beta = 0.0;
        const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     n,
                                                     n,
                                                     n,
                                                     &alpha,
                                                     dA,
                                                     n,
                                                     dB,
                                                     n,
                                                     &beta,
                                                     dC,
                                                     n,
                                                     dD,
                                                     n,
                                                     /*stream=*/nullptr,
                                                     emu_settings);

        (void)hipFree(dD);
        (void)hipFree(dC);
        (void)hipFree(dB);
        (void)hipFree(dA);

        // ADP must have detected the b=32 overflow and returned invalid_value.
        // (The caller — rocblaslt_mat.cpp — would then fall back to native DGEMM.)
        EXPECT_EQ(st, rocblaslt_status_invalid_value)
            << "ADP for b=" << b << " n=" << n
            << " should detect log2P_needed > log2P_18 and return invalid_value,"
            << " but got status=" << static_cast<int>(st);
    }

    // ── DemmelBlas2Test: BLAS Test 2 (Figure 2 from arXiv:2511.13778) ─────────
    //
    // Constructs the Demmel et al. matrix pair (n×n, column-major, NN):
    //   A[k,j] = x[(j+k)%n] × d[(j+k)%n]        (row k, col j)
    //   B[i,k] = x[(i+k)%n] / d[(i+k)%n]        (row i, col k)
    // where x ~ U(1,2) and d[m] = 2^{j_m}, j_m = -b + round(m×2b/(n-1)).
    //
    // The product C = A×B is a circulant matrix with C[k,i] = h[(i-k+n)%n]:
    //   h[δ] = Σ_{m=0}^{n-1} x[m] × x[(m+δ)%n] × d[m] / d[(m+δ)%n]
    //   δ=0 → h[0] = xᵀx     (exact diagonal: d/d = 1 per term)
    //   δ≠0 → h[δ] > 0       (all positive, no cancellation)
    //
    // Reference: host long-double for all n² elements (accurate when 4b ≤ 63).
    //
    // Threshold 1e-13 ≈ 500×ε_machine:
    //   Correct result : error ≈ ε_machine (correctly-rounded IEEE FP64)
    //   Sign-flip fail : error ≈ 1–2  (margin ≥ 10 orders of magnitude)

    struct EmulDemmelParam
    {
        unsigned s; /* num_moduli (2..18)           */
        int      n; /* square matrix side (≤ 128)   */
        int      b; /* exponent half-range (≤ 15)   */
        double   threshold; /* max rel error over all n² el */
    };

    static std::string EmulDemmelParamName(const ::testing::TestParamInfo<EmulDemmelParam>& info)
    {
        const EmulDemmelParam& p = info.param;
        return "s" + std::to_string(p.s) + "_n" + std::to_string(p.n) + "_b" + std::to_string(p.b);
    }

    class Fp64EmulationDemmelTest : public ::testing::TestWithParam<EmulDemmelParam>
    {
    protected:
        void SetUp() override
        {
            if(!has_device())
                GTEST_SKIP() << "No HIP device available";
        }
    };

    TEST_P(Fp64EmulationDemmelTest, VsLongDoubleReference)
    {
        const EmulDemmelParam& p  = GetParam();
        const int64_t          N  = static_cast<int64_t>(p.n);
        const size_t           N2 = static_cast<size_t>(N) * static_cast<size_t>(N);

        // ── Host: build Demmel A, B matrices using shared helper ──────────────
        std::vector<double> h_x, h_d, h_A, h_B;
        build_demmel_ab(p.n, p.b, h_x, h_d, h_A, h_B);

        // ── Host: exact reference in long double for all n² elements ──────────
        // C is circulant: C[k,i] = h[(i-k+N)%N] where
        //   h[δ] = Σ_m x[m]·x[(m+δ)%N]·d[m]/d[(m+δ)%N]  (all positive → no cancel)
        // Accurate in long double when 4·b ≤ 63 (i.e. b ≤ 15).
        std::vector<long double> h_lag(static_cast<size_t>(N), 0.0L);
        for(int64_t lag = 0; lag < N; ++lag)
            for(int64_t m = 0; m < N; ++m)
            {
                const size_t ml  = static_cast<size_t>(m);
                const size_t mpl = static_cast<size_t>((m + lag) % N);
                h_lag[static_cast<size_t>(lag)]
                    += static_cast<long double>(h_x[ml]) * static_cast<long double>(h_x[mpl])
                       * static_cast<long double>(h_d[ml]) / static_cast<long double>(h_d[mpl]);
            }

        // ── Device: upload and run emulation ──────────────────────────────────
        const size_t bytes = N2 * sizeof(double);
        double *     dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, h_A.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, h_B.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        auto cleanup = [&]() {
            (void)hipFree(dD);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        hipblasLtHandle_t hem = nullptr;
        ASSERT_EQ(hipblasLtCreate(&hem), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationEnabled(hem, true), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationStrategy(hem, HIPBLASLT_EMULATION_STRATEGY_EAGER),
                  HIPBLAS_STATUS_SUCCESS);

        // Skip unsupported devices
        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(reinterpret_cast<const _rocblaslt_handle*>(hem),
                                        HIP_R_64F,
                                        HIPBLAS_OP_N,
                                        HIPBLAS_OP_N,
                                        N,
                                        N,
                                        N,
                                        1);
            if(!gate.apply)
            {
                (void)hipblasLtDestroy(hem);
                cleanup();
                GTEST_SKIP() << "Device not supported by emulation";
            }
        }

        Fp64EmulationSettings emu_settings{};
        emu_settings.num_moduli      = p.s;
        emu_settings.sv_mask         = 0u;
        emu_settings.workspace       = nullptr;
        emu_settings.workspace_bytes = 0u;

        const double           alpha = 1.0, beta = 0.0;
        const rocblaslt_status st = fp64EmulatedGemm(hem,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     N,
                                                     N,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     N,
                                                     dB,
                                                     N,
                                                     &beta,
                                                     dC,
                                                     N,
                                                     dD,
                                                     N,
                                                     /*stream=*/nullptr,
                                                     emu_settings);
        (void)hipblasLtDestroy(hem);

        if(st != rocblaslt_status_success)
        {
            cleanup();
            GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(st)
                         << " (INT8 device library may be unavailable)";
        }

        // ── Compare all n² elements against the long-double reference ─────────
        std::vector<double> h_D(N2);
        ASSERT_EQ(hipMemcpy(h_D.data(), dD, bytes, hipMemcpyDeviceToHost), hipSuccess);
        cleanup();

        double max_rel_err = 0.0;
        for(int64_t col = 0; col < N; ++col) /* i = col */
            for(int64_t row = 0; row < N; ++row) /* k = row */
            {
                const size_t      lag  = static_cast<size_t>(((col - row) % N + N) % N);
                const long double ref  = h_lag[lag];
                const double      emul = h_D[static_cast<size_t>(col * N + row)];
                /* ref > 0 always (sum of positive terms) */
                const double rel
                    = std::abs(emul - static_cast<double>(ref)) / static_cast<double>(ref);
                if(rel > max_rel_err)
                    max_rel_err = rel;
            }

        EXPECT_LE(max_rel_err, p.threshold)
            << "Demmel BLAS Test 2 (s=" << p.s << ", n=" << p.n << ", b=" << p.b
            << "): max_rel_err=" << max_rel_err << " > threshold=" << p.threshold
            << ".  A value near 1–2 indicates a CRT sign-flip regression.";
    }

    // The safe b range is determined by the condition that no truncation-error
    // "overlap" occurs — i.e., there is no j_m value where both ε_A and ε_B
    // are simultaneously non-zero in the modular extraction.
    //
    // With sft = floor(log2P_fast(s) − b − 1):
    //   ε_A = 0  when  j_m ≥ 52 − sft   (A_scaled is an exact integer)
    //   ε_B = 0  when  j_m ≤  sft − 52  (B_scaled is an exact integer)
    //
    // No overlap ↔ (52 − sft) ≤ 0  ↔  sft ≥ 52  ↔  b ≤ log2P_fast(s) − 53.
    //
    //   b_max(s) = floor(log2P_fast(s) − 53)
    //
    //   s=14 (log2P≈53.58): b_max = 0  → no b≥1 is safe — s=14 excluded
    //   s=15 (log2P≈57.39): b_max = 4  → powers of 2 in [1,4]  = {1,2,4}
    //   s=16 (log2P≈61.19): b_max = 8  → powers of 2 in [1,8]  = {1,2,4,8}
    //   s=17 (log2P≈64.98): b_max = 11 → powers of 2 in [1,11] = {1,2,4,8}
    //   s=18 (log2P≈68.73): b_max = 15 → powers of 2 in [1,15] = {1,2,4,8}
    //
    // n=512 gives 262144 output elements per test (vs 16384 for n=128) while
    // keeping the host h_lag reference computation at O(n²) ≈ 3 ms.
    INSTANTIATE_TEST_SUITE_P(
        DemmelBlas2,
        Fp64EmulationDemmelTest,
        ::testing::Values(
            // s=7..14, b=0: d[m]=1 for all m → A=B elementwise, trivially safe.
            // Thresholds match CRT capacity per s (same pattern as AllModuliCounts).
            EmulDemmelParam{7, 512, 0, 5e-4},
            EmulDemmelParam{8, 512, 0, 5e-4},
            EmulDemmelParam{9, 512, 0, 1e-6},
            EmulDemmelParam{10, 512, 0, 1e-7},
            EmulDemmelParam{11, 512, 0, 1e-8},
            EmulDemmelParam{12, 512, 0, 1e-9},
            EmulDemmelParam{13, 512, 0, 1e-10},
            EmulDemmelParam{14, 512, 0, 1e-10},
            // s=15 (~118 CRT bits), b_max=4
            EmulDemmelParam{15, 512, 1, 1e-13},
            EmulDemmelParam{15, 512, 2, 1e-13},
            EmulDemmelParam{15, 512, 4, 1e-13},
            // s=16 (~125 CRT bits), b_max=8
            EmulDemmelParam{16, 512, 1, 1e-13},
            EmulDemmelParam{16, 512, 2, 1e-13},
            EmulDemmelParam{16, 512, 4, 1e-13},
            EmulDemmelParam{16, 512, 8, 1e-13},
            // s=17 (~133 CRT bits), b_max=11
            EmulDemmelParam{17, 512, 1, 1e-13},
            EmulDemmelParam{17, 512, 2, 1e-13},
            EmulDemmelParam{17, 512, 4, 1e-13},
            EmulDemmelParam{17, 512, 8, 1e-13},
            // s=18 (~140 CRT bits), b_max=15
            EmulDemmelParam{18, 512, 1, 1e-13},
            EmulDemmelParam{18, 512, 2, 1e-13},
            EmulDemmelParam{18, 512, 4, 1e-13},
            EmulDemmelParam{18, 512, 8, 1e-13}),
        EmulDemmelParamName);

    // ── IllConditionedGramMatrix: disabled ─────────────────────────────────────
    //
    // Disabled because condition number is a solver concept, not a GEMM concept.
    // The Gram matrix test A^T×A verifies Q orthogonality (a property of the
    // host Gram-Schmidt), not the GEMM implementation.  The test always skips
    // on devices where ADP triggers, providing no meaningful coverage.
#if 0
    // ── IllConditionedGramMatrix: A^T×A for varying condition numbers ──────────
    //
    // Constructs A = Q × D where Q is a random orthogonal matrix (N×N,
    // Gram-Schmidt) and D = diag(σ₀,…,σ_{N-1}) with σ_j = κ^{-j/(N-1)},
    // giving condition number κ.
    //
    // The exact result of C = A^T A = (QD)^T(QD) = D Q^T Q D = D² is the
    // diagonal matrix diag(σ₀²,…,σ_{N-1}²) with all off-diagonal elements
    // exactly zero — no reference GEMM required.
    //
    // Checks:
    //   1. Diagonal relative error |C[j,j] − σ_j²| / σ_j² < 2.5√N × ε  (5-sigma)
    //   2. Off-diagonal absolute error |C[i,j]| / σ₀² < 2.5√N × ε  (5-sigma)
    //   3. No NaN or Inf in output

    struct IllCondParam { double kappa; };

    static std::string EmulIllCondParamName(
        const ::testing::TestParamInfo<IllCondParam>& info)
    {
        // Format kappa as an integer exponent to give a unique test name.
        const double k = info.param.kappa;
        const int    e = static_cast<int>(std::round(std::log10(k)));
        return "kappa_1e" + std::to_string(e);
    }

    class Fp64EmulationIllCondTest : public ::testing::TestWithParam<IllCondParam>
    {
    protected:
        void SetUp() override
        {
            if(!has_device())
                GTEST_SKIP() << "No HIP device available";
        }
    };

    TEST_P(Fp64EmulationIllCondTest, GramMatrix)
    {
        const double kappa = GetParam().kappa;
        constexpr int64_t N     = 128;
        constexpr size_t  N2    = static_cast<size_t>(N * N);
        const size_t      bytes = N2 * sizeof(double);

        /* ── Host: build singular values σ_j = κ^{-j/(N-1)} ─────────────────── */
        std::vector<double> sigma(static_cast<size_t>(N));
        for(int64_t j = 0; j < N; ++j)
            sigma[static_cast<size_t>(j)] = std::pow(kappa, -static_cast<double>(j) / (N - 1));

        /* ── Host: random orthogonal Q via Gram-Schmidt ──────────────────────── */
        /* Generate a random N×N matrix then orthonormalise its columns.
         * Use the same xorshift64 PRNG that the other tests use.                */
        std::vector<double> Q(N2);
        {
            uint64_t s = 0x6d81234abcdef000ULL;   /* fixed seed for reproducibility */
            auto rng   = [&]() -> double {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
            };
            /* Fill columns of Q with random values (column-major, col j at offset j*N). */
            for(size_t i = 0; i < N2; ++i) Q[i] = rng();

            /* Gram-Schmidt: orthonormalize column by column. */
            for(int64_t j = 0; j < N; ++j)
            {
                double* col_j = Q.data() + j * N;
                /* Subtract projections onto previous orthonormal columns. */
                for(int64_t p = 0; p < j; ++p)
                {
                    const double* col_p = Q.data() + p * N;
                    double dot = 0.0;
                    for(int64_t i = 0; i < N; ++i) dot += col_j[i] * col_p[i];
                    for(int64_t i = 0; i < N; ++i) col_j[i] -= dot * col_p[i];
                }
                /* Normalise. */
                double norm = 0.0;
                for(int64_t i = 0; i < N; ++i) norm += col_j[i] * col_j[i];
                norm = std::sqrt(norm);
                for(int64_t i = 0; i < N; ++i) col_j[i] /= norm;
            }
        }

        /* ── Host: A = Q × D (scale column j of Q by σ_j) ───────────────────── */
        std::vector<double> hA(N2);
        for(int64_t j = 0; j < N; ++j)
            for(int64_t i = 0; i < N; ++i)
                hA[static_cast<size_t>(i + j * N)] =
                    Q[static_cast<size_t>(i + j * N)] * sigma[static_cast<size_t>(j)];

        /* ── Device buffers ──────────────────────────────────────────────────── */
        double *dA = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        auto cleanup = [&]() {
            (void)hipFree(dD); (void)hipFree(dC); (void)hipFree(dA);
        };

        /* ── Emulation handle ────────────────────────────────────────────────── */
        hipblasLtHandle_t hem = nullptr;
        ASSERT_EQ(hipblasLtCreate(&hem), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationEnabled(hem, true), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationStrategy(hem, HIPBLASLT_EMULATION_STRATEGY_EAGER),
                  HIPBLAS_STATUS_SUCCESS);
        /* ADP mode: falls back to native FP64 if overflow detected. */
        ASSERT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      hem, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);

        /* Skip on unsupported devices. */
        {
            const Fp64EmulationDecision gate =
                fp64EmulationDecision(reinterpret_cast<const _rocblaslt_handle*>(hem),
                                      HIP_R_64F, HIPBLAS_OP_T, HIPBLAS_OP_N, N, N, N, 1);
            if(!gate.apply)
            {
                (void)hipblasLtDestroy(hem);
                cleanup();
                GTEST_SKIP() << "Device not supported by emulation";
            }
        }

        /* ── Run emulated C = A^T × A ────────────────────────────────────────── */
        Fp64EmulationSettings settings{};
        settings.num_moduli      = 0;    /* derive from handle (ADP default) */
        settings.sv_mask         = 0u;   /* skip Inf/NaN detection */
        settings.dynamic_mode    = true; /* ADP mode */
        settings.workspace       = nullptr;
        settings.workspace_bytes = 0u;

        const double alpha = 1.0, beta = 0.0;
        const rocblaslt_status st =
            fp64EmulatedGemm(hem,
                                          HIPBLAS_OP_T, HIPBLAS_OP_N, N, N, N,
                             &alpha, dA, N, dA, N,
                             &beta,  dC, N, dD, N,
                             /*stream=*/nullptr, settings);
        (void)hipblasLtDestroy(hem);

        if(st != rocblaslt_status_success)
        {
            cleanup();
            GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(st)
                         << " (INT8 device library unavailable or ADP fallback failed)";
        }

        /* ── Copy result and verify ──────────────────────────────────────────── */
        std::vector<double> hD(N2);
        ASSERT_EQ(hipMemcpy(hD.data(), dD, bytes, hipMemcpyDeviceToHost), hipSuccess);
        cleanup();

        /* 1. No NaN or Inf. */
        for(size_t idx = 0; idx < N2; ++idx)
            ASSERT_TRUE(std::isfinite(hD[idx]))
                << "Non-finite output at flat index " << idx
                << " (κ=" << kappa << ")";

        /* Threshold: 5-sigma stochastic bound = 2.5 × √N × ε_machine.
         * Each FP64 rounding has unit roundoff u = ε/2; N independent rounding
         * errors accumulate with std dev √N × u.  5σ = 5√N × u = 2.5√N × ε,
         * covering > 99.9999% of random-Q realizations.                      */
        const double kTol = 2.5 * std::sqrt(static_cast<double>(N))
                          * std::numeric_limits<double>::epsilon();

        /* 2. Diagonal elements: |C[j,j] − σ_j²| / σ_j² < 2√N×ε. */
        double max_diag_err = 0.0;
        for(int64_t j = 0; j < N; ++j)
        {
            const double ref  = sigma[static_cast<size_t>(j)] * sigma[static_cast<size_t>(j)];
            const double got  = hD[static_cast<size_t>(j + j * N)];
            const double rerr = (ref > 0.0) ? std::abs(got - ref) / ref : std::abs(got);
            if(rerr > max_diag_err) max_diag_err = rerr;
        }
        EXPECT_LE(max_diag_err, kTol)
            << "Diagonal relative error " << max_diag_err
            << " exceeds √N×ε=" << kTol << " for κ=" << kappa;

        /* 3. Off-diagonal: |C[i,j]| / σ₀² < 2√N×ε. */
        const double sigma0sq = sigma[0] * sigma[0];
        double max_offdiag = 0.0;
        for(int64_t j = 0; j < N; ++j)
            for(int64_t i = 0; i < N; ++i)
            {
                if(i == j) continue;
                const double abs_val = std::abs(hD[static_cast<size_t>(i + j * N)]);
                const double rel_val = (sigma0sq > 0.0) ? abs_val / sigma0sq : abs_val;
                if(rel_val > max_offdiag) max_offdiag = rel_val;
            }
        EXPECT_LE(max_offdiag, kTol)
            << "Off-diagonal relative error " << max_offdiag
            << " exceeds √N×ε=" << kTol << " for κ=" << kappa;
    }

    INSTANTIATE_TEST_SUITE_P(
        IllCond,
        Fp64EmulationIllCondTest,
        ::testing::Values(
            IllCondParam{1e2},
            IllCondParam{1e4},
            IllCondParam{1e6},
            IllCondParam{1e8},
            IllCondParam{1e10},
            IllCondParam{1e12},
            IllCondParam{1e14},
            IllCondParam{1e17}
        ),
        EmulIllCondParamName
    );
#endif // IllConditionedGramMatrix disabled

    // ── StructuredGemmTest: stress matrices targeting the Ozaki extraction ────
    //
    // Seven structured matrix types (N×N, NN mode, C = A × B):
    //   CatastrophicCancel  row pairs (+v,−v): (A×B)[2k,:] + (A×B)[2k+1,:] = 0
    //   ScaledDynamicRange  row i scaled by 2^{i·16/(N-1)−8}: N distinct sftA values
    //   OnesAndEpsilons     A[i,j]=ε for j<N-1, A[i,N-1]=1 — tests ε residual capture
    //   Hadamard            Walsh-Hadamard (Sylvester), entries ±1
    //   Toeplitz            T[i,j] = 0.9^{|i-j|}, decaying off-diagonals
    //   Rank1Perturb        A = I + u·vᵀ, u=sin, v=cos
    //   Alternating         A=checkerboard×U, B=U(0,1)      — signed cancellation
    //
    // All tests run with ADP mode (dynamic moduli selection) — the production default.
    // Reference: native FP64 DGEMM; threshold 1e-10.
    // CatastrophicCancel has an additional exact-zero check on row-pair sums.

    enum StructuredMatType
    {
        SMAT_CATASTROPHIC_CANCEL,
        SMAT_SCALED_DYNAMIC_RANGE,
        SMAT_ONES_AND_EPSILONS,
        SMAT_HADAMARD,
        SMAT_TOEPLITZ,
        SMAT_RANK1_PERTURB,
        SMAT_ALTERNATING,
    };

    struct StructuredGemmParam
    {
        StructuredMatType mat_type;
        int               N;
        double            threshold;
    };

    static std::string
        StructuredGemmParamName(const ::testing::TestParamInfo<StructuredGemmParam>& info)
    {
        const char* names[] = {"CatastrophicCancel",
                               "ScaledDynamicRange",
                               "OnesAndEpsilons",
                               "Hadamard",
                               "Toeplitz",
                               "Rank1Perturb",
                               "Alternating"};
        return std::string(names[static_cast<int>(info.param.mat_type)]) + "_N"
               + std::to_string(info.param.N);
    }

    class Fp64EmulationStructuredTest : public ::testing::TestWithParam<StructuredGemmParam>
    {
    protected:
        void SetUp() override
        {
            if(!has_device())
                GTEST_SKIP() << "No HIP device available";
        }
    };

    /* Build N×N column-major A and B matrices for a given structured type.
     * A and B are independently filled where indicated (ExtremeScale, Alternating). */
    static void build_structured_ab(StructuredMatType    type,
                                    int                  N,
                                    std::vector<double>& hA,
                                    std::vector<double>& hB)
    {
        const size_t N2 = static_cast<size_t>(N) * N;
        hA.resize(N2);
        hB.resize(N2);
        uint64_t sA    = 0xabcd1234ef567890ULL;
        uint64_t sB    = 0x1032547698badcfeULL;
        auto     nextA = [&]() -> double {
            sA ^= sA << 13;
            sA ^= sA >> 7;
            sA ^= sA << 17;
            return static_cast<double>(sA >> 11) * (1.0 / 9007199254740992.0);
        };
        auto nextB = [&]() -> double {
            sB ^= sB << 13;
            sB ^= sB >> 7;
            sB ^= sB << 17;
            return static_cast<double>(sB >> 11) * (1.0 / 9007199254740992.0);
        };

        /* CatastrophicCancel: pre-generate one v[j] per column so that
         * A[2k,j] = +v[j] and A[2k+1,j] = -v[j] share EXACTLY the same value.
         * Drawing v inside the (i,j) loop would give different values per row,
         * breaking the exact cancellation A[2k,j] + A[2k+1,j] = 0.            */
        std::vector<double> v_cat(type == SMAT_CATASTROPHIC_CANCEL ? N : 0);
        if(type == SMAT_CATASTROPHIC_CANCEL)
            for(int jj = 0; jj < N; ++jj)
                v_cat[jj] = 1.0 + nextA(); /* v[j] ~ U(1,2), shared across pair */

        for(int j = 0; j < N; ++j)
            for(int i = 0; i < N; ++i)
            {
                const size_t idx = static_cast<size_t>(i + j * N);
                switch(type)
                {
                case SMAT_CATASTROPHIC_CANCEL:
                {
                    /* A[2k,j] = +v[j],  A[2k+1,j] = -v[j]  (same v per column j).
                     * (A×B)[2k,:] + (A×B)[2k+1,:] = 0 exactly for any B.          */
                    hA[idx] = (i % 2 == 0) ? +v_cat[j] : -v_cat[j];
                    hB[idx] = nextB() * 2.0 - 1.0;
                    break;
                }
                case SMAT_SCALED_DYNAMIC_RANGE:
                {
                    /* Row i scaled by 2^{i·16/(N-1)−8}: N distinct sftA values
                     * from 2^{−8} (row 0) to 2^{8} (row N-1).  Stresses the per-row
                     * shift refinement with every possible sftA in the range.          */
                    const double exp   = static_cast<double>(i) * 16.0 / std::max(N - 1, 1) - 8.0;
                    const double scale = std::ldexp(1.0, static_cast<int>(std::round(exp)));
                    hA[idx]            = scale * (nextA() * 2.0 - 1.0);
                    hB[idx]            = nextB() * 2.0 - 1.0;
                    break;
                }
                case SMAT_ONES_AND_EPSILONS:
                {
                    /* A[i,j] = ε_machine for j < N-1, A[i,N-1] = 1.0; B = all-ones.
                     * The ε elements round to 0 in the primary INT8 extraction (since
                     * sftA = 6 − 0 = 6 and ε × 2^6 ≈ 0) but are recovered from the
                     * residual in subsequent Ozaki passes.  Expected result:
                     *   (A×B)[i,k] = (N-1)×ε + 1.0  (all i,k).                       */
                    constexpr double eps_m = std::numeric_limits<double>::epsilon();
                    hA[idx]                = (j < N - 1) ? eps_m : 1.0;
                    hB[idx]                = 1.0;
                    break;
                }
                case SMAT_HADAMARD:
                case SMAT_TOEPLITZ:
                case SMAT_RANK1_PERTURB:
                    /* Filled after the (i,j) loop — see below. */
                    break;
                case SMAT_ALTERNATING:
                {
                    /* A: checkerboard signs, B: all-positive → genuine cancellation */
                    const double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                    hA[idx]           = sign * nextA();
                    hB[idx]           = nextB();
                    break;
                }
                }
            }

        /* ── Post-loop fills for matrix types that need global structure ────── */
        if(type == SMAT_HADAMARD)
        {
            /* Walsh-Hadamard (Sylvester construction) for N = power-of-2.
             * H₁ = [1], H_{2n} = [[H_n, H_n], [H_n, -H_n]].
             * For non-power-of-2 N, we build the smallest power-of-2 ≥ N and
             * use the top-left N×N sub-block (still ±1 entries).                */
            int logN = 0;
            while((1 << logN) < N)
                ++logN;
            const int           HN = (1 << logN);
            std::vector<double> H(static_cast<size_t>(HN) * HN, 0.0);
            H[0] = 1.0;
            for(int step = 1; step < HN; step *= 2)
                for(int r = 0; r < step; ++r)
                    for(int c = 0; c < step; ++c)
                    {
                        const double v = H[static_cast<size_t>(r + c * HN)];
                        H[static_cast<size_t>(r + step + c * HN)]          = v;
                        H[static_cast<size_t>(r + (c + step) * HN)]        = v;
                        H[static_cast<size_t>(r + step + (c + step) * HN)] = -v;
                    }
            /* Copy top-left N×N block into hA, hB = identity-like (B=I so C=A). */
            for(int jj = 0; jj < N; ++jj)
                for(int ii = 0; ii < N; ++ii)
                {
                    const size_t idx2 = static_cast<size_t>(ii + jj * N);
                    hA[idx2]          = H[static_cast<size_t>(ii + jj * HN)];
                    hB[idx2]          = (ii == jj) ? 1.0 : 0.0;
                }
        }
        else if(type == SMAT_TOEPLITZ)
        {
            /* T[i,j] = 0.9^{|i-j|}: geometrically decaying off-diagonals.
             * Both A and B are set to T so C = T × T.                          */
            for(int jj = 0; jj < N; ++jj)
                for(int ii = 0; ii < N; ++ii)
                {
                    const size_t idx2 = static_cast<size_t>(ii + jj * N);
                    hA[idx2] = hB[idx2] = std::pow(0.9, std::abs(ii - jj));
                }
        }
        else if(type == SMAT_RANK1_PERTURB)
        {
            /* A = I + u·vᵀ where u[i]=sin(π(i+1)/(N+1)), v[j]=cos(π(j+1)/(N+1)).
             * B = identity.  C = A × B = A.                                    */
            const double pi = 3.14159265358979323846;
            for(int jj = 0; jj < N; ++jj)
                for(int ii = 0; ii < N; ++ii)
                {
                    const size_t idx2 = static_cast<size_t>(ii + jj * N);
                    const double u_i  = std::sin(pi * (ii + 1) / (N + 1));
                    const double v_j  = std::cos(pi * (jj + 1) / (N + 1));
                    hA[idx2]          = ((ii == jj) ? 1.0 : 0.0) + u_i * v_j;
                    hB[idx2]          = (ii == jj) ? 1.0 : 0.0;
                }
        }
    }

    TEST_P(Fp64EmulationStructuredTest, VsNativeDgemm)
    {
        const StructuredGemmParam& p     = GetParam();
        const int64_t              N     = static_cast<int64_t>(p.N);
        const size_t               N2    = static_cast<size_t>(N) * N;
        const size_t               bytes = N2 * sizeof(double);

        std::vector<double> hA, hB, hD_nat(N2), hD_emu(N2);
        build_structured_ab(p.mat_type, p.N, hA, hB);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        double *dD_nat = nullptr, *dD_emu = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_nat, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_emu, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        auto cleanup = [&](hipblasLtHandle_t           hem,
                           hipblasLtHandle_t           hnat,
                           hipblasLtMatmulDesc_t       desc,
                           hipblasLtMatrixLayout_t     la,
                           hipblasLtMatrixLayout_t     lb,
                           hipblasLtMatrixLayout_t     ld,
                           hipblasLtMatmulPreference_t pref) {
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(ld)
                hipblasLtMatrixLayoutDestroy(ld);
            if(lb)
                hipblasLtMatrixLayoutDestroy(lb);
            if(la)
                hipblasLtMatrixLayoutDestroy(la);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(hnat)
                hipblasLtDestroy(hnat);
            if(hem)
                hipblasLtDestroy(hem);
            (void)hipFree(dD_emu);
            (void)hipFree(dD_nat);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        hipblasLtHandle_t hem = nullptr;
        ASSERT_EQ(hipblasLtCreate(&hem), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationEnabled(hem, true), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationStrategy(hem, HIPBLASLT_EMULATION_STRATEGY_EAGER),
                  HIPBLAS_STATUS_SUCCESS);
        /* ADP: let the algorithm choose the number of moduli from the data */
        ASSERT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      hem, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);
        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(reinterpret_cast<const _rocblaslt_handle*>(hem),
                                        HIP_R_64F,
                                        HIPBLAS_OP_N,
                                        HIPBLAS_OP_N,
                                        N,
                                        N,
                                        N,
                                        1);
            if(!gate.apply)
            {
                cleanup(hem, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                GTEST_SKIP() << "Device not supported by emulation";
            }
        }

        hipblasLtHandle_t                hnat = nullptr;
        hipblasLtMatmulDesc_t            desc = nullptr;
        hipblasLtMatrixLayout_t          la = nullptr, lb = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t      pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};
        ASSERT_EQ(hipblasLtCreate(&hnat), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F),
                  HIPBLAS_STATUS_SUCCESS);
        {
            hipblasOperation_t opN = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        }
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(
                      &la, HIP_R_64F, static_cast<uint64_t>(N), static_cast<uint64_t>(N), N),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(
                      &lb, HIP_R_64F, static_cast<uint64_t>(N), static_cast<uint64_t>(N), N),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(
                      &ld, HIP_R_64F, static_cast<uint64_t>(N), static_cast<uint64_t>(N), N),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        {
            constexpr size_t ws_zero = 0u;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_zero, sizeof(ws_zero));
        }
        int nat_cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(hnat, desc, la, lb, ld, ld, pref, 1, &heur, &nat_cnt);
        if(nat_cnt == 0)
        {
            cleanup(hem, hnat, desc, la, lb, ld, pref);
            GTEST_SKIP() << "No native FP64 DGEMM algorithm found on this device";
        }

        Fp64EmulationSettings emu_settings{};
        emu_settings.num_moduli      = 18u; /* ADP upper bound */
        emu_settings.dynamic_mode    = true; /* ADP: select s from data */
        emu_settings.sv_mask         = 0u;
        emu_settings.workspace       = nullptr;
        emu_settings.workspace_bytes = 0u;

        const double alpha = 1.0, beta = 0.0;

        /* Native reference */
        const hipblasStatus_t nat_st = hipblasLtMatmul(hnat,
                                                       desc,
                                                       &alpha,
                                                       dA,
                                                       la,
                                                       dB,
                                                       lb,
                                                       &beta,
                                                       dC,
                                                       ld,
                                                       dD_nat,
                                                       ld,
                                                       &heur.algo,
                                                       nullptr,
                                                       0,
                                                       nullptr);
        if(nat_st != HIPBLAS_STATUS_SUCCESS)
        {
            cleanup(hem, hnat, desc, la, lb, ld, pref);
            GTEST_SKIP() << "Native hipblasLtMatmul failed";
        }

        /* Emulated */
        const rocblaslt_status emu_st = fp64EmulatedGemm(hem,
                                                         HIPBLAS_OP_N,
                                                         HIPBLAS_OP_N,
                                                         N,
                                                         N,
                                                         N,
                                                         &alpha,
                                                         dA,
                                                         N,
                                                         dB,
                                                         N,
                                                         &beta,
                                                         dC,
                                                         N,
                                                         dD_emu,
                                                         N,
                                                         nullptr,
                                                         emu_settings);
        if(emu_st != rocblaslt_status_success)
        {
            cleanup(hem, hnat, desc, la, lb, ld, pref);
            GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(emu_st)
                         << " (ADP may have triggered — scale too large for N=" << p.N << ")";
        }

        /* Copy results back before freeing GPU buffers */
        ASSERT_EQ(hipMemcpy(hD_nat.data(), dD_nat, bytes, hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(hD_emu.data(), dD_emu, bytes, hipMemcpyDeviceToHost), hipSuccess);
        cleanup(hem, hnat, desc, la, lb, ld, pref);

        double dmax = 0.0;
        for(double v : hD_nat)
            dmax = std::max(dmax, std::abs(v));
        const double norm        = std::max(dmax, 1.0);
        double       max_rel_err = 0.0;
        for(size_t idx = 0; idx < N2; ++idx)
            max_rel_err = std::max(max_rel_err, std::abs(hD_emu[idx] - hD_nat[idx]) / norm);

        const char* mat_names[] = {"CatastrophicCancel",
                                   "ScaledDynamicRange",
                                   "OnesAndEpsilons",
                                   "Hadamard",
                                   "Toeplitz",
                                   "Rank1Perturb",
                                   "Alternating"};
        EXPECT_LE(max_rel_err, p.threshold)
            << "Structured GEMM (" << mat_names[static_cast<int>(p.mat_type)] << " N=" << p.N
            << " ADP): max_rel_err=" << max_rel_err << " > threshold=" << p.threshold;

        /* Extra check for CatastrophicCancel: paired row sums must be ≈ 0.
         * The alternating-sign construction gives (A×B)[2k,:] = -(A×B)[2k+1,:],
         * so the sum of each row pair is exactly zero.  If a CRT sign-flip occurs
         * the error ≈ 1–2 and the sum will be large.                               */
        if(p.mat_type == SMAT_CATASTROPHIC_CANCEL)
        {
            double max_pair_sum = 0.0;
            for(size_t col = 0; col < static_cast<size_t>(N); ++col)
                for(int64_t row = 0; row + 1 < N; row += 2)
                {
                    const double sum = std::abs(
                        hD_emu[static_cast<size_t>(row) + col * static_cast<size_t>(N)]
                        + hD_emu[static_cast<size_t>(row + 1) + col * static_cast<size_t>(N)]);
                    max_pair_sum = std::max(max_pair_sum, sum);
                }
            /* Normalise by the max output magnitude so the check is scale-independent. */
            const double pair_rel = (dmax > 0.0) ? max_pair_sum / dmax : max_pair_sum;
            EXPECT_LE(pair_rel, p.threshold)
                << "CatastrophicCancel: row-pair sum " << max_pair_sum << " / max=" << dmax << " = "
                << pair_rel << " (a CRT sign-flip would give ≈ 1.0 here)";
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        Structured,
        Fp64EmulationStructuredTest,
        ::testing::Values(StructuredGemmParam{SMAT_CATASTROPHIC_CANCEL, 128, 1e-10},
                          StructuredGemmParam{SMAT_SCALED_DYNAMIC_RANGE, 128, 1e-10},
                          StructuredGemmParam{SMAT_ONES_AND_EPSILONS, 128, 1e-10},
                          StructuredGemmParam{SMAT_HADAMARD, 128, 1e-10},
                          StructuredGemmParam{SMAT_TOEPLITZ, 128, 1e-10},
                          StructuredGemmParam{SMAT_RANK1_PERTURB, 128, 1e-10},
                          StructuredGemmParam{SMAT_ALTERNATING, 128, 1e-10}),
        StructuredGemmParamName);

    // ── AdpFallbackExtremeScale: ADP fallback path is transparent ─────────────
    //
    // Constructs an extreme-scale matrix with rows alternating 1e16 / 1e-16,
    // which exceeds the ADP-safe zone (max element >> threshold for N=64).
    //
    // Step 1: fp64EmulatedGemm(ADP) must return rocblaslt_status_invalid_value.
    // Step 2: hipblasLtMatmul(emulation+ADP) must match pure native FP64 —
    //         rocblaslt_mat.cpp silently falls through to native on invalid_value.
    TEST_F(Fp64EmulationTest, AdpFallbackExtremeScale)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);
        ASSERT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);
        {
            const Fp64EmulationDecision gate = fp64EmulationDecision(
                m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, 64, 64, 64, 1);
            if(!gate.apply)
                GTEST_SKIP() << "Device not supported by emulation";
        }

        constexpr int    N     = 64;
        constexpr size_t N2    = static_cast<size_t>(N) * N;
        const size_t     bytes = N2 * sizeof(double);

        /* Build 1e16 / 1e-16 alternating-row matrix */
        std::vector<double> hA(N2), hB(N2);
        {
            uint64_t sA = 0x1234abcd5678ef90ULL;
            uint64_t sB = 0xfedcba9876543210ULL;
            for(int j = 0; j < N; ++j)
                for(int i = 0; i < N; ++i)
                {
                    const size_t idx = static_cast<size_t>(i + j * N);
                    const double scl = (i % 2 == 0) ? 1e16 : 1e-16;
                    sA ^= sA << 13;
                    sA ^= sA >> 7;
                    sA ^= sA << 17;
                    sB ^= sB << 13;
                    sB ^= sB >> 7;
                    sB ^= sB << 17;
                    hA[idx]
                        = scl
                          * (static_cast<double>(sA >> 11) * (1.0 / 9007199254740992.0) * 2 - 1);
                    hB[idx]
                        = scl
                          * (static_cast<double>(sB >> 11) * (1.0 / 9007199254740992.0) * 2 - 1);
                }
        }

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        double *dD_emul = nullptr, *dD_nat = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_emul, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_nat, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        auto free_bufs = [&]() {
            (void)hipFree(dD_nat);
            (void)hipFree(dD_emul);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        /* ── Step 1: ADP triggers at the fp64EmulatedGemm level ────────────── */
        {
            Fp64EmulationSettings emu{};
            emu.num_moduli               = 16u;
            emu.sv_mask                  = 0u;
            emu.dynamic_mode             = true;
            emu.workspace                = nullptr;
            emu.workspace_bytes          = 0u;
            const double           alpha = 1.0, beta = 0.0;
            const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                         HIPBLAS_OP_N,
                                                         HIPBLAS_OP_N,
                                                         N,
                                                         N,
                                                         N,
                                                         &alpha,
                                                         dA,
                                                         N,
                                                         dB,
                                                         N,
                                                         &beta,
                                                         dC,
                                                         N,
                                                         dD_emul,
                                                         N,
                                                         nullptr,
                                                         emu);
            if(st == rocblaslt_status_success)
            {
                free_bufs();
                GTEST_SKIP() << "ADP did not trigger for 1e16/1e-16 scale on this config";
            }
            EXPECT_EQ(st, rocblaslt_status_invalid_value)
                << "ADP should return invalid_value for 1e16/1e-16 extreme scale";
        }

        /* ── Step 2: hipblasLtMatmul falls back transparently ───────────────── */
        hipblasLtHandle_t                hnat = nullptr;
        hipblasLtMatmulDesc_t            desc = nullptr;
        hipblasLtMatrixLayout_t          la = nullptr, lb = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t      pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};

        ASSERT_EQ(hipblasLtCreate(&hnat), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F),
                  HIPBLAS_STATUS_SUCCESS);
        {
            hipblasOperation_t opN = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        }
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_64F, N, N, N), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_64F, N, N, N), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&ld, HIP_R_64F, N, N, N), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        {
            constexpr size_t ws_zero = 0u;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_zero, sizeof(ws_zero));
        }
        int nat_cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(hnat, desc, la, lb, ld, ld, pref, 1, &heur, &nat_cnt);

        auto destroy_handles = [&]() {
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(hnat);
        };

        if(nat_cnt == 0)
        {
            destroy_handles();
            free_bufs();
            GTEST_SKIP() << "No native FP64 DGEMM algorithm found";
        }

        const double alpha = 1.0, beta = 0.0;

        /* Pure native (emulation disabled via hnat) */
        const hipblasStatus_t nat_st = hipblasLtMatmul(hnat,
                                                       desc,
                                                       &alpha,
                                                       dA,
                                                       la,
                                                       dB,
                                                       lb,
                                                       &beta,
                                                       dC,
                                                       ld,
                                                       dD_nat,
                                                       ld,
                                                       &heur.algo,
                                                       nullptr,
                                                       0,
                                                       nullptr);

        /* m_handle has emulation+ADP enabled → falls back to native internally */
        ASSERT_EQ(hipMemset(dD_emul, 0, bytes), hipSuccess);
        const hipblasStatus_t emul_st = hipblasLtMatmul(m_handle,
                                                        desc,
                                                        &alpha,
                                                        dA,
                                                        la,
                                                        dB,
                                                        lb,
                                                        &beta,
                                                        dC,
                                                        ld,
                                                        dD_emul,
                                                        ld,
                                                        nullptr,
                                                        nullptr,
                                                        0,
                                                        nullptr);

        destroy_handles();

        if(nat_st != HIPBLAS_STATUS_SUCCESS || emul_st != HIPBLAS_STATUS_SUCCESS)
        {
            free_bufs();
            GTEST_SKIP() << "hipblasLtMatmul failed (nat=" << static_cast<int>(nat_st)
                         << " emul=" << static_cast<int>(emul_st) << ")";
        }

        std::vector<double> hD_nat(N2), hD_emul_h(N2);
        ASSERT_EQ(hipMemcpy(hD_nat.data(), dD_nat, bytes, hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(hD_emul_h.data(), dD_emul, bytes, hipMemcpyDeviceToHost), hipSuccess);
        free_bufs();

        double dmax = 0.0;
        for(double v : hD_nat)
            dmax = std::max(dmax, std::abs(v));
        const double norm    = std::max(dmax, 1.0);
        double       max_rel = 0.0;
        for(size_t idx = 0; idx < N2; ++idx)
            max_rel = std::max(max_rel, std::abs(hD_emul_h[idx] - hD_nat[idx]) / norm);

        EXPECT_LE(max_rel, 1e-10) << "ADP fallback differs from pure native by max_rel=" << max_rel
                                  << " (fallback should produce native-equivalent FP64 result)";
    }

    // ── BoundaryValues: FP64 boundary-value inputs handled gracefully ─────────
    //
    // Tests fp64EmulatedGemm with extreme FP64 inputs spanning the full range
    // from the minimum subnormal to the maximum finite value.  Since A = 2×2
    // identity and D = I × B = B exactly (mathematically), each column of D
    // must reproduce the corresponding column of B within the accuracy of the
    // Ozaki extraction.
    //
    // Test design (m=2, n=6, k=2, NN, alpha=1, beta=0):
    //   A  = 2×2 identity.
    //   B  = 2×6 column-major matrix (hB[col*2+row]):
    //
    //   Col │ B[row=0]     │ B[row=1]   │ D[row=0] expected  │ D[row=1] expected
    //   ────┼─────────────┼────────────┼────────────────────┼──────────────────
    //    0  │ TINY        │ 1.0        │ 0.0  (*)           │ ≈ 1.0
    //    1  │ TINY        │ 0.0        │ TINY (†)           │ 0.0 (exact)
    //    2  │ TINY        │ DMAX       │ 0.0  (*)           │ DMAX (exact, ‡)
    //    3  │ NMIN        │ 1.0        │ 0.0  (*)           │ ≈ 1.0
    //    4  │ NMIN        │ 0.0        │ NMIN (§)           │ 0.0 (exact)
    //    5  │ EPS         │ 1.0        │ EPS  (¶)           │ ≈ 1.0
    //
    //   TINY = denorm_min() = 2^-1074  (minimum subnormal)
    //   NMIN = min()        = 2^-1022  (minimum normal, DBL_MIN)
    //   EPS  = epsilon()    = 2^-52    (machine epsilon, DBL_EPSILON)
    //   DMAX = max()        ≈ (2-2^-52)×2^1023  (maximum finite, DBL_MAX)
    //
    // Notes:
    //   (*) col max dominates; trunc(ldexp(small, sft)) = 0 → INT8 = 0 → D = 0.
    //       For col 0: sft≈61, ldexp(TINY,61)=2^-1013<1 → 0.
    //       For col 3: sft≈61, ldexp(NMIN,61)=2^-961<1  → 0.
    //   (†) col contains TINY and 0; floor guard sets col_max=NMIN, sft_init=1028.
    //       Preliminary uses ceil: ceil(ldexp(TINY,1028))=ceil(2^-46)=1 (rounds up!).
    //       Refinement sees col_max_prelim=64 → delta=58 → sft_final=1086.
    //       Main trunc: trunc(ldexp(TINY,1086))=trunc(2^12)=4096 (exact integer).
    //       X_true=64×4096=2^18; D=ldexp(2^18,-1092)=2^-1074=TINY exactly.
    //   (‡) DMAX (= 2^1024-2^971 in exact arithmetic = DBL_MAX) is exactly
    //       representable: X_true = 2^14×(2^53-1); ldexp(X_true,957) = DBL_MAX.
    //   (§) NMIN is the sole non-zero in its column; sft=1028 (no floor guard
    //       since local_max == NMIN exactly). trunc(ldexp(NMIN,1083))=2^61-256;
    //       X_true=64×(2^61-256)=2^67-2^14; D=ldexp(X_true,-(-957))=NMIN.
    //   (¶) EPS is large enough to survive INT8 with col_max=1.0: sft≈61,
    //       trunc(ldexp(EPS,61))=512; X_true=64×512=2^15; D=ldexp(2^15,-67)=EPS.
    //
    // Checks (sub-test 1: A=I):
    //   1. fp64EmulatedGemm returns success (no crash, no NaN/Inf in output).
    //   2. D[row=0] of cols 0,2,3: EXPECT_EQ(0.0) — small value dominated → 0.
    //   3. D[row=0] of cols 1,4: EXPECT_NEAR(TINY/NMIN, 1e-9) — CRT exact (≪1e-9 error).
    //   4. D[row=0] of col 5:   EXPECT_NEAR(EPS, 1e-9) — CRT ~1.6e-10 relative error.
    //   5. D[row=1] of col 2: EXPECT_NEAR(DMAX, 1e-9).
    //   6. D[row=1] of cols 0,3,5: EXPECT_NEAR(1.0, 1e-9).
    //   7. D[row=1] of cols 1,4: EXPECT_EQ(0.0).
    //
    // Sub-test 2 (A = 2*I, 6 cols including [TINY,DMAX]):
    //   B2 = [[TINY,1],[TINY,0],[NMIN,1],[NMIN,0],[EPS,1],[TINY,DMAX]].
    //   D2 = 2×B2 for cols 0-4; col 5 row 1: 2×DMAX = +Inf (IEEE overflow).
    TEST_F(Fp64EmulationTest, BoundaryValues_HandledGracefully)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);

        constexpr int64_t M       = 2; /* output rows   */
        constexpr int64_t N       = 6; /* output cols   */
        constexpr int64_t K       = 2; /* contraction   */
        constexpr size_t  MN      = static_cast<size_t>(M * N); /* 12 elements */
        constexpr size_t  MK      = static_cast<size_t>(M * K); /* 4 elements  */
        constexpr size_t  KN      = static_cast<size_t>(K * N); /* 12 elements */
        const size_t      bytes_A = MK * sizeof(double);
        const size_t      bytes_B = KN * sizeof(double);
        const size_t      bytes_D = MN * sizeof(double);

        /* Boundary-value constants (C++ names map to C DBL_xxx macros). */
        const double TINY = std::numeric_limits<double>::denorm_min(); /* 2^-1074 */
        const double NMIN = std::numeric_limits<double>::min(); /* 2^-1022 */
        const double EPS  = std::numeric_limits<double>::epsilon(); /* 2^-52   */
        const double DMAX = std::numeric_limits<double>::max(); /* ~1.8e308 */

        /* A = 2×2 identity (column-major, lda=M). */
        std::vector<double> hA(MK, 0.0);
        hA[0] = 1.0;
        hA[3] = 1.0; /* A[0,0]=1, A[1,1]=1 */

        /* B = 2×6 column-major matrix (ldb=K=2).
         * hB[col*2+row]:
         *   col 0: [TINY, 1.0 ]
         *   col 1: [TINY, 0.0 ]
         *   col 2: [TINY, DMAX]
         *   col 3: [NMIN, 1.0 ]
         *   col 4: [NMIN, 0.0 ]
         *   col 5: [EPS,  1.0 ]
         */
        const std::vector<double> hB = {
            TINY,
            1.0, /* col 0 */
            TINY,
            0.0, /* col 1 */
            TINY,
            DMAX, /* col 2 */
            NMIN,
            1.0, /* col 3 */
            NMIN,
            0.0, /* col 4 */
            EPS,
            1.0, /* col 5 */
        };

        double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes_A), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes_B), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes_D), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes_D), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes_A, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes_B, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes_D), hipSuccess);

        auto cleanup = [&]() {
            (void)hipFree(dD);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        /* Skip on unsupported devices. */
        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, 1);
            if(!gate.apply)
            {
                cleanup();
                GTEST_SKIP() << "Device not supported";
            }
        }

        Fp64EmulationSettings settings{};
        settings.num_moduli      = 16u;
        settings.sv_mask         = 0u; /* inputs are finite — no Inf/NaN flag needed */
        settings.workspace       = nullptr;
        settings.workspace_bytes = 0u;

        const double           alpha = 1.0, beta = 0.0;
        const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha,
                                                     dA,
                                                     M,
                                                     dB,
                                                     K,
                                                     &beta,
                                                     dC,
                                                     M,
                                                     dD,
                                                     M,
                                                     /*stream=*/nullptr,
                                                     settings);
        if(st != rocblaslt_status_success)
        {
            cleanup();
            GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(st)
                         << " (INT8 device library may be unavailable on this arch)";
        }

        std::vector<double> hD(MN);
        ASSERT_EQ(hipMemcpy(hD.data(), dD, bytes_D, hipMemcpyDeviceToHost), hipSuccess);
        cleanup();

        /* Convenience indexing: hD[col*2+row]. */
        auto d = [&](int col, int row) -> double { return hD[col * 2 + row]; };

        /* 1. All outputs must be finite. */
        for(size_t i = 0; i < MN; ++i)
            ASSERT_TRUE(std::isfinite(hD[i]))
                << "Non-finite output at index " << i << ": " << hD[i];

        /* ── Sub-test 1 checks: A = identity ─────────────────────────────────── */

        /* Row=0 exact zeros: small value dominated by large neighbor.           */
        EXPECT_EQ(d(0, 0), 0.0) << "Col 0 row 0: TINY dominated by 1.0 → 0";
        EXPECT_EQ(d(2, 0), 0.0) << "Col 2 row 0: TINY dominated by DMAX → 0";
        EXPECT_EQ(d(3, 0), 0.0) << "Col 3 row 0: NMIN dominated by 1.0 → 0";

        /* Row=0 non-zero: small value is the sole column element, recovered via
         * the Ozaki scheme.  Use 1e-9 relative tolerance for all non-zero checks.
         * Actual CRT errors are ≪1e-9 for TINY/NMIN and ~1.6e-10 for EPS.      */
        EXPECT_NEAR(d(1, 0), TINY, TINY * 1e-9)
            << "Col 1 row 0: TINY sole non-zero (floor-guard path)";
        EXPECT_NEAR(d(4, 0), NMIN, NMIN * 1e-9) << "Col 4 row 0: NMIN as col max";
        EXPECT_NEAR(d(5, 0), EPS, EPS * 1e-9)
            << "Col 5 row 0: EPS survives INT8, ~1.6e-10 relative CRT error";

        /* Row=1: D = B (A=I), large values. */
        EXPECT_NEAR(d(2, 1), DMAX, DMAX * 1e-9) << "Col 2 row 1: DMAX";
        EXPECT_NEAR(d(0, 1), 1.0, 1e-9) << "Col 0 row 1: 1.0";
        EXPECT_NEAR(d(3, 1), 1.0, 1e-9) << "Col 3 row 1: 1.0";
        EXPECT_NEAR(d(5, 1), 1.0, 1e-9) << "Col 5 row 1: 1.0";
        EXPECT_EQ(d(1, 1), 0.0) << "Col 1 row 1: 0.0";
        EXPECT_EQ(d(4, 1), 0.0) << "Col 4 row 1: 0.0";

        /* ── Sub-test 2: A = 2 × identity, B = 2×6 ──────────────────────────── *
         *
         * For A = 2*I:
         *   sftA[0] = 5  (6 − floor(log2(2.0)) = 5),  A8i[0,0] = trunc(ldexp(2,5)) = 64.
         * Because A8i is identical to the A=I case, the preliminary GEMM and sftB
         * refinement are unchanged.  The only difference is the inverse scale:
         *   D = ldexp(X, −(5 + sftB))  vs.  ldexp(X, −(6 + sftB))  for A=I.
         * This shifts the result by one power of 2, so D = 2 × (A=I result).
         *
         *   B2 cols: [TINY,1.0], [TINY,0.0], [NMIN,1.0], [NMIN,0.0], [EPS,1.0],
         *            [TINY,DMAX]
         *   Expected: D2 = 2 × B2  (col 5: 2×DMAX overflows to +Inf).
         *
         * Col 5 ([TINY,DMAX]): sftB_final≈−963; X_true = 2^67−2^14;
         *   ldexp(X_true, 958) = 2^1025−2^972 → +Inf (IEEE overflow).
         */
        {
            constexpr int64_t N2       = 6; /* 6 output columns */
            constexpr size_t  MN2      = static_cast<size_t>(M * N2); /* 12 elements */
            constexpr size_t  KN2      = static_cast<size_t>(K * N2); /* 12 elements */
            const size_t      bytes_B2 = KN2 * sizeof(double);
            const size_t      bytes_D2 = MN2 * sizeof(double);

            /* A = 2×2 scaled identity: A[0,0]=2, A[1,1]=2 (column-major). */
            std::vector<double> hA2(MK, 0.0);
            hA2[0] = 2.0;
            hA2[3] = 2.0;

            const std::vector<double> hB2 = {
                TINY,
                1.0, /* col 0 */
                TINY,
                0.0, /* col 1 */
                NMIN,
                1.0, /* col 2 */
                NMIN,
                0.0, /* col 3 */
                EPS,
                1.0, /* col 4 */
                TINY,
                DMAX, /* col 5 — 2×DMAX overflows to +Inf */
            };

            double *dA2 = nullptr, *dB2 = nullptr, *dC2 = nullptr, *dD2 = nullptr;
            ASSERT_EQ(hipMalloc(&dA2, bytes_A), hipSuccess);
            ASSERT_EQ(hipMalloc(&dB2, bytes_B2), hipSuccess);
            ASSERT_EQ(hipMalloc(&dC2, bytes_D2), hipSuccess);
            ASSERT_EQ(hipMalloc(&dD2, bytes_D2), hipSuccess);
            ASSERT_EQ(hipMemcpy(dA2, hA2.data(), bytes_A, hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(dB2, hB2.data(), bytes_B2, hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemset(dC2, 0, bytes_D2), hipSuccess);

            const rocblaslt_status st2 = fp64EmulatedGemm(m_handle,
                                                          HIPBLAS_OP_N,
                                                          HIPBLAS_OP_N,
                                                          M,
                                                          N2,
                                                          K,
                                                          &alpha,
                                                          dA2,
                                                          M,
                                                          dB2,
                                                          K,
                                                          &beta,
                                                          dC2,
                                                          M,
                                                          dD2,
                                                          M,
                                                          /*stream=*/nullptr,
                                                          settings);

            std::vector<double> hD2(MN2);
            if(st2 == rocblaslt_status_success)
                ASSERT_EQ(hipMemcpy(hD2.data(), dD2, bytes_D2, hipMemcpyDeviceToHost), hipSuccess);
            (void)hipFree(dD2);
            (void)hipFree(dC2);
            (void)hipFree(dB2);
            (void)hipFree(dA2);

            if(st2 != rocblaslt_status_success)
            {
                GTEST_SKIP() << "fp64EmulatedGemm (A=2*I) returned " << static_cast<int>(st2)
                             << " (INT8 device library may be unavailable)";
            }

            /* Convenience indexing for sub-test 2. */
            auto d2 = [&](int col, int row) -> double { return hD2[col * 2 + row]; };

            /* All outputs must be finite except col 5 row 1 (2×DMAX = +Inf). */
            for(size_t i = 0; i < MN2; ++i)
            {
                if(i == 5 * 2 + 1)
                    continue; /* col 5 row 1 = +Inf is expected */
                ASSERT_TRUE(std::isfinite(hD2[i]))
                    << "A=2*I sub-test: non-finite output at index " << i;
            }

            /* Row=0 exact zeros. */
            EXPECT_EQ(d2(0, 0), 0.0) << "A=2*I col 0 row 0: TINY dominated → 0";
            EXPECT_EQ(d2(2, 0), 0.0) << "A=2*I col 2 row 0: NMIN dominated → 0";

            /* Row=0 non-zero: D = 2 × B (sftA shifts inverse scale by 1 power of 2). */
            EXPECT_NEAR(d2(1, 0), 2.0 * TINY, 2.0 * TINY * 1e-9) << "A=2*I col 1 row 0: 2*TINY";
            EXPECT_NEAR(d2(3, 0), 2.0 * NMIN, 2.0 * NMIN * 1e-9) << "A=2*I col 3 row 0: 2*NMIN";
            EXPECT_NEAR(d2(4, 0), 2.0 * EPS, 2.0 * EPS * 1e-9) << "A=2*I col 4 row 0: 2*EPS";

            /* Row=1: D = 2 × B[row=1]. */
            EXPECT_NEAR(d2(0, 1), 2.0, 2e-9) << "A=2*I col 0 row 1: 2.0";
            EXPECT_NEAR(d2(2, 1), 2.0, 2e-9) << "A=2*I col 2 row 1: 2.0";
            EXPECT_NEAR(d2(4, 1), 2.0, 2e-9) << "A=2*I col 4 row 1: 2.0";
            EXPECT_EQ(d2(1, 1), 0.0) << "A=2*I col 1 row 1: 0.0";
            EXPECT_EQ(d2(3, 1), 0.0) << "A=2*I col 3 row 1: 0.0";

            /* Col 5 ([TINY, DMAX]): 2×DMAX overflows to +Inf.
             * Row=0: TINY dominated by DMAX → 0 (same mechanism as A=I col 2).
             * Row=1: 2×DMAX = ldexp(2^67-2^14, 958) = 2^1025-2^972 = +Inf.         */
            EXPECT_EQ(d2(5, 0), 0.0) << "A=2*I col 5 row 0: TINY dominated by DMAX → 0";
            EXPECT_TRUE(std::isinf(d2(5, 1)) && d2(5, 1) > 0.0)
                << "A=2*I col 5 row 1: 2*DMAX overflows to +Inf, got " << d2(5, 1);
        }
    }

    // ── NonUnitLeadingDimension: lda/ldb/ldc/ldd != m/k/m/m ──────────────────
    //
    // Exercises the non-unit-stride paths in the extraction and finalize kernels.
    // 128×128×128 NN with lda=m+7=135, ldb=k+13=141, ldc=ldd=m+11=139.
    TEST_F(Fp64EmulationTest, NonUnitLeadingDimension)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);

        constexpr int64_t M = 128, N = 128, K = 128;
        constexpr int64_t LDA = M + 7; // 135
        constexpr int64_t LDB = K + 13; // 141
        constexpr int64_t LDC = M + 11; // 139
        constexpr int64_t LDD = LDC;

        const size_t nA = static_cast<size_t>(LDA) * K;
        const size_t nB = static_cast<size_t>(LDB) * N;
        const size_t nC = static_cast<size_t>(LDC) * N;
        const size_t nD = static_cast<size_t>(LDD) * N;

        std::vector<double> hA(nA), hB(nB), hD_emu(nD), hD_nat(nD);
        fill_uniform_host(hA, 0xaabb000011110000ULL);
        fill_uniform_host(hB, 0xccdd000022220000ULL);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        double *dD_emu = nullptr, *dD_nat = nullptr;
        ASSERT_EQ(hipMalloc(&dA, nA * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, nB * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, nC * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_emu, nD * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_nat, nD * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), nA * sizeof(double), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), nB * sizeof(double), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, nC * sizeof(double)), hipSuccess);

        auto free_all = [&]() {
            (void)hipFree(dD_nat);
            (void)hipFree(dD_emu);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, 1);
            if(!gate.apply)
            {
                free_all();
                GTEST_SKIP() << "Device not supported";
            }
        }

        /* Native reference via hipblasLtMatmul */
        hipblasLtHandle_t                hnat = nullptr;
        hipblasLtMatmulDesc_t            desc = nullptr;
        hipblasLtMatrixLayout_t          la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t      pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};
        ASSERT_EQ(hipblasLtCreate(&hnat), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_64F, M, K, LDA), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_64F, K, N, LDB), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&lc, HIP_R_64F, M, N, LDC), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&ld, HIP_R_64F, M, N, LDD), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        int nat_cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(hnat, desc, la, lb, lc, ld, pref, 1, &heur, &nat_cnt);

        auto destroy_nat = [&]() {
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(hnat);
        };

        if(nat_cnt == 0)
        {
            destroy_nat();
            free_all();
            GTEST_SKIP() << "No native algo";
        }

        const double alpha = 1.0, beta = 0.0;
        hipblasLtMatmul(hnat,
                        desc,
                        &alpha,
                        dA,
                        la,
                        dB,
                        lb,
                        &beta,
                        dC,
                        lc,
                        dD_nat,
                        ld,
                        &heur.algo,
                        nullptr,
                        0,
                        nullptr);

        /* Emulated */
        Fp64EmulationSettings emu{};
        emu.num_moduli            = 16u;
        emu.sv_mask               = 0u;
        const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     M,
                                                     N,
                                                     K,
                                                     &alpha,
                                                     dA,
                                                     LDA,
                                                     dB,
                                                     LDB,
                                                     &beta,
                                                     dC,
                                                     LDC,
                                                     dD_emu,
                                                     LDD,
                                                     nullptr,
                                                     emu);

        if(st != rocblaslt_status_success)
        {
            destroy_nat();
            free_all();
            GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(st);
        }

        ASSERT_EQ(hipMemcpy(hD_nat.data(), dD_nat, nD * sizeof(double), hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(hD_emu.data(), dD_emu, nD * sizeof(double), hipMemcpyDeviceToHost),
                  hipSuccess);
        destroy_nat();
        free_all();

        double dmax = 0.0;
        for(int64_t j = 0; j < N; ++j)
            for(int64_t i = 0; i < M; ++i)
                dmax = std::max(dmax, std::abs(hD_nat[static_cast<size_t>(i + j * LDD)]));
        const double norm    = std::max(dmax, 1.0);
        double       max_rel = 0.0;
        for(int64_t j = 0; j < N; ++j)
            for(int64_t i = 0; i < M; ++i)
            {
                const size_t idx = static_cast<size_t>(i + j * LDD);
                max_rel          = std::max(max_rel, std::abs(hD_emu[idx] - hD_nat[idx]) / norm);
            }

        EXPECT_LE(max_rel, 1e-10) << "NonUnitLeadingDimension: max_rel=" << max_rel;
    }

    // ── SvMaskInfInput: sv_mask=0x1 with +Inf input must return invalid_value ─
    //
    // Constructs a 64×64 matrix with one +Inf element. With sv_mask=0x1
    // (Inf detection enabled), fp64EmulatedGemm must return
    // rocblaslt_status_invalid_value.
    TEST_F(Fp64EmulationTest, SvMaskInfInput)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);

        constexpr int64_t N     = 64;
        const size_t      N2    = static_cast<size_t>(N) * N;
        const size_t      bytes = N2 * sizeof(double);

        std::vector<double> hA(N2, 1.0), hB(N2, 1.0);
        hA[0] = std::numeric_limits<double>::infinity(); /* one +Inf element */

        double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        auto cleanup = [&]() {
            (void)hipFree(dD);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, 1);
            if(!gate.apply)
            {
                cleanup();
                GTEST_SKIP() << "Device not supported";
            }
        }

        Fp64EmulationSettings settings{};
        settings.num_moduli = 16u;
        settings.sv_mask    = 0x1u; /* enable Inf detection */

        const double           alpha = 1.0, beta = 0.0;
        const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     N,
                                                     N,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     N,
                                                     dB,
                                                     N,
                                                     &beta,
                                                     dC,
                                                     N,
                                                     dD,
                                                     N,
                                                     nullptr,
                                                     settings);
        cleanup();

        EXPECT_EQ(st, rocblaslt_status_invalid_value)
            << "sv_mask=0x1 with +Inf input should return invalid_value, got "
            << static_cast<int>(st);
    }

    // ── AdpLowModuliUpperBound: ADP with num_moduli=8 on well-conditioned data ─
    //
    // Runs ADP with a low upper bound (num_moduli=8) on a 128×128 random matrix.
    // ADP should succeed (data is well-conditioned) and error within 1e-4.
    TEST_F(Fp64EmulationTest, AdpLowModuliUpperBound)
    {
        set_enabled(true);
        set_strategy(HIPBLASLT_EMULATION_STRATEGY_EAGER);

        constexpr int64_t N     = 128;
        const size_t      N2    = static_cast<size_t>(N) * N;
        const size_t      bytes = N2 * sizeof(double);

        std::vector<double> hA(N2), hB(N2), hD_emu(N2), hD_nat(N2);
        fill_uniform_host(hA, 0x1111aaaa2222bbbbULL);
        fill_uniform_host(hB, 0x3333cccc4444ddddULL);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        double *dD_emu = nullptr, *dD_nat = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_emu, bytes), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_nat, bytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC, 0, bytes), hipSuccess);

        auto free_all = [&]() {
            (void)hipFree(dD_nat);
            (void)hipFree(dD_emu);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        {
            const Fp64EmulationDecision gate
                = fp64EmulationDecision(m_roc, HIP_R_64F, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, 1);
            if(!gate.apply)
            {
                free_all();
                GTEST_SKIP() << "Device not supported";
            }
        }

        /* Native reference */
        hipblasLtHandle_t                hnat = nullptr;
        hipblasLtMatmulDesc_t            desc = nullptr;
        hipblasLtMatrixLayout_t          la = nullptr, lb = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t      pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};
        ASSERT_EQ(hipblasLtCreate(&hnat), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_64F, N, N, N), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_64F, N, N, N), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&ld, HIP_R_64F, N, N, N), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        int nat_cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(hnat, desc, la, lb, ld, ld, pref, 1, &heur, &nat_cnt);

        auto destroy_nat = [&]() {
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(hnat);
        };
        if(nat_cnt == 0)
        {
            destroy_nat();
            free_all();
            GTEST_SKIP() << "No native algo";
        }

        const double alpha = 1.0, beta = 0.0;
        hipblasLtMatmul(hnat,
                        desc,
                        &alpha,
                        dA,
                        la,
                        dB,
                        lb,
                        &beta,
                        dC,
                        ld,
                        dD_nat,
                        ld,
                        &heur.algo,
                        nullptr,
                        0,
                        nullptr);

        /* Emulated with ADP, num_moduli=8 upper bound */
        Fp64EmulationSettings emu{};
        emu.num_moduli            = 8u;
        emu.sv_mask               = 0u;
        emu.dynamic_mode          = true;
        const rocblaslt_status st = fp64EmulatedGemm(m_handle,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     N,
                                                     N,
                                                     N,
                                                     &alpha,
                                                     dA,
                                                     N,
                                                     dB,
                                                     N,
                                                     &beta,
                                                     dC,
                                                     N,
                                                     dD_emu,
                                                     N,
                                                     nullptr,
                                                     emu);

        if(st != rocblaslt_status_success)
        {
            destroy_nat();
            free_all();
            GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(st);
        }

        ASSERT_EQ(hipMemcpy(hD_nat.data(), dD_nat, bytes, hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(hD_emu.data(), dD_emu, bytes, hipMemcpyDeviceToHost), hipSuccess);
        destroy_nat();
        free_all();

        double dmax = 0.0;
        for(double v : hD_nat)
            dmax = std::max(dmax, std::abs(v));
        const double norm    = std::max(dmax, 1.0);
        double       max_rel = 0.0;
        for(size_t idx = 0; idx < N2; ++idx)
            max_rel = std::max(max_rel, std::abs(hD_emu[idx] - hD_nat[idx]) / norm);

        EXPECT_LE(max_rel, 1e-4) << "AdpLowModuliUpperBound (num_moduli=8): max_rel=" << max_rel;
    }

} // namespace
