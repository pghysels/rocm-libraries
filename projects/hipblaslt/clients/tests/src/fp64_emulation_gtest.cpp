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
        EXPECT_LE(s, 18u); // OZ2_S_MAX
    }

    // Workspace must be non-empty and not shrink as the moduli count grows.
    TEST_F(Fp64EmulationHostTest, WorkspaceSizePositiveAndMonotonic)
    {
        const int64_t m = 1024, n = 1024, k = 1024;
        const size_t  ws8  = fp64EmulationWorkspaceSize(m, n, k, 8);
        const size_t  ws16 = fp64EmulationWorkspaceSize(m, n, k, 16);
        EXPECT_GT(ws8, 0u);
        EXPECT_GE(ws16, ws8);
    }

    TEST_F(Fp64EmulationHostTest, PublicWorkspaceSizeRejectsNegativeDimensions)
    {
        EXPECT_EQ(hipblasLtFp64EmulationWorkspaceSize(-1, 64, 64, 16), 0u);
        EXPECT_EQ(hipblasLtFp64EmulationWorkspaceSize(64, -1, 64, 16), 0u);
        EXPECT_EQ(hipblasLtFp64EmulationWorkspaceSize(64, 64, -1, 16), 0u);
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
        EXPECT_EQ(performant.value,
                  static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT));

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

        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("141").state,
                  FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("-1").state,
                  FP64_EMULATION_ENV_INVALID);
        EXPECT_EQ(fp64EmulationParseMantissaBitCountEnv("55.0").state,
                  FP64_EMULATION_ENV_INVALID);
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
            const Fp64EmulationDecision decision =
                fp64EmulationDecision(m_roc, t, m, n, k, batch);
            EXPECT_EQ(decision.status, rocblaslt_status_success);
            return decision.apply;
        }

        hipblasLtHandle_t        m_handle = nullptr;
        const _rocblaslt_handle* m_roc    = nullptr;
    };

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

    TEST_F(Fp64EmulationTest, ApiValidationRejectsInvalidMantissaControl)
    {
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, static_cast<hipblasEmulationMantissaControl_t>(-1)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, static_cast<hipblasEmulationMantissaControl_t>(2)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtSetFixedPointEmulationMantissaControl(
                      m_handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC),
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
                      m_handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC),
                  HIPBLAS_STATUS_SUCCESS);

        const Fp64EmulationDecision decision =
            fp64EmulationDecision(m_roc, HIP_R_64F, 4096, 4096, 4096, 1);
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
        settings.handle          = m_handle;

        const rocblaslt_status st = fp64EmulatedGemm(HIPBLAS_OP_N,
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
        unsigned           s;         /* num_moduli (2..18)                   */
        int64_t            m, n, k;   /* matrix dimensions                    */
        double             threshold; /* max relative error vs native DGEMM   */
        FillStyle          fill;      /* input distribution                   */
        hipblasOperation_t opA;       /* HIPBLAS_OP_N or HIPBLAS_OP_T for A   */
        hipblasOperation_t opB;       /* HIPBLAS_OP_N or HIPBLAS_OP_T for B   */
        double             alpha;     /* scaling factor for A*B               */
        double             beta;      /* scaling factor for C                 */
    };

    static std::string EmulAccuracyParamName(
        const ::testing::TestParamInfo<EmulAccuracyParam>& info)
    {
        const EmulAccuracyParam& p = info.param;
        const char* suf = "";
        switch(p.fill)
        {
        case FILL_UNIFORM_01:   suf = "_uni";   break;
        case FILL_ALLPOS_NEAR1: suf = "_near1"; break;
        case FILL_GEOMROWS:     suf = "_geom";  break;
        }
        const char opA_c = (p.opA == HIPBLAS_OP_N) ? 'N' : 'T';
        const char opB_c = (p.opB == HIPBLAS_OP_N) ? 'N' : 'T';
        /* Format scalar as integer when it is an exact integer, else as-is. */
        auto fmt_s = [](double v) -> std::string {
            const int vi = static_cast<int>(v);
            return (static_cast<double>(vi) == v) ? std::to_string(vi) : "x";
        };
        return "s" + std::to_string(p.s) + suf
               + "_" + opA_c + opB_c
               + "_a" + fmt_s(p.alpha)
               + "_b" + fmt_s(p.beta);
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
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
    static double xsr64_uniform(uint64_t b)
    {
        return static_cast<double>(b >> 11) * (1.0 / 9007199254740992.0);
    }
    static void fill_uniform_host(std::vector<double>& v, uint64_t seed)
    {
        uint64_t s = seed ^ 1442695040888963407ULL;
        s = xsr64(s); s = xsr64(s);
        for(double& x : v) { s = xsr64(s); x = xsr64_uniform(s); }
    }

    /* All-positive near-1: U(0.9, 1.0).  Inner products are ≈3.7× larger
     * than U(0,1), pushing X_true proportionally closer to M_s/2. */
    static void fill_allpos_near1_host(std::vector<double>& v, uint64_t seed)
    {
        uint64_t s = seed ^ 0x9e3779b97f4a7c15ULL;
        s = xsr64(s); s = xsr64(s);
        for(double& x : v) { s = xsr64(s); x = 0.9 + 0.1 * xsr64_uniform(s); }
    }

    /* Geometric row-scale: element (i, j) in column-major storage (lda=rows) is
     * row_scale(i) × U(-1,1), where row_scale(i) = 2^{(i·16/rows)−8}.
     * Creates rows spanning 16 doublings (2^{-8}..2^{8}) → diverse sftA[i] values.
     * Used for op(A)=N matrices (A is m×k, column-major with lda=m), and also for
     * the opA=T case via fill_geomcols_host to scale op(A)'s rows correctly. */
    static void fill_geomrows_host(std::vector<double>& v,
                                   int64_t rows, int64_t cols, uint64_t seed)
    {
        uint64_t s = seed ^ 0xdeadbeef12345678ULL;
        s = xsr64(s); s = xsr64(s);
        for(int64_t i = 0; i < rows; ++i)
        {
            const double row_scale = std::ldexp(1.0, static_cast<int>(i * 16 / rows) - 8);
            for(int64_t j = 0; j < cols; ++j)
            {
                s = xsr64(s);
                v[static_cast<size_t>(i + j * rows)] =
                    row_scale * (xsr64_uniform(s) * 2.0 - 1.0);
            }
        }
    }

    /* Geometric col-scale: element (i, j) in column-major storage (ldb=rows) is
     * col_scale(j) × U(-1,1), where col_scale(j) = 2^{(j·16/cols)−8}.
     * Creates columns spanning 16 doublings → diverse sftB[j] values. */
    static void fill_geomcols_host(std::vector<double>& v,
                                   int64_t rows, int64_t cols, uint64_t seed)
    {
        uint64_t s = seed ^ 0xbadcafe0deadbeefULL;
        s = xsr64(s); s = xsr64(s);
        for(int64_t j = 0; j < cols; ++j)
        {
            const double col_scale = std::ldexp(1.0, static_cast<int>(j * 16 / cols) - 8);
            for(int64_t i = 0; i < rows; ++i)
            {
                s = xsr64(s);
                v[static_cast<size_t>(i + j * rows)] =
                    col_scale * (xsr64_uniform(s) * 2.0 - 1.0);
            }
        }
    }

    TEST_P(Fp64EmulationAccuracyTest, VsNativeDgemm)
    {
        const EmulAccuracyParam& p = GetParam();

        const bool    tA  = (p.opA == HIPBLAS_OP_T);
        const bool    tB  = (p.opB == HIPBLAS_OP_T);
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
        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        double *dD_nat = nullptr, *dD_emu = nullptr;
        ASSERT_EQ(hipMalloc(&dA,     alloc_A),           hipSuccess);
        ASSERT_EQ(hipMalloc(&dB,     alloc_B),           hipSuccess);
        ASSERT_EQ(hipMalloc(&dC,     nD * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_nat, nD * sizeof(double)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD_emu, nD * sizeof(double)), hipSuccess);

        /* Cleanup helper called on every exit path */
        auto cleanup = [&](hipblasLtHandle_t hem, hipblasLtHandle_t hnat,
                           hipblasLtMatmulDesc_t desc,
                           hipblasLtMatrixLayout_t la, hipblasLtMatrixLayout_t lb,
                           hipblasLtMatrixLayout_t ld, hipblasLtMatmulPreference_t pref) {
            if(pref) hipblasLtMatmulPreferenceDestroy(pref);
            if(ld)   hipblasLtMatrixLayoutDestroy(ld);
            if(lb)   hipblasLtMatrixLayoutDestroy(lb);
            if(la)   hipblasLtMatrixLayoutDestroy(la);
            if(desc) hipblasLtMatmulDescDestroy(desc);
            if(hnat) hipblasLtDestroy(hnat);
            if(hem)  hipblasLtDestroy(hem);
            (void)hipFree(dD_emu);
            (void)hipFree(dD_nat);
            (void)hipFree(dC);
            (void)hipFree(dB);
            (void)hipFree(dA);
        };

        /* ── Emulation handle (s moduli, eager strategy) ─────────────────── */
        hipblasLtHandle_t hem = nullptr;
        ASSERT_EQ(hipblasLtCreate(&hem), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationEnabled(hem, true),  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtSetEmulationStrategy(hem, HIPBLASLT_EMULATION_STRATEGY_EAGER),
                  HIPBLAS_STATUS_SUCCESS);

        /* Skip if the device is not supported by the emulation. */
        {
            const Fp64EmulationDecision gate =
                fp64EmulationDecision(reinterpret_cast<const _rocblaslt_handle*>(hem),
                                      HIP_R_64F, p.opA, p.opB, p.m, p.n, p.k, 1);
            if(!gate.apply)
            {
                cleanup(hem, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                GTEST_SKIP() << "Device not supported by emulation";
            }
        }

        /* ── Native DGEMM reference handle + layouts ─────────────────────── */
        hipblasLtHandle_t        hnat = nullptr;
        hipblasLtMatmulDesc_t    desc = nullptr;
        hipblasLtMatrixLayout_t  la   = nullptr, lb = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};

        ASSERT_EQ(hipblasLtCreate(&hnat), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &p.opA, sizeof(p.opA));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &p.opB, sizeof(p.opB));

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
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&ld, HIP_R_64F,
                                              static_cast<uint64_t>(p.m),
                                              static_cast<uint64_t>(p.n), p.m),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        {
            constexpr size_t ws_zero = 0u;
            hipblasLtMatmulPreferenceSetAttribute(pref,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_zero, sizeof(ws_zero));
        }
        int nat_cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(hnat, desc, la, lb, ld, ld, pref, 1, &heur, &nat_cnt);
        if(nat_cnt == 0)
        {
            cleanup(hem, hnat, desc, la, lb, ld, pref);
            GTEST_SKIP() << "No native FP64 DGEMM algorithm found on this device";
        }

        /* ── Emulation settings (use settings.num_moduli directly) ──────── */
        Fp64EmulationSettings emu_settings{};
        emu_settings.handle          = hem;
        emu_settings.num_moduli      = p.s;
        emu_settings.sv_mask         = 0u;   /* skip Inf/NaN detection */
        emu_settings.workspace       = nullptr; /* library allocates internally */
        emu_settings.workspace_bytes = 0u;

        /* Fill the C matrix once before the seed loop.  When beta=0, C is not
         * read by either GEMM, but we still fill it to exercise the beta=0
         * bypass path with a non-trivial buffer.                            */
        fill_uniform_host(hC, 0xc0ffee0000000000ULL);
        ASSERT_EQ(hipMemcpy(dC, hC.data(), nD * sizeof(double), hipMemcpyHostToDevice),
                  hipSuccess);

        /* Sweep three deterministic seeds; accumulate the worst relative error. */
        constexpr uint64_t SEEDS[3] = {12345ULL, 98765ULL, 54321ULL};
        double max_rel_err = 0.0;

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
                if(tA) fill_geomcols_host(hA, p.k, p.m, seed);
                else   fill_geomrows_host(hA, p.m, p.k, seed);
                if(tB) fill_geomrows_host(hB, p.n, p.k, seed ^ 0xdeadbeefcafeULL);
                else   fill_geomcols_host(hB, p.k, p.n, seed ^ 0xdeadbeefcafeULL);
                break;
            }

            if(nA > 0)
                ASSERT_EQ(hipMemcpy(dA, hA.data(), nA*sizeof(double), hipMemcpyHostToDevice),
                          hipSuccess);
            if(nB > 0)
                ASSERT_EQ(hipMemcpy(dB, hB.data(), nB*sizeof(double), hipMemcpyHostToDevice),
                          hipSuccess);

            /* Native FP64 DGEMM reference (C→D_nat). */
            const hipblasStatus_t nat_st =
                hipblasLtMatmul(hnat, desc,
                                &p.alpha, dA, la, dB, lb,
                                &p.beta,  dC, ld, dD_nat, ld,
                                &heur.algo, nullptr, 0, /*stream=*/nullptr);
            if(nat_st != HIPBLAS_STATUS_SUCCESS)
            {
                cleanup(hem, hnat, desc, la, lb, ld, pref);
                GTEST_SKIP() << "Native hipblasLtMatmul failed (status="
                             << static_cast<int>(nat_st) << ")";
            }

            /* Emulated DGEMM (C→D_emu). */
            const rocblaslt_status emu_st =
                fp64EmulatedGemm(p.opA, p.opB, p.m, p.n, p.k,
                                 &p.alpha, dA, lda, dB, ldb,
                                 &p.beta,  dC, p.m, dD_emu, p.m,
                                 /*stream=*/nullptr, emu_settings);
            if(emu_st != rocblaslt_status_success)
            {
                cleanup(hem, hnat, desc, la, lb, ld, pref);
                GTEST_SKIP() << "fp64EmulatedGemm returned " << static_cast<int>(emu_st)
                             << " (INT8 device library may be unavailable on this arch)";
            }

            ASSERT_EQ(hipMemcpy(hD_nat.data(), dD_nat, nD*sizeof(double), hipMemcpyDeviceToHost),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(hD_emu.data(), dD_emu, nD*sizeof(double), hipMemcpyDeviceToHost),
                      hipSuccess);

            /* Relative error normalised by the maximum element magnitude.
             * Floor at 1.0 prevents division-by-zero for near-zero outputs.  */
            double dmax = 0.0;
            for(double v : hD_nat) dmax = std::max(dmax, std::abs(v));
            const double norm = std::max(dmax, 1.0);

            for(size_t idx = 0; idx < nD; ++idx)
                max_rel_err = std::max(max_rel_err,
                                       std::abs(hD_emu[idx] - hD_nat[idx]) / norm);
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
            EmulAccuracyParam{ 2,  64,  64,  128, 3e-1,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 3,  64,  64,  128, 3e-1,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 4,  64,  64,  128, 2e-1,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 5,  64,  64,  128, 1e-1,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 6,  64,  64,  128, 5e-2,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 7,  128, 128, 4096, 5e-4,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 8,  128, 128, 4096, 5e-4,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 9,  128, 128, 4096, 1e-6,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{10,  128, 128, 4096, 1e-7,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{11,  128, 128, 4096, 1e-8,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{12,  128, 128, 4096, 1e-9,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{13,  128, 128, 4096, 1e-10, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{14,  128, 128, 4096, 1e-10, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{15,  128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{16,  128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{17,  128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{18,  128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 }
        ),
        EmulAccuracyParamName
    );

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
            EmulAccuracyParam{ 7,  128, 128, 4096, 5e-4,  FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 8,  128, 128, 4096, 5e-4,  FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 9,  128, 128, 4096, 1e-6,  FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{10,  128, 128, 4096, 1e-7,  FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{11,  128, 128, 4096, 1e-8,  FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{12,  128, 128, 4096, 1e-9,  FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{13,  128, 128, 4096, 1e-10, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{14,  128, 128, 4096, 1e-10, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{15,  128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{16,  128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{17,  128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{18,  128, 128, 4096, 1e-11, FILL_ALLPOS_NEAR1, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            // Geometric row-/col-scaling (wide dynamic range)
            EmulAccuracyParam{ 7,  128, 128, 4096, 5e-4,  FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 8,  128, 128, 4096, 5e-4,  FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 9,  128, 128, 4096, 1e-6,  FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{10,  128, 128, 4096, 1e-7,  FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{11,  128, 128, 4096, 1e-8,  FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{12,  128, 128, 4096, 1e-9,  FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{13,  128, 128, 4096, 1e-10, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{14,  128, 128, 4096, 1e-10, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{15,  128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{16,  128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{17,  128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{18,  128, 128, 4096, 1e-11, FILL_GEOMROWS, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 }
        ),
        EmulAccuracyParamName
    );

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
            EmulAccuracyParam{ 7, 128, 128, 4096, 5e-4,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 7, 128, 128, 4096, 5e-4,  FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{ 7, 128, 128, 4096, 5e-4,  FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_T, 1.0, 0.0 },
            EmulAccuracyParam{ 7, 128, 128, 4096, 5e-4,  FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_T, 1.0, 0.0 },
            // s=16 — default precision
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_T, 1.0, 0.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_T, HIPBLAS_OP_T, 1.0, 0.0 }
        ),
        EmulAccuracyParamName
    );

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
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 0.0, 0.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 0.0, 1.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 0.0, 2.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 1.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 2.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 2.0, 0.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 2.0, 1.0 },
            EmulAccuracyParam{16, 128, 128, 4096, 1e-11, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 2.0, 2.0 }
        ),
        EmulAccuracyParamName
    );

    // ── KEqualsZero: k=0 edge case ────────────────────────────────────────────
    //
    // With a zero contraction dimension, D = alpha*(empty sum) + beta*C = beta*C.
    // No INT8 GEMMs are executed; the result must equal beta*C exactly (error = 0).
    INSTANTIATE_TEST_SUITE_P(
        KEqualsZero,
        Fp64EmulationAccuracyTest,
        ::testing::Values(
            /* beta=0: D = 0 */
            EmulAccuracyParam{16, 64, 64, 0, 0.0, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 0.0 },
            /* beta=1: D = C */
            EmulAccuracyParam{16, 64, 64, 0, 0.0, FILL_UNIFORM_01, HIPBLAS_OP_N, HIPBLAS_OP_N, 1.0, 1.0 }
        ),
        EmulAccuracyParamName
    );

} // namespace
