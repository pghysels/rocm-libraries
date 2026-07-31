// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * fused_prof.cpp
 *
 * Minimal single-shape profiling harness for the fused MFMA+CRT kernel.
 * Runs 2 warmup + 1 profiled launch for the large_sq shape.
 * The kernel configuration is selected via OZ2_FUSED_SHAPE_OVERRIDE.
 *
 * Usage:
 *   OZ2_FUSED_SHAPE_OVERRIDE="4 4 1 1 16" \
 *   HIPBLASLT_EMULATE_DOUBLE_PRECISION=1 \
 *   HIPBLASLT_EMULATION_FUSED=force \
 *   rocprof --stats -i prof_counters.txt ./build/fused_prof
 */

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(expr) \
    do { hipError_t _e=(expr); if(_e!=hipSuccess){ \
        std::fprintf(stderr,"HIP %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); \
        std::exit(1); } } while(0)
#define HLT_CHECK(expr) \
    do { hipblasStatus_t _s=(expr); if(_s!=HIPBLAS_STATUS_SUCCESS){ \
        std::fprintf(stderr,"HLT %s:%d: %d\n",__FILE__,__LINE__,(int)_s); \
        std::exit(1); } } while(0)

int main(int argc, char** argv)
{
    int warmup = 2;
    int64_t M = 32768, N = 32768, K = 32768;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--warmup") && i+1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--m") && i+1 < argc) M = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--n") && i+1 < argc) N = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i+1 < argc) K = std::atoll(argv[++i]);
    }

    /* Mandatory env vars */
    setenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION", "1", 1);
    setenv("HIPBLASLT_EMULATION_FUSED",          "force", 0); /* 0 = don't override if already set */

    const char* cfg = std::getenv("OZ2_FUSED_SHAPE_OVERRIDE");
    std::fprintf(stderr, "OZ2_FUSED_SHAPE_OVERRIDE=%s\n", cfg ? cfg : "(not set, using launcher default)");

    std::fprintf(stderr, "Shape: M=%ld N=%ld K=%ld  warmup=%d  profiled_iters=1\n",
                 (long)M, (long)N, (long)K, warmup);

    /* Handle (eager, max S) */
    hipblasLtHandle_t h;
    HLT_CHECK(hipblasLtCreate(&h));
    HLT_CHECK(hipblasLtSetEmulationEnabled(h, true));
    HLT_CHECK(hipblasLtSetEmulationStrategy(h, HIPBLASLT_EMULATION_STRATEGY_EAGER));
    /* Fix s=16 so OZ2_FUSED_SHAPE_OVERRIDE dispatches the correct kernel
     * and the num_moduli==16 guard in oz2_launch_fused_TN does not abort. */
    HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
        h, HIPBLASLT_EMULATION_MANTISSA_CONTROL_FIXED));
    HLT_CHECK(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(h, 118)); /* s=16 */

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* Allocate buffers: TN layout — A is K×M, B is K×N, C/D is M×N */
    const size_t szA = (size_t)K * M, szB = (size_t)K * N, szCD = (size_t)M * N;
    double *A_d, *B_d, *C_d, *D_d;
    HIP_CHECK(hipMalloc(&A_d,  szA  * sizeof(double)));
    HIP_CHECK(hipMalloc(&B_d,  szB  * sizeof(double)));
    HIP_CHECK(hipMalloc(&C_d,  szCD * sizeof(double)));
    HIP_CHECK(hipMalloc(&D_d,  szCD * sizeof(double)));

    /* Fill A and B with sin/cos values so the scaling kernel sees non-trivial
     * magnitudes and selects a realistic number of moduli S.
     * hipMemset(., 0) produces all-zero doubles → S=2 (minimum), which bypasses
     * the OZ2_FUSED_SHAPE_OVERRIDE condition (num_moduli == 16).              */
    std::fprintf(stderr, "Filling A (%zu MB) and B (%zu MB) with sin/cos...\n",
                 szA * 8 / (1 << 20), szB * 8 / (1 << 20));
    {
        std::vector<double> h_A(szA), h_B(szB);
        for (size_t i = 0; i < szA; ++i) h_A[i] = std::sin(static_cast<double>(i + 1));
        for (size_t i = 0; i < szB; ++i) h_B[i] = std::cos(static_cast<double>(i + 1));
        HIP_CHECK(hipMemcpy(A_d, h_A.data(), szA * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(B_d, h_B.data(), szB * sizeof(double), hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemset(C_d, 0, szCD * sizeof(double)));

    const size_t WS = 40ull * 1024 * 1024 * 1024;  /* 40 GiB — covers S=16 INT8 workspace for K=32768 */
    void* ws_d;
    HIP_CHECK(hipMalloc(&ws_d, WS));

    /* Build matmul descriptor */
    hipblasLtMatrixLayout_t lA, lB, lC, lD;
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&lA, HIP_R_64F, K, M, K));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&lB, HIP_R_64F, K, N, K));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&lC, HIP_R_64F, M, N, M));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&lD, HIP_R_64F, M, N, M));

    hipblasLtMatmulDesc_t mm;
    HLT_CHECK(hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_64F, HIP_R_64F));
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    HLT_CHECK(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    HLT_CHECK(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    hipblasLtMatmulPreference_t pref;
    HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &WS, sizeof(WS)));

    hipblasLtMatmulHeuristicResult_t hres;
    int nalgo = 0;
    HLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(h, mm, lA, lB, lC, lD, pref, 1, &hres, &nalgo));

    const double alpha = 1.0, beta = 0.0;
    auto launch = [&]() {
        HLT_CHECK(hipblasLtMatmul(h, mm, &alpha, A_d, lA, B_d, lB,
                                  &beta, C_d, lC, D_d, lD,
                                  nalgo ? &hres.algo : nullptr, ws_d, WS, stream));
    };

    /* Warmup */
    for (int i = 0; i < warmup; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));

    /* Single profiled launch — rocprof sees this kernel */
    launch();
    HIP_CHECK(hipStreamSynchronize(stream));

    std::fprintf(stderr, "Done — rocprof should have captured the kernel above.\n");

    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK(hipFree(D_d));
    HIP_CHECK(hipFree(ws_d));
    HIP_CHECK(hipStreamDestroy(stream));
    HLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    HLT_CHECK(hipblasLtMatmulDescDestroy(mm));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(lA));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(lB));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(lC));
    HLT_CHECK(hipblasLtMatrixLayoutDestroy(lD));
    HLT_CHECK(hipblasLtDestroy(h));
    return 0;
}
