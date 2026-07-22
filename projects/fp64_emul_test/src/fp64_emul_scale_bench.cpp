// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * fp64_emul_scale_bench.cpp
 *
 * Benchmark specifically targeting scale-kernel-heavy shapes:
 *   large k, small m×n  →  scale kernel is ~40-70% of total emulation time.
 *
 * Usage:
 *   HIPBLASLT_EMULATION_PROFILE=scale_bench.log \
 *   HIP_VISIBLE_DEVICES=0 ./fp64_emul_scale_bench
 *
 * Writes per-call CSV rows via the library's profiling mechanism.
 * Also prints per-run total times to stdout for a quick sanity check.
 */

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* =========================================================================
 * Error-check macros
 * ========================================================================= */
#define HIP_CHECK(e) \
    do { hipError_t _e=(e); if(_e!=hipSuccess){ \
        fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); \
        exit(EXIT_FAILURE); } } while(0)

#define HLT_CHECK(e) \
    do { hipblasStatus_t _s=(e); if(_s!=HIPBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"HLT error %s:%d: status=%d\n",__FILE__,__LINE__,(int)_s); \
        exit(EXIT_FAILURE); } } while(0)

/* =========================================================================
 * Test shapes: (m, n, k, transA, transB)
 * Selected so the scale kernel is a large fraction of total time.
 * ========================================================================= */
struct Shape {
    int64_t m, n, k;
    bool    transA, transB;
    int     num_runs;
};

static const Shape SHAPES[] = {
    /* Large k, small m×n: scale kernel is ~50-70% of total */
    {  256,  256, 131072, true,  false, 6 },
    {  512,  512,  65536, true,  false, 6 },
    { 1024, 1024,  32768, true,  false, 6 },
    /* Same shapes with N,T (exercises SHMEM scale paths for A_N + B_T) */
    {  256,  256, 131072, false, true,  6 },
    {  512,  512,  65536, false, true,  6 },
    { 1024, 1024,  32768, false, true,  6 },
};
static const int NUM_SHAPES = static_cast<int>(sizeof(SHAPES) / sizeof(SHAPES[0]));

/* =========================================================================
 * Simple GPU random-fill kernel
 * ========================================================================= */
__global__ static void fill_rand_kernel(double* A, size_t n, unsigned seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    unsigned s = seed ^ (unsigned)(idx * 2654435761u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    A[idx] = static_cast<double>(s) / 4294967296.0 - 0.5;
}

static void fill_rand(double* d, size_t n, hipStream_t stream)
{
    const unsigned blk = 256;
    const unsigned grd = (n + blk - 1) / blk;
    hipLaunchKernelGGL(fill_rand_kernel, dim3(grd), dim3(blk), 0, stream,
                       d, n, 42u);
    HIP_CHECK(hipGetLastError());
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char**)
{
    (void)argc;

    /* Device info */
    {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        fprintf(stderr, "Device : %s\n", prop.name);
    }

    const char* profile_file = getenv("HIPBLASLT_EMULATION_PROFILE");
    fprintf(stderr, "Profile: %s\n",
            profile_file ? profile_file : "(none — set HIPBLASLT_EMULATION_PROFILE)");
    fprintf(stderr, "Testing %d scale-heavy shapes\n\n", NUM_SHAPES);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* Create emulation handle (EAGER, s=16, no Inf/NaN check) */
    hipblasLtHandle_t handle;
    HLT_CHECK(hipblasLtCreate(&handle));
    HLT_CHECK(hipblasLtSetEmulationEnabled(handle, 1));
    HLT_CHECK(hipblasLtSetEmulationStrategy(handle, HIPBLASLT_EMULATION_STRATEGY_EAGER));
    HLT_CHECK(hipblasLtSetEmulationSpecialValuesSupport(handle, 0u));
    /* Fix at s=16 moduli */
    HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
        handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_FIXED));
    HLT_CHECK(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(handle, 118)); /* s=16 */

    printf("%-5s %-5s %-8s %-5s %-5s  %s\n",
           "m", "n", "k", "tA", "tB", "t_total_ms (per run)");
    printf("-----------------------------------------------------------\n");

    for(int si = 0; si < NUM_SHAPES; si++) {
        const Shape& sh = SHAPES[si];
        const int64_t m = sh.m, n = sh.n, k = sh.k;
        const bool    tA = sh.transA, tB = sh.transB;
        const int     nr = sh.num_runs;

        /* A is stored as (m×k) col-major if transA=T, or (k×m)? No:
         * transA=T means op(A)=A^T and A is stored k×m with lda=k.
         * transA=N means op(A)=A   and A is stored m×k with lda=m. */
        const int64_t lda = tA ? k : m;  /* leading dim of A in storage */
        const int64_t ldb = tB ? n : k;  /* leading dim of B in storage */
        const int64_t ldc = m;

        const size_t szA = static_cast<size_t>(lda) * (tA ? m : k);
        const size_t szB = static_cast<size_t>(ldb) * (tB ? k : n);
        const size_t szD = static_cast<size_t>(m) * n;

        double *dA = nullptr, *dB = nullptr, *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, szA * sizeof(double)));
        HIP_CHECK(hipMalloc(&dB, szB * sizeof(double)));
        HIP_CHECK(hipMalloc(&dD, szD * sizeof(double)));
        fill_rand(dA, szA, stream);
        fill_rand(dB, szB, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Build matmul descriptor */
        hipblasLtMatmulDesc_t   desc   = nullptr;
        hipblasLtMatrixLayout_t layA   = nullptr, layB = nullptr, layD = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulHeuristicResult_t heur{};

        HLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
        {
            hipblasOperation_t opA = tA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = tB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
        }

        /* A layout: rows = inner (k if tA, m if !tA), cols = outer, ld = lda */
        const uint64_t rowsA = tA ? k : m;
        const uint64_t colsA = tA ? m : k;
        const uint64_t rowsB = tB ? n : k;
        const uint64_t colsB = tB ? k : n;

        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layA, HIP_R_64F, rowsA, colsA, lda));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layB, HIP_R_64F, rowsB, colsB, ldb));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layD, HIP_R_64F,
            static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldc));

        HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        constexpr size_t WS_BUDGET = size_t(-1);
        HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WS_BUDGET, sizeof(WS_BUDGET)));

        int cnt = 0;
        HLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, layA, layB, layD, layD, pref, 1, &heur, &cnt));

        void* dWS = nullptr;
        if(heur.workspaceSize > 0)
            HIP_CHECK(hipMalloc(&dWS, heur.workspaceSize));

        const double alpha = 1.0, beta = 0.0;

        auto run_once = [&]() {
            HLT_CHECK(hipblasLtMatmul(
                handle, desc,
                &alpha, dA, layA, dB, layB,
                &beta,  dD, layD, dD, layD,
                cnt > 0 ? &heur.algo : nullptr,
                dWS, heur.workspaceSize, stream));
        };

        /* Warmup (don't count these) */
        run_once(); run_once();
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Timed runs — each emits one profile row (if env var is set) */
        hipEvent_t ev0, ev1;
        HIP_CHECK(hipEventCreate(&ev0));
        HIP_CHECK(hipEventCreate(&ev1));
        HIP_CHECK(hipEventRecord(ev0, stream));
        for(int r = 0; r < nr; r++)
            run_once();
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipEventSynchronize(ev1));
        float ms_total = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms_total, ev0, ev1));

        printf("%5lld %5lld %8lld %5s %5s  %.3f ms/run\n",
               (long long)m, (long long)n, (long long)k,
               tA ? "T" : "N", tB ? "T" : "N",
               ms_total / nr);
        fflush(stdout);

        HIP_CHECK(hipEventDestroy(ev1));
        HIP_CHECK(hipEventDestroy(ev0));
        if(dWS) HIP_CHECK(hipFree(dWS));
        HLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
        HLT_CHECK(hipblasLtMatrixLayoutDestroy(layD));
        HLT_CHECK(hipblasLtMatrixLayoutDestroy(layB));
        HLT_CHECK(hipblasLtMatrixLayoutDestroy(layA));
        HLT_CHECK(hipblasLtMatmulDescDestroy(desc));
        HIP_CHECK(hipFree(dD));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dA));
    }

    HLT_CHECK(hipblasLtDestroy(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return EXIT_SUCCESS;
}
