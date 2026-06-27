// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * gram_illcond_bench.cpp
 *
 * Computes C = A^T * A  (TN mode)  for the ill-conditioned matrix
 *   A = Q * D,  D = diag(σ₀,…,σ_{N-1}),  σⱼ = κ^{-j/(N-1)}
 *   Q : random orthogonal (Gram-Schmidt from a fixed-seed random matrix)
 *
 * The exact result is C = D² (diagonal, zero off-diagonal), so no
 * reference GEMM is required.
 *
 * For each condition number κ the program prints:
 *   κ   |  emul_diag_relerr  |  emul_offdiag_relerr  |
 *       |  native_diag_relerr | native_offdiag_relerr |
 *
 * Usage:
 *   ./build/gram_illcond_bench [N]   (default N=128)
 */

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ─── Error-check macros ──────────────────────────────────────────────────── */
#define HIP_CHECK(expr)                                                       \
    do {                                                                       \
        hipError_t _e = (expr);                                               \
        if(_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, hipGetErrorString(_e));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

#define HLT_CHECK(expr)                                                        \
    do {                                                                       \
        hipblasStatus_t _s = (expr);                                           \
        if(_s != HIPBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "hipBLASLt error %s:%d: %d\n",              \
                         __FILE__, __LINE__, static_cast<int>(_s));           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

/* ─── Simple DGEMM runner (TN mode) ──────────────────────────────────────── */
/* Computes D = alpha * A^T * B + beta * C  using hipblasLtMatmul.
 * All matrices are N×N column-major with leading dimension N.
 * set_emul=true enables FP64 emulation at the given mantissa bit count.    */
struct GemmRunner {
    hipblasLtHandle_t            handle  = nullptr;
    hipblasLtMatmulDesc_t        desc    = nullptr;
    hipblasLtMatrixLayout_t      la      = nullptr;
    hipblasLtMatrixLayout_t      ld      = nullptr;
    hipblasLtMatmulPreference_t  pref    = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    bool                         hasAlgo = false;

    void init(int64_t N, bool set_emul, int mantissa_bits = 0)
    {
        HLT_CHECK(hipblasLtCreate(&handle));
        if(set_emul) {
            HLT_CHECK(hipblasLtSetEmulationEnabled(handle, true));
            HLT_CHECK(hipblasLtSetEmulationStrategy(
                          handle, HIPBLASLT_EMULATION_STRATEGY_EAGER));
            HLT_CHECK(hipblasLtSetEmulationSpecialValuesSupport(handle, 0u));
            HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
                          handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_FIXED));
            if(mantissa_bits > 0)
                HLT_CHECK(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(
                               handle, mantissa_bits));
        }

        HLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
        {
            hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        }
        /* Both A (stored as k×m = N×N with opA=T) and B (N×N opB=N) have
         * the same physical layout: N rows, N cols, lda=ldb=N.           */
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &la, HIP_R_64F, static_cast<uint64_t>(N), static_cast<uint64_t>(N), N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &ld, HIP_R_64F, static_cast<uint64_t>(N), static_cast<uint64_t>(N), N));
        HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        {
            constexpr size_t ws_zero = 0u;
            hipblasLtMatmulPreferenceSetAttribute(pref,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_zero, sizeof(ws_zero));
        }
        int cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, la, la, ld, ld, pref, 1, &heur, &cnt);
        hasAlgo = (cnt > 0);
    }

    void run(const double* A, double* D, int64_t N, hipStream_t stream) const
    {
        const double alpha = 1.0, beta = 0.0;
        /* TN: C = A^T * A.  Same pointer for both A and B operands.      */
        HLT_CHECK(hipblasLtMatmul(
            handle, desc,
            &alpha, A, la, A, la,
            &beta,  D, ld, D, ld,
            hasAlgo ? &heur.algo : nullptr,
            nullptr, 0, stream));
    }

    void destroy()
    {
        if(pref)   { hipblasLtMatmulPreferenceDestroy(pref);   pref   = nullptr; }
        if(ld)     { hipblasLtMatrixLayoutDestroy(ld);         ld     = nullptr; }
        if(la)     { hipblasLtMatrixLayoutDestroy(la);         la     = nullptr; }
        if(desc)   { hipblasLtMatmulDescDestroy(desc);         desc   = nullptr; }
        if(handle) { hipblasLtDestroy(handle);                 handle = nullptr; }
    }
};

/* ─── Main ───────────────────────────────────────────────────────────────── */
int main(int argc, char** argv)
{
    const int64_t N = (argc > 1) ? static_cast<int64_t>(std::atol(argv[1])) : 128;
    if(N <= 0 || N > 4096) {
        std::fprintf(stderr, "Usage: %s [N]  (default N=128, max 4096)\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* Device info */
    {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        std::fprintf(stderr, "Device : %s   N = %lld\n",
                     prop.name, static_cast<long long>(N));
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const size_t N2    = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = N2 * sizeof(double);

    /* ── Build random orthogonal Q (Gram-Schmidt) ─────────────────────────── */
    std::vector<double> Q(N2);
    {
        uint64_t s = 0x6d81234abcdef000ULL;
        auto rng   = [&]() -> double {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
        };
        for(size_t i = 0; i < N2; ++i) Q[i] = rng();

        /* Gram-Schmidt */
        for(int64_t j = 0; j < N; ++j) {
            double* col_j = Q.data() + j * N;
            for(int64_t p = 0; p < j; ++p) {
                const double* col_p = Q.data() + p * N;
                double dot = 0.0;
                for(int64_t i = 0; i < N; ++i) dot += col_j[i] * col_p[i];
                for(int64_t i = 0; i < N; ++i) col_j[i] -= dot * col_p[i];
            }
            double norm = 0.0;
            for(int64_t i = 0; i < N; ++i) norm += col_j[i] * col_j[i];
            norm = std::sqrt(norm);
            for(int64_t i = 0; i < N; ++i) col_j[i] /= norm;
        }
    }

    /* ── GEMM runners ─────────────────────────────────────────────────────── */
    /* Configs to test: native, s=8, s=12, s=16 */
    struct Config {
        const char* label;
        bool        emul;
        int         bits;   /* target mantissa bits (0 = use library default) */
    };
    const Config configs[] = {
        { "native    ", false,   0 },
        { "s=8 (~64b)", true,   55 },  /* 55 bits → 7 moduli (ceiling to s=8 ≈ 63.6 bits) */
        { "s=12(~95b)", true,   87 },  /* 87 bits → 12 moduli (~94.8 bits)                 */
        { "s=16(125b)", true,  125 },  /* 125 bits → 16 moduli (~125.4 bits, default)      */
    };
    constexpr int N_CONFIGS = static_cast<int>(sizeof(configs) / sizeof(configs[0]));

    GemmRunner runners[N_CONFIGS];
    for(int c = 0; c < N_CONFIGS; ++c)
        runners[c].init(N, configs[c].emul, configs[c].bits);

    /* ── GPU buffers ──────────────────────────────────────────────────────── */
    double *dA = nullptr, *dD = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dD, bytes));

    /* ── Header ───────────────────────────────────────────────────────────── */
    std::fprintf(stdout,
        "\n%-12s  %-12s  %-12s  %-12s  %-12s\n"
        "%-12s  %-12s  %-12s  %-12s  %-12s\n",
        "kappa",
        configs[0].label, configs[1].label, configs[2].label, configs[3].label,
        "",
        "diag/offdiag","diag/offdiag","diag/offdiag","diag/offdiag");
    std::fprintf(stdout, "%s\n", std::string(75, '-').c_str());

    /* ── Condition number sweep ───────────────────────────────────────────── */
    const double kappas[] = { 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e17 };

    std::vector<double> hA(N2), hD(N2);

    for(double kappa : kappas)
    {
        /* Build σ_j = κ^{-j/(N-1)} */
        std::vector<double> sigma(static_cast<size_t>(N));
        for(int64_t j = 0; j < N; ++j)
            sigma[static_cast<size_t>(j)] =
                std::pow(kappa, -static_cast<double>(j) / (N - 1));

        /* A = Q × D (scale column j of Q by σ_j) */
        for(int64_t j = 0; j < N; ++j)
            for(int64_t i = 0; i < N; ++i)
                hA[static_cast<size_t>(i + j * N)] =
                    Q[static_cast<size_t>(i + j * N)] * sigma[static_cast<size_t>(j)];

        HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));

        /* Reference: σ_j² on diagonal, 0 off-diagonal */
        const double sigma0sq = sigma[0] * sigma[0];

        std::fprintf(stdout, "%-12.0e  ", kappa);

        for(int c = 0; c < N_CONFIGS; ++c)
        {
            HIP_CHECK(hipMemset(dD, 0, bytes));
            runners[c].run(dA, dD, N, stream);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipMemcpy(hD.data(), dD, bytes, hipMemcpyDeviceToHost));

            /* Diagonal relative error */
            double max_diag = 0.0;
            for(int64_t j = 0; j < N; ++j) {
                const double ref  = sigma[static_cast<size_t>(j)]
                                  * sigma[static_cast<size_t>(j)];
                const double got  = hD[static_cast<size_t>(j + j * N)];
                const double rerr = (ref > 0.0) ? std::abs(got - ref) / ref
                                                : std::abs(got);
                if(rerr > max_diag) max_diag = rerr;
            }

            /* Off-diagonal normalized error */
            double max_off = 0.0;
            for(int64_t j = 0; j < N; ++j)
                for(int64_t i = 0; i < N; ++i) {
                    if(i == j) continue;
                    const double abs_val = std::abs(hD[static_cast<size_t>(i + j * N)]);
                    const double rel_val = (sigma0sq > 0.0) ? abs_val / sigma0sq
                                                            : abs_val;
                    if(rel_val > max_off) max_off = rel_val;
                }

            std::fprintf(stdout, "%.2e/%.2e  ", max_diag, max_off);
        }
        std::fprintf(stdout, "\n");
        std::fflush(stdout);
    }

    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "Columns: max diagonal relative error / max off-diagonal / σ₀²\n");

    /* ── Cleanup ──────────────────────────────────────────────────────────── */
    HIP_CHECK(hipFree(dD));
    HIP_CHECK(hipFree(dA));
    for(int c = 0; c < N_CONFIGS; ++c) runners[c].destroy();
    HIP_CHECK(hipStreamDestroy(stream));
    return EXIT_SUCCESS;
}
