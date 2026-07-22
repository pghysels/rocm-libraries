// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * papers_fig234_bench.cpp
 *
 * Reproduces Figures 2, 3, and 4 from:
 *   "Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores
 *    Through Extensions of the Ozaki Scheme"
 *   arXiv:2511.13778  (SCA/HPCAsia 2026)
 *
 * ── Figure 2 (BLAS Test 2, Section 6 Aspect A1) ─────────────────────────
 *   Matrix construction (Demmel et al. grading test):
 *     x   ~ U(1,2)^n,   n = 1024
 *     D   = diag(2^{j_0}, …, 2^{j_{n-1}}),   j_i = −b + round(i·2b/(n−1))
 *     A_{k,j} = x_{(j+k) mod n} · d_{(j+k) mod n}   (row k, col j)
 *     B_{i,k} = x_{(i+k) mod n} / d_{(i+k) mod n}   (row i, col k)
 *   ⟹  C_{kk} = xᵀx  (exact, any k);  C_{kl} mixes elements at different scales.
 *
 *   Reference:
 *     Diagonal  : xᵀx computed in long double (FP80) on the host
 *     Off-diag  : native FP64 DGEMM
 *
 *   Error metric:
 *     max_{ij} e_{ij},   e_{ii} = |xᵀx − c_{ii}| / |xᵀx|
 *                        e_{ij} = |c^ref_{ij} − c_{ij}| / |c^ref_{ij}|  (i≠j)
 *
 *   Sweep: b = 0, b_step, 2·b_step, …, b_max  (default 0..80 step 5)
 *   Configs: num_moduli ∈ {6, 8, 10, 12, 14, 16, 18}  (no adaptive)
 *
 * ── Figures 3 & 4 (Grade A criterion, Section 6 Aspect A2) ───────────────
 *   Input: A, B ~ U(0,1)^{N×N},  5 seeds,  N ∈ {128,256,512,1024,2048,4096}
 *   Reference: double-double (DD) GEMM  (~106-bit mantissa, GPU kernel)
 *   Error: |fl(AB)_{ij} − DD(AB)_{ij}| / max(|DD(AB)_{ij}|, ε)
 *     max   → Figure 3
 *     median → Figure 4
 *   Compares: native FP64 DGEMM and emulated DGEMM at each moduli count
 *
 * Usage:
 *   ./papers_fig234_bench [--fig 2|34|all]
 *                         [--n-fig2 1024] [--b-max 80] [--b-step 5]
 *                         [--n-list 128,256,512,1024,2048,4096] [--seeds 5]
 *                         [--out-fig2 fig2_results.csv]
 *                         [--out-fig34 fig34_results.csv]
 *                         [--seed <uint64>]
 */

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ═══════════════════════════════════════════════════════════════════════════
 * Error-check macros
 * ═══════════════════════════════════════════════════════════════════════════ */
#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if(_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

#define HLT_CHECK(expr)                                                        \
    do {                                                                       \
        hipblasStatus_t _s = (expr);                                           \
        if(_s != HIPBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "hipBLASLt error %s:%d: status=%d\n",        \
                         __FILE__, __LINE__, static_cast<int>(_s));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * Double-double primitives — host + device
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef M_PI
static constexpr double M_PI = 3.14159265358979323846;
#endif

/* Disable optimisations that break Dekker / EFT operators on the host side. */
#ifndef __HIP_DEVICE_COMPILE__
# pragma clang optimize off
#endif

__host__ __device__ __forceinline__
void two_sum(double a, double b, double& s, double& e)
{
    s = a + b;
    double v = s - a, u = s - v;
    e = (a - u) + (b - v);
}

__host__ __device__ __forceinline__
void fast_two_sum(double a, double b, double& s, double& e)
{ s = a + b; e = (a - s) + b; }

__host__ __device__ __forceinline__
void two_sub(double a, double b, double& s, double& e)
{
    s = a - b;
    double v = s - a, u = s - v;
    e = (a - u) - (b + v);
}

__host__ __device__ __forceinline__
void two_prod(double a, double b, double& p, double& e)
{
    p = a * b;
#ifdef __HIP_DEVICE_COMPILE__
    e = __fma_rn(a, b, -p);
#else
    e = std::fma(a, b, -p);
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
# pragma clang optimize on
#endif

__host__ __device__ __forceinline__
double2 dd_add(double2 a, double2 b)
{
    double2 c;
    two_sum(a.x, b.x, c.x, c.y);
    c.y += a.y + b.y;
    fast_two_sum(c.x, c.y, c.x, c.y);
    return c;
}

__host__ __device__ __forceinline__
double2 dd_sub(double a, double2 b)
{
    double2 c;
    two_sub(a, b.x, c.x, c.y);
    c.y -= b.y;
    fast_two_sum(c.x, c.y, c.x, c.y);
    return c;
}

__host__ __device__ __forceinline__
double2 dd_mul(double a, double b)
{ double2 c; two_prod(a, b, c.x, c.y); return c; }

__host__ __device__ __forceinline__
double2 dd_div(double2 a, double2 b)
{
    double q1 = a.x / b.x;
    double2 q1b; two_prod(q1, b.x, q1b.x, q1b.y); q1b.y += q1 * b.y;
    double rhi, re; two_sub(a.x, q1b.x, rhi, re);
    double rlo = re + (a.y - q1b.y);
    double q2  = rhi / b.x + rlo / b.x;
    double2 r; fast_two_sum(q1, q2, r.x, r.y);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RNG helpers (device + host mirrors)
 * ═══════════════════════════════════════════════════════════════════════════ */
__device__ __forceinline__ uint64_t xorshift64_dev(uint64_t s)
{ s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

__device__ __forceinline__ double bits_to_uniform_dev(uint64_t b)
{ return static_cast<double>(b >> 11) * (1.0 / 9007199254740992.0); }

static inline uint64_t xorshift64_host(uint64_t s)
{ s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static inline double bits_to_uniform_host(uint64_t b)
{ return static_cast<double>(b >> 11) * (1.0 / 9007199254740992.0); }

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU kernels
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Fill x[0..n-1] ~ U(1,2)  (Figure 2 — generates the x vector) */
__global__ static void
randvec_12_kernel(int n, double* __restrict__ x, uint64_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    uint64_t s = seed ^ (static_cast<uint64_t>(idx) * 0x9e3779b97f4a7c15ULL
                         + 1442695040888963407ULL);
    s = xorshift64_dev(s); s = xorshift64_dev(s);
    x[idx] = 1.0 + bits_to_uniform_dev(xorshift64_dev(s));  /* U(1,2) */
}

/* Fill A[0..n_elems-1] ~ U(0,1)  (Figures 3/4) */
__global__ static void
randmat_uniform01_kernel(size_t n_elems, double* __restrict__ A, uint64_t seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n_elems) return;
    uint64_t s = seed ^ (idx * 0x9e3779b97f4a7c15ULL + 1442695040888963407ULL);
    s = xorshift64_dev(s); s = xorshift64_dev(s);
    A[idx] = bits_to_uniform_dev(xorshift64_dev(s));   /* U(0,1) */
}

/*
 * Fill Figure 2's A matrix (n×n, column-major):
 *   A[row=k][col=j]  =  x[(j+k) % n]  *  d[(j+k) % n]
 *   stored at: A[ j*n + k ]
 */
__global__ static void
fill_test2_A_kernel(int n, double* __restrict__ A,
                    const double* __restrict__ x,
                    const double* __restrict__ d)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* j */
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* k */
    if(col >= n || row >= n) return;
    int m = (row + col) % n;
    A[col * n + row] = x[m] * d[m];
}

/*
 * Fill Figure 2's B matrix (n×n, column-major):
 *   B[row=i][col=k]  =  x[(i+k) % n]  /  d[(i+k) % n]
 *   stored at: B[ k*n + i ]
 */
__global__ static void
fill_test2_B_kernel(int n, double* __restrict__ B,
                    const double* __restrict__ x,
                    const double* __restrict__ d)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* k */
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* i */
    if(col >= n || row >= n) return;
    int m = (row + col) % n;
    B[col * n + row] = x[m] / d[m];
}

/*
 * Figure 2 per-element error kernel.
 *
 * Stores |err_{ij}| in err_out[ col*n + row ]:
 *   i == j :  |xTx  −  C_emul[i,i]| / |xTx|        (long-double exact reference)
 *   i != j :  |C_ref[i,j] − C_emul[i,j]| / |C_ref[i,j]|   (native FP64 ref)
 */
__global__ static void
fig2_error_kernel(int n,
                  const double* __restrict__ C_emul,
                  const double* __restrict__ C_ref,
                  double xTx,
                  double* __restrict__ err_out)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col >= n || row >= n) return;

    int pos         = col * n + row;
    double emul_val = C_emul[pos];
    double ref_val  = (row == col) ? xTx : C_ref[pos];

    double relerr = (fabs(ref_val) < 1e-300)
                  ? fabs(emul_val)
                  : fabs(emul_val - ref_val) / fabs(ref_val);
    err_out[pos] = relerr;
}

/*
 * Extract the main diagonal of an n×n column-major matrix C into diag[]:
 *   diag[k] = C[k*n + k]   for k = 0..n-1
 */
__global__ static void
extract_diagonal_kernel(int n, const double* __restrict__ C,
                        double* __restrict__ diag)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= n) return;
    diag[k] = C[k * n + k];
}

/*
 * DD-GEMM kernel: C_dd = A * B  (NN, N×N square, column-major).
 * Double-double arithmetic gives ~106-bit mantissa reference.
 */
static constexpr int DD_TILE = 32;

__global__ static void
dd_gemm_kernel(size_t N,
               const double* __restrict__ A,
               const double* __restrict__ B,
               double2* __restrict__      C_dd)
{
    const size_t row = static_cast<size_t>(blockIdx.y) * DD_TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * DD_TILE + threadIdx.x;

    __shared__ double Asub[DD_TILE][DD_TILE + 1];
    __shared__ double Bsub[DD_TILE][DD_TILE + 1];
    double2 sum = {0.0, 0.0};

    for(int t = 0; t < (int)((N + DD_TILE - 1) / DD_TILE); ++t) {
        size_t ac = (size_t)t * DD_TILE + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] = (row < N && ac < N) ? A[row + ac * N] : 0.0;
        size_t br = (size_t)t * DD_TILE + threadIdx.y;
        Bsub[threadIdx.y][threadIdx.x] = (br < N && col < N) ? B[br + col * N] : 0.0;
        __syncthreads();
#pragma unroll
        for(int i = 0; i < DD_TILE; ++i)
            sum = dd_add(sum, dd_mul(Asub[threadIdx.y][i], Bsub[i][threadIdx.x]));
        __syncthreads();
    }
    if(row < N && col < N) C_dd[row + col * N] = sum;
}

/*
 * Compute relative error vs. DD reference in-place in D[]:
 *   D[idx]  ←  |(D[idx] − C_dd[idx].hi) / C_dd[idx].hi|
 */
__global__ static void
gemm_err_vs_dd_kernel(size_t n_elems,
                      double* __restrict__        D,
                      const double2* __restrict__ C_dd)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n_elems) return;

    const double2 ref = C_dd[idx];
    /* For A,B ~ U(0,1), C[i,j] = Σ_k A[i,k]*B[k,j] > 0 always
     * (sum of N non-negative terms with E[C[i,j]] = N/4 >> 0).
     * This zero-reference branch is dead code for our use case. */
    if(ref.x == 0.0 && ref.y == 0.0) {
        D[idx] = fabs(D[idx]);
        return;
    }
    const double2 gap = dd_sub(D[idx], ref);
    const double2 err = dd_div(gap, ref);
    D[idx] = fabs(err.x);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CRT bit capacity table  (identical to fp64_emul_accuracy.cpp)
 * ═══════════════════════════════════════════════════════════════════════════ */
static constexpr double CRT_BITS[19] = {
    0.0,     0.0,     /*  0,  1  (unused) */
    15.994,  23.976,  /*  2,  3           */
    31.945,  39.894,  /*  4,  5           */
    47.807,  55.708,  /*  6,  7           */
    63.572,  71.411,  /*  8,  9           */
    79.238,  87.040,  /* 10, 11           */
    94.801, 102.522,  /* 12, 13           */
   110.160, 117.782,  /* 14, 15           */
   125.374, 132.949,  /* 16, 17           */
   140.448,           /* 18               */
};

/*
 * Return the maxMantissaBitCount argument that forces exactly s moduli.
 * The library selects the minimum s' such that CRT_BITS[s'] >= maxBits,
 * so setting maxBits = floor(CRT_BITS[s-1]) + 1 forces s' = s.
 */
static int bits_for_moduli(unsigned s)
{
    if(s < 2u)  s = 2u;
    if(s > 18u) s = 18u;
    return static_cast<int>(CRT_BITS[s - 1]) + 1;
}

/* Moduli configurations for Figure 2 (BLAS Test 2) */
static const unsigned MODULI_LIST_FIG2[] = {6, 8, 10, 12, 14, 15, 16, 17, 18};
static constexpr int  N_MODULI_FIG2      = 9;

/* Moduli configurations for Figures 3/4 (Grade A criterion).
 * All s values from 6 to 18 are included.  The former s=14 bug
 * (|X_true| > M_s/2 for specific inputs) has been fixed by replacing
 * the fixed accu::log2P table with a data-dependent log2P computed from
 * the global max of |C32i_prelim|, which adapts the scaling to prevent
 * CRT overflow for all typical inputs. */
static const unsigned MODULI_LIST[] = {6, 8, 10, 12, 14, 16, 18};
static constexpr int  N_MODULI      = 7;

/* ═══════════════════════════════════════════════════════════════════════════
 * DgemmRunner — wraps a hipBLASLt handle for N×N NN-DGEMM
 * ═══════════════════════════════════════════════════════════════════════════ */
struct DgemmRunner {
    hipblasLtHandle_t            handle  = nullptr;
    hipblasLtMatmulDesc_t        desc    = nullptr;
    hipblasLtMatrixLayout_t      layoutA = nullptr;
    hipblasLtMatrixLayout_t      layoutB = nullptr;
    hipblasLtMatrixLayout_t      layoutD = nullptr;
    hipblasLtMatmulPreference_t  pref    = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    bool hasAlgo = false;

    size_t workspaceSize() const { return heur.workspaceSize; }

    void requery()
    {
        int cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, layoutA, layoutB, layoutD, layoutD,
            pref, 1, &heur, &cnt);
        hasAlgo = (cnt > 0);
    }

    void init(int64_t N, bool emulation_enabled)
    {
        constexpr size_t WS_BUDGET = size_t(-1);

        HLT_CHECK(hipblasLtCreate(&handle));
        HLT_CHECK(hipblasLtSetEmulationEnabled(handle, emulation_enabled));

        if(emulation_enabled) {
            /* EAGER: emulate regardless of problem size */
            HLT_CHECK(hipblasLtSetEmulationStrategy(
                handle, HIPBLASLT_EMULATION_STRATEGY_EAGER));
            /* Disable Inf/NaN detection (benchmark uses clean data) */
            HLT_CHECK(hipblasLtSetEmulationSpecialValuesSupport(handle, 0u));
        }

        HLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
        {
            hipblasOperation_t opN = HIPBLAS_OP_N;
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        }

        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layoutA, HIP_R_64F, (uint64_t)N, (uint64_t)N, N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layoutB, HIP_R_64F, (uint64_t)N, (uint64_t)N, N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layoutD, HIP_R_64F, (uint64_t)N, (uint64_t)N, N));

        HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WS_BUDGET, sizeof(WS_BUDGET)));

        requery();
    }

    /* Configure emulation to use exactly s moduli (fixed, no fallback). */
    void set_fixed_s(unsigned s)
    {
        HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
            handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_FIXED));
        HLT_CHECK(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(
            handle, bits_for_moduli(s)));
        requery();
    }

    /* Configure emulation to use ADP (Adaptive Precision) mode.
     * The effective s is chosen per-GEMM from the data; s=num_moduli is
     * the upper bound (default 16 from fp64EmulationNumModuli()). */
    void set_dynamic()
    {
        HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
            handle, HIPBLASLT_EMULATION_MANTISSA_CONTROL_DYNAMIC));
        requery();
    }

    void run(const double* A, const double* B, double* D,
             void* workspace, size_t ws_bytes, hipStream_t stream) const
    {
        const double alpha = 1.0, beta = 0.0;
        HLT_CHECK(hipblasLtMatmul(
            handle, desc,
            &alpha, A, layoutA, B, layoutB,
            &beta,  D, layoutD, D, layoutD,
            hasAlgo ? &heur.algo : nullptr,
            workspace, ws_bytes, stream));
    }

    void destroy()
    {
        if(pref)    { hipblasLtMatmulPreferenceDestroy(pref);    pref    = nullptr; }
        if(layoutD) { hipblasLtMatrixLayoutDestroy(layoutD);     layoutD = nullptr; }
        if(layoutB) { hipblasLtMatrixLayoutDestroy(layoutB);     layoutB = nullptr; }
        if(layoutA) { hipblasLtMatrixLayoutDestroy(layoutA);     layoutA = nullptr; }
        if(desc)    { hipblasLtMatmulDescDestroy(desc);          desc    = nullptr; }
        if(handle)  { hipblasLtDestroy(handle);                  handle  = nullptr; }
    }
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Workspace (grows on demand, never shrinks)
 * ═══════════════════════════════════════════════════════════════════════════ */
struct Workspace {
    void*  ptr   = nullptr;
    size_t bytes = 0;

    void ensure(size_t needed)
    {
        if(needed > bytes) {
            HIP_CHECK(hipFree(ptr));
            bytes = needed;
            HIP_CHECK(hipMalloc(&ptr, bytes));
        }
    }

    void free_all()
    {
        HIP_CHECK(hipFree(ptr));
        ptr   = nullptr;
        bytes = 0;
    }
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Error reduction: copy d_err to host, return (max, mean).
 * Using mean (arithmetic average) as required by Figure 4.
 * ═══════════════════════════════════════════════════════════════════════════ */
static std::pair<double, double>
reduce_errors(size_t n_elems, const double* d_err,
              std::vector<double>& h_buf, hipStream_t stream)
{
    h_buf.resize(n_elems);
    HIP_CHECK(hipMemcpyAsync(h_buf.data(), d_err,
        n_elems * sizeof(double), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double mx  = 0.0;
    double sum = 0.0;
    for(size_t i = 0; i < n_elems; ++i) {
        double v = h_buf[i];
        if(v > mx) mx = v;
        sum += v;
    }
    double avg = sum / static_cast<double>(n_elems);
    return {mx, avg};
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Figure 2: BLAS Test 2
 * ═══════════════════════════════════════════════════════════════════════════ */
static void run_figure2(FILE* fp_out,
                        int n,           /* default 1024 */
                        int b_max,       /* default 64   */
                        int b_step,      /* default 5 (ignored when pow2=true) */
                        bool b_pow2,     /* use b = 1,2,4,8,...,b_max */
                        uint64_t seed,
                        Workspace& ws,
                        hipStream_t stream)
{
    /* Build b sequence */
    std::vector<int> b_vals;
    if(b_pow2) {
        for(int b = 1; b <= b_max; b *= 2)
            b_vals.push_back(b);
    } else {
        for(int b = 0; b <= b_max; b += b_step)
            b_vals.push_back(b);
    }

    std::fprintf(stderr, "\n=== Figure 2: BLAS Test 2,  n = %d ===\n", n);
    std::fprintf(stderr, "  b values:");
    for(int b : b_vals) std::fprintf(stderr, " %d", b);
    std::fprintf(stderr, "\n");
    const size_t N2 = (size_t)n * n;

    /* ── GPU buffers ────────────────────────────────────────────────────── */
    double *d_x, *d_d, *d_A, *d_B, *d_C_ref, *d_C_emul, *d_err, *d_diag;
    HIP_CHECK(hipMalloc(&d_x,      (size_t)n  * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_d,      (size_t)n  * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A,      N2         * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B,      N2         * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C_ref,  N2         * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C_emul, N2         * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_err,    N2         * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_diag,   (size_t)n  * sizeof(double)));

    /* ── Generate x ~ U(1,2) on the host; compute xᵀx in long double ─── */
    std::vector<double> h_x(n), h_d(n);
    for(int i = 0; i < n; ++i) {
        uint64_t s = seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL
                             + 1442695040888963407ULL);
        s = xorshift64_host(s); s = xorshift64_host(s);
        h_x[i] = 1.0 + bits_to_uniform_host(xorshift64_host(s));
    }

    long double xTx_ld = 0.0L;
    for(int i = 0; i < n; ++i) xTx_ld += static_cast<long double>(h_x[i]) * h_x[i];
    const double xTx = static_cast<double>(xTx_ld);
    std::fprintf(stderr, "  xᵀx (long double) = %.15g\n", xTx);

    HIP_CHECK(hipMemcpy(d_x, h_x.data(), (size_t)n * sizeof(double),
                        hipMemcpyHostToDevice));

    /* ── hipBLASLt runners ───────────────────────────────────────────────── */
    DgemmRunner native, emulated, adp_runner;
    native.init(static_cast<int64_t>(n), /*emulation=*/false);
    emulated.init(static_cast<int64_t>(n), /*emulation=*/true);
    adp_runner.init(static_cast<int64_t>(n), /*emulation=*/true);
    adp_runner.set_dynamic();
    ws.ensure(native.workspaceSize());

    /* ── Kernel launch dims for n×n 2-D fill ────────────────────────────── */
    const dim3 blk2d(16, 16);
    const dim3 grd2d(static_cast<unsigned>((n + 15) / 16),
                     static_cast<unsigned>((n + 15) / 16));

    std::vector<double> h_err;
    std::vector<double> h_diag(n);

    /* ── Sweep b ─────────────────────────────────────────────────────────── */
    for(int b : b_vals) {

        /* Compute d vector: d[i] = 2^{j_i},  j_i = −b + round(i·2b/(n−1)) */
        const double delta = (n > 1) ? (2.0 * b) / (n - 1) : 0.0;
        for(int i = 0; i < n; ++i) {
            double ji = -b + std::round(static_cast<double>(i) * delta);
            h_d[i]    = std::ldexp(1.0, static_cast<int>(ji));
        }
        HIP_CHECK(hipMemcpyAsync(d_d, h_d.data(), (size_t)n * sizeof(double),
                                 hipMemcpyHostToDevice, stream));

        /* Fill A and B on GPU */
        hipLaunchKernelGGL(fill_test2_A_kernel, grd2d, blk2d, 0, stream,
                           n, d_A, d_x, d_d);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(fill_test2_B_kernel, grd2d, blk2d, 0, stream,
                           n, d_B, d_x, d_d);
        HIP_CHECK(hipGetLastError());

        /* Native FP64 GEMM → C_ref (off-diagonal reference and native diagonal) */
        ws.ensure(native.workspaceSize());
        native.run(d_A, d_B, d_C_ref, ws.ptr, ws.bytes, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        /* ── Native DGEMM diagonal error ────────────────────────────────── */
        {
            const unsigned blk1d_diag = 256u;
            const unsigned grd1d_diag = static_cast<unsigned>((n + blk1d_diag - 1) / blk1d_diag);
            hipLaunchKernelGGL(extract_diagonal_kernel,
                               dim3(grd1d_diag), dim3(blk1d_diag), 0, stream,
                               n, d_C_ref, d_diag);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipMemcpyAsync(h_diag.data(), d_diag,
                (size_t)n * sizeof(double), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            long double err_nat_max_ld = 0.0L;
            for(int k = 0; k < n; ++k) {
                long double e = fabsl(static_cast<long double>(h_diag[k]) - xTx_ld)
                              / fabsl(xTx_ld);
                if(e > err_nat_max_ld) err_nat_max_ld = e;
            }
            double err_nat_max = static_cast<double>(err_nat_max_ld);
            std::fprintf(fp_out, "DGEMM,%d,0,53.000,%.17e,%.17e\n", b, err_nat_max, err_nat_max);
            std::fflush(fp_out);
        }

        std::fprintf(stderr, "  b = %3d:", b);

    /* Emulated DGEMM for each moduli count (Fig 2 list: {6,8,10,12,14,16,18}) */
    for(int si = 0; si < N_MODULI_FIG2; ++si) {
        unsigned s = MODULI_LIST_FIG2[si];
        emulated.set_fixed_s(s);
        ws.ensure(emulated.workspaceSize());

        emulated.run(d_A, d_B, d_C_emul, ws.ptr, ws.bytes, stream);

        /*
         * Combined all-elements error (matches paper Figure 2 formula):
         *   e_{kk} = |xTx  −  C_emul[k,k]| / |xTx|            (exact LD ref)
         *   e_{ki} = |C_ref[k,i] − C_emul[k,i]| / |C_ref[k,i]|  (native FP64)
         *   err_max = max_{k,i} e_{ki}
         *
         * For large b the native off-diagonal reference is itself inaccurate
         * (individual products span > 53 bits), but the comparison still
         * detects catastrophic failures (sign flips, CRT overflows) because
         * those give error ≈ 1–2 relative to any reasonable reference.
         * The diagonal-only metric (err_diag_max) uses the exact long-double
         * reference and is the most informative for large b.
         */
        hipLaunchKernelGGL(fig2_error_kernel, grd2d, blk2d, 0, stream,
                           n, d_C_emul, d_C_ref, xTx, d_err);
        HIP_CHECK(hipGetLastError());
        double all_max = reduce_errors(N2, d_err, h_err, stream).first;

        const unsigned blk1d_diag = 256u;
        const unsigned grd1d_diag = static_cast<unsigned>((n + blk1d_diag - 1) / blk1d_diag);
        hipLaunchKernelGGL(extract_diagonal_kernel,
                           dim3(grd1d_diag), dim3(blk1d_diag), 0, stream,
                           n, d_C_emul, d_diag);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpyAsync(h_diag.data(), d_diag,
            (size_t)n * sizeof(double), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        long double err_diag_max_ld = 0.0L;
        for(int k = 0; k < n; ++k) {
            long double e = fabsl(static_cast<long double>(h_diag[k]) - xTx_ld)
                          / fabsl(xTx_ld);
            if(e > err_diag_max_ld) err_diag_max_ld = e;
        }
        double err_diag_max = static_cast<double>(err_diag_max_ld);

        std::fprintf(fp_out, "OS2-fixed-s%u,%d,%u,%.3f,%.17e,%.17e\n",
                     s, b, s, CRT_BITS[s], all_max, err_diag_max);
        std::fflush(fp_out);
        std::fprintf(stderr, " s=%u:%.3e(all)/%.3e(diag)", s, all_max, err_diag_max);
    }  /* moduli loop */
        /* ── ADP (dynamic mode) combined + diagonal error ───────────────── */
        {
            ws.ensure(adp_runner.workspaceSize());
            adp_runner.run(d_A, d_B, d_C_emul, ws.ptr, ws.bytes, stream);

            /* Combined error (paper formula): diagonal vs. exact LD,
             * off-diagonal vs. native FP64 DGEMM reference. */
            hipLaunchKernelGGL(fig2_error_kernel, grd2d, blk2d, 0, stream,
                               n, d_C_emul, d_C_ref, xTx, d_err);
            HIP_CHECK(hipGetLastError());
            double adp_all_max = reduce_errors(N2, d_err, h_err, stream).first;

            const unsigned blk1d_diag = 256u;
            const unsigned grd1d_diag = static_cast<unsigned>((n + blk1d_diag - 1) / blk1d_diag);
            hipLaunchKernelGGL(extract_diagonal_kernel,
                               dim3(grd1d_diag), dim3(blk1d_diag), 0, stream,
                               n, d_C_emul, d_diag);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipMemcpyAsync(h_diag.data(), d_diag,
                (size_t)n * sizeof(double), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            long double err_adp_max_ld = 0.0L;
            for(int k = 0; k < n; ++k) {
                long double e = fabsl(static_cast<long double>(h_diag[k]) - xTx_ld)
                              / fabsl(xTx_ld);
                if(e > err_adp_max_ld) err_adp_max_ld = e;
            }
            double err_adp_diag_max = static_cast<double>(err_adp_max_ld);
            /* Report ADP with the absolute max s=18 (the overflow threshold) */
            std::fprintf(fp_out, "ADP,%d,18,%.3f,%.17e,%.17e\n",
                         b, CRT_BITS[18], adp_all_max, err_adp_diag_max);
            std::fflush(fp_out);
            std::fprintf(stderr, " ADP:%.3e(all)/%.3e(diag)", adp_all_max, err_adp_diag_max);
        }

        std::fprintf(stderr, "\n");
    }  /* b loop */

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    adp_runner.destroy();
    emulated.destroy();
    native.destroy();
    HIP_CHECK(hipFree(d_diag));
    HIP_CHECK(hipFree(d_err));
    HIP_CHECK(hipFree(d_C_emul));
    HIP_CHECK(hipFree(d_C_ref));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_d));
    HIP_CHECK(hipFree(d_x));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Figures 3 & 4: Grade A criterion — componentwise error vs. N
 * ═══════════════════════════════════════════════════════════════════════════ */
static void run_figures34(FILE* fp_out,
                          const std::vector<int>& n_list,
                          int num_seeds,
                          uint64_t base_seed,
                          Workspace& ws,
                          hipStream_t stream)
{
    std::fprintf(stderr,
        "\n=== Figures 3 & 4: Grade A criterion — %d seeds, %d N-values ===\n",
        num_seeds, static_cast<int>(n_list.size()));

    for(int N : n_list) {
        std::fprintf(stderr, "  N = %d ...\n", N);
        const size_t N2 = static_cast<size_t>(N) * N;

        /* ── GPU buffers ─────────────────────────────────────────────────── */
        double *d_A, *d_B, *d_D, *d_err;
        double2* d_C_dd;
        HIP_CHECK(hipMalloc(&d_A,    N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_B,    N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_D,    N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_err,  N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_C_dd, N2 * sizeof(double2)));

        /* ── hipBLASLt runners ───────────────────────────────────────────── */
        DgemmRunner native, emulated, adp;
        native.init(static_cast<int64_t>(N), /*emulation=*/false);
        emulated.init(static_cast<int64_t>(N), /*emulation=*/true);
        adp.init(static_cast<int64_t>(N), /*emulation=*/true);
        adp.set_dynamic();

        const unsigned blk1d = 256u;
        const unsigned grd1d = static_cast<unsigned>((N2 + blk1d - 1) / blk1d);
        const dim3 dd_blk(DD_TILE, DD_TILE);
        const dim3 dd_grd(static_cast<unsigned>((N + DD_TILE - 1) / DD_TILE),
                          static_cast<unsigned>((N + DD_TILE - 1) / DD_TILE));

        std::vector<double> h_err;

        /* Accumulators for seed-averaging */
        double nat_max_sum = 0.0, nat_med_sum = 0.0;
        double adp_max_sum = 0.0, adp_med_sum = 0.0;
        std::vector<double> emu_max_sum(N_MODULI, 0.0);
        std::vector<double> emu_med_sum(N_MODULI, 0.0);

        for(int seed_idx = 0; seed_idx < num_seeds; ++seed_idx) {
            /* Use different seeds for A and B (both ~ U(0,1), no cancellation) */
            uint64_t seed_A = base_seed + static_cast<uint64_t>(seed_idx) * 2u;
            uint64_t seed_B = seed_A + 1u;

            /* Fill A, B ~ U(0,1) on GPU */
            hipLaunchKernelGGL(randmat_uniform01_kernel,
                               dim3(grd1d), dim3(blk1d), 0, stream,
                               N2, d_A, seed_A);
            HIP_CHECK(hipGetLastError());
            hipLaunchKernelGGL(randmat_uniform01_kernel,
                               dim3(grd1d), dim3(blk1d), 0, stream,
                               N2, d_B, seed_B);
            HIP_CHECK(hipGetLastError());

            /* DD-GEMM reference */
            hipLaunchKernelGGL(dd_gemm_kernel, dd_grd, dd_blk, 0, stream,
                               static_cast<size_t>(N), d_A, d_B, d_C_dd);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            /* ── Native FP64 DGEMM ────────────────────────────────────────── */
            ws.ensure(native.workspaceSize());
            native.run(d_A, d_B, d_D, ws.ptr, ws.bytes, stream);
            /* Copy D → err buffer; compute error in-place */
            HIP_CHECK(hipMemcpyAsync(d_err, d_D, N2 * sizeof(double),
                                     hipMemcpyDeviceToDevice, stream));
            hipLaunchKernelGGL(gemm_err_vs_dd_kernel,
                               dim3(grd1d), dim3(blk1d), 0, stream,
                               N2, d_err, d_C_dd);
            HIP_CHECK(hipGetLastError());
            auto [nat_max, nat_med] = reduce_errors(N2, d_err, h_err, stream);
            nat_max_sum += nat_max;
            nat_med_sum += nat_med;

            /* ── ADP (dynamic mode) ───────────────────────────────────────── */
            {
                ws.ensure(adp.workspaceSize());
                adp.run(d_A, d_B, d_D, ws.ptr, ws.bytes, stream);
                HIP_CHECK(hipMemcpyAsync(d_err, d_D, N2 * sizeof(double),
                                         hipMemcpyDeviceToDevice, stream));
                hipLaunchKernelGGL(gemm_err_vs_dd_kernel,
                                   dim3(grd1d), dim3(blk1d), 0, stream,
                                   N2, d_err, d_C_dd);
                HIP_CHECK(hipGetLastError());
                auto [a_max, a_med] = reduce_errors(N2, d_err, h_err, stream);
                adp_max_sum += a_max;
                adp_med_sum += a_med;
            }

            /* ── Emulated DGEMM for each s ────────────────────────────────── */
            for(int si = 0; si < N_MODULI; ++si) {
                emulated.set_fixed_s(MODULI_LIST[si]);
                ws.ensure(emulated.workspaceSize());
                emulated.run(d_A, d_B, d_D, ws.ptr, ws.bytes, stream);
                HIP_CHECK(hipMemcpyAsync(d_err, d_D, N2 * sizeof(double),
                                         hipMemcpyDeviceToDevice, stream));
                hipLaunchKernelGGL(gemm_err_vs_dd_kernel,
                                   dim3(grd1d), dim3(blk1d), 0, stream,
                                   N2, d_err, d_C_dd);
                HIP_CHECK(hipGetLastError());
                auto [e_max, e_med] = reduce_errors(N2, d_err, h_err, stream);
                emu_max_sum[si] += e_max;
                emu_med_sum[si] += e_med;
            }
        } /* seed loop */

        /* ── Seed-averaged output ─────────────────────────────────────────── */
        const double inv = 1.0 / num_seeds;
        std::fprintf(fp_out, "DGEMM,%d,%.1f,%.17e,%.17e\n",
                     N, 53.0, nat_max_sum * inv, nat_med_sum * inv);
        /* ADP: report with max s=18 (overflow threshold) as the label */
        std::fprintf(fp_out, "ADP-dynamic,%d,%.3f,%.17e,%.17e\n",
                     N, CRT_BITS[18], adp_max_sum * inv, adp_med_sum * inv);
        for(int si = 0; si < N_MODULI; ++si) {
            unsigned s = MODULI_LIST[si];
            std::fprintf(fp_out, "OS2-accu-s%u,%d,%.3f,%.17e,%.17e\n",
                         s, N, CRT_BITS[s],
                         emu_max_sum[si] * inv, emu_med_sum[si] * inv);
        }
        std::fflush(fp_out);

        std::fprintf(stderr,
            "    Native DGEMM  : max=%.3e  med=%.3e\n",
            nat_max_sum * inv, nat_med_sum * inv);
        for(int si = 0; si < N_MODULI; ++si) {
            std::fprintf(stderr,
                "    s=%u (~%.0f bits): max=%.3e  med=%.3e\n",
                MODULI_LIST[si], CRT_BITS[MODULI_LIST[si]],
                emu_max_sum[si] * inv, emu_med_sum[si] * inv);
        }

        /* ── Cleanup ─────────────────────────────────────────────────────── */
        adp.destroy();
        emulated.destroy();
        native.destroy();
        HIP_CHECK(hipFree(d_C_dd));
        HIP_CHECK(hipFree(d_err));
        HIP_CHECK(hipFree(d_D));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_A));
    } /* N loop */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI parsing helpers
 * ═══════════════════════════════════════════════════════════════════════════ */
static void print_usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --fig  2|34|all        Which figures to reproduce (default: all)\n"
        "  --n-fig2  N            Square matrix size for Fig 2 (default: 1024)\n"
        "  --b-max   B            Max exponent-range param b (default: 80)\n"
        "  --b-step  S            Step for b sweep (default: 5)\n"
        "  --n-list  N1,N2,...    Matrix sizes for Figs 3/4\n"
        "                         (default: 128,256,512,1024,2048,4096)\n"
        "  --seeds   K            Number of random seeds for Figs 3/4 (default: 5)\n"
        "  --out-fig2  FILE       Output CSV for Fig 2 (default: fig2_results.csv)\n"
        "  --out-fig34 FILE       Output CSV for Figs 3/4 (default: fig34_results.csv)\n"
        "  --seed  UINT64         Base RNG seed (default: 98765)\n"
        "  -h, --help             Print this help\n"
        "\n"
        "Output:\n"
        "  fig2_results.csv   columns: b, num_moduli, crt_bits, err_max\n"
        "  fig34_results.csv  columns: algo, N, crt_bits, err_max, err_med\n"
        "\n"
        "Then plot with:\n"
        "  python3 src/papers_fig234_plot.py\n",
        prog);
}

static std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> v;
    size_t pos = 0;
    while(pos <= s.size()) {
        size_t next = s.find(',', pos);
        std::string tok = (next == std::string::npos)
                        ? s.substr(pos) : s.substr(pos, next - pos);
        if(!tok.empty()) v.push_back(std::stoi(tok));
        if(next == std::string::npos) break;
        pos = next + 1;
    }
    return v;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char** argv)
{
    /* ── Defaults ─────────────────────────────────────────────────────────── */
    bool run_fig2  = false;
    bool run_fig34 = false;
    int  n_fig2    = 1024;
    int  b_max     = 64;   /* default: 64 (= 2^6, covers all 5 configs) */
    int  b_step    = 5;
    bool b_pow2    = true; /* default: sample at b = 1,2,4,8,16,32,64 */
    int  num_seeds = 5;
    std::vector<int> n_list = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const char* out_fig2  = "fig2_results.csv";
    const char* out_fig34 = "fig34_results.csv";
    uint64_t base_seed = 98765ULL;

    /* ── Argument parsing ────────────────────────────────────────────────── */
    for(int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if(a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if(a == "--fig" && i + 1 < argc) {
            std::string f = argv[++i];
            if(f == "2")         { run_fig2 = true; }
            else if(f == "34")   { run_fig34 = true; }
            else if(f == "all")  { run_fig2 = run_fig34 = true; }
            else {
                std::fprintf(stderr, "Unknown --fig value: %s\n", f.c_str());
                return EXIT_FAILURE;
            }
        } else if(a == "--n-fig2" && i + 1 < argc) {
            n_fig2 = std::stoi(argv[++i]);
        } else if(a == "--b-max" && i + 1 < argc) {
            b_max = std::stoi(argv[++i]);
        } else if(a == "--b-step" && i + 1 < argc) {
            b_step = std::stoi(argv[++i]);
            b_pow2 = false;   /* explicit step overrides powers-of-2 default */
        } else if(a == "--b-pow2") {
            b_pow2 = true;
        } else if(a == "--b-linear") {
            b_pow2 = false;
        } else if(a == "--seeds" && i + 1 < argc) {
            num_seeds = std::stoi(argv[++i]);
        } else if(a == "--n-list" && i + 1 < argc) {
            n_list = parse_int_list(argv[++i]);
        } else if(a == "--out-fig2" && i + 1 < argc) {
            out_fig2 = argv[++i];
        } else if(a == "--out-fig34" && i + 1 < argc) {
            out_fig34 = argv[++i];
        } else if(a == "--seed" && i + 1 < argc) {
            base_seed = std::stoull(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    /* Default: run everything */
    if(!run_fig2 && !run_fig34) { run_fig2 = run_fig34 = true; }

    /* ── Device info ─────────────────────────────────────────────────────── */
    {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        std::fprintf(stderr, "Device : %s\n", prop.name);
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    Workspace ws;

    /* ── Figure 2 ────────────────────────────────────────────────────────── */
    if(run_fig2) {
        FILE* fp = std::fopen(out_fig2, "w");
        if(!fp) {
            std::fprintf(stderr, "Cannot open output file: %s\n", out_fig2);
            return EXIT_FAILURE;
        }
        std::fprintf(fp, "algo,b,num_moduli,crt_bits,err_max,err_diag_max\n");
        run_figure2(fp, n_fig2, b_max, b_step, b_pow2, base_seed, ws, stream);
        std::fclose(fp);
        std::fprintf(stderr, "\nFigure 2 results → %s\n", out_fig2);
    }

    /* ── Figures 3 & 4 ───────────────────────────────────────────────────── */
    if(run_fig34) {
        FILE* fp = std::fopen(out_fig34, "w");
        if(!fp) {
            std::fprintf(stderr, "Cannot open output file: %s\n", out_fig34);
            return EXIT_FAILURE;
        }
        std::fprintf(fp, "algo,N,crt_bits,err_max,err_med\n");
        run_figures34(fp, n_list, num_seeds, base_seed, ws, stream);
        std::fclose(fp);
        std::fprintf(stderr, "\nFigures 3/4 results → %s\n", out_fig34);
    }

    ws.free_all();
    HIP_CHECK(hipStreamDestroy(stream));
    return EXIT_SUCCESS;
}
