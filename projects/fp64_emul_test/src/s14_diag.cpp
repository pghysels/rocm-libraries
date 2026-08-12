// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * s14_diag.cpp
 *
 * Diagnostic for the s=14 library bug observed at N=128 and N=512
 * (both 2^7 and 2^9, but NOT at 2^8, 2^10, ...).
 *
 * Experiments:
 *   1. Scan N = 64..1280 (step 64) with s=13,14,15 and report max relative
 *      error vs. DD-GEMM reference.  Maps which (N, s) pairs fail.
 *
 *   2. For N=128, s=14: find ALL (row, col) pairs with error > 0.1,
 *      print their indices and values to identify the pattern.
 *
 *   3. Print workspace size reported by the library for each (N, s).
 *
 * Build: added as "s14_diag" target in CMakeLists.txt.
 * Run:
 *   ./build/s14_diag
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

/* ─── Error-check macros ─────────────────────────────────────────────────── */
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

/* ─── Double-double primitives ───────────────────────────────────────────── */
#ifndef __HIP_DEVICE_COMPILE__
# pragma clang optimize off
#endif
__host__ __device__ __forceinline__
void two_sum(double a, double b, double& s, double& e)
{ s = a+b; double v=s-a,u=s-v; e=(a-u)+(b-v); }
__host__ __device__ __forceinline__
void fast_two_sum(double a, double b, double& s, double& e)
{ s=a+b; e=(a-s)+b; }
__host__ __device__ __forceinline__
void two_sub(double a, double b, double& s, double& e)
{ s=a-b; double v=s-a,u=s-v; e=(a-u)-(b+v); }
__host__ __device__ __forceinline__
void two_prod(double a, double b, double& p, double& e)
{ p=a*b;
#ifdef __HIP_DEVICE_COMPILE__
  e=__fma_rn(a,b,-p);
#else
  e=std::fma(a,b,-p);
#endif
}
#ifndef __HIP_DEVICE_COMPILE__
# pragma clang optimize on
#endif

__host__ __device__ __forceinline__
double2 dd_add(double2 a, double2 b) {
    double2 c; two_sum(a.x,b.x,c.x,c.y); c.y+=a.y+b.y; fast_two_sum(c.x,c.y,c.x,c.y); return c;
}
__host__ __device__ __forceinline__
double2 dd_sub(double a, double2 b) {
    double2 c; two_sub(a,b.x,c.x,c.y); c.y-=b.y; fast_two_sum(c.x,c.y,c.x,c.y); return c;
}
__host__ __device__ __forceinline__
double2 dd_mul(double a, double b) { double2 c; two_prod(a,b,c.x,c.y); return c; }
__host__ __device__ __forceinline__
double2 dd_div(double2 a, double2 b) {
    double q1=a.x/b.x; double2 q1b; two_prod(q1,b.x,q1b.x,q1b.y); q1b.y+=q1*b.y;
    double rhi,re; two_sub(a.x,q1b.x,rhi,re); double rlo=re+(a.y-q1b.y);
    double q2=rhi/b.x+rlo/b.x; double2 r; fast_two_sum(q1,q2,r.x,r.y); return r;
}

/* ─── RNG ────────────────────────────────────────────────────────────────── */
__device__ __forceinline__ uint64_t xorshift64_dev(uint64_t s)
{ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
__device__ __forceinline__ double bits_to_uniform_dev(uint64_t b)
{ return static_cast<double>(b>>11)*(1.0/9007199254740992.0); }

/* ─── DD-GEMM kernel (NN, square, column-major) ─────────────────────────── */
static constexpr int DD_TILE = 32;
__global__ static void
dd_gemm_kernel(size_t N, const double* A, const double* B, double2* C_dd)
{
    size_t row = static_cast<size_t>(blockIdx.y)*DD_TILE + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x)*DD_TILE + threadIdx.x;
    __shared__ double Asub[DD_TILE][DD_TILE+1], Bsub[DD_TILE][DD_TILE+1];
    double2 sum = {0.0,0.0};
    for(int t=0;t<(int)((N+DD_TILE-1)/DD_TILE);++t) {
        size_t ac=(size_t)t*DD_TILE+threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] = (row<N&&ac<N) ? A[row+ac*N] : 0.0;
        size_t br=(size_t)t*DD_TILE+threadIdx.y;
        Bsub[threadIdx.y][threadIdx.x] = (br<N&&col<N) ? B[br+col*N] : 0.0;
        __syncthreads();
#pragma unroll
        for(int i=0;i<DD_TILE;++i) sum=dd_add(sum,dd_mul(Asub[threadIdx.y][i],Bsub[i][threadIdx.x]));
        __syncthreads();
    }
    if(row<N&&col<N) C_dd[row+col*N]=sum;
}

/* Fill U(0,1) */
__global__ static void
fill_uniform(size_t n_elems, double* A, uint64_t seed)
{
    size_t idx=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx>=n_elems) return;
    uint64_t s=seed^(idx*0x9e3779b97f4a7c15ULL+1442695040888963407ULL);
    s=xorshift64_dev(s); s=xorshift64_dev(s);
    A[idx]=bits_to_uniform_dev(xorshift64_dev(s));
}


/* ─── DgemmRunner ────────────────────────────────────────────────────────── */
struct Runner {
    hipblasLtHandle_t handle{}; hipblasLtMatmulDesc_t desc{};
    hipblasLtMatrixLayout_t lA{},lB{},lD{};
    hipblasLtMatmulPreference_t pref{};
    hipblasLtMatmulHeuristicResult_t heur{};
    bool hasAlgo=false;

    void requery() { int cnt=0; hipblasLtMatmulAlgoGetHeuristic(handle,desc,lA,lB,lD,lD,pref,1,&heur,&cnt); hasAlgo=(cnt>0); }

    void init(int64_t N, bool emul) {
        HLT_CHECK(hipblasLtCreate(&handle));
        HLT_CHECK(hipblasLtSetEmulationEnabled(handle,emul));
        if(emul){
            HLT_CHECK(hipblasLtSetEmulationStrategy(handle,HIPBLASLT_EMULATION_STRATEGY_EAGER));
            HLT_CHECK(hipblasLtSetEmulationSpecialValuesSupport(handle,0u));
        }
        HLT_CHECK(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_64F,HIP_R_64F));
        hipblasOperation_t opN=HIPBLAS_OP_N;
        HLT_CHECK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&opN,sizeof(opN)));
        HLT_CHECK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(&lA,HIP_R_64F,(uint64_t)N,(uint64_t)N,N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(&lB,HIP_R_64F,(uint64_t)N,(uint64_t)N,N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(&lD,HIP_R_64F,(uint64_t)N,(uint64_t)N,N));
        HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
        size_t wb=size_t(-1);
        HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wb,sizeof(wb)));
        requery();
    }
    void set_s(unsigned s) {
        HLT_CHECK(hipblasLtSetEmulationNumModuli(handle, static_cast<int>(s)));
        requery();
    }
    void run(const double* A,const double* B,double* D,void* ws,size_t wsz,hipStream_t st) const {
        const double al=1.0,be=0.0;
        HLT_CHECK(hipblasLtMatmul(handle,desc,&al,A,lA,B,lB,&be,D,lD,D,lD,hasAlgo?&heur.algo:nullptr,ws,wsz,st));
    }
    size_t workspaceSize() const { return heur.workspaceSize; }
    void destroy() {
        if(pref)    { hipblasLtMatmulPreferenceDestroy(pref); pref={}; }
        if(lD)      { hipblasLtMatrixLayoutDestroy(lD); lD={}; }
        if(lB)      { hipblasLtMatrixLayoutDestroy(lB); lB={}; }
        if(lA)      { hipblasLtMatrixLayoutDestroy(lA); lA={}; }
        if(desc)    { hipblasLtMatmulDescDestroy(desc); desc={}; }
        if(handle)  { hipblasLtDestroy(handle); handle={}; }
    }
};

/* ─── main ───────────────────────────────────────────────────────────────── */
int main()
{
    /* device info */
    {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop,0));
        std::fprintf(stderr,"Device: %s\n\n",prop.name);
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* ── Experiment 1: scan N = 64..1280 step 64, s ∈ {13,14,15} ─────────
     * For each (N,s): compute max relative error vs DD-GEMM reference.    */
    std::fprintf(stdout,"=== Experiment 1: error scan N=64..1280 step 64, s=13,14,15 ===\n");
    std::fprintf(stdout,"%-6s %-4s %8s %12s %8s\n","N","s","ws_MiB","max_relerr","PASS/FAIL");

    static const unsigned S_LIST[] = {13,14,15};
    static const int N_LIST_EXP1[] = {
        64,128,192,256,320,384,448,512,576,640,768,1024,1280
    };
    static constexpr int N_N1 = 13;
    static constexpr double FAIL_THRESH = 1e-4;   /* well above DD noise */

    const uint64_t SEED_A = 12345ULL, SEED_B = 67890ULL;

    for(int ni=0; ni<N_N1; ++ni) {
        int N = N_LIST_EXP1[ni];
        size_t N2 = (size_t)N*N;

        /* allocate GPU buffers */
        double *d_A,*d_B,*d_D,*d_err; double2 *d_dd;
        HIP_CHECK(hipMalloc(&d_A, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_B, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_D, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_err, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_dd,  N2*sizeof(double2)));

        /* fill matrices */
        unsigned blk=256u;
        hipLaunchKernelGGL(fill_uniform,dim3((N2+blk-1)/blk),dim3(blk),0,stream,N2,d_A,SEED_A);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(fill_uniform,dim3((N2+blk-1)/blk),dim3(blk),0,stream,N2,d_B,SEED_B);
        HIP_CHECK(hipGetLastError());

        /* DD-GEMM reference */
        dim3 dd_blk(DD_TILE,DD_TILE);
        dim3 dd_grd((N+DD_TILE-1)/DD_TILE,(N+DD_TILE-1)/DD_TILE);
        hipLaunchKernelGGL(dd_gemm_kernel,dd_grd,dd_blk,0,stream,(size_t)N,d_A,d_B,d_dd);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Emulated GEMM runner */
        Runner emul;
        emul.init((int64_t)N,true);

        for(unsigned s : S_LIST) {
            emul.set_s(s);
            size_t ws_bytes = emul.workspaceSize();
            void* d_ws=nullptr;
            if(ws_bytes>0) HIP_CHECK(hipMalloc(&d_ws,ws_bytes));

            emul.run(d_A,d_B,d_D,d_ws,ws_bytes,stream);

            /* compute per-element relative error on host */
            std::vector<double> h_D(N2), h_err_buf(N2);
            std::vector<double2> h_dd(N2);
            HIP_CHECK(hipMemcpy(h_D.data(), d_D, N2*sizeof(double), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_dd.data(),d_dd,N2*sizeof(double2),hipMemcpyDeviceToHost));

            double err_max = 0.0;
            for(size_t idx=0; idx<N2; ++idx) {
                double ref = h_dd[idx].x;
                double err = (std::fabs(ref)<1e-300) ? std::fabs(h_D[idx])
                           : std::fabs(h_D[idx]-ref)/std::fabs(ref);
                if(err>err_max) err_max=err;
            }

            const char* status = (err_max>FAIL_THRESH) ? "FAIL ***" : "pass";
            std::fprintf(stdout,"%-6d %-4u %8.3f %12.3e %s\n",
                         N, s, ws_bytes/(1024.0*1024.0), err_max, status);
            std::fflush(stdout);

            if(d_ws) HIP_CHECK(hipFree(d_ws)); d_ws=nullptr;
        }
        emul.destroy();

        HIP_CHECK(hipFree(d_dd));
        HIP_CHECK(hipFree(d_err));
        HIP_CHECK(hipFree(d_D));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_A));
    }

    /* ── Experiment 2: for N=128, s=14 — find all failing (row,col) ────── */
    std::fprintf(stdout,"\n=== Experiment 2: N=128, s=14 — location of failing elements ===\n");
    {
        int N=128;
        size_t N2=(size_t)N*N;
        double *d_A,*d_B,*d_D; double2 *d_dd;
        HIP_CHECK(hipMalloc(&d_A, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_B, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_D, N2*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_dd, N2*sizeof(double2)));
        unsigned blk=256u;
        hipLaunchKernelGGL(fill_uniform,dim3((N2+blk-1)/blk),dim3(blk),0,stream,N2,d_A,SEED_A);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(fill_uniform,dim3((N2+blk-1)/blk),dim3(blk),0,stream,N2,d_B,SEED_B);
        HIP_CHECK(hipGetLastError());
        dim3 dd_blk(DD_TILE,DD_TILE);
        dim3 dd_grd((N+DD_TILE-1)/DD_TILE,(N+DD_TILE-1)/DD_TILE);
        hipLaunchKernelGGL(dd_gemm_kernel,dd_grd,dd_blk,0,stream,(size_t)N,d_A,d_B,d_dd);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));

        Runner emul;
        emul.init((int64_t)N,true);
        emul.set_s(14u);
        size_t ws_bytes=emul.workspaceSize();
        void* d_ws=nullptr;
        if(ws_bytes>0) HIP_CHECK(hipMalloc(&d_ws,ws_bytes));
        emul.run(d_A,d_B,d_D,d_ws,ws_bytes,stream);

        std::vector<double> h_D(N2), h_A(N2), h_B(N2);
        std::vector<double2> h_dd(N2);
        HIP_CHECK(hipMemcpy(h_D.data(), d_D,  N2*sizeof(double),  hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_dd.data(),d_dd,  N2*sizeof(double2), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_A.data(), d_A,  N2*sizeof(double),  hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_B.data(), d_B,  N2*sizeof(double),  hipMemcpyDeviceToHost));

        std::fprintf(stdout,"workspace size: %.3f MiB\n", ws_bytes/(1024.0*1024.0));
        std::fprintf(stdout,"%-6s %-6s %14s %14s %14s\n","row","col","emul","DD_ref","relerr");

        int fail_count = 0;
        for(int col=0; col<N; ++col) {
            for(int row=0; row<N; ++row) {
                size_t idx = (size_t)col*N + row;  /* column-major */
                double emul_val = h_D[idx];
                double ref_val  = h_dd[idx].x;
                double err = (std::fabs(ref_val)<1e-300) ? std::fabs(emul_val)
                           : std::fabs(emul_val-ref_val)/std::fabs(ref_val);
                if(err > 0.1) {
                    std::fprintf(stdout,"%-6d %-6d %14.7e %14.7e %14.3e\n",
                                 row, col, emul_val, ref_val, err);
                    ++fail_count;
                }
            }
        }
        std::fprintf(stdout,"Total failing elements: %d / %zu\n", fail_count, N2);

        /* Are failing rows/cols in specific ranges? */
        if(fail_count > 0) {
            /* print row and column histograms */
            std::vector<int> row_count(N,0), col_count(N,0);
            for(int col=0; col<N; ++col) for(int row=0; row<N; ++row) {
                size_t idx=(size_t)col*N+row;
                double ref=h_dd[idx].x;
                double err=(std::fabs(ref)<1e-300)?std::fabs(h_D[idx])
                          :std::fabs(h_D[idx]-ref)/std::fabs(ref);
                if(err>0.1){row_count[row]++; col_count[col]++;}
            }
            std::fprintf(stdout,"\nRows with failures:");
            for(int r=0;r<N;++r) if(row_count[r]>0) std::fprintf(stdout," %d(%d)",r,row_count[r]);
            std::fprintf(stdout,"\nCols with failures:");
            for(int c=0;c<N;++c) if(col_count[c]>0) std::fprintf(stdout," %d(%d)",c,col_count[c]);
            std::fprintf(stdout,"\n");
        }

        if(d_ws) HIP_CHECK(hipFree(d_ws));
        emul.destroy();
        HIP_CHECK(hipFree(d_dd)); HIP_CHECK(hipFree(d_D));
        HIP_CHECK(hipFree(d_B));  HIP_CHECK(hipFree(d_A));
    }

    /* ── Experiment 3: workspace size vs N for s=14 ─────────────────────── */
    std::fprintf(stdout,"\n=== Experiment 3: workspace size for s=14 across N ===\n");
    std::fprintf(stdout,"%-6s %12s\n","N","ws_bytes");
    for(int N : {64,128,192,256,320,384,448,512,576,640,768,1024}) {
        Runner emul; emul.init((int64_t)N,true); emul.set_s(14u);
        std::fprintf(stdout,"%-6d %12zu\n", N, emul.workspaceSize());
        emul.destroy();
    }

    /* ── Experiment 4: workspace dump for N=128, s=14 ───────────────────────
     * Pre-allocate the workspace so we can inspect it after the run.
     * For each of the 3 known failing elements (row,col), at EACH modulus:
     *   - Extract A8i row-slice and B8i col-slice from workspace
     *   - Extract actual C32i[modulus, row, col]
     *   - Compute expected C32i = dot(A8i_row, B8i_col) on CPU
     *   - If actual ≠ expected → INT8 GEMM is buggy for that modulus
     *   - If actual == expected but output is wrong → CRT/finalize is buggy
     * ───────────────────────────────────────────────────────────────────── */
    std::fprintf(stdout, "\n=== Experiment 4: workspace dump N=128, s=14 ===\n");
    {
        const int     N  = 128;
        const unsigned S = 14u;
        const size_t N2  = (size_t)N * N;

        /* Workspace layout (matching fp64_emulation.cpp internals):
         *  lda8i  = oz2_pad(k) = 128  (k = N = 128)
         *  cola8i = oz2_pad(m) = 128  (m = N = 128)
         *  ldb8i  = lda8i = 128
         *  ldc32i = cola8i = 128
         *  szC32i = ldc32i * n = 128 * 128 = 16384 ints
         *
         *  ws layout:
         *    A8i   : [S][lda8i * cola8i] int8   = [14][16384] int8
         *    B8i   : [S][ldb8i * n]      int8   = [14][16384] int8
         *    C32i  : [S][ldc32i * n]     int32  = [14][16384] int32
         *    (followed by Zhi, Zlo, sftA, sftB, nan_flag, row_max)
         */
        const size_t lda8i  = 128, cola8i = 128, ldb8i = 128, ldc32i = 128;
        const size_t szSlice = lda8i * cola8i;           /* 16384 bytes per modulus in A8i/B8i */
        const size_t szC32i  = ldc32i * (size_t)N;       /* 16384 ints per modulus */
        const size_t szA8i   = S * szSlice;
        const size_t szB8i   = S * szSlice;
        const size_t szC32i_total = S * szC32i;

        /* Query actual workspace from the library */
        Runner emul_pre;
        emul_pre.init((int64_t)N, true);
        emul_pre.set_s(S);
        const size_t ws_size = emul_pre.workspaceSize();
        emul_pre.destroy();

        std::fprintf(stdout, "Library workspace: %.3f MiB  (expected ~%.3f MiB)\n",
                     ws_size / (1024.0 * 1024.0),
                     (szA8i + szB8i) * sizeof(int8_t) / (1024.0 * 1024.0)
                     + szC32i_total * sizeof(int32_t) / (1024.0 * 1024.0));

        /* Verify layout assumptions (library must allocate at least our computed A8i+B8i+C32i) */
        const size_t min_expected = (szA8i + szB8i) * sizeof(int8_t)
                                  + szC32i_total * sizeof(int32_t);
        if(ws_size < min_expected) {
            std::fprintf(stdout,
                "ERROR: library workspace (%zu) < expected (%zu) — layout assumptions wrong\n",
                ws_size, min_expected);
        }

        /* Allocate GPU buffers */
        double *d_A, *d_B, *d_D; double2 *d_dd_ref;
        void   *d_ws;
        HIP_CHECK(hipMalloc(&d_A, N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_B, N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_D, N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_dd_ref, N2 * sizeof(double2)));
        HIP_CHECK(hipMalloc(&d_ws, ws_size));

        /* Fill A, B (same seeds as Experiment 2) */
        unsigned blk = 256u;
        hipLaunchKernelGGL(fill_uniform, dim3((N2+blk-1)/blk), dim3(blk), 0, stream,
                           N2, d_A, SEED_A);
        HIP_CHECK(hipGetLastError());
        hipLaunchKernelGGL(fill_uniform, dim3((N2+blk-1)/blk), dim3(blk), 0, stream,
                           N2, d_B, SEED_B);
        HIP_CHECK(hipGetLastError());

        /* DD-GEMM reference */
        dim3 dd_blk(DD_TILE,DD_TILE);
        dim3 dd_grd((N+DD_TILE-1)/DD_TILE, (N+DD_TILE-1)/DD_TILE);
        hipLaunchKernelGGL(dd_gemm_kernel, dd_grd, dd_blk, 0, stream,
                           (size_t)N, d_A, d_B, d_dd_ref);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Run emulation with pre-allocated workspace */
        Runner emul;
        emul.init((int64_t)N, true);
        emul.set_s(S);
        emul.run(d_A, d_B, d_D, d_ws, ws_size, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Download workspace sections */
        std::vector<int8_t>  h_A8i(szA8i), h_B8i(szB8i);
        std::vector<int32_t> h_C32i(szC32i_total);
        std::vector<double>  h_D(N2);
        std::vector<double2> h_dd(N2);

        HIP_CHECK(hipMemcpy(h_A8i.data(), d_ws,
                            szA8i * sizeof(int8_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_B8i.data(),
                            static_cast<char*>(d_ws) + szA8i * sizeof(int8_t),
                            szB8i * sizeof(int8_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_C32i.data(),
                            static_cast<char*>(d_ws) + (szA8i + szB8i) * sizeof(int8_t),
                            szC32i_total * sizeof(int32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_D.data(), d_D, N2 * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_dd.data(), d_dd_ref, N2 * sizeof(double2), hipMemcpyDeviceToHost));

        /* First: verify layout by checking PASSING elements */
        std::fprintf(stdout, "\n--- Layout sanity check: PASSING elements (row=0,col=0 and row=1,col=1) ---\n");
        std::fprintf(stdout, "%-4s  %8s  %12s  %12s  OK?\n", "mod", "actual", "exp(dot)", "diff");
        for(int check = 0; check <= 1; ++check) {
            int pr = check, pc = check;
            std::fprintf(stdout, "(row=%d,col=%d):\n", pr, pc);
            for(unsigned t = 0; t < S; ++t) {
                const int8_t* a_r = h_A8i.data() + t * szSlice + (size_t)pr * lda8i;
                const int8_t* b_c = h_B8i.data() + t * szSlice + (size_t)pc * ldb8i;
                int64_t exp_dot = 0;
                for(int k = 0; k < N; ++k) exp_dot += (int64_t)a_r[k] * (int64_t)b_c[k];
                const int32_t act = h_C32i[t * szC32i + pr + pc * ldc32i];
                std::fprintf(stdout, "  mod=%u actual=%8d exp=%12lld diff=%lld%s\n",
                    t, act, (long long)exp_dot, (long long)act - exp_dot,
                    (act != (int32_t)exp_dot) ? " <MISMATCH" : "");
            }
        }

        /* Known failing elements */
        const int fail_rows[] = {94, 48, 70};
        const int fail_cols[] = {37, 58, 106};
        const int N_FAIL = 3;

        /* For each failing element: print modulus-by-modulus diagnosis */
        for(int fi = 0; fi < N_FAIL; ++fi) {
            const int row = fail_rows[fi];
            const int col = fail_cols[fi];
            const size_t out_idx = (size_t)row + (size_t)col * N;   /* col-major */
            const double actual_D = h_D[out_idx];
            const double ref_D    = h_dd[out_idx].x;
            const double relerr   = std::fabs(actual_D - ref_D) / std::fabs(ref_D);

            std::fprintf(stdout,
                "\n--- Failing element (row=%d, col=%d)  "
                "D_emul=%.5e  D_ref=%.5e  relerr=%.3e ---\n",
                row, col, actual_D, ref_D, relerr);
            std::fprintf(stdout,
                "%-4s  %8s  %12s  %12s  %5s\n",
                "mod", "actual", "exp(dot)", "diff", "WRONG?");

            bool found_bad_gemm   = false;
            bool found_bad_a8i    = false;

            for(unsigned t = 0; t < S; ++t) {
                /* A8i[t][k][row]: A8i layout = [t][k + row*lda8i] */
                const int8_t* a_row = h_A8i.data() + t * szSlice + (size_t)row * lda8i;
                /* B8i[t][k][col]: B8i layout = [t][k + col*ldb8i] */
                const int8_t* b_col = h_B8i.data() + t * szSlice + (size_t)col * ldb8i;

                /* Compute expected C32i = dot(A8i_row, B8i_col) on CPU */
                int64_t expected_dot = 0;
                for(int k = 0; k < N; ++k)
                    expected_dot += (int64_t)a_row[k] * (int64_t)b_col[k];

                /* Actual C32i[t][row + col*ldc32i] */
                const int32_t actual_c32 =
                    h_C32i[t * szC32i + (size_t)row + (size_t)col * ldc32i];

                const int64_t diff = (int64_t)actual_c32 - expected_dot;
                const bool    bad  = (diff != 0);

                if(bad) {
                    std::fprintf(stdout,
                        "%-4u  %8d  %12lld  %12lld  <-- MISMATCH\n",
                        t, actual_c32, (long long)expected_dot, (long long)diff);
                    found_bad_gemm = true;

                    /* Check if A8i or B8i looks suspicious */
                    int nz_a = 0, nz_b = 0;
                    for(int k = 0; k < N; ++k) {
                        if(a_row[k] != 0) ++nz_a;
                        if(b_col[k] != 0) ++nz_b;
                    }
                    std::fprintf(stdout,
                        "       A8i[%d] nonzero: %d/%d, "
                        "B8i[%d] nonzero: %d/%d\n",
                        t, nz_a, N, t, nz_b, N);
                } else {
                    std::fprintf(stdout,
                        "%-4u  %8d  %12lld  %12lld\n",
                        t, actual_c32, (long long)expected_dot, (long long)diff);
                }
            }
            if(!found_bad_gemm)
                std::fprintf(stdout,
                    "  All C32i values correct → bug is in CRT reconstruction\n");
        }

        emul.destroy();
        HIP_CHECK(hipFree(d_dd_ref));
        HIP_CHECK(hipFree(d_ws));
        HIP_CHECK(hipFree(d_D));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_A));
    }

    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
