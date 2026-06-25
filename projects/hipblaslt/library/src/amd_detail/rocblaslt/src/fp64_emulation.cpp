// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/*
 * fp64_emulation.cpp
 *
 * FP64 GEMM emulation via Ozaki Scheme II (accurate mode) using INT8 Tensor Cores.
 *
 * Algorithm (paper: Ozaki, Uchino, Imamura, arXiv:2504.08009)
 * -----------------------------------------------------------
 * Given  D = alpha * op(A) * op(B) + beta * C   (A,B,C,D in FP64)
 *
 * Part 1 – Accurate scaling (OS II-accu):
 *   1a. Per-row 6-bit extraction of op(A) → A8i_high, with per-row shifts sftA[i].
 *   1b. Per-col 6-bit extraction of op(B) → B8i_high, with per-col shifts sftB[j].
 *   1c. Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high  (one extra GEMM).
 *   1d. Refine sftA[i] from per-row max of |C32i_prelim| (tighter bound than Cauchy-Schwarz).
 *   1e. Refine sftB[j] from per-col max of |C32i_prelim|.
 *   1f. Final scaling: A8i[t], B8i[t] for t=0..num_moduli-1 using refined sftA, sftB.
 *
 * Part 2 – For each of s coprime moduli m_t (including implicit m_0=256):
 *   2a.  A'_t = symmetric_mod(A', m_t)  → INT8  in [-m_t/2, m_t/2]
 *   2b.  B'_t = symmetric_mod(B', m_t)  → INT8
 *   2c.  C'_t = A'_t × B'_t             → INT32  (INT8 tensor cores)
 *   2d.  Z    += C'_t * qPi_t           (CRT accumulation, double-double)
 *
 * Part 3 – Range reduction: X = Z mod M  (unique because |X| < M/2)
 *
 * Part 4 – Per-element inverse scale: D[i,j] = alpha * X[i,j] * 2^-(sftA[i]+sftB[j])
 *                                              + beta * C[i,j]
 *
 * The number of moduli s (= number of INT8 GEMMs) is configurable at runtime via
 * HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT (default: s=16, ~125 bits of CRT
 * capacity, sufficient for guaranteed FP64-equivalent results on all inputs).
 *
 * Constants (tables) are taken verbatim from the open-source GEMMul8 implementation
 * (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8).
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 * Inner INT8 GEMMs use hipblasLtMatmul (INT8 tensor cores, INT32 accumulate).
 */

#include "fp64_emulation.hpp"
#include "handle.h"   /* _rocblaslt_handle */

#include "hipblaslt/hipblaslt.h"
#include <hip/hip_runtime.h>

#include <cstdlib>      // std::getenv
#include <cstdio>       // std::fopen / std::fprintf / std::fclose / std::ftell
#include <cstring>      // std::strcmp
#include <cmath>        // std::log2, std::floor, etc.
#include <cerrno>
#include <climits>
#include <atomic>
#include <optional>     // std::optional
#include <unordered_map>// std::unordered_map

/* =========================================================================
 * Tuning constants
 * ========================================================================= */
/* Maximum number of moduli supported (s = 2..OZ2_S_MAX).
 * Capped at 18 (log2(M) ≈ 140 bits, far more than needed for any practical
 * FP64 input).  This keeps the symmetric modular reduction to 2 FP32
 * refinement passes for both A and B, matching the GEMMul8 reference
 * implementation, and eliminates the runtime-fallback scale kernel.          */
static constexpr unsigned OZ2_S_MAX = 18;

/* Alignment for INT8 arrays (128 bytes = 128 INT8 elements) */
static constexpr size_t OZ2_ALIGN = 128;

static __host__ __device__ size_t oz2_pad(size_t n)
{
    return (n + OZ2_ALIGN - 1) / OZ2_ALIGN * OZ2_ALIGN;
}

static constexpr size_t OZ2_CHUNK_TARGET_BYTES = 8ull << 30;   /* 8 GiB    */
static constexpr size_t OZ2_SCALE_CHUNK_TARGET_BYTES = 8ull << 30;  /* 8 GiB */

static unsigned oz2_compute_chunk_size(int64_t m, int64_t n, unsigned num_moduli)
{
    const size_t mn4 = static_cast<size_t>(m) * static_cast<size_t>(n) * 4u;
    if(mn4 == 0u) return num_moduli;
    const size_t k = std::max(size_t(1), OZ2_CHUNK_TARGET_BYTES / mn4);
    return static_cast<unsigned>(std::min(static_cast<size_t>(num_moduli), k));
}

static unsigned oz2_compute_scale_chunk_size(int64_t m, int64_t n, int64_t k,
                                              unsigned num_moduli,
                                              unsigned gemm_chunk_sz)
{
    const size_t lda8i       = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i      = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i       = lda8i;
    const size_t slice_bytes = lda8i * cola8i + ldb8i * static_cast<size_t>(n);
    if(slice_bytes == 0u) return num_moduli;
    const size_t s = std::min(static_cast<size_t>(num_moduli),
                              std::max(static_cast<size_t>(gemm_chunk_sz),
                                       OZ2_SCALE_CHUNK_TARGET_BYTES / slice_bytes));
    return static_cast<unsigned>(s);
}

/* =========================================================================
 * GPU-side constant memory
 * ========================================================================= */
static __constant__ double cNegMod[OZ2_S_MAX];
static __constant__ double cInvMod[OZ2_S_MAX];
static __constant__ float  cInvModF[OZ2_S_MAX];
static __constant__ double cQpiHi[OZ2_S_MAX];
static __constant__ double cQpiLo[OZ2_S_MAX];
static __constant__ double cP_hi;
static __constant__ double cP_lo;
static __constant__ double cInvP;

/* =========================================================================
 * Host-side tables (source: GEMMul8/GEMMul8/src/table.hpp)
 *
 * Moduli in order: 256 (implicit), 255, 253, 251, 247, 241, 239, 233,
 *                  229, 227, 223, 217, 211, 199, 197, 193, 191, 181
 *                  (OZ2_S_MAX = 18 total)
 *
 * All arrays indexed by table_idx = s - 2  (s = number of moduli, 2..18).
 * ========================================================================= */
static const double h_neg_mod[OZ2_S_MAX] = {
    -256.0, -255.0, -253.0, -251.0, -247.0, -241.0, -239.0,
    -233.0, -229.0, -227.0, -223.0, -217.0, -211.0, -199.0,
    -197.0, -193.0, -191.0, -181.0
};
static const double h_inv_mod[OZ2_S_MAX] = {
    0x1.0000000000000p-8,   /* 1/256 */
    0x1.0101010101010p-8,   /* 1/255 */
    0x1.03091b51f5e1ap-8,   /* 1/253 */
    0x1.05197f7d73404p-8,   /* 1/251 */
    0x1.0953f39010954p-8,   /* 1/247 */
    0x1.0fef010fef011p-8,   /* 1/241 */
    0x1.12358e75d3033p-8,   /* 1/239 */
    0x1.19453808ca29cp-8,   /* 1/233 */
    0x1.1e2ef3b3fb874p-8,   /* 1/229 */
    0x1.20b470c67c0d9p-8,   /* 1/227 */
    0x1.25e22708092f1p-8,   /* 1/223 */
    0x1.2e025c04b8097p-8,   /* 1/217 */
    0x1.3698df3de0748p-8,   /* 1/211 */
    0x1.49539e3b2d067p-8,   /* 1/199 */
    0x1.4cab88725af6ep-8,   /* 1/197 */
    0x1.5390948f40febp-8,   /* 1/193 */
    0x1.571ed3c506b3ap-8,   /* 1/191 */
    0x1.6a13cd1537290p-8    /* 1/181 */
};
static const float h_inv_mod_f[OZ2_S_MAX] = {
    0x1.000000p-8F,   /* 1/256  (exact) */
    0x1.010102p-8F,   /* 1/255  */
    0x1.03091cp-8F,   /* 1/253  */
    0x1.051980p-8F,   /* 1/251  */
    0x1.0953f4p-8F,   /* 1/247  */
    0x1.0fef02p-8F,   /* 1/241  */
    0x1.12358ep-8F,   /* 1/239  */
    0x1.194538p-8F,   /* 1/233  */
    0x1.1e2ef4p-8F,   /* 1/229  */
    0x1.20b470p-8F,   /* 1/227  */
    0x1.25e228p-8F,   /* 1/223  */
    0x1.2e025cp-8F,   /* 1/217  */
    0x1.3698e0p-8F,   /* 1/211  */
    0x1.49539ep-8F,   /* 1/199  */
    0x1.4cab88p-8F,   /* 1/197  */
    0x1.539094p-8F,   /* 1/193  */
    0x1.571ed4p-8F,   /* 1/191  */
    0x1.6a13cep-8F    /* 1/181  */
};

/* -M (high part) for s = 2..18 */
static const double h_P_hi_all[OZ2_S_MAX - 1] = {
    -6.5280000000000000e+04,     /* s=2  */
    -1.6515840000000000e+07,     /* s=3  */
    -4.1454758400000000e+09,     /* s=4  */
    -1.0239325324800000e+12,     /* s=5  */
    -2.4676774032768000e+14,     /* s=6  */
    -5.8977489938315520e+16,     /* s=7  */
    -1.3741755155627516e+19,     /* s=8  */
    -3.1468619306387012e+21,     /* s=9  */
    -7.1433765825498518e+23,     /* s=10 */
    -1.5929729779086169e+26,     /* s=11 */
    -3.4567513620616985e+28,     /* s=12 */
    -7.2937453739501847e+30,     /* s=13 */
    -1.4514553294160867e+33,     /* s=14 */
    -2.8593669989496909e+35,     /* s=15 */
    -5.5185783079729035e+37,     /* s=16 */
    -1.0540484568228245e+40,     /* s=17 */
    -1.9078277068493124e+42,     /* s=18 */
};
static const double h_P_lo_all[OZ2_S_MAX - 1] = {
     0.0,                        /* s=2  */
     0.0,                        /* s=3  */
     0.0,                        /* s=4  */
     0.0,                        /* s=5  */
     0.0,                        /* s=6  */
     0.0,                        /* s=7  */
    -2.5600000000000000e+02,     /* s=8  */
     3.1488000000000000e+04,     /* s=9  */
     4.5263360000000000e+06,     /* s=10 */
    -2.6145057280000000e+09,     /* s=11 */
    -2.0448164928000000e+12,     /* s=12 */
     3.3380381295129600e+14,     /* s=13 */
     1.0131963435176704e+16,     /* s=14 */
     1.7272206732770533e+19,     /* s=15 */
     3.2597489231298749e+21,     /* s=16 */
    -2.5574812149594794e+23,     /* s=17 */
     4.6796878119559867e+25,     /* s=18 */
};
static const double h_inv_P_all[OZ2_S_MAX - 1] = {
    1.5318627450980392e-05,      /* s=2  */
    6.0547934588855299e-08,      /* s=3  */
    2.4122683103129606e-10,      /* s=4  */
    9.7662684628055072e-13,      /* s=5  */
    4.0523935530313311e-15,      /* s=6  */
    1.6955621560800549e-17,      /* s=7  */
    7.2770907986268441e-20,      /* s=8  */
    3.1777689076973120e-22,      /* s=9  */
    1.3998981972234855e-24,      /* s=10 */
    6.2775703911367061e-27,      /* s=11 */
    2.8928895811689891e-29,      /* s=12 */
    1.3710377161938337e-31,      /* s=13 */
    6.8896367647931339e-34,      /* s=14 */
    3.4972775455802713e-36,      /* s=15 */
    1.8120609044457363e-38,      /* s=16 */
    9.4872298662080431e-41,      /* s=17 */
    5.2415634619933945e-43,      /* s=18 */
};
/* log2P values for the shift-refinement formula:
 *   sft_delta = floor(-0.5 * log2(amax) + log2P)
 *
 * All entries use GEMMul8 'fast' values: fast::log2P = fld(log2(M-1)/2 - 1.5),
 * exactly 1.0 below the 'accu' values: accu::log2P = fld(log2(M-1)/2 - 0.5).
 *
 * Rationale: for every s, there exists a constructible FP64 input for which
 * the accu::log2P formula pushes a specific element's
 *   X_true = D × 2^{sftA[row]+sftB[col]}
 * above M_s/2 (the OZ2 CRT uniqueness bound), returning the wrong sign.
 * Using fast::log2P for all s guarantees a safety margin of ≥ 2 bits between
 * X_true_max and M_s/2 for any valid FP64 input, without any assumption on the
 * distribution or magnitude of A and B.
 *
 * The preliminary GEMM still provides full per-row/col adaptive scaling via the
 * data-dependent sft_delta = floor(-0.5 × log2(row_max) + log2P); only the
 * global constant log2P is reduced by 1 bit.
 *
 * Values match GEMMul8's table::fast::log2P (RIKEN GEMMul8 reference). */
static const float h_accu_log2P_all[OZ2_S_MAX - 1] = {
    6.49716520e+00F,   /* s=2  — fast */
    1.04886732e+01F,   /* s=3  — fast */
    1.44744443e+01F,   /* s=4  — fast */
    1.84486274e+01F,   /* s=5  — fast */
    2.24050731e+01F,   /* s=6  — fast */
    2.63555068e+01F,   /* s=7  — fast */
    3.02875995e+01F,   /* s=8  — fast */
    3.42071990e+01F,   /* s=9  — fast */
    3.81204757e+01F,   /* s=10 — fast */
    4.20209236e+01F,   /* s=11 — fast */
    4.59016990e+01F,   /* s=12 — fast */
    4.97622489e+01F,   /* s=13 — fast */
    5.35805625e+01F,   /* s=14 — fast */
    5.73915863e+01F,   /* s=15 — fast */
    6.11878166e+01F,   /* s=16 — fast */
    6.49765319e+01F,   /* s=17 — fast */
    6.87264480e+01F,   /* s=18 — fast */
};

static const double h_qpi_hi_all[OZ2_S_MAX - 1][OZ2_S_MAX] = {
    /* s=2  (qPi_1[0]) */
    {0x1.fc02000000000p+15, 0x1.0000000000000p+8,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=3  (qPi_1[1]) */
    {0x1.50ac020000000p+23, 0x1.f60c000000000p+22, 0x1.a45a000000000p+23,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=4  (qPi_1[2]) */
    {0x1.0688601000000p+28, 0x1.f01e000000000p+28, 0x1.4826900000000p+28,
     0x1.6654440000000p+31,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=5  (qPi_1[3]) */
    {0x1.99c1435808000p+37, 0x1.d553914600000p+39, 0x1.cf9d0d8400000p+38,
     0x1.2ff09e4000000p+38, 0x1.dae0172c00000p+39,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=6  (qPi_1[4]) */
    {0x1.24d0f0aa6c020p+47, 0x1.00ffb685c4000p+47, 0x1.7820600df8000p+45,
     0x1.b28fb528de000p+47, 0x1.765c060a1c000p+47, 0x1.56b441a210000p+47,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=7  (qPi_1[5]) */
    {0x1.49071d4742060p+55, 0x1.5fae947039b40p+55, 0x1.42fdb9e1948e0p+55,
     0x1.187c8ee783700p+55, 0x1.e89ef222a1c00p+52, 0x1.0316493fe27a0p+55,
     0x1.1f8e561d65780p+53,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=8  (qPi_2[0]) */
    {0x1.4f3952ae32400p+63, 0x1.f094cf17cf000p+61, 0x1.0f5bef8d36400p+63,
     0x1.e02e9274c5000p+62, 0x1.a403bd5c1a000p+61, 0x1.a1cf7b99c2800p+62,
     0x1.a54e8a8f42000p+60, 0x1.787fdcb9fa000p+62,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=9  (qPi_2[1]) */
    {0x1.9a7c80fe96000p+69, 0x1.43ca2f89db000p+71, 0x1.40f4871424000p+70,
     0x1.2c6790ef15000p+71, 0x1.24d66e4d76000p+70, 0x1.459c5b1ee5800p+71,
     0x1.d43c2b2519000p+70, 0x1.ab93da2aca000p+70, 0x1.dfbe1fda93000p+70,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=10 (qPi_2[2]) */
    {0x1.1ba01a9548000p+75, 0x1.b499060d20000p+76, 0x1.8d00367a82000p+77,
     0x1.348f721e1e000p+77, 0x1.09c9ed1acf000p+79, 0x1.6988bc8c28000p+75,
     0x1.4e2df779b8000p+77, 0x1.54302cc6b7000p+78, 0x1.675767107c000p+76,
     0x1.1fdfa04826000p+77,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=11 (qPi_2[3]) */
    {0x1.ae4dbe76d7000p+86, 0x1.258185fdee000p+86, 0x1.76fdabbf54000p+85,
     0x1.73ade1f823000p+86, 0x1.0cdeb7fb80000p+85, 0x1.0671178918000p+87,
     0x1.c416fd0741000p+86, 0x1.5350d862f8000p+86, 0x1.52567e0ff5000p+86,
     0x1.d0611c1cae000p+85, 0x1.814201f9be000p+86,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=12 (qPi_2[4]) */
    {0x1.42dd4f0c25000p+94, 0x1.71af2232d1000p+94, 0x1.b5f1f25063000p+93,
     0x1.0e8e8784ac000p+93, 0x1.0477c23ba5000p+93, 0x1.ac3c7c8760800p+94,
     0x1.507ba57edc000p+92, 0x1.2b20ca473f000p+93, 0x1.5f2d33fd22000p+92,
     0x1.ab17cae65c800p+94, 0x1.408e48b610000p+90, 0x1.32c582e2cf000p+94,
     0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=13 (qPi_2[5]) */
    {0x1.187ecea5a8800p+102, 0x1.71af223280000p+94,  0x1.5a685a078a000p+102,
     0x1.48a0e93cba000p+102, 0x1.6d422253da000p+102, 0x1.ec015f50a0000p+101,
     0x1.27d31b1920000p+99,  0x1.7b4d942fe0000p+100, 0x1.68332a1fe8000p+101,
     0x1.7859de7afc000p+99,  0x1.317d98db46800p+102, 0x1.08b9be1306800p+102,
     0x1.411e88bd34000p+100, 0.0,0.0,0.0,0.0,0.0},
    /* s=14 (qPi_2[6]) */
    {0x1.4af9bb23b8000p+107, 0x1.e0730f7df3000p+109, 0x1.9e197740a0000p+109,
     0x1.11b44daf38000p+106, 0x1.959dba1ed5000p+109, 0x1.d3f9c70059000p+109,
     0x1.c71fc39610000p+108, 0x1.6e1a9ef495000p+109, 0x1.067fc962e0800p+110,
     0x1.81de6aed04000p+109, 0x1.086d6ad9bc800p+110, 0x1.66ccfaf43f000p+109,
     0x1.d2ae54e567000p+109, 0x1.98842ba66f000p+109,
     0.0,0.0,0.0,0.0},
    /* s=15 (qPi_2[7]) */
    {0x1.8334edf0c0800p+117, 0x1.d9618469e1000p+116, 0x1.4c97d49af8800p+117,
     0x1.3db0f47816800p+117, 0x1.ac11e30d56000p+116, 0x1.d3f9c70000000p+109,
     0x1.0210da6024000p+117, 0x1.2e86f6e52b000p+116, 0x1.f43197eee2000p+115,
     0x1.e913152bf0000p+115, 0x1.775c686f24000p+116, 0x1.44d556f611000p+116,
     0x1.90e2677038000p+115, 0x1.1b5f498bca000p+117, 0x1.9702ab51fa000p+116,
     0.0,0.0,0.0},
    /* s=16 (qPi_2[8]) */
    {0x1.568442b104000p+122, 0x1.23c286bfdb000p+125, 0x1.fffd89ae2f000p+124,
     0x1.9f80a3facf000p+124, 0x1.6b10abb2b0000p+124, 0x1.b90322c900000p+119,
     0x1.ff687bb9b9000p+124, 0x1.494950989a000p+125, 0x1.5c176f9414000p+122,
     0x1.6dca3fa2e7000p+124, 0x1.951e4290e0000p+122, 0x1.a671255128000p+123,
     0x1.b2745cf9ae000p+124, 0x1.2c6cfd90da000p+123, 0x1.a57e7d4e8e000p+124,
     0x1.8f40d0ef24000p+124,
     0.0,0.0},
    /* s=17 (qPi_2[9]) */
    {0x1.e01f9407c4000p+129, 0x1.e201959d63000p+131, 0x1.31982160c4000p+132,
     0x1.7f0fe22eef000p+132, 0x1.00d5bf9f80000p+126, 0x1.8ad801f1a0000p+129,
     0x1.2a9c662802000p+130, 0x1.d836977997000p+131, 0x1.85903a5f3c000p+132,
     0x1.a3320451ba800p+132, 0x1.ce462d2242000p+132, 0x1.d67cf11ca9800p+132,
     0x1.add7c7ba40000p+132, 0x1.57b0afae95000p+131, 0x1.e30840c0e8000p+128,
     0x1.5aabc9d4bf800p+132, 0x1.82a0ee308b800p+132,
     0.0},
    /* s=18 (qPi_2[10]) */
    {0x1.06cf388320000p+134, 0x1.a1bf2dfdc0000p+136, 0x1.bb35a9d83c000p+137,
     0x1.b0c7cfa209000p+139, 0x1.4921eae073800p+140, 0x1.172ab95fd6000p+139,
     0x1.68acfd38e8000p+139, 0x1.f34ce4f4e8000p+138, 0x1.01123dfc72000p+140,
     0x1.9db3f73893000p+139, 0x1.f6d5907a7e000p+138, 0x1.e7abc6d98b000p+139,
     0x1.8e92d65018000p+136, 0x1.1d42b11e83800p+140, 0x1.0579b3ad70800p+140,
     0x1.0cb5cec87c000p+138, 0x1.2009162ca2800p+140, 0x1.3d803cbad1800p+140},
};

static const double h_qpi_lo_all[OZ2_S_MAX - 1][OZ2_S_MAX] = {
    /* s=2..7: lo = 0 */
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=8  (qPi_2[0]) */
    {0x1.16f0100000000p+20, 0x1.89a0000000000p+19, 0x1.8880000000000p+19,
     0x1.d740000000000p+19, 0x1.0b80000000000p+19, 0x1.2880000000000p+19,
     0x1.bcf0000000000p+20, 0x1.2d80000000000p+17,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=9  (qPi_2[1]) */
    {0x1.008cc04000000p+26, 0x1.eca4600000000p+28, 0x1.9a00780000000p+29,
     0x1.e855180000000p+29, 0x1.e9c7f00000000p+29, 0x1.38caf00000000p+29,
     0x1.d6d0600000000p+29, 0x1.e459400000000p+27, 0x1.9cd8200000000p+27,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=10 (qPi_2[2]) */
    {0x1.c6c29fa008000p+37, 0x1.4ddc380000000p+30, 0x1.5e72640800000p+37,
     0x1.5939d00000000p+34, 0x1.acce161000000p+36, 0x1.d3148d7000000p+37,
     0x1.1bca621000000p+37, 0x1.be65b8a000000p+35, 0x1.43b8ee6000000p+36,
     0x1.940b60e000000p+36,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=11 (qPi_2[3]) */
    {0x1.c311739de0100p+44, 0x1.3f5c901690000p+45, 0x1.de7087e210000p+45,
     0x1.6bfc28bd30000p+44, 0x1.de9bee2d48000p+45, 0x1.5646b56780000p+45,
     0x1.5ee3b89260000p+43, 0x1.77449328c0000p+43, 0x1.2e0367d338000p+45,
     0x1.c1e3b22c60000p+45, 0x1.4dba603168000p+45,
     0.0,0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=12 (qPi_2[4]) */
    {0x1.f5cc036fee804p+50, 0x1.9502088c71500p+52, 0x1.f27fe97ac8c00p+52,
     0x1.6a7fd4fb91000p+50, 0x1.9f4e1d77bb800p+52, 0x1.541aed8de8f00p+52,
     0x1.cad9eee787600p+51, 0x1.b754bf1ae1c00p+51, 0x1.d1793fc3ce200p+51,
     0x1.f0f2278772b00p+52, 0x1.59f94c68de600p+52, 0x1.f1f52b3aa8500p+52,
     0.0,0.0,0.0,0.0,0.0,0.0},
    /* s=13 (qPi_2[5]) */
    {0x1.2a800bf67755ap+60, 0x1.459502088c715p+60, 0x1.73141ccb58410p+57,
     0x1.956a7a15d56e0p+60, 0x1.c5f9191e4aa91p+60, 0x1.c69660c475d7bp+60,
     0x1.1a2f5dd7b0278p+60, 0x1.f7e2f13271df4p+60, 0x1.008e6afbfbd20p+59,
     0x1.45eaf7cf70b15p+60, 0x1.b2d9a1321591ap+59, 0x1.2a3c04a60a8a8p+59,
     0x1.25d2c634e54f0p+57, 0.0,0.0,0.0,0.0,0.0},
    /* s=14 (qPi_2[6]) */
    {0x1.ed366131bfd87p+61, 0x1.2a2b688425b37p+67, 0x1.ea31249d190dbp+66,
     0x1.57f8ce0e05580p+65, 0x1.34d662a4fdd1cp+66, 0x1.5013290076958p+67,
     0x1.3e7351822d438p+66, 0x1.1ebdf25941f8bp+67, 0x1.89ef93ae85687p+67,
     0x1.84d8fc60d93d4p+68, 0x1.1340b8f1c34bfp+67, 0x1.590ded2a35e12p+68,
     0x1.34cf70f07ae33p+67, 0x1.d80e799d28f38p+68,
     0.0,0.0,0.0,0.0},
    /* s=15 (qPi_2[7]) */
    {0x1.a4b62a6fdb1e1p+75, 0x1.c75bd2f612d0fp+75, 0x1.3f90fd4ad5142p+74,
     0x1.9e3c7f45d92bfp+75, 0x1.c9413fbd969ffp+75, 0x1.6550132900769p+75,
     0x1.a05bb4379a4c1p+75, 0x1.dae820f5ffc00p+74, 0x1.6405781ac87d9p+75,
     0x1.175dbeffba9cdp+75, 0x1.fe04b43a93e73p+71, 0x1.67335f8e813b8p+73,
     0x1.1bfe09769edb0p+75, 0x1.f1ee6037f1f5dp+71, 0x1.0c08e68bbfe9cp+75,
     0.0,0.0,0.0},
    /* s=16 (qPi_2[8]) */
    {0x1.195bce21a4a4cp+82, 0x1.d368142940c54p+83, 0x1.6961d82d67d29p+81,
     0x1.19861fe645aefp+78, 0x1.4aa2ee5f58c0cp+82, 0x1.42e6d8398ebf0p+83,
     0x1.09590940ec246p+83, 0x1.4ed69939f54a5p+83, 0x1.bccd986816af6p+83,
     0x1.38fff5b887f40p+83, 0x1.8c2bed86953acp+81, 0x1.2544a485cce86p+81,
     0x1.c88c3ec2fb90fp+83, 0x1.a84f9682c93f7p+83, 0x1.0beb05e6abcdfp+81,
     0x1.aeb1b1661a570p+81,
     0.0,0.0},
    /* s=17 (qPi_2[9]) */
    {0x1.1e87e3b708c22p+90, 0x1.4160efcbeef78p+90, 0x1.4480003b19f81p+89,
     0x1.0b25d1ed6a121p+87, 0x1.747bf6c0d8b31p+90, 0x1.bcbc193dd346cp+88,
     0x1.fddf745f1ee5ap+88, 0x1.d1f525311dabfp+90, 0x1.1660b883eb1a4p+90,
     0x1.41dedd270b797p+88, 0x1.79125e4f2418ap+90, 0x1.0272e6220fc37p+90,
     0x1.d2e0f92de9773p+87, 0x1.ea083b704edc0p+90, 0x1.eba48f8e2a378p+90,
     0x1.a977e531befa8p+90, 0x1.2ed72602864a3p+88,
     0.0},
    /* s=18 (qPi_2[10]) */
    {0x1.928222f7c81d9p+98, 0x1.7691a1e475ec2p+97, 0x1.50297632195fep+97,
     0x1.08e60e4fc6baep+96, 0x1.3586f9a06cbf4p+98, 0x1.7a70fdd8b6610p+98,
     0x1.5e172302320e2p+98, 0x1.1c4557a753b6cp+98, 0x1.4622df275a365p+95,
     0x1.cf7c96f698830p+95, 0x1.ebd8e9c0e37a5p+97, 0x1.f1a79e989b457p+97,
     0x1.1c8ffe978c39ep+98, 0x1.c0c39f95f19abp+92, 0x1.f55e10b41e4a2p+97,
     0x1.a546c43a54205p+98, 0x1.c22d132ce1471p+97, 0x1.6f3d636bc541bp+95},
};

/* =========================================================================
 * One-time constant-memory initialisation
 * ========================================================================= */
static hipError_t oz2_init_constants(unsigned num_moduli)
{
    static unsigned done_for = 0u;
    if(done_for == num_moduli) return hipSuccess;
    const unsigned idx = num_moduli - 2;
#define OZ2_CHECK(expr) do { hipError_t _e=(expr); if(_e!=hipSuccess){done_for=0u;return _e;} } while(0)
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cNegMod),  h_neg_mod,   sizeof(h_neg_mod)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvMod),  h_inv_mod,   sizeof(h_inv_mod)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvModF), h_inv_mod_f, sizeof(h_inv_mod_f)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cQpiHi), h_qpi_hi_all[idx], num_moduli*sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cQpiLo), h_qpi_lo_all[idx], num_moduli*sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cP_hi),  &h_P_hi_all[idx],  sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cP_lo),  &h_P_lo_all[idx],  sizeof(double)));
    OZ2_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(cInvP),  &h_inv_P_all[idx], sizeof(double)));
#undef OZ2_CHECK
    done_for = num_moduli;
    return hipSuccess;
}

/* =========================================================================
 * Host-side emulation control functions
 * ========================================================================= */
static constexpr unsigned FP64_EMULATION_DEFAULT_NUM_MODULI = 16u;
static constexpr unsigned FP64_EMULATION_MAX_MANTISSA_BITS  = 140u;

static bool parse_unsigned_env(const char* value, unsigned max_value, unsigned* out)
{
    if(value == nullptr || *value == '\0' || value[0] == '+' || value[0] == '-')
        return false;

    errno       = 0;
    char* end   = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 0);
    if(errno != 0 || end == value || *end != '\0' || parsed > max_value)
        return false;

    *out = static_cast<unsigned>(parsed);
    return true;
}

Fp64EmulationEnvValue fp64EmulationParseEnabledEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "0") == 0) return {FP64_EMULATION_ENV_VALID, 0u};
    if(std::strcmp(value, "1") == 0) return {FP64_EMULATION_ENV_VALID, 1u};
    return {FP64_EMULATION_ENV_INVALID, 0u};
}

Fp64EmulationEnvValue fp64EmulationParseStrategyEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "performant") == 0)
        return {FP64_EMULATION_ENV_VALID,
                static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT)};
    if(std::strcmp(value, "eager") == 0)
        return {FP64_EMULATION_ENV_VALID,
                static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER)};
    return {FP64_EMULATION_ENV_INVALID, 0u};
}

Fp64EmulationEnvValue fp64EmulationParseSpecialValuesMaskEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0x3u};

    unsigned parsed = 0u;
    if(!parse_unsigned_env(value, UINT_MAX, &parsed))
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, parsed};
}

Fp64EmulationEnvValue fp64EmulationParseMantissaBitCountEnv(const char* value)
{
    if(value == nullptr) return {FP64_EMULATION_ENV_UNSET, 0u};

    unsigned parsed = 0u;
    if(!parse_unsigned_env(value, FP64_EMULATION_MAX_MANTISSA_BITS, &parsed))
        return {FP64_EMULATION_ENV_INVALID, 0u};
    return {FP64_EMULATION_ENV_VALID, parsed};
}

bool fp64EmulationIsValidMantissaBitCount(int value)
{
    return value >= -1 && value <= static_cast<int>(FP64_EMULATION_MAX_MANTISSA_BITS);
}

static const Fp64EmulationEnvValue& cached_enabled_env()
{
    static const Fp64EmulationEnvValue parsed =
        fp64EmulationParseEnabledEnv(std::getenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION"));
    return parsed;
}

static const Fp64EmulationEnvValue& cached_strategy_env()
{
    static const Fp64EmulationEnvValue parsed =
        fp64EmulationParseStrategyEnv(std::getenv("HIPBLASLT_EMULATION_STRATEGY"));
    return parsed;
}

static const Fp64EmulationEnvValue& cached_special_values_mask_env()
{
    static const Fp64EmulationEnvValue parsed = fp64EmulationParseSpecialValuesMaskEnv(
        std::getenv("HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK"));
    return parsed;
}

static const Fp64EmulationEnvValue& cached_mantissa_bit_count_env()
{
    static const Fp64EmulationEnvValue parsed = fp64EmulationParseMantissaBitCountEnv(
        std::getenv("HIPBLASLT_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"));
    return parsed;
}

bool fp64EmulationIsEnabled()
{
    const auto& env = cached_enabled_env();
    return env.state == FP64_EMULATION_ENV_VALID && env.value != 0u;
}

bool fp64EmulationPerformanceCheck(int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    if(device < 0 || device >= 64) return std::nullopt;
    auto& entry = oz2_device_params_cache[device];
    if(!entry) {
        int chip_id = 0;
        const hipError_t attr_err =
            hipDeviceGetAttribute(&chip_id,
                                  hipDeviceAttributePciChipId, device);
        const uint32_t pci_device_id = static_cast<uint32_t>(chip_id) & 0xFFFFu;
        auto it = oz2_hw_params_by_pci_id.find(pci_device_id);
        entry = (it != oz2_hw_params_by_pci_id.end())
              ? std::optional<Oz2PerfModelParams>{it->second}
              : std::optional<Oz2PerfModelParams>{};
    }
    return *entry;
}

/* =========================================================================
 * Performance-model predicted times
 * Returns all sub-times in milliseconds.  Used both for the profiling CSV
 * and (via comparison of t_total_ms vs t_native_ms) for the performance
 * heuristic in fp64EmulationPerformanceCheck.
 * ========================================================================= */
struct Fp64PerfModelTimes {
    double t_prelim_ms;      /* prelim kernel (shift + extraction)  */
    double t_prelim_gemm_ms; /* preliminary INT8 GEMM               */
    double t_refine_ms;      /* sft-refinement kernels              */
    double t_scale_ms;       /* multi-modulus scaling kernels       */
    double t_int8_gemms_ms;  /* all INT8 GEMMs                      */
    double t_accum_ms;       /* CRT accumulation / finalize kernels */
    double t_host_ms;        /* per-call host overhead              */
    double t_launch_ms;      /* kernel-launch overhead              */
    double t_fused_ms;       /* fused TN kernel (replaces scale+GEMM+accum when better) */
    double t_total_ms;       /* total predicted emulation time      */
    double t_native_ms;      /* predicted native FP64 DGEMM time    */
};

static Fp64PerfModelTimes fp64EmulationPerfModelTimes(bool tA, bool tB,
                                                      int64_t m, int64_t n, int64_t k,
                                                      unsigned num_moduli, int device)
{
    const auto hw_opt = oz2_get_perf_model_params(device);
    assert(hw_opt.has_value() &&
           "fp64EmulationPerfModelTimes called for a device not in oz2_hw_params_by_pci_id");
    const Oz2PerfModelParams& hw = *hw_opt;

    static constexpr double LATENCY_KERNEL = 5.0e-6;
    static constexpr double LATENCY_MATMUL = 10.0e-6;
    static constexpr double LATENCY_MEMSET = 2.0e-6;

    const double c0 = hw.latency / LATENCY_MATMUL;
    const double c1 = c0 * hw.ai;
    const double c2 = c1 * hw.ratio;
    static constexpr double CHUNK_BYTES_D       = static_cast<double>(OZ2_CHUNK_TARGET_BYTES);
    static constexpr double SCALE_CHUNK_BYTES_D = static_cast<double>(OZ2_SCALE_CHUNK_TARGET_BYTES);

    const double s   = static_cast<double>(num_moduli);
    const double mn  = static_cast<double>(m) * static_cast<double>(n);
    const double mk  = static_cast<double>(m) * static_cast<double>(k);
    const double kn  = static_cast<double>(k) * static_cast<double>(n);
    const double mnk = mn * static_cast<double>(k);

    const double chunk_sz = std::max(1.0, std::min(s, CHUNK_BYTES_D / (mn * 4.0)));
    const double n_chunks = std::ceil(s / chunk_sz);

    const double t_int8_gemm_bw = (mk + kn + 4.0 * mn) / HBM_BW;
    const double t_prelim_gemm  = std::max(2.0 * mnk / INT8_PEAK, t_int8_gemm_bw);
    const double t_prelim_kern  = (mk + kn) * 17.0 / HBM_BW;
    const double t_refine_kern  = mn * 8.0 / HBM_BW;
    const double t_scale_kern   = (mk + kn) * (8.0 + s) / HBM_BW;
    const double t_int8_gemms   = s * std::max(2.0 * mnk / INT8_PEAK, t_int8_gemm_bw);
    const double t_accum_kern  = mn * (4.0 * s + 32.0 * n_chunks - 16.0) / HBM_BW;
    const double t_launch = (5.0 + n_chunks) * LATENCY_KERNEL
                          + (1.0 + n_chunks) * LATENCY_MATMUL
                          + LATENCY_MEMSET;

    const double EFF_PRELIM_KERN = tA ? (tB ? 0.628 : 0.797) : (tB ? 0.550 : 0.622);
    const double EFF_SCALE_KERN  = tA ? (tB ? 0.401 : 0.461) : (tB ? 0.354 : 0.405);
    static constexpr double EFF_REFINE_KERN      = 0.511;
    static constexpr double EFF_ACCUM_KERN       = 0.775;
    static constexpr double OZ2_HOST_OVERHEAD_MS = 0.100;

    const double t_int8_bw     = (mk + kn + 4.0 * mn) / c0;
    const double t_prelim_kern = ((mk + kn) * 17.0 / c0 + 2.0 * LATENCY_KERNEL) / EFF_PRELIM_KERN;
    const double t_prelim_gemm = std::max(2.0 * mnk / c2, t_int8_bw) + LATENCY_MATMUL;
    const double t_refine_kern = (mn * 8.0 / c0 + 3.0 * LATENCY_KERNEL + LATENCY_MEMSET) / EFF_REFINE_KERN;
    const double t_scale_kern  = ((mk + kn) * (8.0 * n_scale_chunks + s) / c0 + 2.0 * n_scale_chunks * LATENCY_KERNEL) / EFF_SCALE_KERN;
    const double t_int8_gemms  = s * std::max(2.0 * mnk / c2, t_int8_bw) + n_chunks * LATENCY_MATMUL;
    const double t_accum_kern  = (mn * (4.0 * s + 32.0 * n_chunks - 16.0) / c0 + n_chunks * LATENCY_KERNEL) / EFF_ACCUM_KERN;
    const double t_host        = OZ2_HOST_OVERHEAD_MS * 1e-3;
    const double t_launch      = 0.0;   /* all launch overhead distributed into components above */

    /* Fused TN kernel: reads pre-computed INT8 A8i/B8i from workspace, performs
     * MFMA + CRT accumulation, writes FP64 D directly.  Scale runs separately.  */
    static constexpr double EFF_FUSED = 0.57;
    const double t_fused_bw   = (s * (mk + kn) + 16.0 * mn) / c0;  /* INT8 + FP64 C/D */
    const double t_fused_int8 = s * 2.0 * mnk / c2;                  /* MFMA              */
    const double t_fused_fp64 = s * 8.0 * mn / c1;                   /* CRT accum only    */
    const double t_fused_cmp  = t_fused_int8 + t_fused_fp64;
    const double t_fused = std::max(t_fused_bw, t_fused_cmp) / EFF_FUSED + LATENCY_KERNEL;

    /* Scale always runs.  Fused kernel replaces only GEMM + CRT accum. */
    const double t_gemm_accum  = std::min(t_int8_gemms + t_accum_kern, t_fused);
    const double t_total       = t_prelim_kern + t_prelim_gemm + t_refine_kern
                               + t_scale_kern  + t_gemm_accum  + t_host;
    const double t_native      = std::max(2.0 * mnk / c1, 8.0 * (mk + kn + mn) / c0) + LATENCY_MATMUL;

    constexpr double s2ms = 1000.0;
    const double t_fused_ret = t_fused * s2ms;
    return { t_prelim_kern * s2ms, t_prelim_gemm * s2ms, t_refine_kern * s2ms,
             t_scale_kern  * s2ms, t_int8_gemms  * s2ms, t_accum_kern  * s2ms,
             t_host        * s2ms, t_launch      * s2ms,
             t_fused_ret,
             t_total       * s2ms, t_native      * s2ms };
}

/* Returns the minimum achievable emulation time in ms, accounting for the
 * recursive binary-halving that fp64EmulatedGemm applies when n_chunks > 1.
 * Both halves execute sequentially so the effective time is additive.
 *
 * The gate comparison uses the RECURSIVE effective times of each half
 * (not the flat perf-model times).  This correctly handles the case where
 * one split does not yet reduce n_chunks but further splitting would: the
 * recursive sub-call for the half discovers and accounts for those deeper
 * splits, returning the true best achievable time for that half.       */
static double oz2_effective_time_ms(bool tA, bool tB,
                                    int64_t m, int64_t n, int64_t k, unsigned s, int device)
{
    const double t_mono = fp64EmulationPerfModelTimes(tA, tB, m, n, k, s, device).t_total_ms;

    const unsigned chunk_sz = oz2_compute_chunk_size(m, n, s);
    const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
    if(n_chunks > 1u) {
        const bool    split_m = (m >= n);
        const int64_t half_m  = split_m ? m / 2 : m;
        const int64_t half_n  = split_m ? n     : n / 2;

        /* Both halves are nearly identical in size (differ by at most 1 when
         * m or n is odd), so approximate t_split = 2 × t_half.             */
        const double t_split = 2. * oz2_effective_time_ms(tA, tB, half_m, half_n, k, s, device);
        if(t_split <= t_mono * 1.01)
            return t_split;
    }
    return t_mono;
}

/* Returns per-component predicted times summed across ALL leaf sub-GEMMs,
 * mirroring the recursive binary-halving of oz2_effective_time_ms.
 * t_native_ms is always set to the top-level (m,n,k) native DGEMM time
 * because native DGEMM does not split.                                    */
static Fp64PerfModelTimes oz2_effective_perf_model_times(bool tA, bool tB,
                                                         int64_t m, int64_t n, int64_t k,
                                                         unsigned s, int device)
{
    Fp64PerfModelTimes mono = fp64EmulationPerfModelTimes(tA, tB, m, n, k, s, device);

    const unsigned chunk_sz = oz2_compute_chunk_size(m, n, s);
    const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
    if(n_chunks > 1u) {
        const bool    split_m = (m >= n);
        const int64_t half_m  = split_m ? m / 2 : m;
        const int64_t half_n  = split_m ? n     : n / 2;

        const double t_split = 2. * oz2_effective_time_ms(tA, tB, half_m, half_n, k, s, device);
        if(t_split <= mono.t_total_ms * 1.01) {
            /* Recurse on one half, then double all components.
             * Both halves are ≈ equal in size so the approximation is exact
             * when m (or n) is even and negligible otherwise.               */
            Fp64PerfModelTimes half = oz2_effective_perf_model_times(tA, tB, half_m, half_n, k, s, device);
            half.t_prelim_ms      *= 2.0;
            half.t_prelim_gemm_ms *= 2.0;
            half.t_refine_ms      *= 2.0;
            half.t_scale_ms       *= 2.0;
            half.t_int8_gemms_ms  *= 2.0;
            half.t_accum_ms       *= 2.0;
            half.t_host_ms        *= 2.0;
            half.t_launch_ms      *= 2.0;
            half.t_fused_ms       *= 2.0;
            half.t_total_ms       *= 2.0;
            /* Native DGEMM does not split: keep the top-level prediction. */
            half.t_native_ms = mono.t_native_ms;
            return half;
        }
    }
    return mono;
}

bool fp64EmulationPerformanceCheck(const _rocblaslt_handle* h,
                                   hipblasOperation_t opA, hipblasOperation_t opB,
                                   int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    const int  device = h->device;
    const bool tA     = (opA != HIPBLAS_OP_N);
    const bool tB     = (opB != HIPBLAS_OP_N);
    const double t_emul   = oz2_effective_time_ms(tA, tB, m, n, k, num_moduli, device);
    const double t_native = fp64EmulationPerfModelTimes(tA, tB, m, n, k, num_moduli, device).t_native_ms;
    return t_emul <= t_native;
}

bool fp64EmulationIsEager()
{
    const auto& env = cached_strategy_env();
    return env.state == FP64_EMULATION_ENV_VALID
           && env.value == static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER);
}

uint32_t fp64EmulationSpecialValuesMask()
{
    const auto& env = cached_special_values_mask_env();
    return env.state == FP64_EMULATION_ENV_VALID ? env.value : 0x3u;
}

static constexpr double oz2_cum_bits[OZ2_S_MAX - 1] = {
     15.994,  /* s=2  */  23.976,  /* s=3  */  31.945,  /* s=4  */
     39.894,  /* s=5  */  47.807,  /* s=6  */  55.708,  /* s=7  */
     63.572,  /* s=8  */  71.411,  /* s=9  */  79.238,  /* s=10 */
     87.040,  /* s=11 */  94.801,  /* s=12 */ 102.522,  /* s=13 */
    110.160,  /* s=14 */ 117.782,  /* s=15 */ 125.374,  /* s=16 */
    132.949,  /* s=17 */ 140.448,  /* s=18 */
};

static unsigned num_moduli_for_mantissa_bits(unsigned target)
{
    for(unsigned s = 2u; s <= OZ2_S_MAX; ++s)
        if(oz2_cum_bits[s - 2u] >= static_cast<double>(target)) return s;
    return OZ2_S_MAX;
}

static rocblaslt_status invalid_if_set(const Fp64EmulationEnvValue& env)
{
    return env.state == FP64_EMULATION_ENV_INVALID ? rocblaslt_status_invalid_value
                                                   : rocblaslt_status_success;
}

struct Fp64EmulationMantissaPolicy {
    rocblaslt_status status;
    unsigned int     num_moduli;
    bool             dynamic_mode;
};

static Fp64EmulationMantissaPolicy resolve_mantissa_policy(const _rocblaslt_handle* h)
{
    const auto& mantissa_env = cached_mantissa_bit_count_env();
    if((h->emulation.mantissa_control < 0
        || (h->emulation.mantissa_control == HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED
            && h->emulation.max_mantissa_bits < 0))
       && invalid_if_set(mantissa_env) != rocblaslt_status_success) {
        return {rocblaslt_status_invalid_value, 0u, false};
    }

    if(h->emulation.mantissa_control == HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED
       && h->emulation.max_mantissa_bits >= 0) {
        return {rocblaslt_status_success,
                num_moduli_for_mantissa_bits(static_cast<unsigned>(h->emulation.max_mantissa_bits)),
                false};
    }

    if(h->emulation.mantissa_control < 0
       && mantissa_env.state == FP64_EMULATION_ENV_VALID) {
        return {rocblaslt_status_success,
                num_moduli_for_mantissa_bits(mantissa_env.value),
                false};
    }

    return {rocblaslt_status_success, FP64_EMULATION_DEFAULT_NUM_MODULI, true};
}

unsigned fp64EmulationEffectiveNumModuli(const _rocblaslt_handle* h)
{
    const Fp64EmulationMantissaPolicy policy = resolve_mantissa_policy(h);
    return policy.status == rocblaslt_status_success ? policy.num_moduli
                                                     : FP64_EMULATION_DEFAULT_NUM_MODULI;
}

Fp64EmulationDecision fp64EmulationDecision(const _rocblaslt_handle* h,
                                            hipDataType              type_a,
                                            int64_t                  m,
                                            int64_t                  n,
                                            int64_t                  k,
                                            int                      batch_count)
{
    Fp64EmulationDecision result{rocblaslt_status_success, false, 0u, 0x3u, false};
    if(type_a != HIP_R_64F || batch_count != 1) return result;

    const auto& enabled_env = cached_enabled_env();
    bool emul_enabled = false;
    if(h->emulation.enabled == 0) {
        return result;
    } else if(h->emulation.enabled == 1) {
        emul_enabled = true;
    } else {
        result.status = invalid_if_set(enabled_env);
        if(result.status != rocblaslt_status_success) return result;
        emul_enabled = enabled_env.state == FP64_EMULATION_ENV_VALID && enabled_env.value;
    }
    if(!emul_enabled) return result;

    const auto& strategy_env = cached_strategy_env();
    result.status = invalid_if_set(strategy_env);
    if(result.status != rocblaslt_status_success) return result;

    const unsigned strategy = (strategy_env.state == FP64_EMULATION_ENV_VALID)
                                  ? strategy_env.value
                                  : static_cast<unsigned>(
                                        (h->emulation.strategy >= 0)
                                            ? h->emulation.strategy
                                            : HIPBLASLT_EMULATION_STRATEGY_PERFORMANT);

    const auto& mask_env = cached_special_values_mask_env();
    result.status = invalid_if_set(mask_env);
    if(result.status != rocblaslt_status_success) return result;
    result.sv_mask = (mask_env.state == FP64_EMULATION_ENV_VALID)
                         ? mask_env.value
                         : ((h->emulation.special_values_mask != ~0u)
                                ? h->emulation.special_values_mask
                                : 0x3u);

    const Fp64EmulationMantissaPolicy mantissa = resolve_mantissa_policy(h);
    result.status = mantissa.status;
    if(result.status != rocblaslt_status_success) return result;

    result.num_moduli   = mantissa.num_moduli;
    result.dynamic_mode = mantissa.dynamic_mode;
    result.apply = (strategy == static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER))
                   || fp64EmulationPerformanceCheck(m, n, k, result.num_moduli);
    return result;
}

void fp64EmulationWarnDynamicTemporary()
{
    static std::atomic<bool> warned{false};
    bool expected = false;
    if(warned.compare_exchange_strong(expected, true)) {
        std::fprintf(stderr,
                     "hipBLASLt FP64 emulation: dynamic mantissa control currently "
                     "uses the existing 16-moduli path; runtime ADP is not enabled yet.\n");
    }
}

/* =========================================================================
 * fp64EmulationWorkspaceSize
 * ========================================================================= */
size_t fp64EmulationWorkspaceSize(int64_t m, int64_t n, int64_t k, unsigned num_moduli)
{
    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);
    const unsigned gemm_sz  = oz2_compute_chunk_size(m, n, num_moduli);
    const unsigned scale_sz = oz2_compute_scale_chunk_size(m, n, k, num_moduli, gemm_sz);

    return   scale_sz * lda8i * cola8i * sizeof(int8_t)
           + scale_sz * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)
           + gemm_sz  * szC32i * sizeof(int32_t)
           + szC32i * sizeof(double) * 2
           + cola8i * sizeof(int16_t)
           + padn   * sizeof(int16_t)
           + sizeof(uint32_t)
           + cola8i * sizeof(int32_t);
}

unsigned fp64EmulationNumModuli()
{
    static const unsigned num_moduli = []() -> unsigned {
        const auto& env = cached_mantissa_bit_count_env();
        if(env.state != FP64_EMULATION_ENV_VALID) return FP64_EMULATION_DEFAULT_NUM_MODULI;
        const unsigned target = env.value;
        for(unsigned s = 2u; s <= OZ2_S_MAX; ++s)
            if(oz2_cum_bits[s - 2u] >= static_cast<double>(target)) return s;
        return OZ2_S_MAX;
    }();
    return num_moduli;
}

/* =========================================================================
 * Device helper: warp-level reductions
 * ========================================================================= */
static __device__ __forceinline__ double warp_reduce_max_abs_d(double val)
{
    val = fabs(val);
    unsigned long long bits; __builtin_memcpy(&bits, &val, 8);
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        unsigned long long other = __shfl_down(bits, off);
        if(other > bits) bits = other;
    }
    double res; __builtin_memcpy(&res, &bits, 8);
    return res;
}

static __device__ __forceinline__ int32_t warp_reduce_max_abs_i32(int32_t val)
{
    if(val < 0) val = -val;
    for(int off = warpSize >> 1; off > 0; off >>= 1) {
        int32_t other = __shfl_down(val, off);
        if(other > val) val = other;
    }
    return val;
}

static __device__ __forceinline__ double
block_reduce_max_d(double warp_max, double* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    double result = 0.0;
    if(threadIdx.x == 0) {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < nw; ++w) if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

static __device__ __forceinline__ int32_t
block_reduce_max_i32(int32_t warp_max, int32_t* __restrict__ s_wmax)
{
    if(threadIdx.x % warpSize == 0) s_wmax[threadIdx.x / warpSize] = warp_max;
    __syncthreads();
    int32_t result = 0;
    if(threadIdx.x == 0) {
        const int nw = (blockDim.x + warpSize - 1) / warpSize;
        result = s_wmax[0];
        for(int w = 1; w < nw; ++w) if(s_wmax[w] > result) result = s_wmax[w];
    }
    return result;
}

/* =========================================================================
 * GPU kernel — fused preliminary shift computation + INT8 extraction
 *
 * oz2_accu_prelim_kernel<TRANS_A, TRANS_B, CHECK_NAN>
 *
 * Fuses per-row/col shift computation (sftA, sftB) and preliminary INT8
 * extraction (A8i_high, B8i_high) into a SINGLE kernel dispatch, eliminating
 * the extra launch overhead of the previous two-kernel approach and achieving
 * coalesced HBM reads AND writes for all four (TRANS_A, TRANS_B) combinations.
 *
 *   TRANS_A=true  (op(A)=A^T, A stored k×m col-major): k-fast (threadIdx.x=j).
 *     Loop 1: read A[row*lda+j] (COALESCED) → block-reduce → sft.
 *     Loop 2: read A[row*lda+j] (COALESCED) → scale → write A8i_high[j+row*lda8i] (COALESCED).
 *
 *   TRANS_A=false (op(A)=A, A stored m×k col-major): tiled SHMEM.
 *     One block per TILE-row tile; two passes over k:
 *       Pass 1: load m-fast tiles into SHMEM (COALESCED), accumulate per-row max → sft.
 *       Pass 2: load m-fast tiles into SHMEM (COALESCED), scale, write A8i_high k-fast
 *               from SHMEM via SHMEM transposition (COALESCED).
 *
 *   TRANS_B=false / TRANS_B=true: symmetric to TRANS_A=true / TRANS_A=false.
 *
 * Grid  = dim3(m_blks + n_blks, 1)
 *   m_blks = TRANS_A ? m       : ceil(m / OZ2_PRELIM_TILE)
 *   n_blks = TRANS_B ? ceil(n / OZ2_PRELIM_TILE) : n
 * Block = dim3(OZ2_PRELIM_TILE * OZ2_PRELIM_TILE, 1) = dim3(256, 1)
 * ========================================================================= */
static constexpr int OZ2_PRELIM_TILE_K = 64;  /* k-tile size (reduces k-tile loop count) */
static constexpr int OZ2_PRELIM_TILE_M = 4;   /* rows/cols per tile (= blockDim.x / TILE_K) */
/* blockDim.x = TILE_K × TILE_M = 256 threads */

template <bool TRANS_A, bool TRANS_B, bool CHECK_NAN>
__global__ static void
oz2_accu_prelim_kernel(const double* __restrict__ A,
                        int64_t m, int64_t k, int64_t lda,
                        int8_t*  __restrict__  A8i_high, size_t lda8i,
                        int16_t* __restrict__  sftA,
                        const double* __restrict__ B,
                        int64_t n, int64_t ldb,
                        int8_t*  __restrict__  B8i_high, size_t ldb8i,
                        int16_t* __restrict__  sftB,
                        uint32_t* __restrict__ nan_flag,
                        unsigned m_blks)
{
    static constexpr int TILE_K = OZ2_PRELIM_TILE_K;  /* k-tile size (256/TILE_M iterations) */
    static constexpr int TILE_M = OZ2_PRELIM_TILE_M;  /* rows/cols per block */

    /* Shared memory:
     *   shmem[TILE_K][TILE_M+1] — FP64 tile (TILE_M+1 padding avoids bank conflicts)
     *   s_sft[TILE_M]           — per-row/col sft broadcast
     *   s_wmax[4]               — warp maxes for block_reduce_max_d (4 warps of 64)
     *
     * TILE_K=64, TILE_M=4 → 4× fewer k-tile iterations (k/64 vs k/16 for TILE=16),
     * 4× fewer __syncthreads() in Pass 2, 4× more A/B blocks → better occupancy. */
    __shared__ double  shmem[TILE_K][TILE_M + 1];
    __shared__ int16_t s_sft[TILE_M];
    __shared__ double  s_wmax[4];

    if(blockIdx.x < m_blks) {
        /* ── A block ────────────────────────────────────────────────────── */
        if constexpr (TRANS_A) {
            /* k-fast: one block per op(A) row; two k-loops.               */
            const int64_t row = static_cast<int64_t>(blockIdx.x);

            /* Loop 1: compute per-row max */
            double local_max = 0.0;
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val = A[row * lda + j];                     /* COALESCED */
                if constexpr (CHECK_NAN)
                    if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > local_max) local_max = av;
            }
            local_max = warp_reduce_max_abs_d(local_max);
            local_max = block_reduce_max_d(local_max, s_wmax);
            if(threadIdx.x == 0) {
                if(local_max < 1e-300) local_max = 1.0;
                s_sft[0] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(local_max))));
                sftA[row] = s_sft[0];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[0]);

            /* Loop 2: scale and write A8i_high */
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val    = A[row * lda + j];                  /* COALESCED */
                double scaled = ceil(ldexp(fabs(val), sft));
                A8i_high[static_cast<size_t>(j)
                         + static_cast<size_t>(row) * lda8i] =
                    static_cast<int8_t>(static_cast<int32_t>(scaled));  /* COALESCED */
            }

        } else {
            /* Tiled SHMEM: one block per TILE_M-row tile.
             * Thread t: k_local = t/TILE_M (k-index in tile, 0..TILE_K-1),
             *           m_local = t%TILE_M (row-index in tile, 0..TILE_M-1).
             * Adjacent threads (same k_local, consecutive m_local) access
             * consecutive rows of A → COALESCED loads.  k-tile iterations =
             * k/TILE_K (e.g. 16 for k=1024, TILE_K=64).                    */
            const int64_t m_base = static_cast<int64_t>(blockIdx.x) * TILE_M;
            const int t       = static_cast<int>(threadIdx.x);
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
            const int64_t i   = m_base + m_local;

            /* Pass 1: accumulate per-row max across all k-tiles */
            double thr_max = 0.0;
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;
                if(i < m && j < k) {
                    /* A[i + j*lda]: adjacent i (= m_local, varies) → COALESCED */
                    double val = A[i + j * lda];
                    if constexpr (CHECK_NAN)
                        if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                    double av = fabs(val);
                    if(av > thr_max) thr_max = av;
                }
            }
            /* Reduce across k_local dimension (TILE_K=64 values per row) */
            shmem[k_local][m_local] = thr_max;
            __syncthreads();
            if(k_local == 0) {   /* TILE_M=4 threads finalise, one per row */
                double row_max = 0.0;
                for(int kl = 0; kl < TILE_K; ++kl)
                    if(shmem[kl][m_local] > row_max) row_max = shmem[kl][m_local];
                if(row_max < 1e-300) row_max = 1.0;
                s_sft[m_local] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(row_max))));
                if(i < m) sftA[i] = s_sft[m_local];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[m_local]);  /* row-specific */

            /* Pass 2: coalesced loads (m-fast) → SHMEM → coalesced writes (k-fast)
             * k_tile iterations = k/TILE_K (32 syncs for k=2048 vs 256 with TILE=16) */
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;

                /* Load: A[i + j*lda], i=m_base+t%TILE_M varies → COALESCED */
                double scaled = 0.0;
                if(i < m && j < k)
                    scaled = ceil(ldexp(fabs(A[i + j * lda]), sft));
                shmem[k_local][m_local] = scaled;
                __syncthreads();

                /* Write: k_write = t%TILE_K varies fast → COALESCED */
                const int k_write = t % TILE_K;
                const int m_write = t / TILE_K;
                const int64_t j_out = k_base  + k_write;
                const int64_t i_out = m_base  + m_write;
                if(i_out < m && j_out < k)
                    A8i_high[static_cast<size_t>(j_out)
                             + static_cast<size_t>(i_out) * lda8i] =
                        static_cast<int8_t>(static_cast<int32_t>(
                            shmem[k_write][m_write]));
                __syncthreads();
            }
        }

    } else {
        /* ── B block (symmetric to A, with TRANS_B) ─────────────────── */
        if constexpr (!TRANS_B) {
            /* j-fast: one block per op(B) col, threadIdx.x = j */
            const int64_t col = static_cast<int64_t>(blockIdx.x - m_blks);

            /* Loop 1: compute per-col max */
            double local_max = 0.0;
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val = B[j + col * ldb];                     /* COALESCED */
                if constexpr (CHECK_NAN)
                    if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                double av = fabs(val);
                if(av > local_max) local_max = av;
            }
            local_max = warp_reduce_max_abs_d(local_max);
            local_max = block_reduce_max_d(local_max, s_wmax);
            if(threadIdx.x == 0) {
                if(local_max < 1e-300) local_max = 1.0;
                s_sft[0] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(local_max))));
                sftB[col] = s_sft[0];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[0]);

            /* Loop 2: scale and write B8i_high */
            for(int64_t j = threadIdx.x; j < k; j += blockDim.x) {
                double val    = B[j + col * ldb];                  /* COALESCED */
                double scaled = ceil(ldexp(fabs(val), sft));
                B8i_high[static_cast<size_t>(j)
                         + static_cast<size_t>(col) * ldb8i] =
                    static_cast<int8_t>(static_cast<int32_t>(scaled));  /* COALESCED */
            }

        } else {
            /* Tiled SHMEM for TRANS_B=T (B stored n×k, B[col,j]=B[col+j*ldb]).
             * One block per TILE_M-col tile.                                */
            const int64_t n_base = static_cast<int64_t>(blockIdx.x - m_blks) * TILE_M;
            const int t       = static_cast<int>(threadIdx.x);
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int l_local = t % TILE_M;   /* col-index within tile */
            const int64_t col = n_base + l_local;

            /* Pass 1: accumulate per-col max */
            double thr_max = 0.0;
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;
                if(col < n && j < k) {
                    /* B[col + j*ldb]: adjacent col (= l_local, varies) → COALESCED */
                    double val = B[col + j * ldb];
                    if constexpr (CHECK_NAN)
                        if(!isfinite(val)) (void)atomicOr(nan_flag, isinf(val) ? 1u : 2u);
                    double av = fabs(val);
                    if(av > thr_max) thr_max = av;
                }
            }
            shmem[k_local][l_local] = thr_max;
            __syncthreads();
            if(k_local == 0) {
                double col_max = 0.0;
                for(int kl = 0; kl < TILE_K; ++kl)
                    if(shmem[kl][l_local] > col_max) col_max = shmem[kl][l_local];
                if(col_max < 1e-300) col_max = 1.0;
                s_sft[l_local] = static_cast<int16_t>(
                    6 - static_cast<int>(floor(log2(col_max))));
                if(col < n) sftB[col] = s_sft[l_local];
            }
            __syncthreads();
            const int sft = static_cast<int>(s_sft[l_local]);

            /* Pass 2: coalesced loads (col-fast) → SHMEM → coalesced writes (k-fast) */
            for(int64_t k_base = 0; k_base < k; k_base += TILE_K) {
                const int64_t j = k_base + k_local;

                double scaled = 0.0;
                if(col < n && j < k)
                    scaled = ceil(ldexp(fabs(B[col + j * ldb]), sft));  /* COALESCED */
                shmem[k_local][l_local] = scaled;
                __syncthreads();

                const int k_write = t % TILE_K;
                const int l_write = t / TILE_K;
                const int64_t j_out   = k_base  + k_write;
                const int64_t col_out = n_base  + l_write;
                if(col_out < n && j_out < k)
                    B8i_high[static_cast<size_t>(j_out)
                             + static_cast<size_t>(col_out) * ldb8i] =
                        static_cast<int8_t>(static_cast<int32_t>(
                            shmem[k_write][l_write]));
                __syncthreads();
            }
        }
    }
}

/* =========================================================================
 * GPU kernels — accu mode Part 1: shift refinement from preliminary GEMM
 * ========================================================================= */
__global__ static void
oz2_refine_sftA_partial_kernel(const int32_t* __restrict__ C32i,
                                int64_t m, int64_t n, size_t ldc32i,
                                int32_t* __restrict__ row_max)
{
    const int64_t row      = static_cast<int64_t>(blockIdx.x) * 64
                           + static_cast<int64_t>(threadIdx.x);
    const int64_t col_base = static_cast<int64_t>(blockIdx.y) * 64;
    if(row >= m) return;
    int32_t local_max = 0;
    const int64_t col_end = (col_base + 64 < n) ? col_base + 64 : n;
    for(int64_t col = col_base; col < col_end; ++col) {
        int32_t v  = C32i[static_cast<size_t>(row) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    if(local_max > 0) atomicMax(row_max + static_cast<size_t>(row), local_max);
}

/* log2P is passed as a host-side float constant (from h_accu_log2P_all[s-2]).
 * For s=13,14,15 this is fast::log2P (1 bit below accu) to prevent the OZ2
 * CRT invariant |X_true| < M_s/2 from being violated by floor discretisation.
 * For all other s the full accu::log2P is used, preserving maximum precision. */
__global__ static void
oz2_refine_sftA_apply_kernel(const int32_t* __restrict__ row_max,
                              int16_t* __restrict__ sftA, int64_t m,
                              float log2P)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * 64
                      + static_cast<int64_t>(threadIdx.x);
    if(row >= m) return;
    int32_t max_val = row_max[row];
    if(max_val < 1) max_val = 1;
    sftA[row] += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(max_val)) + log2P));
}

__global__ static void
oz2_refine_sftB_kernel(const int32_t* __restrict__ C32i,
                       int64_t m, int64_t n, size_t ldc32i,
                       int16_t* __restrict__ sftB, float log2P)
{
    __shared__ int32_t s_wmax[8];
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if(col >= n) return;
    int32_t local_max = 0;
    for(int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        int32_t v  = C32i[static_cast<size_t>(i) + static_cast<size_t>(col) * ldc32i];
        int32_t av = v < 0 ? -v : v;
        if(av > local_max) local_max = av;
    }
    local_max = warp_reduce_max_abs_i32(local_max);
    local_max = block_reduce_max_i32(local_max, s_wmax);
    if(threadIdx.x == 0) {
        if(local_max < 1) local_max = 1;
        sftB[col] += static_cast<int16_t>(floorf(-0.5f * log2f(static_cast<float>(local_max)) + log2P));
    }
}

/* =========================================================================
 * GPU kernels — Part 1f: full multi-modulus scaling (A and B fused)
 *
 * oz2_scaleAB_kernel<T_COUNT, TRANS_A, TRANS_B>: compile-time template.
 *
 * Both A and B branches apply the same 2-pass symmetric modular reduction
 * (FP64 pass + 1 FP32 refinement pass), matching the GEMMul8 reference.
 * With OZ2_S_MAX=18 there is no need for a 3rd FP32 pass or a runtime-
 * fallback kernel.
 *
 * For TRANS_A=true / TRANS_B=false (coalesced reads):
 *   j = t%TILE_K varies fast within warp → stride-1 HBM reads (COALESCED).
 *
 * For TRANS_A=false / TRANS_B=true (non-coalesced reads):
 *   Tiled SHMEM transposition (same structure as oz2_accu_prelim_kernel):
 *     Load:  k_local=t/TILE_M, m_local=t%TILE_M → A[i+j*lda] (m_local fast → COALESCED)
 *     Store raw val in shmem[k_local][m_local]; store per-row sft in s_sft[m_local].
 *     __syncthreads()
 *     Write: k_write=t%TILE_K, m_write=t/TILE_K → shmem transposed read
 *            → A8i[j_out + i_out*lda8i + t_local*stride]  (k_write fast → COALESCED)
 *
 * Grid  = dim3(ceil(k/TILE_K), ceil(m/TILE_M) + ceil(n/TILE_M))  [unchanged]
 * Block = dim3(TILE_K × TILE_M) = dim3(256)                      [was dim3(64,4)]
 *   blockIdx.y <  m_y_blocks → A scaling
 *   blockIdx.y >= m_y_blocks → B scaling
 * ========================================================================= */
static constexpr unsigned OZ2_SCALE_TILE_K = 64;
static constexpr unsigned OZ2_SCALE_TILE_M = 4;

template <unsigned T_COUNT, bool TRANS_A, bool TRANS_B>
__global__ static void
oz2_scaleAB_kernel(const double* __restrict__ A,
                   int64_t m, int64_t lda,
                   int8_t*  __restrict__       A8i, size_t lda8i, size_t cola8i,
                   const int16_t* __restrict__ sftA,
                   const double* __restrict__ B,
                   int64_t n, int64_t ldb,
                   int8_t*  __restrict__       B8i, size_t ldb8i,
                   const int16_t* __restrict__ sftB,
                   int64_t k, unsigned t_start, unsigned m_y_blocks)
{
    static constexpr int TILE_K = static_cast<int>(OZ2_SCALE_TILE_K);
    static constexpr int TILE_M = static_cast<int>(OZ2_SCALE_TILE_M);

    /* SHMEM used only in non-coalesced paths (TRANS_A=false / TRANS_B=true).
     * Declared unconditionally; coalesced paths skip it via if constexpr.   */
    __shared__ double  shmem[TILE_K][TILE_M + 1];  /* +1 avoids bank conflicts */
    __shared__ int16_t s_sft[TILE_M];

    const int t = static_cast<int>(threadIdx.x);

    if(static_cast<unsigned>(blockIdx.y) < m_y_blocks) {
        /* ── A block ─────────────────────────────────────────────────────── */
        const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;

        if constexpr (TRANS_A) {
            /* Coalesced: A stored k×m, A[i,j] = A[j + i*lda].
             * j = t%TILE_K varies fast within warp → stride-1 reads.       */
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + (t % TILE_K);
            const int64_t i = m_base + (t / TILE_K);
            if(i >= m || j >= k) return;
            const double val  = A[i * lda + j];                         /* COALESCED */
            const double ival = trunc(ldexp(val, static_cast<int>(sftA[i])));
            const size_t stride = lda8i * cola8i;
            const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(i) * lda8i;
            #pragma unroll
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                const unsigned tidx = t_start + t_local;
                const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                const float   rf = static_cast<float>(r);
                const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                        static_cast<float>(cNegMod[tidx]), rf);
                __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                            A8i + t_local * stride + offset);
            }
        } else {
            /* Non-coalesced load: A stored m×k, A[i,j] = A[i + j*lda].
             * Use SHMEM transposition for coalesced reads AND writes.       */
            const int k_local = t / TILE_M;   /* 0..TILE_K-1 */
            const int m_local = t % TILE_M;   /* 0..TILE_M-1 */
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_local;
            const int64_t i = m_base + m_local;

            /* Load raw val (m_local varies fast → COALESCED) */
            shmem[k_local][m_local] = (i < m && j < k) ? A[i + j * lda] : 0.0;
            if(k_local == 0 && i < m) s_sft[m_local] = sftA[i];
            __syncthreads();

            /* Write: k_write varies fast → COALESCED writes */
            const int k_write = t % TILE_K;
            const int m_write = t / TILE_K;
            const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
            const int64_t i_out = m_base + m_write;
            if(i_out < m && j_out < k) {
                const double val  = shmem[k_write][m_write];
                const double ival = trunc(ldexp(val, static_cast<int>(s_sft[m_write])));
                const size_t stride = lda8i * cola8i;
                const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i;
                #pragma unroll
                for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                    const unsigned tidx = t_start + t_local;
                    const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                    const float   rf = static_cast<float>(r);
                    const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                            static_cast<float>(cNegMod[tidx]), rf);
                    __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                                A8i + t_local * stride + offset);
                }
            }
        }
    } else {
        /* ── B block (symmetric to A, with TRANS_B) ─────────────────────── */
        const int64_t n_base = static_cast<int64_t>(blockIdx.y - m_y_blocks) * TILE_M;

        if constexpr (!TRANS_B) {
            /* Coalesced: B stored k×n, B[l,j] = B[j + l*ldb].
             * j = t%TILE_K varies fast → stride-1 reads.                   */
            const int64_t j   = static_cast<int64_t>(blockIdx.x) * TILE_K + (t % TILE_K);
            const int64_t col = n_base + (t / TILE_K);
            if(col >= n || j >= k) return;
            const double val  = B[col * ldb + j];                       /* COALESCED */
            const double ival = trunc(ldexp(val, static_cast<int>(sftB[col])));
            const size_t stride = ldb8i * static_cast<size_t>(n);
            const size_t offset = static_cast<size_t>(j) + static_cast<size_t>(col) * ldb8i;
            #pragma unroll
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                const unsigned tidx = t_start + t_local;
                const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                const float   rf = static_cast<float>(r);
                const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                        static_cast<float>(cNegMod[tidx]), rf);
                __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                            B8i + t_local * stride + offset);
            }
        } else {
            /* Non-coalesced: B stored n×k, B[l,j] = B[l + j*ldb].
             * Use SHMEM transposition.                                       */
            const int k_local = t / TILE_M;
            const int l_local = t % TILE_M;
            const int64_t j   = static_cast<int64_t>(blockIdx.x) * TILE_K + k_local;
            const int64_t col = n_base + l_local;

            /* Load raw val (l_local varies fast → COALESCED) */
            shmem[k_local][l_local] = (col < n && j < k) ? B[col + j * ldb] : 0.0;
            if(k_local == 0 && col < n) s_sft[l_local] = sftB[col];
            __syncthreads();

            /* Write: k_write varies fast → COALESCED writes */
            const int k_write = t % TILE_K;
            const int l_write = t / TILE_K;
            const int64_t j_out   = static_cast<int64_t>(blockIdx.x) * TILE_K + k_write;
            const int64_t col_out = n_base + l_write;
            if(col_out < n && j_out < k) {
                const double val  = shmem[k_write][l_write];
                const double ival = trunc(ldexp(val, static_cast<int>(s_sft[l_write])));
                const size_t stride = ldb8i * static_cast<size_t>(n);
                const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(col_out) * ldb8i;
                #pragma unroll
                for(unsigned t_local = 0; t_local < T_COUNT; ++t_local) {
                    const unsigned tidx = t_start + t_local;
                    const double  r  = fma(cNegMod[tidx], rint(ival * cInvMod[tidx]), ival);
                    const float   rf = static_cast<float>(r);
                    const float  rf2 = fmaf(rintf(rf * cInvModF[tidx]),
                                            static_cast<float>(cNegMod[tidx]), rf);
                    __builtin_nontemporal_store(static_cast<int8_t>(static_cast<int32_t>(rf2)),
                                                B8i + t_local * stride + offset);
                }
            }
        }
    }
}

/* =========================================================================
 * GPU kernels — Part 2d: chunked CRT accumulation
 * ========================================================================= */
template <bool HAS_LO>
__global__ static void
oz2_chunk_accum_kernel_rt(const int32_t* __restrict__ C32i_batch,
                           double* __restrict__ Zhi, double* __restrict__ Zlo,
                           int64_t m, int64_t n, size_t ldc32i,
                           unsigned chunk_start, unsigned chunk_size, bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    if(is_first_chunk) { Zhi[idx] = local_hi; Zlo[idx] = local_lo; }
    else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi; Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO>
__global__ static void
oz2_accum_finalize_kernel_rt(const int32_t* __restrict__ C32i_batch,
                              const double* __restrict__ Zhi_in, const double* __restrict__ Zlo_in,
                              const double* __restrict__ C, double* __restrict__ D,
                              int64_t m, int64_t n, size_t ldc32i, int64_t ldc, int64_t ldd,
                              double alpha, double beta,
                              const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                              unsigned chunk_start, unsigned chunk_size, bool is_first_chunk)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    for(unsigned t_local = 0; t_local < chunk_size; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    double Zh, Zl;
    if(is_first_chunk) { Zh = local_hi; Zl = local_lo; }
    else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi; Zl = Zlo_in[idx] + err + local_lo;
    }
    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);
    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    D[d_idx] = alpha * ldexp(X, inv_sft) + beta * C[c_idx];
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_chunk_accum_kernel(const int32_t* __restrict__ C32i_batch,
                       double* __restrict__ Zhi, double* __restrict__ Zlo,
                       int64_t m, int64_t n, size_t ldc32i, unsigned chunk_start)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    if constexpr (IS_FIRST_CHUNK) {
        __builtin_nontemporal_store(local_hi, Zhi + idx);
        __builtin_nontemporal_store(local_lo, Zlo + idx);
    } else {
        const double old_hi = Zhi[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zhi[idx] = s_hi; Zlo[idx] += err + local_lo;
    }
}

template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
__global__ static void
oz2_accum_finalize_kernel(const int32_t* __restrict__ C32i_batch,
                          const double* __restrict__ Zhi_in, const double* __restrict__ Zlo_in,
                          const double* __restrict__ C, double* __restrict__ D,
                          int64_t m, int64_t n, size_t ldc32i, int64_t ldc, int64_t ldd,
                          double alpha, double beta,
                          const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                          unsigned chunk_start)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if(i >= m || l >= n) return;
    const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
    const size_t slice_stride = ldc32i * static_cast<size_t>(n);
    double local_hi = 0.0, local_lo = 0.0;
    #pragma unroll
    for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local) {
        const unsigned t    = chunk_start + t_local;
        const double dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
        const double dc     = fma(cNegMod[t], rint(dc_raw * cInvMod[t]), dc_raw);
        const double hi     = dc * cQpiHi[t];
        const double new_hi = local_hi + hi;
        const double err    = hi - (new_hi - local_hi);
        local_hi = new_hi;
        if constexpr (HAS_LO) local_lo = fma(dc, cQpiLo[t], local_lo + err);
        else                   local_lo += err;
    }
    double Zh, Zl;
    if constexpr (IS_FIRST_CHUNK) { Zh = local_hi; Zl = local_lo; }
    else {
        const double old_hi = Zhi_in[idx];
        const double s_hi   = old_hi + local_hi;
        const double err    = local_hi - (s_hi - old_hi);
        Zh = s_hi; Zl = Zlo_in[idx] + err + local_lo;
    }
    const double q = rint((Zh + Zl) * cInvP);
    const double X = fma(cP_lo, q, fma(cP_hi, q, Zh) + Zl);
    const int inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
    const size_t d_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
    double d_val = alpha * ldexp(X, inv_sft);
    if (beta != 0.0) {
        const size_t c_idx = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
        d_val += beta * C[c_idx];
    }
    __builtin_nontemporal_store(d_val, D + d_idx);
}

/* =========================================================================
 * Fused TN kernel — unified template supporting three macrotile sizes:
 *
 *   TILE=16, WM=4, WN=4 → 64×64   macrotile, 1024 threads (default)
 *   TILE=32, WM=4, WN=2 → 128×64  macrotile,  512 threads (m ≥ n preferred)
 *   TILE=32, WM=2, WN=4 → 64×128  macrotile,  512 threads (m < n preferred)
 *
 * MFMA instructions used:
 *   TILE=16: v_mfma_i32_16x16x32_i8 (gfx94x) / v_mfma_i32_16x16x64_i8 (gfx95x)
 *   TILE=32: v_mfma_i32_32x32x16_i8 (gfx94x) / v_mfma_i32_32x32x32_i8 (gfx95x)
 *
 * Output thread layout (same formula for TILE=16 and TILE=32):
 *   col = lane % TILE
 *   row = (TILE/4)*(lane/TILE) + 8*(e/4) + (e%4)   for e = 0..NREG-1
 *
 * Changes from prior revision:
 *   • Moduli pairing dropped: one modulus per k-loop iteration (simpler code,
 *     halved LDS usage from 32KB→16KB for TILE=16 config).
 *   • WM, WN, TILE are template parameters → one kernel for all three variants.
 *   • Separate A_STEPS / B_STEPS for asymmetric WM≠WN cooperative loads.
 *
 * NOTE: The MFMA output thread layout is based on the AMD CDNA3 ISA and
 * should be validated on hardware before production use.
 * ========================================================================= */

/* Swizzle factor (super-tile column grouping for L2 locality) */
static constexpr int OZ2_FUSED_SWIZZLE = 4;

/* Default (64×64) macrotile configuration — retained for legacy references */
static constexpr unsigned OZ2_FUSED_WM   = 4u;
static constexpr unsigned OZ2_FUSED_WN   = 4u;
static constexpr unsigned OZ2_FUSED_TILE = 16u;

/* Per-architecture MFMA K-block sizes.
 * TILE=16: K_BLOCK = 32 (gfx94x) or 64 (gfx95x)
 * TILE=32: K_BLOCK = 16 (gfx94x) or 32 (gfx95x)                           */
#if defined(__gfx950__)
static constexpr unsigned OZ2_KBLK_16 = 64u;
static constexpr unsigned OZ2_KBLK_32 = 32u;
#else
static constexpr unsigned OZ2_KBLK_16 = 32u;
static constexpr unsigned OZ2_KBLK_32 = 16u;
#endif
/* Maximum KBLK_LOAD across all kernel variants AND all supported architectures —
 * used for the workspace-zero guard in fp64EmulatedGemmImpl (HOST-side).
 * gfx942: TILE=16, K_UNROLL=4, KBLK_16=32 → KBLK_LOAD = 128
 * gfx950: TILE=16, K_UNROLL=4, KBLK_16=64 → KBLK_LOAD = 256
 * Must be the architecture-independent maximum (256) to ensure the workspace is
 * correctly zeroed on gfx950, where host code does not see __gfx950__.           */
static constexpr unsigned OZ2_FUSED_KBLK_LOAD_MAX = 256u;

/* MFMA output vector types */
typedef int v4i32  __attribute__((ext_vector_type(4)));
typedef int v16i32 __attribute__((ext_vector_type(16)));

/* TILE=16 MFMA source type and instruction wrapper */
#if defined(__gfx950__)
typedef long oz2_mfma_src16_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ v4i32
oz2_do_mfma_16(oz2_mfma_src16_t a, oz2_mfma_src16_t b, v4i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_16x16x64_i8(a, b, c, 0, 0, 0); }
#else
typedef int64_t oz2_mfma_src16_t;
__device__ __forceinline__ v4i32
oz2_do_mfma_16(int64_t a, int64_t b, v4i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c, 0, 0, 0); }
#endif

/* TILE=32 MFMA source type and instruction wrapper.
 * The source register size for v_mfma_i32_32x32x16_i8 is KBLK*TILE/64 = 8 bytes
 * (int64_t on gfx94x) — the same as for v_mfma_i32_16x16x32_i8.
 * For gfx95x (v_mfma_i32_32x32x32_i8): KBLK*TILE/64 = 16 bytes (long2), also
 * identical to the TILE=16 source type.  So oz2_mfma_src32_t = oz2_mfma_src16_t. */
typedef oz2_mfma_src16_t oz2_mfma_src32_t;
#if defined(__gfx950__)
__device__ __forceinline__ v16i32
oz2_do_mfma_32(oz2_mfma_src32_t a, oz2_mfma_src32_t b, v16i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_32x32x32_i8(a, b, c, 0, 0, 0); }
#else
__device__ __forceinline__ v16i32
oz2_do_mfma_32(oz2_mfma_src32_t a, oz2_mfma_src32_t b, v16i32 c) noexcept
{ return __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, 0, 0); }
#endif

/* ── Architecture-specific MFMA source load helper ─────────────────────────
 * On gfx95x (K_A_BYTES=16, KBLK_STRIDE=136): two 8-byte loads give 2-way LDS
 * bank conflicts instead of 4-way from a single 16-byte load.
 * On gfx94x (K_A_BYTES=8): single 8-byte load is always conflict-free.       */
#if defined(__gfx950__)
__device__ __forceinline__ oz2_mfma_src16_t
oz2_load_mfma_src16(const int8_t* __restrict__ ptr) noexcept {
    oz2_mfma_src16_t v;
    v[0] = *reinterpret_cast<const long*>(ptr);
    v[1] = *reinterpret_cast<const long*>(ptr + 8);
    return v;
}
#else
__device__ __forceinline__ oz2_mfma_src16_t
oz2_load_mfma_src16(const int8_t* __restrict__ ptr) noexcept {
    return *reinterpret_cast<const oz2_mfma_src16_t*>(ptr);
}
#endif
/* Same helper for TILE=32 source (oz2_mfma_src32_t == oz2_mfma_src16_t). */
__device__ __forceinline__ oz2_mfma_src32_t
oz2_load_mfma_src32(const int8_t* __restrict__ ptr) noexcept {
    return oz2_load_mfma_src16(ptr);
}

/* ── oz2_fused_TN_kernel ─────────────────────────────────────────────────── */
/*
 * Template parameters:
 *   S    – number of moduli (2..OZ2_S_MAX)
 *   HAS_LO – true when S > 7 (double-double CRT accumulation)
 *   WM   – wavefronts in M direction (default 4)
 *   WN   – wavefronts in N direction (default 4)
 *   TILE – MFMA tile: 16 (16×16 MFMA) or 32 (32×32 MFMA)
 *
 * Macrotile = (WM×TILE) × (WN×TILE).   blockDim = WM×WN×64 threads.
 *
 * Prerequisite: A8i/B8i workspace zero-padded to OZ2_FUSED_KBLK_LOAD_MAX
 * alignment (ensured by the hipMemsetAsync in fp64EmulatedGemmImpl).
 */
template <unsigned S, bool HAS_LO, unsigned WM = 4u, unsigned WN = 4u, unsigned TILE = 16u>
__global__ static void
oz2_fused_TN_kernel(
    const int8_t*  __restrict__ A8i,   /* [S × lda8i × cola8i] INT8 A  */
    size_t         stride_A_s,          /* lda8i × cola8i               */
    size_t         lda8i,               /* padded k                      */
    const int8_t*  __restrict__ B8i,   /* [S × ldb8i × n]  INT8 B      */
    size_t         stride_B_s,          /* ldb8i × n                    */
    size_t         ldb8i,               /* padded k                      */
    const double*  __restrict__ C,
    double*        __restrict__ D,
    int64_t m, int64_t n, int64_t k,
    int64_t ldc, int64_t ldd,
    double alpha, double beta,
    const int16_t* __restrict__ sftA,
    const int16_t* __restrict__ sftB,
    int            num_xccs)            /* number of XCCs on this device */
{
    /* ── Derived compile-time constants ────────────────────────────────────── */
    static constexpr unsigned NREG      = TILE * TILE / 64u;    /* 4 (TILE=16) or 16 (TILE=32) */
    static constexpr unsigned KBLK      = (TILE == 16u) ? OZ2_KBLK_16 : OZ2_KBLK_32;
    /* K-loop unroll factor (analogous to DepthU in TensiteLite):
     * K_UNROLL=4 doubles KBLK_LOAD, halves k-loop __syncthreads count, and
     * provides 4 back-to-back MFMAs per barrier interval for better ILP.
     *   gfx94x (MI300): K_UNROLL=4 — measured scratch=108B ✓
     *   gfx95x (MI350): K_UNROLL=2 — KBLK is doubled vs gfx94x, so K_UNROLL=4
     *     would double A_STEPS/B_STEPS giving the same register pressure as
     *     gfx94x K_UNROLL=8 (scratch=388B, 2.5× slower). Confirmed on hardware:
     *     K_UNROLL=4 → scratch=424B ✗  K_UNROLL=2 → scratch=104B ✓          */
    /* Architecture-specific LDS budget per CU:
     *   gfx95x (MI350): 160 KB — allows larger macrotiles and looser padding
     *   gfx94x (MI300): 64 KB                                                 */
#if defined(__gfx950__)
    static constexpr size_t   LDS_BUDGET  = 159u * 1024u;
#else
    static constexpr size_t   LDS_BUDGET  = 63u * 1024u;
#endif
    static constexpr size_t   LDS_MFMA_K4 = 2u * (WM * TILE * (KBLK * 4u)
                                                  + WN * TILE * (KBLK * 4u));
    static constexpr size_t   LDS_SFT     = static_cast<size_t>((WM + WN) * TILE * 2u);
#if defined(__gfx950__)
    /* TILE=32: K_UNROLL=2 — KBLK=32 doubled; A_STEPS≤4 validated (scratch=104B).
     * TILE=16: K_UNROLL=4 — KBLK=64; A_STEPS=4 at boundary (scratch=8B, acceptable).
     * Note: the launcher dispatches TILE=16 for ALL sizes on gfx950, so the TILE=32
     * branch is a compile-time fallback retained for correctness if ever dispatched.  */
    static constexpr unsigned K_UNROLL = (TILE == 32u) ? 2u : 4u;
#else
    static constexpr unsigned K_UNROLL = (LDS_MFMA_K4 + LDS_SFT <= LDS_BUDGET) ? 4u : 2u;
#endif
    static constexpr unsigned KBLK_LOAD   = KBLK * K_UNROLL;
    /* Source register size (bytes): KBLK × TILE / 64.
     * Equals sizeof(oz2_mfma_src16_t) on each architecture.
     *   gfx94x: TILE=16 → 32×16/64=8, TILE=32 → 16×32/64=8  (int64_t)
     *   gfx95x: TILE=16 → 64×16/64=16, TILE=32 → 32×32/64=16 (long2)  */
    static constexpr unsigned K_A_BYTES = KBLK * TILE / 64u;
    /* KBLK_PAD must be a multiple of K_A_BYTES (= the MFMA source read width in
     * bytes) to keep every LDS row naturally aligned for ds_read_b64 / ds_read_b128.
     * Using exactly K_A_BYTES is the minimum aligned padding that also breaks the
     * power-of-two row stride and eliminates LDS bank conflicts:
     *
     *   TILE=32 gfx94x: stride 32→40, GCD(10,64)=2, period 32 = wavefront half → 0 conflicts
     *   TILE=16 gfx94x: stride 64→72, GCD(18,64)=2, period 32 ≥ TILE=16       → 0 conflicts
     *   TILE=32 gfx95x: stride 64→80, GCD(20,64)=4, period 16 → 2-way (acceptable)
     *   TILE=16 gfx95x: skip (LDS budget, see guard below)
     *
     * The padding is suppressed when it would push the block's LDS usage above 63 KB. */
    /* On gfx95x (K_A_BYTES=16), use 8-byte LDS padding unit instead of 16.
     * KBLK_STRIDE=KBLK_LOAD+8 gives bank-stride/4=34, GCD(34,64)=2 → 2-way
     * conflicts for ds_read_b64, vs 4-way for ds_read_b128 with PAD=16.
     * MFMA source reads are split into two int64_t loads (see apply_mfma).   */
    static constexpr unsigned KBLK_PAD_UNIT = (K_A_BYTES == 16u) ? 8u : K_A_BYTES;
    static constexpr size_t   LDS_BYTES_OLD = 2u * (WM * TILE * KBLK_LOAD
                                                   + WN * TILE * KBLK_LOAD);
    static constexpr unsigned KBLK_PAD    = (LDS_BYTES_OLD + KBLK_PAD_UNIT * TILE * (WM + WN) * 2u
                                             <= LDS_BUDGET) ? KBLK_PAD_UNIT : 0u;
    static constexpr unsigned KBLK_STRIDE = KBLK_LOAD + KBLK_PAD;
    static constexpr unsigned BLK_THR   = WM * WN * 64u;
    static constexpr unsigned K4DIM     = KBLK_LOAD / 4u;
    /* Cooperative-load steps per thread (A and B differ for asymmetric WM≠WN) */
    static constexpr unsigned A_STEPS = (WM * TILE * K4DIM) / BLK_THR;
    static constexpr unsigned B_STEPS = (WN * TILE * K4DIM) / BLK_THR;
    static_assert(A_STEPS >= 1u && B_STEPS >= 1u, "cooperative load steps must be positive");

    /* ── Static LDS (double-buffered, no paired-moduli dimension) ─────────────
     * LDS sizes with K_UNROLL (gfx94x, KBLK_PAD=K_A_BYTES=8):
     *   TILE=32, WM=4,WN=2, K_UNROLL=8: A=[2][4][32][136]=34KB  B=[2][2][32][136]=17KB → 51KB
     *   TILE=16, WM=WN=4,   K_UNROLL=4: A=[2][4][16][136]=17KB  B=17KB               → 34KB
     * (KBLK_PAD=K_A_BYTES per row restores alignment and eliminates bank conflicts) */
    __shared__ int8_t  A8i_lds[2][WM][TILE][KBLK_STRIDE];
    __shared__ int8_t  B8i_lds[2][WN][TILE][KBLK_STRIDE];
    /* Per-block scale shifts cached in LDS — loaded cooperatively at kernel start
     * to eliminate global memory reads from the latency-critical finalize step.
     * Size: (WM+WN)*TILE int16_t ≤ 384 bytes (TILE=32 case), negligible.       */
    __shared__ int16_t sftA_lds[WM * TILE];
    __shared__ int16_t sftB_lds[WN * TILE];

    /* ── Thread decomposition ───────────────────────────────────────────────── */
    const int wid  = static_cast<int>(threadIdx.x) / 64;
    const int wm   = wid / static_cast<int>(WN);
    const int wn   = wid % static_cast<int>(WN);
    const int lane = static_cast<int>(threadIdx.x) % 64;
    const int k_int = static_cast<int>(k);

    /* ── XCC-aware block tile mapping ────────────────────────────────────────
     * MI300X / MI350X dispatch blocks strictly round-robin across XCCs
     * (verified via HW_REG_XCC_ID, register 20):  XCC_ID = blockIdx.x % num_xccs.
     *
     * The 1D grid partitions m_tiles across XCCs so each XCC owns a contiguous
     * slice → blocks on the same XCC reuse A8i for their m_tiles via concurrent
     * HBM read merging across all n_tile passes.
     *
     * Within each XCC, swizzle_eff consecutive intra-XCC blocks share the same
     * n_tile → B8i concurrent HBM merge (swizzle_eff reads → 1 HBM fetch).
     *
     * Measured optima (m=n=16K, 128×64 tile, gfx942):
     *   swizzle=4: best for small k (memory-bound); balances A8i and B8i merge
     *   swizzle=1: best for large k (compute-bound); maximizes A8i merge (38×)
     *              → set HIPBLASLT_EMULATION_FUSED_SWIZZLE=1 for large-k workloads */
    static constexpr int SWIZZLE_B8i_PREF = 4;  /* empirically optimal for k~1024 */

    const int m_tiles_total   = (static_cast<int>(m) + static_cast<int>(WM * TILE) - 1)
                                / static_cast<int>(WM * TILE);
    const int n_tiles_total   = (static_cast<int>(n) + static_cast<int>(WN * TILE) - 1)
                                / static_cast<int>(WN * TILE);
    const int m_tiles_per_xcc = (m_tiles_total + num_xccs - 1) / num_xccs;
    const int tiles_per_xcc   = m_tiles_per_xcc * n_tiles_total;

    const int swizzle_eff = (SWIZZLE_B8i_PREF > m_tiles_per_xcc) ? m_tiles_per_xcc : SWIZZLE_B8i_PREF;

    /* Read the hardware XCC ID: HW_REG_XCC_ID = register 20, CDNA3/CDNA4 ISA Table 19. */
    uint32_t hw_xcc_id = 0u;
    asm volatile("s_getreg_b32 %0, hwreg(20)" : "=s"(hw_xcc_id));

    const int linear_block = static_cast<int>(blockIdx.x);
    const int xcc_id       = static_cast<int>(hw_xcc_id);
    const int intra_xcc    = linear_block / num_xccs;

    /* Guard: extra blocks beyond total_tiles return early.                      */
    if (intra_xcc >= tiles_per_xcc) return;

    /* Within XCC: SWIZZLE_eff blocks share n_tile (B8i merge).
     * m_group steps through the XCC's m_tiles in groups of swizzle_eff.        */
    const int swizzle_group = intra_xcc / swizzle_eff;
    const int m_local       = intra_xcc % swizzle_eff;
    const int n_tile        = swizzle_group % n_tiles_total;
    const int m_group       = swizzle_group / n_tiles_total;
    const int m_tile        = xcc_id * m_tiles_per_xcc + m_group * swizzle_eff + m_local;

    if (m_tile >= m_tiles_total) return;   /* guard for uneven XCC boundary     */

    const int block_m_base = m_tile * static_cast<int>(WM * TILE);
    const int block_n_base = n_tile * static_cast<int>(WN * TILE);
    const int m_base = block_m_base + wm * static_cast<int>(TILE);
    const int n_base = block_n_base + wn * static_cast<int>(TILE);

    /* Per-lane output coordinates (constant throughout the S-loop).
     * MFMA output layout:
     *   col = lane % TILE   (N_cols per wavefront = TILE)
     *   row = NREG*(lane/TILE) + e   for e = 0..NREG-1
     *
     * Derivation: N_lane_groups = 64/TILE, each covering TILE rows/group.
     * NREG = TILE²/64 = TILE/N_lane_groups rows per group (= M_per_group).
     *   TILE=16: NREG=4,  4*(lane/16)  → groups covering rows {0..3},{4..7},{8..11},{12..15}
     *   TILE=32: NREG=16, 16*(lane/32) → groups covering rows {0..15},{16..31}              */
    const int col_val       = n_base + (lane % static_cast<int>(TILE));
    const int lane_row_base = m_base + 4 * (lane / static_cast<int>(TILE));

    /* CRT accumulators (persist across all S moduli) */
    double Zhi[NREG] = {};
    double Zlo[NREG] = {};

    /* ── Cooperative-load position tables ──────────────────────────────────────
     * A: covers WM × TILE × K4DIM positions; A_STEPS per thread.
     * B: covers WN × TILE × K4DIM positions; B_STEPS per thread.
     * For symmetric WM=WN: A_STEPS=B_STEPS; for asymmetric (128×64 / 64×128): differ. */
    unsigned wm_A[A_STEPS]; int ml_A[A_STEPS], k4_A[A_STEPS], mi_A[A_STEPS];
    #pragma unroll
    for (unsigned ls = 0; ls < A_STEPS; ++ls) {
        const int flat     = static_cast<int>(threadIdx.x + ls * BLK_THR);
        const int tile_dim = static_cast<int>(TILE * K4DIM);
        wm_A[ls] = static_cast<unsigned>(flat / tile_dim);
        ml_A[ls] = (flat % tile_dim) / static_cast<int>(K4DIM);
        k4_A[ls] = (flat % tile_dim) % static_cast<int>(K4DIM);
        mi_A[ls] = block_m_base + static_cast<int>(wm_A[ls]) * static_cast<int>(TILE) + ml_A[ls];
    }
    unsigned wn_B[B_STEPS]; int nl_B[B_STEPS], k4_B[B_STEPS], ni_B[B_STEPS];
    #pragma unroll
    for (unsigned ls = 0; ls < B_STEPS; ++ls) {
        const int flat     = static_cast<int>(threadIdx.x + ls * BLK_THR);
        const int tile_dim = static_cast<int>(TILE * K4DIM);
        wn_B[ls] = static_cast<unsigned>(flat / tile_dim);
        nl_B[ls] = (flat % tile_dim) / static_cast<int>(K4DIM);
        k4_B[ls] = (flat % tile_dim) % static_cast<int>(K4DIM);
        ni_B[ls] = block_n_base + static_cast<int>(wn_B[ls]) * static_cast<int>(TILE) + nl_B[ls];
    }

    /* ── Boundary-safe 4-byte load helpers (prologue only) ─────────────────── */
    auto cond_load_a = [&](const int8_t* As, unsigned ls, int ki) -> int32_t {
        int32_t v = 0;
        const int mi = mi_A[ls];
        if (mi < static_cast<int>(m)) {
            if (ki + 3 < k_int)
                v = *reinterpret_cast<const int32_t*>(As + static_cast<size_t>(mi) * lda8i + ki);
            else if (ki < k_int) {
                const int8_t* p = As + static_cast<size_t>(mi) * lda8i + ki;
                for (int b = 0; b < 4 && ki + b < k_int; ++b)
                    reinterpret_cast<int8_t*>(&v)[b] = p[b];
            }
        }
        return v;
    };
    auto cond_load_b = [&](const int8_t* Bs, unsigned ls, int ki) -> int32_t {
        int32_t v = 0;
        const int ni = ni_B[ls];
        if (ni < static_cast<int>(n)) {
            if (ki + 3 < k_int)
                v = *reinterpret_cast<const int32_t*>(Bs + static_cast<size_t>(ni) * ldb8i + ki);
            else if (ki < k_int) {
                const int8_t* p = Bs + static_cast<size_t>(ni) * ldb8i + ki;
                for (int b = 0; b < 4 && ki + b < k_int; ++b)
                    reinterpret_cast<int8_t*>(&v)[b] = p[b];
            }
        }
        return v;
    };

    /* ── Cooperative load of sftA/sftB into LDS ────────────────────────────────
     * Thread t < WM*TILE loads sftA_lds[t]; WM*TILE ≤ t < (WM+WN)*TILE loads sftB.
     * Remaining threads are idle.  These reads are small (≤192 int16_t values)
     * and are in flight here; they complete before the __syncthreads below.     */
    if (threadIdx.x < WM * TILE) {
        const int mi = block_m_base + static_cast<int>(threadIdx.x);
        sftA_lds[threadIdx.x] = (mi < static_cast<int>(m)) ? sftA[mi] : 0;
    } else if (threadIdx.x < (WM + WN) * TILE) {
        const int ni = block_n_base + static_cast<int>(threadIdx.x - WM * TILE);
        sftB_lds[threadIdx.x - WM * TILE] = (ni < static_cast<int>(n)) ? sftB[ni] : 0;
    }

    /* ── Pre-fetch modulus s=0 prologue into registers ─────────────────────────
     * Issuing these global loads BEFORE the __syncthreads below hides their
     * ~100-cycle HBM latency behind the barrier wait.                           */
    int32_t rA_pro[A_STEPS] = {};
    int32_t rB_pro[B_STEPS] = {};
    if (k_int > 0) {
        #pragma unroll
        for (unsigned ls = 0; ls < A_STEPS; ++ls)
            rA_pro[ls] = cond_load_a(A8i, ls, k4_A[ls] * 4);
        #pragma unroll
        for (unsigned ls = 0; ls < B_STEPS; ++ls)
            rB_pro[ls] = cond_load_b(B8i, ls, k4_B[ls] * 4);
    }
    /* Barrier: (1) ensures sftA/B LDS writes above are visible to all threads;
     *          (2) acts as the first per-modulus sync for s=0 (no previous
     *              epilogue MFMA exists, so it is correct to merge them).       */
    __syncthreads();

    /* ── MFMA helper: K_UNROLL=2 MFMAs from one LDS ping-pong slot ────────────
     * A_wm_slot = &A8i_lds[cur][wm][0][0], B_wn_slot = &B8i_lds[cur][wn][0][0].
     * LDS layout: [TILE][KBLK_LOAD] → stride KBLK_LOAD per row, 1 byte per k.
     * MFMA source accessed as: slot + (lane%TILE)*KBLK_LOAD + K_A_BYTES*(lane/TILE) + ku*KBLK */
    auto apply_mfma = [&](const int8_t* A_wm_slot, const int8_t* B_wn_slot,
                          int32_t (&C32)[NREG]) __attribute__((always_inline)) {
        const int m_col  = lane % static_cast<int>(TILE);
        const int k_base = static_cast<int>(K_A_BYTES) * (lane / static_cast<int>(TILE));
        if constexpr (TILE == 16u) {
            v4i32 vx;
            for (unsigned e = 0; e < NREG; ++e) vx[static_cast<int>(e)] = C32[e];
            for (unsigned ku = 0; ku < K_UNROLL; ++ku) {
                const int kab = k_base + static_cast<int>(ku * KBLK);
                const int off_A = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                const int off_B = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                /* Use KBLK_STRIDE as the LDS row stride (accounts for KBLK_PAD).
                 * On gfx95x (K_A_BYTES=16): two 64-bit loads → 2-way LDS conflicts
                 * instead of 4-way from a single 128-bit load.                  */
                /* oz2_load_mfma_src16: on gfx95x uses two 64-bit loads (2-way
                 * LDS conflicts); on gfx94x uses one 64-bit load (0 conflicts). */
                const auto sa = oz2_load_mfma_src16(A_wm_slot + off_A);
                const auto sb = oz2_load_mfma_src16(B_wn_slot + off_B);
                vx = oz2_do_mfma_16(sa, sb, vx);
            }
            for (unsigned e = 0; e < NREG; ++e) C32[e] = vx[static_cast<int>(e)];
        } else {
            v16i32 vx;
            for (unsigned e = 0; e < NREG; ++e) vx[static_cast<int>(e)] = C32[e];
            for (unsigned ku = 0; ku < K_UNROLL; ++ku) {
                const int kab = k_base + static_cast<int>(ku * KBLK);
                const int off_A = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                const int off_B = m_col * static_cast<int>(KBLK_STRIDE) + kab;
                const auto sa = oz2_load_mfma_src32(A_wm_slot + off_A);
                const auto sb = oz2_load_mfma_src32(B_wn_slot + off_B);
                vx = oz2_do_mfma_32(sa, sb, vx);
            }
            for (unsigned e = 0; e < NREG; ++e) C32[e] = vx[static_cast<int>(e)];
        }
    };

    /* ── CRT helper: accumulate one modulus's contribution into Zhi/Zlo ──────── */
    auto crt_update = [&](const int32_t* C32x, unsigned sidx) __attribute__((always_inline)) {
        const double nm = cNegMod[sidx], im = cInvMod[sidx];
        for (unsigned e = 0; e < NREG; ++e) {
            const double dc_raw = static_cast<double>(C32x[e]);
            const double dc     = fma(nm, rint(dc_raw * im), dc_raw);
            const double hi     = dc * cQpiHi[sidx];
            const double new_hi = Zhi[e] + hi;
            const double err    = hi - (new_hi - Zhi[e]);
            Zhi[e] = new_hi;
            if constexpr (HAS_LO) Zlo[e] = fma(dc, cQpiLo[sidx], Zlo[e] + err);
            else                  Zlo[e] += err;
        }
    };

    /* ── Main loop: one modulus per iteration (no paired-moduli) ───────────────
     * __syncthreads() at the top of each iteration ensures the previous
     * modulus's epilogue MFMA (which reads LDS) has completed before we
     * overwrite LDS with the next modulus's prologue data.                     */
    for (unsigned s = 0; s < S; ++s) {
        const int8_t* A8i_s = A8i + static_cast<size_t>(s) * stride_A_s;
        const int8_t* B8i_s = B8i + static_cast<size_t>(s) * stride_B_s;
        int32_t C32[NREG] = {};

        if (k_int > 0) {
            __syncthreads(); /* separate from previous modulus's LDS usage */

            /* Prologue: write pre-fetched first k-block (rA_pro/rB_pro) to LDS slot 0.
             * These registers were loaded from global memory BEFORE the __syncthreads
             * above, so their ~100-cycle HBM latency is fully hidden.            */
            #pragma unroll
            for (unsigned ls = 0; ls < A_STEPS; ++ls)
                *reinterpret_cast<int32_t*>(&A8i_lds[0][wm_A[ls]][ml_A[ls]][k4_A[ls] * 4u]) = rA_pro[ls];
            #pragma unroll
            for (unsigned ls = 0; ls < B_STEPS; ++ls)
                *reinterpret_cast<int32_t*>(&B8i_lds[0][wn_B[ls]][nl_B[ls]][k4_B[ls] * 4u]) = rB_pro[ls];

            /* Pre-fetch modulus (s+1)'s first k-block into registers NOW, while the
             * LDS writes above are completing.  The ~100-cycle HBM latency will be
             * hidden behind __syncthreads [2] below and the start of the k-loop.  */
            if (s + 1 < S) {
                #pragma unroll
                for (unsigned ls = 0; ls < A_STEPS; ++ls)
                    rA_pro[ls] = cond_load_a(A8i + static_cast<size_t>(s + 1) * stride_A_s, ls, k4_A[ls] * 4);
                #pragma unroll
                for (unsigned ls = 0; ls < B_STEPS; ++ls)
                    rB_pro[ls] = cond_load_b(B8i + static_cast<size_t>(s + 1) * stride_B_s, ls, k4_B[ls] * 4);
            }

            __syncthreads(); /* [2] wait for prologue LDS writes to complete */

            /* Precompute per-modulus base pointers for the ping-pong prefetch */
            const int8_t* A_base[A_STEPS];
            const int8_t* B_base[B_STEPS];
            #pragma unroll
            for (unsigned ls = 0; ls < A_STEPS; ++ls)
                A_base[ls] = A8i_s + static_cast<size_t>(mi_A[ls]) * lda8i
                           + static_cast<size_t>(k4_A[ls]) * 4u;
            #pragma unroll
            for (unsigned ls = 0; ls < B_STEPS; ++ls)
                B_base[ls] = B8i_s + static_cast<size_t>(ni_B[ls]) * ldb8i
                           + static_cast<size_t>(k4_B[ls]) * 4u;

            int32_t rA[A_STEPS], rB[B_STEPS];
            int cur = 0;

            /* ── Double-buffered k-loop ──────────────────────────────────── */
            for (int k_off = 0; k_off + static_cast<int>(KBLK_LOAD) < k_int;
                 k_off += static_cast<int>(KBLK_LOAD)) {
                const int nxt   = 1 - cur;
                const int nxt_k = k_off + static_cast<int>(KBLK_LOAD);
                /* Prefetch next k-block into registers */
                #pragma unroll
                for (unsigned ls = 0; ls < A_STEPS; ++ls)
                    rA[ls] = *reinterpret_cast<const int32_t*>(A_base[ls] + nxt_k);
                #pragma unroll
                for (unsigned ls = 0; ls < B_STEPS; ++ls)
                    rB[ls] = *reinterpret_cast<const int32_t*>(B_base[ls] + nxt_k);
                /* Compute from current LDS slot */
                apply_mfma(&A8i_lds[cur][wm][0][0], &B8i_lds[cur][wn][0][0], C32);
                /* Write prefetch to next LDS slot */
                #pragma unroll
                for (unsigned ls = 0; ls < A_STEPS; ++ls)
                    *reinterpret_cast<int32_t*>(
                        &A8i_lds[nxt][wm_A[ls]][ml_A[ls]][k4_A[ls] * 4u]) = rA[ls];
                #pragma unroll
                for (unsigned ls = 0; ls < B_STEPS; ++ls)
                    *reinterpret_cast<int32_t*>(
                        &B8i_lds[nxt][wn_B[ls]][nl_B[ls]][k4_B[ls] * 4u]) = rB[ls];
                __syncthreads();
                cur = nxt;
            }
            /* Epilogue: last k-block */
            apply_mfma(&A8i_lds[cur][wm][0][0], &B8i_lds[cur][wn][0][0], C32);
        }

        /* CRT accumulation for modulus s */
        crt_update(C32, s);
    } /* end moduli loop */

    /* ── Finalize: CRT range-reduction + inverse scale + write D ─────────────
     * ISA-verified output layout (AMD CDNA3, Sec. 7.1.4.2):
     *   col = lane % TILE
     *   row = 4*(lane/TILE) + 8*(e/4) + (e%4)   for e = 0..NREG-1
     * i.e. lane_row_base = m_base + 4*(lane/TILE), then:
     *   TILE=16, NREG=4 : 8*(e/4)=0 always → ri = lane_row_base + e        ✓
     *   TILE=32, NREG=16: ri = lane_row_base + 8*(e/4) + (e%4)              ✓ */
    #pragma unroll
    for (unsigned e = 0; e < NREG; ++e) {
        const int ri = lane_row_base + static_cast<int>(8u * (e / 4u) + (e % 4u));
        const int ci = col_val;
        if (ri >= static_cast<int>(m) || ci >= static_cast<int>(n)) continue;
        const double q = rint((Zhi[e] + Zlo[e]) * cInvP);
        const double X = fma(cP_lo, q, fma(cP_hi, q, Zhi[e]) + Zlo[e]);
        /* Read scale shifts from LDS (loaded at kernel start) — avoids global reads
         * after the s-loop has likely evicted sftA/sftB from L1.
         * ri - block_m_base ∈ [0, WM*TILE) and ci - block_n_base ∈ [0, WN*TILE)
         * are guaranteed for all non-skipped (ri, ci) pairs.                    */
        const int inv_sft = -(static_cast<int>(sftA_lds[ri - block_m_base])
                             + static_cast<int>(sftB_lds[ci - block_n_base]));
        const size_t d_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldd;
        double d_val = alpha * ldexp(X, inv_sft);
        /* beta is a uniform scalar — the compiler lowers this to a scalar branch
         * (s_cmp + s_cbranch), skipping the C load for all lanes simultaneously
         * when beta == 0.  No per-lane divergence.                              */
        if (beta != 0.0) {
            const size_t c_idx = static_cast<size_t>(ri) + static_cast<size_t>(ci) * ldc;
            d_val += beta * C[c_idx];
        }
        /* NT store: D is write-only from this kernel's perspective.  Bypassing
         * L2 prevents D write-allocate RFOs from evicting A8i/B8i lines that
         * concurrent blocks depend on for cross-n-tile L2 reuse.               */
        __builtin_nontemporal_store(d_val, D + d_idx);
    }
}

/* ── Host-side launcher for the fused TN kernel ─────────────────────────── */
/* Selects the appropriate macrotile variant based on output aspect ratio:
 *   m ≥ n and (m ≥ 128 or n ≥ 64): WM=4, WN=2, TILE=32 → 128×64 macrotile
 *   m < n and (m ≥ 64 or n ≥ 128): WM=2, WN=4, TILE=32 → 64×128 macrotile
 *   otherwise:                     WM=4, WN=4, TILE=16 → 64×64 macrotile (default)
 */
static rocblaslt_status
oz2_launch_fused_TN(const int8_t*  A8i,    /* workspace INT8 A (all S moduli stacked) */
                    const int8_t*  B8i,    /* workspace INT8 B (all S moduli stacked) */
                    size_t         lda8i,  /* padded k, leading dim of A8i per modulus */
                    size_t         cola8i, /* padded m                                 */
                    size_t         ldb8i,  /* padded k, leading dim of B8i per modulus */
                    const double*  C,
                    double*        D,
                    int64_t m, int64_t n, int64_t k,
                    int64_t ldc, int64_t ldd,
                    double alpha, double beta,
                    const int16_t* sftA, const int16_t* sftB,
                    unsigned num_moduli, hipStream_t stream)
{
    if (oz2_init_constants(num_moduli) != hipSuccess)
        return rocblaslt_status_internal_error;

    const size_t stride_A_s = lda8i * cola8i;
    const size_t stride_B_s = ldb8i * static_cast<size_t>(n);
    const bool   has_lo     = (num_moduli > 7u);

    /* Select macrotile variant based on output shape.
     * VGPR occupancy check for WM=4,WN=4,TILE=32 (blockDim=1024, 16 wf/block):
     *   SIMD needs 4 wf → max VGPRs = 512/4 = 128
     *   Estimated kernel VGPRs ≈ 208 (Zhi[16]+Zlo[16]=64, C32[16]=16, tables+misc)
     *   4×208 = 832 > 512 → block cannot be scheduled on any CU (CDNA4 ISA §3.6.4)
     * WM=4,WN=2 and WM=2,WN=4 use blockDim=512 (8 wf, 2 wf/SIMD): 2×192=384≤512 ✓
     *
     * gfx950 override: always use TILE=16 (64×64 macrotile) with K_UNROLL=4.
     * On gfx950, TILE=16/K_UNROLL=4 and TILE=32/K_UNROLL=2 do equal compute per
     * barrier, but TILE=16/K_UNROLL=4 has 1.7× fewer total barriers (4× fewer k-loop
     * iterations per block, 2× more blocks → net 1.7× reduction). Measured 2.5× fused
     * slowdown on MI355 with TILE=32/K_UNROLL=2 makes TILE=16 the better default.     */
    /* Query current device once; used for both XCC count and gfx950 detection. */
    int cur_dev = 0;
    (void)hipGetDevice(&cur_dev);

    /* On gfx950, use TILE=16 (K_UNROLL=4) for all sizes — fewer total barriers. */
    int is_gfx950 = 0;
    {
        int gfx950_chip_id = 0;
        (void)hipDeviceGetAttribute(&gfx950_chip_id, hipDeviceAttributePciChipId, cur_dev);
        const uint32_t pci_id = static_cast<uint32_t>(gfx950_chip_id) & 0xFFFFu;
        is_gfx950 = (pci_id == 0x75a3u || pci_id == 0x75b3u) ? 1 : 0;
    }
    const bool use_tile32 = !is_gfx950 && (m >= 64 && n >= 64) && (m >= 128 || n >= 128);
    const bool use_128x64 = use_tile32 && (m >= n);

    /* Query number of XCCs (Graphics Compute Dies) on the current device.
     * MI300X and MI350X both have 8 XCCs.  The XCC mapping partitions m_tiles
     * across XCCs to ensure A8i L2 reuse per-XCC (up to 8× HBM traffic reduction).  */
    int num_xccs = 1;
    {
        (void)hipDeviceGetAttribute(&num_xccs, hipDeviceAttributeNumberOfXccs, cur_dev);
        if (num_xccs <= 0) num_xccs = 1;
    }

    /* Compute 1D grid for a given (wm_v, wn_v, tile_v) macrotile.
     * With XCC mapping, total_blocks = m_tiles_per_xcc × n_tiles × num_xccs.
     * The kernel maps blockIdx.x → (XCC, intra-XCC tile) → (m_tile, n_tile).  */
    auto make_grid = [&](unsigned wm_v, unsigned wn_v, unsigned tile_v) -> dim3 {
        const int mt = static_cast<int>(
            (m + static_cast<int64_t>(wm_v * tile_v) - 1) / static_cast<int64_t>(wm_v * tile_v));
        const int nt = static_cast<int>(
            (n + static_cast<int64_t>(wn_v * tile_v) - 1) / static_cast<int64_t>(wn_v * tile_v));
        const int m_per_xcc    = (mt + num_xccs - 1) / num_xccs;
        const int total_blocks = m_per_xcc * nt * num_xccs;
        return dim3(static_cast<unsigned>(total_blocks), 1u);
    };

#define OZ2_FUSED_LAUNCH(S_V, HL, WM_V, WN_V, TILE_V) \
    hipLaunchKernelGGL((oz2_fused_TN_kernel<(S_V),(HL),(WM_V),(WN_V),(TILE_V)>), \
                       make_grid((WM_V),(WN_V),(TILE_V)), \
                       dim3((WM_V)*(WN_V)*64u), 0, stream, \
                       A8i, stride_A_s, lda8i, B8i, stride_B_s, ldb8i, \
                       C, D, m, n, k, ldc, ldd, alpha, beta, sftA, sftB, \
                       num_xccs)

#define OZ2_DISPATCH_SHAPE(S_V) \
    do { \
        if (use_tile32) { \
            if (use_128x64) { \
                if (has_lo) OZ2_FUSED_LAUNCH((S_V), true,  4u, 2u, 32u); \
                else        OZ2_FUSED_LAUNCH((S_V), false, 4u, 2u, 32u); \
            } else { \
                if (has_lo) OZ2_FUSED_LAUNCH((S_V), true,  2u, 4u, 32u); \
                else        OZ2_FUSED_LAUNCH((S_V), false, 2u, 4u, 32u); \
            } \
        } else { \
            if (has_lo) OZ2_FUSED_LAUNCH((S_V), true,  4u, 4u, 16u); \
            else        OZ2_FUSED_LAUNCH((S_V), false, 4u, 4u, 16u); \
        } \
    } while(0)

    switch (num_moduli) {
        case  2: OZ2_DISPATCH_SHAPE( 2); break;
        case  3: OZ2_DISPATCH_SHAPE( 3); break;
        case  4: OZ2_DISPATCH_SHAPE( 4); break;
        case  5: OZ2_DISPATCH_SHAPE( 5); break;
        case  6: OZ2_DISPATCH_SHAPE( 6); break;
        case  7: OZ2_DISPATCH_SHAPE( 7); break;
        case  8: OZ2_DISPATCH_SHAPE( 8); break;
        case  9: OZ2_DISPATCH_SHAPE( 9); break;
        case 10: OZ2_DISPATCH_SHAPE(10); break;
        case 11: OZ2_DISPATCH_SHAPE(11); break;
        case 12: OZ2_DISPATCH_SHAPE(12); break;
        case 13: OZ2_DISPATCH_SHAPE(13); break;
        case 14: OZ2_DISPATCH_SHAPE(14); break;
        case 15: OZ2_DISPATCH_SHAPE(15); break;
        case 16: OZ2_DISPATCH_SHAPE(16); break;
        case 17: OZ2_DISPATCH_SHAPE(17); break;
        case 18: OZ2_DISPATCH_SHAPE(18); break;
        default: return rocblaslt_status_invalid_value;
    }

#undef OZ2_DISPATCH_SHAPE
#undef OZ2_FUSED_LAUNCH

    return rocblaslt_status_success;
}

static const char* oz2_profile_file()
{
    static const char* const fn = std::getenv("HIPBLASLT_EMULATION_PROFILE");
    return fn;
}

/* HIPBLASLT_EMULATION_FUSED controls whether the fused MFMA+CRT kernel is used:
 *   "on"  / "force"      → always use fused (bypasses performance model)
 *   "off" / "never"      → never  use fused (forces non-fused path)
 *   "auto"/ "performant" → performance model decides (default)               */
enum class Oz2FusedMode { AUTO, ON, OFF };
static Oz2FusedMode oz2_fused_mode()
{
    static const Oz2FusedMode v = []() -> Oz2FusedMode {
        const char* e = std::getenv("HIPBLASLT_EMULATION_FUSED");
        if (e == nullptr) return Oz2FusedMode::AUTO;
        if (std::strcmp(e, "on") == 0 || std::strcmp(e, "force") == 0)
            return Oz2FusedMode::ON;
        if (std::strcmp(e, "off") == 0 || std::strcmp(e, "never") == 0)
            return Oz2FusedMode::OFF;
        return Oz2FusedMode::AUTO;   /* "auto", "performant", or unrecognized */
    }();
    return v;
}

/* =========================================================================
 * fp64EmulatedGemm
 * ========================================================================= */

/* Aggregates per-component GPU times across all leaf sub-GEMMs and counts
 * how many leaf (monolithic) sub-GEMMs were executed.  Owned by the public
 * fp64EmulatedGemm wrapper; passed by pointer through recursive calls.     */
struct Fp64ProfileAccum {
    float    t_prelim      = 0.f;
    float    t_prelim_gemm = 0.f;
    float    t_extract     = 0.f;
    float    t_refine      = 0.f;
    float    t_fused       = 0.f;   /* fused TN kernel (non-zero when fused path taken) */
    float    t_scale       = 0.f;
    float    t_int8        = 0.f;
    float    t_accum       = 0.f;
    float    t_finalize    = 0.f;
    unsigned n_sub_gemms   = 0u;
};

/* Internal implementation — called recursively during binary-halving.
 * prof != nullptr enables per-component accumulation across all leaves.   */
static rocblaslt_status
fp64EmulatedGemmImpl(hipblasOperation_t           opA,
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
                     const Fp64EmulationSettings& settings,
                     Fp64ProfileAccum*            prof)
{
    if(settings.num_moduli == 0u
       && cached_mantissa_bit_count_env().state == FP64_EMULATION_ENV_INVALID)
        return rocblaslt_status_invalid_value;
    const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= OZ2_S_MAX)
                                    ? settings.num_moduli : fp64EmulationNumModuli();
    if(settings.dynamic_mode)
        fp64EmulationWarnDynamicTemporary();
    if(oz2_init_constants(num_moduli) != hipSuccess)
        return rocblaslt_status_internal_error;

    const char* const _pf   = oz2_profile_file();
    const bool        _prof = (_pf != nullptr);
    hipEvent_t _ev0{}, _ev1{}, _ev_tot{};
    float _t_prelim = 0, _t_prelim_gemm = 0, _t_extract = 0, _t_refine = 0,
          _t_fused  = 0, _t_scale = 0, _t_int8 = 0, _t_accum = 0, _t_finalize = 0;
    if(_prof) { (void)hipEventCreate(&_ev0); (void)hipEventCreate(&_ev1); }
    auto _pstart = [&]() noexcept { if(_prof) (void)hipEventRecord(_ev0, stream); };
    auto _pstop  = [&](float& t) noexcept {
        if(_prof) {
            (void)hipEventRecord(_ev1, stream); (void)hipStreamSynchronize(stream);
            float ms = 0.f; (void)hipEventElapsedTime(&ms, _ev0, _ev1); t += ms;
        }
    };

    const unsigned chunk_size       = oz2_compute_chunk_size(m, n, num_moduli);
    const unsigned scale_chunk_size = oz2_compute_scale_chunk_size(m, n, k, num_moduli, chunk_size);

    const size_t lda8i  = oz2_pad(static_cast<size_t>(k));
    const size_t cola8i = oz2_pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = oz2_pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);

    const size_t szA8i   = scale_chunk_size * lda8i * cola8i;
    const size_t szB8i   = scale_chunk_size * ldb8i * static_cast<size_t>(n);
    const size_t szZhi   = szC32i;
    const size_t szZlo   = szC32i;
    const size_t szSftA  = cola8i;
    const size_t szSftB  = padn;
    const size_t szNanFlag = 1;
    const size_t szRowMax  = cola8i;

    const size_t wsBytes =
          szA8i    * sizeof(int8_t)
        + szB8i    * sizeof(int8_t)
        + chunk_size * szC32i * sizeof(int32_t)
        + szZhi    * sizeof(double)
        + szZlo    * sizeof(double)
        + szSftA   * sizeof(int16_t)
        + szSftB   * sizeof(int16_t)
        + szNanFlag * sizeof(uint32_t)
        + szRowMax  * sizeof(int32_t);

    bool   ws_owned = false;
    char*  ws       = nullptr;
    if(settings.workspace != nullptr && settings.workspace_bytes >= wsBytes) {
        ws = static_cast<char*>(settings.workspace);
    } else {
        ws_owned = true;
        if(hipMallocAsync(&ws, wsBytes, stream) != hipSuccess)
            return rocblaslt_status_memory_error;
    }

    int8_t*   const A8i        = reinterpret_cast<int8_t*>(ws);
    int8_t*   const B8i        = A8i + szA8i;
    int32_t*  const C32i_batch = reinterpret_cast<int32_t*>(B8i + szB8i);
    double*   const Zhi        = reinterpret_cast<double*>(C32i_batch + chunk_size * szC32i);
    double*   const Zlo        = Zhi + szZhi;
    int16_t*  const sftA       = reinterpret_cast<int16_t*>(Zlo + szZlo);
    int16_t*  const sftB       = sftA + szSftA;
    uint32_t* const nan_flag   = reinterpret_cast<uint32_t*>(sftB + szSftB);
    int32_t*  const row_max    = reinterpret_cast<int32_t*>(nan_flag + szNanFlag);
    int32_t*  const C32i       = C32i_batch;

    if(_prof) (void)hipEventRecord(_ev_tot, stream);

    int8_t* const A8i_high = A8i;
    int8_t* const B8i_high = B8i;

    const bool tA = (opA != HIPBLAS_OP_N);
    const bool tB = (opB != HIPBLAS_OP_N);

    const uint32_t svmask = (settings.sv_mask != ~0u)
                                ? settings.sv_mask : fp64EmulationSpecialValuesMask();

    if(svmask != 0u) {
        if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess) {
            (void)hipFreeAsync(ws, stream); return rocblaslt_status_internal_error;
        }
    }

    hipblasLtMatrixLayout_t layoutA  = nullptr;
    hipblasLtMatrixLayout_t layoutB  = nullptr;
    hipblasLtMatrixLayout_t layoutCD = nullptr;
    hipblasLtMatrixLayout_t layoutA_b  = nullptr;
    hipblasLtMatrixLayout_t layoutB_b  = nullptr;
    hipblasLtMatrixLayout_t layoutCD_b = nullptr;
    hipblasLtMatmulDesc_t   matmulDesc = nullptr;

    auto cleanup = [&]() noexcept {
        bool ok = true;
        if(layoutCD_b) (void)hipblasLtMatrixLayoutDestroy(layoutCD_b);
        if(layoutB_b)  (void)hipblasLtMatrixLayoutDestroy(layoutB_b);
        if(layoutA_b)  (void)hipblasLtMatrixLayoutDestroy(layoutA_b);
        if(matmulDesc) (void)hipblasLtMatmulDescDestroy(matmulDesc);
        if(layoutCD)   (void)hipblasLtMatrixLayoutDestroy(layoutCD);
        if(layoutB)    (void)hipblasLtMatrixLayoutDestroy(layoutB);
        if(layoutA)    (void)hipblasLtMatrixLayoutDestroy(layoutA);
        if(ws_owned && hipFreeAsync(ws, stream) != hipSuccess) ok = false;
        return ok;
    };
    auto fail_internal = [&]() {
        cleanup();
        return rocblaslt_status_internal_error;
    };

    if(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(m), static_cast<int64_t>(lda8i)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(n), static_cast<int64_t>(ldb8i)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutCreate(&layoutCD, HIP_R_32I, static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<int64_t>(ldc32i)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    {
        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        if(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)) != HIPBLAS_STATUS_SUCCESS)
            return fail_internal();
        if(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) != HIPBLAS_STATUS_SUCCESS)
            return fail_internal();
    }

    const int32_t one_i = 1, zero_i = 0;

    /* Fused preliminary shift + extraction (oz2_accu_prelim_kernel).
     * Grid adapts to TRANS_A/TRANS_B:
     *   m_blks = TRANS_A ? m           : ceil(m / OZ2_PRELIM_TILE_M)
     *   n_blks = TRANS_B ? ceil(n/TM)  : n
     * Block = OZ2_PRELIM_TILE_K × OZ2_PRELIM_TILE_M = 256 threads.    */
    const unsigned m_blks_prelim = tA
        ? static_cast<unsigned>(m)
        : static_cast<unsigned>((m + OZ2_PRELIM_TILE_M - 1) / OZ2_PRELIM_TILE_M);
    const unsigned n_blks_prelim = tB
        ? static_cast<unsigned>((n + OZ2_PRELIM_TILE_M - 1) / OZ2_PRELIM_TILE_M)
        : static_cast<unsigned>(n);
    _pstart();
#define OZ2_PRELIM(TA, TB, CN) \
    hipLaunchKernelGGL((oz2_accu_prelim_kernel<(TA),(TB),(CN)>), \
                       dim3(m_blks_prelim + n_blks_prelim), \
                       dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_TILE_M), 0, stream, \
                       A, m, k, lda, A8i_high, lda8i, sftA, \
                       B, n, ldb, B8i_high, ldb8i, sftB, nan_flag, m_blks_prelim)
    if(svmask == 0u) {
        if(tA && !tB)       OZ2_PRELIM(true,  false, false);
        else if(!tA && !tB) OZ2_PRELIM(false, false, false);
        else if(!tA && tB)  OZ2_PRELIM(false, true,  false);
        else                OZ2_PRELIM(true,  true,  false);
    } else {
        if(tA && !tB)       OZ2_PRELIM(true,  false, true);
        else if(!tA && !tB) OZ2_PRELIM(false, false, true);
        else if(!tA && tB)  OZ2_PRELIM(false, true,  true);
        else                OZ2_PRELIM(true,  true,  true);
    }
#undef OZ2_PRELIM
    _pstop(_t_prelim);
    /* _t_extract remains 0: extraction is now fused into _t_prelim */

    if(svmask != 0u) {
        if(hipStreamSynchronize(stream) != hipSuccess) {
            return fail_internal();
        }
        uint32_t detected = 0u;
        if(hipMemcpy(&detected, nan_flag, sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
            return fail_internal();
        }
        if(detected & svmask) {
            cleanup();
            return rocblaslt_status_invalid_value;
        }
    }

    /* Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high */
    _pstart();
    if(hipblasLtMatmul(settings.handle, matmulDesc,
                       &one_i, A8i_high, layoutA, B8i_high, layoutB,
                       &zero_i, C32i, layoutCD, C32i, layoutCD, nullptr, nullptr, 0, stream)
       != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    _pstop(_t_prelim_gemm);

    /* log2P from the mixed table: fast::log2P for s=13,14,15 (prevents CRT overflow);
     * accu::log2P for all other s (preserves maximum precision).               */
    const float accu_log2P = h_accu_log2P_all[num_moduli - 2];
    _pstart();
    const unsigned sftA_m_blks = static_cast<unsigned>((m + 63) / 64);
    const unsigned sftA_n_blks = static_cast<unsigned>((n + 63) / 64);
    (void)hipMemsetAsync(row_max, 0, szRowMax * sizeof(int32_t), stream);
    hipLaunchKernelGGL(oz2_refine_sftA_partial_kernel, dim3(sftA_m_blks, sftA_n_blks), dim3(64), 0, stream,
                       C32i, m, n, ldc32i, row_max);
    hipLaunchKernelGGL(oz2_refine_sftA_apply_kernel, dim3(sftA_m_blks), dim3(64), 0, stream,
                       row_max, sftA, m, accu_log2P);
    hipLaunchKernelGGL(oz2_refine_sftB_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, stream,
                       C32i, m, n, ldc32i, sftB, accu_log2P);
    _pstop(_t_refine);

    /* ── Scale + Fused/non-fused dispatch ────────────────────────────────────
     * Scale grid/block configuration shared by both fused and non-fused paths.
     * Gate: fused kernel is used when pm.t_fused_ms < pm.t_int8_gemms_ms + pm.t_accum_ms.
     * (Scale always runs — its time is excluded from the gate comparison.)
     *
     * Fused path:  scale → oz2_fused_TN_kernel (reads INT8 A8i/B8i → writes FP64 D)
     * Non-fused:   scale → hipblasLtMatmul (INT8 GEMM) → accum/finalize kernels        */
    const dim3 blk_scale_T(OZ2_SCALE_TILE_K * OZ2_SCALE_COALESC_TILE_M);  /* 512  */
    const dim3 blk_scale_N(OZ2_SCALE_TILE_K * OZ2_SCALE_SHMEM_TILE_M);    /* 1024 */
    const unsigned k_x_blks   = static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K);
    const unsigned k_x_blks_c = static_cast<unsigned>((k + 2u * OZ2_SCALE_TILE_K - 1u) / (2u * OZ2_SCALE_TILE_K));
    const dim3 g_scale_A_T(k_x_blks_c, static_cast<unsigned>((m + OZ2_SCALE_COALESC_TILE_M - 1) / OZ2_SCALE_COALESC_TILE_M));
    const dim3 g_scale_A_N(k_x_blks,   static_cast<unsigned>((m + OZ2_SCALE_SHMEM_TILE_M    - 1) / OZ2_SCALE_SHMEM_TILE_M));
    const dim3 g_scale_B_N(k_x_blks_c, static_cast<unsigned>((n + OZ2_SCALE_COALESC_TILE_M - 1) / OZ2_SCALE_COALESC_TILE_M));
    const dim3 g_scale_B_T(k_x_blks,   static_cast<unsigned>((n + OZ2_SCALE_SHMEM_TILE_M    - 1) / OZ2_SCALE_SHMEM_TILE_M));
    const size_t strideA8i = lda8i * cola8i;
    const size_t strideB8i = ldb8i * static_cast<size_t>(n);

#define OZ2_SCALE_LAUNCH_A(TC, SS) \
    do { if(tA) \
        hipLaunchKernelGGL((oz2_scale_A_T_kernel<(TC)>), g_scale_A_T, blk_scale_T, 0, stream, \
                           A, m, lda, A8i, lda8i, cola8i, sftA, k, (SS)); \
    else \
        hipLaunchKernelGGL((oz2_scale_A_N_kernel<(TC)>), g_scale_A_N, blk_scale_N, 0, stream, \
                           A, m, lda, A8i, lda8i, cola8i, sftA, k, (SS)); \
    } while(0)
#define OZ2_SCALE_LAUNCH_B(TC, SS) \
    do { if(!tB) \
        hipLaunchKernelGGL((oz2_scale_B_N_kernel<(TC)>), g_scale_B_N, blk_scale_T, 0, stream, \
                           B, n, ldb, B8i, ldb8i, sftB, k, (SS)); \
    else \
        hipLaunchKernelGGL((oz2_scale_B_T_kernel<(TC)>), g_scale_B_T, blk_scale_N, 0, stream, \
                           B, n, ldb, B8i, ldb8i, sftB, k, (SS)); \
    } while(0)
    /* ── Scale dispatch lambda ─────────────────────────────────────────────── */
    /* Shared by both fused and non-fused paths to avoid duplicating the
     * 18-case switch.  Launches A and B scale kernels for one chunk and
     * accumulates elapsed time into _t_scale.                                   */
    auto launch_scale_chunk = [&](unsigned sc_start, unsigned sc_count) {
        _pstart();
        switch(sc_count) {
            case  1: OZ2_SCALE_LAUNCH_A( 1, sc_start); OZ2_SCALE_LAUNCH_B( 1, sc_start); break;
            case  2: OZ2_SCALE_LAUNCH_A( 2, sc_start); OZ2_SCALE_LAUNCH_B( 2, sc_start); break;
            case  3: OZ2_SCALE_LAUNCH_A( 3, sc_start); OZ2_SCALE_LAUNCH_B( 3, sc_start); break;
            case  4: OZ2_SCALE_LAUNCH_A( 4, sc_start); OZ2_SCALE_LAUNCH_B( 4, sc_start); break;
            case  5: OZ2_SCALE_LAUNCH_A( 5, sc_start); OZ2_SCALE_LAUNCH_B( 5, sc_start); break;
            case  6: OZ2_SCALE_LAUNCH_A( 6, sc_start); OZ2_SCALE_LAUNCH_B( 6, sc_start); break;
            case  7: OZ2_SCALE_LAUNCH_A( 7, sc_start); OZ2_SCALE_LAUNCH_B( 7, sc_start); break;
            case  8: OZ2_SCALE_LAUNCH_A( 8, sc_start); OZ2_SCALE_LAUNCH_B( 8, sc_start); break;
            case  9: OZ2_SCALE_LAUNCH_A( 9, sc_start); OZ2_SCALE_LAUNCH_B( 9, sc_start); break;
            case 10: OZ2_SCALE_LAUNCH_A(10, sc_start); OZ2_SCALE_LAUNCH_B(10, sc_start); break;
            case 11: OZ2_SCALE_LAUNCH_A(11, sc_start); OZ2_SCALE_LAUNCH_B(11, sc_start); break;
            case 12: OZ2_SCALE_LAUNCH_A(12, sc_start); OZ2_SCALE_LAUNCH_B(12, sc_start); break;
            case 13: OZ2_SCALE_LAUNCH_A(13, sc_start); OZ2_SCALE_LAUNCH_B(13, sc_start); break;
            case 14: OZ2_SCALE_LAUNCH_A(14, sc_start); OZ2_SCALE_LAUNCH_B(14, sc_start); break;
            case 15: OZ2_SCALE_LAUNCH_A(15, sc_start); OZ2_SCALE_LAUNCH_B(15, sc_start); break;
            case 16: OZ2_SCALE_LAUNCH_A(16, sc_start); OZ2_SCALE_LAUNCH_B(16, sc_start); break;
            case 17: OZ2_SCALE_LAUNCH_A(17, sc_start); OZ2_SCALE_LAUNCH_B(17, sc_start); break;
            case 18: OZ2_SCALE_LAUNCH_A(18, sc_start); OZ2_SCALE_LAUNCH_B(18, sc_start); break;
            default: break;
        }
        _pstop(_t_scale);
    };

    bool took_fused_path = false;
    rocblaslt_status fused_st = rocblaslt_status_success;

    {
        const int dev = reinterpret_cast<const _rocblaslt_handle*>(settings.handle)->device;
        if (oz2_get_perf_model_params(dev).has_value()) {
            const Fp64PerfModelTimes pm =
                fp64EmulationPerfModelTimes(tA, tB, m, n, k, num_moduli, dev);
            /* Gate: fused replaces only INT8 GEMM + accum; scale always runs.
             * Works for all transpose combinations (A8i/B8i always in canonical format).
             * HIPBLASLT_EMULATION_FUSED=off disables the fused path entirely;
             * HIPBLASLT_EMULATION_FUSED=on/force forces it regardless of perf model. */
            const Oz2FusedMode fused_mode = oz2_fused_mode();
            if (fused_mode != Oz2FusedMode::OFF &&
                (fused_mode == Oz2FusedMode::ON ||
                 (pm.t_fused_ms > 0.0 &&
                  pm.t_fused_ms < pm.t_int8_gemms_ms + pm.t_accum_ms))) {
                /* Zero A8i/B8i workspace padding so the fused kernel's double-buffer
                 * prefetch reads zeros beyond k_int. */
                /* Zero workspace padding so the fused kernel's double-buffer prefetch
                 * reads zeros beyond k.  OZ2_FUSED_KBLK_LOAD_MAX is the maximum
                 * KBLK_LOAD across all kernel variants (TILE=16 and TILE=32).      */
                if (static_cast<int>(k) % static_cast<int>(OZ2_FUSED_KBLK_LOAD_MAX) != 0) {
                    (void)hipMemsetAsync(A8i, 0, szA8i, stream);
                    (void)hipMemsetAsync(B8i, 0, szB8i, stream);
                }
                /* Fused path: run scale first, then MFMA+CRT fused kernel. */
                for (unsigned scale_start = 0; scale_start < num_moduli; scale_start += scale_chunk_size) {
                    const unsigned scale_start_as = (scale_start + scale_chunk_size <= num_moduli)
                                                    ? scale_chunk_size : (num_moduli - scale_start);
                    launch_scale_chunk(scale_start, scale_start_as);
                }
                /* Fused MFMA+CRT kernel: reads A8i/B8i, writes D directly. */
                _pstart();
                fused_st = oz2_launch_fused_TN(A8i, B8i, lda8i, cola8i, ldb8i,
                                               C, D, m, n, k, ldc, ldd,
                                               *alpha, *beta, sftA, sftB, num_moduli, stream);
                _pstop(_t_fused);
                took_fused_path = true;
            }
        }
    }

    if (!took_fused_path) {
    /* Non-fused path: scale + hipblasLtMatmul (INT8 GEMM) + accum/finalize. */
    const dim3 blk_acc(64, 8);
    const dim3 grid_acc((m + 63) / 64, (n + 7) / 8);

    int32_t       batch_cur  = static_cast<int32_t>(chunk_size);
    const int64_t stride_A_b = static_cast<int64_t>(strideA8i);
    const int64_t stride_B_b = static_cast<int64_t>(strideB8i);
    const int64_t stride_C_b = static_cast<int64_t>(szC32i);
    if(hipblasLtMatrixLayoutSetAttribute(layoutA_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutSetAttribute(layoutA_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A_b, sizeof(stride_A_b)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutSetAttribute(layoutB_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutSetAttribute(layoutB_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B_b, sizeof(stride_B_b)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutSetAttribute(layoutCD_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();
    if(hipblasLtMatrixLayoutSetAttribute(layoutCD_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C_b, sizeof(stride_C_b)) != HIPBLAS_STATUS_SUCCESS)
        return fail_internal();

    for(unsigned scale_start = 0; scale_start < num_moduli; scale_start += scale_chunk_size) {
        const unsigned actual_scale = (scale_start + scale_chunk_size <= num_moduli)
                                      ? scale_chunk_size : (num_moduli - scale_start);
        /* Scale: one call dispatches all 18 cases via the shared lambda. */
        launch_scale_chunk(scale_start, actual_scale);

        for(unsigned gemm_local = 0; gemm_local < actual_scale; gemm_local += chunk_size) {
            const unsigned actual_gemm = (gemm_local + chunk_size <= actual_scale)
                                         ? chunk_size : (actual_scale - gemm_local);
            if(static_cast<int32_t>(actual_gemm) != batch_cur) {
                batch_cur = static_cast<int32_t>(actual_gemm);
                if(hipblasLtMatrixLayoutSetAttribute(layoutA_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur)) != HIPBLAS_STATUS_SUCCESS)
                    return fail_internal();
                if(hipblasLtMatrixLayoutSetAttribute(layoutB_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur)) != HIPBLAS_STATUS_SUCCESS)
                    return fail_internal();
                if(hipblasLtMatrixLayoutSetAttribute(layoutCD_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_cur, sizeof(batch_cur)) != HIPBLAS_STATUS_SUCCESS)
                    return fail_internal();
            }
            const int8_t* const A8i_gemm = A8i + gemm_local * strideA8i;
            const int8_t* const B8i_gemm = B8i + gemm_local * strideB8i;
            _pstart();
            if(hipblasLtMatmul(settings.handle, matmulDesc,
                               &one_i, A8i_gemm, layoutA_b, B8i_gemm, layoutB_b,
                               &zero_i, C32i_batch, layoutCD_b, C32i_batch, layoutCD_b,
                               nullptr, nullptr, 0, stream)
               != HIPBLAS_STATUS_SUCCESS)
                return fail_internal();
            _pstop(_t_int8);

            const unsigned global_chunk_start = scale_start + gemm_local;
            const bool is_first = (global_chunk_start == 0);
            const bool is_last  = (global_chunk_start + actual_gemm == num_moduli);
            _pstart();
#define OZ2_FARGS C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd, *alpha, *beta, sftA, sftB, global_chunk_start
#define OZ2_AARGS C32i_batch, Zhi, Zlo, m, n, ldc32i, global_chunk_start
#define OZ2_FINALIZE(HL, CS) \
            do { if(is_first) hipLaunchKernelGGL((oz2_accum_finalize_kernel<(HL),(CS),true>),  grid_acc, blk_acc, 0, stream, OZ2_FARGS); \
                 else         hipLaunchKernelGGL((oz2_accum_finalize_kernel<(HL),(CS),false>), grid_acc, blk_acc, 0, stream, OZ2_FARGS); } while(0)
#define OZ2_ACCUM(HL, CS) \
            do { if(is_first) hipLaunchKernelGGL((oz2_chunk_accum_kernel<(HL),(CS),true>),  grid_acc, blk_acc, 0, stream, OZ2_AARGS); \
                 else         hipLaunchKernelGGL((oz2_chunk_accum_kernel<(HL),(CS),false>), grid_acc, blk_acc, 0, stream, OZ2_AARGS); } while(0)
            if(is_last) {
                if(num_moduli <= 7u) {
                    switch(actual_gemm) {
                        case  1: OZ2_FINALIZE(false,  1); break; case  2: OZ2_FINALIZE(false,  2); break;
                        case  3: OZ2_FINALIZE(false,  3); break; case  4: OZ2_FINALIZE(false,  4); break;
                        case  5: OZ2_FINALIZE(false,  5); break; case  6: OZ2_FINALIZE(false,  6); break;
                        case  7: OZ2_FINALIZE(false,  7); break; case  8: OZ2_FINALIZE(false,  8); break;
                        case  9: OZ2_FINALIZE(false,  9); break; case 10: OZ2_FINALIZE(false, 10); break;
                        case 11: OZ2_FINALIZE(false, 11); break; case 12: OZ2_FINALIZE(false, 12); break;
                        case 13: OZ2_FINALIZE(false, 13); break; case 14: OZ2_FINALIZE(false, 14); break;
                        case 15: OZ2_FINALIZE(false, 15); break; case 16: OZ2_FINALIZE(false, 16); break;
                        case 17: OZ2_FINALIZE(false, 17); break; case 18: OZ2_FINALIZE(false, 18); break;
                        default: hipLaunchKernelGGL((oz2_accum_finalize_kernel_rt<false>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                                     *alpha, *beta, sftA, sftB, global_chunk_start, actual_gemm, is_first);
                    }
                } else {
                    switch(actual_gemm) {
                        case  1: OZ2_FINALIZE(true,  1); break; case  2: OZ2_FINALIZE(true,  2); break;
                        case  3: OZ2_FINALIZE(true,  3); break; case  4: OZ2_FINALIZE(true,  4); break;
                        case  5: OZ2_FINALIZE(true,  5); break; case  6: OZ2_FINALIZE(true,  6); break;
                        case  7: OZ2_FINALIZE(true,  7); break; case  8: OZ2_FINALIZE(true,  8); break;
                        case  9: OZ2_FINALIZE(true,  9); break; case 10: OZ2_FINALIZE(true, 10); break;
                        case 11: OZ2_FINALIZE(true, 11); break; case 12: OZ2_FINALIZE(true, 12); break;
                        case 13: OZ2_FINALIZE(true, 13); break; case 14: OZ2_FINALIZE(true, 14); break;
                        case 15: OZ2_FINALIZE(true, 15); break; case 16: OZ2_FINALIZE(true, 16); break;
                        case 17: OZ2_FINALIZE(true, 17); break; case 18: OZ2_FINALIZE(true, 18); break;
                        default: hipLaunchKernelGGL((oz2_accum_finalize_kernel_rt<true>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, C, D, m, n, ldc32i, ldc, ldd,
                                     *alpha, *beta, sftA, sftB, global_chunk_start, actual_gemm, is_first);
                    }
                }
            } else {
                if(num_moduli <= 7u) {
                    switch(actual_gemm) {
                        case  1: OZ2_ACCUM(false,  1); break; case  2: OZ2_ACCUM(false,  2); break;
                        case  3: OZ2_ACCUM(false,  3); break; case  4: OZ2_ACCUM(false,  4); break;
                        case  5: OZ2_ACCUM(false,  5); break; case  6: OZ2_ACCUM(false,  6); break;
                        case  7: OZ2_ACCUM(false,  7); break; case  8: OZ2_ACCUM(false,  8); break;
                        case  9: OZ2_ACCUM(false,  9); break; case 10: OZ2_ACCUM(false, 10); break;
                        case 11: OZ2_ACCUM(false, 11); break; case 12: OZ2_ACCUM(false, 12); break;
                        case 13: OZ2_ACCUM(false, 13); break; case 14: OZ2_ACCUM(false, 14); break;
                        case 15: OZ2_ACCUM(false, 15); break; case 16: OZ2_ACCUM(false, 16); break;
                        case 17: OZ2_ACCUM(false, 17); break; case 18: OZ2_ACCUM(false, 18); break;
                        default: hipLaunchKernelGGL((oz2_chunk_accum_kernel_rt<false>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, m, n, ldc32i, global_chunk_start, actual_gemm, is_first);
                    }
                } else {
                    switch(actual_gemm) {
                        case  1: OZ2_ACCUM(true,  1); break; case  2: OZ2_ACCUM(true,  2); break;
                        case  3: OZ2_ACCUM(true,  3); break; case  4: OZ2_ACCUM(true,  4); break;
                        case  5: OZ2_ACCUM(true,  5); break; case  6: OZ2_ACCUM(true,  6); break;
                        case  7: OZ2_ACCUM(true,  7); break; case  8: OZ2_ACCUM(true,  8); break;
                        case  9: OZ2_ACCUM(true,  9); break; case 10: OZ2_ACCUM(true, 10); break;
                        case 11: OZ2_ACCUM(true, 11); break; case 12: OZ2_ACCUM(true, 12); break;
                        case 13: OZ2_ACCUM(true, 13); break; case 14: OZ2_ACCUM(true, 14); break;
                        case 15: OZ2_ACCUM(true, 15); break; case 16: OZ2_ACCUM(true, 16); break;
                        case 17: OZ2_ACCUM(true, 17); break; case 18: OZ2_ACCUM(true, 18); break;
                        default: hipLaunchKernelGGL((oz2_chunk_accum_kernel_rt<true>), grid_acc, blk_acc, 0, stream,
                                     C32i_batch, Zhi, Zlo, m, n, ldc32i, global_chunk_start, actual_gemm, is_first);
                    }
                }
            }
#undef OZ2_FARGS
#undef OZ2_AARGS
#undef OZ2_FINALIZE
#undef OZ2_ACCUM
            _pstop(_t_accum);
        }
    }

    } /* end if (!took_fused_path) */

#undef OZ2_SCALE_LAUNCH_B
#undef OZ2_SCALE_LAUNCH_A

    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(layoutCD);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutA);

    if(_prof) {
        /* Accumulate component times into the caller's accumulator. */
        prof->t_prelim      += _t_prelim;
        prof->t_prelim_gemm += _t_prelim_gemm;
        prof->t_extract     += _t_extract;
        prof->t_refine      += _t_refine;
        prof->t_fused       += _t_fused;
        prof->t_scale       += _t_scale;
        prof->t_int8        += _t_int8;
        prof->t_accum       += _t_accum;
        prof->t_finalize    += _t_finalize;
        prof->n_sub_gemms   += 1u;
        (void)hipEventDestroy(_ev1);
        (void)hipEventDestroy(_ev0);
    }
    return took_fused_path ? fused_st : rocblaslt_status_success;
}

/* =========================================================================
 * fp64EmulatedGemm — public wrapper
 *
 * Owns the profiling accumulator.  Records a single HIP event pair around
 * the entire call (including all recursive sub-GEMMs) to measure the true
 * GPU wall-clock time, then writes one summary CSV row with:
 *   – summed component times across all leaf sub-GEMMs
 *   – the measured t_total_ms for the full call
 *   – num_sub_gemms (number of monolithic leaf calls executed)
 * ========================================================================= */
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
                                  const Fp64EmulationSettings& settings)
{
    /* ── Pre-allocate a single workspace for the entire call ──────────────────
     * fp64EmulatedGemmImpl recurses for split shapes; without a pre-allocated
     * buffer each leaf sub-GEMM would do its own hipMallocAsync/hipFreeAsync.
     * Allocating once here and passing it through settings eliminates that
     * overhead (e.g. 16 redundant alloc/free pairs for the 65K square case).
     * If the caller already provided a sufficient workspace we use it as-is.  */
    const unsigned num_moduli = (settings.num_moduli >= 2u && settings.num_moduli <= OZ2_S_MAX)
                                    ? settings.num_moduli : fp64EmulationNumModuli();
    const int device = reinterpret_cast<const _rocblaslt_handle*>(settings.handle)->device;
    const size_t wsNeeded = fp64EmulationWorkspaceSize(
                                reinterpret_cast<const _rocblaslt_handle*>(settings.handle),
                                opA, opB, m, n, k, num_moduli);

    Fp64EmulationSettings effectiveSettings = settings;
    void* ws_toplevel = nullptr;

    if(wsNeeded > 0 &&
       (effectiveSettings.workspace == nullptr ||
        effectiveSettings.workspace_bytes < wsNeeded))
    {
        if(hipMallocAsync(&ws_toplevel, wsNeeded, stream) != hipSuccess)
            return rocblaslt_status_memory_error;
        effectiveSettings.workspace       = ws_toplevel;
        effectiveSettings.workspace_bytes = wsNeeded;
    }
    /* ────────────────────────────────────────────────────────────────────── */

    const char* const _pf   = oz2_profile_file();
    const bool        _prof = (_pf != nullptr);

    Fp64ProfileAccum accum{};
    hipEvent_t ev_start{}, ev_end{};
    if(_prof) {
        (void)hipEventCreate(&ev_start);
        (void)hipEventCreate(&ev_end);
        (void)hipEventRecord(ev_start, stream);
    }

    const rocblaslt_status st =
        fp64EmulatedGemmImpl(opA, opB, m, n, k, alpha, A, lda, B, ldb,
                             beta, C, ldc, D, ldd, stream, effectiveSettings,
                             _prof ? &accum : nullptr);

    /* Release the top-level workspace now that all leaves have finished.    */
    if(ws_toplevel != nullptr)
        (void)hipFreeAsync(ws_toplevel, stream);

    if(_prof) {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        const unsigned chunk_size = oz2_compute_chunk_size(m, n, num_moduli);
        const unsigned scale_chunk_size =
            oz2_compute_scale_chunk_size(m, n, k, num_moduli, chunk_size);
        const bool tA = (opA != HIPBLAS_OP_N);
        const bool tB = (opB != HIPBLAS_OP_N);
        /* Use the split-aware model: each component is the sum across all leaves.
         * t_native_ms remains for the original (m,n,k) problem.           */
        const Fp64PerfModelTimes pm = oz2_effective_perf_model_times(tA, tB, m, n, k, num_moduli, device);

        std::FILE* _f = std::fopen(_pf, "a");
        if(_f) {
            if(std::ftell(_f) == 0)
                std::fprintf(_f,
                    "m,n,k,transA,transB,num_moduli,scale_chunk_size,gemm_chunk_size,"
                    "workspace_bytes,num_sub_gemms,"
                    "t_prelim_ms,t_prelim_gemm_ms,t_extract_ms,t_refine_ms,"
                    "t_fused_ms,t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                    "t_finalize_ms,t_total_ms,"
                    "pred_prelim_ms,pred_prelim_gemm_ms,pred_refine_ms,"
                    "pred_scale_ms,pred_int8_gemm_ms,pred_accum_ms,"
                    "pred_host_ms,pred_launch_ms,pred_fused_ms,pred_total_ms,pred_native_dgemm_ms\n");
            std::fprintf(_f,
                "%lld,%lld,%lld,%c,%c,%u,%u,%u,"
                "%llu,%u,"
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                (long long)m, (long long)n, (long long)k,
                tA ? 'T' : 'N', tB ? 'T' : 'N',
                num_moduli, scale_chunk_size, chunk_size,
                (unsigned long long)wsNeeded, accum.n_sub_gemms,
                accum.t_prelim, accum.t_prelim_gemm, accum.t_extract, accum.t_refine,
                accum.t_fused, accum.t_scale, accum.t_int8, accum.t_accum, accum.t_finalize, t_total,
                pm.t_prelim_ms, pm.t_prelim_gemm_ms, pm.t_refine_ms,
                pm.t_scale_ms, pm.t_int8_gemms_ms, pm.t_accum_ms,
                pm.t_host_ms, pm.t_launch_ms, pm.t_fused_ms, pm.t_total_ms, pm.t_native_ms);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}
