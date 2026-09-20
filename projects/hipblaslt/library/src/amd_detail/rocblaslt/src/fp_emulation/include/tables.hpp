// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/*
 * tables.hpp
 *
 * Constexpr device functions returning the CRT constant data used by the
 * fixed-point emulation kernels (FP32 and FP64).  Embedding the data here
 * (rather than in __constant__ globals) avoids all cross-translation-unit
 * device-symbol issues and requires no initialization step.
 *
 * Naming convention for the s-dependent functions:
 *   s_idx = num_moduli - 2  (0-based index into the per-s tables)
 *
 * This file may be included by any HIP translation unit; it is intentionally
 * separate from emulation.hpp (which must stay free of __device__
 * code so it can be included from plain C++ units).
 *
 * Constants (tables) are base on the open-source GEMMul8 implementation
 * (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8).
 */

#include <array>
#include <cmath>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace FixedPointEmulation
{
    /* Maximum number of moduli for FP64 (full precision range). */
    static constexpr unsigned S_MAX_FP64 = 20u;
    /* Maximum number of moduli for FP32 (reduced range; 12 moduli give
     * ~46 bits of CRT capacity, far exceeding FP32's 23-bit precision). */
    static constexpr unsigned S_MAX_FP32 = 12u;
    /* S_MAX is an alias for S_MAX_FP64 and is used for table sizing only. */
    static constexpr unsigned S_MAX = S_MAX_FP64;

    /* Returns the maximum number of moduli for the given floating-point type T.
     * Use these functions in kernels and workspace-size calculations instead of
     * the raw constants, so that FP32 paths use the smaller S_MAX_FP32. */
    template <typename T>
    __host__ __device__ constexpr unsigned max_num_moduli() noexcept
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "max_num_moduli<T>: T must be float or double");
        return S_MAX_FP64; // specializations below override for float and double
    }
    template <>
    __host__ __device__ constexpr unsigned max_num_moduli<float>() noexcept
    {
        return S_MAX_FP32;
    }
    template <>
    __host__ __device__ constexpr unsigned max_num_moduli<double>() noexcept
    {
        return S_MAX_FP64;
    }

    /* hipDataType overload for non-templated callers. */
    inline __host__ __device__ constexpr unsigned max_num_moduli(hipDataType t) noexcept
    {
        return (t == HIP_R_32F) ? S_MAX_FP32 : (t == HIP_R_64F) ? S_MAX_FP64 : 0u;
    }

    /* Alignment for INT8 arrays (128 bytes = 128 INT8 elements). */
    static constexpr size_t ALIGN = 128u;

    /* Round n up to the nearest multiple of ALIGN. */
    inline __host__ __device__ size_t pad(size_t n) noexcept
    {
        return (n + ALIGN - 1u) / ALIGN * ALIGN;
    }

    /* The S_MAX pairwise-coprime CRT moduli m_i, as positive integers.
     * All derived tables (neg_mod, inv_mod, inv_mod_f) are computed from
     * these values. */
    static constexpr int moduli[S_MAX] = {
        256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
        223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
    };

    /* Negative of the i-th CRT modulus m_i, i.e. -m_i, as a double. */
    __device__ __forceinline__ constexpr double neg_mod(unsigned i) noexcept
    {
        return -(double)moduli[i];
    }

    /* Nearest double to 1/m_i, derived at compile time from moduli[].
     * Used in the fast reduction r_i = X - m_i * round(X * inv_mod[i]). */
    __device__ __forceinline__ constexpr double inv_mod(unsigned i) noexcept
    {
        constexpr auto v = []() constexpr {
            std::array<double, S_MAX> a{};
            for(unsigned k = 0; k < S_MAX; ++k)
                a[k] = 1.0 / (double)moduli[k];
            return a;
        }();
        return v[i];
    }

    /* Nearest float to 1/m_i, derived at compile time from inv_mod[].
     * Used in the fast reduction r_i = X - m_i * round(X * inv_mod_f[i]). */
    __device__ __forceinline__ constexpr float inv_mod_f(unsigned i) noexcept
    {
        constexpr auto v = []() constexpr {
            std::array<float, S_MAX> a{};
            for(unsigned k = 0; k < S_MAX; ++k)
                a[k] = static_cast<float>(inv_mod(k));
            return a;
        }();
        return v[i];
    }

    // UInt256: compile-time 256-bit unsigned integer (4 x uint64_t, little-endian).
    struct UInt256
    {
        uint64_t d[4]{};
        constexpr UInt256() noexcept = default;
        constexpr explicit UInt256(uint64_t v) noexcept
            : d{v, 0, 0, 0}
        {
        }

        constexpr bool bit(int p) const noexcept
        {
            if(p < 0 || p >= 256)
                return false;
            return (d[p >> 6] >> (p & 63)) & 1u;
        }

        constexpr int msb_pos() const noexcept
        {
            for(int i = 3; i >= 0; --i)
            {
                if(d[i] != 0)
                {
                    uint64_t v = d[i];
                    int      b = 0;
                    if(v >> 32)
                    {
                        b += 32;
                        v >>= 32;
                    }
                    if(v >> 16)
                    {
                        b += 16;
                        v >>= 16;
                    }
                    if(v >> 8)
                    {
                        b += 8;
                        v >>= 8;
                    }
                    if(v >> 4)
                    {
                        b += 4;
                        v >>= 4;
                    }
                    if(v >> 2)
                    {
                        b += 2;
                        v >>= 2;
                    }
                    b += (int)(v >> 1);
                    return i * 64 + b;
                }
            }
            return -1;
        }

        constexpr uint64_t extract_bits(int high, int low) const noexcept
        {
            int      ll = low >> 6, lo = low & 63, hl = high >> 6;
            uint64_t r = d[ll] >> lo;
            if(lo != 0 && hl > ll && hl < 4)
                r |= d[hl] << (64 - lo);
            int len = high - low + 1;
            if(len < 64)
                r &= (1ULL << len) - 1u;
            return r;
        }

        constexpr bool any_bit_set_below(int p) const noexcept
        {
            if(p < 0)
                return false;
            int limb = p >> 6, bo = p & 63;
            for(int i = 0; i < limb && i < 4; ++i)
                if(d[i])
                    return true;
            if(limb < 4)
            {
                uint64_t mask = (bo < 63) ? ((1ULL << (bo + 1)) - 1u) : ~uint64_t(0);
                if(d[limb] & mask)
                    return true;
            }
            return false;
        }

        constexpr UInt256 mul(uint32_t rhs) const noexcept
        {
            UInt256  res;
            uint64_t carry = 0;
            for(int i = 0; i < 4; ++i)
            {
                uint64_t lo = d[i] & 0xFFFFFFFFu, hi = d[i] >> 32;
                uint64_t p0 = lo * rhs + (carry & 0xFFFFFFFFu),
                         p1 = hi * rhs + (carry >> 32) + (p0 >> 32);
                res.d[i]    = (p0 & 0xFFFFFFFFu) | ((p1 & 0xFFFFFFFFu) << 32);
                carry       = p1 >> 32;
            }
            return res;
        }

        constexpr UInt256 shl(int n) const noexcept
        {
            if(n == 0)
                return *this;
            UInt256 res;
            int     ls = n >> 6, bs = n & 63;
            for(int i = 3; i >= 0; --i)
            {
                int src = i - ls;
                if(src < 0)
                    res.d[i] = 0;
                else if(bs == 0)
                    res.d[i] = d[src];
                else
                {
                    res.d[i] = d[src] << bs;
                    if(src > 0)
                        res.d[i] |= d[src - 1] >> (64 - bs);
                }
            }
            return res;
        }

        constexpr UInt256 shr(int n) const noexcept
        {
            if(n == 0)
                return *this;
            UInt256 res;
            int     ls = n >> 6, bs = n & 63;
            for(int i = 0; i < 4; ++i)
            {
                int src = i + ls;
                if(src >= 4)
                    res.d[i] = 0;
                else if(bs == 0)
                    res.d[i] = d[src];
                else
                {
                    res.d[i] = d[src] >> bs;
                    if(src + 1 < 4)
                        res.d[i] |= d[src + 1] << (64 - bs);
                }
            }
            return res;
        }

        constexpr bool ge(const UInt256& r) const noexcept
        {
            for(int i = 3; i >= 0; --i)
            {
                if(d[i] > r.d[i])
                    return true;
                if(d[i] < r.d[i])
                    return false;
            }
            return true;
        }

        constexpr UInt256 sub(const UInt256& r) const noexcept
        {
            UInt256  res;
            uint64_t borrow = 0;
            for(int i = 0; i < 4; ++i)
            {
                uint64_t a = d[i], b = r.d[i], d1 = a - borrow, b1 = (a < borrow) ? 1u : 0u,
                         d2 = d1 - b, b2 = (d1 < b) ? 1u : 0u;
                res.d[i] = d2;
                borrow   = b1 + b2;
            }
            return res;
        }

        // Exact division by a 32-bit divisor (caller must ensure zero remainder).
        constexpr UInt256 div(uint32_t rhs) const noexcept
        {
            UInt256  q;
            uint64_t rem = 0;
            for(int i = 3; i >= 0; --i)
            {
                uint64_t hi = (rem << 32) | (d[i] >> 32);
                uint64_t qh = hi / rhs, rh = hi % rhs;
                uint64_t lo = (rh << 32) | (d[i] & 0xFFFFFFFFu);
                uint64_t ql = lo / rhs;
                q.d[i]      = (qh << 32) | ql;
                rem         = lo % rhs;
            }
            return q;
        }

        // Compute this mod rhs (32-bit divisor), returning remainder.
        constexpr uint32_t mod(uint32_t rhs) const noexcept
        {
            uint64_t rem = 0;
            for(int i = 3; i >= 0; --i)
            {
                uint64_t hi = (rem << 32) | (d[i] >> 32);
                uint64_t rh = hi % rhs;
                uint64_t lo = (rh << 32) | (d[i] & 0xFFFFFFFFu);
                rem         = lo % rhs;
            }
            return static_cast<uint32_t>(rem);
        }

        constexpr double to_double() const noexcept
        {
            int top = msb_pos();
            if(top < 0)
                return 0.0;
            if(top <= 52)
                return static_cast<double>(d[0]);
            int      shift = top - 52;
            uint64_t m     = extract_bits(top, top - 52);
            bool     rb = bit(top - 53), st = any_bit_set_below(top - 54);
            if(rb && (st || (m & 1u)))
                ++m;
            double r = static_cast<double>(m);
            int    s = shift;
            while(s >= 32)
            {
                r *= 4294967296.0;
                s -= 32;
            }
            if(s > 0)
                r *= static_cast<double>(1u << s);
            return r;
        }

        // Like to_double(), but forces QPI_N_TZ trailing zeros in the 53-bit
        // mantissa so that dc * result is FP64-exact for all |dc| <= 127.
        //
        // Algorithm (from GEMMul8, Y. Uchino, RIKEN R-CCS):
        //   1. Round Q to the nearest 53-bit double (round-to-nearest-even).
        //   2. Truncate the stored 53-bit mantissa by QPI_N_TZ bits (floor).
        //   3. If the result exceeds Q (can happen when step 1 rounded UP and
        //      the increment crossed an alignment boundary), subtract one
        //      alignment unit to restore the guarantee result <= Q.
        //
        // This ensures qpi_hi <= qpi_exact, so qpi_lo = qpi_exact - qpi_hi >= 0.
        // Non-negative qpi_lo avoids cancellation in the FMA accumulation of
        // chunk_accum_kernel.  QPI_N_TZ = ceil(log2(rho)) where
        // rho = sum(floor(moduli[t]/2)) for all S_MAX moduli; this makes
        // dc * qpi_hi FP64-exact for all balanced residues |dc| <= max(moduli)/2.
        //
        // For small values (top <= 52): the CRT coefficient is an exact integer
        // that fits in a double without rounding, so return it exactly (qpi_lo = 0).
        // This matches the GEMMul8 convention and is required because the kernel
        // uses HAS_LO=false (ignoring qpi_lo) when effective_s <= 7.
        static constexpr int QPI_N_TZ = 12; // ceil(log2(2188)), where 2188 = sum(floor(m/2))
        constexpr double     to_double_aligned() const noexcept
        {
            int top = msb_pos();
            if(top < 0)
                return 0.0;
            // For top <= 52 the value is an exact integer < 2^53, perfectly
            // representable as a double.  Return it exactly so qpi_lo = 0.
            if(top <= 52)
                return static_cast<double>(d[0]);
            int      shift = top - 52;
            uint64_t m     = extract_bits(top, top - 52);
            bool     rb = bit(top - 53), st = any_bit_set_below(top - 54);
            if(rb && (st || (m & 1u)))
                ++m;
            if(m >= (1ULL << 53))
            {
                m >>= 1;
                ++shift;
            }
            // Truncate the 53-bit mantissa by QPI_N_TZ bits (floor).
            uint64_t ma = (m >> QPI_N_TZ) << QPI_N_TZ;
            // Step 3 (GEMMul8 algorithm): overshoot correction.
            // If ma * 2^shift > Q (can happen when step 1 rounded UP and the carry
            // crossed an alignment boundary), subtract one alignment unit to ensure
            // qpi_hi <= qpi_exact, guaranteeing qpi_lo = qpi_exact - qpi_hi >= 0.
            {
                UInt256 ms = UInt256(ma).shl(shift);
                if(!ge(ms))
                    ma -= (1u << QPI_N_TZ);
            }
            double r = static_cast<double>(ma);
            int    s = shift;
            while(s >= 32)
            {
                r *= 4294967296.0;
                s -= 32;
            }
            if(s > 0)
                r *= static_cast<double>(1u << s);
            return r;
        }
    };

    // Compute qpi_lo = Q - to_double_aligned(Q) as a double.
    // Uses the same GEMMul8 algorithm as to_double_aligned(), so qpi_lo >= 0 always.
    constexpr double qpi_lo_val(const UInt256& Q) noexcept
    {
        int top = Q.msb_pos();
        if(top < 0)
            return 0.0;
        // to_double_aligned() returns d[0] exactly for top <= 52,
        // so qpi_lo = Q - qpi_hi = 0 in that case.
        if(top <= 52)
            return 0.0;
        int      shift = top - 52;
        uint64_t m     = Q.extract_bits(top, top - 52);
        bool     rb = Q.bit(top - 53), st = Q.any_bit_set_below(top - 54);
        if(rb && (st || (m & 1u)))
            ++m;
        if(m >= (1ULL << 53))
        {
            m >>= 1;
            ++shift;
        }
        uint64_t ma = (m >> UInt256::QPI_N_TZ) << UInt256::QPI_N_TZ;
        UInt256  ms = UInt256(ma).shl(shift);
        // Ensure ms <= Q (the overshoot case: when nearest-double rounded UP
        // and the increment crossed an alignment boundary).
        if(!Q.ge(ms))
        {
            ma -= (1u << UInt256::QPI_N_TZ);
            ms = UInt256(ma).shl(shift);
        }
        UInt256 diff = Q.sub(ms); // Q >= ms guaranteed
        return static_cast<double>(diff.d[0])
               + static_cast<double>(diff.d[1]) * 18446744073709551616.0;
    }

    constexpr uint32_t mod_inverse(uint32_t a, uint32_t m) noexcept
    {
        int32_t t = 0, newt = 1, r = (int32_t)m, newr = (int32_t)a;
        while(newr != 0)
        {
            int32_t q = r / newr, tmp = t - q * newt;
            t    = newt;
            newt = tmp;
            tmp  = r - q * newr;
            r    = newr;
            newr = tmp;
        }
        if(t < 0)
            t += (int32_t)m;
        return (uint32_t)t;
    }

    constexpr std::array<UInt256, S_MAX - 1> make_P_table() noexcept
    {
        std::array<UInt256, S_MAX - 1> t{};
        UInt256                        prod(static_cast<uint64_t>(moduli[0]));
        for(unsigned i = 0; i < S_MAX - 1; ++i)
        {
            prod = prod.mul(static_cast<uint32_t>(moduli[i + 1]));
            t[i] = prod;
        }
        return t;
    }

    constexpr double p_lo_double(const UInt256& P) noexcept
    {
        int top = P.msb_pos();
        if(top < 0 || top <= 52)
            return 0.0;
        int      shift = top - 52;
        uint64_t m     = P.extract_bits(top, top - 52);
        bool     rb = P.bit(top - 53), st = P.any_bit_set_below(top - 54);
        if(rb && (st || (m & 1u)))
            ++m;
        UInt256 ms = UInt256(m).shl(shift);
        double  res;
        if(P.ge(ms))
        {
            UInt256 diff = P.sub(ms);
            res          = static_cast<double>(diff.d[0])
                  + static_cast<double>(diff.d[1]) * 18446744073709551616.0;
        }
        else
        {
            UInt256 diff = ms.sub(P);
            res          = -(static_cast<double>(diff.d[0])
                    + static_cast<double>(diff.d[1]) * 18446744073709551616.0);
        }
        return -res;
    }

    // Double-double representation of -P_s = {hi, lo} where hi = nearest_double(-P_s)
    // and lo = -P_s - hi (the rounding residual).  Returns hi and lo as a pair so
    // callers always get both components from a single table lookup, preventing
    // mismatches between independently loaded hi and lo values.
    __device__ __forceinline__ std::pair<double, double> P_dd(unsigned s_idx) noexcept
    {
        static constexpr auto v = []() constexpr {
            auto                                          P = make_P_table();
            std::array<std::pair<double, double>, S_MAX - 1> a{};
            for(unsigned i = 0; i < S_MAX - 1; ++i)
            {
                a[i].first  = -P[i].to_double();
                a[i].second = p_lo_double(P[i]);
            }
            return a;
        }();
        return v[s_idx];
    }

    // Nearest double to 1/P_s.
    // Used in the range-reduction step: q = rint((Zhi + Zlo) * inv_P(s))
    // folds the accumulated CRT sum Z into the symmetric range (-P_s/2, P_s/2].
    __device__ __forceinline__ double inv_P(unsigned s_idx) noexcept
    {
        static constexpr auto v = []() constexpr {
            auto                          P = make_P_table();
            std::array<double, S_MAX - 1> a{};
            for(unsigned i = 0; i < S_MAX - 1; ++i)
                a[i] = 1.0 / P[i].to_double();
            return a;
        }();
        return v[s_idx];
    }

    // qpi_k = (P(s)/m_k) * mod_inverse((P(s)/m_k) mod m_k, m_k)
    // This is the CRT coefficient for index k: summing dc[k]*qpi_k over k
    // reconstructs x from its residues dc[k] = x mod m_k.
    //
    // Returns {hi, lo} where:
    //   hi: first (53-QPI_N_TZ) significant bits of qpi_k (floor, always <= qpi_k),
    //       making dc*hi FP64-exact for all balanced residues |dc| <= max(moduli)/2.
    //   lo: double-double residual qpi_k - hi >= 0.
    //
    // Values are taken verbatim from the GEMMul8 open-source implementation
    // (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8),
    // table INT8::qPi_2.  For s <= 6 (s_idx <= 4) the CRT coefficient is an
    // exact integer < 2^53, so lo = 0 (the HAS_LO=false kernel path applies).
    __device__ __forceinline__ std::pair<double, double> qpi(unsigned s_idx,
                                                              unsigned mod_idx) noexcept
    {
        static constexpr double hi[S_MAX - 1][S_MAX] = {
            /* s=2  (s_idx=0) */
            {0x1.fc02000000000p+15, 0x1.0000000000000p+8},
            /* s=3  (s_idx=1) */
            {0x1.50ac020000000p+23, 0x1.f60c000000000p+22, 0x1.a45a000000000p+23},
            /* s=4  (s_idx=2) */
            {0x1.0688601000000p+28, 0x1.f01e000000000p+28, 0x1.4826900000000p+28,
             0x1.6654440000000p+31},
            /* s=5  (s_idx=3) */
            {0x1.99c1435808000p+37, 0x1.d553914600000p+39, 0x1.cf9d0d8400000p+38,
             0x1.2ff09e4000000p+38, 0x1.dae0172c00000p+39},
            /* s=6  (s_idx=4) */
            {0x1.24d0f0aa6c020p+47, 0x1.00ffb685c4000p+47, 0x1.7820600df8000p+45,
             0x1.b28fb528de000p+47, 0x1.765c060a1c000p+47, 0x1.56b441a210000p+47},
            /* s=7  (s_idx=5) */
            {0x1.49071d4742000p+55, 0x1.5fae947039800p+55, 0x1.42fdb9e194800p+55,
             0x1.187c8ee783400p+55, 0x1.e89ef222a0000p+52, 0x1.0316493fe2400p+55,
             0x1.1f8e561d65000p+53},
            /* s=8  (s_idx=6) */
            {0x1.4f3952ae32400p+63, 0x1.f094cf17cf000p+61, 0x1.0f5bef8d36400p+63,
             0x1.e02e9274c5000p+62, 0x1.a403bd5c1a000p+61, 0x1.a1cf7b99c2800p+62,
             0x1.a54e8a8f42000p+60, 0x1.787fdcb9fa000p+62},
            /* s=9  (s_idx=7) */
            {0x1.9a7c80fe96000p+69, 0x1.43ca2f89db000p+71, 0x1.40f4871424000p+70,
             0x1.2c6790ef15000p+71, 0x1.24d66e4d76000p+70, 0x1.459c5b1ee5800p+71,
             0x1.d43c2b2519000p+70, 0x1.ab93da2aca000p+70, 0x1.dfbe1fda93000p+70},
            /* s=10 (s_idx=8) */
            {0x1.1ba01a9548000p+75, 0x1.b499060d20000p+76, 0x1.8d00367a82000p+77,
             0x1.348f721e1e000p+77, 0x1.09c9ed1acf000p+79, 0x1.6988bc8c28000p+75,
             0x1.4e2df779b8000p+77, 0x1.54302cc6b7000p+78, 0x1.675767107c000p+76,
             0x1.1fdfa04826000p+77},
            /* s=11 (s_idx=9) */
            {0x1.ae4dbe76d7000p+86, 0x1.258185fdee000p+86, 0x1.76fdabbf54000p+85,
             0x1.73ade1f823000p+86, 0x1.0cdeb7fb80000p+85, 0x1.0671178918000p+87,
             0x1.c416fd0741000p+86, 0x1.5350d862f8000p+86, 0x1.52567e0ff5000p+86,
             0x1.d0611c1cae000p+85, 0x1.814201f9be000p+86},
            /* s=12 (s_idx=10) */
            {0x1.42dd4f0c25000p+94, 0x1.71af2232d1000p+94, 0x1.b5f1f25063000p+93,
             0x1.0e8e8784ac000p+93, 0x1.0477c23ba5000p+93, 0x1.ac3c7c8760800p+94,
             0x1.507ba57edc000p+92, 0x1.2b20ca473f000p+93, 0x1.5f2d33fd22000p+92,
             0x1.ab17cae65c800p+94, 0x1.408e48b610000p+90, 0x1.32c582e2cf000p+94},
            /* s=13 (s_idx=11) */
            {0x1.187ecea5a8800p+102, 0x1.71af223280000p+94,  0x1.5a685a078a000p+102,
             0x1.48a0e93cba000p+102, 0x1.6d422253da000p+102, 0x1.ec015f50a0000p+101,
             0x1.27d31b1920000p+99,  0x1.7b4d942fe0000p+100, 0x1.68332a1fe8000p+101,
             0x1.7859de7afc000p+99,  0x1.317d98db46800p+102, 0x1.08b9be1306800p+102,
             0x1.411e88bd34000p+100},
            /* s=14 (s_idx=12) */
            {0x1.4af9bb23b8000p+107, 0x1.e0730f7df3000p+109, 0x1.9e197740a0000p+109,
             0x1.11b44daf38000p+106, 0x1.959dba1ed5000p+109, 0x1.d3f9c70059000p+109,
             0x1.c71fc39610000p+108, 0x1.6e1a9ef495000p+109, 0x1.067fc962e0800p+110,
             0x1.81de6aed04000p+109, 0x1.086d6ad9bc800p+110, 0x1.66ccfaf43f000p+109,
             0x1.d2ae54e567000p+109, 0x1.98842ba66f000p+109},
            /* s=15 (s_idx=13) */
            {0x1.8334edf0c0800p+117, 0x1.d9618469e1000p+116, 0x1.4c97d49af8800p+117,
             0x1.3db0f47816800p+117, 0x1.ac11e30d56000p+116, 0x1.d3f9c70000000p+109,
             0x1.0210da6024000p+117, 0x1.2e86f6e52b000p+116, 0x1.f43197eee2000p+115,
             0x1.e913152bf0000p+115, 0x1.775c686f24000p+116, 0x1.44d556f611000p+116,
             0x1.90e2677038000p+115, 0x1.1b5f498bca000p+117, 0x1.9702ab51fa000p+116},
            /* s=16 (s_idx=14) */
            {0x1.568442b104000p+122, 0x1.23c286bfdb000p+125, 0x1.fffd89ae2f000p+124,
             0x1.9f80a3facf000p+124, 0x1.6b10abb2b0000p+124, 0x1.b90322c900000p+119,
             0x1.ff687bb9b9000p+124, 0x1.494950989a000p+125, 0x1.5c176f9414000p+122,
             0x1.6dca3fa2e7000p+124, 0x1.951e4290e0000p+122, 0x1.a671255128000p+123,
             0x1.b2745cf9ae000p+124, 0x1.2c6cfd90da000p+123, 0x1.a57e7d4e8e000p+124,
             0x1.8f40d0ef24000p+124},
            /* s=17 (s_idx=15) */
            {0x1.e01f9407c4000p+129, 0x1.e201959d63000p+131, 0x1.31982160c4000p+132,
             0x1.7f0fe22eef000p+132, 0x1.00d5bf9f80000p+126, 0x1.8ad801f1a0000p+129,
             0x1.2a9c662802000p+130, 0x1.d836977997000p+131, 0x1.85903a5f3c000p+132,
             0x1.a3320451ba800p+132, 0x1.ce462d2242000p+132, 0x1.d67cf11ca9800p+132,
             0x1.add7c7ba40000p+132, 0x1.57b0afae95000p+131, 0x1.e30840c0e8000p+128,
             0x1.5aabc9d4bf800p+132, 0x1.82a0ee308b800p+132},
            /* s=18 (s_idx=16) */
            {0x1.06cf388320000p+134, 0x1.a1bf2dfdc0000p+136, 0x1.bb35a9d83c000p+137,
             0x1.b0c7cfa209000p+139, 0x1.4921eae073800p+140, 0x1.172ab95fd6000p+139,
             0x1.68acfd38e8000p+139, 0x1.f34ce4f4e8000p+138, 0x1.01123dfc72000p+140,
             0x1.9db3f73893000p+139, 0x1.f6d5907a7e000p+138, 0x1.e7abc6d98b000p+139,
             0x1.8e92d65018000p+136, 0x1.1d42b11e83800p+140, 0x1.0579b3ad70800p+140,
             0x1.0cb5cec87c000p+138, 0x1.2009162ca2800p+140, 0x1.3d803cbad1800p+140},
            /* s=19 (s_idx=17) */
            {0x1.b09acf4b80000p+146, 0x1.6f0acc1cea000p+147, 0x1.d8992594f0000p+145,
             0x1.4be496434a000p+146, 0x1.8cc9189a96000p+147, 0x1.c776b470b0000p+143,
             0x1.cd534fe2dc000p+147, 0x1.82fa017336000p+147, 0x1.946f7304e8000p+147,
             0x1.551407a0b7000p+147, 0x1.034c6790f6000p+146, 0x1.452e68b9f4000p+145,
             0x1.407e5f3ab7000p+147, 0x1.a514c77360000p+147, 0x1.840b4e6816000p+147,
             0x1.7a503c2406000p+147, 0x1.9fa0adbac0000p+147, 0x1.0c070c3e0c000p+147,
             0x1.952a21ca4c000p+145},
            /* s=20 (s_idx=18) */
            {0x1.b7d0145780000p+153, 0x1.22e534dde0000p+150, 0x1.157cefeb34000p+153,
             0x1.3ca3f6e300000p+151, 0x1.016a241f28000p+152, 0x1.e66c961dd0000p+154,
             0x1.1945b982ed000p+155, 0x1.3e5ca23c80000p+152, 0x1.1ce0e51379000p+155,
             0x1.98788b5ce0000p+154, 0x1.b19e1bb310000p+154, 0x1.df2e1fa0ce000p+154,
             0x1.8b801f14e4000p+153, 0x1.38d9254eb8000p+153, 0x1.354ce8cdbc000p+154,
             0x1.94eef4587e000p+154, 0x1.3b8c91b979000p+155, 0x1.0b1e374058000p+155,
             0x1.088d7305d7000p+155, 0x1.67ddaa2cae000p+154},
        };
        static constexpr double lo[S_MAX - 1][S_MAX] = {
            /* s=2..6 (s_idx=0..4): exact integers, lo = 0 */
            {0.0}, {0.0}, {0.0}, {0.0}, {0.0},
            /* s=7  (s_idx=5) */
            {0x1.8080000000000p+9,  0x1.a000000000000p+12, 0x1.c000000000000p+10,
             0x1.8000000000000p+12, 0x1.c000000000000p+12, 0x1.d000000000000p+12,
             0x1.e000000000000p+11},
            /* s=8  (s_idx=6) */
            {0x1.16f0100000000p+20, 0x1.89a0000000000p+19, 0x1.8880000000000p+19,
             0x1.d740000000000p+19, 0x1.0b80000000000p+19, 0x1.2880000000000p+19,
             0x1.bcf0000000000p+20, 0x1.2d80000000000p+17},
            /* s=9  (s_idx=7) */
            {0x1.008cc04000000p+26, 0x1.eca4600000000p+28, 0x1.9a00780000000p+29,
             0x1.e855180000000p+29, 0x1.e9c7f00000000p+29, 0x1.38caf00000000p+29,
             0x1.d6d0600000000p+29, 0x1.e459400000000p+27, 0x1.9cd8200000000p+27},
            /* s=10 (s_idx=8) */
            {0x1.c6c29fa008000p+37, 0x1.4ddc380000000p+30, 0x1.5e72640800000p+37,
             0x1.5939d00000000p+34, 0x1.acce161000000p+36, 0x1.d3148d7000000p+37,
             0x1.1bca621000000p+37, 0x1.be65b8a000000p+35, 0x1.43b8ee6000000p+36,
             0x1.940b60e000000p+36},
            /* s=11 (s_idx=9) */
            {0x1.c311739de0100p+44, 0x1.3f5c901690000p+45, 0x1.de7087e210000p+45,
             0x1.6bfc28bd30000p+44, 0x1.de9bee2d48000p+45, 0x1.5646b56780000p+45,
             0x1.5ee3b89260000p+43, 0x1.77449328c0000p+43, 0x1.2e0367d338000p+45,
             0x1.c1e3b22c60000p+45, 0x1.4dba603168000p+45},
            /* s=12 (s_idx=10) */
            {0x1.f5cc036fee804p+50, 0x1.9502088c71500p+52, 0x1.f27fe97ac8c00p+52,
             0x1.6a7fd4fb91000p+50, 0x1.9f4e1d77bb800p+52, 0x1.541aed8de8f00p+52,
             0x1.cad9eee787600p+51, 0x1.b754bf1ae1c00p+51, 0x1.d1793fc3ce200p+51,
             0x1.f0f2278772b00p+52, 0x1.59f94c68de600p+52, 0x1.f1f52b3aa8500p+52},
            /* s=13 (s_idx=11) */
            {0x1.2a800bf67755ap+60, 0x1.459502088c715p+60, 0x1.73141ccb58410p+57,
             0x1.956a7a15d56e0p+60, 0x1.c5f9191e4aa91p+60, 0x1.c69660c475d7bp+60,
             0x1.1a2f5dd7b0278p+60, 0x1.f7e2f13271df4p+60, 0x1.008e6afbfbd20p+59,
             0x1.45eaf7cf70b15p+60, 0x1.b2d9a1321591ap+59, 0x1.2a3c04a60a8a8p+59,
             0x1.25d2c634e54f0p+57},
            /* s=14 (s_idx=12) */
            {0x1.ed366131bfd87p+61, 0x1.2a2b688425b37p+67, 0x1.ea31249d190dbp+66,
             0x1.57f8ce0e05580p+65, 0x1.34d662a4fdd1cp+66, 0x1.5013290076958p+67,
             0x1.3e7351822d438p+66, 0x1.1ebdf25941f8bp+67, 0x1.89ef93ae85687p+67,
             0x1.84d8fc60d93d4p+68, 0x1.1340b8f1c34bfp+67, 0x1.590ded2a35e12p+68,
             0x1.34cf70f07ae33p+67, 0x1.d80e799d28f38p+68},
            /* s=15 (s_idx=13) */
            {0x1.a4b62a6fdb1e1p+75, 0x1.c75bd2f612d0fp+75, 0x1.3f90fd4ad5142p+74,
             0x1.9e3c7f45d92bfp+75, 0x1.c9413fbd969ffp+75, 0x1.6550132900769p+75,
             0x1.a05bb4379a4c1p+75, 0x1.dae820f5ffc00p+74, 0x1.6405781ac87d9p+75,
             0x1.175dbeffba9cdp+75, 0x1.fe04b43a93e73p+71, 0x1.67335f8e813b8p+73,
             0x1.1bfe09769edb0p+75, 0x1.f1ee6037f1f5dp+71, 0x1.0c08e68bbfe9cp+75},
            /* s=16 (s_idx=14) */
            {0x1.195bce21a4a4cp+82, 0x1.d368142940c54p+83, 0x1.6961d82d67d29p+81,
             0x1.19861fe645aefp+78, 0x1.4aa2ee5f58c0cp+82, 0x1.42e6d8398ebf0p+83,
             0x1.09590940ec246p+83, 0x1.4ed69939f54a5p+83, 0x1.bccd986816af6p+83,
             0x1.38fff5b887f40p+83, 0x1.8c2bed86953acp+81, 0x1.2544a485cce86p+81,
             0x1.c88c3ec2fb90fp+83, 0x1.a84f9682c93f7p+83, 0x1.0beb05e6abcdfp+81,
             0x1.aeb1b1661a570p+81},
            /* s=17 (s_idx=15) */
            {0x1.1e87e3b708c22p+90, 0x1.4160efcbeef78p+90, 0x1.4480003b19f81p+89,
             0x1.0b25d1ed6a121p+87, 0x1.747bf6c0d8b31p+90, 0x1.bcbc193dd346cp+88,
             0x1.fddf745f1ee5ap+88, 0x1.d1f525311dabfp+90, 0x1.1660b883eb1a4p+90,
             0x1.41dedd270b797p+88, 0x1.79125e4f2418ap+90, 0x1.0272e6220fc37p+90,
             0x1.d2e0f92de9773p+87, 0x1.ea083b704edc0p+90, 0x1.eba48f8e2a378p+90,
             0x1.a977e531befa8p+90, 0x1.2ed72602864a3p+88},
            /* s=18 (s_idx=16) */
            {0x1.928222f7c81d9p+98, 0x1.7691a1e475ec2p+97, 0x1.50297632195fep+97,
             0x1.08e60e4fc6baep+96, 0x1.3586f9a06cbf4p+98, 0x1.7a70fdd8b6610p+98,
             0x1.5e172302320e2p+98, 0x1.1c4557a753b6cp+98, 0x1.4622df275a365p+95,
             0x1.cf7c96f698830p+95, 0x1.ebd8e9c0e37a5p+97, 0x1.f1a79e989b457p+97,
             0x1.1c8ffe978c39ep+98, 0x1.c0c39f95f19abp+92, 0x1.f55e10b41e4a2p+97,
             0x1.a546c43a54205p+98, 0x1.c22d132ce1471p+97, 0x1.6f3d636bc541bp+95},
            /* s=19 (s_idx=17) */
            {0x1.e0958b3fc5a41p+105, 0x1.586ae0321d89fp+104, 0x1.fd46b49aa2b42p+106,
             0x1.846cf4df36408p+106, 0x1.479c156f25f6cp+106, 0x1.020640df24636p+104,
             0x1.ddd5bd56c8a86p+106, 0x1.9e0161aac9805p+105, 0x1.a622eb926525ap+105,
             0x1.78923f9483982p+106, 0x1.cb97659eca409p+106, 0x1.e91af6550ad66p+105,
             0x1.58efe3be140c5p+106, 0x1.ef9fd35ae1d32p+103, 0x1.a8df1a86177b9p+106,
             0x1.a20f1e945f461p+105, 0x1.03cca141443f9p+106, 0x1.71f2fcf80933dp+106,
             0x1.7334b3dff2d8bp+105},
            /* s=20 (s_idx=18) */
            {0x1.4ca65aa6e2e69p+113, 0x1.3e4218a70dc2fp+114, 0x1.3349341ba5ba9p+114,
             0x1.a8a4cf94c4963p+113, 0x1.a9ef27a2e8284p+113, 0x1.f9453ff49eeb9p+114,
             0x1.54038a5103c5fp+114, 0x1.7e6af65bdb8e7p+114, 0x1.feec500f6cd99p+110,
             0x1.ccccb6fa6a5aep+113, 0x1.e8165d05819c5p+114, 0x1.481a67408850bp+111,
             0x1.d767562c372cdp+112, 0x1.c6ebbea0ef5f4p+113, 0x1.d062d71a7af94p+112,
             0x1.4a1e8a895454cp+111, 0x1.77cf77e873cd7p+114, 0x1.d1d5597316f21p+111,
             0x1.ed07530f9a7fap+114, 0x1.3929cf709cf74p+114},
        };
        return {hi[s_idx][mod_idx], lo[s_idx][mod_idx]};
    }

    // log2P(s_idx): ADP capacity threshold for (s_idx+2) moduli.
    // The minimum s satisfying log2P(s-2) >= log2P_needed is selected by
    // adp_expected_num_moduli().  Computed lazily on first host call; host-only.
    inline double log2P(unsigned s_idx) noexcept
    {
        static const auto v = []() {
            auto                          P = make_P_table();
            std::array<double, S_MAX - 1> a{};
            for(unsigned i = 0; i < S_MAX - 1; ++i)
                // log2(P_s) / 2 - 0.5: GEMMul8 ADP capacity formula
                //   (geometric mean of P_s, shifted by -0.5 for conservative rounding).
                // - 1.0: extra safety margin for fixed-s mode, where CRT overflow is not
                //   detected dynamically.
                //   TODO: could be removed since ADP always detects overflow via
                //   log2P_needed > log2P(S_MAX-2) and falls back to native GEMM.
                a[i] = std::log2(P[i].to_double()) / 2.0 - 0.5 - 1.0;
            return a;
        }();
        return v[s_idx];
    }

} // namespace FixedPointEmulation
