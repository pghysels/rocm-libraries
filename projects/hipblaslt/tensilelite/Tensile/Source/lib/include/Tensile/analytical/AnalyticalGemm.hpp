/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <Tensile/analytical/Hardware.hpp>
#include <vector>

namespace TensileLite
{
    namespace analytical
    {
        // Placeholder for compute_reuse_in_block_gemm function.
        // TODO move over L2 hit rate simulation for tie-breaking.
        double compute_reuse_in_block_gemm(size_t                  grid_m,
                                           size_t                  grid_n,
                                           size_t                  grid_k,
                                           size_t                  A_size,
                                           size_t                  B_size,
                                           size_t                  C_size,
                                           size_t                  nproc,
                                           size_t                  capacity,
                                           const std::vector<int>& radix,
                                           bool                    print_radix,
                                           bool                    print_output,
                                           size_t                  max_timesteps,
                                           size_t                  max_iters);

        // Compute the total latency of a gemm based on the latency of one wave multiplied by the number of waves
        // A wave is defined as : The time it takes for one CU to complete one K-complete output tile
        double compute_total_latency(const Hardware& hardware,
                                     size_t          M,
                                     size_t          N,
                                     size_t          K,
                                     size_t          batch,
                                     bool            transA,
                                     bool            transB,
                                     size_t          MT_M,
                                     size_t          MT_N,
                                     size_t          MT_K,
                                     size_t          MI_M,
                                     size_t          MI_N,
                                     size_t          MI_K,
                                     size_t          split,
                                     double          H_L2,
                                     size_t          element_size_A, //In bits
                                     size_t          element_size_B, //In bits,
                                     size_t          element_size_out, //In bits
                                     int             WGM,
                                     size_t          mx_block_size,
                                     bool            debug);

        std::pair<double, size_t> select_best_wgm(const Hardware&            hardware,
                                                  int                        M,
                                                  int                        N,
                                                  int                        K,
                                                  int                        batch,
                                                  int                        MT_M,
                                                  int                        MT_N,
                                                  int                        MT_K,
                                                  size_t                     element_size_A,
                                                  size_t                     element_size_B,
                                                  const std::vector<size_t>& WGM_list,
                                                  bool                       debug);

        // Compute the performance from the latency.
        // IMPORTANT : This program is NOT meant to be an analytical model for performance, but rather a way to rank different macro tile sizes.
        // These performance values could be wildly inaccurate in absolute terms, but will often result in the correct ranking of MTin relative terms.
        double compute_perf_gflops(const Hardware& hardware,
                                   size_t          M,
                                   size_t          N,
                                   size_t          K,
                                   size_t          batch,
                                   size_t          MT_M,
                                   size_t          MT_N,
                                   size_t          MT_K,
                                   size_t          MI_M,
                                   size_t          MI_N,
                                   size_t          MI_K,
                                   size_t          element_size_A,
                                   size_t          element_size_B,
                                   size_t          element_size_out,
                                   int             WGM,
                                   double          H_mem1,
                                   bool            debug);

        // Check if MT fits in LDS
        bool check_LDS_capacity(const Hardware& hardware,
                                size_t          MT_M,
                                size_t          MT_N,
                                size_t          MT_K,
                                size_t          element_size_A,
                                size_t          element_size_B,
                                bool            debug);

    } // namespace analytical
} // namespace TensileLite
