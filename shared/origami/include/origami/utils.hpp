// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include <set>
#include <tuple>
#include <vector>
#include <functional> // For std::function

namespace origami
{
    /* ---------------------------------------------------------------------------------------- */
    /* Misc. functions                                                                          */
    /* ---------------------------------------------------------------------------------------- */
    // Performs `(n + d - 1) / d`, but is robust against the case where `(n + d - 1)` would
    // overflow.
    template <typename N, typename D>
    constexpr N safe_ceil_div(N n, D d)
    {
        // Static cast to undo integral promotion.
        return static_cast<N>(d == 0 ? 0 : (n / d + (n % d != 0 ? 1 : 0)));
    }

        using result_tuple = std::tuple<double, // latency
                                       size_t, // MT_M
                                       size_t, // MT_N
                                       size_t, // MT_K
                                       size_t, // MI_M
                                       size_t, // MI_N
                                       size_t, // MI_K
                                       size_t,  // Occupancy
                                       int,     // WGM
                                       size_t, // non_temporal_a
                                       size_t>; // non_temporal_b

        using tile_tuple = std::tuple<size_t, // MT_M
                                     size_t, // MT_N
                                     size_t, // MT_K
                                     size_t, // MI_M
                                     size_t, // MI_N
                                     size_t, // MI_K
                                     size_t,  // Occupancy
                                     int,     // WGM
                                     size_t, // non_temporal_a
                                     size_t>; // non_temporal_b

        struct solution_info
        {
            size_t MT_M;
            size_t MT_N;
            size_t MT_K;
            size_t MI_M;
            size_t MI_N;
            size_t MI_K;
            size_t occupancy;
            size_t WGM;
            size_t non_temporal_A;
            size_t non_temporal_B;

            size_t A_loads;
            size_t B_loads;
            size_t Ld_CU_bytes;

            bool is_dot2;
            bool fits_lds_capacity;

            size_t compute_latency;
            size_t memory_latency_penalty;

            double total_latency_penalty;
            double total_latency_penalty_tf32_low_AI;
            double total_latency_penalty_tf32_high_AI;

            solution_info(size_t            macroTile_x,
                          size_t            macroTile_y,
                          size_t            depthU,
                          size_t            matrixInstruction0,
                          size_t            matrixInstruction1,
                          size_t            matrixInstruction2,
                          size_t            CUOoccupancy,
                          size_t            workGroupMapping,
                          size_t            nonTemporalA,
                          size_t            nonTemporalB,
                          const hardware_t& hardware,
                          bool              transA,
                          bool              transB,
                          size_t            element_size_A,
                          size_t            element_size_B,
                          data_type_t       mi_datatype)
            {
                MT_M = macroTile_x;
                MT_N = macroTile_y;
                MT_K = depthU;
                MI_M = matrixInstruction0;
                MI_N = matrixInstruction1;
                MI_K = matrixInstruction2;
                
                // Override dot2 instruction with vector lane widths
                if(MI_N == 0 && MI_M == 0 && MI_K == 0)
                {
                    MI_M = 1;
                    MI_N = 1;
                    MI_K = 64;
                    is_dot2 = true;
                }
                else
                {
                    is_dot2 = false;
                }

                occupancy = CUOoccupancy;
                WGM = std::max(workGroupMapping, 1);
                non_temporal_A = nonTemporalA;
                non_temporal_B = nonTemporalB;

                A_loads = MT_M * MT_K;
                B_loads = MT_N * MT_K;
                Ld_CU_bytes = (A_loads * safe_ceil_div(element_size_A, 8))
                            + (B_loads * safe_ceil_div(element_size_B, 8));

                fits_lds_capacity = Ld_CU_bytes <= hardware.lds_capacity;

                compute_mt_compute_latency(hardware,
                                           transA,
                                           transB,
                                           element_size_A,
                                           element_size_B,
                                           mi_datatype);

                compute_memory_latency_penalty(hardware,
                                               transA,
                                               transB,
                                               element_size_A,
                                               element_size_B,
                                               mi_datatype);

                compute_total_latency_penalty(hardware,
                                              transA,
                                              transB,
                                              element_size_A,
                                              element_size_B,
                                              mi_datatype);
            }

        private:

            void compute_mt_compute_latency(const hardware_t& hardware,
                                            bool              transA,
                                            bool              transB,
                                            size_t            element_size_A,
                                            size_t            element_size_B,
                                            data_type_t       mi_datatype)
            {
                // Total number of matrix instructions for MT_MxMT_NxMT_K tile
                size_t N_MI = safe_ceil_div(MT_M, MI_M) * safe_ceil_div(MT_N, MI_N) * safe_ceil_div(MT_K, MI_K);

                compute_latency = N_MI * hardware.get_mi_latency(MI_M, MI_N, MI_K, mi_datatype);

                // TN
                if(transA && !transB)
                {
                    //We want to penalize tiles that can't be coalesced for T,N where K is contiguous dimension.
                    //In this case, that's when the K dimension is indivisible by 128 bytes.
                    if(MT_K * safe_ceil_div(element_size_A, 8) % 128 != 0)
                        compute_latency *= 1.5;
                    if(MT_K * safe_ceil_div(element_size_B, 8) % 128 != 0)
                        compute_latency *= 1.5;
                }

                // NT: A is contiguous in M and B is contiguous in N
                if(!transA && transB)
                {
                    //LDS Load Granularity is 128 Bytes -> If we load an amount indivisible by 128 bytes in either contiguous
                    //dimesion from LDS then we will get poor LDS utilization. This actually happens as more like
                    //a quantization effect where if either contiguous dimension of the tile is not evenly divisible by 128-bytes
                    //We end up with inefficient loads.
                    //Multiplication by a value is arbitrary, there is probably a better analytical method to quantify the true impact of this
                    //Effect on the efficiency of computation.
                    if((MT_M * safe_ceil_div(element_size_A, 8)) % 128 != 0)
                        compute_latency *= 2;
                    if((MT_N * safe_ceil_div(element_size_B, 8)) % 128 != 0)
                        compute_latency *= 2;
                    //NT Transpose Overhead Scales in both.
                }

                // TT: A is contiguous in K and B is contiguous in N
                if(transA && transB)
                {
                    if(MT_K * safe_ceil_div(element_size_A, 8) < 128)
                        compute_latency *= 2;
                    if(MT_N * safe_ceil_div(element_size_B, 8) < 128)
                        compute_latency *= 2;
                }

                // NN: A is contiguous in M and B is contiguous in K
                if(!transA && !transB)
                {
                    if(MT_M * safe_ceil_div(element_size_A, 8) < 128)
                        compute_latency *= 2;
                    if(MT_K * safe_ceil_div(element_size_B, 8) < 128)
                        compute_latency *= 2;
                }
            }

            void compute_memory_latency_penalty(const hardware_t& hardware,
                                                bool              transA,
                                                bool              transB,
                                                size_t            element_size_A,
                                                size_t            element_size_B,
                                                data_type_t       mi_datatype)
            {
                memory_latency_penalty = 1;

                // NT
                if(!transA && transB)
                {
                    //LDS Load Granularity is 128 Bytes -> If we load an amount indivisible by 128 bytes in either contiguous
                    //dimesion from LDS then we will get poor LDS utilization. This actually happens as more like
                    //a quantization effect where if either contiguous dimension of the tile is not evenly divisible by 128-bytes
                    //We end up with inefficient loads.
                    //Multiplication by a value is arbitrary, there is probably a better analytical method to quantify the true impact of this
                    //Effect on the efficiency of computation.
                    if((MT_M * safe_ceil_div(element_size_A, 8)) % 128 != 0)
                        memory_latency_penalty *= 2;
                    if((MT_N * safe_ceil_div(element_size_B, 8)) % 128 != 0)
                        memory_latency_penalty *= 2;
                }

                // TT : A is contiguous in K and B is contiguous in N
                if(transA && transB)
                {
                    if(MT_K * safe_ceil_div(element_size_A, 8) < 128)
                        memory_latency_penalty *= 2;
                    if(MT_N * safe_ceil_div(element_size_B, 8) < 128)
                        memory_latency_penalty *= 2;
                }

                // NN : A is contiguous in M and B is contiguous in K
                if(!transA && !transB)
                {
                    if(MT_M * safe_ceil_div(element_size_A, 8) < 128)
                        memory_latency_penalty *= 2;
                    if(MT_K * safe_ceil_div(element_size_B, 8) < 128)
                        memory_latency_penalty *= 2;
                }
            }

            void compute_total_latency_penalty(const hardware_t& hardware,
                                               bool              transA,
                                               bool              transB,
                                               size_t            element_size_A,
                                               size_t            element_size_B,
                                               data_type_t       mi_datatype)
            {
                total_latency_penalty = 1.;
                total_latency_penalty_tf32_low_AI = 1.;
                total_latency_penalty_tf32_high_AI = 1.;

                bool tf32_emu = mi_datatype == data_type_t::XFloat32 &&
                            hardware.arch == hardware_t::architecture_t::gfx950;

                // Heuristics for TF32
                if(tf32_emu)
                {
                    // The kernel for this is more optimized (Custom kernel NT)
                    if((!transA && transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
                    {
                        total_latency_penalty_tf32_low_AI *= 0.6;
                        total_latency_penalty_tf32_high_AI *= 0.4;
                    }

                    // The kernel for this is more optimized (Custom kernel NN)
                    if((!transA && !transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
                    {
                        total_latency_penalty_tf32_low_AI *= 0.8;
                        total_latency_penalty_tf32_high_AI *= 0.6;
                    }

                    // The kernel for this is more optimized (Custom kernel TN)
                    if((transA && !transB) && MT_M == 256 && MT_N == 256 && MT_K == 32)
                    {
                        total_latency_penalty_tf32_low_AI *= 0.8;
                        total_latency_penalty_tf32_high_AI *= 0.4;
                    }
                }
                else  // !TF32
                {
                    // Bias Model towards at least one dim being power of 2
                    bool MT_M_is_power_two = (MT_M > 0) && (MT_M & (MT_M - 1)) == 0;
                    bool MT_N_is_power_two = (MT_N > 0) && (MT_N & (MT_N - 1)) == 0;
                    if(!MT_M_is_power_two && !MT_N_is_power_two)
                        total_latency_penalty *= 1.1;

                    // Bias Model towards both dims being a power of 2
                    if(MT_M_is_power_two && MT_N_is_power_two)
                        total_latency_penalty *= 0.9;

                    // Heuristics for FP16
                    if(element_size_A == 16)
                    {
                        // These kernels are more optimized (Custom kernels)
                        // All layouts
                        if(MT_M == 256 && MT_N == 256 && MT_K == 64)
                            total_latency_penalty *= 0.85;

                        // The kernel for this is less optimized, for some reason
                        if(MT_M == 256 && MT_N == 16 && MT_K == 128)
                            total_latency_penalty *= 2;

                        // The kernel for this is less optimized, for some reason
                        if(MT_M == 16 && MT_N == 256 && MT_K == 128)
                            total_latency_penalty *= 2;
                    }

                    // Heuristics for FP8
                    if(element_size_A == 8)
                    {
                        // The kernel for this is more optimized (Custom kernel)
                        if(transA && !transB && MT_M == 256 && MT_N == 256 && MT_K == 128)
                            total_latency_penalty *= 0.8;

                        // Bias towards dimensions divisible by 64 for 8-bit datatypes
                        if((MT_M > 64) && (MT_M % 64 != 0))
                            total_latency_penalty *= 1.2;
                        if((MT_N > 64) && (MT_N % 64 != 0))
                            total_latency_penalty *= 1.2;
                    }
                }
            }

        };

        struct analytical_model
        {
            bool transA;
            bool transB;
            const hardware_t& hardware;

            size_t element_size_A;
            size_t element_size_B;
            size_t element_size_out;
            data_type_t mi_datatype;
            size_t mx_block_size;

            bool tf32_emu;

            // TODO store this in PredictionLibrary, then pass to select_best_macro_tile_size
            std::vector<solution_info> tile_list;

            analytical_model(bool                           transA_,
                             bool                           transB_,
                             const hardware_t&              hardware_,
                             size_t                         element_size_A_, //In bits
                             size_t                         element_size_B_, //In bits
                             size_t                         element_size_out_, //In bits
                             data_type_t                    mi_datatype_,
                             size_t                         mx_block_size_,
                             const std::vector<tile_tuple>& MT_list)
                : transA(transA_), transB(transB_), hardware(hardware_),
                  element_size_A(element_size_A_), element_size_B(element_size_B_),
                  element_size_out(element_size_out_), mi_datatype(mi_datatype_),
                  mx_block_size(mx_block_size_)
            {
                for(auto& t : MT_list)
                {
                    solution_info ti(t, hardware, transA, transB, element_size_A, element_size_B, mi_datatype);
                    if(ti.fits_lds_capacity) 
                        tile_list.push_back(std::move(ti));
                }

                tf32_emu = mi_datatype == data_type_t::XFloat32 &&
                           hardware.arch == hardware_t::architecture_t::gfx950;
            }

            std::vector<result_tuple> select_best_macro_tile_size(size_t M,
                                                                  size_t N,
                                                                  size_t K,
                                                                  size_t batch,
                                                                  double H_L2,
                                                                  bool   print,
                                                                  size_t defaultWGM) const
            {
                std::vector<result_tuple> valid_results;
                valid_results.reserve(tile_list.size());

                for(const auto& mt : tile_list)
                {
                    double L = compute_total_latency(M, N, K, batch, mt, 0);
                    valid_results.emplace_back(
                        L,
                        mt.MT_M, mt.MT_N, mt.MT_K,
                        mt.MI_M, mt.MI_N, mt.MI_K,
                        mt.occupancy,
                        mt.WGM,
                        mt.non_temporal_A, mt.non_temporal_B);
                }

                if(valid_results.empty())
                    throw std::runtime_error("No valid macro-tile sizes found.");

                // 1) Sort results by ascending latency.
                std::stable_sort(valid_results.begin(), valid_results.end(), [](auto const& a, auto const& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });

                // TODO tie-breaking logic

                return valid_results;
            }

        private:

            double compute_total_latency(size_t M,
                                         size_t N,
                                         size_t K,
                                         size_t batch,
                                         const solution_info& mt,
                                         size_t split)
            {
                // 0) Short-circuit
                // We don't need to compute latency for all MTs. With this, we can shortcut.
                bool shortCircuit = true;
                if(shortCircuit)
                {
                    // When problem dimensions are small enough that we can fit them in one tile, we should do so.
                    // This short circuit condition also decreases selection latency when problems are very small :)
                    // TODO 256 and 256 here should be largest M and N tile dimensions in library
                    if(M <= 256 && N <= 256 && K < 1024 && batch != 1 && (mt.MT_M < M || mt.MT_N < N))
                        return std::numeric_limits<double>::max();

                    // We only use Dot2 for NN layout where M < 3
                    if(mt.is_dot2 && (M > 2 || transA || transB))
                        return std::numeric_limits<double>::max();
                }

                // 1-2) Find CU occupancy
                auto [numActiveCUs, numWaves, splittingFactor] =
                    compute_CU_occupancy(M, N, K, batch, mt,
                                         std::numeric_limits<size_t>::max(), // workspace
                                         std::numeric_limits<size_t>::max(), // workspace per c
                                         0, // occupancy
                                         6, // dynamic_grid
                                         split);

                // 2) Compute latency of a wave
                // Compute latency of a wave
                double L_wave = compute_tile_wave_latency(M, N, K, batch, mt, numActiveCUs, splittingFactor);
                // Compute latency for all waves and return it as the latency for the MT/problem
                double total_latency = L_wave * numWaves;

                // 3) Customized heuristics
                // TODO These are quantifying effects that don't work in the current math.
                // TODO THESE SHOULD BE TEMPORARY FIXES AND BE MORE SOLIDLY INTEGRATED LATER
                if(hardware_t::is_heuristics_enabled())
                {

                    // Heuristics for TF32
                    if(tf32_emu)
                    {
                        double bytes_per_element = static_cast<double>(element_size_A) / 8.0;
                        double arith = emulated_tf32_arithmetic_intensity(M, N, K, bytes_per_element);
                        double compute_threshold = 1000; // threshold empirically determined.

                        total_latency *= arith < compute_threshold ?
                            mt.total_latency_penalty_tf32_low_AI : mt.total_latency_penalty_tf32_high_AI;

                        // Bias large DU where K-dimension is large and M and N are small.
                        if((K >= (M * 16) && K >= (N * 16)) && (MT_K >= 128))
                            total_latency *= 0.5;
                    }
                    else
                    {
                        // Penalize tiles that lead to edge waste
                        const size_t numMT_M = safe_ceil_div(M, MT_M);
                        const size_t numMT_N = safe_ceil_div(N, MT_N);
                        const double waste = static_cast<double>(numMT_M * MT_M * numMT_N * MT_N) / (M * N);
                        double edge_penalty = std::pow(waste, 0.8);
                        if(batch > 10)
                            edge_penalty = std::pow(waste, 1.5) * std::log10((double)batch) * std::pow((double)numMT_M * numMT_N, 0.2);
                        total_latency *= edge_penalty;

                        // Penalize K iterations
                        size_t K_iters = safe_ceil_div(K, MT_K);
                        if(K_iters <= 2)
                            total_latency *= 8;
                        else if(K_iters <= 4)
                            total_latency *= 4;
                        else if(K_iters <= 8)
                            total_latency *= 2.1;

                        // Bias toward not splitting for small K values
                        // This should actually come from SK grid prediction
                        if(splittingFactor > 1 && K < 2048)
                            total_latency *= splittingFactor;

                        // There is no case where a kernel with MT_K > K wins unless K < MI_K.
                        // Unless it is Dot2.
                        if(K < MT_K && MI_M != 1)
                            total_latency *= (MT_K - K);


                        total_latency *= mt.total_latency_penalty;


                        // Bias toward 512 tiles for sizes "very skinny" sizes
                        // "very skinny" definition: either N or M less than 16 (1 tile) and the other one requires
                        // more than 100 waves (100*numCUs tiles)
                        if(M < 16 && N > 100 * hardware.N_CU * 512 && MT_N == 512)
                            total_latency *= 0.25;
                        if(N < 16 && M > 100 * hardware.N_CU * 512 && MT_M == 512)
                            total_latency *= 0.25;

                        // DOT2 Kernels
                        if(mt.is_dot2)
                        {
                            // Bias DOT2 kernels in which the tile dimensions in M and K are equal to the problem dimensions
                            if(MT_M == M || MT_K == K)
                                total_latency *= 0.8;
                        }

                        // Heuristics for FP16
                        if(element_size_A == 16)
                        {
                            // These kernels are more optimized (Custom kernels)
                            // All layouts
                            if(MT_M == 256 && MT_N == 256 && MT_K == 64)
                            {
                                if((transA && !transB) && (M == MT_M && N > 256 * MT_N && K >= 4*MT_K))
                                    total_latency *= total_latency;
                            }
                        }

                    }
                }

                if(hardware_t::is_debug_enabled())
                {
                    hardware.log_debug("Total_latency (with heuristics)", total_latency);
                    hardware.print_debug_info();
                }

                return total_latency;
            }

            double compute_tile_wave_latency(size_t           M,
                                             size_t           N,
                                             size_t           K,
                                             size_t           batch,
                                             const solution_info& mt,
                                             size_t           numActiveCUs,
                                             size_t           splittingFactor)
            {
                double L_compute = mt.compute_latency;

                double L_mem = compute_memory_latency(M, N, K, batch, mt,
                                                      numActiveCUs,
                                                      splittingFactor);

                // 2) Work-group setup & iteration latencies
                double L_WG_setup = 1; // WG_setup_Latency

                // 3) Prologue: 2.2× memory latency
                double L_prologue = 1.5 * L_mem; // 1.5 chosen emprically

                // 4) Epilogue: writes from all active CUs with limited bandwidth
                double epilogue_limite = (static_cast<double>(numActiveCUs) / hardware.N_CU);
                double limited_mem1 = hardware.mem1_perf_ratio * epilogue_limite;
                if(limited_mem1 < 1)
                {
                    limited_mem1 = 10;
                }
                double out_bytes = static_cast<double>(numActiveCUs)
                                        * mt.MT_M * mt.MT_N * safe_ceil_div(element_size_out, 8);
                double L_epilogue = out_bytes / limited_mem1;

                // 4') K-split reductions are globally coherent, we need to write and read split-1 MT_M*MT_N tiles to coherent memory
                if(splittingFactor > 1)
                {
                    size_t n_partials              = splittingFactor - 1;
                    double partial_readwrite_bytes = (2 * out_bytes * n_partials);
                    double L_reduce = partial_readwrite_bytes / (hardware.mem3_perf_ratio);
                    L_epilogue += L_reduce * 1;
                }

                // 4'') tf32 emu has some more overhead
                double L_cvt = 0;
                if(tf32_emu)
                {
                    // TODO

                    // L_cvt = compute_cvt_overhead(hardware,
                    //                              MT_M,
                    //                              MT_N,
                    //                              MT_K,
                    //                              MI_M,
                    //                              MI_N,
                    //                              MI_K,
                    //                              element_size_A,
                    //                              element_size_B);
                }

                // 5) Single-tile latency (always additive)
                double L_tile_single = std::max(L_compute, L_mem) + L_cvt;

                // 6) Number of K-iterations (excluding epilogue), at least 1
                // long num_iter = static_cast<long>(((K + MT_K - 1) / MT_K)) - 1;
                // num_iter      = std::ceil(num_iter / splittingFactor);
                // num_iter      = std::max(num_iter, 1L);
                long splittedK = static_cast<long>(safe_ceil_div(K, splittingFactor));
                long num_iter = static_cast<long>(safe_ceil_div(splittedK, mt.MT_K)) - 1;

                // 7) Total tile latency
                double L_tile_total = (L_tile_single * num_iter)
                                        + L_prologue
                                        + L_epilogue
                                        + L_WG_setup
                                        + (28 * num_iter); // 7 instructions (each with 4 cycles) at the end of the loop

                if(hardware_t::is_debug_enabled())
                {
                    hardware.log_debug("L_compute", L_compute);
                    hardware.log_debug("L_mem", L_mem);
                    hardware.log_debug("L_cvt", L_cvt);
                    hardware.log_debug("L_tile_single", L_tile_single);
                    hardware.log_debug("num_iter", num_iter);
                    hardware.log_debug("L_prologue", L_prologue);
                    hardware.log_debug("L_epilogue", L_epilogue);
                    hardware.log_debug("L_tile_total", L_tile_total);
                }

                return L_tile_total;

            }

            double compute_memory_latency(size_t           M,
                                          size_t           N,
                                          size_t           K,
                                          size_t           batch,
                                          const solution_info& mt,
                                          size_t           numActiveCUs,
                                          size_t           splittingFactor)
            {
                // Compute grid dimensions
                int grid_m = static_cast<int>(safe_ceil_div(M, MT_M));
                int grid_n = static_cast<int>(safe_ceil_div(N, MT_N));

                // 1) Estimate L2 hit-rate
                double H_mem1
                    = estimate_l2_hit(M, N, K, batch, mt, grid_m, grid_n, splittingFactor);

                // 2) Estimate mall hit-rate
                double H_mem2
                    = estimate_mall_hit(M, N, K, batch, mt, grid_m, grid_n, numActiveCUs, splittingFactor);

                // 3) Total loads are loads from A and loads from B
                auto Ld_CU_bytes = mt.Ld_CU_bytes;


                // Logic for block scaled datatypes (Assuming BS=32 and 8-bit scales)
                // TODO This is technically wrong, need separate flag to enable MX so we can differentiate FP8 and MX8
                if(mx_block_size != 0)
                {
                    if(element_size_A < 8)
                        Ld_CU_bytes += safe_ceil_div(mt.A_loads, mx_block_size); //One Byte per scale
                    if(element_size_B < 8)
                        Ld_CU_bytes += safe_ceil_div(mt.B_loads, mx_block_size); //One Byte per scale
                }

                // 4) total loads by all CUs
                double total_Ld = Ld_CU_bytes * static_cast<double>(numActiveCUs);

                // 5) mem1‐limited factor (simple linear model)
                double mem1_bw_limited = static_cast<double>(numActiveCUs) / static_cast<double>(hardware.N_CU);
                double limited_mem1_bw = hardware.mem1_perf_ratio * mem1_bw_limited;

                // 6) mem1 latency
                double L_mem_mem1 = (limited_mem1_bw > 0) ? (total_Ld / (limited_mem1_bw)) : 0.0;

                // 7) mem2‐limited from occupancy (Can't Issue enough load/stores)
                double bw_limited = compute_mem_bw_from_occupancy(hardware, numActiveCUs);

                // 8) loads that reach each level
                double Ld_mem2 = (1.0 - H_mem1) * total_Ld;
                double Ld_MEM  = (1.0 - H_mem2) * Ld_mem2;

                // 9) enforce whole‐problem minimum loads
                if(numActiveCUs < hardware.N_CU)
                {
                    double min_load
                        = static_cast<double>(mt.MT_K * splittingFactor *
                            (M * safe_ceil_div(element_size_A, 8) + N * safe_ceil_div(element_size_B, 8)));
                    Ld_MEM  = std::max(Ld_MEM, min_load) * batch;
                    Ld_mem2 = std::max(Ld_mem2, min_load) * batch;
                }

                // 10) mem2 latency
                double limited_mem2_bw = hardware.mem2_perf_ratio * bw_limited;
                double L_mem_mem2 = (limited_mem2_bw > 0) ? (Ld_mem2 / limited_mem2_bw) : 0.0;

                // 11) MEM latency
                double limited_mem_bw = hardware.mem3_perf_ratio * bw_limited;
                double L_mem_MEM      = (limited_mem_bw > 0) ? (Ld_MEM / limited_mem_bw) : 0.0;
                L_mem_MEM += 200; // Load Latency

                // 12) pick the worst‐case bound
                double L_mem = std::max({L_mem_mem1, L_mem_mem2, L_mem_MEM});

                L_mem *= mt.memory_latency_penalty;

                if(hardware_t::is_debug_enabled())
                {
                    hardware.log_debug("mem1_perf_ratio", hardware.mem1_perf_ratio);
                    hardware.log_debug("mem2_perf_ratio", hardware.mem2_perf_ratio);
                    hardware.log_debug("mem3_perf_ratio", hardware.mem3_perf_ratio);
                    hardware.log_debug("mem_bw_per_wg_coefficients(0)", std::get<0>(hardware.mem_bw_per_wg_coefficients));
                    hardware.log_debug("mem_bw_per_wg_coefficients(1)", std::get<1>(hardware.mem_bw_per_wg_coefficients));
                    hardware.log_debug("mem_bw_per_wg_coefficients(2)", std::get<2>(hardware.mem_bw_per_wg_coefficients));
                    hardware.log_debug("H_mem1 (mem1 hit ratio)", H_mem1);
                    hardware.log_debug("H_mem2 (mem2 hit ratio)", H_mem2);
                    hardware.log_debug("Total Load (bytes)", total_Ld);
                    hardware.log_debug("Ld_mem2 (bytes)", Ld_mem2);
                    hardware.log_debug("Ld_MEM (bytes)", Ld_MEM);
                    hardware.log_debug("L_mem_mem1 (cycles)", L_mem_mem1);
                    hardware.log_debug("L_mem_mem2 (cycles)", L_mem_mem2);
                    hardware.log_debug("L_mem_MEM (cycles)", L_mem_MEM);
                }

                return L_mem;
            }

            double estimate_l2_hit(size_t           M,
                                   size_t           N,
                                   size_t           K,
                                   size_t           batch,
                                   const solution_info& mt,
                                   int              grid_m,
                                   int              grid_n,
                                   size_t           numActiveCUs,
                                   size_t           splittingFactor)
            {
                // Distribute CUs per XCD
                // Modify cu_per_xcd to only take into account the CUs that might share same K-tiles
                // This is to factor in the effect of splitting on L2
                int cu_per_xcd = safe_ceil_div(grid_m * grid_n, hardware.NUM_XCD);
                cu_per_xcd /= splittingFactor;

                // N dimension of mem1 tile is divided by whichever is smaller between WGM and grid
                int l2_n = std::min(WGM, grid_n);
                int l2_m = cu_per_xcd / l2_n;

                // If a single mem1 tile is larger than the grid, extend M dimension
                if(l2_m > grid_m)
                {
                    int num_wraps   = (l2_m / grid_m) - 1; // how many times we wrap
                    l2_n += (num_wraps * WGM);
                    l2_m = grid_m;
                }

                // Clamp mem1 tile dimensions to at least 1 and at most grid size
                l2_m = std::max(std::min(grid_m, l2_m), 1);
                l2_n = std::max(std::min(grid_n, l2_n), 1);

                // Compute "uncached" reads based on mem1 tile dimensions
                long long l2_A_uncached_reads = static_cast<long long>(l2_m) * mt.A_loads;
                long long l2_B_uncached_reads = static_cast<long long>(l2_n) * mt.B_loads;
                long long uncached_read       = l2_A_uncached_reads + l2_B_uncached_reads;

                // If bigger than cache capacity, reduce mem1 tile size and recompute uncached reads
                while(uncached_read > hardware.L2_capacity / safe_ceil_div(element_size, 8))
                {
                    // Reduce M dimension by 1
                    l2_m -= 1;
                    if(l2_m < 1)
                    {
                        // We cannot shrink any more without going to zero or negative
                        l2_m = 1;
                        break;
                    }
                    l2_A_uncached_reads = static_cast<long long>(l2_m) * mt.A_loads;
                    uncached_read = l2_A_uncached_reads + l2_B_uncached_reads
                }

                // Total reads considering repeated usage
                long long l2_A_reads = l2_n * l2_A_uncached_reads;
                long long l2_B_reads = l2_m * l2_B_uncached_reads;

                long long total_reads         = std::max(l2_A_reads + l2_B_reads, 1LL);
                long long total_uncached_read = l2_A_uncached_reads + l2_B_uncached_reads;
                long long cached_reads        = total_reads - total_uncached_read;

                double l2_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

                // Guard against numeric anomalies
                if(l2_hit > 1.0)
                {
                    std::cerr << "mem1 hit was greater than 1, which isn't possible.\n"
                            << "Problem Size: " << M << "x" << N << "x" << K << "\n"
                            << "Macro-Tile:  " << MT_M << "x" << MT_N << "x" << MT_K << "\n"
                            << "cu_per_xcd:  " << cu_per_xcd << "\n"
                            << "l2_m: " << l2_m << ", l2_n: " << l2_n << ", l2_hit: " << l2_hit
                            << "\n";
                }

                if(hardware_t::is_debug_enabled())
                {
                    hardware.log_debug("L2Tile_M", l2_m);
                    hardware.log_debug("L2Tile_N", l2_n);
                }

                return l2_hit;
            }

            double estimate_mall_hit(size_t           M,
                                     size_t           N,
                                     size_t           K,
                                     size_t           batch,
                                     const solution_info& mt,
                                     int              grid_m,
                                     int              grid_n,
                                     size_t           numActiveCUs,
                                     size_t           splittingFactor)
            {
                // mem2 tile dimensions
                int mall_m = grid_m * grid_n / WGM;         // M dimension of mem2 tile
                int mall_n = std::min(WGM, grid_n); // N dimension of mem2 tile

                // If a single mem2 tile is larger than the grid, extend its M dimension
                if(mall_m > grid_m)
                {
                    int num_wraps   = (mall_m / grid_m) - 1;
                    mall_n += (num_wraps * WGM);
                    mall_m = grid_m;
                }

                // Clamp the tile dimensions to valid ranges
                mall_m = std::max(std::min(grid_m, mall_m), 1);
                mall_n = std::max(std::min(grid_n, mall_n), 1);

                // Unique “uncached” entries of A/B for this XCD
                int mall_A_uncached_reads = mall_m * mt.A_loads;
                int mall_B_uncached_reads = mall_n * mt.B_loads;
                int total_uncached_read   = mall_A_uncached_reads + mall_B_uncached_reads;

                // Total A/B reads considering repeated usage
                long long mall_A_reads = mall_n * static_cast<long long>(mall_A_uncached_reads);
                long long mall_B_reads = mall_m * static_cast<long long>(mall_B_uncached_reads);

                // Avoid division by zero
                long long total_reads  = std::max(mall_A_reads + mall_B_reads, 1LL);
                long long cached_reads = total_reads - total_uncached_read;

                double mall_hit = static_cast<double>(cached_reads) / static_cast<double>(total_reads);

                if(hardware_t::is_debug_enabled())
                {
                    hardware.log_debug("MallTile_M", mall_m);
                    hardware.log_debug("MallTile_N", mall_n);
                }

                return mall_hit;
            }

        };




        size_t select_best_grid_size(size_t          M,
                                     size_t          N,
                                     size_t          K,
                                     size_t          batch,
                                     bool            transA,
                                     bool            transB,
                                     const hardware_t& hardware,
                                     size_t          MT_M,
                                     size_t          MT_N,
                                     size_t          MT_K,
                                     size_t          MI_M,
                                     size_t          MI_N,
                                     size_t          MI_K,
                                     size_t          element_size_A,
                                     size_t          element_size_B,
                                     size_t          element_size_out,
                                     data_type_t     mi_datatype,
                                     size_t          mx_block_size,
                                     double          H_L2,
                                     size_t          WGM,
                                     size_t          biggest_allowable_split = 8);

        std::vector<result_tuple> select_best_macro_tile_size(size_t                        M,
                                                             size_t                        N,
                                                             size_t                        K,
                                                             size_t                        batch,
                                                             bool                          transA,
                                                             bool                          transB,
                                                             const hardware_t&             hardware,
                                                             const std::vector<tile_tuple>& MT_list,
                                                             size_t element_size_A,
                                                             size_t element_size_B,
                                                             size_t element_size_out,
                                                             data_type_t mi_datatype,
                                                             size_t mx_block_size,
                                                             double H_L2,
                                                             bool   print,
                                                             size_t WGM);

        std::vector<result_tuple> sweep_macro_tile_sizes(size_t    M,
                                                        size_t    N,
                                                        size_t    K,
                                                        bool      transA,
                                                        bool      transB,
                                                        hardware_t& hardware,
                                                        size_t    element_size = 2,
                                                        size_t    max_MT_M     = 256,
                                                        size_t    max_MT_N     = 256,
                                                        size_t    max_MT_K     = 128,
                                                        size_t    step_MT_M    = 32,
                                                        size_t    step_MT_N    = 32,
                                                        size_t    step_MT_K    = 32,
                                                        double    H_L2         = 0.8,
                                                        const std::vector<tile_tuple>& tiles_to_add
                                                        = {},
                                                        bool print = false);

        std::pair<double, size_t> select_best_wgm(
            size_t                     M,
            size_t                     N,
            size_t                     K,
            size_t                     batch,
            const hardware_t&          hardware,
            size_t                     MT_M,
            size_t                     MT_N,
            size_t                     MT_K,
            size_t                     MI_M,
            size_t                     MI_N,
            size_t                     MI_K,
            const std::vector<size_t>& WGM_list,
            size_t                     element_size,
            double H_L2, // not needed for L2 hit rate but retained if your code expects it
            bool   print);

        double compute_tflops_from_latency(double latency_cycles,
                                           size_t M,
                                           size_t N,
                                           size_t K,
                                           double clock_GHz);

} // namespace origami
