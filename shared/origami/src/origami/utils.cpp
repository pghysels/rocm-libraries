// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/utils.hpp"

#include <algorithm>
#include <chrono> // For timing
#include <cmath>
#include <iomanip> // For output formatting
#include <iostream>

namespace origami
{
    //
    // Tiebreaker function.
    //
    void pick_best_tile_by_arithmetic_intensity(std::vector<result_tuple>& top_results,
                                                size_t                     num_to_sort)
    {
        if(top_results.empty())
        {
            throw std::runtime_error("pick_best_tile_by_arithmetic_intensity received empty list.");
        }

        // 1) Define a helper function to compute the arithmetic intensity of a tile.
        //    Here we assume:
        //    - Flops for tile (MT_M, MT_N, MT_K) is: 2 * MT_M * MT_N * MT_K
        //    - Memory traffic approximated as: MT_M*MT_K + MT_K*MT_N + MT_M*MT_N
        //    - Arithmetic intensity = flops / memory_traffic
        auto compute_arithmetic_intensity = [](const result_tuple& t) -> double {
            // The tuple is: (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
            auto MT_M = std::get<1>(t);
            auto MT_N = std::get<2>(t);
            auto MT_K = std::get<3>(t);

            double flops          = static_cast<double>(2ull * MT_M * MT_N * MT_K);
            double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

            // Avoid division by zero.
            if(memory_traffic == 0.0)
                return 0.0;

            return flops / memory_traffic;
        };
        // 2) Sort the results in descending order of arithmetic intensity
        //    (highest arithmetic intensity first).
        std::stable_sort(top_results.begin(),
                         top_results.begin() + num_to_sort,
                         [&](const result_tuple& a, const result_tuple& b) {
                             double ai_a = compute_arithmetic_intensity(a);
                             double ai_b = compute_arithmetic_intensity(b);
                             return ai_a > ai_b; // descending
                         });
        // 3) Return the tile with the highest arithmetic intensity
    }

    result_tuple pick_best_tile_with_dimension_priority(
        const std::vector<result_tuple>& top_results, size_t M, size_t N, size_t K)
    {
        if(top_results.empty())
        {
            throw std::runtime_error("pick_best_tile_with_dimension_priority received empty list.");
        }

        // 1) Determine whether M or N is more important
        //    (based on which is larger), and always place K last.
        //    This yields a priority order of either { 'M', 'N', 'K' }
        //    or { 'N', 'M', 'K' }.
        std::vector<char> dimPriority;
        if(M >= N)
            dimPriority = {'M', 'N', 'K'};
        else
            dimPriority = {'N', 'M', 'K'};

        // 2) Helper function to extract the tile dimension:
        //    (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
        auto getTileSize = [](const result_tuple& t, char dimChar) -> size_t {
            switch(dimChar)
            {
            case 'M':
                return std::get<1>(t); // MT_M
            case 'N':
                return std::get<2>(t); // MT_N
            case 'K':
                return std::get<3>(t); // MT_K
            default:
                return 0;
            }
        };

        // 3) Sort in descending order according to the dimension priority.
        //    - Compare dimensionPriority[0] first
        //    - If there's a tie, compare dimensionPriority[1]
        //    - If still a tie, compare dimensionPriority[2]
        //    - If they're all equal, consider them tied
        std::vector<result_tuple> sorted = top_results; // copy
        std::stable_sort(
            sorted.begin(), sorted.end(), [&](const result_tuple& a, const result_tuple& b) {
                for(char d : dimPriority)
                {
                    size_t ta = getTileSize(a, d);
                    size_t tb = getTileSize(b, d);
                    if(ta > tb)
                        return true;
                    if(ta < tb)
                        return false;
                }
                // If all relevant dimensions are the same, treat as a tie
                return false;
            });

        // 4) Return the best tile (the first after sorting).
        return sorted.front();
    }

    size_t select_best_grid_size(size_t            M,
                                 size_t            N,
                                 size_t            K,
                                 size_t            batch,
                                 bool              transA,
                                 bool              transB,
                                 const hardware_t& hardware,
                                 size_t            MT_M,
                                 size_t            MT_N,
                                 size_t            MT_K,
                                 size_t            MI_M,
                                 size_t            MI_N,
                                 size_t            MI_K,
                                 size_t            element_size_A,
                                 size_t            element_size_B,
                                 size_t            element_size_out,
                                 data_type_t       mi_datatype,
                                 size_t            mx_block_size,
                                 double            H_L2,
                                 size_t            WGM,
                                 size_t            biggest_allowable_split)
    {
        // compute how many 32×32 tiles are needed in each dim,
        // then multiply to get total grid size:
        size_t grid = ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N) * batch;

        size_t max_hw_split = std::floor(hardware.N_CU / grid);
        size_t MAX_SPLIT    = std::min(biggest_allowable_split, max_hw_split);

        size_t best_split   = 1;
        double best_latency = std::numeric_limits<double>::infinity();

        for(size_t split = 1; split <= MAX_SPLIT; ++split)
        {
            double latency = compute_total_latency(hardware,
                                                   M,
                                                   N,
                                                   K, // problem dims
                                                   batch,
                                                   transA,
                                                   transB,
                                                   MT_M,
                                                   MT_N,
                                                   MT_K,
                                                   MI_M,
                                                   MI_N,
                                                   MI_K,
                                                   element_size_A, //ElementSizeA
                                                   element_size_B, //ElementSizeB
                                                   element_size_out, //ElementSizeout
                                                   mi_datatype,
                                                   mx_block_size,
                                                   WGM,
                                                   0,
                                                   0,
                                                   0,
                                                   split);

            if(latency < best_latency)
            {
                best_latency = latency;
                best_split   = split;
            }
        }
        size_t best_grid = best_split * grid;

        // you now have both `grid` and `best_split`—
        // return whichever is appropriate (here we stick with split):
        return best_grid;
    }

    std::vector<result_tuple> select_best_macro_tile_size(size_t                         M,
                                                          size_t                         N,
                                                          size_t                         K,
                                                          size_t                         batch,
                                                          bool                           transA,
                                                          bool                           transB,
                                                          const hardware_t&              hardware,
                                                          const std::vector<tile_tuple>& MT_list,
                                                          size_t      element_size_A, //In bits
                                                          size_t      element_size_B, //In bits
                                                          size_t      element_size_out, //In bits
                                                          data_type_t mi_datatype,
                                                          size_t      mx_block_size,
                                                          double      H_L2,
                                                          bool        print,
                                                          size_t      defaultWGM)
    {
        std::vector<result_tuple> valid_results;
        valid_results.reserve(MT_list.size());

        // bool tf32_emu = ((mi_datatype == data_type_t::XFloat32)
        //                  && (hardware.arch == hardware_t::architecture_t::gfx950));

        for(const auto& mt : MT_list)
        {
            size_t MT_M           = std::get<0>(mt);
            size_t MT_N           = std::get<1>(mt);
            size_t MT_K           = std::get<2>(mt);
            size_t MI_M           = std::get<3>(mt);
            size_t MI_N           = std::get<4>(mt);
            size_t MI_K           = std::get<5>(mt);
            size_t occupancy      = std::get<6>(mt);
            size_t WGM            = std::get<7>(mt);
            size_t non_temporal_a = std::get<8>(mt);
            size_t non_temporal_b = std::get<9>(mt);

            if(hardware_t::is_debug_enabled())
            {
                std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << ", MI_M=" << MI_M << ", MI_N=" << MI_N << ", MI_K=" << MI_K << "\n";
            }

            if(check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size_A))
            {
                double Total_latency = compute_total_latency(hardware,
                                                             M,
                                                             N,
                                                             K,
                                                             batch,
                                                             transA,
                                                             transB,
                                                             MT_M,
                                                             MT_N,
                                                             MT_K,
                                                             MI_M,
                                                             MI_N,
                                                             MI_K,
                                                             element_size_A,
                                                             element_size_B,
                                                             element_size_out,
                                                             mi_datatype,
                                                             mx_block_size,
                                                             WGM,
                                                             non_temporal_a,
                                                             non_temporal_b,
                                                             occupancy,
                                                             0);

                valid_results.emplace_back(Total_latency,
                                           MT_M,
                                           MT_N,
                                           MT_K,
                                           MI_M,
                                           MI_N,
                                           MI_K,
                                           occupancy,
                                           WGM,
                                           non_temporal_a,
                                           non_temporal_b);
            }
            else if(hardware_t::is_debug_enabled())
            {
                std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << " due to LDS capacity\n";
            }
        }

        if(valid_results.empty())
        {
            throw std::runtime_error("No valid macro-tile sizes found.");
        }

        // 1) Sort results by ascending latency.
        std::stable_sort(
            valid_results.begin(), valid_results.end(), [](auto const& a, auto const& b) {
                return std::get<0>(a) < std::get<0>(b);
            });

        // 2) Collect results that tie for the absolute best latency.
        double best_latency = std::get<0>(valid_results.front());
        size_t num_the_same = 0;

        // Count the number of similar latencies
        for(const auto& res : valid_results)
        {
            double diff = std::fabs(std::get<0>(res) - best_latency);
            diff /= best_latency;
            // If it's within 1%, include it.
            if(diff < 0.01)
                num_the_same++;
            else
                break; // Once we pass best_latency, we can stop.
        }
        // 3) If that tie group has at least 10 entries, we only use those.
        // 4) Otherwise, keep adding the next best latencies until we have 10 total or run out.
        // std::vector<result_tuple> top_candidates = tie_results;
        // if(tie_results.size() < 10)
        // {
        //     size_t i = tie_results.size();
        //     while(top_candidates.size() < 10 && i < valid_results.size())
        //     {
        //         top_candidates.push_back(valid_results[i]);
        //         i++;
        //     }
        // }
        // Now ‘top_candidates’ is either all the tied best-latency results (if >=10),
        // or the top 10 latencies overall (including however many best-latency entries there were).

        // Finally, use your existing tie-breaker on top_candidates
        pick_best_tile_by_arithmetic_intensity(valid_results, num_the_same);
        if(print)
        {
            for(const auto& tile : valid_results)
            {
                std::cout << M << "x" << N << "x" << K
                          << "Selected Macro-Tile: Latency=" << std::get<0>(tile)
                          << ", MT_M=" << std::get<0>(tile) << ", MT_N=" << std::get<1>(tile)
                          << ", MT_K=" << std::get<2>(tile) << ", MI_M=" << std::get<3>(tile)
                          << ", MI_N=" << std::get<4>(tile) << ", MI_K=" << std::get<5>(tile)
                          << ", Occupancy=" << std::get<6>(tile) << ", WGM=" << std::get<7>(tile)
                          << ", NonTemporalA=" << std::get<8>(tile)
                          << ", NonTemporalB=" << std::get<9>(tile) << "\n";
            }
        }

        return valid_results;
    }

    /*!
         * \brief Selects the best WGM (maximizing L2 hit rate) given fixed macro tile sizes.
         *
         * \param[in] M, N, K    - your overall problem sizes
         * \param[in] hardware   - a struct describing your hardware capabilities
         * \param[in] MT_M,MT_N,MT_K,MI_M,MI_N,MI_K - chosen macro/MI tile sizes
         * \param[in] WGM_list   - candidate WGM values to try
         * \param[in] element_size
         * \param[in] H_L2       - some hardware-related constant or factor (no longer used here,
         *                         but kept if your signature or other usage requires it)
         * \param[in] print      - whether to print the final best result
         *
         * \return A pair: (best_l2_hit_rate, best_WGM).
         */
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
        bool   print)
    {
        using WGMResult = std::pair<double, size_t>; // (l2_hit_rate, WGM)

        std::vector<WGMResult> valid_results;
        valid_results.reserve(WGM_list.size());

        // Iterate over all candidate WGM values
        for(const auto& candidate_wgm : WGM_list)
        {
            if(hardware_t::is_debug_enabled())
            {
                std::cout << "Evaluating WGM=" << candidate_wgm << "\n";
            }

            // Optionally ensure we do not exceed LDS capacity
            // (If you want to factor in WGM, add it to your check_lds_capacity signature.)
            // For now, let's just check the tile itself:
            if(!check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size))
            {
                if(hardware_t::is_debug_enabled())
                {
                    std::cout << "Skipping WGM=" << candidate_wgm << " due to LDS capacity.\n";
                }
                continue;
            }

            // Compute L2 hit rate for this WGM
            double current_hit = estimate_l2_hit(hardware,
                                                 M,
                                                 N,
                                                 K,
                                                 batch,
                                                 MT_M,
                                                 MT_N,
                                                 MT_K,
                                                 element_size,
                                                 static_cast<int>(candidate_wgm),
                                                 1 /* splittingFactor */);

            valid_results.emplace_back(current_hit, candidate_wgm);
        }

        // If no valid WGM was found, throw an error
        if(valid_results.empty())
        {
            throw std::runtime_error("No valid WGM found.");
        }

        // Find the maximum L2 hit rate in valid_results
        // (Use max_element on the first value in the pair.)
        auto best_it = std::max_element(
            valid_results.begin(), valid_results.end(), [](const WGMResult& a, const WGMResult& b) {
                return a.first < b.first; // "less" => a has smaller hit rate than b
            });

        double best_l2_hit = best_it->first;
        size_t best_wgm    = best_it->second;

        // Return (l2_hit_rate, WGM)
        return std::make_pair(best_l2_hit, best_wgm);
    }

    /*!
     * \brief Selects the best WGM (using DecisionTreeClassifier), should be in [1, 36].
     * At the moment, macrotile, matrix instruction and element_size are not used.
     *
     * \param[in] M, N, K, batch                - problem sizes
     * \param[in] hardware                      - a struct describing your hardware capabilities
     * \param[in] MT_M,MT_N,MT_K,MI_M,MI_N,MI_K - chosen macro/MI tile sizes
     * \param[in] element_size                  - element size in bytes
     * \param[in] WorkGroupMappingXCC
     * \param[in] StreamKXCCMapping
     *
     * \return best WGM
     */
    size_t select_best_wgm_tree(
        size_t            M,
        size_t            N,
        size_t            K,
        size_t            batch,
        const hardware_t& hardware,
        size_t            MT_M,
        size_t            MT_N,
        size_t            MT_K,
        size_t            MI_M,
        size_t            MI_N,
        size_t            MI_K,
        size_t            element_size,
        int               WorkGroupMappingXCC,
        int               StreamKXCCMapping)
    {
        // tree only tuned for gfx950

        const std::array left =
                {1, 2, 3, 4, 5, 6, 7, -1, -1, 10, -1, -1, 13, 14, -1, -1, 17, -1, -1, 20, 21, 22, -1, -1, 25, -1, -1, 28, 29, -1,
                 -1, 32, -1, -1, 35, 36, 37, 38, -1, -1, 41, -1, -1, 44, 45, -1, -1, 48, -1, -1, 51, 52, 53, -1, -1, 56, -1, -1, 59, 60,
                 -1, -1, 63, -1, -1, 66, 67, 68, 69, 70, -1, -1, 73, -1, -1, 76, 77, -1, -1, 80, -1, -1, 83, 84, 85, -1, -1, 88, -1, -1,
                 91, 92, -1, -1, 95, -1, -1, 98, 99, 100, 101, -1, -1, 104, -1, -1, 107, 108, -1, -1, 111, -1, -1, 114, 115, 116, -1, -1, 119, -1,
                 -1, 122, 123, -1, -1, 126, -1, -1, 129, 130, 131, 132, 133, 134, -1, -1, 137, -1, -1, 140, 141, -1, -1, 144, -1, -1, 147, 148, 149, -1,
                 -1, 152, -1, -1, 155, 156, -1, -1, 159, -1, -1, 162, 163, 164, 165, -1, -1, 168, -1, -1, 171, 172, -1, -1, 175, -1, -1, 178, 179, 180,
                 -1, -1, 183, -1, -1, 186, 187, -1, -1, 190, -1, -1, 193, 194, 195, 196, 197, -1, -1, 200, -1, -1, 203, 204, -1, -1, 207, -1, -1, 210,
                 211, 212, -1, -1, 215, -1, -1, 218, 219, -1, -1, 222, -1, -1, 225, 226, 227, 228, -1, -1, 231, -1, -1, 234, 235, -1, -1, 238, -1, -1,
                 241, 242, 243, -1, -1, 246, -1, -1, 249, 250, -1, -1, 253, -1, -1, };
        const std::array right =
                {128, 65, 34, 19, 12, 9, 8, -1, -1, 11, -1, -1, 16, 15, -1, -1, 18, -1, -1, 27, 24, 23, -1, -1, 26, -1, -1, 31, 30, -1,
                 -1, 33, -1, -1, 50, 43, 40, 39, -1, -1, 42, -1, -1, 47, 46, -1, -1, 49, -1, -1, 58, 55, 54, -1, -1, 57, -1, -1, 62, 61,
                 -1, -1, 64, -1, -1, 97, 82, 75, 72, 71, -1, -1, 74, -1, -1, 79, 78, -1, -1, 81, -1, -1, 90, 87, 86, -1, -1, 89, -1, -1,
                 94, 93, -1, -1, 96, -1, -1, 113, 106, 103, 102, -1, -1, 105, -1, -1, 110, 109, -1, -1, 112, -1, -1, 121, 118, 117, -1, -1, 120, -1,
                 -1, 125, 124, -1, -1, 127, -1, -1, 192, 161, 146, 139, 136, 135, -1, -1, 138, -1, -1, 143, 142, -1, -1, 145, -1, -1, 154, 151, 150, -1,
                 -1, 153, -1, -1, 158, 157, -1, -1, 160, -1, -1, 177, 170, 167, 166, -1, -1, 169, -1, -1, 174, 173, -1, -1, 176, -1, -1, 185, 182, 181,
                 -1, -1, 184, -1, -1, 189, 188, -1, -1, 191, -1, -1, 224, 209, 202, 199, 198, -1, -1, 201, -1, -1, 206, 205, -1, -1, 208, -1, -1, 217,
                 214, 213, -1, -1, 216, -1, -1, 221, 220, -1, -1, 223, -1, -1, 240, 233, 230, 229, -1, -1, 232, -1, -1, 237, 236, -1, -1, 239, -1, -1,
                 248, 245, 244, -1, -1, 247, -1, -1, 252, 251, -1, -1, 254, -1, -1, };
        const std::array threshold =
                {10018.5, 2340.5, 791.5, 194.5, 415.5, 1.5, 403.5, -2.0, -2.0, 4.5, -2.0, -2.0, 481.5, 8021.0, -2.0, -2.0, 471.5, -2.0, -2.0, 1.5,
                 1723.5, 466.5, -2.0, -2.0, 992.0, -2.0, -2.0, 2332.0, 2.5, -2.0, -2.0, 3988.0, -2.0, -2.0, 147.5, 1.5, 34.5, 118.5, -2.0, -2.0,
                 109.5, -2.0, -2.0, 11.5, 139.0, -2.0, -2.0, 140.5, -2.0, -2.0, 2.5, 422.5, 2888.5, -2.0, -2.0, 2262.5, -2.0, -2.0, 6.5, 49160.0,
                 -2.0, -2.0, 283.5, -2.0, -2.0, 2332.5, 1.5, 525.5, 183.5, 3217.5, -2.0, -2.0, 167.5, -2.0, -2.0, 797.5, 18.5, -2.0, -2.0, 7464.5,
                 -2.0, -2.0, 3555.0, 2.5, 4.0, -2.0, -2.0, 4.5, -2.0, -2.0, 64.5, 3.5, -2.0, -2.0, 4125.5, -2.0, -2.0, 2951.0, 252.5, 6316.5,
                 2349.5, -2.0, -2.0, 1.5, -2.0, -2.0, 4.5, 758.5, -2.0, -2.0, 65.0, -2.0, -2.0, 1.5, 8223.0, 8921.5, -2.0, -2.0, 5977.5, -2.0,
                 -2.0, 32747.0, 1.5, -2.0, -2.0, 19566.0, -2.0, -2.0, 1.5, 4596.5, 909.5, 333.5, 2.5, 1.5, -2.0, -2.0, 4.5, -2.0, -2.0, 31.5,
                 1048.5, -2.0, -2.0, 347.0, -2.0, -2.0, 527.5, 8929.0, 2633.0, -2.0, -2.0, 51.5, -2.0, -2.0, 8351.0, 4.5, -2.0, -2.0, 1.5, -2.0,
                 -2.0, 493.5, 21001.0, 1.5, 109.5, -2.0, -2.0, 7.5, -2.0, -2.0, 32978.0, 47916.0, -2.0, -2.0, 42.0, -2.0, -2.0, 1.5, 10076.0, 3326.5,
                 -2.0, -2.0, 9307.0, -2.0, -2.0, 23229.0, 3.5, -2.0, -2.0, 15553.0, -2.0, -2.0, 3.0, 19069.0, 7103.0, 4.0, 2222.0, -2.0, -2.0, 7845.0,
                 -2.0, -2.0, 12327.0, 11739.0, -2.0, -2.0, 13820.0, -2.0, -2.0, 20158.0, 12455.0, 1615.0, -2.0, -2.0, 26066.0, -2.0, -2.0, 17205.0, 14943.0, -2.0,
                 -2.0, 24302.0, -2.0, -2.0, 5576.0, 641.0, 3986.0, 637.0, -2.0, -2.0, 13187.0, -2.0, -2.0, 10628.0, 3145.0, -2.0, -2.0, 22746.0, -2.0, -2.0,
                 4.0, 12.0, 20337.0, -2.0, -2.0, 8216.0, -2.0, -2.0, 24482.0, 8586.0, -2.0, -2.0, 14509.0, -2.0, -2.0, };
        const std::array feature_wgm =
                {1, 2, 1, 1, 0, 3, 0, 33, 33, 3, 18, 1, 2, 0, 1, 1, 0, 34, 1, 3, 0, 2, 1, 1, 2, 18, 22, 2, 3, 35,
                 1, 0, 4, 33, 0, 3, 0, 2, 33, 1, 0, 1, 1, 3, 0, 1, 32, 2, 1, 1, 3, 2, 1, 32, 32, 0, 21, 18, 3, 0,
                 23, 20, 2, 1, 3, 1, 3, 0, 1, 2, 14, 33, 0, 1, 19, 1, 1, 33, 17, 2, 6, 3, 0, 3, 5, 34, 30, 3, 17, 34,
                 1, 3, 31, 34, 2, 27, 36, 0, 0, 2, 2, 23, 1, 3, 27, 34, 3, 0, 36, 6, 3, 2, 3, 3, 0, 2, 6, 3, 2, 5,
                 3, 2, 4, 4, 16, 0, 34, 31, 4, 0, 2, 0, 3, 3, 1, 1, 3, 1, 1, 3, 0, 31, 31, 0, 24, 1, 0, 2, 2, 1,
                 27, 3, 33, 36, 2, 3, 15, 4, 3, 3, 2, 2, 0, 3, 2, 31, 15, 3, 18, 1, 1, 0, 9, 7, 2, 2, 2, 3, 2, 2,
                 7, 4, 0, 3, 2, 1, 3, 5, 5, 0, 2, 4, 4, 2, 0, 5, 0, 4, 36, 2, 24, 35, 1, 1, 20, 27, 1, 12, 12, 0,
                 0, 0, 29, 20, 2, 20, 20, 1, 1, 24, 16, 2, 24, 20, 0, 0, 2, 2, 1, 1, 1, 21, 35, 1, 2, 36, 19, 2, 2, 1,
                 5, 4, 2, 6, 14, 2, 32, 2, 2, 2, 16, 36, 0, 30, 24, };

        const std::array<float,6> key =
            {static_cast<float>(M * element_size),
             static_cast<float>(N * element_size),
             static_cast<float>(K * element_size),
             static_cast<float>(batch * element_size),
             static_cast<float>(WorkGroupMappingXCC),
             static_cast<float>(StreamKXCCMapping)};

        int node = 0, branches = 0;
        while(left[node] >= 0)
        {
            node = (key[feature_wgm[node]] <= threshold[node]) ? left[node] : right[node];
            branches++;
        }

        if(hardware_t::is_debug_enabled())
        {
            std::cout << "Traversed DecisionTreeClassifier for WGM prediction." << std::endl;
            std::cout << "  Key = {M: " << key[0] << ", N: " << key[1] << ", K: " << key[2] << ", batch: " << key[3]
                      << ", WGMXCC: " << key[4] << ", SKXCCM: " << key[5] << "}" << std::endl;
            std::cout << "  Tree has " << left.size() << " nodes. Traversal required "
                      << branches << " comparisons." << std::endl;
            std::cout << "  Predicted WGM: " << feature_wgm[node] << std::endl;
        }

        return feature_wgm[node];
    }


    // Logic to decide between two MT that are "tied"
    std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_macro_tile_sizes(
        const std::vector<std::tuple<double, size_t, size_t, size_t>>& top_results,
        size_t                                                         M,
        size_t                                                         N,
        size_t                                                         K,
        hardware_t&                                                    hardware,
        std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, hardware_t&)>
            tie_breaker_fn)
    {
        std::vector<std::tuple<double, size_t, size_t, size_t>> tie_breaker_results;

        for(const auto& res : top_results)
        {
            size_t MT_M = std::get<1>(res);
            size_t MT_N = std::get<2>(res);
            size_t MT_K = std::get<3>(res);

            // Call user-provided tie-breaking function
            double precise_latency = tie_breaker_fn(M, N, K, MT_M, MT_N, MT_K, hardware);

            tie_breaker_results.emplace_back(precise_latency, MT_M, MT_N, MT_K);
        }

        // Sort results by precise_latency (ascending order)
        std::stable_sort(tie_breaker_results.begin(), tie_breaker_results.end());

        return tie_breaker_results;
    }

    std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>>
        rank_macro_tile_sizes(
            size_t                         M,
            size_t                         N,
            size_t                         K,
            bool                           transA,
            bool                           transB,
            hardware_t&                    hardware,
            const std::vector<tile_tuple>& MT_list,
            size_t                         element_size,
            data_type_t                    mi_datatype,
            double                         H_L2,
            bool                           print,
            size_t                         WGM,
            std::function<double(size_t, size_t, size_t, size_t, size_t, size_t, hardware_t&)>
                tie_breaker_fn)
    {
        std::vector<std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t>> results;

        typedef std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t> result_tuple;

        for(size_t i = 0; i < MT_list.size(); ++i)
        {
            size_t MT_M = std::get<0>(MT_list[i]);
            size_t MT_N = std::get<1>(MT_list[i]);
            size_t MT_K = std::get<2>(MT_list[i]);
            size_t MI_M = std::get<3>(MT_list[i]);
            size_t MI_N = std::get<4>(MT_list[i]);
            size_t MI_K = std::get<5>(MT_list[i]);

            if(hardware_t::is_debug_enabled())
            {
                std::cout << "Evaluating MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << ", MI_M=" << MI_M << ", MI_N=" << MI_N << ", MI_K=" << MI_K << "\n";
            }

            if(check_lds_capacity(hardware, MT_M, MT_N, MT_K, element_size))
            {
                size_t split         = 1;
                size_t mx_block_size = 0;
                double Total_latency = compute_total_latency(hardware,
                                                             M,
                                                             N,
                                                             K,
                                                             1, //Batch
                                                             transA,
                                                             transB,
                                                             MT_M,
                                                             MT_N,
                                                             MT_K,
                                                             MI_M,
                                                             MI_N,
                                                             MI_K,
                                                             element_size * 8, //Element Size A
                                                             element_size * 8, //Element Size B
                                                             element_size * 8, //Element Size out
                                                             mi_datatype,
                                                             mx_block_size,
                                                             WGM,
                                                             0,
                                                             0,
                                                             0,
                                                             split);

                results.push_back(
                    std::make_tuple(Total_latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K));
            }
            else if(hardware_t::is_debug_enabled())
            {
                std::cout << "Skipping MT_M=" << MT_M << ", MT_N=" << MT_N << ", MT_K=" << MT_K
                          << " due to LDS capacity\n";
            }
        }

        // Sort results by Total_latency, from worst (largest latency) to best (smallest latency)
        std::stable_sort(
            results.begin(), results.end(), [](const result_tuple& a, const result_tuple& b) {
                return std::get<0>(a) > std::get<0>(b);
            });

        if(!results.empty())
        {
            double best_latency = std::get<0>(results.back());

            std::vector<result_tuple> top_results;
            for(size_t i = 0; i < results.size(); ++i)
            {
                if(std::abs(std::get<0>(results[i]) - best_latency) < 1e-6)
                {
                    top_results.push_back(results[i]);
                }
            }

            if(top_results.size() > 1)
            {
                if(hardware_t::is_debug_enabled())
                {
                    std::cout << "Tie detected among top-ranked tile sizes. Applying "
                                 "tie-breaker...\n";
                }

                // Compute tie-breaker scores and store them along with the result indices
                std::vector<std::pair<double, size_t>>
                    tie_breaker_scores; // (score, index in top_results)

                for(size_t i = 0; i < top_results.size(); ++i)
                {
                    const result_tuple& res  = top_results[i];
                    size_t              MT_M = std::get<1>(res);
                    size_t              MT_N = std::get<2>(res);
                    size_t              MT_K = std::get<3>(res);
                    size_t              MI_M = std::get<4>(res);
                    size_t              MI_N = std::get<5>(res);
                    size_t              MI_K = std::get<6>(res);
                    double score = tie_breaker_fn(MT_M, MT_N, MT_K, MI_M, MI_N, MI_K, hardware);

                    tie_breaker_scores.push_back(std::make_pair(score, i));
                }

                // Now sort the tie_breaker_scores based on score
                std::stable_sort(
                    tie_breaker_scores.begin(),
                    tie_breaker_scores.end(),
                    [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                        return a.first > b.first;
                    });

                // Now re-order 'top_results' based on sorted indices
                std::vector<result_tuple> sorted_top_results;
                for(size_t i = 0; i < tie_breaker_scores.size(); ++i)
                {
                    size_t idx = tie_breaker_scores[i].second;
                    sorted_top_results.push_back(top_results[idx]);
                }

                // Remove the tied results from 'results' and insert the sorted 'sorted_top_results'
                results.erase(std::remove_if(results.begin(),
                                             results.end(),
                                             [best_latency](const result_tuple& res) {
                                                 return std::abs(std::get<0>(res) - best_latency)
                                                        < 1e-6;
                                             }),
                              results.end());

                results.insert(results.end(), sorted_top_results.begin(), sorted_top_results.end());
                // No need to re-sort results as total_latency remains same for tied results
            }
        }

        if(print)
        {
            std::cout << "Total Latency\tMT_M\tMT_N\tMT_K\tMI_M\tMI_N\tMI_K\n";
            for(size_t i = 0; i < results.size(); ++i)
            {
                double latency = std::get<0>(results[i]);
                size_t MT_M    = std::get<1>(results[i]);
                size_t MT_N    = std::get<2>(results[i]);
                size_t MT_K    = std::get<3>(results[i]);
                size_t MI_M    = std::get<4>(results[i]);
                size_t MI_N    = std::get<5>(results[i]);
                size_t MI_K    = std::get<6>(results[i]);
                std::cout << std::fixed << std::setprecision(2) << latency << "\t" << MT_M << "\t"
                          << MT_N << "\t" << MT_K << "\t" << MI_M << "\t" << MI_N << "\t" << MI_K
                          << "\n";
            }
        }

        return results;
    }

    double compute_tflops_from_latency(
        double latency_cycles, size_t M, size_t N, size_t K, double clock_GHz)
    {
        // Compute total FLOPs
        double total_FLOPs = 2.0 * M * N * K; // For GEMM, each multiply-add is 2 FLOPs
        // Compute total time in seconds
        double cycles_per_second  = clock_GHz * 1e9; // 1 GHz = 1e9 cycles per second
        double total_time_seconds = latency_cycles / cycles_per_second;
        // Compute performance in FLOPS
        double FLOPS = total_FLOPs / total_time_seconds;
        // Convert to TFLOPS
        double TFLOPS = FLOPS / 1e12; // 1 TFLOP = 1e12 FLOPs

        if(hardware_t::is_debug_enabled())
        {
            std::cout << "Total FLOPs: " << total_FLOPs << "\n";
            std::cout << "Total Time: " << total_time_seconds << " seconds\n";
            std::cout << "Performance: " << FLOPS << " FLOPS\n";
            std::cout << "Achieved Performance: " << TFLOPS << " TFLOPS\n";
        }

        return TFLOPS;
    }
} // namespace origami
