/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <queue>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/MLFeatures.hpp>
#include <Tensile/MLPClassification.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a small neural network to rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MLPClassificationLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using MLPNet           = MLPClassification::MLPNet;
        using SolutionFeatures = std::vector<std::shared_ptr<MLFeatures::MLFeature<MySolution>>>;
        using ProblemFeatures  = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;

        std::map<int, std::shared_ptr<MySolution>> solutionmap;
        std::shared_ptr<MLPNet>                    model;
        SolutionFeatures                           solFeatures;
        ProblemFeatures                            probFeatures;
        std::vector<analytical::TileTuple>         tile_list;

        static std::string Type()
        {
            return "MLPClassification";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(model == nullptr)
                return concatenate(type(), ", MLPNet: nullptr");
            else
                return concatenate(type(), ": ", model->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // If the experimental library mode is not on treat it like it asserted out
                return nullptr;
            }
            // ;
            auto indexMatch = solutionmap.find(index);
            if(indexMatch != solutionmap.end())
                return indexMatch->second;
            return nullptr;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            SolutionVector<MySolution>  solutions = findTopSolutions(problem, hardware, 1);
            std::shared_ptr<MySolution> solution  = nullptr;
            if(solutions.size() > 0)
                solution = solutions[0];
            return solution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }
            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            std::vector<float> problemkey
                = ProblemKey::keyForProblem<std::vector<float>, MyProblem, float>(
                    problem, this->probFeatures);

            bool                  debug   = Debug::Instance().printPropertyEvaluation();
            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            const analytical::Hardware& hw = *(pAMDGPU->analyticalHardware);
            int WGM = std::sqrt(std::floor(hw.N_CU / hw.NUM_XCD));
            analytical::DataType miDataType = static_cast<analytical::DataType>(problem.computeInputType());
            if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32) // Check F32 compute type
                miDataType = analytical::DataType::XFloat32;
            auto selected_tiles = analytical::select_best_macro_tile_size(
                problemkey[0],
                problemkey[1],
                problemkey[3],
                problemkey[2],
                problem.transA(),
                problem.transB(),
                hw,
                tile_list,
                problem.a().elementBytes() * 8,
                problem.b().elementBytes() * 8,
                problem.c().elementBytes() * 8,
                miDataType,
                0, //mx_block_size -> MX Data types come from rocroller.
                0.8,
                debug,
                false,
                WGM);

            auto Fhidden = model->predict_hidden(problemkey);
            std::vector<bool> mask(solutionmap.size());

            SolutionVector<MySolution> rv;

            for(const auto& tile : selected_tiles)
            {
                size_t i = 0;
                for(const auto& s : solutionmap)
                {
                    mask[i] =
                        std::get<1>(tile) == s.second->sizeMapping.macroTile.x &&
                        std::get<2>(tile) == s.second->sizeMapping.macroTile.y &&
                        std::get<3>(tile) == s.second->sizeMapping.depthU &&
                        std::get<4>(tile) == s.second->sizeMapping.matrixInstruction[0] &&
                        std::get<5>(tile) == s.second->sizeMapping.matrixInstruction[1] &&
                        std::get<6>(tile) == s.second->sizeMapping.matrixInstruction[2] &&
                        std::get<7>(tile) == s.second->sizeMapping.CUOccupancy;
                    i++;
                }
                auto logits = model->dense(Fhidden, mask);

                std::vector<std::pair<decltype(logits)::value_type,
                                      std::shared_ptr<MySolution>*>> solution_ranking;
                solution_ranking.reserve(solutionmap.size());
                i = 0;
                for(const auto& s : solutionmap)
                {
                    if(mask[i])
                        solution_ranking.emplace_back(logits[s.second->libraryLogicIndex],
                            (std::shared_ptr<MySolution>*)(&s.second));
                    i++;
                }
                std::sort(solution_ranking.begin(), solution_ranking.end());
                for(auto& s : solution_ranking)
                {
                    auto& solution = *s.second;
                    if((*solution->hardwarePredicate)(hardware) &&
                       (*solution->problemPredicate)(problem))
                    {
                        rv.emplace_back(solution);
                        if(rv.size() == numSolutions)
                            return rv;
                    }
                }
            }
            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }
    };

} // namespace TensileLite
