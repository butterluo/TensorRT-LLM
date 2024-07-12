/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tensorrt_llm/common/quantization.h"
#include "btllm/kernels/cutlass_kernels/btfpA_intB_gemm/btfpA_intB_gemm.h" //@#
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv//kernelLauncher.h"
#include "btllm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <cutlass/numeric_types.h>

#include <cassert>
#include <cuda_runtime.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace btllm::plugins
{

using WeightOnlyGemmRunner = btllm::kernels::cutlass_kernels::BTCutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class BTWeightOnlyGroupwiseQuantGemmPluginProfiler
    : public btllm::plugins::GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
          btllm::plugins::GemmIdCore, btllm::plugins::GemmIdCoreHash>
{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

    void setQuantAlgo(int quantAlgo)
    {
        mQuantAlgo = quantAlgo;
    }

    void setGroupSize(int groupSize)
    {
        mGroupSize = groupSize;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    int mQuantAlgo;
    int mGroupSize;
};

class BTWeightOnlyGroupwiseQuantMatmulPlugin //: public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<BTWeightOnlyGroupwiseQuantGemmPluginProfiler>;

    BTWeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    BTWeightOnlyGroupwiseQuantMatmulPlugin(
        nvinfer1::DataType type, int quant_algo, int group_size, PluginProfilerPtr const& profiler);
    
    BTWeightOnlyGroupwiseQuantMatmulPlugin(
        nvinfer1::DataType type, int quant_algo, int group_size, void* profiler)
    : mPluginProfiler( ((PluginProfilerPtr*)profiler)[0] ) {
        printf("**** CAN NOT init BTWeightOnlyGroupwiseQuantMatmulPlugin !!! Use creator instead! *****");
        // init(type, quant_algo, group_size);
    }

    BTWeightOnlyGroupwiseQuantMatmulPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~BTWeightOnlyGroupwiseQuantMatmulPlugin() = default;

    // IPluginV2DynamicExt Methods
    // nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    // nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
    //     nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    // bool supportsFormatCombination(
    //     int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    // void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    //     nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept;// override;
    void configurePlugin(const int maxM, const int maxK, const int maxN) noexcept;
    size_t getWorkspaceSize() const noexcept;// override;
    int enqueue(int const m, int const real_n, int const k, 
          void const* zeros_ptr, void const* weight_scales_ptr, 
          void const* act_ptr, void const* weight_ptr, void const* biases_ptr, void* output_ptr,  
          void* workspace, cudaStream_t stream) noexcept;// override;

    // IPluginV2Ext Methods
    // nvinfer1::DataType getOutputDataType(
    //     int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;
    //
    // IPluginV2 Methods
    // char const* getPluginType() const noexcept override;
    // char const* getPluginVersion() const noexcept override;
    // int getNbOutputs() const noexcept override;
    int initialize() noexcept; // override;
    // void terminate() noexcept override;
    // size_t getSerializationSize() const noexcept override;
    // void serialize(void* buffer) const noexcept override;
    // void destroy() noexcept; // override;

private:
    // group_size: 64, 128
    void init(nvinfer1::DataType type, int quant_algo, int group_size);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGroupwiseGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    bool mCudaKernelEnabled;
    tensorrt_llm::kernels::weight_only::KernelType mCudaKernelType;
    int mArch;

    // When M is smaller than this value, we trigger a fast path
    // I.e. a tailored kernel instead of cutlass.
    static constexpr int SMALL_M_FAST_PATH = 5;

    int mQuantAlgo;

    int mGroupSize;

    int mPreQuantScaleInputIdx;
    int mWeightInputIdx;
    int mScalesInputIdx;
    int mZerosInputIdx;
    int mBiasesInputIdx;
    int mAlphaInputIdx;

    btllm::plugins::GemmDims mDims{};
    btllm::plugins::GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class BTWeightOnlyGroupwiseQuantMatmulPluginCreator //: public BaseCreator
{
public:
    BTWeightOnlyGroupwiseQuantMatmulPluginCreator();

    // char const* getPluginName() const noexcept override;
    //
    // char const* getPluginVersion() const noexcept override;
    //
    // nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    // nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    BTWeightOnlyGroupwiseQuantMatmulPlugin* createPlugin(char const* name, nvinfer1::DataType type, int quant_algo, int group_size,
          const int maxM, const int maxK, const int maxN) noexcept;

    // nvinfer1::IPluginV2* deserializePlugin(
    //     char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    btllm::plugins::GemmPluginProfilerManager<BTWeightOnlyGroupwiseQuantGemmPluginProfiler> gemmPluginProfileManager;
    // static nvinfer1::PluginFieldCollection mFC;
    // static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace btllm::plugins
