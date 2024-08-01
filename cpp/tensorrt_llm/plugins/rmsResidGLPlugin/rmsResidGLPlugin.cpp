/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "rmsResidGLPlugin.h"

// #include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
// #include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/lookupGLRmsKernels.h"
// #include <nccl.h>
#include <unordered_set>

using namespace nvinfer1;
using tensorrt_llm::plugins::RmsResidGLPluginCreator;
using tensorrt_llm::plugins::RmsResidGLPlugin;

static char const* RMSRESID_PLUGIN_VERSION{"1"};
static char const* RMSRESID_PLUGIN_NAME{"RmsResidGL"};
PluginFieldCollection RmsResidGLPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RmsResidGLPluginCreator::mPluginAttributes;

RmsResidGLPlugin::RmsResidGLPlugin(nvinfer1::DataType type)
    : mType(type)
{
}

// Parameterized constructor
RmsResidGLPlugin::RmsResidGLPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RmsResidGLPlugin::clone() const noexcept
{
    auto* plugin = new RmsResidGLPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RmsResidGLPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool RmsResidGLPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    bool res = false;
    if(nbInputs == 3) {
        switch (pos) {
            case 0:
            case 2:
            case 3:
            case 4:
                res = ((inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR)); break;
            case 1:
                res = ((inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR)); break;//TODO can be change to quant type
        }
        
    }
    return res;
}

void RmsResidGLPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t RmsResidGLPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RmsResidGLPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{   // inputs
    //     hidStat  [batchSz,seqLen,hidSz]
    //     gamma    [hidSz]
    //     residual [batchSz,seqLen,hidSz]
    // outputs
    //     hidAfterRms      [batchSz,seqLen,hidSz]
    //     residualAfterAdd [batchSz,seqLen,hidSz]
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    auto const sizePerElem = common::getDTypeSize(mType);

    tensorrt_llm::kernels::gl::RmsParams params;
    params.fusion_params.bias_buffer = nullptr;
    params.fusion_params.residual_buffer = inputs[2];
    params.fusion_params.weight_buffer = inputs[1];
    params.local_output_buffer_ptr = outputs[0];
    params.fusion_params.residual_out = outputs[1];
    params.elts_total = size;
    params.fusion_params.hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    params.fusion_params.intermediate_buffer = inputs[0];
    tensorrt_llm::kernels::gl::residualRmsNorm(params, mType, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RmsResidGLPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return mType;
}

// IPluginV2 Methods

char const* RmsResidGLPlugin::getPluginType() const noexcept
{
    return RMSRESID_PLUGIN_NAME;
}

char const* RmsResidGLPlugin::getPluginVersion() const noexcept
{
    return RMSRESID_PLUGIN_VERSION;
}

int RmsResidGLPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int RmsResidGLPlugin::initialize() noexcept
{
    return 0;
}

void RmsResidGLPlugin::terminate() noexcept {}

size_t RmsResidGLPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType);
}

void RmsResidGLPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void RmsResidGLPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

RmsResidGLPluginCreator::RmsResidGLPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* RmsResidGLPluginCreator::getPluginName() const noexcept
{
    return RMSRESID_PLUGIN_NAME;
}

char const* RmsResidGLPluginCreator::getPluginVersion() const noexcept
{
    return RMSRESID_PLUGIN_VERSION;
}

PluginFieldCollection const* RmsResidGLPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RmsResidGLPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new RmsResidGLPlugin(type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RmsResidGLPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RmsResidGLPlugin::destroy()
    try
    {
        auto* obj = new RmsResidGLPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
