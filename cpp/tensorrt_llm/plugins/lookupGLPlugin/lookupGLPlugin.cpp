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

#include <cstdio>

#include "lookupGLPlugin.h"
#include "tensorrt_llm/kernels/lookupGLRmsKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::LookupGLPluginCreator;
using tensorrt_llm::plugins::LookupGLPlugin;

static char const* LOOKUP_PLUGIN_VERSION{"1"};
static char const* LOOKUP_PLUGIN_NAME{"LookupGL"};
PluginFieldCollection LookupGLPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LookupGLPluginCreator::mPluginAttributes;

LookupGLPlugin::LookupGLPlugin(nvinfer1::DataType type, int rank)
    : mType(type)
    , mRank(rank)
{
    mArch = tensorrt_llm::common::getSMVersion();
}

// Parameterized constructor
LookupGLPlugin::LookupGLPlugin(void const* data, size_t length)
{
    mArch = tensorrt_llm::common::getSMVersion();
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mRank);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LookupGLPlugin::clone() const noexcept
{
    auto* plugin = new LookupGLPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs LookupGLPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 3 || nbInputs == 4);//input_id, weight(embTbl),gamma(RMS) or additionaly per_token_scale
        TLLM_CHECK(outputIndex == 0 || outputIndex == 1);//output includes residual(batchSz,seqLen,hidSz),hidAfterRMS
        DimsExprs ret;
        int const nbDimsInput = inputs[0].nbDims;
        int const nbDimsWeight = inputs[1].nbDims;
        ret.nbDims = nbDimsInput + 1;

        for (int i = 0; i < nbDimsInput; ++i)
        {
            ret.d[i] = inputs[0].d[i];
        }
        ret.d[nbDimsInput] = inputs[1].d[nbDimsWeight - 1];

        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LookupGLPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    bool res = false;
    if (nbInputs == 3)
    {
        switch (pos)
        {
        case 0: res = ((inOut[0].type == DataType::kINT32) && (inOut[0].format == TensorFormat::kLINEAR)); break;//input_ids
        case 1: res = ((inOut[1].type == mType) && (inOut[1].format == TensorFormat::kLINEAR)); break;//weight(embTbl)
        case 2: res = ((inOut[2].type == mType) && (inOut[2].format == TensorFormat::kLINEAR)); break;//gama(RMS)
        case 3: res = ((inOut[3].type == mType) && (inOut[3].format == TensorFormat::kLINEAR)); break;//output[0](residual) 
        case 4: res = ((inOut[4].type == mType) && (inOut[4].format == TensorFormat::kLINEAR)); break;//output[1](hidAfterRMS)
        default: // should NOT be here!
            res = false;
        }
    }
    // else
    // {
    //     TLLM_CHECK_WITH_INFO(mArch == 90, "int8 weight only lookupPlugin is only supported in SM 90 now.");
    //     switch (pos)
    //     {
    //     case 0: res = ((inOut[0].type == DataType::kINT32) && (inOut[0].format == TensorFormat::kLINEAR)); break;
    //     case 1:
    //         res = ((inOut[1].type == DataType::kINT8 || inOut[1].type == mType)
    //             && (inOut[1].format == TensorFormat::kLINEAR));
    //         break;
    //     case 2: res = ((inOut[2].type == mType) && (inOut[2].format == TensorFormat::kLINEAR)); break;
    //     case 3: res = ((inOut[3].type == mType) && (inOut[3].format == TensorFormat::kLINEAR)); break;
    //     default: // should NOT be here!
    //         res = false;
    //     }
    // }
    return res;
}

void LookupGLPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    mNbInputs = nbInputs;
}

size_t LookupGLPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int LookupGLPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     input  [tokenNum]
    //     weight [localVocabSize, hidden]
    //     per_token_scales [localVocabSize], optional
    // outputs
    //     embedding [tokenNum, hidden]

    int64_t tokenNum = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        tokenNum *= inputDesc[0].dims.d[i];
    }

    int const localVocabSize = inputDesc[1].dims.d[0];
    int const hidden = inputDesc[1].dims.d[inputDesc[1].dims.nbDims - 1];
    int const* input = reinterpret_cast<int const*>(inputs[0]);

    int offset = mRank * localVocabSize;

    if (mNbInputs == 3) {
        if (mType == DataType::kBF16) {
            __nv_bfloat16 const* weight = reinterpret_cast<__nv_bfloat16 const*>(inputs[1]);
            __nv_bfloat16 const* gamma = reinterpret_cast<__nv_bfloat16 const*>(inputs[2]);
            __nv_bfloat16* residual = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
            __nv_bfloat16* aftRMS = reinterpret_cast<__nv_bfloat16*>(outputs[1]);
            invokeLookUpRms<__nv_bfloat16, __nv_bfloat16, int>(residual,
                aftRMS, input, weight, tokenNum, offset, localVocabSize, hidden, gamma, stream);
        }
        sync_check_cuda_error();
    } else {
        TLLM_THROW("Unsupported data type");
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType LookupGLPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return mType;
}

// IPluginV2 Methods

char const* LookupGLPlugin::getPluginType() const noexcept
{
    return LOOKUP_PLUGIN_NAME;
}

char const* LookupGLPlugin::getPluginVersion() const noexcept
{
    return LOOKUP_PLUGIN_VERSION;
}

int LookupGLPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int LookupGLPlugin::initialize() noexcept
{
    return 0;
}

void LookupGLPlugin::destroy() noexcept
{
    delete this;
}

size_t LookupGLPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType) + sizeof(mRank);
}

void LookupGLPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mRank);

    assert(d == a + getSerializationSize());
}

void LookupGLPlugin::terminate() noexcept {}

///////////////

LookupGLPluginCreator::LookupGLPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rank", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LookupGLPluginCreator::getPluginName() const noexcept
{
    return LOOKUP_PLUGIN_NAME;
}

char const* LookupGLPluginCreator::getPluginVersion() const noexcept
{
    return LOOKUP_PLUGIN_VERSION;
}

PluginFieldCollection const* LookupGLPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LookupGLPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    int rank;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            rank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new LookupGLPlugin(type, rank);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LookupGLPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call LookupGLPlugin::destroy()
    try
    {
        auto* obj = new LookupGLPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
