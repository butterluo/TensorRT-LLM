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

#include "btllm/plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

#include "btllm/btcommon/bt.h"

namespace btllm::plugins
{

template <typename T, typename Idx>
void invokeLookUp(T* out, Idx const* input, T const* weight, const int64_t token_num, const Idx offset, const Idx size,
    Idx const n_embed, cudaStream_t stream = 0);

class LookupPlugin //: public BasePlugin
{
public:

struct BTParam: BTBaseParam {//for run
  void* weight = nullptr;
  int* input_ids = nullptr; 
  int64_t tokenNum = 0;
  void* outputs = nullptr;
};
public:
    LookupPlugin() = delete;
    LookupPlugin(const int localVocabSize, const int hidden, nvinfer1::DataType dataType);
    // LookupPlugin(nvinfer1::DataType type, int rank);

    ~LookupPlugin() = default;

    // // IPluginV2DynamicExt Methods
    // nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    // nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
    //     nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    // bool supportsFormatCombination(
    //     int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    // void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    //     nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    // size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    //     nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(BTParam param, cudaStream_t stream) noexcept;

    // // IPluginV2Ext Methods
    // nvinfer1::DataType getOutputDataType(
    //     int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;
    //
    // // IPluginV2 Methods
    // char const* getPluginType() const noexcept override;
    // char const* getPluginVersion() const noexcept override;
    // int getNbOutputs() const noexcept override;
    // int initialize() noexcept override;
    // void terminate() noexcept override;
    // size_t getSerializationSize() const noexcept override;
    // void serialize(void* buffer) const noexcept override;
    // void destroy() noexcept override;

  const int localVocabSize;
  const int hidden;
  const nvinfer1::DataType mType;
  const int dataTypeBitSz;
  
  int rank = 0;

private:
    // const std::string mLayerName;

};

// class LookupPluginCreator //: public BaseCreator
// {
// public:
//     LookupPluginCreator();
//
//     char const* getPluginName() const noexcept override;
//
//     char const* getPluginVersion() const noexcept override;
//
//     nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
//
//     nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
//
//     nvinfer1::IPluginV2* deserializePlugin(
//         char const* name, void const* serialData, size_t serialLength) noexcept override;
//
// private:
//     static nvinfer1::PluginFieldCollection mFC;
//     static std::vector<nvinfer1::PluginField> mPluginAttributes;
// };

// LookupPlugin* createLookupPlugin(char const* name, nvinfer1::DataType type) noexcept
// {
//     // PluginField const* fields = fc->fields;
//     // nvinfer1::DataType type;
//     // int rank;
//     // // Read configurations from each fields
//     // for (int i = 0; i < fc->nbFields; ++i)
//     // {
//     //     char const* attrName = fields[i].name;
//     //     if (!strcmp(attrName, "type_id"))
//     //     {
//     //         TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
//     //         type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
//     //     }
//     //     else if (!strcmp(attrName, "rank"))
//     //     {
//     //         TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
//     //         rank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
//     //     }
//     // }
//     try
//     {
//         auto* obj = new LookupPlugin(type, 0);
//         return obj;
//     }
//     catch (std::exception const& e)
//     {
//         caughtError(e);
//     }
//     return nullptr;
// }


} // namespace tensorrt_llm::plugins
