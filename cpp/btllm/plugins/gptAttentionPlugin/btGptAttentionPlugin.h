#pragma once

#include "btllm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "btllm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include <nlohmann/json.hpp>
#include "btllm/btcommon/bt.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <algorithm>
#include <functional>
#include <numeric>


namespace btllm::plugins
{

class BTGPTAttentionPluginCreator //: public GPTAttentionPluginCreatorCommon
{
public:
    BTGPTAttentionPluginCreator();

    // char const* getPluginName() const noexcept override;

    // char const* getPluginVersion() const noexcept override;

    // nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    tensorrt_llm::plugins::GPTAttentionPlugin* createPlugin(int layer_idx, nvinfer1::DataType actType, Json &jsnObj) noexcept;

    // nvinfer1::IPluginV2* deserializePlugin(
    //     char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace tensorrt_llm::plugins
