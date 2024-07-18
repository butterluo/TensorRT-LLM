#pragma once

#include <type_traits>
#include <cuda_bf16.h>
#include <NvInferRuntime.h>
#include <nlohmann/json.hpp>

#include "tensorrt_llm/common/cudaUtils.h"

using Json = typename nlohmann::json::basic_json;

namespace btllm {

#define EPS 1e-5

struct BTBaseArg {};

struct BTBaseParam {};

template<typename T>
nvinfer1::DataType toTrtDataType() ;

int trtDataTypeToBitSz(nvinfer1::DataType t) ;

template <typename FieldType>
FieldType parseJsonFieldOr(Json const& json, std::string_view name, FieldType defaultValue)
{
    auto value = defaultValue;
    try
    {
        value = json.at(name).template get<FieldType>();
    }
    catch (nlohmann::json::out_of_range& e)
    {
        TLLM_LOG_WARNING("Parameter %s cannot be read from json:", std::string(name).c_str());
        TLLM_LOG_WARNING(e.what());
    }
    return value;
}

}