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

#include "tensorrt_llm/common/workspace.h"
// #include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "btllm/plugins/common/checkMacrosPlugin.h"

#include <NvInferRuntime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <cstring>
#include <map>
#include <memory>
#include <nvml.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace btllm::plugins
{

inline size_t typeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBOOL: return 1UL;
    case nvinfer1::DataType::kFP8: return 1UL;
    case nvinfer1::DataType::kHALF: return 2UL;
    case nvinfer1::DataType::kBF16: return 2UL;
    case nvinfer1::DataType::kFLOAT: return 4UL;
    case nvinfer1::DataType::kINT8: return 1UL;
    case nvinfer1::DataType::kUINT8: return 1UL;
    case nvinfer1::DataType::kINT32: return 4UL;
    case nvinfer1::DataType::kINT64: return 8UL;
    }

    TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    return 0;
}

inline cudaDataType_t trtToCublasDtype(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return CUDA_R_32F;
    case nvinfer1::DataType::kHALF: return CUDA_R_16F;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kBF16: return CUDA_R_16BF;
#endif
    default: TLLM_THROW("Not supported data type for cuBLAS");
    }
}


}

//! To save GPU memory, all the plugins share the same cublas and cublasLt handle globally.
//! Get cublas and cublasLt handle for current cuda context
std::shared_ptr<cublasHandle_t> getCublasHandle();
std::shared_ptr<cublasLtHandle_t> getCublasLtHandle();
