/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

#include <btllm/plugins/weightOnlyGroupwiseQuantMatmulPlugin/btweightOnlyGroupwiseQuantMatmulPlugin.h>

namespace py = pybind11;

template <typename T>
using OptVec = std::optional<std::vector<T>>;

using btllm::plugins::BTWeightOnlyGroupwiseQuantMatmulPlugin;
using btllm::plugins::BTWeightOnlyGroupwiseQuantMatmulPluginCreator;

#if not defined(BTLLM_PYBIND_MODULE)
#error "BTLLM_PYBIND_MODULE must be defined"
#endif

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;

    ~Pet() {
      printf("~~~~~~~~~~~~~LLLLL~~~~~ \n");
    }
};

typedef struct BTWeightOnlyGroupwiseQuantMatmulPlugin* BTWeightOnlyGroupwiseQuantMatmulPlugin_t;

std::shared_ptr<BTWeightOnlyGroupwiseQuantMatmulPlugin> createBTWeightOnlyGroupwiseQuantMatmulPlugin(
    // int layer_id, const torch::Tensor &input
    int dataTypeInt /* nvinfer1::DataType type */, int quant_algo, int group_size,
  const int maxM, const int maxN, const int maxK) {
  // auto res = torch::zeros_like(input);
  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // cudaDeviceProp* deviceProp = at::cuda::getCurrentDeviceProperties();
  // printf("\nDevice %d: \"%s\"\n", 0, deviceProp->name);

  // printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
  //          deviceProp->major, deviceProp->minor);
  // return {res};
  nvinfer1::DataType type = nvinfer1::DataType(dataTypeInt);
  BTWeightOnlyGroupwiseQuantMatmulPluginCreator* creator = new BTWeightOnlyGroupwiseQuantMatmulPluginCreator();
  BTWeightOnlyGroupwiseQuantMatmulPlugin* ret = creator->createPlugin(nullptr, type, quant_algo, group_size, maxM, maxN, maxK);
  return std::shared_ptr<BTWeightOnlyGroupwiseQuantMatmulPlugin>(ret);
}

torch::Tensor gemmWeightOnlyGrpwisBf16W8(std::shared_ptr<BTWeightOnlyGroupwiseQuantMatmulPlugin> int8plugin
          , int const m, int const real_n, int const k
          , const torch::Tensor &zeros_tsr, const torch::Tensor &weight_scales_tsr
          , const torch::Tensor &act_tsr, const torch::Tensor &weight_tsr, const torch::Tensor &biases_tsr 
          // , void const* act_ptr, void const* weight_ptr, void const* biases_ptr
          // , void* output_ptr
        ) {
  uint8_t* ptr = 0;
  size_t wrkSz = int8plugin->getWorkspaceSize();
  cudaError_t cuda_error = cudaMalloc((void**)&ptr, wrkSz);
  if (cuda_error != cudaSuccess) {
    TLLM_THROW("Failed to allocate memory with CUDA_ERR %s \n", cudaGetErrorString(cuda_error));
  }
  const void* zeros_ptr = nullptr;
  if(zeros_tsr.numel() > 0) {
    zeros_ptr = zeros_tsr.data_ptr();
  }
  const void* weight_scales_ptr = nullptr;
  if(weight_scales_tsr.numel() > 0) {
    weight_scales_ptr = weight_scales_tsr.data_ptr();
  }
  const void* act_ptr = nullptr;
  if(act_tsr.numel() > 0) {
    act_ptr = act_tsr.data_ptr();
  }
  const void* weight_ptr = nullptr;
  if(weight_tsr.numel() > 0) {
    weight_ptr = weight_tsr.data_ptr();
  }
  const void* biases_ptr = nullptr;
  if(biases_tsr.numel() > 0) {
    biases_ptr = biases_tsr.data_ptr();
  }
  auto options = torch::TensorOptions()
                     .dtype(act_tsr.scalar_type())
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, act_tsr.device().index());//.requires_grad(true);
  // torch::Tensor output = torch::empty({m, real_n}, options).contiguous();
  torch::Tensor output = torch::ones({m, real_n}, options).contiguous();
  // void *out_ptr = output.data_ptr();
  return output;
}

PYBIND11_MODULE(BTLLM_PYBIND_MODULE, m)
{
  //@# 在c++和py之间传递对象指针,参考 https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
  py::class_<BTWeightOnlyGroupwiseQuantMatmulPlugin, std::shared_ptr<BTWeightOnlyGroupwiseQuantMatmulPlugin>>(m, "BTWeightOnlyGroupwiseQuantMatmulPlugin")
        // .def(py::init<nvinfer1::DataType, int, int, void*>())
        // .def("setName", &Pet::setName)
        ;
  m.def("createBTWeightOnlyGroupwiseQuantMatmulPlugin",
        &createBTWeightOnlyGroupwiseQuantMatmulPlugin,
        "BT createBTWeightOnlyGroupwiseQuantMatmulPlugin");
  m.def("gemmWeightOnlyGrpwisBf16W8",
        &gemmWeightOnlyGrpwisBf16W8,
        "BT gemmWeightOnlyGrpwisBf16W8");
}
