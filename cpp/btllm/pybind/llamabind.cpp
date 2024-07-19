#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>


#include "tensorrt_llm/common/cudaUtils.h"
#include "btllm/mdl/llama/llama.h"

namespace py = pybind11;

template <typename T>
using OptVec = std::optional<std::vector<T>>;
using btllm::mdl::Llama;



std::shared_ptr<Llama> createLlama(const std::string& jsn, size_t max_batch_tokens,
          const torch::Tensor &weights
  ) {
  Llama* obj = new Llama(jsn, max_batch_tokens);
  // tensorrt_llm::common::check_cuda_error(cudaGetLastError());
  obj->setWAndGrd(weights.data_ptr(), nullptr);
  // tensorrt_llm::common::check_cuda_error(cudaGetLastError());
  return std::shared_ptr<Llama>(obj);
}

torch::Tensor runLlama(std::shared_ptr<Llama> lma, const torch::Tensor &inpIdTsr, int mxOutputLen) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(torch::kStrided)
                     .device(inpIdTsr.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int batchSz = inpIdTsr.size(0);
  int seqLen = inpIdTsr.size(1);
  int tokenNum = batchSz * seqLen;
  int hidSz = lma->mArg.hidSz;
  lma->setStream(stream);
  lma->initRunParam(batchSz, seqLen, mxOutputLen);
  torch::Tensor output = torch::empty({mxOutputLen}, options).contiguous();
  lma->Forward(reinterpret_cast<int*>(inpIdTsr.data_ptr()), reinterpret_cast<int*>(output.data_ptr()));
  // tensorrt_llm::common::check_cuda_error(cudaGetLastError());
  
  auto embOutOpt = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .layout(torch::kStrided)
                     .device(inpIdTsr.device());
  torch::Tensor embOutTsr = torch::empty({batchSz, seqLen, hidSz}, embOutOpt).contiguous();
  tensorrt_llm::common::check_cuda_error(cudaMemcpy(embOutTsr.data_ptr(), lma->_buf,
                              size_t(batchSz*seqLen*hidSz * 2)/* sizeof(bf16) */, cudaMemcpyDeviceToDevice/*cudaMemcpyDefault*/));
  
  return embOutTsr;
}


PYBIND11_MODULE(BTLLM_PYBIND_MODULE, m)
{
  //@# 在c++和py之间传递对象指针,参考 https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
  py::class_<Llama, std::shared_ptr<Llama>>(m, "Llama")
        // .def(py::init<nvinfer1::DataType, int, int, void*>())
        // .def("setName", &Pet::setName)
        ;
  m.def("createLlama",
        &createLlama,
        "BT createLlama");
  m.def("runLlama",
        &runLlama,
        "BT runLlama");
}