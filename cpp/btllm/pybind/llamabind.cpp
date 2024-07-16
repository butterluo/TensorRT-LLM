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
using btllm::mdl::LlamaA16W4;

std::shared_ptr<LlamaA16W4> createLlama(int mxBtchTkn,
          int vocabSz,
          int hidSz,
          const torch::Tensor &weights
  ) {
  Llama::BTArg lmaArg;
  lmaArg._max_batch_tokens = mxBtchTkn;
  lmaArg.vocabSz = vocabSz;
  lmaArg.hidSz = hidSz;
  LlamaA16W4* obj = new LlamaA16W4(lmaArg);
  obj->setWAndGrd(weights.data_ptr(), nullptr);
  return std::shared_ptr<LlamaA16W4>(obj);
}

torch::Tensor runLlama(std::shared_ptr<LlamaA16W4> lma, const torch::Tensor &inpIdTsr, int seqLen, int mxOutputLen) {
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .layout(torch::kStrided)
                     .device(inpIdTsr.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  lma->setStream(stream);
  lma->initRunParam(seqLen, mxOutputLen);
  torch::Tensor output = torch::empty({mxOutputLen}, options).contiguous();
  lma->Forward(reinterpret_cast<int*>(inpIdTsr.data_ptr()), reinterpret_cast<int*>(output.data_ptr()));
  return output;
}


PYBIND11_MODULE(BTLLM_PYBIND_MODULE, m)
{
  //@# 在c++和py之间传递对象指针,参考 https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
  py::class_<LlamaA16W4, std::shared_ptr<LlamaA16W4>>(m, "LlamaA16W4")
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