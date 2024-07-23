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

torch::Tensor getLlamaW(std::shared_ptr<Llama> lma) {
    auto outOpt = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .layout(torch::kStrided)
                     .device("cuda:0");
    // //pre rms norm weight
    // torch::Tensor tsr = torch::empty({lma->mArg.hidSz}, outOpt).contiguous();
    // tensorrt_llm::common::check_cuda_error(cudaMemcpy(tsr.data_ptr(), (void*)lma->_prerms_w_ptr, 
    //       size_t(lma->mArg.hidSz * sizeof(__nv_bfloat16)), cudaMemcpyDeviceToDevice));
    // return tsr;

    //qkv 0 weight
    torch::Tensor tsr = torch::empty({lma->mArg.qkvSz, lma->mArg.hidSz}, outOpt).contiguous();
    tensorrt_llm::common::check_cuda_error(cudaMemcpy(tsr.data_ptr(), (void*)lma->_qkvPrj_w_ptr, 
          size_t(lma->mArg.qkvSz*lma->mArg.hidSz * sizeof(__nv_bfloat16)), cudaMemcpyDeviceToDevice));
    return tsr;
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
  size_t byteSzOfT = sizeof(__nv_bfloat16);
  auto outOpt = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .layout(torch::kStrided)
                     .device(inpIdTsr.device());

  //AFter emb and rms
  // torch::Tensor embOutTsr = torch::empty({batchSz, seqLen, hidSz}, outOpt).contiguous();
  // tensorrt_llm::common::check_cuda_error(cudaMemcpy(embOutTsr.data_ptr(), lma->_buf,
  //                             size_t(batchSz*seqLen*hidSz * byteSzOfT)/* sizeof(bf16) */, cudaMemcpyDeviceToDevice/*cudaMemcpyDefault*/));
  // return embOutTsr;

  //AFter qkvPrj
  torch::Tensor tsr = torch::empty({batchSz,seqLen,lma->mArg.qkvSz}, outOpt).contiguous();
  __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(lma->_buf);
  ptr = ptr + (batchSz*seqLen*hidSz);
  tensorrt_llm::common::check_cuda_error(cudaMemcpy(tsr.data_ptr(), ptr,
                              size_t(batchSz*seqLen*lma->mArg.qkvSz * byteSzOfT), cudaMemcpyDeviceToDevice/*cudaMemcpyDefault*/));
  return tsr;
}


PYBIND11_MODULE(BTLLM_PYBIND_MODULE, m)
{
  //@# 在c++和py之间传递对象指针,参考 https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
  py::class_<Llama, std::shared_ptr<Llama>>(m, "Llama")
        // .def(py::init<nvinfer1::DataType, int, int, void*>())
        // .def("setName", &Pet::setName)
        ;
  m.def("createLlama", &createLlama, "BT createLlama");
  m.def("runLlama", &runLlama, "BT runLlama");
  m.def("getLlamaW", &getLlamaW, "BT getLlamaW");
}