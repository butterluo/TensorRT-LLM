#include "llama.h"


namespace btllm::mdl {

LlamaA16W4::LlamaA16W4(BTArg arg):
    mArg(arg),
    _lookupPlugin(mArg.hidSz, mArg.hidSz, toTrtDataType<__nv_bfloat16>())
{
  initBuf();
}

void LlamaA16W4::setStream(cudaStream_t stream) {
  _stream = stream;
}

void LlamaA16W4::initBuf() {
  size_t embTblSz = (mArg.vocabSz * mArg.hidSz) * sizeof(__nv_bfloat16);
  size_t ttlBytSz = embTblSz;
  tensorrt_llm::common::check_cuda_error(cudaMalloc((void **)&_buf, ttlBytSz));
}

void LlamaA16W4::setWAndGrd(void* weightPtr, void* grdPtr) {
  _emb_w_ptr = reinterpret_cast<__nv_bfloat16*>(weightPtr);
  //TODO
  void* nxt_ptr = _emb_w_ptr + (mArg.vocabSz * mArg.hidSz);
}

void LlamaA16W4::initRunParam(int batchSz, int seqLen, int mxOutputLen, bool initAll) {
  if(initAll) {
    _param.lookupParam.weight = _emb_w_ptr;
  }
  _param.lookupParam.tokenNum = batchSz * seqLen;
  _param.mxOutputLen = mxOutputLen;
}

void LlamaA16W4::Forward(const int *input_ptr, int *out_ptr) {
  _param.lookupParam.input_ids = const_cast<int*>(input_ptr);
  _param.lookupParam.outputs = _buf;
  _lookupPlugin.enqueue(_param.lookupParam, _stream);
  void* nxtMidPtr = _buf + _param.lookupParam.tokenNum * mArg.hidSz * (_lookupPlugin.dataTypeBitSz / 8);
}

}