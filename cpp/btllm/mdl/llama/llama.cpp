#include "llama.h"


namespace btllm::mdl {

Llama::Llama(const std::string& jsn, size_t max_batch_tokens):
    _jsn(nlohmann::json::parse(jsn, nullptr, true/*allow_exceptions*/, true/*ignore_comments*/)),
    _lookupPlugin(parseJsonFieldOr(_jsn,"vocab_size",0), parseJsonFieldOr(_jsn,"hidden_size",0), toTrtDataType<__nv_bfloat16>())
{
  mArg.hidSz = _lookupPlugin.hidden;
  mArg.vocabSz = _lookupPlugin.localVocabSize;
  mArg._max_batch_tokens = max_batch_tokens;
  initBuf();
}

void Llama::setStream(cudaStream_t stream) {
  _stream = stream;
}

void Llama::initBuf() {
  size_t embTblSz = size_t(mArg._max_batch_tokens * mArg.hidSz) * size_t(_lookupPlugin.dataTypeBitSz / 8);
  size_t ttlBytSz = embTblSz;
  tensorrt_llm::common::check_cuda_error(cudaMalloc((void **)&_buf, ttlBytSz));
}

void Llama::setWAndGrd(void* weightPtr, void* grdPtr) {
  _emb_w_ptr = reinterpret_cast<__nv_bfloat16*>(weightPtr);
  //TODO
  void* nxt_ptr = _emb_w_ptr + (mArg.vocabSz * mArg.hidSz);
  _prerms_w_ptr = reinterpret_cast<__nv_bfloat16*>(nxt_ptr);
  nxt_ptr = _prerms_w_ptr + mArg.hidSz;
}

void Llama::initRunParam(int batchSz, int seqLen, int mxOutputLen, bool initAll) {
  _param.lookupParam.tokenNum = batchSz * seqLen;
  assert(_param.lookupParam.tokenNum <= mArg._max_batch_tokens);
  if(initAll) {
    _param.lookupParam.weight = _emb_w_ptr;
    _param.lookupParam.gamma = _prerms_w_ptr;
  }
  
  _param.mxOutputLen = mxOutputLen;
}

void Llama::Forward(const int *input_ptr, int *out_ptr) {
  _param.lookupParam.input_ids = const_cast<int*>(input_ptr);
  _param.lookupParam.outputs = _buf;
  _lookupPlugin.enqueue(_param.lookupParam, _stream);
  void* nxtMidPtr = _buf + _param.lookupParam.tokenNum * mArg.hidSz * size_t(_lookupPlugin.dataTypeBitSz / 8);
}

}