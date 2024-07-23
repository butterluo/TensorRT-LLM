#include "llama.h"


namespace btllm::mdl {

Llama::Llama(const std::string& jsn, size_t max_batch_tokens):
    _jsn(nlohmann::json::parse(jsn, nullptr, true/*allow_exceptions*/, true/*ignore_comments*/)),
    _lookupPlugin(parseJsonFieldOr(_jsn,"vocab_size",0), parseJsonFieldOr(_jsn,"hidden_size",0), toTrtDataType<__nv_bfloat16>())
{
  mArg.hidSz = _lookupPlugin.hidden;
  mArg.vocabSz = _lookupPlugin.localVocabSize;
  mArg._max_batch_tokens = max_batch_tokens;


  GemmPluginCreator gemmCreator;
  size_t min_batch_tokens = 1;
  mArg.qHead = parseJsonFieldOr(_jsn,"num_attention_heads",0);
  mArg.kvHead = parseJsonFieldOr(_jsn,"num_key_value_heads",0);
  mArg.headSz = mArg.hidSz / mArg.qHead;
  mArg.qkvSz = 2*(mArg.headSz * mArg.kvHead) + mArg.hidSz;
  _qkvPrjPtr = gemmCreator.createPlugin(min_batch_tokens, max_batch_tokens, mArg.qkvSz, mArg.hidSz, false/*transA*/, true/*transB*/, 0, 0, toTrtDataType<__nv_bfloat16>());

  initBuf();
}

void Llama::setStream(cudaStream_t stream) {
  _stream = stream;
}

void Llama::initBuf() {
  size_t embTblSz = size_t(mArg._max_batch_tokens * mArg.hidSz)* size_t(_lookupPlugin.dataTypeBitSz / 8);
  size_t qkvOutSz = mArg._max_batch_tokens * mArg.qkvSz * sizeof(__nv_bfloat16);
  size_t ttlBytSz = (embTblSz + qkvOutSz);
  tensorrt_llm::common::check_cuda_error(cudaMalloc((void **)&_buf, ttlBytSz));

  size_t qkvWS = _qkvPrjPtr->getWorkspaceSize();
  size_t wsBytSz = qkvWS;
  tensorrt_llm::common::check_cuda_error(cudaMalloc((void **)&_ws, wsBytSz));
}

void Llama::setWAndGrd(void* weightPtr, void* grdPtr) {
  //emb
  _emb_w_ptr = reinterpret_cast<__nv_bfloat16*>(weightPtr);
  //pre rms norm
  void* nxt_ptr = _emb_w_ptr + (mArg.vocabSz * mArg.hidSz);
  _prerms_w_ptr = reinterpret_cast<__nv_bfloat16*>(nxt_ptr);
  //qkv 0
  nxt_ptr = _prerms_w_ptr + mArg.hidSz;
  _qkvPrj_w_ptr = reinterpret_cast<__nv_bfloat16*>(nxt_ptr);
  nxt_ptr = _qkvPrj_w_ptr + mArg.hidSz * mArg.qkvSz;
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
  __nv_bfloat16* nxtMidPtr = reinterpret_cast<__nv_bfloat16*>(_param.lookupParam.outputs);
  nxtMidPtr = nxtMidPtr + _param.lookupParam.tokenNum * mArg.hidSz;
  _qkvPrjPtr->enqueue(_param.lookupParam.tokenNum/*rowA*/, mArg.hidSz/*colA*/, mArg.qkvSz/*rowB*/, mArg.hidSz/*colB*/,
             _param.lookupParam.outputs/*A*/, _qkvPrj_w_ptr/*B*/, nxtMidPtr/*C*/, _ws, _stream);
}

}