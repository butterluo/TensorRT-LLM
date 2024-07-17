#pragma once

#include "btllm/btcommon/bt.h"
#include "btllm/plugins/lookupPlugin/lookupPlugin.h"

using btllm::plugins::LookupPlugin;

namespace btllm {
namespace mdl {

class Llama {

public:

struct BTArg: BTBaseArg { //for init
  int vocabSz;
  int hidSz;
  size_t _max_batch_tokens;
};

struct BTParam: BTBaseParam {//for run
  LookupPlugin::BTParam lookupParam;
  int batchSz;
  int seqLen;
  int mxOutputLen;
};

virtual ~Llama() = default;

};

class LlamaA16W4 : public Llama{

public:

BTArg mArg;

LlamaA16W4() = delete;

LlamaA16W4(BTArg arg);

~LlamaA16W4() = default;

void setStream(cudaStream_t stream);

void setWAndGrd(void* weights_ptr, void* grads_ptr);

void initBuf();

void initRunParam(int batchSz, int seqLen, int mxOutputLen, bool initAll=true);

void Forward(const int *input_ptr, int *out_ptr);


char* _buf = nullptr;

private:


BTParam _param;
cudaStream_t _stream = NULL;
LookupPlugin _lookupPlugin;

__nv_bfloat16* _emb_w_ptr = nullptr;
};

}
}