#pragma once

#include <nlohmann/json.hpp>

#include "btllm/btcommon/bt.h"
#include "btllm/plugins/lookupPlugin/lookupPlugin.h"
#include "btllm/plugins/gemmPlugin/gemmPlugin.h"

using btllm::plugins::LookupPlugin;
using btllm::plugins::GemmPluginCreator;
using btllm::plugins::GemmPlugin;

namespace btllm {
namespace mdl {


class Llama {

public:

struct BTArg: BTBaseArg { //for init
  int vocabSz;
  int hidSz;
  int headSz;
  int qkvSz;
  int qHead;
  int kvHead;
  size_t _max_batch_tokens;
};

struct BTParam: BTBaseParam {//for run
  LookupPlugin::BTParam lookupParam;
  int batchSz;
  int seqLen;
  int mxOutputLen;
};

public:

BTArg mArg;

Llama() = delete;

Llama(const std::string& jsn, size_t max_batch_tokens);

~Llama() = default;

void setStream(cudaStream_t stream);

void setWAndGrd(void* weights_ptr, void* grads_ptr);

void initBuf();

void initRunParam(int batchSz, int seqLen, int mxOutputLen, bool initAll=true);

void Forward(const int *input_ptr, int *out_ptr);


void* _buf = nullptr;
void* _buf2 = nullptr;

// private:

Json _jsn;
BTParam _param;
cudaStream_t _stream = NULL;
LookupPlugin _lookupPlugin;
btllm::plugins::GemmPlugin* _qkvPrjPtr;

__nv_bfloat16* _emb_w_ptr = nullptr;
__nv_bfloat16* _prerms_w_ptr = nullptr;
__nv_bfloat16* _qkvPrj_w_ptr = nullptr;

void* _ws = nullptr;
};

}
}