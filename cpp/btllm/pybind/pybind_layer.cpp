#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "cudas/context.h"
#include "ops/norm_head.h"
// #include "quant_linear_layer.h"
// #include "transformer_decoder_layer.h"
#include "ops/llama_embedding_layer.h"
// #include "transformer_encoder_layer.h"
#include "mdl/llama.h"

using namespace torch::indexing;

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

static std::unordered_map<int, std::shared_ptr<void>>
    s_transformer_encoder_layers;
// static std::unordered_map<int, std::shared_ptr<void>> s_cross_entropy_layers;
// static std::unordered_map<int, std::shared_ptr<void>> s_quant_linear_layers;
namespace bt {
  using namespace mdl;
namespace pybind {
template <typename T>
int create_transformer_encoder_layer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_dim,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_dropout_ratio, /* bool is_pre_ln, @#lma is pre norm */
    std::string activation_fn, bool mask_future_tokens, int num_headsK) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaDeviceProp* dprops = at::cuda::getCurrentDeviceProperties();
  Context::Instance().set_stream(stream);
  auto layer = std::make_shared<LlamaAttnLayer<T>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, activation_fn, mask_future_tokens, dprops, num_headsK);

  s_transformer_encoder_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> transformer_encoder_layer_fw(
    int layer_id, const torch::Tensor &input, /* const torch::Tensor &input_mask, */
    bool training_mode/*, bool prelayernorm , bool quant_mode */) {
  CHECK_INPUT(input);
  // CHECK_INPUT(input_mask);

  const T *input_ptr = (const T *)input.data_ptr();
  // const T *input_mask_ptr = (const T *)input_mask.data_ptr();

  std::shared_ptr<LlamaAttnLayer<T>> layer =
      std::static_pointer_cast<LlamaAttnLayer<T>>(
          s_transformer_encoder_layers[layer_id]);

  auto output = torch::empty_like(input);// origin
  //FORTEST START
  // auto dtype =
  //     (std::is_same<T, __half>::value) ? torch::kFloat16 : torch::kFloat32;
  // //output of qkv proj
  // auto options = torch::TensorOptions()
  //                    .dtype(dtype)
  //                    .layout(torch::kStrided)
  //                    .device(torch::kCUDA, input.device().index());
  // auto output = torch::empty({int(input.size(0)/*batchSz*/ * input.size(1)/*seqLen*/ * int(layer->_heads + 2*layer->_heads_k) * layer->_head_size)}, options);//FORTEST forward RoPE output
  // auto output = torch::empty({input.size(0), layer->_heads, input.size(1), input.size(1)}, options);//FORTEST forward sftMx(Q*K) output
  // auto output = torch::empty({input.size(0), input.size(1), input.size(2)}, options);//FORTEST forward before o_proj. Output of ffnLayer's rmsNorm
  // auto output = torch::empty({2 * input.size(0), input.size(1), layer->_intermediate_size}, options);//FORTEST forward . Output of ffnLayer's cat[gate_proj_out,up_proj_out]
  // auto output = torch::empty({input.size(0), input.size(1), layer->_intermediate_size}, options);//FORTEST forward . Output of ffnLayer's silu
  //FORTEST END
  T *out_ptr = (T *)output.data_ptr();

  
  layer->set_cur_batch_shape(input.size(0)/*batchSz*/, input.size(1)/*seqLen*/);
  // layer->SetTrainingMode(training_mode);//lma has no dropout
  // layer->SetQuantMode(quant_mode);
  layer->Forward(input_ptr, out_ptr);
  return {output};
}

template <typename T>
std::vector<torch::Tensor> transformer_encoder_layer_bw(
    int layer_id, const torch::Tensor &g_output) {
  // auto g_output = grad_dec_output.contiguous();//@# 上一层做完backward后传过来的grad of this layer's output(本层在forward时output给到上一层的上一层的input的grad)
  CHECK_INPUT(g_output);
  // CHECK_INPUT(output);           //@# 本层在forward时的output
  // CHECK_INPUT(input);            //本层在forward时的input
  // CHECK_INPUT(input_mask);       //本层在forward时的input mask

  std::shared_ptr<LlamaAttnLayer<T>> layer =
      std::static_pointer_cast<LlamaAttnLayer<T>>(
          s_transformer_encoder_layers[layer_id]);
  auto grad_input = torch::empty_like(g_output); //@# 求出来的是在forward时输入到本层的input的grad，有给到下一层做backward时的grad of output

  // inputs.
  const T *grad_dec_output_ptr = (const T *)g_output.data_ptr();
  // const T *input_ptr = (const T *)input.data_ptr();
  // const T *output_ptr = (const T *)output.data_ptr();
  // const T *input_mask_ptr = (const T *)input_mask.data_ptr();
  // outputs.
  T *grad_input_ptr = (T *)grad_input.data_ptr();

  
  layer->set_cur_batch_shape(g_output.size(0), g_output.size(1));
  layer->Backward(grad_dec_output_ptr, /* input_ptr, output_ptr, input_mask_ptr, */
                  grad_input_ptr);
  // CUDA_SYNC_CHK_ERR();
  return {grad_input};
}


static std::unordered_map<int, std::shared_ptr<void>>
    s_llama_embedding_layers;

template <typename T>
int create_llama_embedding_layer(int layer_id,
                                       int max_batch_tokens, int embedding_dim,
                                       int vocab_size, int max_seq_len,
                                       int padding_idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);

  auto layer = std::make_shared<LlamaEmbeddingLayer<T>>(
      layer_id, max_batch_tokens, embedding_dim, vocab_size,
      max_seq_len, padding_idx);

  s_llama_embedding_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "Embedding layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> llama_embedding_layer_fw(
    int layer_id, const torch::Tensor &input) {
  CHECK_INPUT(input);
  const int *input_ptr = (const int *)input.data_ptr();

  std::shared_ptr<LlamaEmbeddingLayer<T>> layer =
      std::static_pointer_cast<LlamaEmbeddingLayer<T>>(
          s_llama_embedding_layers[layer_id]);

  auto dtype =
      (std::is_same<T, __half>::value) ? torch::kFloat16 : torch::kFloat32;

  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA)
                     .requires_grad(true);
  auto output = torch::empty(
      {input.size(0), input.size(1), layer->_embedding_dim}, options);
  T *out_ptr = (T *)output.data_ptr();

  layer->set_cur_batch_shape(input.size(0), input.size(1));
  layer->Forward(input_ptr, out_ptr);

  return {output};
}

template <typename T>
void llama_embedding_layer_bw(int layer_id,
                                    const torch::Tensor &grad_output,
                                    const torch::Tensor &input) {
  auto g_output = grad_output.contiguous();
  CHECK_INPUT(g_output);
  CHECK_INPUT(input);

  const T *grad_output_ptr = (const T *)g_output.data_ptr();
  const int *input_ptr = (const int *)input.data_ptr();

  std::shared_ptr<LlamaEmbeddingLayer<T>> layer =
      std::static_pointer_cast<LlamaEmbeddingLayer<T>>(
          s_llama_embedding_layers[layer_id]);

  layer->set_cur_batch_shape(g_output.size(0), g_output.size(1));
  layer->Backward(grad_output_ptr, input_ptr);
  // CUDA_SYNC_CHK_ERR();
  return;
}

static std::unordered_map<int, std::shared_ptr<void>>
    s_llama_normhead_layers;

template <typename T>
int create_llama_normhead_layer(int layer_id,
                                       int max_batch_tokens, int hidden_size,
                                       int vocab_size, int max_seq_len) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);

  auto layer = std::make_shared<NormHeadLayer<T>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, vocab_size);

  s_llama_normhead_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "NormHead layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> llama_normhead_layer_fw(
    int layer_id, const torch::Tensor &input) {
  CHECK_INPUT(input);
  const T *input_ptr = (const T *)input.data_ptr();

  std::shared_ptr<NormHeadLayer<T>> layer =
      std::static_pointer_cast<NormHeadLayer<T>>(
          s_llama_normhead_layers[layer_id]);

  // auto dtype =
  //     (std::is_same<T, __half>::value) ? torch::kFloat16 : torch::kFloat32;

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, input.device().index());
                     //.requires_grad(true);
  auto output = torch::empty(
      {input.size(0), input.size(1), layer->_vocab_size}, options).contiguous();
  T *out_ptr = (T *)output.data_ptr();

  layer->set_cur_batch_shape(input.size(0), input.size(1));
  layer->Forward(input_ptr, out_ptr, Context::Instance().get_stream());

  return {output};
}

template <typename T>
void  llama_normhead_layer_bw(int layer_id,
                                    const torch::Tensor &grad_output, torch::Tensor &grad_input) {
  auto g_output = grad_output.contiguous();
  CHECK_INPUT(g_output);
  // CHECK_INPUT(input);

  const T *grad_output_ptr = (const T *)g_output.data_ptr();
  // const int *input_ptr = (const int *)input.data_ptr();
  T *grad_input_ptr = (T *)grad_input.data_ptr();

  std::shared_ptr<NormHeadLayer<T>> layer =
      std::static_pointer_cast<NormHeadLayer<T>>(
          s_llama_normhead_layers[layer_id]);

  layer->set_cur_batch_shape(g_output.size(0), g_output.size(1));
  layer->Backward(grad_output_ptr, grad_input_ptr, at::cuda::getCurrentCUDAStream());
  // CUDA_SYNC_CHK_ERR();
  return; 
}

template <typename T>
void assign_layer_weight_grad(const torch::Tensor &weights,
                              torch::Tensor &grads, std::string layer_name,
                              int layer_id) {
  CHECK_INPUT(weights);
  const T *wptr = (const T *)weights.data_ptr();

  CHECK_INPUT(grads);
  T *gptr = (T *)grads.data_ptr();

  // if (layer_name == "TransformerDecoderLayer") {
  //   std::shared_ptr<TransformerDecoderLayer<T>> layer =
  //       std::static_pointer_cast<TransformerDecoderLayer<T>>(
  //           s_transformer_decoder_layers[layer_id]);
  //   layer->assign_weight_ptr(wptr);
  //   layer->assign_grad_ptr(gptr);
  // } else 
  if (layer_name == "TransformerEncoderLayer") {
    std::shared_ptr<LlamaAttnLayer<T>> layer =
        std::static_pointer_cast<LlamaAttnLayer<T>>(
            s_transformer_encoder_layers[layer_id]);
    layer->assign_weight_ptr(wptr);
    layer->assign_grad_ptr(gptr);
  } 
  else if (layer_name == "LlamaEmbeddingLayer") {
    std::shared_ptr<LlamaEmbeddingLayer<T>> layer =
        std::static_pointer_cast<LlamaEmbeddingLayer<T>>(
            s_llama_embedding_layers[layer_id]);
    layer->assign_tsr_ptr(wptr, gptr);
  } 
  else if (layer_name == "NormHeadLayer") {
    std::shared_ptr<NormHeadLayer<T>> layer =
        std::static_pointer_cast<NormHeadLayer<T>>(
            s_llama_normhead_layers[layer_id]);
    layer->assign_tsr_ptr(wptr, gptr);
  }
  else {
    throw std::runtime_error("NOT SUPPORT");
  }
  std::cout << layer_name << " #" << layer_id << " bind weights and grads."
            << std::endl;
  return;
}


}  // namespace pybind
}  // namespace bt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("transformer_encoder_layer_fw_fp32",
  //       &bt::pybind::transformer_encoder_layer_fw<float>,
  //       "BT Transformer Encoder forward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_fw_fp16",
        &bt::pybind::transformer_encoder_layer_fw<__half>,
        "BT Transformer Encoder forward with fp16 (CUDA)");
  // m.def("transformer_encoder_layer_bw_fp32",
  //       &bt::pybind::transformer_encoder_layer_bw<float>,
  //       "BT Transformer Encoder backward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_bw_fp16",
        &bt::pybind::transformer_encoder_layer_bw<__half>,
        "BT Transformer Encoder backward with fp16 (CUDA)");
  // m.def("create_transformer_encoder_layer_fp32",
  //       &bt::pybind::create_transformer_encoder_layer<float>,
  //       "Create BT Transformer Encoder Layer with fp32 (CUDA)");
  m.def("create_transformer_encoder_layer_fp16",
        &bt::pybind::create_transformer_encoder_layer<__half>,
        "Create BT Transformer Encoder Layer with fp16 (CUDA)");
  m.def("llama_embedding_layer_fw_fp16",
        &bt::pybind::llama_embedding_layer_fw<__half>,
        "BT Transformer Embedding forward with fp16 (CUDA)");
  // m.def("transformer_embedding_layer_bw_fp32",
  //       &bt::pybind::llama_embedding_layer_bw<float>,
  //       "BT Transformer Embedding backward with fp32 (CUDA)");
  m.def("llama_embedding_layer_bw_fp16",
        &bt::pybind::llama_embedding_layer_bw<__half>,
        "BT Transformer Embedding backward with fp16 (CUDA)");
  // m.def("create_transformer_embedding_layer_fp32",
  //       &bt::pybind::create_llama_embedding_layer<float>,
  //       "Create BT Transformer Embedding Layer with fp32 (CUDA)");
  m.def("create_llama_embedding_layer_fp16",
        &bt::pybind::create_llama_embedding_layer<__half>,
        "Create BT Transformer Embedding Layer with fp16 (CUDA)");
  m.def("create_llama_normhead_layer_fp16",
        &bt::pybind::create_llama_normhead_layer<__half>,
        "Create BT Transformer NormHead Layer with fp16 (CUDA)");
  m.def("llama_normhead_layer_fw_fp16",
        &bt::pybind::llama_normhead_layer_fw<__half>,
        "BT Transformer NormHead forward with fp16 (CUDA)");
  m.def("llama_normhead_layer_bw_fp16",
        &bt::pybind::llama_normhead_layer_bw<__half>,
        "BT Transformer NormHead forward with fp16 (CUDA)");
  // m.def("assign_layer_weight_grad_fp32",
  //       &bt::pybind::assign_layer_weight_grad<float>,
  //       "Bind layer weights and grads");
  m.def("assign_layer_weight_grad_fp16",
        &bt::pybind::assign_layer_weight_grad<__half>,
        "Bind layer weights and grads");
}
