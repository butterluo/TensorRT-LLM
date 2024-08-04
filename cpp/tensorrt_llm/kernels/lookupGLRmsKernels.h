
#pragma once

// #include <assert.h>
#include <cuda_runtime.h>

#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tensor.h"

namespace tensorrt_llm
{
namespace kernels
{

#define EPS 1e-5



template <typename Tout, typename Tw, typename Idx>
void invokeLookUpRms(Tout* residual, Tout* out, Idx const* input, Tw const* weight, int64_t const token_num, Idx const offset,
    Idx const size, Idx const n_embed, Tw const* gamma, cudaStream_t stream = 0);

template <typename T>
void invokeRmsResid(T* output_resid, T* output, T const* input, T const* input_resid, int const n_embed/*hidSz*/, T const* gamma, int64_t const token_num, cudaStream_t stream);


namespace gl {

struct RmsFusionParams
{
    RmsFusionParams()
        : bias_buffer(nullptr)
        , residual_buffer(nullptr)
        , weight_buffer(nullptr)
        , intermediate_buffer(nullptr)
        , residual_out(nullptr)
    {
    }

    // gemm bias
    void const* bias_buffer;
    // residuial add
    void const* residual_buffer;
    // rms norm
    int hidden_size;           // equal to normalized_shape
    void const* weight_buffer; // norm elem-wise affine gamma
    void* residual_out;         //original input + residual
    // new residual
    void const* intermediate_buffer; //input hiddenstate
};

struct RmsParams
{
    size_t elts_total;
    void* local_output_buffer_ptr;

    RmsFusionParams fusion_params;

};

void residualRmsNorm(RmsParams& params, nvinfer1::DataType dataType, cudaStream_t stream);

}// namespace gl


} // namespace kernels
} // namespace tensorrt_llm
