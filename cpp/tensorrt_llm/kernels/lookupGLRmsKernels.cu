/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/lookupGLRmsKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

/////////////////////////////////////////// from customAllReduceKernels //////////////////////////////
namespace gl {
using PackedFloat = union
{
    int4 packed;
    float unpacked[4];
};

using PackedHalf = union
{
    int4 packed;
    half2 unpacked[4];
};

template <typename T>
struct PackedOn16Bytes
{
};

template <>
struct PackedOn16Bytes<float>
{
    using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half>
{
    using Type = PackedHalf;
};

#ifdef ENABLE_BF16
using PackedBFloat16 = union
{
    int4 packed;
    __nv_bfloat162 unpacked[4];
};

template <>
struct PackedOn16Bytes<__nv_bfloat16>
{
    using Type = PackedBFloat16;
};

#endif

// add two 128b data
template <typename T>
inline __device__ int4 add128b(T& a, T& b)
{
    T c;
    c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
    c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
    c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
    c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
    return c.packed;
}

namespace details
{
static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
}; // namespace details

inline __device__ float warp_reduce_sum(float val)
{
    val += __shfl_xor_sync(~0, val, 16);
    val += __shfl_xor_sync(~0, val, 8);
    val += __shfl_xor_sync(~0, val, 4);
    val += __shfl_xor_sync(~0, val, 2);
    val += __shfl_xor_sync(~0, val, 1);
    return val;
}

inline __device__ float block_reduce_sum(float val)
{
    __shared__ float smem[details::kWarpSize];
    int lane_id = threadIdx.x % details::kWarpSize, warp_id = threadIdx.x / details::kWarpSize,
        warp_num = blockDim.x / details::kWarpSize;
    val = warp_reduce_sum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();
    val = lane_id < warp_num ? smem[lane_id] : 0.f;
    val = warp_reduce_sum(val);
    return val;
}

template <typename T, typename PackedStruct>
inline __device__ float accumulate(float acc, PackedStruct& vec)
{
    static constexpr int kLoopNum = sizeof(PackedStruct) / sizeof(T);
#pragma unroll
    for (int i = 0; i < kLoopNum; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<T*>(vec.unpacked)[i]);
        acc += v * v;
    }
    return acc;
}

template <typename T, bool Affine, typename PackedStruct>
inline __device__ int4 rms_norm(float denom, PackedStruct& vec, PackedStruct& weight)
{
    static constexpr int kLoopNum = sizeof(PackedStruct) / sizeof(T);
    PackedStruct ret;
#pragma unroll
    for (int i = 0; i < kLoopNum; ++i)
    {
        float v1 = static_cast<float>(reinterpret_cast<T*>(vec.unpacked)[i]);
        if constexpr (Affine)
        {
            float v2 = static_cast<float>(reinterpret_cast<T*>(weight.unpacked)[i]);
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(__fdividef(v1, denom) * v2);
        }
        else
        {
            reinterpret_cast<T*>(ret.unpacked)[i] = static_cast<T>(__fdividef(v1, denom));
        }
    }
    return ret.packed;
}

template <typename T, bool Bias = false, bool Residual = false, bool Affine = false, bool UseSmem = false>
__global__ void rms_norm_kernel(RmsParams params)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    extern __shared__ uint8_t smem_ptr[];
    T* smem = reinterpret_cast<T*>(smem_ptr);

    int bid = blockIdx.x, tid = threadIdx.x;

    T const* bias_buffer = reinterpret_cast<T const*>(params.fusion_params.bias_buffer);
    T const* residual_buffer = reinterpret_cast<T const*>(params.fusion_params.residual_buffer);
    T const* weight_buffer = reinterpret_cast<T const*>(params.fusion_params.weight_buffer);
    T* local_final_output_buffer = reinterpret_cast<T*>(params.local_output_buffer_ptr);
    T const* intermediate_buffer = reinterpret_cast<T const*>(params.fusion_params.intermediate_buffer);
    T* residual_out = reinterpret_cast<T*>(params.fusion_params.residual_out);  //@#

    int block_offset = bid * params.fusion_params.hidden_size;
    int thread_offset = tid * kPackedSize;

    if constexpr (Residual)
    {
        residual_buffer += block_offset;
    }
    local_final_output_buffer += block_offset;
    intermediate_buffer += block_offset;
    residual_out += block_offset;

    PackedStruct inter_vec, weight_vec;
    float acc = 0.f;
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        inter_vec.packed = *reinterpret_cast<int4 const*>(intermediate_buffer + offset);
        if constexpr (Bias)
        {
            PackedStruct bias_vec;
            bias_vec.packed = *reinterpret_cast<int4 const*>(bias_buffer + offset);
            inter_vec.packed = add128b(inter_vec, bias_vec);
        }
        if constexpr (Residual)
        {
            PackedStruct residual_vec;
            residual_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer + offset);
            inter_vec.packed = add128b(inter_vec, residual_vec);
            // *reinterpret_cast<int4*>(intermediate_buffer + offset) = inter_vec.packed;
            *reinterpret_cast<int4*>(residual_out + offset) = inter_vec.packed;
        }
        acc = accumulate<T>(acc, inter_vec);
        if constexpr (UseSmem)
        {
            *reinterpret_cast<int4*>(&smem[offset]) = inter_vec.packed;
        }
    }
    acc = block_reduce_sum(acc);
    float denom = __fsqrt_rn(__fdividef(acc, params.fusion_params.hidden_size) + EPS);
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        if constexpr (UseSmem)
        {
            inter_vec.packed = *reinterpret_cast<int4 const*>(&smem[offset]);
        }
        if constexpr (Affine)
        {
            weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + offset);
        }
        inter_vec.packed = rms_norm<T, Affine>(denom, inter_vec, weight_vec);
        *reinterpret_cast<int4*>(&local_final_output_buffer[offset]) = inter_vec.packed;
    }
}


template <typename T, bool Bias = false, bool Residual = false, bool Affine = false>
void rms_norm_kernel_launcher(RmsParams params, cudaStream_t stream)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    TLLM_CHECK(params.fusion_params.hidden_size % kPackedSize == 0);
    int need_threads = params.fusion_params.hidden_size / kPackedSize;
    int cta_size;
    if (need_threads <= details::kMaxCtaSize)
    {
        cta_size = (need_threads + details::kWarpSize - 1) / details::kWarpSize * details::kWarpSize;
    }
    else
    {
        cta_size = details::kMaxCtaSize;
    }
    int cta_num = params.elts_total / params.fusion_params.hidden_size;
    int smem_size = 0;
    if (cta_size * details::kBytesPerAccess / sizeof(T) < params.fusion_params.hidden_size)
    {
        smem_size = params.fusion_params.hidden_size * sizeof(T);
        rms_norm_kernel<T, Bias, Residual, Affine, true><<<cta_num, cta_size, smem_size, stream>>>(params);
    }
    else
    {
        rms_norm_kernel<T, Bias, Residual, Affine, false><<<cta_num, cta_size, smem_size, stream>>>(params);
    }
}

template <typename T>
void launchResidualRmsNormKernel(RmsParams& params, cudaStream_t stream)
{
    if (params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        rms_norm_kernel_launcher<T, true, true, true>(params, stream);
    }
    else if (params.fusion_params.bias_buffer && !params.fusion_params.weight_buffer)
    {
        rms_norm_kernel_launcher<T, true, true, false>(params, stream);
    }
    else if (!params.fusion_params.bias_buffer && params.fusion_params.weight_buffer)
    {
        rms_norm_kernel_launcher<T, false, true, true>(params, stream);
    }
    else
    {
        rms_norm_kernel_launcher<T, false, true, false>(params, stream);
    }
}

void residualRmsNorm(RmsParams& params, nvinfer1::DataType dataType, cudaStream_t stream)
{
    sync_check_cuda_error();
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: launchResidualRmsNormKernel<float>(params, stream); break;
    case nvinfer1::DataType::kHALF: launchResidualRmsNormKernel<half>(params, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: launchResidualRmsNormKernel<__nv_bfloat16>(params, stream); break;
#endif
    default: TLLM_THROW("Unsupported dataType for customAllReduce");
    }
    sync_check_cuda_error();
}
} // namespace gl




template <typename T, typename Tw = T, typename Idx>
__global__ void lookupRmsKernel(T* residual, T* output, Idx const* input, Tw const* weight, int64_t const token_num, Idx const rankoffset,
    Idx const size, Idx const n_elems, Tw const* __restrict gamma)
{
    static constexpr int kPackedSize = gl::details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename gl::PackedOn16Bytes<T>::Type;

    // extern __shared__ uint8_t smem_ptr[];
    // T* smem = reinterpret_cast<T*>(smem_ptr);

    int bid = blockIdx.x, tid = threadIdx.x;

    __shared__ int s_word_index;
    if(tid == 0) {
        s_word_index = input[blockIdx.x];
    }
    __syncthreads();
    int word_index = s_word_index;
    int offset = tid * kPackedSize;//thread_offset
    int block_offset = bid * n_elems;
    T* residual_out = residual + block_offset;
    T* local_final_output_buffer = output + block_offset;
    PackedStruct inter_vec, gamma_vec;
    float acc = 0.f;
    if (word_index >= 0 && word_index < size) {
      // int wrd_offset = word_index * n_elems;
      //for (int offset = thread_offset; offset < n_elems; offset += blockDim.x * kPackedSize) {
      inter_vec.packed = *reinterpret_cast<int4 const*>(weight + (word_index * n_elems + offset));
      *reinterpret_cast<int4*>(residual_out + offset) = inter_vec.packed;
      acc = gl::accumulate<T>(acc, inter_vec);
      //}end for
    }
    
    acc = gl::block_reduce_sum(acc);
    float denom = __fsqrt_rn(__fdividef(acc, n_elems) + EPS);
    //for (int offset = thread_offset; offset < n_elems; offset += blockDim.x * kPackedSize) {
    // gamma_vec.packed = *reinterpret_cast<int4 const*>(gamma + offset);
    gamma_vec.packed = ldg(reinterpret_cast<int4 const*>(gamma + offset));
    inter_vec.packed = gl::rms_norm<T, true>(denom, inter_vec, gamma_vec);
    *reinterpret_cast<int4*>(&local_final_output_buffer[offset]) = inter_vec.packed;
    //}end for
}



template <typename T, typename Tw, typename Idx>
void invokeLookUpRms(T* residual, T* out, Idx const* input, Tw const* weight, int64_t const token_num, Idx const offset, Idx const size,
    Idx const n_embed, Tw const* gamma, cudaStream_t stream)
{
    sync_check_cuda_error();
    static constexpr int kPackedSize = gl::details::kBytesPerAccess / sizeof(T);
    TLLM_CHECK(n_embed % kPackedSize == 0);
    int need_threads = n_embed / kPackedSize;
    int cta_size;
    if (need_threads <= gl::details::kMaxCtaSize)
    {
        cta_size = (need_threads + gl::details::kWarpSize - 1) / gl::details::kWarpSize * gl::details::kWarpSize;
    }
    else
    {
        cta_size = gl::details::kMaxCtaSize;
    }
    int cta_num = token_num;
    int smem_size = 0;
    if (cta_size * gl::details::kBytesPerAccess / sizeof(T) >= n_embed)
    {
        lookupRmsKernel<T, Tw, Idx><<<cta_num, cta_size, smem_size, stream>>>(residual, out, input, weight, token_num, offset, size, n_embed, gamma);
    }
    else
    {
        // smem_size = params.fusion_params.hidden_size * sizeof(T);
        // rms_norm_kernel<T, Bias, Residual, Affine, true><<<cta_num, cta_size, smem_size, stream>>>(params);
        TLLM_THROW("hidSz is too large. NOT supported");
    }
    sync_check_cuda_error();
}


#define INSTANTIATE_LOOK_UPRMS(T, Tw, Idx)                                                                                    \
    template void invokeLookUpRms<T, Tw, Idx>(T* residual, T * out, Idx const* input, Tw const* weight, int64_t const token_num,            \
        Idx const offset, Idx const size, Idx const n_embed, Tw const* gamma, cudaStream_t stream)

// INSTANTIATE_LOOK_UPRMS(float, int);
// INSTANTIATE_LOOK_UPRMS(half, int);

#ifdef ENABLE_BF16
INSTANTIATE_LOOK_UPRMS(__nv_bfloat16, __nv_bfloat16, int);
#endif


template <typename T>
__global__ void rmsRsidKernel(T* output_resid, T* output, T const* input, T const* input_resid, int const n_elems/*hidSz*/, T const* __restrict gamma)
{
    static constexpr int kPackedSize = gl::details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename gl::PackedOn16Bytes<T>::Type;

    // extern __shared__ uint8_t smem_ptr[];
    // T* smem = reinterpret_cast<T*>(smem_ptr);

    int bid = blockIdx.x, tid = threadIdx.x;

    int offset = tid * kPackedSize;//thread_offset
    int block_offset = bid * n_elems;
    T const* intermediate_buffer = input + block_offset;
    T const* residual_buffer = input_resid + block_offset;
    T* residual_out = output_resid + block_offset;
    T* local_final_output_buffer = output + block_offset;
    PackedStruct inter_vec, resid_gamma_vec;
    float acc = 0.f;
    //for (int offset = thread_offset; offset < n_elems; offset += blockDim.x * kPackedSize) {
    inter_vec.packed = *reinterpret_cast<int4 const*>(intermediate_buffer + offset);
    resid_gamma_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer + offset);
    inter_vec.packed = gl::add128b(inter_vec, resid_gamma_vec);
    *reinterpret_cast<int4*>(residual_out + offset) = inter_vec.packed;
    acc = gl::accumulate<T>(acc, inter_vec);
    //}end for
    
    acc = gl::block_reduce_sum(acc);
    float denom = __fsqrt_rn(__fdividef(acc, n_elems) + EPS);
    //for (int offset = thread_offset; offset < n_elems; offset += blockDim.x * kPackedSize) {
    // resid_gamma_vec.packed = *reinterpret_cast<int4 const*>(gamma + offset);
    resid_gamma_vec.packed = ldg(reinterpret_cast<int4 const*>(gamma + offset));
    inter_vec.packed = gl::rms_norm<T, true>(denom, inter_vec, resid_gamma_vec);
    *reinterpret_cast<int4*>(&local_final_output_buffer[offset]) = inter_vec.packed;
    //}end for
}

template <typename T>
void invokeRmsResid(T* output_resid, T* output, T const* input, T const* input_resid, int const n_embed/*hidSz*/, T const* gamma, int64_t const token_num, cudaStream_t stream)
{
    sync_check_cuda_error();
    static constexpr int kPackedSize = gl::details::kBytesPerAccess / sizeof(T);
    TLLM_CHECK(n_embed % kPackedSize == 0);
    int need_threads = n_embed / kPackedSize;
    int cta_size;
    if (need_threads <= gl::details::kMaxCtaSize)
    {
        cta_size = (need_threads + gl::details::kWarpSize - 1) / gl::details::kWarpSize * gl::details::kWarpSize;
    }
    else
    {
        cta_size = gl::details::kMaxCtaSize;
    }
    int cta_num = token_num;
    int smem_size = 0;
    if (cta_size * gl::details::kBytesPerAccess / sizeof(T) >= n_embed)
    {
        rmsRsidKernel<T><<<cta_num, cta_size, smem_size, stream>>>(output_resid, output, input, input_resid, n_embed, gamma);
    }
    else
    {
        // smem_size = params.fusion_params.hidden_size * sizeof(T);
        // rms_norm_kernel<T, Bias, Residual, Affine, true><<<cta_num, cta_size, smem_size, stream>>>(params);
        TLLM_THROW("hidSz is too large. NOT supported");
    }
    sync_check_cuda_error();
}

#define INSTANTIATE_LOOK_RMSRESID(T)                                                                                    \
    template void invokeRmsResid(T* output_resid, T* output, T const* input, T const* input_resid, int const n_embed/*hidSz*/, T const* gamma, int64_t const token_num, cudaStream_t stream)

// INSTANTIATE_LOOK_UPRMS(float, int);
// INSTANTIATE_LOOK_UPRMS(half, int);

#ifdef ENABLE_BF16
INSTANTIATE_LOOK_RMSRESID(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm
