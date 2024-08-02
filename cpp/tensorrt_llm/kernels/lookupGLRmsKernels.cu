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
/* When running with multiple GPUs, we split the embedding lookup table across multiple GPUs to save the memory
requirements of embedding lookup table ([vocab_size, hidden]). This operation is equivalent to the single GPU version of
embedding() (i.e.add_gather() operation in TensorRT). As only a portion of embedding lookup table
([ceil(vocab_size/world_size), hidden]) is stored in each GPU and the value range of input IDs is [0, vocab_size]. The
add_gather() operation in TensorRT cannot get the correct results. So, we need to write a plugin to add an offset to
input IDs and get the correct results.

 * Input: Input IDs (input[token_num])
   Input: Embedding Lookup Table (weight[ceil(vocab_size/world_size), hidden])
   Output: weight[input[idx]-offset,hidden]

 * The total thread number equals to token_num*hidden
 *
 * If the input ids is out of range it writes zero, otherwise it writes the correct embedding result.
 */
template <typename T, typename Tw, typename Idx, int tmpNum>
__global__ void lookupRmsKernel(T* residual, T* output, Idx const* input, Tw const* weight, int64_t const token_num, Idx const offset,
    Idx const size, Idx const n_elems, Tw const* __restrict gamma)
{
    __shared__ int word_index;
    __shared__ float rms;
    constexpr auto num_elems_T = num_elems<T>::value;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    // int const n_elems = n_embed / num_elems_T;
    int const n_embed = n_elems * num_elems_T;
    int curIdx = 0;
    T tmpArr[tmpNum] = {0};
    float sqrSum = 0;
    
    if (threadIdx.x == 0)
    {
        word_index = input[blockIdx.x];
    }
    __syncthreads();
    
    if (word_index >= 0 && word_index < size) {
        const float_packed_t rld = cuda_cast<float_packed_t,float>(1.f / float(n_embed));
        for (int64_t index = threadIdx.x; index < n_elems; index += blockDim.x) {
            // int64_t const word_index = input[index / n_embed] - offset;
            // Idx const col_index = index % n_embed;
            T embedding = weight[word_index * n_elems + index];
            residual[blockIdx.x * n_elems + index] = embedding;
            // output[index] = embedding;
            tmpArr[curIdx] = embedding;
            ++curIdx;
            sqrSum = sqrSum + cuda_sum<float>( rld * cuda_cast<float_packed_t,T>(embedding) * cuda_cast<float_packed_t,T>(embedding) ); //@# MAYBUG
        } // end for index

    }
    
    ///////////////// RMS //////////////////
    float blkSum = blockReduceSum(sqrSum);
    curIdx = 0;
    if (threadIdx.x == 0) {
      rms = rsqrtf(blkSum + EPS);
    }
    __syncthreads();
    float_packed_t tRms = cuda_cast<float_packed_t,float>(rms);
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
      T tmpVal = tmpArr[curIdx];
      int const index = blockIdx.x * n_elems + i;
      T g = ldg(&gamma[i]);
      float_packed_t tVal = cuda_cast<float_packed_t,T>(tmpVal) * tRms * cuda_cast<float_packed_t,T>(g);
      // output[index] = tmpVal * tRms * g;
      output[index] = cuda_cast<T, float_packed_t>(tVal);
    }
}

template <typename T, typename Tw, typename Idx>
void invokeLookUpRms(T* residual, T* out, Idx const* input, Tw const* weight, int64_t const token_num, Idx const offset, Idx const size,
    Idx const n_embed, Tw const* gamma, cudaStream_t stream)
{
    // int64_t constexpr max_block_num = 65536;
    // Idx constexpr max_block_size = 512;
    // dim3 grid(min(token_num, max_block_num));
    // dim3 block(min(n_embed, max_block_size));
    // lookup_kernel<T, Idx><<<grid, block, 0, stream>>>(out, input, weight, token_num, offset, size, n_embed);

    dim3 grid(token_num);
    constexpr int num_elems_T = num_elems<T>::value;
    assert(n_embed % num_elems_T); //use macro to avoid assert in release version
    int packedEmb = n_embed / num_elems_T;
    dim3 block(min(packedEmb, 1024));
    // Make sure block.x is multiple of 32 for warp shuffle to work
    block.x = 32 * ((block.x + 31) / 32);

    const int flod = packedEmb > 1024 ? 1 : ((packedEmb + 1023) /1024);//TODO remove, use n_embed as condition instead
    if(flod < 2) {
      lookupRmsKernel<T, Tw, Idx, 1><<<grid, block, 0, stream>>>(residual, out, input, weight, token_num, offset, size, packedEmb, gamma);
    } else if (flod < 5) {
      lookupRmsKernel<T, Tw, Idx, 4><<<grid, block, 0, stream>>>(residual, out, input, weight, token_num, offset, size, packedEmb, gamma);
    } else {
      lookupRmsKernel<T, Tw, Idx, 8><<<grid, block, 0, stream>>>(residual, out, input, weight, token_num, offset, size, packedEmb, gamma);
    }
    
}

#define INSTANTIATE_LOOK_UPRMS(T, Tw, Idx)                                                                                    \
    template void invokeLookUpRms<T, Tw, Idx>(T* residual, T * out, Idx const* input, Tw const* weight, int64_t const token_num,            \
        Idx const offset, Idx const size, Idx const n_embed, Tw const* gamma, cudaStream_t stream)

// INSTANTIATE_LOOK_UPRMS(float, int);
// INSTANTIATE_LOOK_UPRMS(half, int);

#ifdef ENABLE_BF16
INSTANTIATE_LOOK_UPRMS(__nv_bfloat16, __nv_bfloat16, int);
#endif

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




template <typename T, typename Tw = T, int tmpNum =1, bool Bias = false, bool Residual = false, bool Affine = false, bool UseSmem = false>
__global__ void rms_norm_kernel(RmsParams params, T* residual, T* output, Idx const* input, Tw const* weight, int64_t const token_num, Idx const offset,
    Idx const size, Idx const n_elems, Tw const* __restrict gamma)
{
    static constexpr int kPackedSize = details::kBytesPerAccess / sizeof(T);
    using PackedStruct = typename PackedOn16Bytes<T>::Type;

    // extern __shared__ uint8_t smem_ptr[];
    // T* smem = reinterpret_cast<T*>(smem_ptr);

    int bid = blockIdx.x, tid = threadIdx.x;

    __shared__ int s_word_index;
    if(tid == 0) {
        s_word_index = input[blockIdx.x];
    }
    __syncthreads();
    int word_index = s_word_index;

    

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
        // if constexpr (Bias)
        // {
        //     PackedStruct bias_vec;
        //     bias_vec.packed = *reinterpret_cast<int4 const*>(bias_buffer + offset);
        //     inter_vec.packed = add128b(inter_vec, bias_vec);
        // }
        // if constexpr (Residual)
        // {
        //     PackedStruct residual_vec;
        //     residual_vec.packed = *reinterpret_cast<int4 const*>(residual_buffer + offset);
        //     inter_vec.packed = add128b(inter_vec, residual_vec);
        //     // *reinterpret_cast<int4*>(intermediate_buffer + offset) = inter_vec.packed;
            *reinterpret_cast<int4*>(residual_out + offset) = inter_vec.packed;
        // }
        acc = gl::accumulate<T>(acc, inter_vec);
        // if constexpr (UseSmem)
        // {
        //     *reinterpret_cast<int4*>(&smem[offset]) = inter_vec.packed;
        // }
    }
    acc = gl::block_reduce_sum(acc);
    float denom = __fsqrt_rn(__fdividef(acc, params.fusion_params.hidden_size) + EPS);
    for (int offset = thread_offset; offset < params.fusion_params.hidden_size; offset += blockDim.x * kPackedSize)
    {
        // if constexpr (UseSmem)
        // {
        //     inter_vec.packed = *reinterpret_cast<int4 const*>(&smem[offset]);
        // }
        // if constexpr (Affine)
        {
            weight_vec.packed = *reinterpret_cast<int4 const*>(weight_buffer + offset);
        // }
        inter_vec.packed = rms_norm<T, Affine>(denom, inter_vec, weight_vec);
        *reinterpret_cast<int4*>(&local_final_output_buffer[offset]) = inter_vec.packed;
    }
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
    if (cta_size * details::kBytesPerAccess / sizeof(T) >= params.fusion_params.hidden_size)
    {
        rms_norm_kernel<T, Bias, Residual, Affine, false><<<cta_num, cta_size, smem_size, stream>>>(params);
    }
    else
    {
        // smem_size = params.fusion_params.hidden_size * sizeof(T);
        // rms_norm_kernel<T, Bias, Residual, Affine, true><<<cta_num, cta_size, smem_size, stream>>>(params);
        TLLM_THROW("hidSz is too large. NOT supported");
    }
    sync_check_cuda_error();
}




} // namespace kernels
} // namespace tensorrt_llm
