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

} // namespace kernels
} // namespace tensorrt_llm
