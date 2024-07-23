/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "btllm/plugins/common/plugin.h"

#include "tensorrt_llm/common/mpiUtils.h"

#include "checkMacrosPlugin.h"
#include "cuda.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <functional>
#include <mutex>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif


namespace
{

// Get current cuda context, a default context will be created if there is no context.
inline CUcontext getCurrentCudaCtx()
{
    CUcontext ctx{};
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_ERROR_NOT_INITIALIZED || ctx == nullptr)
    {
        TLLM_CUDA_CHECK(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    TLLM_CHECK(err == CUDA_SUCCESS);
    return ctx;
}

// Helper to create per-cuda-context singleton managed by std::shared_ptr.
// Unlike conventional singletons, singleton created with this will be released
// when not needed, instead of on process exit.
// Objects of this class shall always be declared static / global, and shall never own CUDA
// resources.
template <typename T>
class PerCudaCtxSingletonCreator
{
public:
    using CreatorFunc = std::function<std::unique_ptr<T>()>;
    using DeleterFunc = std::function<void(T*)>;

    // creator returning std::unique_ptr is by design.
    // It forces separation of memory for T and memory for control blocks.
    // So when T is released, but we still have observer weak_ptr in mObservers, the T mem block can be released.
    // creator itself must not own CUDA resources. Only the object it creates can.
    PerCudaCtxSingletonCreator(CreatorFunc creator, DeleterFunc deleter)
        : mCreator{std::move(creator)}
        , mDeleter{std::move(deleter)}
    {
    }

    std::shared_ptr<T> operator()()
    {
        std::lock_guard<std::mutex> lk{mMutex};
        CUcontext ctx{getCurrentCudaCtx()};
        std::shared_ptr<T> result = mObservers[ctx].lock();
        if (result == nullptr)
        {
            // Create the resource and register with an observer.
            result = std::shared_ptr<T>{mCreator().release(),
                [this, ctx](T* obj)
                {
                    if (obj == nullptr)
                    {
                        return;
                    }
                    mDeleter(obj);

                    // Clears observer to avoid growth of mObservers, in case users creates/destroys cuda contexts
                    // frequently.
                    std::shared_ptr<T> observedObjHolder; // Delay destroy to avoid dead lock.
                    std::lock_guard<std::mutex> lk{mMutex};
                    // Must check observer again because another thread may created new instance for this ctx just
                    // before we lock mMutex. We can't infer that the observer is stale from the fact that obj is
                    // destroyed, because shared_ptr ref-count checking and observer removing are not in one atomic
                    // operation, and the observer may be changed to observe another instance.
                    observedObjHolder = mObservers.at(ctx).lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers.erase(ctx);
                    }
                }};
            mObservers.at(ctx) = result;
        }
        return result;
    }

private:
    CreatorFunc mCreator;
    DeleterFunc mDeleter;
    mutable std::mutex mMutex;
    // CUDA resources are per-context.
    std::unordered_map<CUcontext, std::weak_ptr<T>> mObservers;
};
} // namespace

std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    static PerCudaCtxSingletonCreator<cublasHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasHandle_t>(new cublasHandle_t);
            TLLM_CUDA_CHECK(cublasCreate(handle.get()));
            return handle;
        },
        [](cublasHandle_t* handle)
        {
            TLLM_CUDA_CHECK(cublasDestroy(*handle));
            delete handle;
        });
    return creator();
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    static PerCudaCtxSingletonCreator<cublasLtHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasLtHandle_t>(new cublasLtHandle_t);
            TLLM_CUDA_CHECK(cublasLtCreate(handle.get()));
            return handle;
        },
        [](cublasLtHandle_t* handle)
        {
            TLLM_CUDA_CHECK(cublasLtDestroy(*handle));
            delete handle;
        });
    return creator();
}