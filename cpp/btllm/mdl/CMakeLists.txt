#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

file(GLOB SRCS *.cpp)
file(GLOB CU_SRCS *.cu)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(MDL_SOURCES)
set(MDL_CU_SOURCES)

set(MDL_LISTS
    # bertAttentionPlugin
    # gptAttentionCommon
    # gptAttentionPlugin
    # identityPlugin
    # gemmPlugin
    # smoothQuantGemmPlugin
    # quantizePerTokenPlugin
    # quantizeTensorPlugin
    # layernormQuantizationPlugin
    # rmsnormQuantizationPlugin
    llama
    # weightOnlyQuantMatmulPlugin
    # lookupPlugin
    # loraPlugin
    # mixtureOfExperts
    # selectiveScanPlugin
    # mambaConv1dPlugin
    # lruPlugin
    # cumsumLastDimPlugin
  )
foreach(MDL_ITER ${MDL_LISTS})
  include_directories(${MDL_ITER})
  add_subdirectory(${MDL_ITER})
endforeach(MDL_ITER)

list(APPEND MDL_SOURCES "${MDL_CU_SOURCES}")

add_library(btllm_mdl OBJECT ${MDL_SOURCES})
set_property(TARGET btllm_mdl PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET btllm_mdl PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


target_include_directories(btllm_mdl
  PUBLIC ${CUDA_INSTALL_DIR}/include
         ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
  btllm_mdl
  PROPERTIES CXX_STANDARD "17"
             CXX_STANDARD_REQUIRED "YES"
             CXX_EXTENSIONS "NO"
             ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

set_target_properties(
  btllm_mdl
  PROPERTIES
    LINK_FLAGS
    "-Wl,--exclude-libs,ALL -Wl,-rpath,'$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
)

set_property(TARGET btllm_mdl PROPERTY CUDA_STANDARD 17)

target_link_libraries(
  btllm_mdl
  ${CUBLAS_LIB}
  nvinfer
  ${CUDA_RT_LIB}
  nvinfer_plugin_tensorrt_llm   #@#from cpp/tensorrt_llm/plugins/CMakeLists.txt
  ${BTSHARED_TARGET}
  btllm_plugin
  )