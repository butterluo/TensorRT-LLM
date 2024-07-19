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

set(BTPLUGIN_TARGET_NAME btllm_plugin)
set(BTPLUGIN_SHARED_TARGET ${BTPLUGIN_TARGET_NAME})

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map) # Linux

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
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
    weightOnlyGroupwiseQuantMatmulPlugin
    # weightOnlyQuantMatmulPlugin
    lookupPlugin
    # loraPlugin
    # mixtureOfExperts
    # selectiveScanPlugin
    # mambaConv1dPlugin
    # lruPlugin
    # cumsumLastDimPlugin
  )

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
  include_directories(${PLUGIN_ITER})
  add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

include_directories(${TRTLLMDIR_SRC}/plugins/common)
# add_subdirectory(${TRTLLMDIR_SRC}/plugins/common tensorrt_llm/plugins/common) #@# 添加非子目录参考：https://stackoverflow.com/questions/35260552/how-do-i-explicitly-specify-an-out-of-tree-source-in-cmake
include_directories(common)
add_subdirectory(common)

set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "-G -O0 -Xptxas=-O0 -Xptxas=-disable-optimizer-consts")

# Set gencodes
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

# list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/api/tllmPlugin.cpp")

# ################################# SHARED LIBRARY
# ##############################################################################

add_library(${BTPLUGIN_SHARED_TARGET} SHARED ${PLUGIN_SOURCES})

target_include_directories(
  ${BTPLUGIN_SHARED_TARGET}
  PUBLIC ${CUDA_INSTALL_DIR}/include
  PRIVATE ${TARGET_DIR})

set_target_properties(
  ${BTPLUGIN_SHARED_TARGET}
  PROPERTIES CXX_STANDARD "17"
             CXX_STANDARD_REQUIRED "YES"
             CXX_EXTENSIONS "NO"
             ARCHIVE_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             LIBRARY_OUTPUT_DIRECTORY "${TRT_OUT_DIR}"
             RUNTIME_OUTPUT_DIRECTORY "${TRT_OUT_DIR}")

# set_target_properties(
#   ${BTPLUGIN_SHARED_TARGET}
#   PROPERTIES
#     LINK_FLAGS
#     "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,-rpath,'$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
# ) #@# change to below
set_target_properties(
  ${BTPLUGIN_SHARED_TARGET}
  PROPERTIES
    LINK_FLAGS
    "-Wl,--exclude-libs,ALL -Wl,-rpath,'$ORIGIN' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
)

# set_target_properties(
#   ${BTPLUGIN_SHARED_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION
#                                                             ${TRT_SOVERSION})

set_property(TARGET ${BTPLUGIN_SHARED_TARGET} PROPERTY CUDA_STANDARD 17)

target_link_libraries(
  ${BTPLUGIN_SHARED_TARGET}
  ${CUBLAS_LIB}
  ${CUBLASLT_LIB}
  nvinfer
  ${CUDA_DRV_LIB}
  ${CUDA_NVML_LIB}
  ${CUDA_RT_LIB}
  ${CMAKE_DL_LIBS}
  nvinfer_plugin_tensorrt_llm   #@#from cpp/tensorrt_llm/plugins/CMakeLists.txt
  ${BTSHARED_TARGET})
