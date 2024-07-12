
set(BTTARGET_NAME btllm)
set(BTSHARED_TARGET ${BTTARGET_NAME})
set(BTSHARED_TARGET
    ${BTSHARED_TARGET}
    PARENT_SCOPE)
set(API_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cutlass_extensions/include
#                     ${API_INCLUDE_DIR})  #@# have included in tensorrt_llm/CMakeLists.txt
set(TRTLLMDIR_SRC ${PROJECT_SOURCE_DIR}/tensorrt_llm)
include_directories(${PROJECT_SOURCE_DIR}
                    ${API_INCLUDE_DIR}
                    ${TRTLLMDIR_SRC}/cutlass_extensions/include
          )


find_package(Threads REQUIRED)

add_subdirectory(kernels)

# set(BTLLM_LINK_LIBS
#     ${CUBLAS_LIB}
#     ${CUBLASLT_LIB}
#     ${CMAKE_DL_LIBS}
#     ${TRT_LIB}
#     common_src
#     kernels_src
#     context_attention_src
#     decoder_attention_src
#     fpA_intB_gemm_src
#     moe_gemm_src
#     cutlass_src
#     layers_src
#     runtime_src)
set(BTLLM_LINK_LIBS
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CMAKE_DL_LIBS}
    ${TRT_LIB} 
    common_src
    kernels_src
    fpA_intB_gemm_src
    cutlass_src

    btfpA_intB_gemm_src
  )

set(UNDEFINED_FLAG "-Wl,--no-undefined")
set(AS_NEEDED_FLAG "-Wl,--as-needed")


add_library(${BTSHARED_TARGET} SHARED)

set_target_properties(
  ${BTSHARED_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")

function(link_whole_archive TARGET LIBRARY_TO_LINK)
  # Assume everything else is like gcc
  target_link_libraries(
    ${TARGET} PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:${LIBRARY_TO_LINK}>
                      "-Wl,--no-whole-archive")
endfunction()

target_link_libraries(${BTSHARED_TARGET} PUBLIC ${BTLLM_LINK_LIBS})

set_target_properties(${BTSHARED_TARGET} PROPERTIES LINK_FLAGS
                                                  "-Wl,-rpath='$ORIGIN'")



# if(BUILD_PYT)
#   add_subdirectory(thop)
# endif()

# if(BUILD_PYBIND)
#   add_subdirectory(pybind)
# endif() 

add_subdirectory(plugins)
add_subdirectory(pybind)
