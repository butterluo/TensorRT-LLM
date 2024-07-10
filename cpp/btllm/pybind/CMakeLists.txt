set(BTLLM_PYBIND_MODULE btpybind)
set(BTLLM_PYBIND_MODULE
    ${BTLLM_PYBIND_MODULE}
    PARENT_SCOPE)

execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c"
          "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='');"
  RESULT_VARIABLE PYBIND_CMAKE_DIR_RET
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR)

if(PYBIND_CMAKE_DIR_RET MATCHES 0)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND_CMAKE_DIR}")
else()
  message(ERROR "pybind11 CMake directory not found.")
endif()

find_package(pybind11 REQUIRED)

set(SRCS
    bindings.cpp
  )

pybind11_add_module(${BTLLM_PYBIND_MODULE} ${SRCS})

set_property(TARGET ${BTLLM_PYBIND_MODULE} PROPERTY POSITION_INDEPENDENT_CODE
                                                     ON)

target_link_directories(${BTLLM_PYBIND_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(
  ${BTLLM_PYBIND_MODULE}
  PUBLIC ${BTSHARED_TARGET} ${Python3_LIBRARIES} ${TORCH_LIBRARIES} torch_python btllm_plugin
         ${UNDEFINED_FLAG})
target_compile_definitions(${BTLLM_PYBIND_MODULE}
                           PUBLIC BTLLM_PYBIND_MODULE=${BTLLM_PYBIND_MODULE})
target_compile_definitions(${BTLLM_PYBIND_MODULE}
                           PUBLIC PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set_target_properties(
    ${BTLLM_PYBIND_MODULE}
    PROPERTIES LINK_FLAGS
               "-Wl,-rpath,'$ORIGIN/libs' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")
endif()
