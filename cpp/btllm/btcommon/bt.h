#pragma once

#include <type_traits>
#include <cuda_bf16.h>
#include <NvInferRuntime.h>

namespace btllm {


struct BTBaseArg {};

struct BTBaseParam {};

template<typename T>
nvinfer1::DataType toTrtDataType() ;

int trtDataTypeToBitSz(nvinfer1::DataType t) ;

}