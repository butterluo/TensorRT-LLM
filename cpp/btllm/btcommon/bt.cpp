#include "bt.h"


namespace btllm {

// template<typename T>
// nvinfer1::DataType toTrtDataType() {
//   if(std::is_same<T,__nv_bfloat16>::value) {
//     return nvinfer1::DataType::kBF16;
//   } else {
//     return nvinfer1::DataType::kFLOAT;
//   }
// }

template<>
nvinfer1::DataType toTrtDataType<__nv_bfloat16>() {
  return nvinfer1::DataType::kBF16;
}

int trtDataTypeToBitSz(nvinfer1::DataType t) {
  switch(t) {
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF:
      return 16;
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 32;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
      return 8;
    case nvinfer1::DataType::kINT4:
      return 4;
    default:
      return 0;
  }
  return 0;
}

}