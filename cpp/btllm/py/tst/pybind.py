import sys
new_path = '/home/SRC/MLSys/Framework/TRT/TensorRT-LLM/cpp/build_Debug'
sys.path.append(new_path)

import torch
import tensorrt as trt
from btllm.pybind.btpybind import *

use_w4a8_awq = False
has_pre_quant = False
has_zero = True
has_bias = True

BIAS = 1
ZERO = 2
PRE_QUANT_SCALE = 4
W4A8_AWQ = 8

quant_algo = use_w4a8_awq * W4A8_AWQ + has_pre_quant * PRE_QUANT_SCALE + has_zero * ZERO + has_bias * BIAS

quantPlugin = createBTWeightOnlyGroupwiseQuantMatmulPlugin(
  int(trt.bfloat16), #str_dtype_to_trt("bfloat16"),
  quant_algo,  
  64,
  32,64,64)
print(quantPlugin)
gemmWeightOnlyGrpwisBf16W8(quantPlugin, 32, 64, 64)
print("lasssssssssssssssssssssssssssssssssss")