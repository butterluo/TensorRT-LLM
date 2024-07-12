import os
import sys

sys.path.append(os.path.dirname(__file__))

from path_util import *
sys.path.append(f"{PRJ_ROOT_PATH}/cpp/build_Debug")  #@# 以便加载pybind11生成的so包
sys.path.append(f"{PRJ_ROOT_PATH}/tests/quantization/")
sys.path.append(f"{PRJ_ROOT_PATH}/tests/")


import _utils

# isort: off
import torch
import tensorrt as trt
# isort: on
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import \
    weight_only_groupwise_quant_matmul


# from utils.util import (skip_pre_ada_unittest, skip_pre_ampere_unittest,
#                         skip_pre_hopper_unittest, unittest_name_func)

tensorrt_llm.logger.set_level('verbose')

from btllm.pybind.btpybind import *

def _run_matmul_plugin( th_activation,
                        th_pre_quant_scale,
                        th_weight,
                        th_scale,
                        th_zero,
                        th_bias,
                        th_alpha,
                        dtype,
                        quant_algo,
                        group_size=128):
    # Create builder
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(dtype)#@#quant set activation data type
    with tensorrt_llm.net_guard(net):
        network = tensorrt_llm.default_trtnet()
        # Init TensorRT-LLM tensor for activation
        activation = Tensor(
            name='activation',
            shape=th_activation.shape,
            dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
        # Init TensorRT-LLM tensor for pre_quant_scale
        pre_quant_scale = Tensor(
            name='pre_quant_scale',
            shape=th_pre_quant_scale.shape,
            dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
        # Init TensorRT-LLM tensor for weight
        weight = Tensor(name='weight',
                        shape=th_weight.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
        # Init TensorRT-LLM tensor for scale
        scale = Tensor(name='scale',
                        shape=th_scale.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
        # Init TensorRT-LLM tensor for zero
        zero = Tensor(name='zero',
                      shape=th_zero.shape,
                      dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
        # Init TensorRT-LLM tensor for bias
        bias = Tensor(name='bias',
                      shape=th_bias.shape,
                      dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))
        # Init TensorRT-LLM tensor for alpha
        alpha = Tensor(
            name='alpha',
            shape=th_alpha.shape,
            dtype=tensorrt_llm._utils.str_dtype_to_trt("float32"))

        # Get output tensor for WOQ Matmul
        output = weight_only_groupwise_quant_matmul(activation,
                                                    pre_quant_scale,
                                                    weight,
                                                    scale,
                                                    zero,
                                                    bias,
                                                    alpha,
                                                    quant_algo,
                                                    group_size,
                                                    dtype=dtype).trt_tensor
        output.name = 'output'
        network.mark_output(output)
        output.dtype = tensorrt_llm._utils.str_dtype_to_trt(dtype)

    # Build engine consisting of only WBQ Matmul
    build_engine = EngineFromNetwork(
        (builder.trt_builder, net.trt_network),
        config=CreateConfig(
            fp16=(dtype == "float16"),
            bf16=(dtype == "bfloat16"),
            memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

    # Infer engine
    with TrtRunner(build_engine) as runner:
        outputs = runner.infer(
            feed_dict={
                'activation': th_activation,
                'pre_quant_scale': th_pre_quant_scale,
                'weight': th_weight,
                'scale': th_scale,
                'zero': th_zero,
                'bias': th_bias,
                'alpha': th_alpha
            })

    return outputs['output']

def _woq_groupwise_matmul(m,
                          n,
                          k,
                          activation_dtype_str,
                          quantized_weight_dtype,
                          has_pre_quant,
                          has_zero,
                          has_bias,
                          group_size=128,
                          use_w4a8_awq=False):
    seed = 131
    # original quant gemm version: 0(atol=1e-7),1(1e-6),131(1e-7) BUT with bias in only can pass atol=1e-1,rtol=1e-2
    torch.manual_seed(seed)
    print(f"***** seed: {seed} *****")
    activation_dtype = tensorrt_llm._utils.str_dtype_to_torch(
        activation_dtype_str)

    total_groups = (k + group_size - 1) // group_size
    activation = torch.randn(m, k, dtype=activation_dtype)
    bias = torch.randn(
        1, n, dtype=activation_dtype) if has_bias else torch.Tensor().to(
            activation_dtype)
    zero = torch.randn(
        total_groups, n, dtype=activation_dtype
    ) if has_zero else torch.Tensor().to(activation_dtype)

    scale = torch.rand(total_groups, n, dtype=activation_dtype)
    pre_quant_scale = torch.rand(1, k, dtype=activation_dtype)
    fp8_alpha = torch.rand(
        1, dtype=torch.float32) if use_w4a8_awq else torch.Tensor().float()

    num_weights_in_32_bits = 0
    if quantized_weight_dtype == torch.int8:
        num_weights_in_32_bits = 4
    elif quantized_weight_dtype == torch.quint4x2:
        num_weights_in_32_bits = 8
    else:
        assert False, "Unsupported weight dtype."

    assert n % num_weights_in_32_bits == 0, f"n must be a multiple of {num_weights_in_32_bits}"
    unprocessed_int_weight = torch.randint(-2**31,
                                            2**31,
                                            (k, n // num_weights_in_32_bits),
                                            dtype=torch.int32)

    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8

    unprocessed_weight = unprocessed_int_weight.view(torch.int8)

    ref_q_weight = unpacker(unprocessed_weight)
    if use_w4a8_awq:
        activation_type = torch.float8_e4m3fn
    else:
        activation_type = torch.float16
    cuda_q_weight = preprocessor(unprocessed_weight, quantized_weight_dtype,
                                  activation_type).view(activation_dtype)

    # Flags for indicating whether the corresponding inputs are applied in quant_algo
    BIAS = 1
    ZERO = 2
    PRE_QUANT_SCALE = 4
    W4A8_AWQ = 8

    quant_algo = use_w4a8_awq * W4A8_AWQ + has_pre_quant * PRE_QUANT_SCALE + has_zero * ZERO + has_bias * BIAS

    scale_ref = scale.repeat_interleave(group_size, dim=0)[:k, :]
    ref_th_weight = ref_q_weight.to(activation_dtype) * scale_ref

    if has_zero:
        zero_ref = zero.repeat_interleave(group_size, dim=0)[:k, :]
        ref_th_weight += zero_ref

    # output = _run_matmul_plugin(activation, pre_quant_scale,
    #                                   cuda_q_weight, scale, zero, bias,
    #                                   fp8_alpha, activation_dtype_str,
    #                                   quant_algo, group_size).cpu()
    
    quantPlugin = createBTWeightOnlyGroupwiseQuantMatmulPlugin(
                int(trt.bfloat16), #str_dtype_to_trt("bfloat16"),
                quant_algo,  
                group_size,
                m,n,k)
    zero = zero.cuda()
    scale = scale.cuda()
    activation=activation.cuda() 
    cuda_q_weight = cuda_q_weight.cuda()
    bias = bias.cuda()
    output = gemmWeightOnlyGrpwisBf16W8(quantPlugin, m, n, k, zero, scale, activation, cuda_q_weight, bias)
    # print(output)
    if use_w4a8_awq:
        activation *= fp8_alpha

    if has_pre_quant:
        pre_quant_scale = pre_quant_scale.repeat(m, 1)
        activation = torch.mul(activation, pre_quant_scale)

    ref = _utils.woq_groupwise_gt_matmul(activation, ref_th_weight, bias) #@# coment for temporaly
    ref = activation.cuda() + ref.cuda()
    _utils.woq_assert_near_eq(ref, output, 2)
    print("************DONE****************")


# @parameterized.expand(
#     [(1, 1024, 64, 'bfloat16', False, True, True, 64),
#       (16, 1024, 256, 'bfloat16', False, True, False, 64),
#       (32, 2048, 384, 'bfloat16', False, False, True, 64),
#       (64, 2048, 1024, 'bfloat16', False, False, False, 64),
#       (2, 1024, 128, 'bfloat16', False, True, True, 128),
#       (8, 1024, 256, 'bfloat16', False, True, False, 128),
#       (48, 2048, 384, 'bfloat16', False, False, True, 128),
#       (96, 2048, 1024, 'bfloat16', False, False, False, 128)],
#     name_func=unittest_name_func)
# @skip_pre_ampere_unittest
def test_matmul_bf16_int4_input(m,
                                n,
                                k,
                                dtype,
                                has_pre_quant,
                                has_zero,
                                has_bias,
                                group_size=128):
    _woq_groupwise_matmul(m, n, k, dtype, torch.quint4x2,
                                has_pre_quant, has_zero, has_bias,
                                group_size)


# @parameterized.expand([(3, 1024, 64, 'bfloat16', True, True, 64),
#                        (128, 1024, 256, 'bfloat16', True, False, 64),
#                        (192, 2048, 384, 'bfloat16', False, True, 64),
#                        (256, 2048, 1024, 'bfloat16', False, False, 64),
#                        (4, 1024, 128, 'bfloat16', True, True, 128),
#                        (64, 1024, 256, 'bfloat16', True, False, 128),
#                        (384, 2048, 384, 'bfloat16', False, True, 128),
#                        (512, 2048, 1024, 'bfloat16', False, False, 128)],
#                       name_func=unittest_name_func)
# @skip_pre_ampere_unittest
# def test_prequant_matmul_bf16_int4_input(self,
#                                          m,
#                                          n,
#                                          k,
#                                          dtype,
#                                          has_zero,
#                                          has_bias,
#                                          group_size=128):
#     has_pre_quant = True
#     self._woq_groupwise_matmul(m, n, k, dtype, torch.quint4x2,
#                                has_pre_quant, has_zero, has_bias,
#                                group_size)

if __name__ == '__main__':
    test_matmul_bf16_int4_input(32, 128, 128, 'bfloat16', False, True, False, 64)
