# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import tensorrt_llm.functionalGL

sys.path.append(os.path.dirname(__file__))

from path_util import *
sys.path.append(f"{PRJ_ROOT_PATH}/cpp/build_Debug")  #@# 以便加载pybind11生成的so包
sys.path.append(f"{PRJ_ROOT_PATH}/tests/")
sys.path.append(f"{PRJ_ROOT_PATH}/tests/quantization/")
sys.path.append(f"{PRJ_ROOT_PATH}/tests/model/")

import unittest

import torch
from torch import nn
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.parameter import Parameter

import time
torch.random.manual_seed(time.time_ns())

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import create_session, run_session, unittest_name_func


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.rand(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# class TestEmbedding(unittest.TestCase):
#
#     def setUp(self):
#         torch.random.manual_seed(0)
#         tensorrt_llm.logger.set_level('error')
#
#     @parameterized.expand([(
#         'float32',
#         True,
#     ), (
#         'float32',
#         False,
#     ), (
#         'float16',
#         True,
#     ), (
#         'float16',
#         False,
#     )],
#                           name_func=unittest_name_func)
def test_rmsResid(dtype):

    # meta data
    batch_size = 10
    seqLen = 1000
    n_embed = 6656#1024,2560,3072,4096,5120,6656,8192

    # test data
    input_data = torch.rand(batch_size, seqLen,
                              n_embed,
                              dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                              device="cuda")
    resid_data = torch.rand(batch_size, seqLen,
                              n_embed,
                              dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                              device="cuda")
    gamma_data = torch.rand(n_embed,
                              dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                              device="cuda")
    # construct trt network
    builder = tensorrt_llm.Builder()
    network = builder.create_network()

    with tensorrt_llm.net_guard(network):
        inp = Tensor(name='input',
                        shape=input_data.shape,
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))

        resid = Tensor(name='resid',
                        shape=resid_data.shape,
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))
        normalized_shape = (n_embed, )
        gamma = Tensor(name='gamma',
                        shape=tuple(normalized_shape), #gamma_data.shape,
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))
        # gamma = Parameter(shape=gamma_data.shape, dtype=dtype).value

        hid_aft_norm, resid2 = tensorrt_llm.functionalGL.resd_rms_plugin(input=inp,
                                                    residual=resid,
                                                    gamma=gamma,
                                                    type=dtype)
        hid_aft_norm.mark_output('hid_aft_norm', dtype)
        resid2.mark_output('resid2', dtype)

    # trt run
    session = create_session(builder, network, precision=dtype)
    inputs = {
        'input': input_data,
        'gamma' : gamma_data,
        'resid': resid_data
    }
    outputs = run_session(session, inputs)

    # pytorch run
    ref_resid2 = input_data + resid_data
    norm = LlamaRMSNorm(n_embed, eps=1e-5).cuda()
    norm.weight.data = gamma_data
    ref_hid = norm(ref_resid2)
    # print(ned2)

    # compare diff
    torch.testing.assert_close(ref_resid2, outputs['resid2'])
    print('-----------------------DONE resid2---------------------')
    torch.testing.assert_close(ref_hid, outputs['hid_aft_norm'])
    print('-----------------------DONE hid_aft_norm---------------------')


test_rmsResid('bfloat16')
