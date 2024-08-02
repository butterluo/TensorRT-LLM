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
        self.weight = nn.Parameter(torch.ones(hidden_size))
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
def test_embedding(dtype, use_lookup_plugin):

    # meta data
    batch_size = 10
    vocab_size = 1000
    n_embed = 1024

    # test data
    ## input index
    index_shape = (batch_size, )
    index_data = torch.randint(0,
                                vocab_size,
                                index_shape,
                                dtype=torch.int32,
                                device="cuda")

    ## weight data
    weight_data = torch.rand(vocab_size,
                              n_embed,
                              dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                              device="cuda")
    gamma_data = torch.rand(n_embed,
                              dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                              device="cuda")
    # construct trt network
    builder = tensorrt_llm.Builder()
    network = builder.create_network()

    # if use_lookup_plugin:
    #     network.plugin_config.lookupGL_plugin = dtype

    with tensorrt_llm.net_guard(network):
        index = Tensor(name='index',
                        shape=index_data.shape,
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))

        weight = Tensor(name='weight',
                        shape=weight_data.shape,
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))
        
        gamma = Tensor(name='gamma',
                        shape=gamma_data.shape,
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))

        resid, hid_aft_norm = tensorrt_llm.functionalGL.emb_rms(input=index,
                                                    weight=weight,
                                                    gamma=gamma,
                                                    dtype=dtype)
        resid.mark_output('resid', dtype)
        hid_aft_norm.mark_output('hid_aft_norm', dtype)

    # trt run
    session = create_session(builder, network, precision=dtype)
    inputs = {
        'index': index_data,
        'weight': weight_data,
        'gamma' : gamma_data
    }
    outputs = run_session(session, inputs)

    # pytorch run
    embedding = torch.nn.Embedding.from_pretrained(weight_data).cuda()
    norm = LlamaRMSNorm(n_embed, eps=1e-5).cuda()
    ref = embedding(index_data)
    # ned = norm(ref)
    # print(ned)
    # print('-----------------------')
    norm.weight.data = gamma_data
    ned2 = norm(ref)
    # print(ned2)

    # compare diff
    torch.testing.assert_close(ref, outputs['resid'])
    print('-----------------------DONE resid---------------------')
    torch.testing.assert_close(ned2, outputs['hid_aft_norm'])
    print('-----------------------DONE hid_aft_norm---------------------')


test_embedding('bfloat16', True)
