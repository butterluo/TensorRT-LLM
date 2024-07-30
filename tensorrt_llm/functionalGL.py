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
import math
import weakref
from collections import OrderedDict
from enum import IntEnum, IntFlag, auto
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# isort: off
import tensorrt as trt
# isort: on

from . import graph_rewriting as gw
from ._common import default_net, default_trtnet, precision
from ._utils import (bf16_array, bool_array, dim_resolve_negative,
                     dim_to_trt_axes, dims_array, fp16_array, fp32_array,
                     int32_array, int64_array, np_dtype_to_trt,
                     str_dtype_to_trt, trt_dtype_to_np, trt_dtype_to_str,
                     trt_gte_10)
from .network import PluginInfo, set_np_weight, set_plugin_info
from .plugin import TRT_LLM_PLUGIN_NAMESPACE, current_all_reduce_helper
from .quantization import QuantMode
from .functional import Tensor, _create_tensor,_add_plugin_info


def _lookupGL_plugin(input: Tensor, weight: Tensor,  gamma: Tensor, rank: int,
                   per_token_scale: Tensor) -> Tensor:
    '''
    Add an operation to perform lookup in a tensor.

    That operation performs the lookup needed by embedding layers. Given a
    'weight' tensor of shape [rows, cols], it produces a tensor of shape
    [inputs.size(0), cols] where the ith row corresponds to the input[i] row in
    the weight tensor.

    It inserts a IPluginV2Layer.

    Parameters:
        input : Tensor
            The input tensor contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        rank :  int
            The mpi rank.

    Returns:
        The output tensor of the lookup layer.
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'LookupGL', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    p_dtype = default_net().plugin_config.lookup_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)

    rank = trt.PluginField("rank", np.array([int(rank)], np.int32),
                           trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_type, rank])
    lookup_plug = plg_creator.create_plugin("lookupGL", pfc)
    plug_inputs = [input.trt_tensor, weight.trt_tensor, gamma.trt_tensor]
    if per_token_scale is not None:
        plug_inputs.append(per_token_scale.trt_tensor)
        weight.trt_tensor.set_dynamic_range(-127, 127)
    layer = default_trtnet().add_plugin_v2(plug_inputs, lookup_plug)
    _add_plugin_info(layer, plg_creator, "lookupGL", pfc)
    resid =_create_tensor(layer.get_output(0), layer)
    hid_aft_norm = _create_tensor(layer.get_output(1), layer)
    return resid, hid_aft_norm


def emb_rms(input: Tensor,
              weight: Tensor,
              gamma: Tensor,
              tp_size=1,
              tp_group=None,
              sharding_dim=0,
              tp_rank=None,
              per_token_scale=None) -> Tensor:
    '''
    Add an operation to perform embedding lookup.

    That operation performs the embedding lookup. The 'input' tensor contains
    the identifiers of the rows of 'weight' to gather.

    1. Distribute the embedding lookup table over multiple GPU
    When 'tp_size' is greater than 1 and the 'tp_group' is defined, this
    embedding lookup is distributed among multiple GPUs.

    When 'sharding_dim==0', each GPU stores a subset of the rows of the embedding
    table rows(that number of rows per GPU is given by weights.shape[0] and the offset to
    the 1st row stored on the GPU is given by rank * weights.shape[0]). Each
    parallel rank will query all the indices and set 0s for the weights that
    are not stored on the associated GPU. To compute the final result, a
    parallel all-reduce operation is added to the TensorRT graph. That lookup
    can be performed using either the plugin or the operators TensorRT support.

    When'sharding_dim==1', each GPU stores a subset of the embedding table's columns.
    Each rank can obtain a portion of the embedding results.
    Then the embedding is collected using the  all-gather operation.
    Related transposition operations are also used to obtain the final results.

    2. Store embedding lookup table as a whole
    When 'tp_size' is not greater than 1, the embedding lookup table will not
    be divided. In this case, when the default_net().plugin_config.lookup_plugin is set,
    the operation is implemented using a plugin (without the all-reduce operation).
    Otherwise, this operation is implemented using the standard IGatherLayer in TensorRT.

    Parameters:
        input : Tensor
            The input tensor the contains the indices to perform the lookup.

        weight : Tensor
            The table to gather from.

        tp_size : int
            The number of GPUs collaborating to perform that embedding.

        tg_group : Optional[List[int]]
            The group of world ranks participating in the all-reduce when
            tp_size > 1.

        sharding_dim : int
            sharding_dim = 0 means that we shard the embedding table in vocab dim;
            sharding_dim = 1 means that we shard the embedding table in embedding dim.

        tp_rank : int
            The tensor parallelism rank. Used to calculate offset in TP on vocab dim.

    Returns:
        The tensor produced by the embedding lookup layer.
    '''
    if tp_size <=1 and tp_group is None:
        x = _lookupGL_plugin(input,
                               weight,
                               gamma,
                               rank=0,
                               per_token_scale=per_token_scale)
    else:
      raise ValueError(
                  'NOT support parallelism now'
              )
    return x




