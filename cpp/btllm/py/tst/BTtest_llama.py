import os
import sys

sys.path.append(os.path.dirname(__file__))

from path_util import *
sys.path.append(f"{PRJ_ROOT_PATH}/cpp/build_Debug")  #@# 以便加载pybind11生成的so包
sys.path.append(f"{PRJ_ROOT_PATH}/tests/")
sys.path.append(f"{PRJ_ROOT_PATH}/tests/quantization/")
sys.path.append(f"{PRJ_ROOT_PATH}/tests/model/")


import random
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np
np.set_printoptions(
    threshold=np.inf,
    precision=3,
    suppress=True,
    linewidth=640)
import json
import pytest
import torch
from parameterized import parameterized
from transformers import LlamaConfig, LlamaForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_str
from tensorrt_llm.models.llama.weight import (load_from_hf_llama,
                                              load_from_meta_llama)
from tensorrt_llm.models.modeling_utils import PretrainedConfig, optimize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import (skip_bf16_pre_ampere, skip_fp32_accum_pre_ampere,
                        unittest_name_func)

from btllm.pybind.btpybind import *


# class TestLLaMA(unittest.TestCase):
EOS_TOKEN = 2
PAD_TOKEN = 2

def _gen_tensorrt_llm_network(network, hf_llama,
                              llama_config: LlamaConfig, batch_size,
                              beam_width, input_len, output_len, dtype,
                              rank, tensor_parallel, **opt_flags):
    list(range(tensor_parallel))

    with net_guard(network):
        str_dtype_to_trt(dtype)

        config = {
            'architecture': "LlamaForCausalLM",
            'dtype': dtype,
            'logits_dtype': 'float32',
            'num_hidden_layers': llama_config.num_hidden_layers,
            'num_attention_heads': llama_config.num_attention_heads,
            'hidden_size': llama_config.hidden_size,
            'intermediate_size': llama_config.intermediate_size,
            'num_key_value_heads': llama_config.num_key_value_heads,
            'vocab_size': llama_config.vocab_size,
            'position_embedding_type': 'rope_gpt_neox',
            'max_position_embeddings': llama_config.max_position_embeddings,
            'hidden_act': llama_config.hidden_act,
            'rotary_base': getattr(llama_config, 'rotary_base', 10000.0),
            'rotary_scaling': getattr(llama_config, 'rotary_scaling', None),
            'norm_epsilon': llama_config.rms_norm_eps,
            'mapping': {
                'world_size': tensor_parallel,
                'tp_size': tensor_parallel,
            },
            "moe_config": {
                "num_experts": 0,
                "top_k": 0,
                "tp_mode": 2,
                "normalization_mode": 1
            },
            'use_parallel_embedding': False,
            'embedding_sharding_dim': 0,
            'moe_num_experts': 0,
            'moe_top_k': 0,
            'moe_tp_mode': 2,
            'moe_normalization_mode': 1,
        }

        # Initialize model
        tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(
            PretrainedConfig.from_dict(config))
        weights = load_from_hf_llama(tensorrt_llm_llama,
                                      hf_llama,
                                      dtype=dtype,
                                      mapping=tensorrt_llm.Mapping(
                                          world_size=tensor_parallel,
                                          rank=rank,
                                          tp_size=tensor_parallel))
        tensorrt_llm_llama.load(weights)
        optimize_model(tensorrt_llm_llama, **opt_flags)

        # Prepare
        network.set_named_parameters(tensorrt_llm_llama.named_parameters())
        inputs = tensorrt_llm_llama.prepare_inputs(
            max_batch_size=batch_size,
            max_input_len=input_len,
            max_seq_len=input_len + output_len,
            use_cache=True,
            max_beam_width=beam_width)
        # Forward
        tensorrt_llm_llama(**inputs)

    return network

def _gen_tensorrt_llm_engine(        
                              dtype,
                              rank,
                              world_size,
                              llama_config,
                              hf_llama,
                              model_name,
                              use_plugin,
                              batch_size,
                              beam_width,
                              input_len,
                              output_len,
                              use_refit,
                              fast_building=False,
                              context_fmha_flag=ContextFMHAType.disabled,
                              enable_remove_input_padding=False,
                              **opt_flags):

    builder = Builder()

    with tempfile.TemporaryDirectory() as tmpdirname:
        builder_config = builder.create_builder_config(
            name=model_name,
            precision=dtype,
            timing_cache='model.cache',
            tensor_parallel=world_size,  # TP only
            use_refit=use_refit,
            strongly_typed=(dtype in ["float16", "bfloat16"]),
        )
        network = builder.create_network()
        network.plugin_config.to_legacy_setting()
        if use_plugin:
            network.plugin_config.set_gpt_attention_plugin(dtype)
        if fast_building:
            network.plugin_config.set_gemm_plugin(dtype)
        if enable_remove_input_padding:
            network.plugin_config.enable_remove_input_padding()
        network.plugin_config.set_context_fmha(context_fmha_flag)

        _gen_tensorrt_llm_network(network, hf_llama, llama_config,
                                        batch_size, beam_width, input_len,
                                        output_len, dtype, rank, world_size,
                                        **opt_flags)

        engine_buffer = builder.build_engine(network, builder_config)
        return engine_buffer

def _gen_tensorrt_llm_runtime(
                              log_level,
                              dtype,
                              world_size,
                              rank,
                              llama_config,
                              hf_llama,
                              model_name,
                              use_plugin,
                              batch_size,
                              beam_width,
                              input_len,
                              output_len,
                              use_refit,
                              fast_building=False,
                              context_fmha_flag=ContextFMHAType.disabled,
                              enable_remove_input_padding=False,
                              **opt_flags):
    tensorrt_llm.logger.set_level(log_level)
    mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)
    engine_buffer = _gen_tensorrt_llm_engine(
        dtype, rank, world_size, llama_config, hf_llama, model_name,
        use_plugin, batch_size, beam_width, input_len, output_len,
        use_refit, fast_building, context_fmha_flag,
        enable_remove_input_padding, **opt_flags)
    runtime = tensorrt_llm.runtime.generation._Runtime(
        engine_buffer, mapping)
    return runtime, engine_buffer

def load_test_cases():
    test_cases = list(
        product([False], [False, True], [
            ContextFMHAType.disabled, ContextFMHAType.enabled,
            ContextFMHAType.enabled_with_fp32_acc
        ], [False, True], ['float16'], [0], ['silu'],
                [{
                    "use_fused_mlp": True
                }, {
                    "use_fused_mlp": False
                }]))
    test_cases.append((False, True, ContextFMHAType.disabled, False,
                        'bfloat16', 0, 'silu', dict()))
    test_cases.append(
        (False, True, ContextFMHAType.enabled, False, 'float16', 1, 'silu',
          dict()))  # MQA
    test_cases.append(
        (False, True, ContextFMHAType.disabled, False, 'float32', 0, 'silu',
          dict()))
    test_cases.append((False, True, ContextFMHAType.disabled, False,
                        'bfloat16', 2, 'silu', dict()))  # GQA
    test_cases.append(
        (False, True, ContextFMHAType.enabled, False, 'float16', 2, 'silu',
          dict()))  # GQA
    test_cases.append((False, True, ContextFMHAType.enabled_with_fp32_acc,
                        False, 'float16', 4, 'silu', dict()))  # GQA
    test_cases.append((False, True, ContextFMHAType.disabled, False,
                        'float16', 2, 'gelu', {
                            "use_fused_mlp": True
                        }))  # GQA
    test_cases.append((False, True, ContextFMHAType.disabled, False,
                        'float16', 2, 'silu', {
                            "use_fused_mlp": True
                        }))  # GQA
    return test_cases

# @parameterized.expand(load_test_cases, name_func=unittest_name_func)  
def test_llama(use_refit, fast_building, context_fmha_flag,
                enable_remove_input_padding, dtype, num_kv_heads, hidden_act,
                opt_flags):

    # Skip tests that are not supported in pre-ampere architecture
    skip_bf16_pre_ampere(dtype)
    skip_fp32_accum_pre_ampere(context_fmha_flag)

    PRECHECKED_GOOD_RANDOM_SEEDS = [1, 4, 5, 8]
    model = 'llama'
    log_level = 'error'
    use_plugin = True  # gpt plugin
    batch_size = 4
    beam_width = 1
    input_len = 5 #4
    output_len = 2
    max_seq_len = input_len + output_len
    world_size = 1
    head_size = 32
    rank = 0
    llama_config = LlamaConfig()
    llama_config.hidden_act = hidden_act
    llama_config.num_hidden_layers = 2
    llama_config.max_position_embeddings = 64
    llama_config.vocab_size = 128
    llama_config.num_attention_heads = 2 if num_kv_heads <= 1 else 2 * num_kv_heads
    llama_config.hidden_size = llama_config.num_attention_heads * head_size
    llama_config.intermediate_size = ((
        (llama_config.hidden_size * 4 * 2 // 3) + head_size - 1) //
                                      head_size) * head_size
    if hasattr(llama_config, "num_key_value_heads"):
        llama_config.num_key_value_heads = num_kv_heads if num_kv_heads != 0 else llama_config.num_attention_heads
        print(llama_config.num_key_value_heads)
        assert (llama_config.num_attention_heads %
                llama_config.num_key_value_heads) == 0
    llama_config.pad_token_id = PAD_TOKEN
    llama_config.eos_token_id = EOS_TOKEN
    seed_idx = random.randint(0, len(PRECHECKED_GOOD_RANDOM_SEEDS) - 1)
    torch.manual_seed(PRECHECKED_GOOD_RANDOM_SEEDS[seed_idx])
    hf_llama = LlamaForCausalLM(llama_config).cuda()
    # runtime, _ = _gen_tensorrt_llm_runtime(
    #     log_level, dtype, world_size, rank, llama_config, hf_llama, model,
    #     use_plugin, batch_size, beam_width, input_len, output_len,
    #     use_refit, fast_building, context_fmha_flag,
    #     enable_remove_input_padding, **opt_flags)
    key_value_cache_buffers = []
    head_size = llama_config.hidden_size // llama_config.num_attention_heads
    for i in range(llama_config.num_hidden_layers):
        key_value_cache_buffers.append(
            torch.zeros((
                batch_size,
                2,
                llama_config.num_key_value_heads,
                max_seq_len,
                head_size,
            ),
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                        device='cuda'))

    # compare context
    step = 0
    ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
    ctx_context_lengths = input_len * torch.ones(
        (batch_size), dtype=torch.int32, device='cuda')
    ctx_position_ids = torch.tensor(range(input_len),
                                    dtype=torch.int32).reshape([
                                        1, input_len
                                    ]).expand([batch_size,
                                                input_len]).cuda()
    ctx_last_token_ids = ctx_context_lengths.clone()
    ctx_host_request_types = torch.tensor([0] * batch_size,
                                          dtype=torch.int32)

    # We need sequence_lengths start as context_lengths for step 0,
    # and it will be added one after each step.
    sequence_length_buffer = ctx_context_lengths.detach().clone()

    with torch.no_grad():
        hf_outputs = hf_llama.forward(ctx_ids, output_hidden_states=True)
    torch.cuda.synchronize()
    ref = hf_outputs.logits[:, -1, :]
    ###################### Remove origin and Add below
    print("-------------------compute ref hf lma---------------------")
    btref = hf_llama.model.embed_tokens(ctx_ids)
    btref = hf_llama.model.layers[0].input_layernorm(btref)
    query_states = hf_llama.model.layers[0].self_attn.q_proj(btref)
    key_states = hf_llama.model.layers[0].self_attn.k_proj(btref)
    value_states = hf_llama.model.layers[0].self_attn.v_proj(btref)
    btref = torch.cat([query_states, key_states, value_states], dim=-1)
    btref = btref.clone().detach()
    btref_np = btref.to(torch.float32).cpu().numpy();
    print(btref_np)
    print("-------------------crate cpp lma----------")
    hf_llama.cpu()
    w = mrgeFfW(hf_llama)
    # jsnStr = llama_config.to_json_string(use_diff=False)
    cfgDic = llama_config.to_dict()
    cfgDic['other'] = 1
    jsnStr = json.dumps(cfgDic)
    print(jsnStr)
    lma = createLlama(
            jsnStr,
            256, #batch_size * llama_config.max_position_embeddings,  can be assign a different val manually
            w.cuda()
            )
    # print("----------compare w--------------")
    # cppw = getLlamaW(lma)
    # hfw = hf_llama.model.layers[0].input_layernorm.weight.detach().to(torch.bfloat16)
    # hfw = torch.cat([
    #         hf_llama.model.layers[0].self_attn.q_proj.weight.detach().to(torch.bfloat16),
    #         hf_llama.model.layers[0].self_attn.k_proj.weight.detach().to(torch.bfloat16),
    #         hf_llama.model.layers[0].self_attn.v_proj.weight.detach().to(torch.bfloat16)
    #     ], dim=0)
    # np.testing.assert_allclose(cppw.to(torch.float32).cpu().numpy(), 
    #                            hfw.to(torch.float32).cpu().numpy(), 
    #                            rtol=1e-7, atol=0, verbose=True)
    print("-------------------RUN cpp lma----------")
    btres = runLlama(lma, ctx_ids, output_len)
    # btres = btres.permute(2, 1,0).contiguous()
    btres_np = btres.to(torch.float32).cpu().numpy()
    print(btres_np)
    
    print("--------------manual compare--------------------")
    rtol = 1e-7  # Set your relative tolerance
    atol = 0.12  # Set your absolute tolerance (optional)

    # Calculate the absolute difference between elements
    difference = np.abs(btref_np - btres_np)

    # Find indices where the difference is greater than tolerance
    mismatch_indices = np.where(difference > (atol + rtol * np.abs(btres_np)))
    print("----Mismatch indices:  \n", mismatch_indices)
    print("-----------numpy compare------------")
    np.testing.assert_allclose(btref_np, btres_np, rtol=rtol, atol=atol,verbose=True)
    print("---------DONE----------")


def calc_offset(sizes):
    offsets = [int(0)]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(int(tmp))
    return offsets
def cpyWeights(para, para_offset, w, i):
    cur_para =  para.data.narrow(
        0, para_offset[i], para_offset[i + 1] - para_offset[i]
    )
    assert cur_para.numel() == w.numel()
    cur_para.copy_(w.view(-1))
def mrgeFfW(hf:LlamaForCausalLM):
    cfg:LlamaConfig = hf.model.config
    vcbSz = cfg.vocab_size
    hs = cfg.hidden_size
    ims = cfg.intermediate_size
    hsKV = cfg.num_key_value_heads *  (cfg.hidden_size / cfg.num_attention_heads)
    sizes = [vcbSz * hs]
    for i in range(cfg.num_hidden_layers):
        subSz = [
            hs,  # attn_nw
            hs * (hs + 2 * hsKV) ,  # attn_qkvw
            # hs * 3,  # attn_qkvb
            hs * hs,  # attn_ow
            # hs,  # attn_ob
            # hs,  # attn_nb
            hs,  # ffn_nw
            hs * ims,  # inter_w (up_w)
            # ims,  # inter_b
            hs * ims,  # gate_w
            hs * ims,  # output_w
            # hs,  # output_b
        ]
        sizes.extend(subSz)
    sizes.extend([
        hs, hs * vcbSz
    ])
    para_offset = calc_offset(sizes)
    # para = torch.nn.Parameter(torch.Tensor(para_offset[-1]).to(torch.bfloat16))
    para = torch.Tensor(para_offset[-1]).to(torch.bfloat16)

    idx = 0
    cpyWeights(para, para_offset, 
               hf.model.embed_tokens.weight.detach().to(torch.bfloat16), idx)
    idx+=1
    for i in range(cfg.num_hidden_layers):
        cpyWeights(para, para_offset, 
            hf.model.layers[i].input_layernorm.weight.detach().to(torch.bfloat16), idx)
        idx+=1
        #@# assume attn_implementation is eager
        qkvw = torch.cat([
            hf.model.layers[i].self_attn.q_proj.weight.detach().to(torch.bfloat16),
            hf.model.layers[i].self_attn.k_proj.weight.detach().to(torch.bfloat16),
            hf.model.layers[i].self_attn.v_proj.weight.detach().to(torch.bfloat16)
        ], dim=0)#.transpose(0,1).contiguous()
        cpyWeights(para, para_offset, 
            qkvw.to(torch.bfloat16), idx)
        idx+=1
        cpyWeights(para, para_offset, 
            hf.model.layers[i].self_attn.o_proj.weight.detach().to(torch.bfloat16), idx)
        idx+=1
        cpyWeights(para, para_offset, 
            hf.model.layers[i].post_attention_layernorm.weight.detach().to(torch.bfloat16), idx)
        idx+=1
        cpyWeights(para, para_offset, 
            hf.model.layers[i].mlp.up_proj.weight.detach().to(torch.bfloat16), idx)
        idx+=1
        cpyWeights(para, para_offset, 
            hf.model.layers[i].mlp.gate_proj.weight.detach().to(torch.bfloat16), idx)
        idx+=1
        cpyWeights(para, para_offset, 
            hf.model.layers[i].mlp.down_proj.weight.detach().to(torch.bfloat16), idx)
        idx+=1
    cpyWeights(para, para_offset, 
            hf.model.norm.weight.detach().to(torch.bfloat16), idx)
    idx+=1
    cpyWeights(para, para_offset, 
            hf.lm_head.weight.detach().to(torch.bfloat16), idx)
    idx+=1
    assert(idx == len(sizes))
    return para













if __name__ == '__main__':
    # unittest.main()
    test_llama(
        use_refit=False, 
        fast_building=True, 
        context_fmha_flag=ContextFMHAType.enabled,
        enable_remove_input_padding=False, 
        dtype='bfloat16', 
        num_kv_heads=0, #@#???
        hidden_act='silu',
        opt_flags={"use_fused_mlp": True}
    )
