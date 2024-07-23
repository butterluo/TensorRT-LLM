
#include <cstdio>
#include "btllm/plugins/gptAttentionPlugin/btGptAttentionPlugin.h"

using btllm::plugins::BTGPTAttentionPluginCreator;

tensorrt_llm::plugins::GPTAttentionPlugin* BTGPTAttentionPluginCreator::createPlugin(int layer_idx, nvinfer1::DataType actType, Json &jsnObj) noexcept {
    try
    {
        float rotary_embedding_percentage = 1.0;
        float rotary_embedding_scale = 1.0;
        float mscale = 0;
        int headSz = parseJsonFieldOr(jsnObj,"attention_head_size",0);

        tensorrt_llm::plugins::GPTAttentionPlugin* obj = new tensorrt_llm::plugins::GPTAttentionPlugin(
            layer_idx,
            parseJsonFieldOr(jsnObj,"num_attention_heads",0), //p.getScalar<int32_t>("num_heads").value(), 
            -1,                                               //p.getScalar<int32_t>("vision_start").value(),
            -1,                                               //p.getScalar<int32_t>("vision_length").value(), 
            parseJsonFieldOr(jsnObj,"num_attention_kv_heads",0), //p.getScalar<int32_t>("num_kv_heads").value(),
            headSz,                                              //p.getScalar<int32_t>("head_size").value(), 
            1,                                                   //p.getScalar<int32_t>("unidirectional").value(),         @#???
            1,                                                   //p.getScalar<float>("q_scaling").value(),
            tensorrt_llm::kernels::PositionEmbeddingType(parseJsonFieldOr(jsnObj,"position_embedding_type",0)),    
                                                              //static_cast<PositionEmbeddingType>(p.getScalar<int8_t>("position_embedding_type").value()),
                                                              //@#TODO add PositionEmbeddingType to py cfg, refer to attention.py
            headSz * rotary_embedding_percentage ,               //p.getScalar<int32_t>("rotary_embedding_dim").value(), 
            parseJsonFieldOr(jsnObj,"rotary_base",10000.0f), //p.getScalar<float>("rotary_embedding_base").value(),
            tensorrt_llm::kernels::RotaryScalingType(parseJsonFieldOr(jsnObj,"rotary_embedding_scale_type",0)), 
                                                              //static_cast<RotaryScalingType>(p.getScalar<int8_t>("rotary_embedding_scale_type").value()),
                                                              //@#TODO add RotaryScalingType to py cfg, refer to attention.py
            rotary_embedding_scale,                               //p.getScalar<float>("rotary_embedding_scale").value(),
            mscale,                                               //p.getScalar<float>("rotary_embedding_m_scale").value(),
            parseJsonFieldOr(jsnObj,"max_position_embeddings",0), //p.getScalar<int32_t>("rotary_embedding_max_positions").value(),
            1,                                                    //static_cast<int32_t>(p.getScalar<int32_t>("tp_size").value()),    @#TODO have NOT support multi node
            0,                                                    //static_cast<int32_t>(p.getScalar<int32_t>("tp_rank").value()),    @#TODO have NOT support multi node
            0,                                                    //static_cast<bool>(p.getScalar<int8_t>("unfuse_qkv_gemm").value()),   @#only support fuse qkv
            tensorrt_llm::kernels::ContextFMHAType::ENABLED_WITH_FP32_ACC,   //static_cast<ContextFMHAType>(p.getScalar<int8_t>("context_fmha_type").value()),   @#TODO need to change to ENABLE that is use bf16 as accumulation
            0,                                               //static_cast<bool>(p.getScalar<int8_t>("multi_block_mode").value()),  @#TODO have NOT support multi_block
            0,                                               //static_cast<bool>(p.getScalar<int8_t>("enable_xqa").value()),    @#TODO have NOT support xqa(e.g. MQA,GQA)
            0,                                               //p.getScalar<int32_t>("kv_cache_quant_mode").value(),             @#TODO have NOT support kv_cache_quant
            1,                                               //static_cast<bool>(p.getScalar<int8_t>("remove_input_padding").value()),   @#must rm pad
            tensorrt_llm::kernels::AttentionMaskType::CAUSAL,//static_cast<AttentionMaskType>(p.getScalar<int32_t>("mask_type").value()),@#only support causal masking right now
            0,                                               //static_cast<bool>(p.getScalar<int32_t>("paged_kv_cache").value()),@#TODO have NOT support paged_kv_cache
            64,                                              //p.getScalar<int32_t>("tokens_per_block").value(),@#TODO have NOT support paged_kv_cache so tokens_per_block will not work, but its default val is 64 according to trtllm's code
            actType,                                         //static_cast<nvinfer1::DataType>(p.getScalar<int32_t>("type_id").value()),
            parseJsonFieldOr(jsnObj,"max_context_length",0),   //p.getScalar<int32_t>("max_context_length").value(), @#same as 'prepare_inputs( max_input_len=...' in modeling_util.py and max_batch_token_size in llama.cpp @#TODO mv to py cfg
                //@#TODO !!! mv above line to py cfg
            0,                                              //static_cast<bool>(p.getScalar<int8_t>("qkv_bias_enabled").value()),
            0,                                              //static_cast<bool>(p.getScalar<int8_t>("do_cross_attention").value()),
            0,                                              //static_cast<int32_t>(p.getScalar<int32_t>("max_distance").value()),  @#???
            0,                                            //static_cast<bool>(p.getScalar<int8_t>("pos_shift_enabled").value()),   @#TODO have NOT support streamingllm
            0,                                            //static_cast<bool>(p.getScalar<int8_t>("dense_context_fmha").value()),   @#TODO have NOT support streamingllm
            0,                                              //static_cast<bool>(p.getScalar<int8_t>("use_paged_context_fmha").value()),@# NOT support FP8 Context FMHA
            0,                                              //static_cast<bool>(p.getScalar<int8_t>("use_fp8_context_fmha").value()),@# NOT support FP8 Context FMHA
            1,                                              //static_cast<bool>(p.getScalar<int32_t>("use_cache").value()), @#always use kv cache
            0                                           //static_cast<bool>(p.getScalar<int8_t>("is_spec_decoding_enabled").value()));  @#TODO have NOT support spec_decod
        );
        // obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}