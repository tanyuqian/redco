#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

from jax_llama import LLaMAConfig, FlaxLLaMAForCausalLM


FLAX_PT_KEY_MAPPING = {
    ("transformer", "wte", "embedding"): ("model.embed_tokens.weight", False),
    ("transformer", "ln_f", "kernel"): ("model.norm.weight", False),
    ("lm_head", "kernel"): ("lm_head.weight", True),
    ("transformer", "h", "{layer}", "attention_norm", "kernel"): ("model.layers.{layer}.input_layernorm.weight", False),
    ("transformer", "h", "{layer}", "ffn_norm", "kernel"): ("model.layers.{layer}.post_attention_layernorm.weight", False),
    ("transformer", "h", "{layer}", "feed_forward", "w2", "kernel"): ("model.layers.{layer}.mlp.down_proj.weight", True),
    ("transformer", "h", "{layer}", "feed_forward", "w1", "kernel"): ("model.layers.{layer}.mlp.gate_proj.weight", True),
    ("transformer", "h", "{layer}", "feed_forward", "w3", "kernel"): ("model.layers.{layer}.mlp.up_proj.weight", True),
    ("transformer", "h", "{layer}", "attention", "wk", "kernel"): ("model.layers.{layer}.self_attn.k_proj.weight", True),
    ("transformer", "h", "{layer}", "attention", "wv", "kernel"): ("model.layers.{layer}.self_attn.v_proj.weight", True),
    ("transformer", "h", "{layer}", "attention", "wq", "kernel"): ("model.layers.{layer}.self_attn.q_proj.weight", True),
    ("transformer", "h", "{layer}", "attention", "wo", "kernel"): ("model.layers.{layer}.self_attn.o_proj.weight", True)
}


def convert_llama_pt_params_to_flax(pt_state_dict, llama_config):
    n_layers = llama_config.num_hidden_layers
    n_heads = llama_config.num_attention_heads
    dim = llama_config.hidden_size

    def permute(w):
        return w.view(
            n_heads, 2, dim // n_heads // 2, dim
        ).transpose(1, 2).reshape(dim, dim)

    params = {}
    for flax_key_raw, (pt_key_raw, do_transpose) in FLAX_PT_KEY_MAPPING.items():
        if '{layer}' not in pt_key_raw:
            flax_key, pt_key = flax_key_raw, pt_key_raw

            params[flax_key] = pt_state_dict[pt_key]
            if do_transpose:
                params[flax_key] = params[flax_key].transpose(0, 1)
        else:
            for layer in range(n_layers):
                flax_key = tuple([s.format(layer=layer) for s in flax_key_raw])
                pt_key = pt_key_raw.format(layer=layer)

                if ('q_proj.weight' in pt_key) or ('k_proj.weight' in pt_key):
                    params[flax_key] = permute(pt_state_dict[pt_key])
                else:
                    params[flax_key] = pt_state_dict[pt_key]

                if do_transpose:
                    params[flax_key] = params[flax_key].transpose(0, 1)

    return jax.tree_util.tree_map(jnp.asarray, unflatten_dict(params))


def get_llama(model_name_or_path, bf16):
    config = LLaMAConfig.from_dict(
        HFLlamaConfig.from_pretrained(model_name_or_path).to_dict())
    model = FlaxLLaMAForCausalLM(
        config=config,
        dtype=jnp.bfloat16 if bf16 else jnp.float32,
        _do_init=False)

    pt_state_dict = LlamaForCausalLM.from_pretrained(
        model_name_or_path).state_dict()
    params = convert_llama_pt_params_to_flax(
        pt_state_dict=pt_state_dict, llama_config=config)

    return model, params