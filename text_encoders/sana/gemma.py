# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import importlib

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def sdpa_attention_forward(config, query, key, value, mask=None, **kwargs):
    key = repeat_kv(key, config["num_key_value_groups"])
    value = repeat_kv(value, config["num_key_value_groups"])

    causal_mask = mask
    if mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query.device.type == "cuda" and causal_mask is not None:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and query.shape[1] > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=config["scaling"],
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None

def eager_attention_forward(config, query, key, value, mask, **kwargs):
    key_states = repeat_kv(key, config["num_key_value_groups"])
    value_states = repeat_kv(value, config["num_key_value_groups"])

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * config["scaling"]

    if config["attn_logit_softcapping"] is not None:
        attn_weights = attn_weights / config["attn_logit_softcapping"]
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * config["attn_logit_softcapping"]
    if mask is not None:  # no matter the length, we just slice it
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    #attn_weights = nn.functional.dropout(attn_weights, p=0, training=config.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

# torch 2.0 can't pass scale arg to sdpa
if int((torch.__version__).split(".")[1]) >= 1:
    attention_forward = sdpa_attention_forward
else:
    attention_forward = eager_attention_forward

class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class Gemma2MLP(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype, device=device)
        if config["hidden_activation"] != "gelu_pytorch_tanh":
            raise NotImplementedError("Unknown act mode")
        self.act_fn = torch.nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config["attention_dropout"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rope_theta = config["rope_theta"]
        self.is_causal = True
        self.scaling = config["query_pre_attn_scalar"]**-0.5
        self.sliding_window = config["sliding_window"] if not bool(layer_idx % 2) else None
        self.attn_logit_softcapping = config["attn_logit_softcapping"]
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = operations.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config["attention_bias"], dtype=dtype, device=device)
        self.k_proj = operations.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config["attention_bias"], dtype=dtype, device=device)
        self.v_proj = operations.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config["attention_bias"], dtype=dtype, device=device)
        self.o_proj = operations.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config["attention_bias"], dtype=dtype, device=device)
        self.rotary_emb = Gemma2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position= None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        config = {
            "scaling": self.scaling,
            "num_key_value_groups": self.num_key_value_groups,
            "max_position_embeddings": self.max_position_embeddings,
            "attn_logit_softcapping": self.attn_logit_softcapping,
        }
        attn_output, attn_weights = attention_forward(config, query_states, key_states, value_states, attention_mask, output_attentions=output_attentions)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.config = config
        self.is_sliding = not bool(layer_idx % 2)

        self.self_attn = Gemma2Attention(config=config, layer_idx=layer_idx, dtype=dtype, device=device, operations=operations)
        self.mlp = Gemma2MLP(config, dtype=dtype, device=device, operations=operations)

        self.input_layernorm = Gemma2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = Gemma2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

        self.pre_feedforward_layernorm = Gemma2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_feedforward_layernorm = Gemma2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.sliding_window = config["sliding_window"]

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None):
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # # Flash-attn is a 2D tensor
            # if self.config["_attn_implementation == "flash_attention_2":
            #     if past_key_value is not None:  # when decoding
            #         attention_mask = attention_mask[:, -self.sliding_window :]
            # else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

def prepare_causal_mask(input_tensor, attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    batch_size=input_tensor.shape[0]
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    #causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    return causal_mask

class Gemma2Model(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.padding_idx = 0
        self.hidden_size = config_dict["hidden_size"]
        self.embed_tokens = operations.Embedding(config_dict["vocab_size"], self.hidden_size, self.padding_idx, device=device, dtype=dtype)
        self.num_layers = config_dict["num_hidden_layers"]
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config_dict, layer_idx, dtype=dtype, device=device, operations=operations) for layer_idx in range(config_dict["num_hidden_layers"])]
        )
        self.norm = Gemma2RMSNorm(self.hidden_size, eps=config_dict["rms_norm_eps"])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, intermediate_output=None, final_layer_norm_intermediate=False, *args, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        hidden_states = inputs_embeds
        intermediate = None

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        causal_mask = prepare_causal_mask(inputs_embeds, attention_mask)

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        for i, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            if i == intermediate_output:
                intermediate = hidden_states.clone()
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)

        return hidden_states, intermediate
