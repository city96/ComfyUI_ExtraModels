# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM

from .norms import RMSNorm
from .utils import get_same_padding, to_2tuple
from .basic_modules import Mlp

sdpa_32b = None
Q_4GB_LIMIT = 32000000
"""If q is greater than this, the operation will likely require >4GB VRAM, which will fail on Intel Arc Alchemist GPUs without a workaround."""
# 2k   = 37 748 736
# 1024 =  9 437 184
# 2k model goes very slightly over 4GB

from comfy import model_management
if model_management.xformers_enabled():
    import xformers
    import xformers.ops
else:
    if model_management.xpu_available:
        import intel_extension_for_pytorch as ipex # type: ignore
        import os
        if not torch.xpu.has_fp64_dtype() and not os.environ.get('IPEX_FORCE_ATTENTION_SLICE', None):
            from ...utils.IPEX.attention import scaled_dot_product_attention_32_bit
            sdpa_32b = scaled_dot_product_attention_32_bit
            print("Using IPEX 4GB SDPA workaround")
        else:
            print("No IPEX 4GB workaround")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, dtype=None, device=None, operations=None, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = operations.Linear(d_model, d_model, dtype=dtype, device=device)
        self.kv_linear = operations.Linear(d_model, d_model * 2, dtype=dtype, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = operations.Linear(d_model, d_model, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            # not used for now
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        if model_management.xformers_enabled():
            attn_bias = None
            if mask is not None:
                attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            x = xformers.ops.memory_efficient_attention(
                q, k, v,
                p=self.attn_drop.p,
                attn_bias=attn_bias
            )
        else:
            q, k, v = map(lambda t: t.permute(0, 2, 1, 3),(q, k, v),)
            attn_mask = None
            if mask is not None and len(mask) > 1:

                # Create equivalent of xformer diagonal block mask, still only correct for square masks
                # But depth doesn't matter as tensors can expand in that dimension
                attn_mask_template = torch.ones(
                    [q.shape[2] // B, mask[0]],
                    dtype=torch.bool,
                    device=q.device
                )
                attn_mask = torch.block_diag(attn_mask_template)

                # create a mask on the diagonal for each mask in the batch
                for n in range(B - 1):
                    attn_mask = torch.block_diag(attn_mask, attn_mask_template)

            p = getattr(self.attn_drop, "p", 0) # IPEX.optimize() will turn attn_drop into an Identity()

            if sdpa_32b is not None and (q.element_size() * q.nelement()) > Q_4GB_LIMIT:
                sdpa = sdpa_32b
            else:
                sdpa = torch.nn.functional.scaled_dot_product_attention

            x = sdpa(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=p
            ).permute(0, 2, 1, 3).contiguous()
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LiteLA(torch.nn.Module): # from attention
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        heads = heads or int(out_dim // dim * heads_ratio)

        # assert dim % heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = operations.Linear(in_dim, in_dim * 3, bias=use_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(in_dim, in_dim, dtype=dtype, device=device)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads  # TODO: need some change
        self.eps = eps

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, 3, C --> B, N, C
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()


class PAGCFGIdentitySelfAttnProcessorLiteLA:
    r"""Self Attention with Perturbed Attention & CFG Guidance"""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        x_uncond, x_org, x_ptb = x.chunk(3)
        x_org = torch.cat([x_uncond, x_org])
        B, N, C = x_org.shape

        qkv = self.attn.qkv(x_org).reshape(B, N, 3, C)
        # B, N, 3, C --> B, N, C
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.attn.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.attn.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.attn.dim, self.attn.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)

        out = self.attn.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.attn.proj(out)

        # perturbed path (identity attention)
        v_weight = self.attn.qkv.weight[C * 2 : C * 3, :]  # Shape: (dim, dim)
        if self.attn.qkv.bias:
            v_bias = self.attn.qkv.bias[C * 2 : C * 3]  # Shape: (dim,)
            x_ptb = (torch.matmul(x_ptb, v_weight.t()) + v_bias).to(dtype)
        else:
            x_ptb = torch.matmul(x_ptb, v_weight.t()).to(dtype)
        x_ptb = self.attn.proj(x_ptb)

        out = torch.cat([out, x_ptb])

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


class PAGIdentitySelfAttnProcessorLiteLA:
    r"""Self Attention with Perturbed Attention Guidance"""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        x_org, x_ptb = x.chunk(2)
        B, N, C = x_org.shape

        qkv = self.attn.qkv(x_org).reshape(B, N, 3, C)
        # B, N, 3, C --> B, N, C
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.attn.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.attn.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.attn.dim, self.attn.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)

        out = self.attn.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.attn.proj(out)

        # perturbed path (identity attention)
        v_weight = self.attn.qkv.weight[C * 2 : C * 3, :]  # Shape: (dim, dim)
        if self.attn.qkv.bias:
            v_bias = self.attn.qkv.bias[C * 2 : C * 3]  # Shape: (dim,)
            x_ptb = (torch.matmul(x_ptb, v_weight.t()) + v_bias).to(dtype)
        else:
            x_ptb = torch.matmul(x_ptb, v_weight.t()).to(dtype)
        x_ptb = self.attn.proj(x_ptb)

        out = torch.cat([out, x_ptb])

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


class SelfAttnProcessorLiteLA:
    r"""Self Attention with Lite Linear Attention"""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW
        qkv = self.attn.qkv(x).reshape(B, N, 3, C)
        # B, N, 3, C --> B, N, C
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.attn.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.attn.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.attn.dim, self.attn.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)

        out = self.attn.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.attn.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


class FlashAttention(torch.nn.Module): # from attention
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, mask=None, HW=None, block_id=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if model_management.xformers_enabled():
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if mask is not None and mask.ndim == 2:
                mask = (1 - mask.to(x.dtype)) * -10000.0
                mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            x = x.clip(-65504, 65504)

        return x


#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################
class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        sampling='conv',
        sr_ratio=1,
        qk_norm=False,
        dtype=None,
        device=None,
        operations=None,
        **block_kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, HW=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B,N,3,H,C -> B,H,N,C
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q, k = q.float(), k.float()

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        #attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        #x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        dtype = x.dtype
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x.to(dtype))
        return x


class MaskFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(c_emb_size, 2 * final_hidden_size, bias=True, dtype=dtype, device=device)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, decoder_hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_decoder = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, decoder_hidden_size, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(dtype))
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0,
            dtype=dtype, device=device, operations=operations
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
        self.uncond_prob = uncond_prob

    def initialize_gemma_params(self, model_name="google/gemma-2b-it"):
        num_layers = len(self.custom_gemma_layers)
        text_encoder = AutoModelForCausalLM.from_pretrained(model_name).get_decoder()
        pretrained_layers = text_encoder.layers[-num_layers:]
        for custom_layer, pretrained_layer in zip(self.custom_gemma_layers, pretrained_layers):
            info = custom_layer.load_state_dict(pretrained_layer.state_dict(), strict=False)
            print(f"**** {info} ****")
        print(f"**** Initialized {num_layers} Gemma layers from pretrained model: {model_name} ****")

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None, mask=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)

        caption = self.y_proj(caption)

        return caption


class CaptionEmbedderDoubleBr(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate="tanh"), token_num=120):
        super().__init__()
        self.proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.embedding = nn.Parameter(torch.randn(1, in_channels) / 10**0.5)
        self.y_embedding = nn.Parameter(torch.randn(token_num, in_channels) / 10**0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, global_caption, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(global_caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        global_caption = torch.where(drop_ids[:, None], self.embedding, global_caption)
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return global_caption, caption

    def forward(self, caption, train, force_drop_ids=None):
        assert caption.shape[2:] == self.y_embedding.shape
        global_caption = caption.mean(dim=2).squeeze()
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            global_caption, caption = self.token_drop(global_caption, caption, force_drop_ids)
        y_embed = self.proj(global_caption)
        return y_embed, caption


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        self.proj = operations.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias, dtype=dtype, device=device
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        assert (W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbedMS(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        self.proj = operations.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias, dtype=dtype, device=device
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
