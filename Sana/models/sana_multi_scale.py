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
import torch
import torch.nn as nn

from .basic_modules import DWMlp, GLUMBConv, MBConvPreGLU, Mlp
from .sana import Sana, get_2d_sincos_pos_embed
from .sana_blocks import (
    Attention,
    CaptionEmbedder,
    TimestepEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    PatchEmbedMS,
    T2IFinalLayer,
    t2i_modulate,
)
from .norms import RMSNorm

class SanaMSBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        dtype=None,
        device=None,
        operations=None,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        if attn_type == "flash":
            # flash self attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                dtype=dtype,
                device=device,
                operations=operations,
                **block_kwargs,
            )
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(
                hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm,
                dtype=dtype, device=device, operations=operations,
            )
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=True, dtype=dtype, device=device, operations=operations,
            )
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, dtype=dtype, device=device, operations=operations, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0,
                dtype=dtype, device=device, operations=operations,
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0,
                dtype=dtype, device=device, operations=operations,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=mlp_acts,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = nn.Identity() # DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None].to(x.dtype) + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), HW=HW))

        return x


#############################################################################
#                                 Core Sana Model                                #
#################################################################################
class SanaMS(Sana):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False,
        pred_sigma=False,
        drop_path: float = 0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=300,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="linear",
        ffn_type="glumbconv",
        use_pe=False,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.y_norm = y_norm
        self.model_max_length = model_max_length
        self.fp32_attention = kwargs.get("use_fp32_attention", False)
        self.h = self.w = 0

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device)
        )

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device, operations=operations)

        self.pos_embed_ms = None
        if input_size is not None:
            self.base_size = input_size // self.patch_size
        else:
            self.base_size = None

        kernel_size = patch_embed_kernel or patch_size
        self.x_embedder = PatchEmbedMS(
            patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True,
            dtype=dtype, device=device, operations=operations,
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                SanaMSBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_size // patch_size, input_size // patch_size),
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(
            hidden_size, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations
        )

    def forward(self, x, timesteps, context, **kwargs):
        """
        Forward pass that adapts comfy input to original forward function
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (N,) tensor of diffusion timesteps
        context: (N, 1, 120, C) conditioning
        """
        ## size/ar from cond with fallback based on the latent image shape.
        bs = x.shape[0]
        ## Still accepts the input w/o that dim but returns garbage
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        ## run original forward pass
        out = self.forward_orig(
            x = x.to(self.dtype),
            timestep = timesteps.to(self.dtype),
            y = context.to(self.dtype),
        )

        ## only return EPS
        out = out.to(torch.float)
        
        return out

    def forward(self, x, timestep, context, mask=None, data_info=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        bs = x.shape[0]
        y = context
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if self.use_pe:
            x = self.x_embedder(x)
            if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x.shape[1:]:
                self.pos_embed_ms = (
                    torch.from_numpy(
                        get_2d_sincos_pos_embed(
                            self.pos_embed.shape[-1],
                            (self.h, self.w),
                            pe_interpolation=self.pe_interpolation,
                            base_size=self.base_size,
                        )
                    ).unsqueeze(0).to(x.device).to(x.dtype)
                )
            x += self.pos_embed_ms  # (N, T, D), where T = H * W / patch_size ** 2
        else:
            x = self.x_embedder(x)

        t = self.t_embedder(timestep, x.dtype)  # (N, D)

        y_lens = ((y != 0).sum(dim=3) > 0).sum(dim=2).squeeze().tolist()
        y_lens = [y_lens[1]] * bs

        mask = torch.zeros((len(y_lens), self.model_max_length), dtype=torch.int).to(x.device)
        for i, count in enumerate(y_lens):
            mask[i, :count] = 1

        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training, mask=mask)  # (N, D)
        if self.y_norm:
            y = self.attention_y_norm(y)

        y = y.squeeze(1).masked_select(mask.unsqueeze(-1).bool()).view(1, -1, y.shape[-1])

        for block in self.blocks:
            x = block(x, y, t0, y_lens, (self.h, self.w), **kwargs) # (N, T, D) #

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs
