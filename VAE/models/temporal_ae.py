import math
import torch
import numpy as np
from torch import nn
from typing import Callable, Iterable, Union, Optional
from einops import rearrange, repeat

from comfy import model_management
from .kl import (
	Encoder, Decoder, Upsample, Normalize,
	AttnBlock, ResnetBlock, #MemoryEfficientAttnBlock, 
	DiagonalGaussianDistribution, nonlinearity
)

class AutoencoderKL(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.embed_dim = config["embed_dim"]
		self.encoder = Encoder(**config)
		self.decoder = VideoDecoder(**config)
		assert config["double_z"]
		# these aren't used here for some reason
		# self.quant_conv = torch.nn.Conv2d(2*config["z_channels"], 2*self.embed_dim, 1)
		# self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, config["z_channels"], 1)

	def encode(self, x):
		## batched
		# n_samples = x.shape[0]
		# n_rounds = math.ceil(x.shape[0] / n_samples)
		# all_out = []
		# for n in range(n_rounds):
			# h = self.encoder(
				# x[n * n_samples : (n + 1) * n_samples]
			# )
			# moments = h # self.quant_conv(h)
			# posterior = DiagonalGaussianDistribution(moments)
			# all_out.append(posterior.sample())
		# z = torch.cat(all_out, dim=0)
		# return z

		## default
		h = self.encoder(x)
		moments = h # self.quant_conv(h)
		posterior = DiagonalGaussianDistribution(moments)
		return posterior.sample()


	def decode(self, z):
		## batched - seems the same as default?
		# n_samples = z.shape[0]
		# n_rounds = math.ceil(z.shape[0] / n_samples)
		# all_out = []
		# for n in range(n_rounds):
			# dec = self.decoder(
				# z[n * n_samples : (n + 1) * n_samples],
				# timesteps=len(z[n * n_samples : (n + 1) * n_samples]),
			# )
			# all_out.append(dec)
		# out = torch.cat(all_out, dim=0)

		## default
		out = self.decoder(
			z, timesteps=len(z)
		)
		return out

	def forward(self, input, sample_posterior=True):
		posterior = self.encode(input)
		if sample_posterior:
			z = posterior.sample()
		else:
			z = posterior.mode()
		dec = self.decode(z)
		return dec, posterior

class VideoDecoder(nn.Module):
	available_time_modes = ["all", "conv-only", "attn-only"]
	def __init__(
		self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
		attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
		resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
		attn_type="vanilla",
		video_kernel_size: Union[int, list] = 3, alpha: float = 0.0, merge_strategy: str = "learned", time_mode: str = "conv-only",
		**ignorekwargs
	):
		super().__init__()
		if use_linear_attn: attn_type = "linear"
		self.ch = ch
		self.temb_ch = 0
		self.num_resolutions = len(ch_mult)
		self.num_res_blocks = num_res_blocks
		self.resolution = resolution
		self.in_channels = in_channels
		self.give_pre_end = give_pre_end
		self.tanh_out = tanh_out

		self.video_kernel_size = video_kernel_size
		self.alpha = alpha
		self.merge_strategy = merge_strategy
		self.time_mode = time_mode
		assert (
			self.time_mode in self.available_time_modes
		), f"time_mode parameter has to be in {self.available_time_modes}"

		# compute in_ch_mult, block_in and curr_res at lowest res
		in_ch_mult = (1,)+tuple(ch_mult)
		block_in = ch*ch_mult[self.num_resolutions-1]
		curr_res = resolution // 2**(self.num_resolutions-1)
		self.z_shape = (1,z_channels,curr_res,curr_res)
		print("Working with z of shape {} = {} dimensions.".format(
			self.z_shape, np.prod(self.z_shape)))

		# z to block_in
		self.conv_in = torch.nn.Conv2d(
			z_channels,
			block_in,
			kernel_size=3,
			stride=1,
			padding=1
		)

		# middle
		self.mid = nn.Module()
		self.mid.block_1 = VideoResBlock(
			in_channels=block_in,
			out_channels=block_in,
			temb_channels=self.temb_ch,
			dropout=dropout,
			video_kernel_size=self.video_kernel_size,
			alpha=self.alpha,
			merge_strategy=self.merge_strategy,
		)
		self.mid.attn_1 = make_time_attn(
			block_in,
			attn_type=attn_type,
			alpha=self.alpha,
			merge_strategy=self.merge_strategy,
		)
		self.mid.block_2 = VideoResBlock(
			in_channels=block_in,
			out_channels=block_in,
			temb_channels=self.temb_ch,
			dropout=dropout,
			video_kernel_size=self.video_kernel_size,
			alpha=self.alpha,
			merge_strategy=self.merge_strategy,
		)

		# upsampling
		self.up = nn.ModuleList()
		for i_level in reversed(range(self.num_resolutions)):
			block = nn.ModuleList()
			attn = nn.ModuleList()
			block_out = ch*ch_mult[i_level]
			for i_block in range(self.num_res_blocks+1):
				block.append(VideoResBlock(
					in_channels=block_in,
					out_channels=block_out,
					temb_channels=self.temb_ch,
					dropout=dropout,
					video_kernel_size=self.video_kernel_size,
					alpha=self.alpha,
					merge_strategy=self.merge_strategy,
				))
				block_in = block_out
				if curr_res in attn_resolutions:
					attn.append(make_time_attn(
						block_in,
						attn_type=attn_type,
						alpha=self.alpha,
						merge_strategy=self.merge_strategy,
					))
			up = nn.Module()
			up.block = block
			up.attn = attn
			if i_level != 0:
				up.upsample = Upsample(block_in, resamp_with_conv)
				curr_res = curr_res * 2
			self.up.insert(0, up) # prepend to get consistent order

		# end
		self.norm_out = Normalize(block_in)
		self.conv_out = AE3DConv(
			in_channels = block_in,
			out_channels = out_ch,
			video_kernel_size=self.video_kernel_size,
			kernel_size=3,
			stride=1,
			padding=1,
		)

	def get_last_layer(self, skip_time_mix=False, **kwargs):
		if self.time_mode == "attn-only":
			raise NotImplementedError("TODO")
		else:
			return (
				self.conv_out.time_mix_conv.weight
				if not skip_time_mix
				else self.conv_out.weight
			)

	def forward(self, z, **kwargs):
		#assert z.shape[1:] == self.z_shape[1:]
		self.last_z_shape = z.shape

		# timestep embedding
		temb = None

		# z to block_in
		h = self.conv_in(z)

		# middle
		h = self.mid.block_1(h, temb, **kwargs)
		h = self.mid.attn_1(h, **kwargs)
		h = self.mid.block_2(h, temb, **kwargs)

		# upsampling
		for i_level in reversed(range(self.num_resolutions)):
			for i_block in range(self.num_res_blocks+1):
				h = self.up[i_level].block[i_block](h, temb, **kwargs)
				if len(self.up[i_level].attn) > 0:
					h = self.up[i_level].attn[i_block](h, **kwargs)
			if i_level != 0:
				h = self.up[i_level].upsample(h)

		# end
		if self.give_pre_end:
			return h

		h = self.norm_out(h)
		h = nonlinearity(h)
		h = self.conv_out(h, **kwargs)
		if self.tanh_out:
			h = torch.tanh(h)
		return h

class CrossAttention(nn.Module):
	def __init__(
		self,
		query_dim,
		context_dim=None,
		heads=8,
		dim_head=64,
		dropout=0.0,
		backend=None,
	):
		super().__init__()
		inner_dim = dim_head * heads
		context_dim = context_dim or query_dim

		self.scale = dim_head**-0.5
		self.heads = heads

		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
		self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
		self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
		)
		self.backend = backend

	def forward(
		self,
		x,
		context=None,
		mask=None,
		additional_tokens=None,
		n_times_crossframe_attn_in_self=0,
	):
		h = self.heads

		if additional_tokens is not None:
			# get the number of masked tokens at the beginning of the output sequence
			n_tokens_to_mask = additional_tokens.shape[1]
			# add additional token
			x = torch.cat([additional_tokens, x], dim=1)

		q = self.to_q(x)
		context = context or x
		k = self.to_k(context)
		v = self.to_v(context)

		if n_times_crossframe_attn_in_self:
			# reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
			assert x.shape[0] % n_times_crossframe_attn_in_self == 0
			n_cp = x.shape[0] // n_times_crossframe_attn_in_self
			k = repeat(
				k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
			)
			v = repeat(
				v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
			)

		q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

		## old
		"""
		sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
		del q, k

		if exists(mask):
			mask = rearrange(mask, 'b ... -> b (...)')
			max_neg_value = -torch.finfo(sim.dtype).max
			mask = repeat(mask, 'b j -> (b h) () j', h=h)
			sim.masked_fill_(~mask, max_neg_value)

		# some quote about attention, that I just about had enough of
		sim = sim.softmax(dim=-1)

		out = einsum('b i j, b j d -> b i d', sim, v)
		"""
		## new
		with sdp_kernel(**BACKEND_MAP[self.backend]):
			# print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
			out = F.scaled_dot_product_attention(
				q, k, v, attn_mask=mask
			)  # scale is dim_head ** -0.5 per default

		del q, k, v
		out = rearrange(out, "b h n d -> b n (h d)", h=h)

		if additional_tokens is not None:
			# remove additional token
			out = out[:, n_tokens_to_mask:]
		return self.to_out(out)

class VideoBlock(AttnBlock):
	def __init__(
		self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"
	):
		super().__init__(in_channels)
		# no context, single headed, as in base class
		self.time_mix_block = VideoTransformerBlock(
			dim=in_channels,
			n_heads=1,
			d_head=in_channels,
			checkpoint=False,
			ff_in=True,
			attn_mode="softmax",
		)

		time_embed_dim = self.in_channels * 4
		self.video_time_embed = torch.nn.Sequential(
			torch.nn.Linear(self.in_channels, time_embed_dim),
			torch.nn.SiLU(),
			torch.nn.Linear(time_embed_dim, self.in_channels),
		)

		self.merge_strategy = merge_strategy
		if self.merge_strategy == "fixed":
			self.register_buffer("mix_factor", torch.Tensor([alpha]))
		elif self.merge_strategy == "learned":
			self.register_parameter(
				"mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
			)
		else:
			raise ValueError(f"unknown merge strategy {self.merge_strategy}")

	def forward(self, x, timesteps, skip_video=False):
		if skip_video:
			return super().forward(x)

		x_in = x
		x = self.attention(x)
		h, w = x.shape[2:]
		x = rearrange(x, "b c h w -> b (h w) c")

		x_mix = x
		num_frames = torch.arange(timesteps, device=x.device)
		num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
		num_frames = rearrange(num_frames, "b t -> (b t)")
		t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
		emb = self.video_time_embed(t_emb)  # b, n_channels
		emb = emb[:, None, :]
		x_mix = x_mix + emb

		alpha = self.get_alpha()
		x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
		x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

		x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
		x = self.proj_out(x)

		return x_in + x

	def attention(self, h_: torch.Tensor) -> torch.Tensor:
		h_ = self.norm(h_)
		q = self.q(h_)
		k = self.k(h_)
		v = self.v(h_)

		b, c, h, w = q.shape
		q, k, v = map(
			lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
		)
		h_ = torch.nn.functional.scaled_dot_product_attention(
			q, k, v
		)  # scale is dim ** -0.5 per default
		# compute attention

		return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

	def forward(self, x, **kwargs):
		h_ = x
		h_ = self.attention(h_)
		h_ = self.proj_out(h_)
		return x + h_

	def get_alpha(
		self,
	):
		if self.merge_strategy == "fixed":
			return self.mix_factor
		elif self.merge_strategy == "learned":
			return torch.sigmoid(self.mix_factor)
		else:
			raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")

class VideoTransformerBlock(nn.Module):
	ATTENTION_MODES = {
		"softmax": CrossAttention,
		# "softmax-xformers": MemoryEfficientCrossAttention,
	}

	def __init__(
		self,
		dim,
		n_heads,
		d_head,
		dropout=0.0,
		context_dim=None,
		gated_ff=True,
		checkpoint=True,
		timesteps=None,
		ff_in=False,
		inner_dim=None,
		attn_mode="softmax",
		disable_self_attn=False,
		disable_temporal_crossattention=False,
		switch_temporal_ca_to_sa=False,
	):
		super().__init__()

		attn_cls = self.ATTENTION_MODES.get(attn_mode, "softmax")

		self.ff_in = ff_in or inner_dim is not None
		if inner_dim is None:
			inner_dim = dim

		assert int(n_heads * d_head) == inner_dim

		self.is_res = inner_dim == dim

		if self.ff_in:
			self.norm_in = nn.LayerNorm(dim)
			self.ff_in = FeedForward(
				dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
			)

		self.timesteps = timesteps
		self.disable_self_attn = disable_self_attn
		if self.disable_self_attn:
			self.attn1 = attn_cls(
				query_dim=inner_dim,
				heads=n_heads,
				dim_head=d_head,
				context_dim=context_dim,
				dropout=dropout,
			)  # is a cross-attention
		else:
			self.attn1 = attn_cls(
				query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
			)  # is a self-attention

		self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

		if disable_temporal_crossattention:
			if switch_temporal_ca_to_sa:
				raise ValueError
			else:
				self.attn2 = None
		else:
			self.norm2 = nn.LayerNorm(inner_dim)
			if switch_temporal_ca_to_sa:
				self.attn2 = attn_cls(
					query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
				)  # is a self-attention
			else:
				self.attn2 = attn_cls(
					query_dim=inner_dim,
					context_dim=context_dim,
					heads=n_heads,
					dim_head=d_head,
					dropout=dropout,
				)  # is self-attn if context is none

		self.norm1 = nn.LayerNorm(inner_dim)
		self.norm3 = nn.LayerNorm(inner_dim)
		self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

		self.checkpoint = checkpoint
		if self.checkpoint:
			print(f"{self.__class__.__name__} is using checkpointing")

	def forward(
		self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None
	) -> torch.Tensor:
		if self.checkpoint:
			return checkpoint(self._forward, x, context, timesteps)
		else:
			return self._forward(x, context, timesteps=timesteps)

	def _forward(self, x, context=None, timesteps=None):
		assert self.timesteps or timesteps
		assert not (self.timesteps and timesteps) or self.timesteps == timesteps
		timesteps = self.timesteps or timesteps
		B, S, C = x.shape
		x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

		if self.ff_in:
			x_skip = x
			x = self.ff_in(self.norm_in(x))
			if self.is_res:
				x += x_skip

		if self.disable_self_attn:
			x = self.attn1(self.norm1(x), context=context) + x
		else:
			x = self.attn1(self.norm1(x)) + x

		if self.attn2 is not None:
			if self.switch_temporal_ca_to_sa:
				x = self.attn2(self.norm2(x)) + x
			else:
				x = self.attn2(self.norm2(x), context=context) + x
		x_skip = x
		x = self.ff(self.norm3(x))
		if self.is_res:
			x += x_skip

		x = rearrange(
			x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
		)
		return x

	def get_last_layer(self):
		return self.ff.net[-1].weight

class ResBlock(nn.Module):
	"""
	A residual block that can optionally change the number of channels.
	:param channels: the number of input channels.
	:param emb_channels: the number of timestep embedding channels.
	:param dropout: the rate of dropout.
	:param out_channels: if specified, the number of out channels.
	:param use_conv: if True and out_channels is specified, use a spatial
		convolution instead of a smaller 1x1 convolution to change the
		channels in the skip connection.
	:param dims: determines if the signal is 1D, 2D, or 3D.
	:param use_checkpoint: if True, use gradient checkpointing on this module.
	:param up: if True, use this block for upsampling.
	:param down: if True, use this block for downsampling.
	"""

	def __init__(
		self,
		channels: int,
		emb_channels: int,
		dropout: float,
		out_channels: Optional[int] = None,
		use_conv: bool = False,
		use_scale_shift_norm: bool = False,
		dims: int = 2,
		use_checkpoint: bool = False,
		up: bool = False,
		down: bool = False,
		kernel_size: int = 3,
		exchange_temb_dims: bool = False,
		skip_t_emb: bool = False,
	):
		super().__init__()
		self.channels = channels
		self.emb_channels = emb_channels
		self.dropout = dropout
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.use_checkpoint = use_checkpoint
		self.use_scale_shift_norm = use_scale_shift_norm
		self.exchange_temb_dims = exchange_temb_dims

		if isinstance(kernel_size, Iterable):
			padding = [k // 2 for k in kernel_size]
		else:
			padding = kernel_size // 2

		self.in_layers = nn.Sequential(
			normalization(channels),
			nn.SiLU(),
			conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
		)

		self.updown = up or down

		if up:
			self.h_upd = Upsample(channels, False, dims)
			self.x_upd = Upsample(channels, False, dims)
		elif down:
			self.h_upd = Downsample(channels, False, dims)
			self.x_upd = Downsample(channels, False, dims)
		else:
			self.h_upd = self.x_upd = nn.Identity()

		self.skip_t_emb = skip_t_emb
		self.emb_out_channels = (
			2 * self.out_channels if use_scale_shift_norm else self.out_channels
		)
		if self.skip_t_emb:
			print(f"Skipping timestep embedding in {self.__class__.__name__}")
			assert not self.use_scale_shift_norm
			self.emb_layers = None
			self.exchange_temb_dims = False
		else:
			self.emb_layers = nn.Sequential(
				nn.SiLU(),
				linear(
					emb_channels,
					self.emb_out_channels,
				),
			)

		self.out_layers = nn.Sequential(
			normalization(self.out_channels),
			nn.SiLU(),
			nn.Dropout(p=dropout),
			zero_module(
				conv_nd(
					dims,
					self.out_channels,
					self.out_channels,
					kernel_size,
					padding=padding,
				)
			),
		)

		if self.out_channels == channels:
			self.skip_connection = nn.Identity()
		elif use_conv:
			self.skip_connection = conv_nd(
				dims, channels, self.out_channels, kernel_size, padding=padding
			)
		else:
			self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

	def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
		"""
		Apply the block to a Tensor, conditioned on a timestep embedding.
		:param x: an [N x C x ...] Tensor of features.
		:param emb: an [N x emb_channels] Tensor of timestep embeddings.
		:return: an [N x C x ...] Tensor of outputs.
		"""
		if self.use_checkpoint:
			return checkpoint(self._forward, x, emb)
		else:
			return self._forward(x, emb)

	def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
		if self.updown:
			in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
			h = in_rest(x)
			h = self.h_upd(h)
			x = self.x_upd(x)
			h = in_conv(h)
		else:
			h = self.in_layers(x)

		if self.skip_t_emb:
			emb_out = torch.zeros_like(h)
		else:
			emb_out = self.emb_layers(emb).type(h.dtype)
		while len(emb_out.shape) < len(h.shape):
			emb_out = emb_out[..., None]
		if self.use_scale_shift_norm:
			out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
			scale, shift = torch.chunk(emb_out, 2, dim=1)
			h = out_norm(h) * (1 + scale) + shift
			h = out_rest(h)
		else:
			if self.exchange_temb_dims:
				emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
			h = h + emb_out
			h = self.out_layers(h)
		return self.skip_connection(x) + h

class VideoResBlock(ResnetBlock):
	def __init__(
		self,
		out_channels,
		*args,
		dropout=0.0,
		video_kernel_size=3,
		alpha=0.0,
		merge_strategy="learned",
		**kwargs,
	):
		super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
		if video_kernel_size is None:
			video_kernel_size = [3, 1, 1]
		self.time_stack = ResBlock(
			channels=out_channels,
			emb_channels=0,
			dropout=dropout,
			dims=3,
			use_scale_shift_norm=False,
			use_conv=False,
			up=False,
			down=False,
			kernel_size=video_kernel_size,
			use_checkpoint=False,
			skip_t_emb=True,
		)

		self.merge_strategy = merge_strategy
		if self.merge_strategy == "fixed":
			self.register_buffer("mix_factor", torch.Tensor([alpha]))
		elif self.merge_strategy == "learned":
			self.register_parameter(
				"mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
			)
		else:
			raise ValueError(f"unknown merge strategy {self.merge_strategy}")

	def get_alpha(self, bs):
		if self.merge_strategy == "fixed":
			return self.mix_factor
		elif self.merge_strategy == "learned":
			return torch.sigmoid(self.mix_factor)
		else:
			raise NotImplementedError()

	def forward(self, x, temb, skip_video=False, timesteps=None):
		if timesteps is None:
			timesteps = self.timesteps

		b, c, h, w = x.shape

		x = super().forward(x, temb)

		if not skip_video:
			x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

			x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

			x = self.time_stack(x, temb)

			alpha = self.get_alpha(bs=b // timesteps)
			x = alpha * x + (1.0 - alpha) * x_mix

			x = rearrange(x, "b c t h w -> (b t) c h w")
		return x

class AE3DConv(torch.nn.Conv2d):
	def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
		super().__init__(in_channels, out_channels, *args, **kwargs)
		if isinstance(video_kernel_size, Iterable):
			padding = [int(k // 2) for k in video_kernel_size]
		else:
			padding = int(video_kernel_size // 2)

		self.time_mix_conv = torch.nn.Conv3d(
			in_channels=out_channels,
			out_channels=out_channels,
			kernel_size=video_kernel_size,
			padding=padding,
		)

	def forward(self, input, timesteps, skip_video=False):
		x = super().forward(input)
		if skip_video:
			return x
		x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
		x = self.time_mix_conv(x)
		return rearrange(x, "b c t h w -> (b t) c h w")

def make_time_attn(in_channels, attn_type="vanilla", attn_kwargs=None, alpha: float = 0, merge_strategy: str = "learned"):
	if attn_type == "vanilla":
		assert attn_kwargs is None
		return VideoBlock(
			in_channels,
			alpha=alpha,
			merge_strategy=merge_strategy,
		)
	# lazy to add the xformers code
	else:
		return NotImplementedError()

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
	"""
	Create sinusoidal timestep embeddings.
	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""
	if not repeat_only:
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period)
			* torch.arange(start=0, end=half, dtype=torch.float32)
			/ half
		).to(device=timesteps.device)
		args = timesteps[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat(
				[embedding, torch.zeros_like(embedding[:, :1])], dim=-1
			)
	else:
		embedding = repeat(timesteps, "b -> b d", d=dim)
	return embedding

def normalization(channels):
	"""
	Make a standard normalization layer.
	:param channels: number of input channels.
	:return: an nn.Module for normalization.
	"""
	return GroupNorm32(32, channels)

class SiLU(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
	def forward(self, x):
		return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
	"""
	Create a 1D, 2D, or 3D convolution module.
	"""
	if dims == 1:
		return nn.Conv1d(*args, **kwargs)
	elif dims == 2:
		return nn.Conv2d(*args, **kwargs)
	elif dims == 3:
		return nn.Conv3d(*args, **kwargs)
	raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module

# feedforward
class GEGLU(nn.Module):
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.proj = nn.Linear(dim_in, dim_out * 2)

	def forward(self, x):
		x, gate = self.proj(x).chunk(2, dim=-1)
		return x * nn.functional.gelu(gate)

class FeedForward(nn.Module):
	def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
		super().__init__()
		inner_dim = int(dim * mult)
		dim_out = dim_out or dim
		project_in = (
			nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
			if not glu
			else GEGLU(dim, inner_dim)
		)

		self.net = nn.Sequential(
			project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
		)

	def forward(self, x):
		return self.net(x)
