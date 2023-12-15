# For using the diffusers format weights
#  Based on the original ComfyUI function + 
#  https://github.com/PixArt-alpha/PixArt-alpha/blob/master/tools/convert_pixart_alpha_to_diffusers.py
import torch

conversion_map = [ # main SD conversion map (PixArt reference, HF Diffusers)
	# Patch embeddings
	("x_embedder.proj.weight", "pos_embed.proj.weight"),
	("x_embedder.proj.bias",   "pos_embed.proj.bias"),
	# Caption projection
	("y_embedder.y_embedding", "caption_projection.y_embedding"),
	("y_embedder.y_proj.fc1.weight", "caption_projection.linear_1.weight"),
	("y_embedder.y_proj.fc1.bias",   "caption_projection.linear_1.bias"),
	("y_embedder.y_proj.fc2.weight", "caption_projection.linear_2.weight"),
	("y_embedder.y_proj.fc2.bias",   "caption_projection.linear_2.bias"),
	# AdaLN-single LN
	("t_embedder.mlp.0.weight", "adaln_single.emb.timestep_embedder.linear_1.weight"),
	("t_embedder.mlp.0.bias",   "adaln_single.emb.timestep_embedder.linear_1.bias"),
	("t_embedder.mlp.2.weight", "adaln_single.emb.timestep_embedder.linear_2.weight"),
	("t_embedder.mlp.2.bias",   "adaln_single.emb.timestep_embedder.linear_2.bias"),
	# Shared norm
	("t_block.1.weight", "adaln_single.linear.weight"),
	("t_block.1.bias",   "adaln_single.linear.bias"),
	# Final block
	("final_layer.linear.weight", "proj_out.weight"),
	("final_layer.linear.bias",   "proj_out.bias"),
	("final_layer.scale_shift_table", "scale_shift_table"),
]

conversion_map_ms = [ # for multi_scale_train (MS)
	# Resolution
	("csize_embedder.mlp.0.weight", "adaln_single.emb.resolution_embedder.linear_1.weight"),
	("csize_embedder.mlp.0.bias",   "adaln_single.emb.resolution_embedder.linear_1.bias"),
	("csize_embedder.mlp.2.weight", "adaln_single.emb.resolution_embedder.linear_2.weight"),
	("csize_embedder.mlp.2.bias",   "adaln_single.emb.resolution_embedder.linear_2.bias"),
	# Aspect ratio
	("ar_embedder.mlp.0.weight", "adaln_single.emb.aspect_ratio_embedder.linear_1.weight"),
	("ar_embedder.mlp.0.bias",   "adaln_single.emb.aspect_ratio_embedder.linear_1.bias"),
	("ar_embedder.mlp.2.weight", "adaln_single.emb.aspect_ratio_embedder.linear_2.weight"),
	("ar_embedder.mlp.2.bias",   "adaln_single.emb.aspect_ratio_embedder.linear_2.bias"),
]

# Add actual transformer blocks
for depth in range(28):
	# Transformer blocks
	conversion_map += [
		(f"blocks.{depth}.scale_shift_table", f"transformer_blocks.{depth}.scale_shift_table"),
		# Projection
		(f"blocks.{depth}.attn.proj.weight", f"transformer_blocks.{depth}.attn1.to_out.0.weight"),
		(f"blocks.{depth}.attn.proj.bias",   f"transformer_blocks.{depth}.attn1.to_out.0.bias"),
		# Feed-forward
		(f"blocks.{depth}.mlp.fc1.weight", f"transformer_blocks.{depth}.ff.net.0.proj.weight"),
		(f"blocks.{depth}.mlp.fc1.bias",   f"transformer_blocks.{depth}.ff.net.0.proj.bias"),
		(f"blocks.{depth}.mlp.fc2.weight", f"transformer_blocks.{depth}.ff.net.2.weight"),
		(f"blocks.{depth}.mlp.fc2.bias",   f"transformer_blocks.{depth}.ff.net.2.bias"),
		# Cross-attention (proj)
		(f"blocks.{depth}.cross_attn.proj.weight" ,f"transformer_blocks.{depth}.attn2.to_out.0.weight"),
		(f"blocks.{depth}.cross_attn.proj.bias"   ,f"transformer_blocks.{depth}.attn2.to_out.0.bias"),
	]

def convert_pixart_state_dict(unet_state_dict):
	if "adaln_single.emb.resolution_embedder.linear_1.weight" in unet_state_dict.keys():
		cmap = conversion_map + conversion_map_ms
	else:
		cmap = conversion_map

	new_state_dict = {k: unet_state_dict.pop(v) for k,v in cmap}
	
	for depth in range(28):
		# Self Attention
		q = unet_state_dict.pop(f"transformer_blocks.{depth}.attn1.to_q.weight")
		k = unet_state_dict.pop(f"transformer_blocks.{depth}.attn1.to_k.weight")
		v = unet_state_dict.pop(f"transformer_blocks.{depth}.attn1.to_v.weight")
		new_state_dict[f"blocks.{depth}.attn.qkv.weight"] = torch.cat((q,k,v), dim=0)
		qb = unet_state_dict.pop(f"transformer_blocks.{depth}.attn1.to_q.bias")
		kb = unet_state_dict.pop(f"transformer_blocks.{depth}.attn1.to_k.bias")
		vb = unet_state_dict.pop(f"transformer_blocks.{depth}.attn1.to_v.bias")
		new_state_dict[f"blocks.{depth}.attn.qkv.bias"] = torch.cat((qb,kb,vb), dim=0)

		# Cross-attention (linear)
		q = unet_state_dict.pop(f"transformer_blocks.{depth}.attn2.to_q.weight")
		k = unet_state_dict.pop(f"transformer_blocks.{depth}.attn2.to_k.weight")
		v = unet_state_dict.pop(f"transformer_blocks.{depth}.attn2.to_v.weight")
		new_state_dict[f"blocks.{depth}.cross_attn.q_linear.weight"] = q
		new_state_dict[f"blocks.{depth}.cross_attn.kv_linear.weight"] = torch.cat((k,v), dim=0)
		qb = unet_state_dict.pop(f"transformer_blocks.{depth}.attn2.to_q.bias")
		kb = unet_state_dict.pop(f"transformer_blocks.{depth}.attn2.to_k.bias")
		vb = unet_state_dict.pop(f"transformer_blocks.{depth}.attn2.to_v.bias")
		new_state_dict[f"blocks.{depth}.cross_attn.q_linear.bias"] = qb
		new_state_dict[f"blocks.{depth}.cross_attn.kv_linear.bias"] = torch.cat((kb,vb), dim=0)

	if len(unet_state_dict.keys()) > 0:
		print(f"PixArt: UNET conversion has leftover keys!:\n{unet_state_dict.keys()}")
	
	return new_state_dict
