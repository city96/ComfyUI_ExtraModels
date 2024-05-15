import os
import json
import torch
import folder_paths

from .conf import dit_conf
from .loader import load_dit

class HunYuanDitCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"model": (list(dit_conf.keys()),),
				"image_size": ([256, 512],),
				# "num_classes": ("INT", {"default": 1000, "min": 0,}),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/DiT"
	TITLE = "HunYuanDitCheckpointLoader"

	def load_checkpoint(self, ckpt_name, model, image_size):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		model_conf = dit_conf[model]
		model_conf["unet_config"]["input_size"]  = image_size // 8
		# model_conf["unet_config"]["num_classes"] = num_classes
		dit = load_dit(
			model_path = ckpt_path,
			model_conf = model_conf,
		)
		return (dit,)

NODE_CLASS_MAPPINGS = {
	"HunYuanDitCheckpointLoader" : HunYuanDitCheckpointLoader,
}
