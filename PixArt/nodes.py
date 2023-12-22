import os
import json
import torch
import folder_paths

from comfy import utils
from .conf import pixart_conf, pixart_res
from .lora import load_pixart_lora
from .loader import load_pixart
from .sampler import sample_pixart

class PixArtCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"model": (list(pixart_conf.keys()),),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Checkpoint Loader"

	def load_checkpoint(self, ckpt_name, model):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		model_conf = pixart_conf[model]
		model = load_pixart(
			model_path = ckpt_path,
			model_conf = model_conf,
		)
		return (model,)

class PixArtResolutionSelect():
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (list(pixart_conf.keys()),),
				# keys are the same for both
				"ratio": (list(pixart_res["PixArtMS_XL_2"].keys()),{"default":"1.00"}),
			}
		}
	RETURN_TYPES = ("INT","INT")
	RETURN_NAMES = ("width","height")
	FUNCTION = "get_res"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Resolution Select"

	def get_res(self, model, ratio):
		width, height = pixart_res[model][ratio]
		return (width,height)

class PixArtLoraLoader:
	def __init__(self):
		self.loaded_lora = None

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL",),
				"lora_name": (folder_paths.get_filename_list("loras"), ),
				"strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
			}
		}
	RETURN_TYPES = ("MODEL",)
	FUNCTION = "load_lora"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt Load LoRA"

	def load_lora(self, model, lora_name, strength,):
		if strength == 0:
			return (model)

		lora_path = folder_paths.get_full_path("loras", lora_name)
		lora = None
		if self.loaded_lora is not None:
			if self.loaded_lora[0] == lora_path:
				lora = self.loaded_lora[1]
			else:
				temp = self.loaded_lora
				self.loaded_lora = None
				del temp

		if lora is None:
			lora = utils.load_torch_file(lora_path, safe_load=True)
			self.loaded_lora = (lora_path, lora)

		model_lora = load_pixart_lora(model, lora, lora_path, strength,)
		return (model_lora,)

class PixArtDPMSampler:
	"""
	The sampler from the reference code.
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("MODEL", ),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
				"noise_schedule": (["linear","squaredcos_cap_v2"],{"default":"linear"}),
				"noise_schedule_vp": (["linear","discrete"],{"default":"discrete"}),
				"positive": ("CONDITIONING", ),
				"negative": ("CONDITIONING", ),
				"latent_image": ("LATENT", ),
			}
		}
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "sample"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt DPM Sampler [Reference]"

	def sample(self, model, seed, steps, cfg, noise_schedule, noise_schedule_vp, positive, negative, latent_image):
		samples = sample_pixart(
			model    = model,
			seed     = seed,
			steps    = steps,
			cfg      = cfg,
			positive = positive,
			negative = negative,
			latent_image = latent_image["samples"],
			noise_schedule = noise_schedule,
			noise_schedule_vp = noise_schedule_vp,
		)
		return ({"samples":samples},)

class PixArtT5TextEncode:
	"""
	Reference code, mostly to verify compatibility.
	 Once everything works, this should instead inherit from the
	 T5 text encode node and simply add the extra conds (res/ar).
	"""
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"text": ("STRING", {"multiline": True}),
				"T5": ("T5",),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	FUNCTION = "encode"
	CATEGORY = "ExtraModels/PixArt"
	TITLE = "PixArt T5 Text Encode [Reference]"

	def mask_feature(self, emb, mask):
		if emb.shape[0] == 1:
			keep_index = mask.sum().item()
			return emb[:, :, :keep_index, :], keep_index
		else:
			masked_feature = emb * mask[:, None, :, None]
			return masked_feature, emb.shape[2]

	def encode(self, text, T5):
		text = text.lower().strip()
		tokenizer_out = T5.tokenizer.tokenizer(
			text,
			max_length            = 120,
			padding               = 'max_length',
			truncation            = True,
			return_attention_mask = True,
			add_special_tokens    = True,
			return_tensors        = 'pt'
		)
		tokens = tokenizer_out["input_ids"]
		mask = tokenizer_out["attention_mask"]
		embs = T5.cond_stage_model.transformer(
			input_ids      = tokens.to(T5.load_device),
			attention_mask = mask.to(T5.load_device),
		)['last_hidden_state'].float()[:, None]
		masked_embs, keep_index = self.mask_feature(
			embs.detach().to("cpu"),
			mask.detach().to("cpu")
		)
		masked_embs = masked_embs.squeeze(0) # match CLIP/internal
		print("Encoded T5:", masked_embs.shape)
		return ([[masked_embs, {}]], )

NODE_CLASS_MAPPINGS = {
	"PixArtCheckpointLoader" : PixArtCheckpointLoader,
	"PixArtResolutionSelect" : PixArtResolutionSelect,
	"PixArtLoraLoader" : PixArtLoraLoader,
	"PixArtDPMSampler" : PixArtDPMSampler,
	"PixArtT5TextEncode" : PixArtT5TextEncode,
}
