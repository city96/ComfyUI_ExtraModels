import os
import json
import torch
import folder_paths

from comfy.model_management import get_torch_device, soft_empty_cache, text_encoder_offload_device
from comfy import utils
from .conf import sana_conf, sana_res
from .loader import load_sana
from ..utils.dtype import string_to_dtype

dtypes = [
	"auto",
	"FP32",
	"FP16",
	"BF16"
]

class SanaCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"model": (list(sana_conf.keys()),),
			}
		}
	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Checkpoint Loader"

	def load_checkpoint(self, ckpt_name, model):
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		model_conf = sana_conf[model]
		model = load_sana(
			model_path = ckpt_path,
			model_conf = model_conf,
		)
		return (model,)


class SanaResolutionSelect():
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": (list(sana_res.keys()),),
				"ratio": (list(sana_res["1024px"].keys()),{"default":"1.00"}),
			}
		}
	RETURN_TYPES = ("INT","INT")
	RETURN_NAMES = ("width","height")
	FUNCTION = "get_res"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Resolution Select"

	def get_res(self, model, ratio):
		width, height = sana_res[model][ratio]
		return (width,height)


class SanaResolutionCond:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"cond": ("CONDITIONING", ),
				"width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
				"height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	RETURN_NAMES = ("cond",)
	FUNCTION = "add_cond"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Resolution Conditioning"
	
	def add_cond(self, cond, width, height):
		for c in range(len(cond)):
			cond[c][1].update({
				"img_hw": [[height, width]],
				"aspect_ratio": [[height/width]],
			})
		return (cond,)


class SanaTextEncode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"text": ("STRING", {"multiline": True}),
				"preset_styles": (STYLE_NAMES,),
				"GEMMA": ("GEMMA",),
			}
		}

	RETURN_TYPES = ("CONDITIONING",)
	FUNCTION = "encode"
	CATEGORY = "ExtraModels/Sana"
	TITLE = "Sana Text Encode"

	def encode(self, text, preset_styles, GEMMA=None):
		tokenizer = GEMMA["tokenizer"]
		text_encoder = GEMMA["text_encoder"]
		
		# 应用预设样式 - 只使用正面提示词部分
		text, _ = apply_style(preset_styles, text)
		
		with torch.no_grad():
			# 处理正面提示词
			chi_prompt = "\n".join(preset_te_prompt)
			full_prompt = chi_prompt + text
			num_chi_tokens = len(tokenizer.encode(chi_prompt))
			max_length = num_chi_tokens + 300 - 2  # 减去[bos]和[_]标记
			
			tokens = tokenizer(
				[full_prompt],
				max_length=max_length,
				padding="max_length",
				truncation=True,
				return_tensors="pt"
			).to(text_encoder.device)
			
			select_idx = [0] + list(range(-300 + 1, 0))
			embs = text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][:, :, select_idx]
			emb_masks = tokens.attention_mask[:, select_idx]
		# 利用emb_masks将有效的embs选出来，其他置零
		embs = embs * emb_masks.unsqueeze(-1)
			
		return ([[embs, {}]], )

# 需要添加style相关的辅助函数
style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, "
        "glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, "
        "disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
	p, n = styles.get(style_name, styles[style_name])
	if not negative:
		negative = ""
	return p.replace("{prompt}", positive), n + negative

preset_te_prompt = ['Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:', '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: ']

NODE_CLASS_MAPPINGS = {
	"SanaCheckpointLoader" : SanaCheckpointLoader,
	"SanaResolutionSelect" : SanaResolutionSelect,
	"SanaTextEncode" : SanaTextEncode,
	"SanaResolutionCond" : SanaResolutionCond,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sana Checkpoint Loader": "SanaCheckpointLoader",
    "Sana Resolution Select": "SanaResolutionSelect",
    "Sana Text Encoder": "SanaTextEncode",
    "Sana Resolution Cond": "SanaResolutionCond",
}
