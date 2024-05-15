import os
import json
import torch
import folder_paths

from .conf import dit_conf
from .loader import load_dit
from .models.text_encoder import MT5Embedder
from transformers import BertModel, BertTokenizer

class MT5Loader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"HunyuanDiTfolder": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "HunyuanDiT"}),
				"device": (["cpu", "cuda"], {"default": "cuda"}),
			}
		}
	RETURN_TYPES = ("MT5","CLIP","Tokenizer",)
	FUNCTION = "load_model"
	CATEGORY = "ExtraModels/T5"
	TITLE = "MT5 Loader"

	def load_model(self, HunyuanDiTfolder, device):
		HunyuanDiTfolder=os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),HunyuanDiTfolder)
		mt5folder=os.path.join(HunyuanDiTfolder,"t2i/mt5")
		clipfolder=os.path.join(HunyuanDiTfolder,"t2i/clip_text_encoder")
		tokenizerfolder=os.path.join(HunyuanDiTfolder,"t2i/tokenizer")
		torch_dtype=torch.float16
		if device=="cpu":
			torch_dtype=torch.float32
		clip_text_encoder = BertModel.from_pretrained(str(clipfolder), False, revision=None).to(device)
		tokenizer = BertTokenizer.from_pretrained(str(tokenizerfolder))
		embedder_t5 = MT5Embedder(mt5folder, torch_dtype=torch_dtype, max_length=256, device=device)

		return (embedder_t5,clip_text_encoder,tokenizer,)

def clip_get_text_embeddings(clip_text_encoder,tokenizer,text,device):
	max_length=tokenizer.model_max_length
	text_inputs = tokenizer(
		text,
		padding="max_length",
		max_length=max_length,
		truncation=True,
		return_attention_mask=True,
		return_tensors="pt",
	)
	text_input_ids = text_inputs.input_ids
	attention_mask = text_inputs.attention_mask.to(device)
	prompt_embeds = clip_text_encoder(
		text_input_ids.to(device),
		attention_mask=attention_mask,
	)
	prompt_embeds = prompt_embeds[0]
	attention_mask = attention_mask.repeat(1, 1)

	return (prompt_embeds,attention_mask)

class MT5TextEncode:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"embedder_t5": ("MT5",),
				"clip_text_encoder": ("CLIP",),
				"tokenizer": ("Tokenizer",),
				"prompt": ("STRING", {"multiline": True}),
				"negative_prompt": ("STRING", {"multiline": True}),
			}
		}

	RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
	RETURN_NAMES = ("positive","negative",)
	FUNCTION = "encode"
	CATEGORY = "ExtraModels/T5"
	TITLE = "MT5 Text Encode"

	def encode(self, embedder_t5, clip_text_encoder, tokenizer, prompt, negative_prompt):
		print(f'prompt{prompt}')
		clip_prompt_embeds,clip_attention_mask = clip_get_text_embeddings(clip_text_encoder,tokenizer,prompt,embedder_t5.device)
		#clip_prompt_embeds=clip_prompt_embeds[0].permute(1,0)
		#clip_prompt_embeds=torch.cat((clip_prompt_embeds,clip_attention_mask))
		#print(f'clip_prompt_embeds{clip_prompt_embeds}')

		clip_negative_prompt_embeds,clip_negative_attention_mask = clip_get_text_embeddings(clip_text_encoder,tokenizer,negative_prompt,embedder_t5.device)
		#clip_negative_prompt_embeds=clip_negative_prompt_embeds[0].permute(1,0)
		#clip_negative_prompt_embeds=torch.cat((clip_negative_prompt_embeds,clip_negative_attention_mask))

		mt5_prompt_embeds,mt5_attention_mask = embedder_t5.get_text_embeddings(prompt)
		#mt5_prompt_embeds=mt5_prompt_embeds[0].permute(1,0)
		#mt5_prompt_embeds=torch.cat((mt5_prompt_embeds,mt5_attention_mask))

		mt5_negative_prompt_embeds,mt5_negative_attention_mask = embedder_t5.get_text_embeddings(negative_prompt)
		#mt5_negative_prompt_embeds=mt5_negative_prompt_embeds[0].permute(1,0)
		#mt5_negative_prompt_embeds=torch.cat((mt5_negative_prompt_embeds,mt5_negative_attention_mask))
		
		torch.save(clip_prompt_embeds,"/home/admin/ComfyUI/output/clip_prompt_embeds.pt")
		torch.save(clip_attention_mask,"/home/admin/ComfyUI/output/clip_attention_mask.pt")
		torch.save(clip_negative_prompt_embeds,"/home/admin/ComfyUI/output/clip_negative_prompt_embeds.pt")
		torch.save(clip_negative_attention_mask,"/home/admin/ComfyUI/output/clip_negative_attention_mask.pt")
		torch.save(mt5_prompt_embeds,"/home/admin/ComfyUI/output/mt5_prompt_embeds.pt")
		torch.save(mt5_attention_mask,"/home/admin/ComfyUI/output/mt5_attention_mask.pt")
		torch.save(mt5_negative_prompt_embeds,"/home/admin/ComfyUI/output/mt5_negative_prompt_embeds.pt")
		torch.save(mt5_negative_attention_mask,"/home/admin/ComfyUI/output/mt5_negative_attention_mask.pt")

		return ([[clip_prompt_embeds, {"clip_attention_mask":clip_attention_mask}]],[[clip_negative_prompt_embeds, {"clip_attention_mask":clip_negative_attention_mask}]], )

class HunYuanDitCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"HunyuanDiTfolder": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "HunyuanDiT"}),
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

	def load_checkpoint(self, HunyuanDiTfolder, model, image_size):
		HunyuanDiTfolder=os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),HunyuanDiTfolder)
		ckpt_path=os.path.join(HunyuanDiTfolder,"t2i/model/pytorch_model_ema.pt")
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
	"MT5Loader" : MT5Loader,
	"MT5TextEncode" : MT5TextEncode,
}
