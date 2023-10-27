import os
import json
import torch
import folder_paths

from .loader import load_t5

# initialize custom folder path
# TODO: integrate with `extra_model_paths.yaml`
os.makedirs(
	os.path.join(folder_paths.models_dir,"t5"),
	exist_ok = True,
)
folder_paths.folder_names_and_paths["t5"] = (
	[os.path.join(folder_paths.models_dir,"t5")],
	folder_paths.supported_pt_extensions
)

class T5v11Loader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"t5v11_name": (folder_paths.get_filename_list("t5"),),
				"t5v11_ver": (["xxl"],),
				"path_type": (["folder", "file"],),
				"device": (["auto", "cpu", "bnb8bit", "bnb4bit"],{"default":"cpu"})
			}
		}
	RETURN_TYPES = ("T5",)
	FUNCTION = "load_model"
	CATEGORY = "ExtraModels/T5"
	TITLE = "T5v1.1 Loader"

	def load_model(self, t5v11_name, t5v11_ver, path_type, device):
		return (load_t5(
			model_type = "t5v11",
			model_ver  = t5v11_ver,
			model_path = folder_paths.get_full_path("t5", t5v11_name),
			path_type  = path_type,
			device     = device,
		),)

class T5TextEncode:
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
	CATEGORY = "ExtraModels/T5"
	TITLE = "T5 Text Encode"

	def encode(self, text, T5=None):
		tokens = T5.tokenize(text)
		cond = T5.encode_from_tokens(tokens)
		return ([[cond, {}]], )

NODE_CLASS_MAPPINGS = {
	"T5v11Loader"  : T5v11Loader,
	"T5TextEncode" : T5TextEncode,
}
