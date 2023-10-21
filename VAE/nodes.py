import folder_paths

from .conf import vae_conf
from .loader import EXVAE

class ExtraVAELoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"vae_name": (folder_paths.get_filename_list("vae"),),
				"vae_type": (list(vae_conf.keys()), {"default":"kl-f8"}),
				"dtype"   : (["auto","fp32","fp16","bf16"],),
			}
		}
	RETURN_TYPES = ("VAE",)
	FUNCTION = "load_vae"
	CATEGORY = "ExtraModels"
	TITLE = "ExtraVAELoader"

	def load_vae(self, vae_name, vae_type, dtype):
		model_path = folder_paths.get_full_path("vae", vae_name)
		model_conf = vae_conf[vae_type]
		vae = EXVAE(model_path, model_conf, dtype)
		return (vae,)

NODE_CLASS_MAPPINGS = {
	"ExtraVAELoader" : ExtraVAELoader,
}
