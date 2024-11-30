import folder_paths
import torch
import comfy

from .conf import vae_conf
from .loader import EXVAE

from ..utils.dtype import string_to_dtype

dtypes = [
	"auto",
	"FP32",
	"FP16",
	"BF16"
]

MAX_RESOLUTION=16384

class ExtraVAELoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"vae_name": (folder_paths.get_filename_list("vae"),),
				"vae_type": (list(vae_conf.keys()), {"default":"kl-f8"}),
				"dtype"   : (dtypes,),
			}
		}
	RETURN_TYPES = ("VAE",)
	FUNCTION = "load_vae"
	CATEGORY = "ExtraModels"
	TITLE = "ExtraVAELoader"

	def load_vae(self, vae_name, vae_type, dtype):
		model_path = folder_paths.get_full_path("vae", vae_name)
		model_conf = vae_conf[vae_type]
		vae = EXVAE(model_path, model_conf, string_to_dtype(dtype, "vae"))
		return (vae,)


class EmptyDCAELatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"
    TITLE = "Empty DCAE Latent Image"

    CATEGORY = "latent"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 32, height // 32, width // 32], device=self.device)
        return ({"samples":latent}, )


NODE_CLASS_MAPPINGS = {
	"ExtraVAELoader" : ExtraVAELoader,
	"EmptyDCAELatentImage" : EmptyDCAELatentImage,
}
