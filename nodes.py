import folder_paths
import comfy.utils

from .PixArt.loader import load_pixart_state_dict

loaders = {
    "PixArt": load_pixart_state_dict,
}

class EXMUnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "model_type": (list(loaders.keys()),)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "ExtraModels"
    TITLE = "Load Diffusion Model (ExtraModels)"

    def load_unet(self, unet_name, model_type):
        model_options = {}
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        loader_fn = loaders[model_type]
        sd = comfy.utils.load_torch_file(unet_path)
        return (loader_fn(sd),)

NODE_CLASS_MAPPINGS = {
    "EXMUnetLoader": EXMUnetLoader,
}
