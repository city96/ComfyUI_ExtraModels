import folder_paths
import comfy.utils

from .PixArt.loader import load_pixart_state_dict
from .Sana.loader import load_sana_state_dict
from .text_encoders.tenc import load_text_encoder, tenc_names

loaders = {
    "PixArt": load_pixart_state_dict,
    "Sana": load_sana_state_dict,
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

class EXMCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        # if "clip_gguf" in folder_paths.folder_names_and_paths:
        #     files += folder_paths.get_filename_list("clip_gguf")
        return {
            "required": {
                "clip_name": (files, ),
                "type": (["PixArt", "MiaoBi", "Sana"],),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "ExtraModels"
    TITLE = "CLIPLoader (ExtraModels)"

    def load_clip(self, clip_name, type):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = tenc_names.get(type, None)

        clip = load_text_encoder(
            ckpt_paths =[clip_path],
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
            clip_type = clip_type
        )
        return (clip,)

#class EXMResolutionSelect:

NODE_CLASS_MAPPINGS = {
    "EXMUnetLoader": EXMUnetLoader,
    "EXMCLIPLoader": EXMCLIPLoader,
}
