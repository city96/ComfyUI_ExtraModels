import math
import logging

import comfy.utils
import comfy.model_base
import comfy.model_detection

import comfy.supported_models_base
import comfy.supported_models
import comfy.latent_formats

from .models.pixart import PixArt
from .models.pixartms import PixArtMS
from .diffusers_convert import convert_state_dict
from ..utils.loader import load_state_dict_from_config
from ..text_encoders.pixart.tenc import PixArtTokenizer, PixArtT5XXL

class PixArtConfig(comfy.supported_models_base.BASE):
    unet_class = PixArtMS
    unet_config = {}
    unet_extra_config = {}

    latent_format = comfy.latent_formats.SD15
    sampling_settings = {
        "beta_schedule" : "sqrt_linear",
        "linear_start"  : 0.0001,
        "linear_end"    : 0.02,
        "timesteps"     : 1000,
    }

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.EPS

    def get_model(self, state_dict, prefix="", device=None):
        return PixArtModel(model_config=self, unet_model=self.unet_class, device=device)

    def clip_target(self, state_dict={}):
        return comfy.supported_models_base.ClipTarget(PixArtTokenizer, PixArtT5XXL)

class PixArtModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        return out

def load_pixart_state_dict(sd, model_options={}):
    # prefix / format
    sd = sd.get("model", sd) # ref ckpt
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    # diffusers convert
    if "adaln_single.linear.weight" in sd:
        sd = convert_state_dict(sd)

    # model config
    model_config = model_config_from_unet(sd)
    return load_state_dict_from_config(model_config, sd, model_options)

def model_config_from_unet(sd):
    """
    Guess config based on (converted) state dict.
    """
    # Shared settings based on DiT_XL_2 - could be enumerated
    config = {
        "num_heads"   :   16, # get from attention
        "patch_size"  :    2, # final layer I guess?
        "hidden_size" : 1152, # pos_embed.shape[2]
    }
    config["depth"] = sum([key.endswith(".attn.proj.weight") for key in sd.keys()]) or 28

    try:
        # this is not present in the diffusers version for sigma?
        config["model_max_length"] = sd["y_embedder.y_embedding"].shape[0]
    except KeyError:
        # need better logic to guess this
        config["model_max_length"] = 300

    if "pos_embed" in sd:
        config["input_size"] = int(math.sqrt(sd["pos_embed"].shape[1])) * config["patch_size"]
        config["pe_interpolation"] = config["input_size"] // (512//8) # dumb guess

    model_config = PixArtModel
    if config["model_max_length"] == 300:
        # Sigma
        model_class = PixArtMS
        model_config.latent_format = comfy.latent_formats.SDXL
        config["micro_condition"] = False
        if "input_size" not in config:
            # The diffusers weights for 1K/2K are exactly the same...?
            # replace patch embed logic with HyDiT?
            logging.warn(f"PixArt: diffusers weights - 2K model will be broken, use manual loading!")
            config["input_size"] = 1024//8
    else:
        # Alpha
        if "csize_embedder.mlp.0.weight" in sd:
            # MS (microconds)
            model_class = PixArtMS
            config["micro_condition"] = True
            if "input_size" not in config:
                config["input_size"] = 1024//8
                config["pe_interpolation"] = 2
        else:
            # PixArt
            model_class = PixArt
            if "input_size" not in config:
                config["input_size"] = 512//8
                config["pe_interpolation"] = 1
    model_config = PixArtConfig(config)
    model_config.unet_class = model_class
    logging.debug(f"PixArt config: {model_class}\n{config}")
    return model_config

resolutions = {
    "PixArt 512": {
        0.25: [256,1024], 0.26: [256, 992], 0.27: [256, 960], 0.28: [256, 928],
        0.32: [288, 896], 0.33: [288, 864], 0.35: [288, 832], 0.40: [320, 800],
        0.42: [320, 768], 0.48: [352, 736], 0.50: [352, 704], 0.52: [352, 672],
        0.57: [384, 672], 0.60: [384, 640], 0.68: [416, 608], 0.72: [416, 576],
        0.78: [448, 576], 0.82: [448, 544], 0.88: [480, 544], 0.94: [480, 512],
        1.00: [512, 512], 1.07: [512, 480], 1.13: [544, 480], 1.21: [544, 448],
        1.29: [576, 448], 1.38: [576, 416], 1.46: [608, 416], 1.67: [640, 384],
        1.75: [672, 384], 2.00: [704, 352], 2.09: [736, 352], 2.40: [768, 320],
        2.50: [800, 320], 2.89: [832, 288], 3.00: [864, 288], 3.11: [896, 288],
        3.62: [928, 256], 3.75: [960, 256], 3.88: [992, 256], 4.00: [1024,256]
    },
    "PixArt 1024": {
        0.25: [512, 2048], 0.26: [512, 1984], 0.27: [512, 1920], 0.28: [512, 1856],
        0.32: [576, 1792], 0.33: [576, 1728], 0.35: [576, 1664], 0.40: [640, 1600],
        0.42: [640, 1536], 0.48: [704, 1472], 0.50: [704, 1408], 0.52: [704, 1344],
        0.57: [768, 1344], 0.60: [768, 1280], 0.68: [832, 1216], 0.72: [832, 1152],
        0.78: [896, 1152], 0.82: [896, 1088], 0.88: [960, 1088], 0.94: [960, 1024],
        1.00: [1024,1024], 1.07: [1024, 960], 1.13: [1088, 960], 1.21: [1088, 896],
        1.29: [1152, 896], 1.38: [1152, 832], 1.46: [1216, 832], 1.67: [1280, 768],
        1.75: [1344, 768], 2.00: [1408, 704], 2.09: [1472, 704], 2.40: [1536, 640],
        2.50: [1600, 640], 2.89: [1664, 576], 3.00: [1728, 576], 3.11: [1792, 576],
        3.62: [1856, 512], 3.75: [1920, 512], 3.88: [1984, 512], 4.00: [2048, 512],
    },
    "PixArt 2K": {
        0.25: [1024, 4096], 0.26: [1024, 3968], 0.27: [1024, 3840], 0.28: [1024, 3712],
        0.32: [1152, 3584], 0.33: [1152, 3456], 0.35: [1152, 3328], 0.40: [1280, 3200],
        0.42: [1280, 3072], 0.48: [1408, 2944], 0.50: [1408, 2816], 0.52: [1408, 2688],
        0.57: [1536, 2688], 0.60: [1536, 2560], 0.68: [1664, 2432], 0.72: [1664, 2304],
        0.78: [1792, 2304], 0.82: [1792, 2176], 0.88: [1920, 2176], 0.94: [1920, 2048],
        1.00: [2048, 2048], 1.07: [2048, 1920], 1.13: [2176, 1920], 1.21: [2176, 1792],
        1.29: [2304, 1792], 1.38: [2304, 1664], 1.46: [2432, 1664], 1.67: [2560, 1536],
        1.75: [2688, 1536], 2.00: [2816, 1408], 2.09: [2944, 1408], 2.40: [3072, 1280],
        2.50: [3200, 1280], 2.89: [3328, 1152], 3.00: [3456, 1152], 3.11: [3584, 1152],
        3.62: [3712, 1024], 3.75: [3840, 1024], 3.88: [3968, 1024], 4.00: [4096, 1024]
    }
}
