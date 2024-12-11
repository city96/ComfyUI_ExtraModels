import logging

import comfy.utils
import comfy.model_base
import comfy.model_detection

import comfy.supported_models_base
import comfy.supported_models
import comfy.latent_formats

from .models.sana import Sana
from .models.sana_multi_scale import SanaMS
from .diffusers_convert import convert_state_dict
from ..utils.loader import load_state_dict_from_config

class SanaLatent(comfy.latent_formats.LatentFormat):
    scale_factor = 0.41407
    latent_channels = 32

class SanaConfig(comfy.supported_models_base.BASE):
    unet_class = SanaMS
    unet_config = {}
    unet_extra_config = {}

    latent_format = SanaLatent
    sampling_settings = {
        "shift": 3.0,
    }

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.FLOW

    def get_model(self, state_dict, prefix="", device=None):
        return SanaModel(
            model_config=self,
            model_type=comfy.model_base.ModelType.FLOW,
            unet_model=self.unet_class,
            device=device
        )

class SanaModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def load_sana_state_dict(sd, model_options={}):
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
    # shared settings that match between all models
    # TODO: some can (should) be enumerated
    config = {
        "in_channels": 32,
        "linear_head_dim": 32,
        "model_max_length": 300,
        "y_norm": True,
        "attn_type": "linear",
        "ffn_type": "glumbconv",
        "mlp_ratio": 2.5,
        "mlp_acts": ["silu", "silu", None],
        "use_pe": False,
        "pred_sigma": False,
        "learn_sigma": False,
        "fp32_attention": True,
        "patch_size": 1,
    }
    config["depth"] = sum([key.endswith(".point_conv.conv.weight") for key in sd.keys()]) or 28

    if "x_embedder.proj.bias" in sd:
        config["hidden_size"] = sd["x_embedder.proj.bias"].shape[0]
    
    if config["hidden_size"] == 1152:
        config["num_heads"] = 16
    elif config["hidden_size"] == 2240:
        config["num_heads"] = 20
    else:
        raise RuntimeError(f"Unknown model config.")
    
    model_config = SanaConfig(config)
    logging.debug(f"Sana config:\n{config}")
    return model_config

# 512/1024/2K match, TODO: 4K is new, add on release
from ..PixArt.loader import resolutions as pixart_res
resolutions = {
    "Sana 512": pixart_res["PixArt 512"],
    "Sana 1024": pixart_res["PixArt 1024"],
    "Sana 2K": pixart_res["PixArt 2K"],
}
