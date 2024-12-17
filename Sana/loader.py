import math
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
    latent_channels = 32
    def __init__(self):
        self.scale_factor = 0.41407
        self.latent_rgb_factors = latent_rgb_factors.copy()
        self.latent_rgb_factors_bias = latent_rgb_factors_bias.copy()

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
        return SanaModel(model_config=self, unet_model=self.unet_class, device=device)

class SanaModel(comfy.model_base.BaseModel):
    def __init__(self, *args, model_type=comfy.model_base.ModelType.FLOW, unet_model=SanaMS, **kwargs):
        super().__init__(*args, model_type=model_type, unet_model=unet_model, **kwargs)

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

    if "pos_embed" in sd:
        config["input_size"] = int(math.sqrt(sd["pos_embed"].shape[1])) * config["patch_size"]
    else:
        # TODO: this isn't optimal though most models don't use it
        config["use_pe"] = False

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

# for fast latent preview
latent_rgb_factors = [
    [-2.0022e-03, -6.0736e-03, -1.7096e-03],
    [ 1.4221e-03,  3.6703e-03,  4.1083e-03],
    [ 1.0081e-02,  2.6456e-04, -1.4333e-02],
    [-2.4253e-03,  3.0967e-03, -1.0301e-03],
    [ 2.2158e-03,  7.7274e-03, -1.3151e-02],
    [ 1.1235e-02,  5.7630e-03,  3.6146e-03],
    [-7.2899e-02,  1.1062e-02,  3.6103e-02],
    [ 3.2346e-02,  2.8678e-02,  2.5014e-02],
    [ 1.6469e-03, -1.1364e-03,  2.8366e-03],
    [-3.5597e-02, -2.3447e-02, -3.1172e-03],
    [-1.9985e-04, -2.0647e-03, -1.2702e-02],
    [ 2.1318e-04,  1.2196e-03, -8.3461e-04],
    [ 1.3766e-02,  2.7559e-03,  7.3567e-03],
    [ 1.3027e-02,  2.6365e-03,  3.0405e-03],
    [ 1.5335e-02,  9.4682e-03,  6.7312e-03],
    [ 5.1827e-03, -9.4865e-03,  8.5080e-03],
    [ 1.4365e-02, -3.2867e-03,  9.5108e-03],
    [-4.1216e-03, -1.9177e-03, -3.3726e-03],
    [-2.4757e-03,  5.1739e-04,  2.0280e-03],
    [-3.5950e-03,  1.0720e-03,  5.3043e-03],
    [-5.1758e-03,  8.1040e-03, -3.7564e-02],
    [-3.8555e-03, -1.5529e-03, -3.5799e-03],
    [-6.6175e-03, -6.8484e-03, -9.9609e-03],
    [-2.1656e-03,  5.5770e-05,  1.4936e-03],
    [-9.2857e-02, -1.1379e-01, -1.0919e-01],
    [ 7.7044e-04,  5.5594e-03,  3.4755e-02],
    [ 1.2714e-02,  2.9729e-02,  3.1989e-03],
    [-1.1805e-03,  9.0548e-03, -4.1063e-04],
    [ 8.3309e-04,  4.9694e-03,  2.3087e-03],
    [ 7.8456e-03,  3.9750e-03,  3.5655e-03],
    [-1.7552e-03,  4.9306e-03,  1.4210e-02],
    [-1.4790e-03,  2.8837e-03, -4.5687e-03]
]
latent_rgb_factors_bias = [0.4358, 0.3814, 0.3388]

# 512/1024/2K match, TODO: 4K is new, add on release
from ..PixArt.loader import resolutions as pixart_res
resolutions = {
    "Sana 512": pixart_res["PixArt 512"],
    "Sana 1024": pixart_res["PixArt 1024"],
    "Sana 2K": pixart_res["PixArt 2K"],
}
