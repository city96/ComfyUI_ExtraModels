import logging
from enum import Enum

import comfy.sd
import comfy.utils
import comfy.text_encoders
import comfy.model_management

from .pixart.tenc import pixart_te, PixArtTokenizer
from .sana.tenc import SanaClipModel, SanaTokenizer

class TencType(Enum):
    # offset in case we ever integrate w/ original
    PixArt = 1001
    MiaoBi = 1002
    # HunYuan = 1003 # deprecated
    Sana = 1004

tenc_names = {
    # for node readout
    "PixArt": TencType.PixArt,
    "MiaoBi": TencType.MiaoBi,
    # "HunYuan": TencType.HunYuan,
    "Sana": TencType.Sana,
}


def load_text_encoder(ckpt_paths, embedding_directory=None, clip_type=TencType.PixArt, model_options={}):
    # Partial duplicate of ComfyUI/comfy/sd:load_clip
    clip_data = []
    for p in ckpt_paths:
        if p.lower().endswith(".gguf"):
            # TODO: cross-node call w/o code duplication
            raise NotImplementedError("Planned!")
        else:
            clip_data.append(comfy.utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)

def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=TencType.PixArt, model_options={}):
    # Partial duplicate of ComfyUI/comfy/sd:load_text_encoder_state_dicts
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = comfy.utils.clip_text_transformers_convert(clip_data[i], "", "")
        elif "model.layers.25.post_feedforward_layernorm.weight" in clip_data[i]:
            clip_data[i] = {k[len("model."):]:v for k,v in clip_data[i].items()}
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1) #old models saved with the CLIPSave node
    
    clip_target = EmptyClass()
    clip_target.params = {}

    if clip_type == TencType.PixArt:
        clip_target.clip = pixart_te(**comfy.sd.t5xxl_detect(clip_data))
        clip_target.tokenizer = PixArtTokenizer
    elif clip_type == TencType.Sana:
        clip_target.clip = SanaClipModel
        clip_target.tokenizer = SanaTokenizer
    else:
        raise NotImplementedError(f"Unknown tenc: {clip_type}")

    parameters = 0
    tokenizer_data = {}
    for c in clip_data:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = comfy.sd.CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip
