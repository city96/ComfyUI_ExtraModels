class PixArtResolutionCond:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING", ),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "add_cond"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt Resolution Conditioning"
    
    def add_cond(self, cond, width, height):
        for c in range(len(cond)):
            cond[c][1].update({
                "img_hw": [[height, width]],
                "aspect_ratio": [[height/width]],
            })
        return (cond,)

class PixArtT5FromSD3CLIP:
    """
    Split the T5 text encoder away from SD3
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sd3_clip": ("CLIP",),
                "padding": ("INT", {"default": 1, "min": 1, "max": 300}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("t5",)
    FUNCTION = "split"
    CATEGORY = "ExtraModels/PixArt"
    TITLE = "PixArt T5 from SD3 CLIP"

    def split(self, sd3_clip, padding):
        try:
            from comfy.text_encoders.sd3_clip import SD3Tokenizer, SD3ClipModel
        except ImportError:
            # fallback for older ComfyUI versions
            from comfy.sd3_clip import SD3Tokenizer, SD3ClipModel # type: ignore
        import copy
    
        clip = sd3_clip.clone()
        assert clip.cond_stage_model.t5xxl is not None, "CLIP must have T5 loaded!"

        # remove transformer
        transformer = clip.cond_stage_model.t5xxl.transformer
        clip.cond_stage_model.t5xxl.transformer = None

        # clone object
        tmp = SD3ClipModel(clip_l=False, clip_g=False, t5=False)
        tmp.t5xxl = copy.deepcopy(clip.cond_stage_model.t5xxl)
        # put transformer back
        clip.cond_stage_model.t5xxl.transformer = transformer
        tmp.t5xxl.transformer = transformer

        # override special tokens
        tmp.t5xxl.special_tokens = copy.deepcopy(clip.cond_stage_model.t5xxl.special_tokens)
        tmp.t5xxl.special_tokens.pop("end") # make sure empty tokens match
        
        # add attn mask opt if present in original
        if hasattr(sd3_clip.cond_stage_model, "t5_attention_mask"):
            tmp.t5_attention_mask = False

        # tokenizer
        tok = SD3Tokenizer()
        tok.t5xxl.min_length = padding
        
        clip.cond_stage_model = tmp
        clip.tokenizer = tok

        return (clip, )

NODE_CLASS_MAPPINGS = {
    "PixArtResolutionCond" : PixArtResolutionCond,
    "PixArtT5FromSD3CLIP": PixArtT5FromSD3CLIP,
}
