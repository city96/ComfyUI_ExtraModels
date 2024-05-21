import folder_paths
from . import sd
    
class MiaoBi_CLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "MiaoBi_CLIPLoader"
    CATEGORY = "ExtraModels//MiaoBi"
    TITLE = "MiaoBi_CLIP"

    def MiaoBi_CLIPLoader(self, clip_name):
        clip_type = sd.CLIPType.STABLE_DIFFUSION
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = sd.load_clip(ckpt_paths=[clip_path], clip_type=clip_type)
        return (clip,)
        
NODE_CLASS_MAPPINGS = {
	"MiaoBi_CLIP"  : MiaoBi_CLIP,
}