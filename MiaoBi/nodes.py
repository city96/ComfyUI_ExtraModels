import os
import folder_paths
from . import sd
from . import diffusers_load
    
class MiaoBi_CLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "MiaoBi_CLIPLoader"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi_CLIP"

    def MiaoBi_CLIPLoader(self, clip_name):
        clip_type = sd.CLIPType.STABLE_DIFFUSION
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)
        return (clip,)
        
class MiaoBi_Diffusers:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "MiaoBi_Diffusers"
    CATEGORY = "ExtraModels/MiaoBi"
    TITLE = "MiaoBi_Diffusers"

    def MiaoBi_Diffusers(self, model_path, output_vae=True, output_clip=True):
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                # 拼接os.walk索引的前置路径和选择的文件夹名字，适用于本地。
                if os.path.exists(path):
                    model_path = path
                    break

        return diffusers_load.load_diffusers(model_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))

NODE_CLASS_MAPPINGS = {
	"MiaoBi_CLIP"  : MiaoBi_CLIP,
    "MiaoBi_Diffusers" :MiaoBi_Diffusers,
}