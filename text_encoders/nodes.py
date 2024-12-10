import folder_paths

from .tenc import load_text_encoder, tenc_names

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

NODE_CLASS_MAPPINGS = {
    "EXMCLIPLoader": EXMCLIPLoader,
}
