import os
import torch
import folder_paths
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.dtype import string_to_dtype
from huggingface_hub import snapshot_download


tenc_root = (
    folder_paths.folder_names_and_paths.get(
        "text_encoders",
        folder_paths.folder_names_and_paths.get("clip", [[], set()])
    )
)

dtypes = [
    "default",
    "auto (comfy)",
    "BF16",
    "FP32",
    "FP16",
]
try: torch.float8_e5m2
except AttributeError: print("Torch版本过旧,不支持FP8")
else: dtypes += ["FP8 E4M3", "FP8 E5M2"]

class GemmaLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["auto", "cpu", "cuda"]
        # 支持多GPU
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "model_name": (["Efficient-Large-Model/gemma-2-2b-it", "google/gemma-2-2b-it", "unsloth/gemma-2-2b-it-bnb-4bit"],),
                "device": (devices, {"default":"cpu"}),
                "dtype": (dtypes,),
            }
        }
    RETURN_TYPES = ("GEMMA",)
    FUNCTION = "load_model"
    CATEGORY = "ExtraModels/Gemma" 
    TITLE = "Gemma Loader"

    def load_model(self, model_name, device, dtype):
        dtype = string_to_dtype(dtype, "text_encoder")
        if device == "cpu":
            assert dtype in [None, torch.float32], f"Can't use dtype '{dtype}' with CPU! Set dtype to 'default'."

        if model_name == 'google/gemma-2-2b-it':
            text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', 'models--google--gemma-2-2b-it')
            if not os.path.exists(os.path.join(text_encoder_dir, 'model.safetensors')):
                snapshot_download('google/gemma-2-2b-it', local_dir=text_encoder_dir)
        elif model_name == 'unsloth/gemma-2-2b-it-bnb-4bit':
            text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', 'models--unsloth--gemma-2-2b-it-bnb-4bit')
            if not os.path.exists(os.path.join(text_encoder_dir, 'model.safetensors')):
                snapshot_download('unsloth/gemma-2-2b-it-bnb-4bit', local_dir=text_encoder_dir)
        elif model_name == 'Efficient-Large-Model/gemma-2-2b-it':
            text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', 'models--Efficient-Large-Model--gemma-2-2b-it')
            if not os.path.exists(os.path.join(text_encoder_dir, 'model.safetensors')):
                snapshot_download('Efficient-Large-Model/gemma-2-2b-it', local_dir=text_encoder_dir)
        else:
            raise ValueError('Not implemented!')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        tokenizer.padding_side = "right"
        text_encoder = text_encoder_model.get_decoder()

        if device != "cpu":
            text_encoder = text_encoder.to(device)

        return ({
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "text_encoder_model": text_encoder_model
        },)


class GemmaTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "GEMMA": ("GEMMA",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "ExtraModels/Gemma"
    TITLE = "Gemma Text Encode"

    def encode(self, text, GEMMA=None):
        print(text)
        tokenizer = GEMMA["tokenizer"]
        text_encoder = GEMMA["text_encoder"]
        
        with torch.no_grad():
            tokens = tokenizer(
                text,
                max_length=300,
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            ).to(text_encoder.device)
            
            cond = text_encoder(tokens.input_ids, tokens.attention_mask)[0]
            emb_masks = tokens.attention_mask
            
        cond = cond * emb_masks.unsqueeze(-1)

        return ([[cond, {}]], )

NODE_CLASS_MAPPINGS = {
    "GemmaLoader": GemmaLoader,
    "GemmaTextEncode": GemmaTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaLoader": "Gemma Loader",
    "GemmaTextEncode": "Gemma Text Encode",
} 
