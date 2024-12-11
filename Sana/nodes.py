import torch
import folder_paths
from nodes import EmptyLatentImage

class EmptySanaLatentImage(EmptyLatentImage):
    CATEGORY = "ExtraModels/Sana"
    TITLE = "Empty Sana Latent Image"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 32, height // 32, width // 32], device=self.device)
        return ({"samples":latent}, )

class SanaTextEncode:
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
    CATEGORY = "ExtraModels/Sana"
    TITLE = "Sana Text Encode"

    def encode(self, text, GEMMA=None):
        tokenizer = GEMMA["tokenizer"]
        text_encoder = GEMMA["text_encoder"]
        
        with torch.no_grad():
            chi_prompt = "\n".join(preset_te_prompt)
            full_prompt = chi_prompt + text
            num_chi_tokens = len(tokenizer.encode(chi_prompt))
            max_length = num_chi_tokens + 300 - 2
            
            tokens = tokenizer(
                [full_prompt],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(text_encoder.device)
            
            select_idx = [0] + list(range(-300 + 1, 0))
            embs = text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][:, :, select_idx]
            emb_masks = tokens.attention_mask[:, select_idx]
        embs = embs * emb_masks.unsqueeze(-1)
            
        return ([[embs, {}]], )

preset_te_prompt = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
    '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
    'Here are examples of how to transform or refine prompts:',
    '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
    '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
    'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
    'User Prompt: '
]

NODE_CLASS_MAPPINGS = {
    "SanaTextEncode" : SanaTextEncode,
    "EmptySanaLatentImage": EmptySanaLatentImage,
}
