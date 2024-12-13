import os
import torch

from comfy import sd1_clip
import comfy.model_management
from .gemma import Gemma2Model

from transformers import GemmaTokenizer as TFGemmaTokenizer

class GemmaClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        special_tokens = {"start": 2, "end": 1, "pad": 0}
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens=special_tokens, model_class=Gemma2Model, enable_attention_masks=True, return_attention_masks=False, model_options=model_options)

class SanaClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="gemma", clip_model=GemmaClipModel, model_options=model_options)

class GemmaTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "tokenizers", "gemma_tokenizer",
        )
        # TODO: reenable proper logic here - needs comfy version 44db978 or higher
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=2304, embedding_key='gemma', tokenizer_class=TFGemmaTokenizer, has_start_token=False, pad_to_max_length=True, max_length=300, min_length=1)
        self.start_token = 2
        self.end_token = 1
        self.pad_token = 0

    def tokenize_with_weights(self, text, return_word_ids=False):
        # TODO: see above, this is still just a wrapper for now
        tokens = self.tokenizer(
            text,
            max_length=300,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        batched_tokens = [(x.item(), 1.0) for x in tokens.input_ids[0]]
        return [batched_tokens]

class SanaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="gemma", tokenizer=GemmaTokenizer)
