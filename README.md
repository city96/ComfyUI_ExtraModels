# Extra Models for ComfyUI

This repository aims to add support for various random image diffusion models to ComfyUI.

## Installation

Simply clone this repo to your custom_nodes folder using the following command: `git clone https://github.com/city96/ComfyUI_ExtraModels custom_nodes/ComfyUI_ExtraModels`.

## DiT
### Model info / implementation
- Uses class labels instead of prompts
- Limited to 256x256 or 512x512 images
- Same latent space as SD1.5 (works with the SD1.5 VAE)
- Works in FP16, but no other optimization (yet)

### Usage

1. Download the original model weights from the [DiT Repo](https://github.com/facebookresearch/DiT) or the converted [FP16 safetensor ones from Huggingface](https://huggingface.co/city96/DiT/tree/main).
2. Place them in `ComfyUI\models\dit` (created on first run after installing the extension)
3. Load the model and select the class labels as shown in the image below
4. **Make sure to use the Empty label conditioning for the Negative input of the KSampler!**

ConditioningCombine nodes *should* work for combining multiple labels. The area ones don't since the model currently can't handle dynamic input dimensions.

[Image with sample workflow](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/33bfb812-23ea-4bb0-b1e2-082756e53010)

![DIT_WORKFLOW_IMG](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/cdd4ec94-b0eb-436a-bf23-a3bcef8d7b90)
