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

1. Download the model weights from the [DiT Repo](https://github.com/facebookresearch/DiT) or via this [direct link to the XL/2-512x512 model](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt)
2. Place them in `ComfyUI\models\dit` (created on first run after installing the extension)
3. Load the model and select the class labels as shown in the image below

Conditioning nodes *should* work for combining multiple labels. The area ones don't since the model currently can't handle dynamic input dimensions.

