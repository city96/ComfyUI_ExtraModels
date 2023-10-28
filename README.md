# Extra Models for ComfyUI

This repository aims to add support for various random image diffusion models to ComfyUI.

## Installation

Simply clone this repo to your custom_nodes folder using the following command: `git clone https://github.com/city96/ComfyUI_ExtraModels custom_nodes/ComfyUI_ExtraModels`.

### Portable install

I haven't tested this completely, so if you know what you're doing, use the regular VEVN/`git clone` install option when installing ComfyUI.

Go to the where you unpacked `ComfyUI_windows_portable` to (where your run_nvidia_gpu.bat file is) and open a command line window. Press `CTRL+SHIFT+Right click` in an empty space and click "Open PowerShell window here".

In case you haven't installed in through the manager, run `git clone https://github.com/city96/ComfyUI_ExtraModels .\ComfyUI\custom_nodes\ComfyUI_ExtraModels`

To install the requirements on windows, run these commands in the same window:
```
.\python_embeded\python.exe -m pip install -r  .\ComfyUI\custom_nodes\ComfyUI_ExtraModels\requirements.txt
.\python_embeded\python.exe -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

To update, open the command line window like before and run the following commands:

```
cd .\ComfyUI\custom_nodes\ComfyUI_ExtraModels\
git pull
```

Alternatively, use the manager, assuming it has an update function.


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


## PixArt

This is mostly a proof of concept, as the model weights have not been released officially. [Sample workflow here](https://github.com/city96/ComfyUI_ExtraModels/files/13192747/PixArt.json)

Make sure to `pip install timm==0.6.13`. xformers is optional but strongly recommended as torch SDP is only partially implemented, if that.

Limitations:
- The default `KSampler` uses a different noise schedule/sampling algo (I think), so it most likely won't work as expected.
- `PixArt DPM Sampler` requires the negative prompt to be shorter than the positive prompt.
- `PixArt DPM Sampler` can only work with a batch size of 1.
- `PixArt T5 Text Encode` is from the reference implementation, therefore it doesn't support weights. `T5 Text Encode` support weights, but I can't attest to the correctness of the implementation.

PixArt uses the same T5v1.1-xxl text encoder as DeepFloyd, so the T5 section of the readme also applies.


## T5
### Model

The model files can be downloaded from the [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main) repository.

You will need to download the following 4 files:
 - `config.json`
 - `pytorch_model-00001-of-00002.bin`
 - `pytorch_model-00002-of-00002.bin`
 - `pytorch_model.bin.index.json`

Place them in your `ComfyUI/models/t5` folder. You can put them in a subfolder called "t5-v1.1-xxl" though it doesn't matter. There are int8 safetensor files in the other DeepFloyd repo, thought they didn't work for me.

### Usage

Loaded onto the CPU, it'll use about 22GBs of system RAM. Depending on which weights you use, it might use slightly more during loading.

Loaded in bnb4bit mode, it only takes around 6GB VRAM, making it work with 12GB cards. The only drawback is that it'll constantly stay in VRAM since BitsAndBytes doesn't allow moving the weights to the system RAM temporarily. Switching to a different workflow *should* still release the VRAM as expected. Pascal cards (1080ti, P40) seem to struggle with 4bit. Select "cpu" if you encounter issues.

On windows, you may need a newer version of bitsandbytes for 4bit. Try `python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui`

You may also need to upgrade transformers and install spiece for the tokenizer. `pip install -r requirements.txt`


## VAE

A few custom VAE models are supported. The option to select a different dtype when loading is also possible, which can be useful for testing/comparisons.

### AutoencoderKL / VQModel

`kl-f4/8/16/32` from the [compvis/latent diffusion repo](https://github.com/CompVis/latent-diffusion/tree/main#pretrained-autoencoding-models).

`vq-f4/8/16` from the taming transformers repo, weights for both vq and kl models available [here](https://ommer-lab.com/files/latent-diffusion/)

`vq-f8` can accepts latents from the SD unet but just like xl with v1 latents, output largely garbage. The rest are completely useless without a matching UNET that uses the correct channel count.

![VAE_TEST](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/316c7029-ee78-4ff7-a46a-b56ef91477eb)
