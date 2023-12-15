# Extra Models for ComfyUI

This repository aims to add support for various different image diffusion models to ComfyUI.

## Installation

Simply clone this repo to your custom_nodes folder using the following command:

`git clone https://github.com/city96/ComfyUI_ExtraModels custom_nodes/ComfyUI_ExtraModels`

You will also have to install the requirements from the provided file by running `pip install -r requirements.txt` inside your VENV/conda env. If you downloaded the standalone version of ComfyUI, then follow the steps below.

### Standalone ComfyUI

I haven't tested this completely, so if you know what you're doing, use the regular venv/`git clone` install option when installing ComfyUI.

Go to the where you unpacked `ComfyUI_windows_portable` to (where your run_nvidia_gpu.bat file is) and open a command line window. Press `CTRL+SHIFT+Right click` in an empty space and click "Open PowerShell window here".

Clone the repository to your custom nodes folder, assuming haven't installed in through the manager.

`git clone https://github.com/city96/ComfyUI_ExtraModels .\ComfyUI\custom_nodes\ComfyUI_ExtraModels`

To install the requirements on windows, run these commands in the same window:
```
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_ExtraModels\requirements.txt
.\python_embeded\python.exe -s -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

To update, open the command line window like before and run the following commands:

```
cd .\ComfyUI\custom_nodes\ComfyUI_ExtraModels\
git pull
```

Alternatively, use the manager, assuming it has an update function.



## PixArt

[Original Repo](https://github.com/PixArt-alpha/PixArt-alpha)

### Model info / implementation
- Uses T5 text encoder instead of clip
- Available in 512 and 1024 versions, needs specific pre-defined resolutions to work correctly
- Same latent space as SD1.5 (works with the SD1.5 VAE)
- Attention needs optimization, images look worse without xformers.

### Usage

1. Download the model weights from the [PixArt alpha repo](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main) - you most likely want the 1024px one - `PixArt-XL-2-1024-MS.pth`
2. Place them in your checkpoints folder
3. Load them with the correct PixArt checkpoint loader
4. **Follow the T5v11 section of this readme** to set up the T5 text encoder

> [!TIP]
> You should be able to use the model with the default KSampler if you're on the latest version of the node.
> In theory, this should allow you to use longer prompts as well as things like doing img2img.

Limitations:
- `PixArt DPM Sampler` requires the negative prompt to be shorter than the positive prompt.
- `PixArt DPM Sampler` can only work with a batch size of 1.
- `PixArt T5 Text Encode` is from the reference implementation, therefore it doesn't support weights. `T5 Text Encode` support weights, but I can't attest to the correctness of the implementation.

> [!IMPORTANT]  
> Installing `xformers` is optional but strongly recommended as torch SDP is only partially implemented, if that.

[Sample workflow here](https://github.com/city96/ComfyUI_ExtraModels/files/13617463/PixArtV3.json)

![PixArtT12](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/eb1a02f9-6114-47eb-a066-261c39c55615)

### PixArt LCM

The LCM model also works if you're on the latest version. To use it:

1. Download the [PixArt LCM model](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS/blob/main/transformer/diffusion_pytorch_model.safetensors) and place it in your checkpoints folder.
2. Add a `ModelSamplingDiscrete` node and set "sampling" to "lcm"
3. Adjust the KSampler settings - Set the sampler to "lcm". Your CFG should be fairly low (1.1-1.5), your steps should be around 5.

Everything else can be the same the same as in the example above.

![PixArtLCM](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/558f8b30-449b-4973-ad7e-6aa69832adcb)



## DiT

[Original Repo](https://github.com/facebookresearch/DiT)

### Model info / implementation
- Uses class labels instead of prompts
- Limited to 256x256 or 512x512 images
- Same latent space as SD1.5 (works with the SD1.5 VAE)
- Works in FP16, but no other optimization

### Usage

1. Download the original model weights from the [DiT Repo](https://github.com/facebookresearch/DiT) or the converted [FP16 safetensor ones from Huggingface](https://huggingface.co/city96/DiT/tree/main).
2. Place them in your checkpoints folder. (You may need to move them if you had them in `ComfyUI\models\dit` before)
3. Load the model and select the class labels as shown in the image below
4. **Make sure to use the Empty label conditioning for the Negative input of the KSampler!**

ConditioningCombine nodes *should* work for combining multiple labels. The area ones don't since the model currently can't handle dynamic input dimensions.

[Sample workflow here](https://github.com/city96/ComfyUI_ExtraModels/files/13619259/DiTV2.json)

![DIT_WORKFLOW_IMG](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/cdd4ec94-b0eb-436a-bf23-a3bcef8d7b90)



## T5

### T5v11

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

> [!IMPORTANT]  
> You may also need to upgrade transformers and install spiece for the tokenizer. `pip install -r requirements.txt`



## VAE

A few custom VAE models are supported. The option to select a different dtype when loading is also possible, which can be useful for testing/comparisons.

### Consistency Decoder

[Original Repo](https://github.com/openai/consistencydecoder)

This now works thanks to the work of @mrsteyk and @madebyollin - [Gist with more info](https://gist.github.com/madebyollin/865fa6a18d9099351ddbdfbe7299ccbf).

- Download the converted safetensor VAE from [this HF repository](https://huggingface.co/mrsteyk/consistency-decoder-sd15/blob/main/stk_consistency_decoder_amalgamated.safetensors). If you downloaded the OpenAI model before, it won't work, as it is a TorchScript file. Feel free to delete it.
- Put the file in your VAE folder
- Load it with the ExtraVAELoader
- Set it to fp16 or bf16 to not run out of VRAM
- Use tiled VAE decode if required

### Deflickering Decoder / VideoDecoder

This is the VAE that comes baked into the [Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) model.

It doesn't seem particularly good as a normal VAE (color issues, pretty bad with finer details).

Still for completeness sake the code to run it is mostly implemented. To obtain the weights just extract them from the sdv model:

```py
from safetensors.torch import load_file, save_file

pf = "first_stage_model." # Key prefix
sd = load_file("svd_xt.safetensors")
vae = {k.replace(pf, ''):v for k,v in sd.items() if k.startswith(pf)}
save_file(vae, "svd_xt_vae.safetensors")
```

### AutoencoderKL / VQModel

`kl-f4/8/16/32` from the [compvis/latent diffusion repo](https://github.com/CompVis/latent-diffusion/tree/main#pretrained-autoencoding-models).

`vq-f4/8/16` from the taming transformers repo, weights for both vq and kl models available [here](https://ommer-lab.com/files/latent-diffusion/)

`vq-f8` can accepts latents from the SD unet but just like xl with v1 latents, output largely garbage. The rest are completely useless without a matching UNET that uses the correct channel count.

![VAE_TEST](https://github.com/city96/ComfyUI_ExtraModels/assets/125218114/316c7029-ee78-4ff7-a46a-b56ef91477eb)
