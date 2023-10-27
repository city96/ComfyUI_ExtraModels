import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import torch
from comfy import model_management

from .models import PixArtMS

class EXM_PixArt(comfy.supported_models_base.BASE):
	unet_config = {}
	unet_extra_config = {}
	latent_format = comfy.latent_formats.SD15

	def model_type(self, state_dict, prefix=""):
		return comfy.model_base.ModelType.EPS

def load_pixart(model_path, model_conf):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)
	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)

	model = comfy.model_base.BaseModel(
		EXM_PixArt({"disable_unet_model_creation" : True }),
		model_type=comfy.model_base.ModelType.EPS,
		device=model_management.get_torch_device()
	)

	model.pixart_config = model_conf
	if model_conf["target"] == "PixArtMS":
		from .models.PixArtMS import PixArtMS
		model.diffusion_model = PixArtMS(**model_conf)
	elif model_conf["target"] == "PixArt":
		from .models.PixArt import PixArt
		model.diffusion_model = PixArt(**model_conf)
	else:
		raise NotImplementedError

	model.diffusion_model.load_state_dict(state_dict)
	model.diffusion_model.dtype = unet_dtype
	model.diffusion_model.eval()
	model.diffusion_model.to(unet_dtype)

	model_patcher = comfy.model_patcher.ModelPatcher(
		model,
		load_device    = comfy.model_management.get_torch_device(),
		offload_device = comfy.model_management.unet_offload_device(),
		current_device = "cpu",
	)
	return model_patcher
