import comfy.supported_models_base
import comfy.supported_models
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import torch
from comfy import model_management

from .model import DiT

class EXMDiT(comfy.supported_models.SD15):
	unet_config = {}
	unet_extra_config = {}

	def model_type(self, state_dict, prefix=""):
		return comfy.model_base.ModelType.EPS

def load_dit(model_path, model_conf):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)
	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = model_management.unet_dtype(model_params=parameters)

	offload_device = model_management.unet_offload_device()
	model = comfy.model_base.BaseModel(
		EXMDiT({"disable_unet_model_creation" : True }),
		model_type=comfy.model_base.ModelType.EPS,
		device=model_management.get_torch_device()
	)
	model_conf["num_classes"] = state_dict["y_embedder.embedding_table.weight"].shape[0] - 1 # adj. for empty
	model.dit_config = model_conf
	model.diffusion_model = DiT(**model_conf).eval()
	model.diffusion_model.load_state_dict(state_dict)
	model.diffusion_model.eval()
	model.diffusion_model.dtype = unet_dtype
	model.diffusion_model.to(unet_dtype)

	model_patcher = comfy.model_patcher.ModelPatcher(
		model,
		load_device=comfy.model_management.get_torch_device(),
		offload_device=comfy.model_management.unet_offload_device(),
		current_device="cpu"
	)
	return model_patcher
