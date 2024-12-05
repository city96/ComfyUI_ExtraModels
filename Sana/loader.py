import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds
import torch
import math 
from comfy import model_management
from comfy.latent_formats import LatentFormat
from .diffusers_convert import convert_state_dict


class SanaLatent(LatentFormat):
    latent_channels = 32
    def __init__(self):
        self.scale_factor = 0.41407


class EXM_Sana(comfy.supported_models_base.BASE):
	unet_config = {}
	unet_extra_config = {}
	latent_format = SanaLatent

	def __init__(self, model_conf):
		self.model_target = model_conf.get("target")
		self.unet_config = model_conf.get("unet_config", {})
		self.sampling_settings = model_conf.get("sampling_settings", {})
		self.latent_format = self.latent_format()
		# UNET is handled by extension
		self.unet_config["disable_unet_model_creation"] = True

	def model_type(self, state_dict, prefix=""):
		return comfy.model_base.ModelType.FLOW


class EXM_Sana_Model(comfy.model_base.BaseModel):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def extra_conds(self, **kwargs):
		out = super().extra_conds(**kwargs)

		cn_hint = kwargs.get("cn_hint", None)
		if cn_hint is not None:
			out["cn_hint"] = comfy.conds.CONDRegular(cn_hint)

		return out


def load_sana(model_path, model_conf):
	state_dict = comfy.utils.load_torch_file(model_path)
	state_dict = state_dict.get("model", state_dict)

	# prefix
	for prefix in ["model.diffusion_model.",]:
		if any(True for x in state_dict if x.startswith(prefix)):
			state_dict = {k[len(prefix):]:v for k,v in state_dict.items()}

	# diffusers
	if "adaln_single.linear.weight" in state_dict:
		state_dict = convert_state_dict(state_dict) # Diffusers

	parameters = comfy.utils.calculate_parameters(state_dict)
	unet_dtype = comfy.model_management.unet_dtype()
	load_device = comfy.model_management.get_torch_device()
	offload_device = comfy.model_management.unet_offload_device()

	# ignore fp8/etc and use directly for now
	manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device)
	if manual_cast_dtype:
		print(f"Sana: falling back to {manual_cast_dtype}")
		unet_dtype = manual_cast_dtype

	model_conf = EXM_Sana(model_conf) # convert to object
	model = EXM_Sana_Model( # same as comfy.model_base.BaseModel
		model_conf,
		model_type=comfy.model_base.ModelType.FLOW,
		device=model_management.get_torch_device()
	)

	if model_conf.model_target == "SanaMS":
		from .models.sana_multi_scale import SanaMS
		model.diffusion_model = SanaMS(**model_conf.unet_config)
	else:
		raise NotImplementedError(f"Unknown model target '{model_conf.model_target}'")

	m, u = model.diffusion_model.load_state_dict(state_dict, strict=False)
	if len(m) > 0: print("Missing UNET keys", m)
	if len(u) > 0: print("Leftover UNET keys", u)
	model.diffusion_model.dtype = unet_dtype
	model.diffusion_model.eval()
	model.diffusion_model.to(unet_dtype)

	model_patcher = comfy.model_patcher.ModelPatcher(
		model,
		load_device = load_device,
		offload_device = offload_device,
	)
	return model_patcher
