import logging
import comfy.utils
import comfy.model_patcher
from comfy import model_management

def load_state_dict_from_config(model_config, sd, model_options={}):
    parameters = comfy.utils.calculate_parameters(sd)
    load_device = model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    dtype = model_options.get("dtype", None)
    weight_dtype = comfy.utils.weight_dtype(sd)
    unet_weight_dtype = list(model_config.supported_inference_dtypes)

    if weight_dtype is not None and model_config.scaled_fp8 is None:
        unet_weight_dtype.append(weight_dtype)

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(sd, "")
    model = model.to(offload_device).eval()
    model.load_model_weights(sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
