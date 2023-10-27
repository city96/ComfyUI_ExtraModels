import torch
from .sampling import gaussian_diffusion as gd
from .sampling.dpm_solver import model_wrapper, DPM_Solver, NoiseScheduleVP

from comfy.sample import prepare_sampling, prepare_noise, cleanup_additional_models, get_models_from_cond
import comfy.utils
import latent_preview

def sample_pixart(model, seed, steps, cfg, noise_schedule, noise_schedule_vp, positive, negative, latent_image):
	"""
	Mostly just a wrapper around the reference code.
	"""
	# prepare model
	noise = prepare_noise(latent_image, seed)
	real_model, _, _, _, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask=None)

	# negative cond
	cond = positive[0][0]
	raw_uncond = negative[0][0]

	# Sampler seems to want the same dim for cond and uncond
	#  truncate uncond to the length of cond
	#  if shorter, pad uncond with y_null
	null_y = real_model.diffusion_model.y_embedder.y_embedding[None].repeat(latent_image.shape[0], 1, 1)
	uncond = null_y[:, :cond.shape[1], :]
	uncond[:, :raw_uncond.shape[1], :] = raw_uncond[:, :cond.shape[1], :]
	if raw_uncond.shape[1] > cond.shape[1]:
		print("PixArt: Warning. Your negative prompt is too long.")
		uncond[:, -1, :] = raw_uncond[:, -1, :] # add back EOS token

	# Move inputs
	cond = cond.to(model.load_device).to(real_model.diffusion_model.dtype)
	uncond = uncond.to(model.load_device).to(real_model.diffusion_model.dtype)
	noise = noise.to(model.load_device).to(real_model.diffusion_model.dtype)

	# preview
	pbar = comfy.utils.ProgressBar(steps)
	previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)

	## Noise schedule.
	betas = torch.tensor(gd.get_named_beta_schedule(noise_schedule, steps))
	noise_schedule = NoiseScheduleVP(schedule=noise_schedule_vp, betas=betas)

	## Convert your discrete-time `model` to the continuous-time
	## noise prediction model. Here is an example for a diffusion model
	## `model` with the noise prediction type ("noise") .
	model_fn = model_wrapper(
		real_model.diffusion_model.forward,
		noise_schedule,
		model_type="noise", # 'noise', "x_start", "v", "score"
		model_kwargs={},
		guidance_type="classifier-free",
		condition=cond,
		unconditional_condition=uncond,
		guidance_scale=cfg,
	)
	dpm_solver = DPM_Solver(
		model_fn,
		noise_schedule,
		algorithm_type="dpmsolver++"
	)
	samples = dpm_solver.sample(
		noise,
		steps=steps,
		order=2,
		skip_type="time_uniform",
		method="multistep",
		pbar=pbar,
		previewer=previewer,
	)

	cleanup_additional_models(models)
	cleanup_additional_models(set(get_models_from_cond(positive, "control")))
	return samples.detach().cpu().float() * (1 / model.model.latent_format.scale_factor)
