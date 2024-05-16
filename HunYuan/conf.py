"""
List of all DiT model types / settings
"""
sampling_settings = {
	"beta_schedule" : "linear",
	"linear_start"  : 0.00085,
	"linear_end"    : 0.03,
	"timesteps"     : 1000,
	'steps_offset': 1,
	'clip_sample': False,
	'clip_sample_range': 1.0,
	'beta_start': 0.00085,
	'beta_end': 0.03,
	'prediction_type': 'v_prediction',
}

dit_conf = {
	"DiT-g/2": { # DiT-g/2
		"unet_config": {
			"depth"       :   40,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1408,
			'mlp_ratio': 4.3637,
		},
		"sampling_settings" : sampling_settings,
	},
	"DiT-XL/2": { # DiT_XL_2
		"unet_config": {
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1152,
		},
		"sampling_settings" : sampling_settings,
	},
	"DiT-L/2": { # DiT_L_2
		"unet_config": {
			"depth"       :   24,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1024,
		},
		"sampling_settings" : sampling_settings,
	},
	"DiT-B/2": { # DiT_B_2
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :   12,
			"patch_size"  :    2,
			"hidden_size" :  768,
		},
		"sampling_settings" : sampling_settings,
	},
}
