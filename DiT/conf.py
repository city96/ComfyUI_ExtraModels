"""
List of all DiT model types / settings
"""
dit_conf = {
	"XL/2": { # DiT_XL_2
		"depth"       :   28,
		"num_heads"   :   16,
		"patch_size"  :    2,
		"hidden_size" : 1152,
	},
	"XL/4": { # DiT_XL_4
		"depth"       :   28,
		"num_heads"   :   16,
		"patch_size"  :    4,
		"hidden_size" : 1152,
	},
	"XL/8": { # DiT_XL_8
		"depth"       :   28,
		"num_heads"   :   16,
		"patch_size"  :    8,
		"hidden_size" : 1152,
	},
	"L/2": { # DiT_L_2
		"depth"       :   24,
		"num_heads"   :   16,
		"patch_size"  :    2,
		"hidden_size" : 1024,
	},
	"L/4": { # DiT_L_4
		"depth"       :   24,
		"num_heads"   :   16,
		"patch_size"  :    4,
		"hidden_size" : 1024,
	},
	"L/8": { # DiT_L_8
		"depth"       :   24,
		"num_heads"   :   16,
		"patch_size"  :    8,
		"hidden_size" : 1024,
	},
	"B/2": { # DiT_B_2
		"depth"       :   12,
		"num_heads"   :   12,
		"patch_size"  :    2,
		"hidden_size" :  768,
	},
	"B/4": { # DiT_B_4
		"depth"       :   12,
		"num_heads"   :   12,
		"patch_size"  :    4,
		"hidden_size" :  768,
	},
	"B/8": { # DiT_B_8
		"depth"       :   12,
		"num_heads"   :   12,
		"patch_size"  :    8,
		"hidden_size" :  768,
	},
	"S/2": { # DiT_S_2
		"depth"       :   12,
		"num_heads"   :    6,
		"patch_size"  :    2,
		"hidden_size" :  384,
	},
	"S/4": { # DiT_S_4
		"depth"       :   12,
		"num_heads"   :    6,
		"patch_size"  :    4,
		"hidden_size" :  384,
	},
	"S/8": { # DiT_S_8
		"depth"       :   12,
		"num_heads"   :    6,
		"patch_size"  :    8,
		"hidden_size" :  384,
	},
}
