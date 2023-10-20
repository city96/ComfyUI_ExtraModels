# only import if running as a custom node
try:
	import comfy.utils
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}

	# Deci Diffusion
	# from .DeciDiffusion.nodes import NODE_CLASS_MAPPINGS as DeciDiffusion_Nodes
	# NODE_CLASS_MAPPINGS.update(DeciDiffusion_Nodes)

	# DiT
	from .DiT.nodes import NODE_CLASS_MAPPINGS as DiT_Nodes
	NODE_CLASS_MAPPINGS.update(DiT_Nodes)
	
	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
