# only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    NODE_CLASS_MAPPINGS = {}

    # All text encoders
    from .text_encoders.nodes import NODE_CLASS_MAPPINGS as Tenc_Nodes
    NODE_CLASS_MAPPINGS.update(Tenc_Nodes)

    # Generic nodes
    from .nodes import NODE_CLASS_MAPPINGS as Base_Nodes
    NODE_CLASS_MAPPINGS.update(Base_Nodes)

    # DiT
    from .DiT.nodes import NODE_CLASS_MAPPINGS as DiT_Nodes
    NODE_CLASS_MAPPINGS.update(DiT_Nodes)

    # PixArt
    from .PixArt.nodes import NODE_CLASS_MAPPINGS as PixArt_Nodes
    NODE_CLASS_MAPPINGS.update(PixArt_Nodes)

    # VAE
    from .VAE.nodes import NODE_CLASS_MAPPINGS as VAE_Nodes
    NODE_CLASS_MAPPINGS.update(VAE_Nodes)
 
    # MiaoBi
    from .MiaoBi.nodes import NODE_CLASS_MAPPINGS as MiaoBi_Nodes
    NODE_CLASS_MAPPINGS.update(MiaoBi_Nodes)

    # Extra
    from .utils.nodes import NODE_CLASS_MAPPINGS as Extra_Nodes
    NODE_CLASS_MAPPINGS.update(Extra_Nodes)

    # Sana
    from .Sana.nodes import NODE_CLASS_MAPPINGS as Sana_Nodes
    NODE_CLASS_MAPPINGS.update(Sana_Nodes)

    # Gemma
    from .Gemma.nodes import NODE_CLASS_MAPPINGS as Gemma_Nodes
    NODE_CLASS_MAPPINGS.update(Gemma_Nodes)

    NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
