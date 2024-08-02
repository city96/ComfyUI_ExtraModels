NODE_CLASS_MAPPINGS = {}

from .offload import NODE_CLASS_MAPPINGS as Offload_Nodes
NODE_CLASS_MAPPINGS.update(Offload_Nodes)

for name, node in NODE_CLASS_MAPPINGS.items():
	cat = node.CATEGORY
	if not cat.startswith("ExtraModels/"):
		node.CATEGORY = f"ExtraModels/{cat}"

__all__ = ["NODE_CLASS_MAPPINGS"]
