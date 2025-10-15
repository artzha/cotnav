# cotnav/eval/registry.py
import importlib

def build_from_path(class_path: str, **kwargs):
    """
    class_path='cotnav.dataset.frodo:FrodoDataset' or 'cotnav.models.openvla:OpenVLA'
    """
    if ":" in class_path:
        module, name = class_path.split(":")
    else:
        parts = class_path.split("."); module, name = ".".join(parts[:-1]), parts[-1]
    cls = getattr(importlib.import_module(module), name)
    return cls(**kwargs)