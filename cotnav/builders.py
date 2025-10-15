# cotnav/builders.py
"""
Super-simple factories for datasets and models.
Usage:
  ds = build_dataset("generic_nav", cfg=..., split="full")
  m  = build_model("openvla", ckpt=".../model.ckpt")
Or dotted path fallback:
  ds = build_dataset("cotnav.dataset.template_dataset:GenericNavDataset", cfg=..., split="full")
"""

import importlib
import inspect
from typing import Any, Callable

# ---- optional: readable errors ----
def _oops(kind: str, name: str) -> str:
    return (
        f"Unknown {kind} '{name}'. "
        f"Add an entry to cotnav/builders.py or pass a dotted path like 'pkg.mod:Symbol'."
    )

def build_from_path(path: str, **kwargs) -> Any:
    """Accepts 'pkg.mod:Symbol' or 'pkg.mod.Symbol'."""
    if ":" in path:
        mod, sym = path.split(":")
    else:
        mod, sym = path.rsplit(".", 1)
    obj = getattr(importlib.import_module(mod), sym)
    if inspect.isclass(obj):
        return obj(**kwargs)
    if callable(obj):
        return obj(**kwargs)
    return obj

# ---- DATASET FACTORY ----
def build_dataset(name: str, **kwargs) -> Any:
    # dotted path fallback
    if "." in name or ":" in name:
        return build_from_path(name, **kwargs)

    # Hand-wired aliases (add as needed)
    if name == "grandtour_raw":
        from cotnav.dataset.template_dataset import GenericNavDataset
        return GenericNavDataset(kwargs, **kwargs)

    # Add more:
    # elif name == "frodo":
    #     from spinflow.dataset.frodo import FrodoDataset
    #     return FrodoDataset(**kwargs)

    raise ValueError(_oops("dataset", name))

# ---- MODEL FACTORY ----
def build_model(name: str, **kwargs) -> Any:
    # dotted path fallback
    if "." in name or ":" in name:
        return build_from_path(name, **kwargs)

    # Hand-wired aliases (add as needed)
    if name == "pivot":
        from cotnav.models.vlms.pivot_wrapper import PIVOT
        return PIVOT(**kwargs)

    # Add more:
    # elif name == "qwen2vl":
    #     from cotnav.models.qwen2vl import Qwen2VL
    #     return Qwen2VL(**kwargs)

    raise ValueError(_oops("model", name))