# cotnav/dataset/dataset_builders.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Optional

import yaml
from torch.utils.data import DataLoader

from cotnav.dataset.template_dataset import GenericNavDataset

def load_yaml(path_or_dict) -> Dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, "r") as f:
        return yaml.safe_load(f)

def make_dataset(cfg: Dict[str, Any], split: str, **overrides) -> GenericNavDataset:
    """
    Build a GenericNavDataset (or swap to your concrete class) from cfg for a given split.
    """
    ds = GenericNavDataset(cfg, split=split, **overrides)
    return ds

def make_loader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dataset.collate_fn,
    )

def build_datasets_from_cfg(cfg_path_or_dict, splits: Iterable[str] = ("train","val","test"),
                            **dataset_overrides) -> Dict[str, GenericNavDataset]:
    cfg = load_yaml(cfg_path_or_dict)
    # You can also apply split-specific overrides here if your YAML has them.
    out = {}
    for sp in splits:
        out[sp] = make_dataset(cfg, split=sp, **dataset_overrides)
    return out

def build_loaders_from_cfg(
    cfg_path_or_dict,
    splits: Iterable[str] = ("train","val","test"),
    batch_size_map: Optional[Dict[str, int]] = None,
    num_workers_map: Optional[Dict[str, int]] = None,
    shuffle_map: Optional[Dict[str, bool]] = None,
    **dataset_overrides,
) -> Dict[str, DataLoader]:
    """
    Auto-build datasets & dataloaders from the dataset YAML.
    Example:
      loaders = build_loaders_from_cfg("config/dataset/policy/fai_lhy.yaml",
                                       splits=("train","full"),
                                       batch_size_map={"train": 8, "full": 1},
                                       num_workers_map={"train": 8, "full": 2},
                                       shuffle_map={"train": True, "full": False})
    """
    ds_map = build_datasets_from_cfg(cfg_path_or_dict, splits, **dataset_overrides)
    loaders = {}
    for sp, ds in ds_map.items():
        bs = (batch_size_map or {}).get(sp, 1)
        nw = (num_workers_map or {}).get(sp, 0)
        sh = (shuffle_map or {}).get(sp, sp == "train")
        loaders[sp] = make_loader(ds, batch_size=bs, num_workers=nw, shuffle=sh)
    return loaders

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="Path to dataset YAML config")
    parser.add_argument("--splits", type=str, nargs="+", default=["train","val","test"],
                        help="Which splits to build")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers")
    args = parser.parse_args()

    loaders = build_loaders_from_cfg(
        args.cfg,
        splits=args.splits,
        batch_size_map={sp: args.batch_size for sp in args.splits},
        num_workers_map={sp: args.num_workers for sp in args.splits},
        shuffle_map={sp: (sp=="train") for sp in args.splits},
    )
    for sp, loader in loaders.items():
        print(f"Split '{sp}' has {len(loader.dataset)} samples and {len(loader)} batches.")