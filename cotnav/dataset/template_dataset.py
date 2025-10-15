# cotnav/dataset/template_dataset.py
from __future__ import annotations
import copy, logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Optional

import json
import yaml
from PIL import Image
from decord import VideoReader, cpu, bridge as decord_bridge
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torchvision.transforms import ToTensor

import pandas as pd

from cotnav.utils.loader_utils import (load_timestamps, load_odom, build_transforms, load_intrinsics)
from cotnav.utils.math_utils import (interpolate_se3, se3_matrix)

from cotnav.dataset import dataset_helpers as dh

DTYPE_TO_TORCH = {
    "float": torch.float32,
    "half": torch.float16,
    "double": torch.float64,
    "long": torch.int64,
    "int": torch.int32,
    "byte": torch.uint8,
    "bool": torch.bool,
    "str": str,   # special-cased
}

# ------------------------------
# Utilities
# ------------------------------

def read_split_txt(split_txt: Path) -> List[str]:
    """
    Support BOTH formats:
      1) CSV with columns: ride_id, drive_id0, drive_id1, timestamp, start_frame
      2) CSV with columns: ride_name, start_frame (ride_name = 'ride d0 d1 ts')
    Returns a list of single-string sample_ids: 'ride d0 d1 ts start'
    """
    assert Path(split_txt).exists(), f"Missing split file: {split_txt}"
    df = pd.read_csv(split_txt, sep='\t', header=0, dtype=str)

    columns = ["dataset", "mission_id", "start_frame", "end_frame"]
    assert all(c in df.columns for c in columns), f"split.txt missing columns: {columns}"
    df = pd.concat([df[c] for c in columns], axis=1)
    
    rows = df.values.tolist()
    return [" ".join(r) for r in rows]


def parse_sample_id(sample_id: str) -> Dict[str, str]:
    """Reverse of read_split_txt output."""
    dataset, mission_id, start_frame, end_frame = sample_id.split(" ")
    return {
        "dataset": dataset,
        "mission_id": mission_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


# ------------------------------
# Dataset
# ------------------------------

class GenericNavDataset(Dataset):
    """
    Boilerplate dataset that mirrors your FrodoDataset style:
    - `cfg` mirrored into `self`
    - `setup_paths` builds path attributes from cfg.subdirs
    - `setup_samples` reads split_dir/{split}.txt
    - `load_cfgs` is converted to dict keyed by name
    - dispatch in __getitem__ via add_load_key() and load_frame_data()
    """

    def __init__(self, cfg: Dict[str, Any], split: str = "train",
                 do_augmentation: bool = False, **kwargs):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.split = split
        self.do_augmentation = do_augmentation

        self.merge_kwargs(kwargs)
        assert Path(self.root_dir).exists(), f"Root directory {self.root_dir} does not exist"

        # small file cache pattern (you can expand later)
        self._h5_cache_max = 2
        self._h5_cache: OrderedDict[str, Any] = OrderedDict()

        self.setup_paths(self.root_dir, self.subdirs)
        self.idx_to_sample, self.sample_to_idx = self.setup_samples(split)

        if self.do_augmentation:
            self.augmentation_dict = self.setup_augmentations()

    # ---------- cfg wiring ----------

    def merge_kwargs(self, kwargs: Dict[str, Any]) -> None:
        for k, v in kwargs.items():
            if k in self.cfg:
                logging.info(f"Override cfg.{k} -> {v}")
            self.cfg[k] = v
        for k, v in self.cfg.items():
            setattr(self, k, v)

        # normalize load_cfgs to dict keyed by 'name'
        _d = {}
        for task in self.load_cfgs:
            _d[task["name"]] = {kk: vv for kk, vv in task.items() if kk != "name"}
        self.__dict__["load_cfgs"] = _d

    def setup_paths(self, root_dir: str, subdirs: Iterable[Dict[str, str]]) -> None:
        for sd in subdirs:
            name, rel = sd["name"], sd["path"]
            p = Path(root_dir) / rel
            p.mkdir(parents=True, exist_ok=True)
            setattr(self, name, str(p))
            logging.info(f"Set path: {name} -> {p}")

    def setup_samples(self, split: str) -> Tuple[Dict[int, str], Dict[str, int]]:
        split_path = Path(self.split_dir) / f"{split}.txt"
        ride_names = read_split_txt(split_path)
        idx_to_sample = {i: rn for i, rn in enumerate(ride_names)}
        sample_to_idx = {rn: i for i, rn in idx_to_sample.items()}
        logging.info(f"Loaded split='{split}' with N={len(ride_names)}")
        return idx_to_sample, sample_to_idx

    # ---------- augmentation hooks (templates) ----------

    def setup_augmentations(self) -> Dict[str, Any]:
        """
        Mirror your 'augmentations' config. Fill in real functions as needed.
        """
        return {}  # TEMPLATE: plug your Image/Action/Heading augmentors here

    def renew_augmentation(self):
        for aug in getattr(self, "augmentation_dict", {}).values():
            if hasattr(aug, "renew_augmentation"):
                aug.renew_augmentation()

    # ---------- io helpers ----------

    def _get_h5(self, path: Path):
        """Simple worker-safe pattern; expand with LRU if desired."""
        return hkl.load(str(path))

    # ---------- tensor allocation ----------

    def add_load_key(self, key: str, num_frames: int):
        key_cfg = self.load_cfgs[key]
        key_type = key_cfg["type"]
        try:
            if key_type == "scalar":
                return torch.zeros(num_frames, dtype=DTYPE_TO_TORCH[key_cfg["dtype"]])
            elif key_type in {"image", "depth", "bev", "tensor"}:
                dims = key_cfg["kwargs"]["dimensions"]
                dtype = DTYPE_TO_TORCH[key_cfg["kwargs"]["dtype"]]
                channel_dim = key_cfg["kwargs"].get("channel_dim", True)
                out_dims = (num_frames, *dims) if channel_dim else dims
                return torch.zeros(out_dims, dtype=dtype)
            elif key_type == "odom":
                dims = key_cfg["kwargs"]["dimensions"]
                dtype = DTYPE_TO_TORCH[key_cfg["kwargs"]["dtype"]]
                return torch.zeros(dims, dtype=dtype)
            elif key_type in {"dict", "list"}:
                return [] if key_type == "list" else None
            else:
                raise NotImplementedError(f"Key type '{key_type}' not implemented")
        except KeyError as e:
            logging.error(f"load_cfgs missing for key={key}: {e}")

    # ---------- item plumbing ----------

    def __len__(self) -> int:
        return len(self.idx_to_sample)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.idx_to_sample[idx]
        infos = self._load_infos(sample_id)

        if self.do_augmentation:
            self.renew_augmentation()

        out: Dict[str, Any] = {"infos": infos}
        load_keys = getattr(self, "fload_keys", []) + getattr(self, "sload_keys", [])
        num_frames = getattr(self, "num_views", 1)

        # allocate holders
        for lk in load_keys:
            buf = self.add_load_key(lk, num_frames)
            if buf is not None:
                out[lk] = buf

        # fill them
        for lk in getattr(self, "fload_keys", []):
            data = self.load_frame_data(infos, lk)
            if isinstance(data, dict):
                out.update(data)
            elif not torch.is_tensor(data):
                out[lk] = data
            elif lk in out and data.shape[0] == out[lk].shape[0]:
                out[lk] = data
            else:
                raise ValueError(f"Shape mismatch for key {lk}")

        return out

    # ---------- dispatch to loaders ----------

    def load_frame_data(self, infos: Dict[str, Any], key: str):
        key_cfg = self.load_cfgs[key]
        key_type = key_cfg["type"]
        if key_type == "image":
            return self._load_image(infos, key_cfg)
        elif key_type == "odom":
            return self._load_odom(infos, key_cfg)
        elif key_type == "dict":
            # Delegate on dict name (e.g., 'path_meta', 'stlang_meta', etc.)
            name = key
            loader = getattr(self, f"_load_{name}", None)
            if loader is None:
                raise NotImplementedError(f"{name} meta loader not implemented")
            return loader(infos, key_cfg)
        else:
            # Scalars / strings can come from infos directly
            try:
                return str(infos[key])
            except KeyError as e:
                raise KeyError(f"Key {key} not found in infos") from e

    # ---------- template: infos ----------

    def _load_infos(self, sample_id: str) -> Dict[str, Any]:
        """
        TEMPLATE: adapt to your directory scheme.
        """
        toks = parse_sample_id(sample_id)
        dataset, mission_id, start_frame, end_frame = toks["dataset"], toks["mission_id"], toks["start_frame"], toks["end_frame"]
        seq_dir = Path(self.root_dir) / mission_id

        # Load calibrations if available

        return {
            "sequence": f"{dataset} {mission_id} {start_frame} {end_frame}",
            "mission_id": mission_id,
            "end_frame": end_frame,
            "start_frame": start_frame,
            "path": str(seq_dir),
        }

    # ---------- template: normalization helpers ----------

    @staticmethod
    def _normalize_image(th: torch.Tensor, low: float, high: float) -> torch.Tensor:
        if th.max() > 1.0:  # assume 0..255
            th = th / 255.0
        th = torch.clamp(th, 0.0, 1.0)
        dom = abs(high - low)
        out = th * dom + low
        return out

    # ---------- template loaders you can fill in ----------

    def _load_calib(self, infos: Dict[str, Any], key_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Required cfg:
          kwargs.intr_filename, kwargs.tf_filename
        """
        intr_path = Path(self.root_dir) / infos['mission_id'] / key_cfg['kwargs']['intr_filename']
        tf_path = Path(self.root_dir) / infos['mission_id'] / key_cfg['kwargs']['tf_filename']
        assert intr_path.exists(), f"Missing intrinsics file: {intr_path}"
        assert tf_path.exists(), f"Missing tf file: {tf_path}"
        world_frame = key_cfg['kwargs'].get('world_frame', 'base')

        # Resize intrinsics if needed
        ds_rgb = key_cfg['kwargs'].get('ds_rgb', 1.0)
        calib = load_intrinsics(intr_path, tf_path, world_frame, ds_rgb=ds_rgb)

        return calib

    def _load_image(self, infos: Dict[str, Any], key_cfg: Dict[str, Any]) -> torch.Tensor:
        """
        TEMPLATE: load image → tensor in range [low, high]
        Required cfg:
          kwargs.camid, kwargs.dimensions=[C,H,W], kwargs.dtype, kwargs.range=[lo,hi]
        """
        dims = key_cfg["kwargs"]["dimensions"]
        dtype = DTYPE_TO_TORCH[key_cfg["kwargs"]["dtype"]]
        lo, hi = key_cfg["kwargs"]["range"]

        video_path = Path(self.root_dir) / infos['mission_id'] / key_cfg['kwargs']['filename']
        assert video_path.exists(), "Video not found"
        start_frame, end_frame = int(infos['start_frame']), int(infos['end_frame'])
        num_views = key_cfg['kwargs']['num_views']

        idx = np.linspace(start_frame, end_frame+1, num_views)
        vr = VideoReader(str(video_path), ctx=cpu(0), width=dims[-1], height=dims[-2])
        decord_bridge.set_bridge('torch')
        th = vr.get_batch(idx)                     # (T, H, W, C), uint8 torch tensor
        th = th.permute(0, 3, 1, 2).to(torch.float32)  # (T, C, H, W)
        th = self._normalize_image(th, lo, hi)

        # Optional augmentation hook
        if self.do_augmentation and "ImageAugmentation" in getattr(self, "augmentation_dict", {}):
            th = self.augmentation_dict["ImageAugmentation"](th)

        return th

    def _load_odom(self, infos: Dict[str, Any], key_cfg: Dict[str, Any]) -> torch.Tensor:
        """
        TEMPLATE: fill with your se(2)/se(3) conversion.
        Required: kwargs.dimensions, kwargs.dtype, kwargs.horizon, kwargs.skip_factor
        """
        ts_path = Path(self.root_dir) / infos['mission_id'] / key_cfg['kwargs']['ts_filename']
        tf_path = Path(self.root_dir) / infos['mission_id'] / key_cfg['kwargs']['tf_filename']
        odom_path = Path(self.root_dir) / infos['mission_id'] / key_cfg['kwargs']['odom_filename']
        assert all([p.exists() for p in [ts_path, tf_path, odom_path]]), "Missing odom/ts/tf files"
        odom_np = load_odom(odom_path)
        tm = build_transforms(tf_path)

        out_dict = {}
        subkeys = key_cfg['kwargs'].get('subkeys', [])
        for subkey_cfg in subkeys:
            name = subkey_cfg['name']
            dtype = DTYPE_TO_TORCH[subkey_cfg['dtype']]
            convert_fn = subkey_cfg['converter_fn']
            start_frame, end_frame = None, None

            if name == 'action_ctx':
                start_frame, end_frame = int(infos['start_frame']), int(infos['end_frame'])
            elif name == 'action_label':
                start_frame = int(infos['start_frame']) + 1
                end_frame = start_frame + subkey_cfg['horizon']
            else:
                raise NotImplementedError(f"Unknown odom subkey '{name}'")

            # Align odom to video timestamps if available
            ref_ts = load_timestamps(ts_path)[start_frame:end_frame+1]
            interp_odom = interpolate_se3(ref_ts, odom_np[:, 0], odom_np[:, 1:4], odom_np[:, 4:8])
            T_base_local = getattr(dh, convert_fn)(interp_odom, ts_path, tm)

            # Pad or truncate to fixed length if needed
            num_actions = subkey_cfg.get('num_actions', None)
            if num_actions is not None:
                T_base_local = dh.pad_or_trunc_last(T_base_local, num_actions)

            out_dict[name] = torch.from_numpy(T_base_local).to(dtype)  # (T,4,4)

        return out_dict

    def _load_prompt(self, infos: Dict[str, Any], key_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        TEMPLATE: load text prompt from a JSON file.
        Required cfg:
          kwargs.prompts: path to JSON file with structure {"0": "prompt text", ...}
        """
        # Start prompts from project directory
        prompts_path = Path("./") / key_cfg['kwargs']['prompts']
        assert prompts_path.exists(), f"Prompts file not found: {prompts_path}"

        with open(prompts_path, 'r') as f:
            prompts_dict = json.load(f)

        # TODO: Preprocess Language Command
        for key, prompt_dict in prompts_dict.items():
            if key == 'system_prompt':
                system_prompt = open(prompt_dict['path'], 'r').read().strip()
                prompts_dict[key] = system_prompt
            else:
                prompts_dict[key] = tuple(prompt_dict)
        return prompts_dict

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples: Dict[str, Any] = {}
        for k in batch[0].keys():
            if isinstance(batch[0][k], torch.Tensor):
                samples[k] = torch.stack([d[k] for d in batch], dim=0)
            elif isinstance(batch[0][k], list):
                out = []
                for d in batch:
                    out.extend(d[k])
                samples[k] = out
            else:
                samples[k] = [d[k] for d in batch]
        return samples

if __name__ == "__main__":
    import yaml
    with open("configs/dataset/grandtour_raw.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ds = GenericNavDataset(cfg, split="full", do_augmentation=True)
    print(f"Dataset N={len(ds)}")

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                    collate_fn=ds.collate_fn)
    for batch in dl:
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"{k}: {v.shape} {v.dtype} {v.device}")
            else:
                print(f"{k}: {type(v)} len={len(v)}")
        break