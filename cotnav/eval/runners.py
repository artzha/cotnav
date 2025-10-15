# runners.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from cotnav.utils.metric_utils import MetricManager
from cotnav.utils import train_utils as tu

from cotnav.builders import build_from_path

# ----------------------------
# Minimal JSONL writer
# ----------------------------
class JSONLWriter:
    def __init__(self, out_dir: Path | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.out_dir / "episodes.jsonl", "w")

    def write(self, record: Dict[str, Any]):
        self.fp.write(json.dumps(_to_json_safe(record)) + "\n")
        self.fp.flush()

    def close(self):
        self.fp.close()

def _to_json_safe(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_json_safe(v) for v in x]
    return x

# ----------------------------
# Eval config loader (simple)
# ----------------------------
def load_eval_cfg(cfg_or_path: Dict[str, Any] | str) -> Dict[str, Any]:
    """Accept a dict or a YAML path; always return a plain dict."""
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    import yaml
    with open(cfg_or_path, "r") as f:
        return yaml.safe_load(f)

# ----------------------------
# Episodic Runner (simplified)
# ----------------------------
class EpisodicRunner:
    def __init__(
        self,
        dataloader,
        model: Any,
        eval_cfg: Dict[str, Any],
        model_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        metric_registry: Optional[Dict[str, Any]] = None,  # optional short-name registry
        writer: Any = None,                                 # optional scalar writer
    ):
        self.dl = dataloader
        self.model = model
        self.model_cfg = model_cfg or {}
        self.eval_cfg = eval_cfg
        self.device = torch.device(device)

        # device + eval
        if hasattr(self.model, "to"):   self.model.to(self.device)
        if hasattr(self.model, "eval"): self.model.eval()

        # Writer for JSONL (records); separate scalar writer can be passed to MetricManager
        out_dir = self.eval_cfg.get("logging", {}).get("out_dir", "outputs/eval")
        self.writer = JSONLWriter(out_dir)

        # Build MetricManager
        self.metrics = MetricManager(
            metrics=self.eval_cfg.get("metrics", []),
            writer=writer,
        )

        # mapping for in/out
        self.mapping = self.model_cfg['model_inputs']

    @torch.no_grad()
    def run(self, max_batches: Optional[int] = None):
        n = 0
        for batch in self.dl:
            bsz = _infer_batch_size(batch)
            for i in range(bsz):
                rec = self._process_one(batch, i, step=n)
                # TODO: Replace writer with wandb or tensorboard logging
                self.writer.write(rec)
            n += 1
            if max_batches is not None and n >= max_batches:
                break
        # TODO: Log dataset aggregated metrics using self.metrics 

        self.writer.close()

    def _process_one(self, batch: Dict[str, Any], i: int, step: Optional[int] = None) -> Dict[str, Any]:
        inputs = self._pack_for_model(batch, i)
        outputs = self.model(**inputs)
        merged_dict = tu.merge_dict(('inputs', inputs), ('outputs', outputs))

        # compute & accumulate metrics via the manager
        metrics_now = self.metrics.update(merged_dict, step=step, n=1)

        # TODO: Add visualization for the image if available

        infos = _pluck(batch, "infos", i) or {}
        return {
            "sequence": infos.get("sequence", None),
            "metrics": metrics_now,
            # Add image to record
        }

    # ---------- helpers ----------
    def _pack_for_model(self, batch: Dict[str, Any], i: int) -> Dict[str, Any]:
        """Produce {out_key: sample_i_of(batch[in_key])} on the correct device."""
        out: Dict[str, Any] = {}
        for m in self.mapping:
            in_key = m.get("in_key")
            out_key = m.get("out_key", in_key)
            if in_key not in batch:
                continue
            v = batch[in_key]
            if torch.is_tensor(v):
                out[out_key] = v[i].to(self.device)
            elif isinstance(v, (list, tuple)):
                out[out_key] = v[i]
            elif isinstance(v, dict):
                out[out_key] = _index_dict(v, i, self.device)
            else:
                out[out_key] = v
        return out

# ----------------------------
# Utilities
# ----------------------------
def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for v in batch.values():
        if torch.is_tensor(v):
            return v.shape[0]
        if isinstance(v, list) and v:
            return len(v)
        if isinstance(v, dict):
            for vv in v.values():
                if torch.is_tensor(vv):
                    return vv.shape[0]
                if isinstance(vv, list) and vv:
                    return len(vv)
    return 1

def _pluck(batch: Dict[str, Any], key: str, i: int):
    if key not in batch:
        return None
    v = batch[key]
    if isinstance(v, list):
        return v[i]
    if isinstance(v, dict):
        return v if i == 0 else v
    return v

def _index_dict(d: Dict[str, Any], i: int, device: torch.device):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v[i].to(device)
        elif isinstance(v, list):
            out[k] = v[i]
        elif isinstance(v, dict):
            out[k] = _index_dict(v, i, device)
        else:
            out[k] = v
    return out