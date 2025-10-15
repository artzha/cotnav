# metric_manager.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from scipy.spatial.distance import cdist
import torch
    
class MetricManager:
    """
    Lightweight metric aggregator.

    Each metric config can be one of:
      - {"fn": "package.module:function_name", "name": "optional_alias", "pred_key": "...", "lab_key": "...", "kwargs": {...}}
      - {"name": "compute_accuracy", "pred_key": "...", "lab_key": "...", "kwargs": {...}}  # resolved via `registry`

    registry: optional dict[str, Callable] to resolve short names.
    writer: optional summary writer (must support .add_scalar(tag, scalar_value, global_step=step))
    """

    def __init__(self,
                 metrics: List[Dict[str, Any]],
                 writer: Any = None):
        self.writer = writer
        self._cfgs: Dict[str, Dict[str, Any]] = {}
        self._fns: Dict[str, Callable] = {}
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

        # Optional streaming store for special metrics (e.g., precision/recall curves)
        self._stream: Dict[str, Dict[str, np.ndarray]] = {}

        # Resolve each metric to a callable
        for cfg in metrics:
            # display name
            disp = cfg.get("name")
            fn = None
            key = cfg["name"]
            try:
                fn = globals()[key]
            except KeyError:
                raise ValueError(f"Metric '{key}' not found in globals.")

            self._cfgs[disp] = cfg
            self._fns[disp] = fn
            self._totals[disp] = 0.0
            self._counts[disp] = 0

    def reset(self):
        for k in self._totals:
            self._totals[k] = 0.0
            self._counts[k] = 0
        self._stream.clear()

    def set_writer(self, writer: Any):
        self.writer = writer

    def update(self,
               tensor_dict: Dict[str, Any],
               step: Optional[int] = None,
               n: int = 1) -> Dict[str, Any]:
        """
        Compute all metrics for the current sample (or mini-batch),
        update running averages, and return the instantaneous values.

        Each metric config may specify:
          - pred_key: key inside predictions or model_inputs (predictions searched first)
          - lab_key:  key inside predictions or model_inputs (model_inputs searched second)
          - kwargs:   extra args to pass to the metric function

        Special-case: if prediction tensor has ndim==5 (B,E,...) → mean over ensemble (dim=1).
        """
        out: Dict[str, Any] = {}
        for name, fn in self._fns.items():
            cfg = self._cfgs[name]
            kwargs = dict(cfg.get("kwargs", {}) or {})
            
            metric_inputs = { k: tensor_dict[v] for k, v in kwargs.items() }
            try:
                val = fn(**metric_inputs)
            except Exception:
                print(f"Warning: metric '{name}' failed with inputs {metric_inputs}.")
                raise

            # Convert scalar
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()
                val = val.item() if val.numel() == 1 else val.mean().item()

            if isinstance(val, np.ndarray) and val.size == 1:
                val = float(val.reshape(()))

            out[name] = val

            # Update running stats
            self._totals[name] += float(val) * n
            self._counts[name] += n

            # Optional writer
            if self.writer is not None and hasattr(self.writer, "add_scalar"):
                self.writer.add_scalar(name, float(val), global_step=0 if step is None else step)

        return out

    def averages(self) -> Dict[str, float]:
        """Return running averages for each metric."""
        avg = {}
        for k in self._totals:
            c = max(1, self._counts[k])
            avg[k] = self._totals[k] / c
        return avg

def compute_accuracy(arcs: np.ndarray, pred: np.ndarray, odom: np.ndarray) -> float:
    """
    arcs: (B, K, N, 3), pred: (B, N, 3), odom: (B, M, 3)
    Returns average accuracy over batch.
    """
    
    if isinstance(arcs, torch.Tensor):
        arcs, pred, odom = arcs.detach().cpu().numpy(), pred.detach().cpu().numpy(), odom.detach().cpu().numpy()
    B, K, N, _ = arcs.shape
    if odom.ndim == 2:
        odom = odom[None, ...]  # (1, M, 3)

    # Get best prediction for each batch element (B,)
    
    # Compute Hausdorff distances for all combinations (B, K)
    hausdorff_distances = np.zeros((B, K))
    hausdorff_distances_pred = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            hausdorff_distances[b, k] = hausdorff_xyz(arcs[b, k], odom[b])
            hausdorff_distances_pred[b, k] = hausdorff_xyz(arcs[b, k], pred[b])
    
    selected_k_indices = np.argmin(hausdorff_distances_pred, axis=1)
    # Find ground truth indices (best Hausdorff distance for each batch) (B,)
    ground_truth_indices = np.argmin(hausdorff_distances, axis=1)
    
    # Compute accuracy as fraction of correct predictions
    acc = np.mean(selected_k_indices == ground_truth_indices)
    
    return acc

def compute_hausdorff_distance(arcs: np.ndarray, pred: np.ndarray, odom: np.ndarray, oneway=True) -> float:
    """
    arcs: (B, K, N, 3), pred: (B, N, 3), odom: (B, M, 3)
    Returns average hdist(model, odom) over batch.
    """
    if isinstance(arcs, torch.Tensor):
        arcs, pred, odom = arcs.detach().cpu().numpy(), pred.detach().cpu().numpy(), odom.detach().cpu().numpy()
    B, K, N, _ = arcs.shape
    if odom.ndim == 2:
        odom = odom[None, ...]  # (1, M, 3)
    total_distance = 0.0
    for b in range(B):
        total_distance += hausdorff_xyz(pred[b], odom[b], oneway=oneway)
    return total_distance / B

def compute_relative_hausdorff_distance(arcs: np.ndarray, pred: np.ndarray, odom: np.ndarray, oneway=True) -> float:
    """
    arcs: (B, K, N, 3), pred: (B, N, 3), odom: (B, M, 3)
    Returns average hdist(model arc, odom arc)
    """
    if isinstance(arcs, torch.Tensor):
        arcs, pred, odom = arcs.detach().cpu().numpy(), pred.detach().cpu().numpy(), odom.detach().cpu().numpy()
    B, K, N, _ = arcs.shape
    if odom.ndim == 2:
        odom = odom[None, ...]  # (1, M, 3)
    total_distance = 0.0
    # Compute Hausdorff distances for all combinations (B, K)
    hausdorff_distances = np.zeros((B, K))
    hausdorff_distances_pred = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            hausdorff_distances[b, k] = hausdorff_xyz(arcs[b, k], odom[b])
            hausdorff_distances_pred[b, k] = hausdorff_xyz(arcs[b, k], pred[b])

    total_distance = np.sum(np.min(hausdorff_distances_pred, axis=1) - np.min(hausdorff_distances, axis=1))
    return total_distance / B

def hausdorff_xyz(A: np.ndarray, B: np.ndarray, oneway=True) -> float:
    """Oneway Hausdorff distance between two polylines A(N,3) and B(M,3)."""
    if A.size == 0 or B.size == 0: 
        return np.inf
    D = cdist(A, B)  # (N,M)
    if oneway:
        return float(D.min(axis=1).max())
    else:
        return float(max(D.min(axis=1).max(), D.min(axis=0).max()))

if __name__ == "__main__":
    print("Testing compute_accuracy with specific probabilities...")