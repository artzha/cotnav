"""
Utility functions for loading configurations and data.
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple
from pytransform3d.transform_manager import TransformManager

from cotnav.utils.log_utils import logging
from cotnav.geometry.camera import Calib, compute_projections

def construct_filters(cfg):
    """Loops throgh and constructs filters"""
    filters = {}
    if not cfg.get('filters', None):
        return filters

    for filter_dict in cfg['filters']:
        name = filter_dict['name']
        filters[name] = {
            'type': filter_dict['type'],
            'params': filter_dict['params']
        }
    return filters

def build_transforms(tf_yaml: str | Path) -> TransformManager:
    """
    Build a TransformManager from a YAML dict whose entries look like:
      "src dst": [16 floats]  # row-major 4x4
    or:
      "src dst": [12 floats]  # row-major 3x4 (we append [0 0 0 1])
    """
    tf_cfg = yaml.safe_load(Path(tf_yaml).read_text())
    tm = TransformManager()

    # allow either a top-level dict or nested under "transforms"
    items = tf_cfg.get("transforms", tf_cfg)

    def add_pair(key: str, mat_like: Any) -> None:
        parts = key.strip().split(None, 1)  # split on first whitespace
        if len(parts) != 2:
            return  # skip malformed keys
        src, dst = parts[0], parts[1]

        arr = np.asarray(mat_like, dtype=float).ravel()
        if arr.size == 16:
            T = arr.reshape(4, 4)
        elif arr.size == 12:
            T = np.eye(4, dtype=float)
            T[:3, :] = arr.reshape(3, 4)
        else:
            raise ValueError(f"Transform '{key}' must have 12 or 16 numbers, got {arr.size}")

        tm.add_transform(src, dst, T)

    if isinstance(items, dict):
        for k, v in items.items():
            add_pair(k, v)
    elif isinstance(items, list):
        # also tolerate a list of { "src dst": [..] } maps
        for entry in items:
            for k, v in entry.items():
                add_pair(k, v)
    else:
        raise ValueError("TF YAML must be a dict (or list of dicts) with keys like 'src dst'.")

    return tm

def load_intrinsics(cam_yaml: str | Path, tf_yaml: str | Path, world_frame: str) -> Calib:
    cam_cfg = yaml.safe_load(Path(cam_yaml).read_text())
    K = np.array(cam_cfg["K"], dtype=float).reshape(3, 3)
    D = np.array(cam_cfg.get("D", []), dtype=float)
    R = np.array(cam_cfg.get("R", np.eye(3)), dtype=float).reshape(3, 3)
    P = np.array(cam_cfg.get("P", []), dtype=float).reshape(3, 4) if "P" in cam_cfg else np.hstack((K, np.zeros((3, 1))))
    H, W = int(cam_cfg["image_height"]), int(cam_cfg["image_width"])
    cam_frame = cam_cfg["frame_id"]
    tm = build_transforms(tf_yaml)

    # Compose across the graph (e.g., "map base" + "base camera")
    T_world_cam = tm.get_transform(cam_frame, world_frame) # tgt src
    P_world_pix, B_pix_world = compute_projections(K, R, T_world_cam)

    return Calib(
        K=K, D=D, R=R, P=P, size_hw=(H, W), T_world_cam=T_world_cam,
        P_world_pix=P_world_pix, B_pix_world=B_pix_world
    )

def load_odom(odom_path, format='numpy'):
    assert format in ['numpy', 'pandas'], "Unsupported odometry loader format. Use 'numpy' or 'pandas'."
    try:
        odometry = pd.read_csv(odom_path, header=0)

        # Rearrange colummns to [timestamp, x, y, z, qw, qx, qy, qz]
        if not {"timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"}.issubset(odometry.columns):
            logging.error(f"[ERROR] odometry_data.csv in {odom_path} is missing required columns.")
            return None
        
        odometry = odometry[["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"]].dropna()
        if format == 'pandas':
            return odometry.astype(np.float64)
        elif format == 'numpy':
            odometry_np = odometry.to_numpy(dtype=np.float64)
            if odometry_np.shape[1] != 8:
                logging.error(f"[ERROR] odometry_data.csv in {odom_path} has incorrect number of columns.")
                return None
            return odometry_np

    except Exception as e:
        logging.error(f"Failed to load odometry data from {odom_path}: {e}")
        return None

def load_timestamps(timestamps_path):
    """
    Loads timestamps from a CSV file.
    Returns a numpy array of timestamps.
    """
    try:
        timestamps = np.loadtxt(timestamps_path, delimiter=',', skiprows=1, dtype=np.float64)
        if timestamps.ndim == 1:
            timestamps = timestamps.reshape(-1, 1)
        return timestamps[:, -1] # [N,]
    except Exception as e:
        logging.error(f"Failed to load timestamps from {timestamps_path}: {e}")
        return None

if __name__ == "__main__":
    cam_info_path = "/robodata/arthurz/Research/cotnav/data/grandtour_raw/mission_2024-10-01-11-29-55/front_camera_info_anymal.yaml"
    tf_path = "/robodata/arthurz/Research/cotnav/data/grandtour_raw/mission_2024-10-01-11-29-55/tf_static_anymal.yaml"

    calib = load_intrinsics(cam_info_path, tf_path, world_frame="base")
    print("Calibration loaded successfully.")
    print("Calib: ", calib)