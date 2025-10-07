from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import yaml

@dataclass
class Calib:
    K: np.ndarray                 # (3,3)
    R: np.ndarray                 # (3,3)
    D: np.ndarray                 # (N,)
    P: np.ndarray                 # (3,4)
    size_hw: Tuple[int, int]      # (H, W)
    T_world_cam: np.ndarray       # (4,4) world->camera
    P_world_pix: np.ndarray       # (3,4) world->pixel
    B_pix_world: np.ndarray       # (4,4) pixel->world

def compute_projections(K: np.ndarray, R: np.ndarray, T_world_cam: np.ndarray) -> None:
    """
    Given a calibration and extrinsic matrix, compute the projection matrices
    """
    assert T_world_cam.shape == (4, 4)
    assert K.shape == (3, 3)

    # Compute P_world_pix = K [I | 0] T_world_cam
    A = np.eye(4)
    A[:3, :3] = R[:3, :3]
    P_world_pix = K @ A[:3, :] @ T_world_cam

    # Compute B_pix_world = T_cam_world [A_inv | 0] K_inv
    Kinv = np.eye(4)
    Kinv[:3, :3] = np.linalg.inv(K)
    A[:3, :3] = R.T
    T_cam_world = np.linalg.inv(T_world_cam)

    B_pix_world = T_cam_world @ A @ Kinv

    return P_world_pix, B_pix_world

def project_to_pixel(
    xyz: np.ndarray,            # (N,3) points in world frame
    calib: Calib
) -> np.ndarray:         # → (N,2) pixel coords
    """
    Project 3D points in the world frame to pixel coordinates using the
    provided calibration.
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    N = xyz.shape[0]

    # Convert to homogeneous (N,4)
    xyz_h = np.hstack([xyz, np.ones((N, 1))])

    # Project to pixel homogeneous (N,3)
    pix_h = (calib.P_world_pix @ xyz_h.T).T

    # Normalize to get pixel coordinates (N,2)
    pix = pix_h[:, :2] / (pix_h[:, 2:3] + 1e-6)

    # Clip to image bounds
    image_h, image_w = calib.size_hw
    valid_mask = ((pix[:, 0] >= 0) & (pix[:, 0] < image_w) &
                  (pix[:, 1] >= 0) & (pix[:, 1] < image_h))
    
    if not valid_mask.any():
        return None, valid_mask

    return pix, valid_mask