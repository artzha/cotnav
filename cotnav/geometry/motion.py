
from __future__ import annotations
import math
from typing import List, Optional
import numpy as np

_EPS = 1e-8

class ConstantCurvatureArc:
    """Represents a constant-curvature arc with a given curvature and length."""
    def __init__(self, curvature: float, length: float):
        self.curvature = curvature
        self.length = length

    def __repr__(self):
        return f"ConstantCurvatureArc(curvature={self.curvature}, length={self.length})"
    
    def xy_at_s(self, s: float):
        k = self.curvature
        if abs(k) < _EPS:
            return (s, 0.0)
        r = 1.0 / k
        a = s * k     # swept angle
        return (r * np.sin(a), r * (1.0 - np.cos(a)))

class MotionTemplateLibrary:
    """Creates a bank of constant-curvature arcs."""
    def __init__(self, max_curvature: float, max_path_len: float, num_options: int):
        assert num_options >= 2
        self.max_curvature = float(max_curvature)
        self.max_path_len  = float(max_path_len)
        self.num_options   = int(num_options)

        # Uniform in curvature like the C++ example
        self.curvatures = np.linspace(-self.max_curvature, self.max_curvature, self.num_options)

    def arcs(self) -> List[ConstantCurvatureArc]:
        return [ConstantCurvatureArc(float(k), self.max_path_len) for k in self.curvatures]
    
# def project_xyz_to_uv(xyz, intrinsics, T_base_to_optical):
#     assert xyz.shape[1] == 3
#     xyz_homo = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

#     P = compute_projection_matrices(
#         intrinsics,
#         T_cam_base=np.linalg.inv(T_base_to_optical)
#     )['T_base_to_optical']

#     uvd = (P @ xyz_homo.T).T
#     z = uvd[:, 2]
#     eps = 1e-9
#     z_safe = np.where(np.abs(z) < eps, eps, z)
#     u, v = uvd[:, 0] / z_safe, uvd[:, 1] / z_safe

#     H, W = intrinsics['image_height'], intrinsics['image_width']
#     valid_mask = (z_safe > 0) & (u >= 0) & (v >= 0) & (u < W) & (v < H)

#     uv_all = np.stack([u, v], axis=-1)   # keep *all* arcs, original order
#     return uv_all, valid_mask
