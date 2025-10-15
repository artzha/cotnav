
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

    def sample_along_arc(self, num_samples: int) -> np.ndarray:
        """Sample points along the arc in the local frame."""
        assert num_samples >= 2
        ss = np.linspace(0, self.length, num_samples)
        points = [self.xy_at_s(s) for s in ss]
        return np.array(points)  # Shape (num_samples, 2)

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
