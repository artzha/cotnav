import numpy as np
from scipy.spatial.distance import cdist



def sample_arc_xy(arc, max_len_m: float, samples_per_meter: int = 10) -> np.ndarray:
    """Return Nx2 XY samples (in base frame) along arc up to max_len_m."""
    s_end = float(min(max_len_m, getattr(arc, "length", max_len_m)))
    n = max(2, int(samples_per_meter * max(0.05, s_end)))
    s_vals = np.linspace(0.0, s_end, n)
    return np.array([arc.xy_at_s(float(s)) for s in s_vals], dtype=np.float32)  # (N,2)

def hausdorff_xy(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two polylines A(N,2) and B(M,2)."""
    if A.size == 0 or B.size == 0: 
        return np.inf
    D = cdist(A, B)  # (N,M)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))

def gt_local_polyline_from_odom_corrected(
    interp_odom: np.ndarray, 
    T_base_local: np.ndarray, 
    i: int, 
    lookahead_m: float = 6.0
) -> np.ndarray:
    """
    Extract forward-looking ground truth polyline using the existing coordinate transformation pipeline.
    
    This function reuses the coordinate transformations already computed in the notebook:
    - Uses interpolated odometry (interp_odom) instead of raw odometry
    - Uses the existing T_base_local transformation that accounts for proper frame conversions
    
    Parameters:
    -----------
    interp_odom : np.ndarray, shape (N, 8)
        Interpolated odometry array [ts, x, y, z, qw, qx, qy, qz] 
        (already aligned to camera timestamps)
    T_base_local : np.ndarray, shape (N, 4, 4)
        Pre-computed transformation matrices from global to local frame
        (accounts for hesai_lidar → base → local frame transformations)
    i : int
        Index of the current frame in the interpolated odometry
    lookahead_m : float
        Maximum along-track distance to include in the polyline
        
    Returns:
    --------
    np.ndarray, shape (K, 2)
        Local polyline coordinates (x, y) in the local frame of pose i
    """
    # Step 1: Extract positions in local frame (reusing existing computation)
    # T_base_local already contains the transformation to the local frame of the first pose
    # We need to adjust to make the frame relative to pose i instead
    T_current_local = np.linalg.inv(T_base_local[i]) @ T_base_local  # Transform to frame of pose i
    p_local = T_current_local[:, :3, 3]  # Extract positions (x, y, z)

    # Step 2: Extract future trajectory segment (xy only)
    xy_local = p_local[:, :2]  # Take only x, y coordinates
    future_segment = xy_local[i:]  # Take poses from current index onwards
    
    if len(future_segment) < 2:
        return future_segment

    # Step 3: Compute cumulative along-track distance and truncate
    deltas = np.diff(future_segment, axis=0)  # (K-1, 2)
    distances = np.sqrt((deltas**2).sum(axis=1))  # (K-1,)
    cum_distances = np.concatenate([[0.0], np.cumsum(distances)])  # (K,)

    # Step 4: Find cutoff index based on lookahead distance
    cutoff_idx = int(np.searchsorted(cum_distances, lookahead_m, side="right"))
    cutoff_idx = max(1, min(cutoff_idx, len(future_segment)-1))
    
    return future_segment[:cutoff_idx+1]  # (K,2)