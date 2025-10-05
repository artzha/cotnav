import numpy as np

def interpolate_mask_to_timestamps( 
    src_timestamps: np.ndarray,
    src_mask: np.ndarray,
    target_timestamps: np.ndarray
) -> np.ndarray:
    """
    Nearest-neighbor “interpolation” of a boolean mask from GPS times to target times.

    Args:
        src_timestamps    (shape=(N_ts,))         : sorted array of timestamps
        src_mask          (shape=(N_ts,))         : boolean array at each timestamp
        target_timestamps (shape=(N_timestamps,))  : sorted array of target timestamps

    Returns:
        np.ndarray of shape (N_timestamps,) of booleans,
        where each entry is gps_mask at the GPS timestamp nearest to that target timestamp.
    """
    src_ts = np.asarray(src_timestamps).flatten()
    src_b  = np.asarray(src_mask).flatten()
    tgt_ts = np.asarray(target_timestamps).flatten()

    # Find insertion indices: idx[i] is first j such that gps_ts[j] >= tgt_ts[i]
    idx = np.searchsorted(src_ts, tgt_ts, side="left")

    # Compute candidate indices to pick from gps_b
    ng = len(src_ts)
    left = np.clip(idx - 1, 0, ng - 1)
    right = np.clip(idx,     0, ng - 1)

    # Distances to left and right neighbors
    dist_left  = tgt_ts - src_ts[left]    # could be negative if idx=0, left=0
    dist_right = src_ts[right] - tgt_ts   # could be negative if idx>=ng, right=ng-1

    # By default, pick left; override when right is strictly closer
    use_right = dist_right < dist_left
    chosen_idx = np.where(use_right, right, left)

    return src_b[chosen_idx]