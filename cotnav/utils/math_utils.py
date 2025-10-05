
import numpy as np
from scipy.spatial.transform import Rotation as R
from cotnav.utils.log_utils import logging

try:
    from pymlg import SO3, SE3
except ImportError:
    logging.error("Please install 'pymlg' package to use this script.")
    raise ImportError("Missing 'pymlg' package. Install it with: pip install pymlg")


# ── tiny helpers to “batchify” scalar SE3 ops ───────────────────────────
def se3_inverse_batch(Ts: np.ndarray) -> np.ndarray:         # (B,4,4)
    return np.stack([SE3.inverse(T) for T in Ts], axis=0)

def se3_log_batch(Ts: np.ndarray) -> np.ndarray:             # (B,6)
    return np.stack([SE3.Log(T).reshape(-1) for T in Ts], 0)

def se3_exp_batch(xis: np.ndarray) -> np.ndarray:            # (B,4,4)
    return np.stack([SE3.Exp(xi) for xi in xis], axis=0)

# ── quat+xyz → SE3 ----------------------------------------------------------

def se3_matrix(xyz: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """
    Given arrays x, y, z of shape (N,) and quaternion q_wxyz of shape (N, 4),
    return an array of SE(3) mats shape (N,4,4).
    """
    N = xyz.shape[0]
    Rt = R.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # Convert to SciPy's (x y z w) order

    T = np.zeros((N, 4, 4), dtype=float)
    T[:, :3, :3] = Rt
    T[:, :3, 3] = xyz
    T[:, 3, 3] = 1.0

    return T

def quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def angular_diff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def interpolate_se3(
    cam_ts:     np.ndarray,        # (F,)
    odo_ts:     np.ndarray,        # (N,)
    odo_xyz:    np.ndarray,        # (N,3)
    odo_q_wxyz: np.ndarray         # (N,4)  qw qx qy qz
) -> np.ndarray:                   # → (F,8) ts x y z  qw qx qy qz
    # 1) Convert odometry poses to SE3 matrices
    T_odo = se3_matrix(odo_xyz, odo_q_wxyz)             # (N,4,4)

    # 2) Bracket each camera timestamp
    idx0 = np.searchsorted(odo_ts, cam_ts, side="right") - 1
    idx0 = np.clip(idx0, 0, len(odo_ts) - 2)
    idx1 = idx0 + 1
    alpha = ((cam_ts - odo_ts[idx0]) /
             (odo_ts[idx1] - odo_ts[idx0] + 1e-6))[:, None]     # (F,1)

    T0, T1 = T_odo[idx0], T_odo[idx1]                    # (F,4,4)
    # 3) Relative motion & SE3 interpolation
    Delta = se3_inverse_batch(T0) @ T1                   # (F,4,4)
    xi    = se3_log_batch(Delta)                         # (F,6)
    T_d   = se3_exp_batch(alpha * xi)                    # (F,4,4)
    T_cam = T0 @ T_d                                     # (F,4,4)

    # 4) Unpack xyz + quaternion (qw qx qy qz)
    xyz_cam = T_cam[:, :3, 3]
    try:
        q_xyzw  = R.from_matrix(T_cam[:, :3, :3]).as_quat()
    except Exception as e:
        import pdb; pdb.set_trace()
        logging.error(f"Failed to convert rotation matrix to quaternion: {e}")
    q_wxyz  = q_xyzw[:, [3, 0, 1, 2]]
    return np.hstack([cam_ts.reshape(-1, 1), xyz_cam, q_wxyz])                  # (F,7)

def odom_to_local_pose(
    odom: np.ndarray,
    mode: str = "se3",
) -> np.ndarray:
    """
    Convert a global odometry trace to **local coordinates w.r.t. the first pose**.

    Parameters
    ----------
    odom : (N,8) ndarray
        Global poses ordered `[x  y  z  qw  qx  qy  qz]`
        (scalar–first quaternion, right-handed).
    mode : {"se2","se3"}, default="se2"
        • "se2" → return local **(x,y,yaw)** – identical to the old function.  
        • "se3" → return local **(x,y,z,qw,qx,qy,qz)**.

    Returns
    -------
    out :
        - If *mode="se2"*  → shape **(N,3)**  = `[x_local, y_local, yaw_local]`
        - If *mode="se3"*  → shape **(N,7)**  = `[x_local, y_local, z_local, qw, qx, qy, qz]`
    """
    if odom.shape[1] != 8:
        raise ValueError("odom must be (N,8) with [ts, x y z qw qx qy qz]")

    if mode not in {"se3"}:
        raise ValueError("mode must be 'se3'")

    odom_ts   = odom[:, 0:1]          # (N,1)
    xyz_world = odom[:, 1:4]          # (N,3)
    q_wxyz    = odom[:, 4:]          # (N,4) scalar-first

    # ------------------------------------------------------------------ #
    #                      3-D  (x,y,z,q)   branch                       #
    # ------------------------------------------------------------------ #
    # Build SE(3) matrices
    Tt_world = se3_matrix(
        xyz_world[:,:3], q_wxyz
    )                                                # (N,4,4)

    T0_inv   = np.linalg.inv(Tt_world[0])
    Tt_local = T0_inv @ Tt_world                     # (N,4,4)

    # local translation
    xyz_local = Tt_local[:, :3, 3]                   # (N,3)

    # local rotation → quaternion (scalar-first)
    Rt_local  = Tt_local[:, :3, :3]
    q_xyzw_local = R.from_matrix(Rt_local).as_quat() # (N,4) xyzw
    q_wxyz_local = q_xyzw_local[:, [3, 0, 1, 2]]     # back to wxyz

    return np.column_stack([odom_ts, xyz_local, q_wxyz_local])

def transform_poses(
    odom: np.ndarray,               # (N,8) ts x y z qw qx qy qz
    T_world_newworld: np.ndarray    # (4,4) newworld ← world
) -> np.ndarray:                   # → (N,8) ts x y z qw qx qy qz
    """
    Transform a global odometry trace to a new world frame.

    Parameters
    ----------
    odom : (N,8) ndarray
        Global poses ordered `[ts x  y  z  qw  qx  qy  qz]`
        (scalar–first quaternion, right-handed).
    T_world_newworld : (4,4) ndarray
        SE(3) matrix transforming points in the *old* "world" frame to the *new* "newworld" frame.

    Returns
    -------
    out : (N,8) ndarray
        Transformed global poses ordered `[ts x  y  z  qw  qx  qy  qz]`
        (scalar–first quaternion, right-handed).
    """
    if odom.shape[1] != 8:
        raise ValueError("odom must be (N,8) with [ts, x y z qw qx qy qz]")
    if T_world_newworld.shape != (4, 4):
        raise ValueError("T_world_newworld must be a (4,4) SE(3) matrix")

    odom_ts   = odom[:, 0:1]          # (N,1)
    xyz_world = odom[:, 1:4]          # (N,3)
    q_wxyz    = odom[:, 4:]           # (N,4) scalar-first

    # Build SE(3) matrices
    Tt_world = se3_matrix(
        xyz_world[:,:3], q_wxyz
    )                                                # (N,4,4)

    Tt_newworld = T_world_newworld @ Tt_world       # (N,4,4)

    # newworld translation
    xyz_newworld = Tt_newworld[:, :3, 3]            # (N,3)

    # newworld rotation → quaternion (scalar-first)
    Rt_newworld  = Tt_newworld[:, :3, :3]
    q_xyzw_newworld = R.from_matrix(Rt_newworld).as_quat() # (N,4) xyzw
    q_wxyz_newworld = q_xyzw_newworld[:, [3, 0, 1, 2]]     # back to wxyz
    return np.column_stack([odom_ts, xyz_newworld, q_wxyz_newworld])  # (N,8)