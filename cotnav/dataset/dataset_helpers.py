from pathlib import Path
from typing import Dict
import numpy as np

from cotnav.utils.math_utils import (se3_matrix)

def get_mission_id(mission_path, cfg):
    dataset, robot = get_dataset_info(cfg)

    mission_path = Path(mission_path)
    if dataset == 'grandtour':
        if mission_path.is_file():
            mission_path = mission_path.parent
        mission_name = mission_path.stem.split('_')[-1]
        return dataset, robot, mission_name
    else:
        raise NotImplementedError

def set_dataset_dir(save_dir, cfg: Dict, bag_paths):
    dataset, robot = get_dataset_info(cfg)
    assert len(bag_paths) > 0, "set_dataset_dir failed. bag_paths must be > 0"

    if dataset == "scand":
        did0 = Path(bag_paths[0]).stem
        return Path(save_dir) / f"{dataset}_{robot}" / f"mission_{did0}"
    elif dataset == "grandtour":
        did0 = Path(bag_paths[0]).stem.split('_')[0]
        return Path(save_dir) / f"mission_{did0}"
    else:
        raise NotImplementedError

def get_dataset_info(cfg: Dict):
    assert 'dataset' in cfg, "The 'dataset' key is missing from the cfg"
    dataset = cfg['dataset']

    if dataset == 'grandtour':
        robot = 'anymal'
    elif dataset == 'scand':
        robot = 'spot'

    return dataset, robot

"""BEGIN DATASET SPECIFIC HELPERS"""

def convert_grandtour_odom(odom_np, ts_np, tf_manager):
    """Convert GrandTour odometry to world frame SE3 matrices."""

    # Get base to front camera transform
    T_alpha_base = tf_manager.get_transform("base", "alphasense_front_center") # tgt, src
    p_alpha_base = T_alpha_base[:3, 3]

    # Get base to odom transforms
    T_hesai_odom = se3_matrix(odom_np[:, 1:4], odom_np[:, 4:8])
    T_base_hesai = tf_manager.get_transform("hesai_lidar", "base") # tgt, src
    T_base_odom = T_hesai_odom @ T_base_hesai

    # Transform T^base_odom -> T^alpha_odom -> T^alpha_local
    T_alpha_odom = T_base_odom
    T_alpha_odom[:, :3, 3] += p_alpha_base
    T_alpha_local = np.linalg.inv(T_alpha_odom[0]) @ T_alpha_odom
    
    return T_alpha_local


"""END DATASET SPECIFIC HELPERS"""