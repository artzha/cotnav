from pathlib import Path
from typing import Dict

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