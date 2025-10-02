from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd

from scripts.utils.log_utils import logging
from scripts.preprocessing.bag_utils import (
    get_ros_type, 
    process_rgb,
    process_odometry,
    process_tf,
    transforms_to_yaml
)
from cotnav.dataset.dataset_helpers import (get_dataset_info, set_dataset_dir)

DEBUG_MODE=False

# ───────────────────────── local bag indexing ────────────────────────────
def index_bags_by_topic_local(
    session_dir: str | Path,
    topics_dict: Dict[str, Dict[str, any]],
) -> Dict[str, List[str]]:
    """
    Scan every *.bag under session_dir and index which bags contain which ROS topics
    declared in topics_dict (keys mirror your previous 'key_seg' blocks).

    Returns
    -------
    bags_dict: { key_seg -> [bag_path, ...] }
    """
    session_dir = Path(session_dir)
    all_bags = sorted(str(p) for p in session_dir.rglob("*.bag"))
    if not all_bags:
        return {}

    # Normalize to a map: key_seg -> list_of_ros_topics
    def _ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    target_topics: Dict[str, List[str]] = {
        key: _ensure_list(spec["ros_topic"])
        for key, spec in topics_dict.items()
    }

    # For each bag, open once and see which target topics are present
    from rosbags.highlevel import AnyReader
    bags_dict: Dict[str, List[str]] = defaultdict(list)

    for bag in all_bags:
        try:
            with AnyReader([Path(bag)]) as reader:
                avail = {c.topic for c in reader.connections}
                for key, ros_topics in target_topics.items():
                    if any(t in avail for t in ros_topics):
                        bags_dict[key].append(bag)
        except Exception as e:
            logging.warning(f"[index] Failed to read {bag}: {e}")
            continue
    return bags_dict

# ───────────────────────── local orchestrator  ───────────────────────────
def inspect_and_stream_local(
    cfg: Dict, 
    sess_dir: str | Path,
    save_root_dir: str | Path,
    topics_dict: Dict[str, Dict[str, any]],
    filters_dict: Dict[str, any] | None = None,
    debug_mode=False
):
    """
    Local-only counterpart to inspect_and_stream:
      * sess_dir: local directory containing *.bag (recursively)
      * topics_dict: same structure as before (keys like 'rgb', 'odometry', 'control', 'tf_static', etc.)
    """
    global DEBUG_MODE
    DEBUG_MODE = debug_mode
    sess_dir = Path(sess_dir)
    save_root_dir = Path(save_root_dir)

    # Index bags by topic groups (keys in topics_dict)
    bags_dict = index_bags_by_topic_local(sess_dir, topics_dict)
    if not bags_dict:
        logging.info(f"[local] No matching bags in {sess_dir}")
        return pd.DataFrame()
    
    # A simple parent/child naming scheme for local sessions:
    dataset, robot = get_dataset_info(cfg)
    child_groups = {"chunk0": []}       # single chunk by default

    # -------------------- process odometry (optional) ---------------------
    odom_key = next((k for k, v in topics_dict.items() if v["save_prefix"] == "odometry"), None)
    odom_df = None
    if odom_key and bags_dict.get(odom_key):
        try:
            odom_art = process_odometry(
                bags=bags_dict[odom_key],
                topic=topics_dict[odom_key]["ros_topic"],
                out_dir="",
                save_prefix=topics_dict[odom_key]["save_prefix"],
                save_file=False
            )
            if odom_art:
                odom_df = odom_art["df"]
        except Exception as e:
            logging.warning(f"[local] Odometry extraction failed in {sess_dir}: {e}")
            odom_df = None
    else:
        logging.info(f"[local] No odometry found in {sess_dir}")

    # -------------------- process static TFs (optional) -------------------
    tf_static_key = next((k for k, v in topics_dict.items()
                          if v.get("save_prefix") == "tf_static"), None)
    static_transforms = None
    if tf_static_key and bags_dict.get(tf_static_key):
        try:
            tf_art = process_tf(
                bag=bags_dict[tf_static_key][0],
                topics=topics_dict[tf_static_key]["ros_topic"],
                frame_pairs=topics_dict[tf_static_key].get("frames", [])
            )
            static_transforms = tf_art.get("transforms", {})
        except Exception as e:
            logging.warning(f"[local] TF static extraction failed in {sess_dir}: {e}")
   
    # -------------------- process controls (optional) ---------------------
    ctrl_key = next((k for k, v in topics_dict.items()
                     if v.get("save_prefix") == "control"), None)
    ctrl_df = None
    if ctrl_key and bags_dict.get(ctrl_key):
        try:
            ctrl_art = process_controls(
                bags=bags_dict[ctrl_key],
                topic=topics_dict[ctrl_key]["ros_topic"],
                out_dir="",
                save_prefix=topics_dict[ctrl_key]["save_prefix"],
                save_file=False,
            )
            if ctrl_art:
                ctrl_df = ctrl_art["df"]
        except Exception as e:
            logging.warning(f"[local] Controls extraction failed in {sess_dir}: {e}")

    # -------------------- apply optional filters --------------------------
    if filters_dict:
        for fkey, fdict in filters_dict.items():
            if fkey == "velocity_filter" and ctrl_df is not None:
                data_dict = {
                    "controls": np.column_stack(
                        (ctrl_df["linear"], ctrl_df["angular"], ctrl_df["timestamp"])
                    )
                }
                from scripts.preprocessing.run_engine import apply_velocity_filter
                from scripts.utils.time_utils import find_contiguous_true_intervals
                mask = apply_velocity_filter(data_dict, fdict)
                subsequences = find_contiguous_true_intervals(mask, window_len=fdict['params']['min_length'])
                if len(subsequences) == 0:
                    logging.warning(f"[local] Failed {fkey}: no valid subsequences.")
                    return pd.DataFrame()
            elif fkey == "odometry_filter" and odom_df is not None:
                data_dict = {
                    "odometry": odom_df.to_numpy(),
                    "timestamps": odom_df["timestamp"].to_numpy()
                }
                from scripts.preprocessing.run_engine import apply_odometry_filter
                from scripts.utils.time_utils import find_contiguous_true_intervals
                mask = apply_odometry_filter(data_dict, fdict)
                subsequences = find_contiguous_true_intervals(mask, window_len=fdict['params']['horizon_frames'])
                if len(subsequences) == 0:
                    logging.warning(f"[local] Failed {fkey}: no valid subsequences.")
                    return pd.DataFrame()

    # -------------------- images & metadata per (single) chunk ------------
    metadata_rows = []
    for key_seg, local_bags in bags_dict.items():
        spec = topics_dict[key_seg]
        ros_topics    = spec["ros_topic"] if isinstance(spec["ros_topic"], list) else [spec["ros_topic"]]
        save_prefixes = spec["save_prefix"] if isinstance(spec["save_prefix"], list) else [spec["save_prefix"]]
        # One output directory (chunk0) for this local session
        out_dir = set_dataset_dir(save_root_dir, cfg, local_bags)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Run RGB processor
        art = process_rgb(
            bags=local_bags,
            out_dir=str(out_dir),
            topics=ros_topics,
            save_prefixes=save_prefixes,
            seq=robot,
            write_archival=False
            fps=10,
            process_bag=True,
            save_video=cfg.get('save_video', True)
        )

        if not art:
            logging.warning(f"[local] No frames found for {ros_topics} in {sess_dir}")
            continue
        
        # Write odometry if available; else blank path
        odom_path = ""
        if odom_df is not None and not odom_df.empty:
            odom_path = Path(out_dir) / f"{topics_dict[odom_key]['save_prefix']}_data_{robot}.csv" if odom_key else (Path(out_dir) / "odometry_data_local.csv")
            odom_df.to_csv(odom_path, index=False)
            odom_path = str(odom_path)

        # Write controls if available; else blank path
        ctrl_path = ""
        if ctrl_df is not None and not ctrl_df.empty:
            ctrl_path = Path(out_dir) / f"{topics_dict[ctrl_key]['save_prefix']}_data_{robot}.csv" if ctrl_key else (Path(out_dir) / "controls_data_local.csv")
            ctrl_df.to_csv(ctrl_path, index=False)
            ctrl_path = str(ctrl_path)

        # Write static TF if available; else blank path
        tf_path = ""
        if static_transforms:
            tf_path = Path(out_dir) / f"{topics_dict[tf_static_key]['save_prefix']}_{robot}.yaml" if tf_static_key else (Path(out_dir) / "tf_static_local.yaml")
            with Path(tf_path).open("w") as f:
                transforms_to_yaml(static_transforms, f)
            tf_path = str(tf_path)

        # Assume first image topic is the main video
        main_topic = ros_topics[0]
        main_video = art.get(main_topic, {}).get("video", "")

        metadata_rows.append(
            {
                "video":    main_video,
                "odometry": odom_path,
                "controls": ctrl_path,
                "tf_static": tf_path,
                "robot":    robot,
            }
        )

    return pd.DataFrame(metadata_rows)
