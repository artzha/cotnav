import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from cotnav.utils.math_utils import (interpolate_se3, se3_matrix)
from cotnav.utils.loader_utils import (load_intrinsics, load_odom, load_timestamps, build_transforms)
from cotnav.utils.eval_utils import (gt_local_polyline_from_odom_corrected, sample_arc_xy, hausdorff_xy)


motion_cfg = {
    "max_curvature": 0.6,
    "max_free_path_length": 3.0,
    "num_options": 7,
    "pixels_per_meter": 120.0,
    "colors": [(255, 0, 0), (255, 255, 255), (0, 255, 0)],
    "thickness": 3,
    "endpoint_radius": 4,
    "alpha": 230,
}

annotation_cfg = {
    "thickness": 5,
    "endpoint_radius": 25,
    "overlay_alpha": 0.9,
    "label_font_size": 40,
    "label_font_color": (0, 0, 0),
}



def evaluate_model_outputs(arcs, model_responses):

    total_correct = 0
    total_responses = len(model_responses)

    for i, response in enumerate(model_responses):
        mission = response['mission']
        start_frame = response['start_frame']
        end_frame = response['end_frame']
        mission_dir = Path(f"../../cotnav/dataset/grandtour_raw/{mission}")
        video_path = mission_dir / "front_camera_lossy.mp4"
        ts_path   = mission_dir / "front_camera_timestamps_anymal.csv"
        odom_path = mission_dir / "odometry_data_anymal.csv"
        info_path = mission_dir / "front_camera_info_anymal.yaml"
        tf_path = mission_dir / "tf_static_anymal.yaml"

        assert all(p.exists() for p in [mission_dir, video_path, odom_path, info_path, ts_path, tf_path]), "One or more paths do not exist"

        calib = load_intrinsics(info_path, tf_path, world_frame="base")

        vr = VideoReader(str(video_path), ctx=cpu(0))
        batch = vr.get_batch(np.arange(start_frame, end_frame, 1))
        frames = batch.asnumpy()
        odom = load_odom(odom_path)

        calib = load_intrinsics(info_path, tf_path, world_frame="base")
        cam_ts = load_timestamps(ts_path)
        interp_odom = interpolate_se3(cam_ts, odom[:, 0], odom[:, 1:4], odom[:, 4:8])

        print("Loaded odometry with shape:", odom.shape)
        print("Loaded timestamps with shape:", cam_ts.shape)
        print("Interpolated odometry shape:", interp_odom.shape)

        print("Image shape (H, W):", calib.size_hw)
        print("Intrinsics:\n", calib.K)
        print("TF world to cam:\n", calib.T_world_cam)
        tm = build_transforms(tf_path)

        odom_window = interp_odom[start_frame:start_frame + 100]
        T_hesai_odom = se3_matrix(odom_window[:, 1:4], odom_window[:, 4:8])
        T_base_hesai = tm.get_transform("hesai_lidar", "base") # tgt, src
        T_base_odom = T_hesai_odom @ T_base_hesai
        T_base_local = np.linalg.inv(T_base_odom[0]) @ T_base_odom
        gt_xy = gt_local_polyline_from_odom_corrected(interp_odom, T_base_local, 0, lookahead_m=6.0)
        hd_vals = []
        for j, arc in enumerate(arcs):
            arc_xy = sample_arc_xy(
                arc,
                max_len_m=6.0,
                samples_per_meter=annotation_cfg.get("samples_per_meter", 10),
            )
            hd_vals.append(hausdorff_xy(gt_xy, arc_xy))
        if len(hd_vals):
            best_idx = int(np.argmin(hd_vals))
            correct = (best_idx == response['chosen_path'])
            total_correct += int(correct)
            print(f"Response {i}: Mission {mission}, Frames {start_frame}-{end_frame}, Chosen Path: {response['chosen_path']}, Best Path: {best_idx}, Correct: {correct}")

    print(f"Total Correct: {total_correct}/{total_responses}")