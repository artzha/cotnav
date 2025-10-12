import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from cotnav.utils.math_utils import (interpolate_se3, se3_matrix, get_T_base_local, heading_from_start)
from cotnav.utils.loader_utils import (load_intrinsics, load_odom, load_timestamps, build_transforms)
from cotnav.utils.eval_utils import (gt_local_polyline_from_odom_corrected, sample_arc_xy, hausdorff_xy, convert_response_to_unified_format)
from cotnav.models.vlms import infer_registry
from PIL import Image
from cotnav.models.vlms.openaimodel import (
    ChatQuery, get_openai_cost
)
import json


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

        ## TODO: implement these metrics:
        # Pass @ 3, hdist(model, odom), hdist(model, odom) - hdist(gt, odom)

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


def build_messages_batch(base_dir, data_dict, vlm_cfg, motion_cfg, conditional=False):

    messages_batch = []
    
    pivot = infer_registry.get("pivot", vlm=vlm_cfg, motion_parameters=motion_cfg)
    motion_arcs = pivot.motion_templates()
    missions, start_frames, end_frames = data_dict['mission'], data_dict['start_frame'], data_dict['future_frame_idx']
    for mission, start_frame, end_frame in zip(missions, start_frames, end_frames):
        print(f"Processing mission {mission}, frames {start_frame} to {end_frame}")
        FRAME_STEP = 1
        LANGUAGE_CMD = "Go to the railway conductor building."
        mission_dir = Path(os.path.join(base_dir, mission))
        video_path = mission_dir / "front_camera_lossy.mp4"
        ts_path   = mission_dir / "front_camera_timestamps_anymal.csv"
        odom_path = mission_dir / "odometry_data_anymal.csv"
        info_path = mission_dir / "front_camera_info_anymal.yaml"
        tf_path = mission_dir / "tf_static_anymal.yaml"
        assert all(p.exists() for p in [mission_dir, video_path, odom_path, info_path, ts_path, tf_path]), "One or more paths do not exist"

        calib = load_intrinsics(info_path, tf_path, world_frame="base")

        vr = VideoReader(str(video_path), ctx=cpu(0))
        batch = vr.get_batch(np.arange(start_frame, end_frame, FRAME_STEP))
        frames = batch.asnumpy()
        num_plots = 5
        frame_indices = np.linspace(0, len(frames) - 1, num_plots, dtype=int)
        T_base_local = get_T_base_local(
            load_odom(odom_path),
            load_intrinsics(info_path, tf_path, world_frame="base"),
            load_timestamps(ts_path),
            build_transforms(tf_path),
            start_frame,
            start_frame + 700
        )
        if not conditional:
            # Response gathering loop
            with open("../../cotnav/models/prompts/parallel_prompt.txt", "r") as f:
                system_prompt = f.read()

            intermediate_prompts = [
                f"""
                For each path option from 0 to 6, describe how promising of an option it is for {LANGUAGE_CMD} and provide each reason for each choice in a list of json objects. After describing each path option, then append one more choice in the list of json objects for best path option and the reason. ONLY output the result in the format as described in the system prompt.
                """
            ]
        else:
            with open("../../cotnav/models/prompts/conditional_prompt.txt", "r") as f:
                system_prompt = f.read()

            intermediate_prompts = [
                f"""
                For each path option from 0 to 6, describe how promising of an option it is for {LANGUAGE_CMD} and provide each reason for each choice in a list of json objects. ONLY output the result in the format as described in the system prompt.
                """,
                f"""
                Given the previous reasons explaining each path option, select the best overall choice that follows the {LANGUAGE_CMD} and its reason in a json format. ONLY output the result in the format as described in the system prompt.
                """
            ]

        messages = []
        images = []
        for i, frame_idx in enumerate(frame_indices):
            frame = Image.fromarray(frames[frame_idx])
            if i == len(frame_indices) - 1:
                frame, centers = pivot.annotate_constant_curvature(
                    frame, arcs=motion_arcs, calib=calib, **annotation_cfg
                )
                goal_hdg_deg = heading_from_start(T_base_local, frame_idx, -1, degrees=True)
                frame = pivot.annotate_goal_heading(frame, goal_hdg_deg)
                annotated_frame = frame.copy()
            images.append(ChatQuery("image", "user", frame))

        messages.extend(images)
        messages_batch.append(messages)

    return messages_batch, system_prompt, intermediate_prompts, pivot

def batch_inference(messages_batch, system_prompt, intermediate_prompts, pivot):

    batch_intermediate_responses = [[] for _ in range(len(messages_batch))]
    for i, prompt in enumerate(intermediate_prompts):
        # Add user prompt and query VLM
        for messages in messages_batch:
            messages.append(ChatQuery("text", "user", prompt))
        # response = pivot.vqa(system_prompt, messages)
        
        responses = pivot.batch_vqa(system_prompt, messages_batch)
        for j, response in enumerate(responses):
            for output in response['response']['body']['output']:
                if 'role' in output:
                    response_output_str = output['content'][0]['text']
                    # Parse output and add to messages
                    stage_response = ChatQuery("text", "assistant", response_output_str)
                    messages_batch[j].append(stage_response)
            batch_intermediate_responses[j].append(response_output_str)

#     cost, cost_breakdown = get_openai_cost(
    return batch_intermediate_responses

def generate_batch_mission_results(base_dir, data_dict, vlm_cfg, motion_cfg, conditional=False):

    messages_batch, system_prompt, intermediate_prompts, pivot = build_messages_batch(base_dir, data_dict, vlm_cfg, motion_cfg, conditional)

    batch_intermediate_responses = batch_inference(messages_batch, system_prompt, intermediate_prompts, pivot)
    batch_unified_responses = []

    missions, start_frames, end_frames = data_dict['mission'], data_dict['start_frame'], data_dict['future_frame_idx']

    for i, intermediate_responses in enumerate(batch_intermediate_responses):
        # print("Intermediate Responses:", len(intermediate_responses), intermediate_responses[0])
        unified_response = convert_response_to_unified_format(
            response_json=intermediate_responses,   # list/str/pydantic per your pipeline
            dataset="grandtour",
            mission=missions[i],
            start_frame=start_frames[i],
            end_frame=end_frames[i],
            conditional_enabled=conditional,
        )
        print(json.dumps(unified_response, indent=2))
        batch_unified_responses.append(unified_response)
    return batch_unified_responses
