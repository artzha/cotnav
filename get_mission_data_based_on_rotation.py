#!/usr/bin/env python3
"""
Scan mission directories under a dataset root, compute high-angular-velocity segments
from odometry CSV files, and dump a CSV of (dataset, mission_dir, start_idx, future_frame_idx).

Usage:
    python get_mission_data_based_on_rotation.py --data_root dataset/grandtour_raw \
        --out rotation_segments.csv

The script tries to estimate camera fps from a camera timestamps file if present
(`front_camera_timestamps_anymal.csv`). If unavailable, it falls back to a default fps.
"""
import argparse
import csv
from pathlib import Path
import sys
import math
import os

import numpy as np
import pandas as pd


def quaternion_to_yaw(qw, qx, qy, qz):
    """Convert quaternion to yaw angle (heading)"""
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def calculate_angular_velocity(df):
    angular_velocities = []
    for i in range(1, len(df)):
        yaw_diff = df['yaw'].iloc[i] - df['yaw'].iloc[i - 1]
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi

        time_diff = df['timestamp'].iloc[i] - df['timestamp'].iloc[i - 1]
        if time_diff > 0:
            angular_vel = yaw_diff / time_diff
        else:
            angular_vel = 0
        angular_velocities.append(angular_vel)
    angular_velocities.insert(0, 0)
    return np.array(angular_velocities)


def find_rotation_segments(sudden_rotation_mask):
    rotation_segments = []
    in_rotation = False
    start_idx = None
    for i, is_rot in enumerate(sudden_rotation_mask):
        if is_rot and not in_rotation:
            in_rotation = True
            start_idx = i
        elif not is_rot and in_rotation:
            in_rotation = False
            rotation_segments.append((start_idx, i - 1))
        elif i == len(sudden_rotation_mask) - 1 and in_rotation:
            rotation_segments.append((start_idx, i))
    return rotation_segments


def timestamp_to_frame(timestamp, start_time, fps, total_frames):
    elapsed_time = timestamp - start_time
    frame_idx = int(elapsed_time * fps)
    return max(0, min(frame_idx, max(0, total_frames - 1)))


def process_mission(mission_dir: Path, angular_velocity_threshold=20.0, min_duration=1.0, future_seconds=4.0, fps_default=30):
    """Process a single mission directory. Returns list of tuples (start_idx, future_frame_idx)."""
    # Prefer the canonical filename used in analysis notebook
    odom_path = mission_dir / 'odometry_data_anymal.csv'
    if not odom_path.exists():
        # fallback to any odometry-like csv
        odom_candidates = list(mission_dir.glob('*odometry*.csv'))
        if not odom_candidates:
            print(f"  Skipping {mission_dir} (no odometry csv found)")
            return []
        odom_path = odom_candidates[0]
    try:
        df = pd.read_csv(odom_path)
    except Exception as e:
        print(f"  Failed reading {odom_path}: {e}")
        return []

    required_cols = {'timestamp', 'qw', 'qx', 'qy', 'qz'}
    if not required_cols.issubset(set(df.columns)):
        print(f"  Skipping {mission_dir} (missing required odom columns). Found: {df.columns.tolist()}")
        return []

    df = df.reset_index(drop=True)
    df['yaw'] = quaternion_to_yaw(df['qw'].values, df['qx'].values, df['qy'].values, df['qz'].values)
    angular_vel = calculate_angular_velocity(df)
    df['angular_velocity'] = angular_vel
    df['angular_velocity_deg'] = np.degrees(df['angular_velocity'])
    df['sudden_rotation'] = np.abs(df['angular_velocity_deg']) > angular_velocity_threshold

    rotation_segments = find_rotation_segments(df['sudden_rotation'].values)

    # Load camera timestamps if available to estimate total_frames and fps
    cam_ts_path = mission_dir / 'front_camera_timestamps_anymal.csv'
    total_frames = None
    estimated_fps = fps_default
    video_duration = df['timestamp'].max() - df['timestamp'].min()

    if cam_ts_path.exists():
        try:
            cam_ts = pd.read_csv(cam_ts_path, header=None).squeeze().values
            total_frames = len(cam_ts)
            if video_duration > 0:
                estimated_fps = total_frames / video_duration
        except Exception:
            total_frames = None
    # If no timestamps file, try to estimate using a video file exists (but avoid heavy deps). Fallback to default fps.
    if total_frames is None:
        total_frames = int(max(1, round(video_duration * fps_default)))

    results = []
    for start_idx, end_idx in rotation_segments:
        start_timestamp = float(df.iloc[start_idx]['timestamp'])
        end_timestamp = float(df.iloc[end_idx]['timestamp'])
        duration = end_timestamp - start_timestamp
        if duration < min_duration:
            continue

        # Map odometry timestamp to camera frame index
        start_frame_idx = timestamp_to_frame(start_timestamp, df['timestamp'].min(), estimated_fps, total_frames)
        future_frame_idx = min(total_frames - 1, start_frame_idx + int(round(future_seconds * estimated_fps)))
        results.append((start_idx, future_frame_idx))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/scratch/cluster/carlq/research/cotnav/cotnav/dataset/grandtour_raw', help='Root folder containing mission_* directories')
    parser.add_argument('--out', type=str, default='rotation_segments.txt', help='Output text file to write results')
    parser.add_argument('--threshold_deg', type=float, default=20.0, help='Angular velocity threshold in deg/s')
    parser.add_argument('--min_duration', type=float, default=1.0, help='Minimum rotation segment duration (s) to include')
    parser.add_argument('--future_seconds', type=float, default=8.0, help='Seconds into future for future_frame_idx')
    parser.add_argument('--fps_default', type=float, default=10.0, help='Default fps to use if camera timestamps missing (assumed 10)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    mission_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir() and p.name.startswith('mission')]
    print(f"Found {len(mission_dirs)} mission directories under {data_root}")

    out_path = Path(args.out)
    with out_path.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        # write simple header
        writer.writerow(['dataset', 'mission_dir', 'start_idx', 'future_frame_idx'])

        for mission_dir in mission_dirs:
            print(f"Processing {mission_dir}...")
            try:
                segments = process_mission(mission_dir, angular_velocity_threshold=args.threshold_deg,
                                           min_duration=args.min_duration, future_seconds=args.future_seconds,
                                           fps_default=args.fps_default)
            except Exception as e:
                print(f"  Error processing {mission_dir}: {e}")
                continue

            for start_idx, future_frame_idx in segments:
                # write dataset as the folder name (e.g., 'grandtour_raw')
                writer.writerow([data_root.name, os.path.split(mission_dir)[-1], int(start_idx), int(future_frame_idx)])
            print(f"  -> Found {len(segments)} valid segments")

    print(f"Wrote results to {out_path}")


if __name__ == '__main__':
    main()
