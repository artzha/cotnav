"""
Track Points along Robot Path using **Online** CoTracker (sparse‑query mode)
==========================================================================

-----------------------
* **Shape mismatch fixed**  we now allocate per‑segment arrays based on the
  actual number of tracks returned (some points can disappear when masked by
  the support-grid logic).
* **Occlusion-aware pruning** - after stitching segments we invalidate any
  coordinates that stay invisible for more than `--occlusion_tol` consecutive
  frames.
* **Absolute frame index for *t*** - when creating the query tensor we store
  each point's **absolute original frame index** in the *t* column.  Internally
  we still reset it to *0* for CoTracker (it expects clip‑local indices), but
  we keep a copy so the saved pickle now contains a `query_t_abs` entry.

  Note: We want to operate in BGR space as much as possible, since we use cv2 SIFT
  for feature extraction, which expects BGR images.  We convert to RGB only 
"""

from __future__ import annotations
from profilehooks import timecall

import os
import os
for var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'
import argparse
import pickle
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging
import kornia as K

import cv2
cv2.setNumThreads(0)        # or 1 – completely disables OpenCV’s TBB/OpenMP pool
cv2.ocl.setUseOpenCL(False) # avoid extra OpenCL threads on laptops / Macs
import h5py
import hickle
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import decord
import pickle
from pandas.errors import EmptyDataError

from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
from scripts.utils.log_utils import logging
from scripts.utils.robust_hull import swept_mask
from scripts.utils.image_utils import (save_video_np, sift_in_mask, sample_edge_points, smooth_contour, enhance_ground_batch, apply_intrinsics)
from scripts.utils.loader_utils import (
    load_odom, 
    load_timestamps
)
from scripts.mapping.compute_odom import (
    interpolate_se3
)
from spinflow.dataset.frodo_constants import FRONT_CAMERA_INTRINSICS
from spinflow.dataset.frodo_helpers import (get_frodo_raw_id, set_frodo_dir, set_frodo_id, get_frodo_id)
# from scripts.utils.polyline_utils import (build_mask_from_crumbs)
from scripts.utils.polyline_utils_smoothed import (build_mask_from_crumbs)

# random.seed(42)

"""BEGIN FRODO CONFIGURATION"""
# ~40 GB with #frames=250, MAX_QUERY_POINTS=1152, TEMPORAL_DS_RESOLUTION=1, SPATIAL_DS_RESOLUTION=1
# SAVE_VIDEO_MASKS = False # Expensive ~10 seconds per call
# ACCEPTABLE_RESOLUTIONS = [ (576, 1024) ]
# ENHANCE_VIDEO = False # Enhance video for easier tracking (white balance, clache clip, sharpen)
# APPLY_INTRINSICS = False  # Enable undistorted video tracking
# SAMPLE_AT_EDGES = True  # Sample queries at the edges of the robot path
# ROBOT_WIDTH_PCT = 0.2      # Higher is wider
# ROBOT_HEIGHT_PCT = 0.15      # Higher is taller
# TEMPORAL_DS_RESOLUTION = 1  # 1 means no downsampling, 2 means downsample by 2x
# SPATIAL_DS_RESOLUTION = 1   # 1 means no downsampling, 2 means downsample by 2x
# NUM_FRAMES = 250            # 250frames / 20 fps -> 12.5 seconds of video
# EDGE_OFFSET_PCT = 0.1
# FRAME_STEP = 100 
# WINDOW_SIZE = 20            # window size to sample SIFT queries
# MAX_QUERY_POINTS = 384
# MAX_REINITIALIZATION_ITERATIONS = 20  # Maximum number of re-initializations per segment
"""END FRODO CONFIGURATION"""

OVERRIDE_CACHE=None
ODOMETRY_STEP=None
SAVE_VIDEO_MASKS=None
ACCEPTABLE_RESOLUTIONS=None
ENHANCE_VIDEO=None
APPLY_INTRINSICS=None
SAMPLE_AT_EDGES=None
ROBOT_WIDTH_PCT=None
ROBOT_HEIGHT_PCT=None
TEMPORAL_DS_RESOLUTION=None
SPATIAL_DS_RESOLUTION=None
NUM_FRAMES=None
EDGE_OFFSET_PCT=None
FRAME_STEP=None
WINDOW_SIZE=None
MAX_QUERY_POINTS=None
MAX_REINITIALIZATION_ITERATIONS=None

# """BEGIN USER CONFIGURATION"""
# # ~40 GB with #frames=250, MAX_QUERY_POINTS=1152, TEMPORAL_DS_RESOLUTION=1, SPATIAL_DS_RESOLUTION=1
# SAVE_VIDEO_MASKS = False # Expensive ~10 seconds per call
# ACCEPTABLE_RESOLUTIONS = [ (576, 1024) ]
# ENHANCE_VIDEO = False # Enhance video for easier tracking (white balance, clache clip, sharpen)
# APPLY_INTRINSICS = False  # Enable undistorted video tracking
# SAMPLE_AT_EDGES = True  # Sample queries at the edges of the robot path
# ROBOT_WIDTH_PCT = 0.15      # Higher is wider
# ROBOT_HEIGHT_PCT = 0.15      # Higher is taller
# TEMPORAL_DS_RESOLUTION = 1  # 1 means no downsampling, 2 means downsample by 2x
# SPATIAL_DS_RESOLUTION = 1   # 1 means no downsampling, 2 means downsample by 2x
# NUM_FRAMES = 250            # 250frames / 20 fps -> 12.5 seconds of video
# EDGE_OFFSET_PCT = 0.2
# FRAME_STEP = 100           # Every 5 seconds at 20 fps
# WINDOW_SIZE = 20            # window size to sample SIFT queries
# MAX_QUERY_POINTS = 384
# MAX_REINITIALIZATION_ITERATIONS = 20  # Maximum number of re-initializations per segment
# """END USER CONFIGURATION"""
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_global_params(cfg):
    global OVERRIDE_CACHE, SAVE_VIDEO_MASKS, ACCEPTABLE_RESOLUTIONS, ENHANCE_VIDEO, APPLY_INTRINSICS, \
        SAMPLE_AT_EDGES, ROBOT_WIDTH_PCT, ROBOT_HEIGHT_PCT, TEMPORAL_DS_RESOLUTION, \
        SPATIAL_DS_RESOLUTION, NUM_FRAMES, EDGE_OFFSET_PCT, FRAME_STEP, WINDOW_SIZE, \
        MAX_QUERY_POINTS, MAX_REINITIALIZATION_ITERATIONS
    track_params = cfg["track_params"]
    logging.info("Global tracking parameters:")
    for key, value in track_params.items():
        globals()[key] = value
        print(f"  {key}: {value}\n")

# data/frodo8k/output_rides_12/ride_46811_e7808b_20240518091924/front_camera.mp4
def choose_h5_file(data_dir: Path, chosen=None) -> Path:
    if chosen is None:
        files = sorted(data_dir.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No .h5 files in {data_dir}")
        return random.choice(files)
    p = data_dir / chosen
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def choose_frodo_ride(data_dir: Path) -> Path:
    if not Path(data_dir).is_file():
        
        if data_dir.is_dir() and not any(data_dir.glob("output_rides_*")):
            ride_dirs = [data_dir]
        else:
            ride_dirs = sorted(data_dir.glob("output_rides_*"))

        # Choose a random ride for the files that match "output_rides_{} pattern"
        # ride_dirs = sorted(data_dir.glob("output_rides_*"))
        while True:
            chosen = Path(random.choice(ride_dirs).name)
            # Choose a random drive subdirectory
            drive_dirs = list((data_dir / chosen).glob("ride_*"))
            drive_dir = random.choice(drive_dirs)
            video_path = drive_dir / "front_camera.mp4"
            if video_path.exists():
                break
    else:
        video_path = data_dir
        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_path} does not exist")
    return video_path

def choose_split_ride(split_path: Path) -> Path:
    # Open split file and choose a random ride
    rides = pd.read_csv(split_path)

    # Choose random ride
    chosen_ride = rides.sample(1).iloc[0]
    ride_name, start, end = chosen_ride['ride_name'], chosen_ride['start_frame'], chosen_ride['end_frame']
    video_dir = set_frodo_dir("./data/frodobots8k", *ride_name.split(" "))
    video_path = video_dir / "front_camera.mp4"
    assert video_path.exists(), f"Video file {video_path} does not exist"

    return video_path, start, end, ride_name

def load_video(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        vid = f["images"]["front_camera"][()]
    if vid.dtype != np.uint8:
        vid = (vid.clip(0, 1) * 255).astype(np.uint8)
    if vid.shape[-1] != 3:
        raise ValueError("Expect RGB video")
    start_frame = 0
    end_frame = vid.shape[0]
    # cv2.imwrite("test.jpg", vid[0].astype(np.uint8))  # Save first frame for testing    
    return vid, start_frame, end_frame

def load_video_from_file(video_file: Path, start_frame=None, end_frame=None) -> np.ndarray:
    assert video_file.exists(), f"Video file {video_file} does not exist"
    decord.bridge.set_bridge("torch")  # or "numpy"  (torch gives zero-copy tensors)

    vr = decord.VideoReader(str(video_file), num_threads=4)   # lazy – doesn’t decode yet
    # get the number of frames
    num_frames = len(vr)
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = min(start_frame + NUM_FRAMES, num_frames)
    else:
        end_frame = min(end_frame + NUM_FRAMES, num_frames)

    requested_frames = int(end_frame - start_frame)
    assert requested_frames > 0, \
        f"Requested frames {requested_frames} must be greater than 0, "
   
    if start_frame < 0 or end_frame >= num_frames:
        print(
            f"Video {video_file} has only {num_frames} frames, "
            f"but requested {NUM_FRAMES} frames. Skipping."
        )
        return (None, start_frame, end_frame)

    video_th = vr.get_batch(range(len(vr)))
    video_np = video_th.numpy()
    # video_np = video_np.astype(np.uint8) #[:, :, :, [2, 1, 0]]  # Convert from RGB to BGR

    # mid = video_np.shape[0] // 2
    # cv2.imwrite("test.jpg", video_np[mid].astype(np.uint8))  # Save first frame for testing
    return (video_np, start_frame, end_frame)

@timecall(immediate=False)
def preprocess_video(video: np.ndarray) -> np.ndarray:
    # Downsample the video if needed
    if TEMPORAL_DS_RESOLUTION > 1:
        video = video[::TEMPORAL_DS_RESOLUTION]
        print(f"Downsampled video to {video.shape[0]} frames")

    if APPLY_INTRINSICS:
        video = apply_intrinsics(video, FRONT_CAMERA_INTRINSICS)

    if video.shape[0] == 0:
        return None

    if SPATIAL_DS_RESOLUTION != 1:
        # Downsample spatially by resizing each frame
        video = np.array([
            cv2.resize(frame, 
                       ( int(frame.shape[1] / SPATIAL_DS_RESOLUTION), 
                        int(frame.shape[0] / SPATIAL_DS_RESOLUTION)), 
                       interpolation=cv2.INTER_LINEAR)
            for frame in video
        ])
        print(f"Resampled video spatially to {video.shape[1:]}")

    # Reverse video frames for tracking
    video = np.flip(video, axis=0).copy()  # Reverse the order of frames
    # print("Reversed video frames")
    # # video = video[..., ::-1].copy()  # Convert RGB to BGR
    # # Save reversed video 
    # video_path = "reversed_video.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # height, width, _ = video.shape[1:]
    # out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
    # for frame in video:
    #     out.write(
    #         cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    #     )
    # out.release()
    # Convert to torch tensor
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()

    if ENHANCE_VIDEO:
        # Preprocess the video for easier tracking with kornia
        assert video.max() >= 1.0 and video.max() <= 255.0, \
            "Video frames must be in the range [0, 255] for uint8 or [0, 1] for float32"
        video = enhance_ground_batch(video / 255.0)
        video = video * 255.0

    video = video.unsqueeze(0)  # Add batch dimension
    return video

def plot_queries(queries: torch.Tensor, H, W):
    # queries [1, K, 3] where K is the number of queries
    frame_numbers = queries[0,:,0].int().tolist()

    # Create query dictionary from frame number to query points
    queries_dict = { frame: [] for frame in frame_numbers }
    for i, query in enumerate(queries[0]):
        frame_number = frame_numbers[i]
        queries_dict[frame_number].append(query)
    num_unique_frames = len(queries_dict)
    num_subplots = (num_unique_frames // 2) + num_unique_frames % 2
    fig, axs = plt.subplots(2, num_subplots)
    axs = axs.flatten()

    for i, (frame_number, query_points) in enumerate(queries_dict.items()):
        ax = axs[i]
        query_points = torch.stack(query_points).numpy()  # Convert to numpy array
        ax.scatter(query_points[:, 1], query_points[:, 2], s=10, c='blue', label='Query Points')
        # ax.plot(query[1].item(), query[2].item(), 'ro')

        ax.set_title("Frame {}".format(frame_number))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("queries_plot.png")

def get_colormap(time_ids: np.ndarray) -> np.ndarray:
    """Returns a colormap from unique ids to colors"""
    # Get indices and ids
    unique_ids, color_idxs = np.unique(time_ids, return_inverse=True)   # [U,] [Q,]
    num_colors = len(unique_ids)
    colormap = plt.get_cmap('hsv', num_colors)
    colors = colormap(np.arange(num_colors))[:, :3]  # Get RGB colors, ignore alpha
    colors = (colors * 255).astype(np.uint8)  # Convert to uint8

    return colors[color_idxs]   # [Q, 3]

def sample_grid_queries(H, W, grid_width, grid_height, num_points=4, t=0):
    """
    Creates query points in a grid pattern at the bottom center of the frame for a specifix time
    Args:
        H (int): Height of the frame
        W (int): Width of the frame
        grid_size (int): Size of the grid
        time (int): Time index for the queries
    Returns:
        queries (torch.Tensor): (1, K, 3) where K is the number of queries
            (time, x, y) for each query point
    """
    queries_x = torch.linspace(
        W // 2 - (grid_width // 2),
        W // 2 + (grid_width // 2),
        num_points,
    ).long().clip(0, W - 1)
    queries_y = torch.linspace(
        H - grid_height,
        H,
        num_points,
    ).long().clip(0, H - 1)
    queries = torch.meshgrid(
        queries_x, queries_y, indexing='ij'
    )
    queries = torch.stack(queries, dim=-1).reshape(-1, 2)
    time = torch.ones(queries.shape[0]).unsqueeze(1) * t  # Shape (K, 1)
    queries = torch.cat(
        [time, queries], 
        dim=1
    ).unsqueeze(0).float()  # Shape (1, K, 3)

    return queries

def get_trapezoid_mask(H, W, grid_width, grid_height, sample_at_edges):
    """Compute a trapezoidal mask"""
    mask = np.zeros((H, W), dtype=np.uint8)
    # Define the trapezoid vertices
    left_top_x = W // 2 - grid_width // 2
    right_top_x = W // 2 + grid_width // 2
    top_left = (left_top_x, H - grid_height)
    top_right = (right_top_x, H - grid_height)

    offset = (right_top_x - left_top_x) * 0.2  # 20% offset for the bottom
    bottom_left = (left_top_x - offset, H)
    bottom_right = (right_top_x + offset, H)

    # Fill the trapezoid area
    pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    # cv2.imwrite("trapezoid_mask.jpg", mask)  # Save for debugging
    # import pdb; pdb.set_trace()

    if sample_at_edges:
        inner_offset = (right_top_x - left_top_x) * 0.15  # 30% offset for the inner edges
        pts = np.array([
            (top_left[0] + inner_offset, top_left[1]),
            (top_right[0] - inner_offset, top_right[1]),
            (bottom_right[0] - inner_offset, bottom_right[1]),
            (bottom_left[0] + inner_offset, bottom_left[1]),
        ], np.int32)
        negative_mask = np.ones((H, W), dtype=np.uint8) * 255  # Reset mask to full
        cv2.fillPoly(negative_mask, [pts], 0)  # Fill the inner trapezoid area
        mask = cv2.bitwise_and(mask, negative_mask)  # Apply the negative mask
    cv2.imwrite("trapezoid_mask.jpg", mask)  # Save for debugging
    return mask


def get_rectangular_mask(H, W, grid_width, grid_height, sample_at_edges):
    """
    Compute a rectangular mask of size (H, W), with a rectangle centered horizontally,
    grid_width wide and grid_height tall, anchored at the bottom of the image. If
    sample_at_edges is True, the center portion of that rectangle is removed, leaving
    two vertical strips at the left and right edges of the rectangle.

    Args:
        H (int): image height (rows)
        W (int): image width (columns)
        grid_width (int): width of the rectangle (in pixels)
        grid_height (int): height of the rectangle (in pixels)
        sample_at_edges (bool): if True, cut out an inner vertical band, leaving two
                                side strips; otherwise leave the full rectangle.

    Returns:
        mask (np.ndarray of shape (H, W), dtype=np.uint8): 255 inside the rectangle
            (or side strips), 0 elsewhere.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    # Compute the top‐left and top‐right x‐coordinates of the rectangle
    left_x = (W // 2) - (grid_width // 2)
    right_x = (W // 2) + (grid_width // 2)
    top_y = H - grid_height
    bottom_y = H

    # Define the four corners of the rectangle (bottom‐anchored)
    rect_pts = np.array([
        (left_x,  top_y),
        (right_x, top_y),
        (right_x, bottom_y),
        (left_x,  bottom_y)
    ], dtype=np.int32)

    # Fill the main rectangle
    cv2.fillPoly(mask, [rect_pts], 255)

    if sample_at_edges:
        # Compute an inner_offset (e.g. 15% of grid_width) to cut out the center
        inner_offset = int(grid_width * EDGE_OFFSET_PCT)

        # Ensure offsets don't overlap: clamp if needed
        inner_left = left_x + inner_offset
        inner_right = right_x - inner_offset
        if inner_right <= inner_left:
            # If the inner region is too small or inverted, skip edge sampling
            return mask

        # Define the inner rectangle to subtract
        inner_pts = np.array([
            (inner_left,  top_y),
            (inner_right, top_y),
            (inner_right, bottom_y),
            (inner_left,  bottom_y)
        ], dtype=np.int32)

        # Create a negative mask: start with all white, then zero out the inner rectangle
        negative = np.full((H, W), 255, dtype=np.uint8)
        cv2.fillPoly(negative, [inner_pts], 0)

        # Combine: keep only the side strips of the original rectangle
        mask = cv2.bitwise_and(mask, negative)
    cv2.imwrite("rectangular_mask.jpg", mask)  # Save for debugging
    return mask

@timecall(immediate=False)
def sample_sift_mask(video, max_num_points, mask=None, window_size=20, sift_kwargs=None):
    """Inputs
        video: (B, T, 3, H, W) torch tensor of video frames
        mask: (H, W) np array of mask to apply to the video frames
    Return:
        queries: (1, N*K, 3) torch tensor of queries where N is #windows, K is #keypoints
                 each query is (t, x, y)
        sides:   (1, N*K) torch tensor of side‐ids
    """
    assert video.ndim == 5 and video.shape[2] == 3, \
        "Video must be in (B, T, 3, H, W) format"

    # 1) Convert to H×W×3 numpy
    video = video[0].permute(0, 2, 3, 1).cpu().numpy()
    if video.max() <= 1.0:
        video = (video * 255).astype(np.uint8)
    else:
        video = video.astype(np.uint8)
    T, H, W, _ = video.shape

    # 2) Prepare mask
    if mask is None:
        mask = np.ones((H, W), dtype=bool)
    else:
        assert mask.shape == (H, W), f"Mask must be {H}×{W}, got {mask.shape}"
    video_mask = np.repeat(mask[None, :, :], T, axis=0)

    # 3) Pick one frame per window (uniformly), include first & last
    num_windows = max(1, T // window_size)
    # linspace gives us evenly spaced indices 0..T-1
    idxs = np.linspace(0, T - 1, num=num_windows, dtype=int)
    # ensure 0 and T-1 are present
    idxs = np.unique(np.concatenate(([0], idxs, [T - 2])))
    
    sub_video      = video[idxs]
    sub_video_mask = video_mask[idxs]

    # 4) Run SIFT only on those frames
    # Clip range of max_num_points to avoid excessive sampling
    samples_per_frame = int(max_num_points / len(idxs))
    if samples_per_frame < 8:
        samples_per_frame = 8
    elif samples_per_frame > 20:
        samples_per_frame = 20

    sift_dict = sift_in_mask(
        sub_video,
        n_samples=samples_per_frame,
        seg_mask=sub_video_mask,
        sift_kwargs=sift_kwargs
    )

    # 5) Build queries exactly as before, but using our subsampled frames `idxs`
    keyps       = sift_dict['keyps']   # shape (N_w, K, 2)
    sides       = sift_dict['sides']   # shape (N_w, K)
    Nw, K, _    = keyps.shape
    # repeat each timestamp K times
    ts          = np.repeat(idxs[:, None], K, axis=1)
    queries_np  = np.concatenate([ts[..., None], keyps], axis=2).reshape(-1, 3)
    sides_np    = sides.reshape(-1)

    # 6) filter out any NaNs
    valid_mask  = ~np.isnan(queries_np).any(axis=1)
    queries_np  = queries_np[valid_mask]
    sides_np    = sides_np[valid_mask]

    # 7) to tensors and return
    crumb_queries     = torch.from_numpy(queries_np).float().unsqueeze(0)   # (1, M, 3)
    crumb_query_sides = torch.from_numpy(sides_np).float().unsqueeze(0)    # (1, M)

    print(f"Total queries sampled: {crumb_queries.shape[1]}")
    return crumb_queries, crumb_query_sides

def sample_sift_mask_adaptive(
    video: torch.Tensor,
    max_num_points: int,
    mask: Optional[np.ndarray] = None,
    window_size: int = 20,
    tiers: Optional[List[Dict]] = None,
    min_points: Optional[int] = None,  # fallback: aim for 80% of max_num_points
    dedup_radius_px: int = 3
):
    """
    Try stricter SIFT first; if not enough points, progressively relax.
    Returns (queries, sides) exactly like sample_sift_mask.
    """
    if tiers is None:
        tiers = [
            # Tier 0 (baseline / current-ish)
            {"contrastThreshold": 0.018, "edgeThreshold": 18, "nOctaveLayers": 4, "sigma": 1.6},
            # Tier 1 (slightly more lenient)
            {"contrastThreshold": 0.012, "edgeThreshold": 22, "nOctaveLayers": 4, "sigma": 1.4},
            # Tier 2 (lenient but still decent quality)
            {"contrastThreshold": 0.008, "edgeThreshold": 24, "nOctaveLayers": 4, "sigma": 1.2},
        ]
    if min_points is None:
        min_points = int(0.8 * max_num_points)

    all_q = []
    all_s = []
    for p in tiers:
        q, s = sample_sift_mask(
            video, max_num_points, mask, window_size=window_size, sift_kwargs=p
        )
        if q.numel() > 0:
            all_q.append(q)
            all_s.append(s)

        total = sum(x.shape[1] for x in all_q) if all_q else 0
        if total >= min_points:
            break

    if not all_q:
        # nothing found in any tier
        return torch.zeros((1,0,3), dtype=torch.float32), torch.zeros((1,0), dtype=torch.float32)

    # merge and dedup (by (t, x, y) grid within radius)
    q_cat = torch.cat(all_q, dim=1)   # (1,K,3)
    s_cat = torch.cat(all_s, dim=1)   # (1,K)

    # round positions to a coarse grid to remove near-duplicates
    q_np = q_cat[0].cpu().numpy()
    cell = max(1, dedup_radius_px)
    t = q_np[:, 0].astype(int)
    x = (q_np[:, 1] / cell).round().astype(int)
    y = (q_np[:, 2] / cell).round().astype(int)
    _, uniq_idx = np.unique(np.stack([t, x, y], axis=1), axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)

    q_cat = q_cat[:, uniq_idx]
    s_cat = s_cat[:, uniq_idx]

    # cap to max_num_points
    if q_cat.shape[1] > max_num_points:
        idx = np.random.choice(q_cat.shape[1], size=max_num_points, replace=False)
        idx.sort()
        q_cat = q_cat[:, idx]
        s_cat = s_cat[:, idx]

    return q_cat, s_cat

def last_visible_timestep(visible: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    visible : Bool tensor of shape  (B, T, Q)
              visible[b, t, q] == True  ⟺  query q is visible at timestep t
                                          in batch-item b.

    Returns
    -------
    last_ts : Int64 tensor of shape (B, Q)
              last_ts[b, q]  : last timestep index (0-based) where that
                               query is visible;  -1  if it is never visible.
    """
    if visible.dtype != torch.bool:
        raise ValueError("`visible` must be a boolean tensor.")

    B, T, Q = visible.shape

    # --- 1.  build a 0..T-1 index grid, broadcast to (B,T,Q) --------------
    t_idx = torch.arange(T, device=visible.device, dtype=torch.long)
    t_idx = t_idx.view(1, T, 1).expand(B, T, Q)          # (B,T,Q)

    # --- 2.  mask timesteps where the query is *not* visible --------------
    t_idx_masked = t_idx.masked_fill(~visible, -1)        # invisible → -1

    # --- 3.  maximum along the time dimension → last visible --------------
    last_ts, _ = t_idx_masked.max(dim=1)                  # shape (B,Q)

    return last_ts

@timecall(immediate=False)
def annotate_query_points(
        image: torch.Tensor or np.ndarray, 
        queries: torch.Tensor or np.ndarray, 
        visibility: torch.Tensor or np.ndarray, 
        save_image=False, 
        image_name="annotated_image.png"
    ) -> torch.Tensor:
    """
    Annotate query points on the image. Color code by the timestep
    Args:
        image (torch.Tensor): Image tensor of shape (H, W, 3) or (3, H, W)
        queries (torch.Tensor): (1, K, 3) tensor or (K, 3) np of query points
            Each query is (t, x, y) with t being the frame index and (x, y) being the keypoint coordinates
        visibility (torch.Tensor): (1, K) tensor or (K) np array of visibility flags
    Returns:
        annotated_image (torch.Tensor): Annotated image tensor of shape (H, W, 3) or (3, H, W)
    """
    if type(queries) is torch.Tensor:
        assert queries.ndim == 3 and queries.size(0) == 1 and queries.size(-1) == 3, \
            "Queries must be a tensor of shape (1, K, 3) where K is the number of queries and last dim is (t, x, y)"
        queries = queries.squeeze(0).cpu().numpy()  # Shape (K, 3)
    if type(visibility) is torch.Tensor:
        assert visibility.ndim == 2 and visibility.size(0) == 1, \
            "Visibility must be a tensor of shape (1, K) where K is the number of queries"
        visibility = visibility.squeeze(0).cpu().numpy()
    if type(image) is torch.Tensor:
        assert image.ndim == 3 and image.shape[0] == 3, \
            "Image must be a tensor of shape (3, H, W) or (H, W, 3)"
        image = image.permute(1, 2, 0).clone().cpu().numpy()
    
    assert image.ndim == 3 and image.shape[2] == 3, \
        "Image must be a 3D array of shape (H, W, 3) or (3, H, W)"
    assert queries.ndim == 2 and queries.shape[1] == 3 and visibility.ndim == 1, \
        "Queries must be a 2D array of shape (K, 3) and visibility must be a 1D array of shape (K,)"

    annotated_image = image.astype(np.uint8)  # Convert to numpy array for OpenCV
    annotated_image = np.ascontiguousarray(annotated_image)

    # Only show visibile queries
    queries = queries[visibility]  # [K, 3] where K is the # visible queries
         
    colormap = get_colormap(queries[:, 0])  # Get colors for the timestamps
    for i, query in enumerate(queries):
        t, x, y = query
        color = colormap[i]  # Get color for the query point based on its timestamp
        cv2.circle(annotated_image, (int(x), int(y)), radius=3, color=color.tolist(), thickness=-1)
   
    if save_image:
        cv2.imwrite(image_name, annotated_image.astype(np.uint8))  # Save the annotated image
        print(f"Annotated image saved to {image_name}")

    return torch.from_numpy(annotated_image).permute(2, 0, 1)  # Convert back to (C, H, W)

def get_query_mask(
        queries: torch.Tensor,   # (1, K, 3)  ── last dim = (x, y, …)
        H: int,
        W: int,
        kernel_size: int = 3
) -> torch.Tensor:
    """
    Build an HxW binary mask that is 1 inside a (kernel_size×kernel_size)
    square around each query point.

    Parameters
    ----------
    queries : (1,K,3) tensor - point coordinates (x,y,␣).
    H, W    : output height / width.
    kernel_size : odd integer ≥1. 3 → a 3x3 patch, 5 → 5x5 …

    Returns
    -------
    mask : (H,W) bool tensor.
    """
    if queries.ndim != 3 or queries.size(0) != 1 or queries.size(-1) < 2:
        raise ValueError("queries must be (1, K, 3) and contain x/y in last dim.")

    # ------------------------------------------------------------------ #
    # 1) integer pixel coordinates
    xy = queries[0, :, :2].round().long()     # (K,2)  where [:,0]=x, [:,1]=y
    x, y = xy[:, 0], xy[:, 1]

    # 2) clip to image bounds
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not torch.any(inside):
        return torch.zeros((H, W), dtype=torch.bool, device=queries.device)

    x, y = x[inside], y[inside]

    # ------------------------------------------------------------------ #
    # 3) build mesh-grid offsets relative to each point
    r = kernel_size // 2
    dy, dx = torch.meshgrid(
        torch.arange(-r, r + 1, device=queries.device),
        torch.arange(-r, r + 1, device=queries.device),
        indexing="ij"
    )                                   # each is (k,k)
    dx = dx.flatten(); dy = dy.flatten()                 # (k²,)

    # (K,1)+(1,k²) → broadcast to (K, k²)
    xs = (x[:, None] + dx[None, :]).clamp(0, W - 1)
    ys = (y[:, None] + dy[None, :]).clamp(0, H - 1)

    # ------------------------------------------------------------------ #
    # 4) scatter into mask
    mask = torch.zeros((H, W), dtype=torch.bool, device=queries.device)
    mask[ys, xs] = True                # advanced indexing, all at once
    return mask

@timecall(immediate=False)
def draw_video_masks(
    video: torch.Tensor,    
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    crumbs: torch.Tensor,
    crumb_sides: torch.Tensor,
    save_video: bool = False,
    filename: str = "annotated_video.mp4"
):
    """Draws masks on the video frames based on the tracks and visibility."""
    assert video.ndim == 5, "Video must be in (B, T, C, H, W) format"
    assert tracks.ndim == 4, "Tracks must be in (B, T, N, 2) format"
    assert visibility.ndim == 3, "Visibility must be in (B, T, N) format"
    assert crumbs.ndim == 3, "Crumbs must be in (B, N, 3) format"
    assert crumb_sides.ndim == 2, "Crumb sides must be in (B, N) format"
    B, T, _, H, W = video.shape

    crumb_times = crumbs[:, :, 0].int().cpu().numpy()  # Shape (B, N) for crumb times
    crumb_sides = crumb_sides.cpu().numpy()

    video = video[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # Convert to numpy array
    annotated_video = video.copy()  # Copy the video to draw on
    
    # For each frame, draw the trajectory masks using the visible points
    # for idx, frame in enumerate(video):
    for idx in range(T):
        frame = np.ascontiguousarray(video[idx])  # Ensure contiguous memory layout for OpenCV
        ftracks = tracks[0, idx].cpu().numpy()  # Shape (N, 2) for the current frame
        fvisibility = visibility[0, idx].cpu().numpy()  # Shape (N,)

        # override for testing
        fvisibility = np.ones_like(fvisibility, dtype=np.float32)  # All visible for testing

        # Filter all tracks, crumbs, and times by visibility
        visible_mask = fvisibility > 0
        ftracks = ftracks[visible_mask]  # Shape (N_visible, 2)
        fcrumb_times = crumb_times[0, visible_mask]  # Shape (N_visible, 3)
        fcrumb_sides = crumb_sides[0, visible_mask]  # Shape (N_visible,)

        mask = build_mask_from_crumbs(
            frame, ftracks, fcrumb_times, fcrumb_sides
        )["mask"]
        if mask is None:
            print(f"No visible crumbs in frame {idx}, skipping mask drawing.")
            continue

        # Draw the mask on the frame
        overlay_mask = annotated_video[idx].copy()  # Copy the frame to draw on
        overlay_mask[mask] = (51, 255, 255)
        annotated_video[idx] = cv2.addWeighted(
            annotated_video[idx], 0.5, overlay_mask, 0.5, 0)
        
        # Draw current tracks on frame
        for track in ftracks:
            x, y = int(track[0]), int(track[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(annotated_video[idx], (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    if save_video:
        success = save_video_np(annotated_video, filename)
        if success:
            print(f"Annotated video saved to {filename}")

def filter_tracks_horizon(
    tracks: torch.Tensor,
    horizon_line: int
) -> torch.Tensor:
    """Remove points above the horizon line from the tracks.

    Args:
        tracks: Tensor of shape (B, T, N, 2), where last dim is (x, y).
        horizon_line: Pixel row (from top) defining the horizon. Points with y < horizon are above.

    Returns:
        Tensor of shape (B, T, N) of bools: True if point is below the horizon, False otherwise.
    """
    assert tracks.ndim == 4, "Tracks must be in (B, T, N, 2) format"

    # Extract y-coordinate for each track point (pixel row)
    y_coords = tracks[..., 1]  # shape (B, T, N)
    # Visible if below (or on) the horizon line
    vis = y_coords >= horizon_line
    return vis.bool()

def sample_grid_points(
    video: torch.Tensor,
    max_num_points: int,
    mask: Optional[np.ndarray] = None,
    window_size: int = 20,
    grid_size: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample up to `max_num_points` points per temporal window by placing
    a regular spatial grid spaced `grid_size` pixels apart, optionally restricted
    by a binary mask.

    Args:
        video: (1, T, 3, H, W) torch tensor of video frames.
        max_num_points: total number of points to sample (across all windows).
        mask: optional (H, W) boolean array; if provided, only sample inside mask.
        window_size: number of frames per temporal window.
        grid_size: spacing (in pixels) between grid points.

    Returns:
        queries: (1, K, 3) tensor of (t, x, y) coordinates
        sides:   (1, K) tensor of side indicators (-1 for left half, +1 for right half)
    """
    import numpy as np
    import torch

    # -- validate inputs --
    assert video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3, \
        "Expected video of shape (1, T, 3, H, W)"
    B, T, C, H, W = video.shape
    assert max_num_points > 0, "max_num_points must be positive"

    # -- prepare mask for sampling --
    if mask is None:
        frame_mask = np.ones((H, W), dtype=bool)
    else:
        assert mask.shape == (H, W), f"Mask must be (H, W), got {mask.shape}"
        frame_mask = mask.astype(bool)

    # -- temporal windows --
    num_windows = T // window_size
    # build window ranges, including first & last partial windows
    buffer = 1
    ranges = []
    # first small window
    ranges.append((0, buffer))
    # full windows
    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, T)
        if end - start > buffer:
            ranges.append((start, end))
    # last small window
    ranges.append((T - buffer, T))

    n_per_window = int(np.ceil(max_num_points / len(ranges)))

    all_queries = []
    all_sides = []

    # convert video to numpy just to get shape (we don't actually use pixel values)
    # video_np = video[0].permute(0, 2, 3, 1).cpu().numpy()

    # precompute spatial grid
    xs = np.arange(0, W, grid_size)
    ys = np.arange(0, H, grid_size)
    grid_xy = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)  # (G, 2)

    for (start, end) in ranges:
        # pick representative frame (middle of window)
        t = int((start + end) // 2)

        # filter grid by mask at this frame
        pts = grid_xy
        if frame_mask is not None:
            inside = frame_mask[pts[:,1].astype(int), pts[:,0].astype(int)]
            pts = pts[inside]

        if pts.shape[0] == 0:
            continue

        # sample up to n_per_window without replacement
        k = min(n_per_window, pts.shape[0])
        choice = np.random.choice(pts.shape[0], size=k, replace=False)
        sel = pts[choice]

        # build queries and sides
        ts = np.full((k, 1), t, dtype=np.float32)
        qs = np.hstack((ts, sel.astype(np.float32)))   # (k, 3)
        sd = np.where(sel[:, 0] < (W / 2), -1.0, 1.0)   # (k,)

        all_queries.append(qs)
        all_sides.append(sd)

    if not all_queries:
        # no points sampled
        return torch.zeros((1,0,3)), torch.zeros((1,0))

    queries_np = np.vstack(all_queries)  # (K, 3)
    sides_np   = np.hstack(all_sides)    # (K,)

    # clamp to max_num_points if needed
    if queries_np.shape[0] > max_num_points:
        idx = np.random.choice(queries_np.shape[0], size=max_num_points, replace=False)
        queries_np = queries_np[idx]
        sides_np   = sides_np[idx]

    # to torch
    queries = torch.from_numpy(queries_np).float().unsqueeze(0)  # (1, K, 3)
    sides   = torch.from_numpy(sides_np).float().unsqueeze(0)    # (1, K)

    print(f"Total grid points sampled: {queries.shape[1]}")
    return queries, sides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: compute all window start indices efficiently
# ---------------------------------------------------------------------------
def _compute_window_starts(
    T: int,
    num_frames: int,
    frame_step: int,
    odom_rev: Optional[np.ndarray] = None,   # reversed odom for *snippet*
    odom_step: Optional[float] = None
) -> List[int]:
    """
    Return reversed-time indices that start each sliding window.

    • With odometry → next start is first frame whose XY displacement from the
      current start ≥ odom_step.
    • Otherwise → add frame_step.
    """
    starts = []
    cur = 0
    while cur + num_frames <= T:
        starts.append(cur)

        if odom_rev is not None and odom_step is not None:
            # displacement from current frame to every *future* frame
            disp = np.linalg.norm(
                odom_rev[cur + 1:, 1:3] - odom_rev[cur, 1:3], axis=1
            )
            nxt = np.where(disp >= odom_step)[0]
            if nxt.size == 0:
                break
            cur = cur + nxt[0] + 1          # +1 → index of that future frame
        else:
            cur += frame_step

    if not starts:                          # make sure we emit at least one
        starts.append(0)
    return starts


# ---------------------------------------------------------------------------
# Helper: save one window (unchanged logic, compacted)
# ---------------------------------------------------------------------------
def _export_window(
    win_start: int,
    win_end: int,
    ride_info: Dict,
    tr_video: torch.Tensor,               # (1,T,C,H,W)
    pred_tracks_all: torch.Tensor,        # (1,T,Ntot,2)  anchors+crumbs
    pred_vis_all: torch.Tensor,           # (1,T,Ntot)
    num_anchors: int,                     # count of anchor queries (front part)
    tr_queries: torch.Tensor,             # (1,Ntot,3)   original (t,x,y) in reversed timeline
    tr_crumb_sides: torch.Tensor,         # (1,Ncrumb)
    out_dir: Path,
    snippet_start: int,                   # orig-timeline idx of first frame
    snippet_len: int                      # = tr_video.shape[1]
):
    """
    Save one window [win_start, win_end) with the original 4-pane mosaic.
    We first sample *support* SIFT crumbs inside a trapezoid on the window's
    last frame, then *augment* the inputs (tracks/vis/queries/sides) so the
    support points are treated as if they were present from the start.
    """
    if win_end <= win_start:
        return

    # --- slice tensors for this window ----------------------------------------
    v_win = tr_video[:, win_start:win_end]                 # (1,L,C,H,W)
    t_all = pred_tracks_all[:, win_start:win_end].clone()  # (1,L,Ntot,2)
    v_all = pred_vis_all[:, win_start:win_end].clone()     # (1,L,Ntot)
    L     = v_win.shape[1]

    # --- last frame image ------------------------------------------------------
    last_img = v_win[0, -1].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    H, W    = last_img.shape[:2]

    # --- sample support crumbs on LAST frame (trapezoid mask) -----------------
    grid_w = W * ROBOT_WIDTH_PCT
    grid_h = H * ROBOT_HEIGHT_PCT
    ground_mask = get_rectangular_mask(H, W, grid_w, grid_h, SAMPLE_AT_EDGES).astype(bool)

    n_sup = max(32, int(MAX_QUERY_POINTS // 6))  # small, stable bump
    sift_kwargs_sup = {"contrastThreshold": 0.006, "edgeThreshold": 24,
                       "nOctaveLayers": 4, "sigma": 1.2}
    sup = sift_in_mask(
        last_img[None, ...],               # (1,H,W,3)
        n_samples=n_sup,
        seg_mask=ground_mask[None, ...],   # (1,H,W)
        sift_kwargs=sift_kwargs_sup
    )
    keyps_sup = sup.get("keyps", [])
    keyps_sup = keyps_sup[0] if len(keyps_sup) > 0 else np.zeros((0, 2), dtype=np.float32)
    if keyps_sup.size > 0:
        valid = ~np.isnan(keyps_sup).any(axis=1)
        keyps_sup = keyps_sup[valid].astype(np.float32)
    Ksup = keyps_sup.shape[0]

    # --- augment inputs with support points (treat as if always present) ------
    if Ksup > 0:
        # side = -1 for left half, +1 for right half
        sides_sup_np = np.where(keyps_sup[:, 0] < (W / 2), -1, 1).astype(np.int8)
        # time index = window's last reversed frame
        t_sup_np = np.full((Ksup,), float(win_end - 1), dtype=np.float32)

        # queries: (1, Ksup, 3)
        sup_queries = torch.from_numpy(
            np.column_stack([t_sup_np, keyps_sup])  # (Ksup,3)
        ).unsqueeze(0).to(tr_queries.device, dtype=tr_queries.dtype)

        # sides: (1, Ksup)
        sup_sides = torch.from_numpy(sides_sup_np).unsqueeze(0).to(tr_crumb_sides.device)

        # tracks: replicate last-frame coords across all L timesteps
        pts_sup = torch.from_numpy(keyps_sup).to(t_all.device, dtype=t_all.dtype)  # (Ksup,2)
        sup_tracks = pts_sup.view(1, 1, Ksup, 2).repeat(1, L, 1, 1)                # (1,L,Ksup,2)

        # visibility: ones across all timesteps
        sup_vis = torch.ones((1, L, Ksup), dtype=v_all.dtype, device=v_all.device)

        # concat to *front-facing* arrays
        t_all = torch.cat([t_all, sup_tracks], dim=2)            # (1,L,Ntot+Ksup,2)
        v_all = torch.cat([v_all, sup_vis],   dim=2)             # (1,L,Ntot+Ksup)
        tr_queries = torch.cat([tr_queries, sup_queries], dim=1) # (1,Ntot+Ksup,3)
        tr_crumb_sides = torch.cat([tr_crumb_sides, sup_sides], dim=1)  # (1,Ncrumb+Ksup)

    # --- split crumbs (everything after anchors) -------------------------------
    t_crumb = t_all[:, :, num_anchors:]                    # (1,L,Ncrumb',2)
    v_crumb = v_all[:, :, num_anchors:]                    # (1,L,Ncrumb')
    q_crumb = tr_queries[:, num_anchors:]                  # (1,Ncrumb',3)

    # --- pick points alive in last frame (now includes support) ----------------
    alive_last = v_crumb[0, -1].cpu().numpy().astype(bool)
    if alive_last.sum() == 0:
        return

    xy_last = t_crumb[0, -1].cpu().numpy()[alive_last]     # (Na,2)
    tm_last = q_crumb[0, alive_last, 0].cpu().numpy()      # (Na,)
    sd_last = tr_crumb_sides[0, alive_last].cpu().numpy()  # (Na,)

    # --- fit path mask on tracked + support -----------------------------------
    mask_dict = build_mask_from_crumbs(last_img, xy_last, tm_last, sd_last)
    if (mask_dict["mask"] is None) or (not mask_dict["success"]):
        return

    # --- 4-pane mosaic (all queries include support) ---------------------------
    orig_img = last_img.copy()

    all_track = torch.cat([tr_queries[:, :, 0:1], t_all[:, -1]], dim=-1)      # (1,Ntot',3)
    all_img = annotate_query_points(
        v_win[0, -1], all_track, v_all[:, -1], save_image=False
    ).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    crumb_track = torch.cat([q_crumb[:, :, 0:1], t_crumb[:, -1]], dim=-1)     # (1,Ncrumb',3)
    crumb_img = annotate_query_points(
        v_win[0, -1], crumb_track, v_crumb[:, -1], save_image=False
    ).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    mask_img = last_img.copy()
    overlay  = mask_img.copy()
    overlay[mask_dict["mask"]] = (51, 255, 255)
    mask_img = cv2.addWeighted(mask_img, 0.5, overlay, 0.5, 0)

    mosaic = np.vstack((np.hstack((orig_img, all_img)),
                        np.hstack((crumb_img, mask_img))))

    # --- filenames / directories (different seq tags) --------------------------
    orig_start = snippet_start + (snippet_len - win_end)
    orig_end   = snippet_start + (snippet_len - win_start)      # exclusive
    seq_tag_dir    = f"seq_{orig_start}"                        # for directory
    seq_tag_mosaic = f"seq_{orig_start}_{orig_end}"             # for mosaic

    ride_id, d0, d1, t_id = get_frodo_id(ride_info["ride_name"])
    out_lbl_dir = set_frodo_dir(out_dir, ride_id, d0, d1, t_id) / seq_tag_dir
    out_img_dir = out_dir / "path_tracker_debug" / "images"
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # --- save front-camera clip ------------------------------------------------
    save_video_np(
        v_win[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8),
        out_lbl_dir / "front_camera.mp4", fps=15, quality=20
    )

    # --- save labels (support already concatenated) ----------------------------
    hickle.dump({
        "tracks"      : t_crumb.cpu().numpy(),                 # crumbs+support
        "visibility"  : v_crumb.cpu().numpy(),
        "crumbs"      : q_crumb.cpu().numpy(),
        "sides"       : tr_crumb_sides.cpu().numpy().astype(np.int8),
        "path_mask"   : mask_dict["mask"].astype(bool),
        "front_rgb"   : orig_img,
        "mask_success": True,
    }, out_lbl_dir / "path_tracker.h5", mode='w')

    # --- save 4-pane mosaic with start+end tag --------------------------------
    mosaic_name = f"{ride_info['ride_name']}_{seq_tag_mosaic}.jpg"
    cv2.imwrite(out_img_dir / mosaic_name, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_track.jpg", cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))

    # --- save annotated video with all crumbs and support points ------
    if SAVE_VIDEO_MASKS:
        out_vid_dir = out_dir / "path_tracker_debug" / "videos"
        out_vid_dir.mkdir(parents=True, exist_ok=True)
        out_vid_path = out_vid_dir / f"{ride_info['ride_name']}_{seq_tag_mosaic}.mp4"
        print(f"Saving annotated video to {out_vid_path}")
        draw_video_masks(
            v_win,              # (1, L, C, H, W)
            t_crumb,            # (1, L, N_crumb, 2)  ← was t_all (mismatched N)
            v_crumb,            # (1, L, N_crumb)     ← was v_all (mismatched N)
            q_crumb,            # (1, N_crumb, 3)
            tr_crumb_sides,     # (1, N_crumb)
            save_video=True,
            filename=out_vid_path
        )

# ---------------------------------------------------------------------------
# Full migrated process_ride_single
# ---------------------------------------------------------------------------
@timecall(immediate=False)
def process_ride_single(video, ride_info, out_dir, reinit_tracks, odom=None):
    """
    One-shot CoTracker over the reversed clip, then export windows.
    Odometry is first clipped to the *same* [start_frame:end_frame) range
    so that indices match the snippet’s length.
    """
    # ------------------------------------------------------------------ #
    # 0.  Pre-process video (reverses internally)
    # ------------------------------------------------------------------ #
    video = preprocess_video(video)
    if video is None:
        print("Pre-processing produced empty clip.")
        return

    B, T, _, H, W = video.shape
    device = ("cuda" if torch.cuda.is_available()
              else "mps"  if torch.backends.mps.is_available()
              else "cpu")

    # ------------------------------------------------------------------ #
    # 1.  Clip / reverse odometry so length matches T
    # ------------------------------------------------------------------ #
    if odom is not None:
        # odom indexes were absolute for the *whole* drive → clip to segment
        start_frame = ride_info['start_frame']
        seg_odom = odom[start_frame:start_frame + T]
        if seg_odom.shape[0] != T:
            raise ValueError("Odometry / video length mismatch after clipping")
        odom_rev = seg_odom[::-1]          # reverse to match reversed frames
    else:
        odom_rev = None

    # ------------------------------------------------------------------ #
    # 2.  Sample queries (anchors + crumbs) as before
    # ------------------------------------------------------------------ #
    robot_w_px = W * ROBOT_WIDTH_PCT
    robot_h_px = H * ROBOT_HEIGHT_PCT
    qmask = get_rectangular_mask(H, W, robot_w_px, robot_h_px, SAMPLE_AT_EDGES)

    crumbs, crumb_sides = sample_sift_mask(
        video, MAX_QUERY_POINTS, qmask,
        sift_kwargs={"edgeThreshold": 20, "sigma": 1.5,
                     "nOctaveLayers": 5, "contrastThreshold": 0.018},
        window_size=WINDOW_SIZE)
    # crumbs, crumb_sides = sample_sift_mask_adaptive(
    #     video, MAX_QUERY_POINTS, qmask,
    #     window_size=WINDOW_SIZE,
    #     tiers=[
    #         # {"edgeThreshold": 18, "sigma": 1.6, "nOctaveLayers": 7, "contrastThreshold": 0.02},
    #         {"contrastThreshold": 0.018, "edgeThreshold": 20, "nOctaveLayers": 5, "sigma": 1.5},
    #         {"contrastThreshold": 0.012, "edgeThreshold": 24, "nOctaveLayers": 5, "sigma": 1.4},
    #         # {"contrastThreshold": 0.008, "edgeThreshold": 26, "nOctaveLayers": 4, "sigma": 1.2},
    #     ],
    #     min_points=int(0.8 * MAX_QUERY_POINTS),
    #     dedup_radius_px=1,
    # )

    anchors, _ = sample_sift_mask(
        video, MAX_QUERY_POINTS,
        window_size=WINDOW_SIZE,
        sift_kwargs={'contrastThreshold': 0.008, 'edgeThreshold': 15,
                     'nOctaveLayers': 3, 'sigma': 1.2})
    # anchors, _ = sample_sift_mask_adaptive(
    #     video, MAX_QUERY_POINTS,
    #     mask=None,
    #     window_size=1,
    #     tiers=[
    #         {'contrastThreshold': 0.008, 'edgeThreshold': 15, 'nOctaveLayers': 3, 'sigma': 1.2}
    #         # {"contrastThreshold": 0.010, "edgeThreshold": 18, "nOctaveLayers": 3, "sigma": 1.4},
    #         # {"contrastThreshold": 0.008, "edgeThreshold": 22, "nOctaveLayers": 3, "sigma": 1.3},
    #         # {"contrastThreshold": 0.006, "edgeThreshold": 24, "nOctaveLayers": 3, "sigma": 1.2},
    #     ],
    #     min_points=int(MAX_QUERY_POINTS),
    #     dedup_radius_px=0,
    # )

    if anchors.shape[1] == 0 or crumbs.shape[1] == 0:
        print("No anchors or crumbs – abort.")
        return

    num_anchors  = anchors.shape[1]
    queries      = torch.cat([anchors, crumbs], dim=1).to(device)
    video        = video.to(device)
    crumb_sides  = crumb_sides.to(device)

    # ------------------------------------------------------------------ #
    # 3.  One CoTracker pass over full clip
    # ------------------------------------------------------------------ #
    model = CoTrackerOnlinePredictor(
        checkpoint='./external/co-tracker/checkpoints/scaled_online.pth'
    ).to(device)
    model(video_chunk=video, is_first_step=True,
          grid_size=0, queries=queries, add_support_grid=True)

    for ind in range(0, T - model.step, model.step):
        pred_tracks, pred_visibility = model(
            video_chunk=video[:, ind: ind + model.step * 2],
            grid_size=0, queries=queries, add_support_grid=True)
    
    # ------------------------------------------------------------------ #
    # 4.  Separate crumbs
    # ------------------------------------------------------------------ #
    pred_tracks_all = pred_tracks           # keep full set (anchors+crumbs)
    pred_vis_all    = pred_visibility
    tr_crumb_sides        = crumb_sides
    tr_video              = video

    # ------------------------------------------------------------------ #
    # 5.  Pre-compute all window starts once
    # ------------------------------------------------------------------ #

    starts = _compute_window_starts(
        T, NUM_FRAMES, FRAME_STEP,
        odom_rev=odom_rev, odom_step=ODOMETRY_STEP
    )

    # ------------------------------------------------------------------ #
    # 6.  Export each window
    # ------------------------------------------------------------------ #
    for s in starts:
        _export_window(
            s, s + NUM_FRAMES,
            ride_info,
            tr_video,
            pred_tracks_all, pred_vis_all,
            num_anchors,
            queries,                  # = tr_queries (anchors+crumbs)
            tr_crumb_sides,
            out_dir,
            snippet_start=ride_info['start_frame'],
            snippet_len=T
        )

    return

def main():
    ap = argparse.ArgumentParser("Backward sparse tracking with CoTracker-online")
    ap.add_argument("--cfg_file", type=str, default="scripts/preprocessing/config/frodo_mining.yaml",)
    ap.add_argument("--root_dir", type=Path, default=None, help="Path to the dataset directory or video file")
    ap.add_argument("--out_dir", default=None, type=Path)
    ap.add_argument("--split_path", type=Path, default=None, help="Path to the split CSV file with rides")
    ap.add_argument("--dataset", type=str, default="frodo", choices=["frodo", "h5", "mp4", "split"])
    ap.add_argument("--rows", type=str, default=None, help="Comma-separated list of rows to process from the split file")
    ap.add_argument("--reinit_tracks", action="store_true", help="Reinitialize tracks if they are not visible in the last frame")
    ap.add_argument("-s", "--start_frame", type=int, default=None, help="Start frame for the video")
    ap.add_argument("-e", "--end_frame", type=int, default=None, help="End frame for the video")
    args = ap.parse_args()

    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.root_dir is not None:
        cfg['root_dir'] = str(args.root_dir)
    if args.out_dir is not None:
        cfg['out_dir'] = str(args.out_dir)
    root_dir = Path(cfg['root_dir'])
    out_dir = Path(cfg['out_dir'])
    assert root_dir.exists(), f"Root directory {root_dir} does not exist"

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Set global parameters from the config
    set_global_params(cfg)

    ride_info = {
        "ride_name": "test_drive",
        "start_frame": args.start_frame,
        "end_frame": args.end_frame
    }
    if args.dataset == "split":
        try:
            rides = pd.read_csv(args.split_path)
            # Extract day from ride_name columns 2025-08-06-16-25-00 ferrite7
            # rides["day"] = rides["ride_name"].str.split(" ").str[0]
            # date_cutoff = "2025-08-12-14-35-00" # Last day
            # rides = rides[rides["day"] > date_cutoff]
            
            # if rides.empty:
            #     print(f"No rides found after {date_cutoff} in {args.split_path}")
            #     return
        except EmptyDataError:
            print(f"Split file {args.split_path} is empty or not found.")
            return
        
        if args.rows is not None and args.rows.strip():
            try:
                idx = [int(x) for x in args.rows.split(",") if x.strip() != ""]
            except ValueError:
                raise ValueError(f"Invalid --rows argument: {args.rows}")
            rides = rides.iloc[idx]

        # Create tqdm with ride
        start_flag = False
        for og_ride_name, og_start_frame, og_end_frame in tqdm(zip(rides["ride_name"], rides["start_frame"], rides["end_frame"]), desc="Processing rides"):
            ride_id, drive_id0, drive_id1, timestamp = og_ride_name.split(" ")
            video_dir = set_frodo_dir(root_dir, *og_ride_name.split(" "))
            video_path = video_dir / "front_camera.mp4"
            assert video_path.exists(), f"Video file {video_path} does not exist"

            # if ride_name != "6 30470 39b458 20240413114534" and not start_flag:
            #     continue
            # start_flag = True

            ride_name = set_frodo_id(*og_ride_name.split(" "))  # Set the ride id
            video_full, start_frame, end_frame = \
                load_video_from_file(video_path, og_start_frame, og_end_frame)

            # Load odometry if provided
            odometry_path = video_dir / f"odometry_data_{drive_id0}.csv"
            camera_ts_path = video_dir / f"front_camera_timestamps_{drive_id0}.csv"
            if odometry_path.exists() and camera_ts_path.exists():
                odo = load_odom(odometry_path)
                cam_ts = load_timestamps(camera_ts_path)
                odo_full = interpolate_se3(cam_ts, odo[:, 0], odo[:, 1:4], odo[:, 4:8])
                assert odo_full.shape[0] == cam_ts.shape[0] == video_full.shape[0], \
                    f"Odometry length {odo_full.shape[0]} does not match camera timestamps length {cam_ts.shape[0]}"
            else:
                odo_full = None

            if video_full is None:
                print(f"Video {video_path} is empty or could not be loaded, skipping ride {ride_name}")
                continue
            H, W = video_full.shape[1], video_full.shape[2]
            if [H, W] not in ACCEPTABLE_RESOLUTIONS:
                print(f"Skipping ride {ride_name} with unexpected resolution {H}x{W}")
                continue
            ride_info = {
                "ride_name": ride_name,
                "start_frame": start_frame,
                "end_frame": og_end_frame, # Only process until requested end frame
            }

            video = video_full[start_frame:end_frame]  # Slice video to the current segment
            process_ride_single(
                video,
                ride_info,
                out_dir,
                args.reinit_tracks,
                odom=odo_full
            )

        print(f"Finished processing {len(rides)} rides from {args.split_path}")
        return
    elif args.dataset == "frodo":
        video_path = choose_frodo_ride(root_dir)
        print(f"Loading Frodo ride video from {video_path}")
        video, start_frame, end_frame = load_video_from_file(video_path)
        ride_id, drive_id0, drive_id1, timestamp = get_frodo_raw_id(video_path.parent)
        ride_info = {
            'ride_name': set_frodo_id(ride_id, drive_id0, drive_id1, timestamp),
            'start_frame': start_frame,
            'end_frame': end_frame
        }
    elif args.dataset == "h5":
        h5_path = choose_h5_file(root_dir)
        print(f"Loading h5 dataset video from {h5_path}")
        video, start_frame, end_frame = load_video(h5_path)
        ride_info = {
            'ride_name': h5_path.stem,
            'start_frame': start_frame,
            'end_frame': end_frame
        }
    elif args.dataset == "mp4":
        print(f"Loading video from {root_dir}")
        video, start_frame, end_frame = load_video_from_file(root_dir, args.start_frame, args.end_frame)
        ride_id, drive_id0, drive_id1, timestamp = get_frodo_raw_id(root_dir)
        ride_info = {
            'ride_name': set_frodo_id(ride_id, drive_id0, drive_id1, timestamp),
            'start_frame': start_frame,
            'end_frame': end_frame
        }
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")

    H, W = video.shape[1], video.shape[2]
    if (H, W) not in ACCEPTABLE_RESOLUTIONS:
        print(f"Skipping ride {ride_info['ride_name']} with unexpected resolution {H}x{W}")
        return
    process_ride_single(video, ride_info, out_dir, args.reinit_tracks)

if __name__ == "__main__":
    main()
