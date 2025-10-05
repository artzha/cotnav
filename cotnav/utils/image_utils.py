import os
from typing import Optional, Union
import numpy as np
import torch
import imageio
import cv2

def _to_uint8_rgb_numpy(frames: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """
    Convert a numpy or torch tensor of shape (B, H, W, 3) into a contiguous uint8 numpy array.
    Assumes RGB channel order.
    """
    # Move torch -> cpu numpy
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames of shape (B, H, W, 3), got {frames.shape}")

    arr = frames

    # Scale to [0, 255] uint8 if needed
    if arr.dtype == np.float32 and arr.max() <= 1.0:
        arr = (arr * 255.0).round().astype(np.uint8)

    # Ensure contiguous memory for fast I/O
    arr = np.ascontiguousarray(arr)
    return arr


def save_video(
    frames: Union[np.ndarray, "torch.Tensor"],
    out_path: Union[str, os.PathLike],
    fps: int = 30,
    codec: str = "libx264",
    quality: int = 5,
    bitrate: Optional[str] = None,
) -> None:
    """
    Save a batch of RGB frames (B, H, W, 3) to a video file efficiently.

    Args:
        frames: np.ndarray or torch.Tensor, shape (B, H, W, 3), RGB.
        out_path: Output file path (e.g., 'out.mp4').
        fps: Frames per second.
        codec: FFmpeg codec (e.g., 'libx264', 'libx265', 'mpeg4').
        quality: ImageIO-ffmpeg quality (-1..10), lower is better quality/bitrate.
        bitrate: Optional FFmpeg bitrate string (e.g., '8M'). Overrides quality if provided.
    """
    frames_u8 = _to_uint8_rgb_numpy(frames)
    if frames_u8.shape[0] == 0:
        raise ValueError("No frames to write (B == 0).")

    out_path = str(out_path)

    writer_args = {
        "fps": fps,
        "codec": codec,
        "pixelformat": "yuv420p",  # widely compatible
        "macro_block_size": 1,     # allow any resolution
    }
    if bitrate is not None:
        writer_args["bitrate"] = bitrate
    else:
        writer_args["quality"] = quality

    with imageio.get_writer(out_path, **writer_args) as writer:
        for f in frames_u8:
            writer.append_data(f)
    return