from pathlib import Path
import cv2
from imageio import get_writer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from rosbags.highlevel import AnyReader
from typing import List, Dict, Tuple, Iterable, Any, Union, Optional


from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from pytransform3d.transform_manager import TransformManager

def get_ros_type(bag_file: str | Path, topic: str) -> Optional[str]:
    """
    Return the fully-qualified ROS message type for *topic* inside *bag_file*.
    """
    bag_file = Path(bag_file)
    try:
        with AnyReader([bag_file]) as reader:
            # `reader.topics` is populated from the bag header(s) only
            info = reader.topics.get(topic)
            if info is not None:
                return info.msgtype          # e.g. "nav_msgs/Odometry"
    except FileNotFoundError:
        raise
    return None

def _topic_to_filename(topic: str, ext: str | None = None) -> Path:
    """"""
    topic = topic.strip()                      # trim accidental spaces
    segs  = [s for s in topic.lstrip("/").split("/") if s]

    if len(segs) < 2:                          # nothing to parse
        return Path(topic.lstrip("/").replace("/", "_") + ext)

    sensor = segs[1]                           # usually the sensor label

    # special-case: Ouster IMU   →  "ouster_imu_data"
    if sensor == "ouster" and (len(segs) >= 3 and segs[2] == "imu"):
        sensor = "ouster_imu_data"

    if ext is None:
        return f"{sensor}"
    return f"{sensor}.{ext}"

def build_transforms(
    bag: Union[str, Path],
    topics: List[str],
) -> TransformManager:
    """
    Read all TF messages on *any* of `topics` in `bag` (e.g. ['/tf', '/tf_static']),
    build a TransformManager containing every parent->child edge.
    Note: parent_frame == header.frame_id, child_frame == child_frame_id.
    """
    tm = TransformManager()
    with AnyReader([Path(bag)]) as reader:
        # collect every connection whose topic is in our list
        conns = [c for c in reader.connections if c.topic in topics]
        if not conns:
            raise ValueError(f"No TF topics {topics} found in {bag}")

        # iterate them all
        for conn in conns:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                for tf in msg.transforms:
                    parent = tf.header.frame_id
                    child  = tf.child_frame_id
                    t = tf.transform.translation
                    q = tf.transform.rotation

                    # build 4×4 homogeneous matrix:
                    # rotation first
                    R_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    T = np.eye(4, dtype=float)
                    T[:3, :3] = R_mat
                    T[:3,  3] = [t.x, t.y, t.z]

                    """
                    ROS uses transform from child to parent,
                    so we need to invert it for TransformManager.
                    """
                    tm.add_transform(
                        to_frame=parent,
                        from_frame=child,
                        A2B=T
                    )
    return tm

def transforms_to_yaml(
    transforms: Dict[str, List[float]],
    stream
) -> None:
    """
    Write a mapping of frame-pair keys to 16-element transform lists,
    using inline sequences. Cast values to native Python floats.
    """
    m = CommentedMap()
    for key, flat in transforms.items():
        flat_py = [float(val) for val in flat]
        seq = CommentedSeq(flat_py)
        seq.fa.set_flow_style()
        m[key] = seq

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.dump(m, stream)

def _camera_info_to_yaml(info: Dict[str, Any], stream) -> None:
    """
    Write a ROS-style CameraInfo mapping to `stream` (file or stdout),
    with D/K/R/P as inline [ … ] sequences.
    """
    # 1) Build a CommentedMap for insertion‐order safety and ruamel hooks
    m = CommentedMap()
    m["image_width"]      = int(info["width"])
    m["image_height"]     = int(info["height"])
    m["distortion_model"] = info["distortion_model"]
    m["frame_id"]         = info["frame_id"]

    # 2) Wrap each of your lists in a CommentedSeq and force flow style
    for key in ("D", "K", "R", "P"):
        arr = np.array(info[key], dtype=float).reshape(-1).tolist()
        seq = CommentedSeq(arr)
        seq.fa.set_flow_style()         # ← this makes [a, b, c, …]
        m[key] = seq

    # 3) Dump with ruamel.yaml
    yaml = YAML()
    yaml.default_flow_style = False    # block style for maps
    yaml.dump(m, stream)

def process_tf(
    bag: str | Path,
    topics: List[str],
    frame_pairs: List[Tuple[str, str]],
) -> Dict[str, str]:
    """
    Build a TransformManager from `bag`/`topic`, then for each (src, tgt)
    in `frame_pairs`, lookup the 4×4 transform matrix (skipping pairs
    that cannot be connected). Store into a dict and write to YAML.

    Returns
    -------
    {"yaml": <path_to_yaml>}
    """
    # build the transform graph
    tm = build_transforms(bag, topics)

    # collect transforms in a dict: 'src tgt' -> flat 16-element list
    transforms = {}
    for src, tgt in frame_pairs:
        try:
            """
            ROS uses transform from parent to child
            so we need to invert it for TransformManager.
            """
            T = tm.get_transform(
                from_frame=tgt,
                to_frame=src
            )
        except Exception:
            continue

        key = f"{src} {tgt}"
        transforms[key] = list(T.flatten())

    return {"transforms": transforms}

def _collect_timestamps_per_topic(bags: List[str], topics: List[str]) -> Dict[str, List[float]]:
    """Pre-pass: collect timestamps (seconds) for each image-like topic across all bags."""
    ts_map = {t: [] for t in topics}
    topics_set = set(topics)
    for bag in bags:
        with AnyReader([Path(bag)]) as reader:
            conns = {c.topic: c for c in reader.connections if c.topic in topics_set}
            if not conns:
                continue
            conn_to_topic = {c: t for t, c in conns.items()}
            for conn, _, raw in reader.messages(connections=list(conns.values())):
                # Only care about image-ish messages (includes CameraInfo timestamps too,
                # which we keep separate later; here we only want frames)
                mtype = conn.msgtype
                if not (mtype.endswith("Image") or mtype.endswith("CompressedImage")):
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                # Prefer header time; assume ROS2 style is used in your bags
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                ts_map[conn_to_topic[conn]].append(ts)
    # Ensure sorted
    for t in ts_map:
        ts_map[t].sort()
    return ts_map

def _estimate_fps(timestamps: List[float], default_fps: float) -> float:
    """Estimate FPS from timestamps (seconds). Robust to outliers using median Δt."""
    n = len(timestamps)
    if n <= 1:
        return float(default_fps)
    ts = np.asarray(timestamps, dtype=np.float64)
    dts = np.diff(ts)
    # guard against zero/negative dt (clock glitches); keep positive only
    dts = dts[dts > 0]
    if dts.size == 0:
        return float(default_fps)
    # Use median Δt when enough samples; else mean; else span-based fallback
    if dts.size >= 3:
        dt = float(np.median(dts))
        fps = 1.0 / dt if dt > 0 else default_fps
    else:
        span = ts[-1] - ts[0]
        fps_span = (n - 1) / span if span > 0 else default_fps
        fps_mean = 1.0 / float(np.mean(dts)) if np.mean(dts) > 0 else default_fps
        fps = float(np.clip(np.mean([fps_span, fps_mean]), 0.1, 300.0))
    # Clamp to sane range
    return float(np.clip(fps, 0.1, 300.0))

def _collect_timestamps_per_topic(bags: List[str], topics: List[str]) -> Dict[str, List[float]]:
    ts_map = {t: [] for t in topics}
    topics_set = set(topics)
    for bag in bags:
        with AnyReader([Path(bag)]) as reader:
            conns = {c.topic: c for c in reader.connections if c.topic in topics_set}
            if not conns:
                continue
            conn_to_topic = {c: t for t, c in conns.items()}
            for conn, _, raw in reader.messages(connections=list(conns.values())):
                mtype = conn.msgtype
                if not (mtype.endswith("Image") or mtype.endswith("CompressedImage")):
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                # Prefer header time (ROS2); fall back to iterator time if needed
                ts = getattr(getattr(msg, "header", None), "stamp", None)
                if ts and hasattr(ts, "sec") and hasattr(ts, "nanosec"):
                    ts_s = ts.sec + ts.nanosec * 1e-9
                else:
                    # If no header, we can’t read ts here without iter timestamp; skip for prepass
                    continue
                ts_map[conn_to_topic[conn]].append(ts_s)
    for t in ts_map:
        ts_map[t].sort()
    return ts_map

def _estimate_fps(timestamps: List[float], default_fps: float) -> float:
    n = len(timestamps)
    if n <= 1:
        return float(default_fps)
    ts = np.asarray(timestamps, dtype=np.float64)
    dts = np.diff(ts)
    dts = dts[dts > 0]
    if dts.size == 0:
        return float(default_fps)
    if dts.size >= 3:
        dt = float(np.median(dts))
        fps = 1.0 / dt if dt > 0 else default_fps
    else:
        span = ts[-1] - ts[0]
        fps_span = (n - 1) / span if span > 0 else default_fps
        fps_mean = 1.0 / float(np.mean(dts)) if np.mean(dts) > 0 else default_fps
        fps = float(np.clip(np.mean([fps_span, fps_mean]), 0.1, 300.0))
    return float(np.clip(fps, 0.1, 300.0))

def _open_mp4_writer(path, fps_used, crf=18, preset="veryfast"):
    """
    Cross-platform MP4 writer (libx264 + yuv420p) with auto-padding to even dims.
    """
    return get_writer(
        str(path), fps=float(fps_used),
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=[
            # keep original resolution; pad to even if needed for yuv420p
            "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:x=0:y=0",
            "-crf", str(crf),
            "-preset", preset,         # "veryfast" for lossy, "slow"/"medium" for near-lossless if you want
            "-profile:v", "high",
            "-level", "4.2",
            "-movflags", "+faststart",
        ],
        # prevent imageio from forcing macroblock multiples
        macro_block_size=1,
    )


def process_rgb(
    bags: List[str],
    out_dir: str,
    topics: List[str],
    save_prefixes: List[str],
    seq: str,
    fps: int = 10,
    process_bag: bool = True,
    save_video: bool = True,
    crf_lossy: int = 20,        # smaller
    crf_near: int  = 10,        # near-lossless but still yuv420p (very compatible)
    near_preset: str = "slow",  # you can pick "medium" for speed/size tradeoff
) -> Dict[str, str]:
    """
    Writes per topic:
      • <prefix>.mp4              (H.264 yuv420p, CRF=crf_lossy)
      • <prefix>_near.mp4         (H.264 yuv420p, CRF=crf_near, high quality)
      • <prefix>_timestamps_<seq>.csv
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    artefacts          = {t: {}   for t in topics}
    writers_lossy      = {t: None for t in topics}
    writers_near       = {t: None for t in topics}
    cam_ts             = {t: []   for t in topics}
    infos              = {t: {"timestamps": [], "camera_info": []} for t in topics}
    prefix_map         = {t: save_prefixes[i] for i, t in enumerate(topics)}

    # Pre-pass FPS
    ts_map    = _collect_timestamps_per_topic(bags, topics)
    topic_fps = {t: _estimate_fps(ts_map.get(t, []), fps) for t in topics}

    for bag in bags:
        with AnyReader([Path(bag)]) as reader:
            conns = {c.topic: c for c in reader.connections if c.topic in topics}
            if not conns:
                continue
            conn_to_topic = {c: t for t, c in conns.items()}

            for conn, bag_ts, raw in reader.messages(connections=list(conns.values())):
                topic = conn_to_topic[conn]
                msg   = reader.deserialize(raw, conn.msgtype)

                # timestamp (prefer header)
                ts = bag_ts * 1e-9
                h  = getattr(msg, "header", None)
                if h is not None and getattr(h, "stamp", None) is not None:
                    st = h.stamp
                    if hasattr(st, "sec") and hasattr(st, "nanosec"):
                        ts = st.sec + st.nanosec * 1e-9
                    elif hasattr(st, "secs") and hasattr(st, "nsecs"):
                        ts = st.secs + st.nsecs * 1e-9

                # CameraInfo
                if conn.msgtype.endswith("CameraInfo"):
                    intr = {
                        "K": msg.K.tolist(), "D": msg.D.tolist(), "R": msg.R.tolist(),
                        "P": msg.P.tolist(), "width": msg.width, "height": msg.height,
                        "distortion_model": msg.distortion_model,
                        "frame_id": msg.header.frame_id,
                    }
                    infos[topic]["camera_info"].append(intr)
                    infos[topic]["timestamps"].append(ts)
                    continue

                # Image / CompressedImage → RGB uint8
                if conn.msgtype.endswith("CompressedImage"):
                    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif conn.msgtype.endswith("Image"):
                    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
                    if getattr(msg, "encoding", "").lower() in ("bgr8", "bgr"):
                        img = img[..., ::-1]
                else:
                    continue

                if img is None or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
                    continue

                cam_ts[topic].append(ts)
                H, W = img.shape[:2]
                artefacts[topic].setdefault("orig_height", int(H))
                artefacts[topic].setdefault("orig_width",  int(W))

                if not process_bag or not save_video:
                    continue

                # open writers lazily
                if writers_lossy[topic] is None:
                    prefix   = prefix_map[topic]
                    fps_used = float(topic_fps.get(topic, fps))

                    lossy_path = out_dir / f"{prefix}_lossy.mp4"
                    near_path  = out_dir / f"{prefix}.mp4"

                    writers_lossy[topic] = _open_mp4_writer(lossy_path, fps_used, crf=crf_lossy, preset="veryfast")
                    writers_near[topic]  = _open_mp4_writer(near_path,  fps_used, crf=crf_near,  preset=near_preset)

                    artefacts[topic]["video"]        = str(lossy_path)
                    artefacts[topic]["video_near"]   = str(near_path)
                    artefacts[topic]["fps"]          = fps_used

                writers_lossy[topic].append_data(img)
                writers_near[topic].append_data(img)

    # finalize + metadata
    for topic in topics:
        if writers_lossy[topic]:
            writers_lossy[topic].close()
        if writers_near[topic]:
            writers_near[topic].close()

        prefix   = prefix_map[topic]
        csv_path = out_dir / f"{prefix}_timestamps_{seq}.csv"

        if cam_ts[topic]:
            ts_arr = np.asarray(cam_ts[topic], dtype=np.float64)
            pd.DataFrame({"timestamp": ts_arr}).to_csv(csv_path, index=False)
            artefacts[topic]["timestamps"]  = str(csv_path)
            artefacts[topic]["frame_count"] = int(len(ts_arr))

        if infos[topic]["camera_info"]:
            caminfo_path = out_dir / f"{prefix}_{seq}.yaml"
            with caminfo_path.open("w") as f:
                _camera_info_to_yaml(infos[topic]["camera_info"][0], f)
            artefacts[topic]["camera_info"] = str(caminfo_path)

    return artefacts

# ───────────────────────── odometry processor ───────────────────────
def process_odometry(
    bags: List[str],
    topic: str,
    out_dir: str | Path,
    save_prefix: str,
    save_file: bool = True,
):
    """
    Collapse *bags* on *topic* (nav_msgs/Odometry) into a single CSV:

        ts,x,y,z,qw,qx,qy,qz   (comma-separated, ts in **seconds**)

    Returns
    -------
    {"csv": <path>, "df": pandas.DataFrame}
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "odom.csv"
    rows: list[dict[str, float]] = []

    # ── gather rows across all bags ───────────────────────────────────────
    for bag in bags:
        with AnyReader([Path(bag)]) as reader:
            conn = next((c for c in reader.connections if c.topic == topic), None)
            if conn is None:
                continue

            for _, bag_ts, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)

                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                # nav_msgs/Odometry -> msg.pose.pose.position / orientation
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                rows.append(
                    {
                        "timestamp": ts,      # nanoseconds → seconds
                        "x":  pos.x,
                        "y":  pos.y,
                        "z":  pos.z,
                        "qw": ori.w,             # note: qw FIRST
                        "qx": ori.x,
                        "qy": ori.y,
                        "qz": ori.z,
                    }
                )

    if not rows:
        return None

    # ── write CSV ---------------------------------------------------------
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if save_file:
        df.to_csv(csv_path, index=False)            # comma-separated by default

    return {"csv": str(csv_path), "df": df}