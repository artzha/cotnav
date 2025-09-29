#!/usr/bin/env python3
import os
import argparse, yaml
from pathlib import Path
from typing import Dict, List
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
from scripts.utils.log_utils import logging
from scripts.utils.loader_utils import construct_filters

from scripts.preprocessing.process_utils import inspect_and_stream_local

DEBUG_MODE=True

# ------------------ helpers ----------------------------------------------
def load_cfg(fp: Path) -> dict:
    with fp.open() as f:
        return yaml.safe_load(f)

def find_sessions_local(bags_root: Path) -> List[Path]:
    """
    Treat every directory that contains at least one *.bag as a 'session'.
    Recurse from bags_root to find them.
    """
    sessions = set()
    for root, _, files in os.walk(bags_root):
        if any(f.endswith(".bag") for f in files):
            sessions.add(Path(root))
    return sorted(sessions)

# ------------------ main --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cfg", required=True,
                    help="YAML config that includes 'topics' and optional 'n_jobs', 'root_dir'")
    ap.add_argument("--bags_dir", required=True,
                    help="Local directory containing bags (recursively). Each directory with *.bag is a session.")
    args = ap.parse_args()

    cfg       = load_cfg(Path(args.cfg))
    save_root = cfg.get("root_dir", "./data/fai_raw_local")
    filters   = construct_filters(cfg)

    # Topic spec stays the same structure as before (per-robot OR flat).
    # If your cfg has per-robot topics, set robot via --robot or infer from folder name.
    topics_cfg = cfg["topics"]  # either {robot: {...}} or flat {...}

    # Discover sessions (local dirs containing .bag)
    sessions = find_sessions_local(Path(args.bags_dir))
    if not sessions:
        logging.info("No local sessions (folders containing *.bag) found — exiting.")
        return

    if DEBUG_MODE:
        dfs = []
        for sess_dir in sessions:
            df = inspect_and_stream_local(
                sess_dir=str(sess_dir),
                save_root_dir=save_root,
                topics_dict=topics_cfg,
                filters_dict=filters
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs.append(df)
    else:
        n_jobs = cfg.get("n_jobs", 8)
        dfs = []
        with tqdm(total=len(sessions), desc="Sessions") as pbar:
            for df in Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(inspect_and_stream_local)(
                    sess_dir=str(sess),
                    save_root_dir=save_root,
                    topics_dict=topics_cfg,
                    filters_dict=filters
                ) for sess in sessions
            ):
                if isinstance(df, pd.DataFrame) and not df.empty:
                    dfs.append(df)
                pbar.update(1)

    if not dfs:
        logging.info("No sessions produced metadata; nothing to combine.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    (Path(save_root) / "all_metadata.csv").parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(Path(save_root) / "all_metadata.csv", index=False)

    # Build simple per-ride manifests (compatible with your downstream)
    from collections import defaultdict
    rides: dict[str, list[dict]] = defaultdict(list)
    for row in combined.to_dict("records"):
        ride_dir = str(Path(row["video"]).parent.parent) if row.get("video") else None
        if ride_dir:
            rides[ride_dir].append(row)

    for ride_dir, rows in rides.items():
        txt_path = Path(ride_dir) / "ride_manifest.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_path.open("w") as f:
            f.write("ride_name,child_dt,video,odometry,controls\n")
            for r in rows:
                ride_info = get_frodo_raw_id(r["video"]) if r.get("video") else ("unknown",)
                ride_name = " ".join(ride_info)
                f.write(f"{ride_name},{r.get('child_dt','')},{r.get('video','')},{r.get('odometry','')},{r.get('controls','')}\n")

    logging.info(f"✅  {len(dfs)}/{len(sessions)} sessions processed "
                 f"({len(combined)} rows total, {len(rides)} ride manifests)")

if __name__ == "__main__":
    main()
