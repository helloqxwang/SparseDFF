import argparse
from pathlib import Path

from tqdm import tqdm
import time

from manager import MultiCameraManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./workspace")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--disable_color_stream", action="store_true")
    parser.add_argument("--disable_depth_stream", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = MultiCameraManager(
        save_dir=Path(args.save_dir),
        enable_color_stream=not args.disable_color_stream,
        enable_depth_stream=not args.disable_depth_stream,
        tranformed_color=False
    )

    for i in tqdm(range(10000)):
        frames = manager.wait_for_frames()

    else:
        input()