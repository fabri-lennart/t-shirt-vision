"""Extract video frames for YOLO bounding-box annotation."""

import argparse
from pathlib import Path

import cv2


def extract_frames(video: Path, destination: Path, every_seconds: float) -> int:
    if every_seconds <= 0:
        raise ValueError("--every-seconds must be greater than zero")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("The video does not report a valid frame rate")
    stride = max(1, round(fps * every_seconds))
    destination.mkdir(parents=True, exist_ok=True)
    frame_index = count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                target = destination / f"{video.stem}_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise OSError(f"Cannot save frame: {target}")
                count += 1
            frame_index += 1
    finally:
        cap.release()
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/test.mp4"))
    parser.add_argument("--output", type=Path, default=Path("data/frames_to_label"))
    parser.add_argument("--every-seconds", type=float, default=1.0)
    args = parser.parse_args()
    print(f"Frames extracted: {extract_frames(args.video, args.output, args.every_seconds)}")
