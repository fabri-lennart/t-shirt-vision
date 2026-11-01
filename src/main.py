"""Count tracked T-shirts when they cross the white line in the video."""

import argparse
import sys
from pathlib import Path

import cv2
from loguru import logger
import supervision as sv
import torch
from trackers import ByteTrackTracker
from ultralytics import YOLO

from config import DB_PATH, LINE_Y, MODEL_PATH, VIDEO_SOURCE
from counter import CrossingCounter
from database import init_db, save_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO_SOURCE)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--line-y", type=int, default=LINE_Y)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", colorize=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run uv sync --locked and check nvidia-smi.")
    if not args.model.is_file():
        raise FileNotFoundError(f"Trained model not found: {args.model}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not 0 <= args.line_y < height:
        capture.release()
        raise ValueError(f"--line-y must be between 0 and {height - 1}")

    stored_total = init_db(args.db)
    counter = CrossingCounter()
    model = YOLO(str(args.model))
    tracker = ByteTrackTracker()
    line_zone = sv.LineZone(
        start=sv.Point(x=width - 1, y=args.line_y),
        end=sv.Point(x=0, y=args.line_y),
    )
    box_annotator = sv.BoxAnnotator()

    logger.info("Video: {} | GPU: {}", args.video, torch.cuda.get_device_name(0))
    logger.info("Counting only tracked T-shirts crossing the white line at y={}", args.line_y)
    logger.info("Previous database total: {}", stored_total)

    if not args.headless:
        cv2.namedWindow("T-Shirt Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("T-Shirt Vision", 1000, 600)
        logger.info("Press q in the video window to stop")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model(frame, device=0, verbose=False)[0]
            detections = tracker.update(sv.Detections.from_ultralytics(result))
            crossed_in, _ = line_zone.trigger(detections=detections)
            if detections.tracker_id is not None:
                for index, crossed in enumerate(crossed_in):
                    if not crossed:
                        continue
                    tracker_id = int(detections.tracker_id[index])
                    if counter.record(tracker_id):
                        stored_total = save_count("Line_1", args.db)
                        logger.info(
                            "T-shirt crossed | tracker_id={} | this_run={} | database_total={}",
                            tracker_id, counter.total, stored_total,
                        )

            if not args.headless:
                frame = box_annotator.annotate(scene=frame, detections=detections)
                cv2.line(frame, (0, args.line_y), (width - 1, args.line_y), (255, 255, 255), 3)
                cv2.rectangle(frame, (10, 10), (390, 84), (0, 0, 0), -1)
                cv2.putText(frame, f"T-shirts crossed: {counter.total}", (20, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Database total: {stored_total}", (20, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("T-Shirt Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Stopped by user")
                    break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()

    logger.info("Finished | this_run={} | database_total={}", counter.total, stored_total)


if __name__ == "__main__":
    main()
