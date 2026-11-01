"""Import the Roboflow ZIP, convert YOLO polygons to boxes, and split by time."""

import argparse
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path.home() / "Desktop" / "t_shirt_counts.v1-tshirts-version.yolov8.zip"
DATASET = ROOT / "data" / "dataset"


def box_from_polygon(line: str) -> str:
    values = [float(item) for item in line.split()]
    if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0:
        raise ValueError(f"Invalid polygon: {line[:100]}")
    coordinates = values[1:]
    if any(not 0 <= value <= 1 for value in coordinates):
        raise ValueError(f"Coordinate outside [0,1]: {line[:100]}")
    xs, ys = coordinates[::2], coordinates[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if left == right or top == bottom:
        raise ValueError(f"Degenerate polygon: {line[:100]}")
    return f"0 {(left + right) / 2:.8f} {(top + bottom) / 2:.8f} {right - left:.8f} {bottom - top:.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if not args.zip.is_file():
        raise FileNotFoundError(args.zip)
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        images = [name for name in names if name.startswith("train/images/") and name.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(images) < 2:
            raise ValueError("At least two training images are required")
        # File names contain frame indices. Reserve the final video segment for validation.
        images.sort(key=lambda name: int(re.search(r"test_(\d+)", Path(name).name).group(1)))
        split_at = max(1, round(len(images) * 0.8))
        counts = {"train": 0, "val": 0}
        boxes = {"train": 0, "val": 0}
        for position, name in enumerate(images):
            split = "train" if position < split_at else "val"
            image_name = Path(name).name
            label_name = f"train/labels/{Path(image_name).stem}.txt"
            if label_name not in names:
                raise FileNotFoundError(f"Label missing from ZIP: {label_name}")
            image_dir = DATASET / "images" / split
            label_dir = DATASET / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / image_name).write_bytes(archive.read(name))
            lines = archive.read(label_name).decode("utf-8").splitlines()
            converted = [box_from_polygon(line) for line in lines if line.strip()]
            (label_dir / f"{Path(image_name).stem}.txt").write_text(
                "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
            )
            counts[split] += 1
            boxes[split] += len(converted)
    (DATASET / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnc: 1\nnames:\n  0: polera\n",
        encoding="utf-8",
    )
    print(f"Images: {counts}; converted boxes: {boxes}; dataset: {DATASET}")


if __name__ == "__main__":
    main()
