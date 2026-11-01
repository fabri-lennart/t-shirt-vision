"""Validate a YOLO dataset and train/evaluate a T-shirt detector on CUDA."""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def validate_dataset(dataset: Path) -> dict:
    dataset = dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    config = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("names") != {0: "polera"}:
        raise ValueError("data.yaml must define names: {0: polera}")
    if config.get("nc") not in (None, 1):
        raise ValueError("The dataset must contain exactly one class")
    root = dataset.parent
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"Images or labels missing in {split}")
        files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"No images found in {images}")
        for image in files:
            label = labels / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing label {label}; use an empty file for negative images")
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                try:
                    values = [float(value) for value in parts]
                except ValueError as exc:
                    raise ValueError(f"Invalid label: {label}:{line_number}") from exc
                if (len(values) != 5 or values[0] != 0
                        or any(not 0 <= value <= 1 for value in values[1:])
                        or values[3] <= 0 or values[4] <= 0
                        or values[1] - values[3] / 2 < -1e-6 or values[1] + values[3] / 2 > 1 + 1e-6
                        or values[2] - values[4] / 2 < -1e-6 or values[2] + values[4] / 2 > 1 + 1e-6):
                    raise ValueError(f"Invalid label: {label}:{line_number}")
    return {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "polera"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset/data.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    data = validate_dataset(args.data)
    if args.check_only:
        print("Dataset valid")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run uv sync --locked and check nvidia-smi")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("epochs and batch must be positive, and imgsz must be at least 32")
    normalized = Path("output/training/dataset_absolute.yaml")
    normalized.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    print(f"Training on {torch.cuda.get_device_name(0)}")
    model = YOLO("yolov8n.pt")
    model.train(data=str(normalized.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=0, project=str(Path("output/training").resolve()),
                name="polera", exist_ok=True)
    best = Path("output/training/polera/weights/best.pt")
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")
    YOLO(str(best)).val(data=str(normalized.resolve()), device=0, split="val")
    print(f"Model ready: {best}")


if __name__ == "__main__":
    main()
