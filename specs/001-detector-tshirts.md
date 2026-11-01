# Specification 001: CUDA T-shirt detection and line counting

## Goal

Train a one-class YOLOv8 detector on annotated conveyor footage using the RTX 4050, then count each tracked T-shirt when it crosses the white line.

## Acceptance criteria

1. `uv sync --locked` installs CUDA-enabled PyTorch on Windows with Python 3.12. `torch.cuda.is_available()` returns `True` on the RTX 4050.
2. `import_roboflow.py` reads the supplied ZIP, converts YOLO polygons to detection boxes, and splits frames by time into `data/dataset/{images,labels}/{train,val}`.
3. `train.py --check-only` rejects missing files, wrong classes, and malformed boxes. Training writes `output/training/polera/weights/best.pt` and runs validation on the GPU.
4. `main.py` counts only tracked detections crossing the white line at `LINE_Y`. A tracker ID contributes at most once per run. The visible run count and Loguru messages increase with every crossing; the SQLite lifetime total is shown separately.
5. The README is in English and provides a preview and a link to the full sample video. Source code and logs contain no emoji or Spanish user-facing text.

## Implementation notes

- The supplied Roboflow ZIP contained 35 images and polygon labels. The import converted 328 polygons to boxes, using 28 images for training and the last 7 for validation. The source ZIP remains unchanged.
- The 100-epoch run on October 31, 2026 produced precision 1.000, recall 0.979, mAP50 0.986, and mAP50-95 0.901 on the 7 reserved frames. All frames come from the same video, so these numbers do not establish accuracy on another recording.
- The white line is horizontal at source-frame coordinate `y=350`. Its endpoints are oriented so the conveyor's motion is reported as an inbound crossing by Supervision. The application draws the line itself instead of displaying Supervision's `in`/`out` labels.
- `output/production.db` accumulates events across runs. The run count starts at zero each time. A separate database can be selected with `--db` for a smoke test.
- The 6 GiB GPU can use `batch=8` at `imgsz=640` for this dataset. Lower the batch size if the dataset or model grows.

## Remaining validation

- [ ] Manually count crossings in an independent video and compare with the program.
- [ ] Test with new lighting, garments, camera positions, and conveyor speeds.
