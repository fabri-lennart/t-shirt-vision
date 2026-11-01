# Agent instructions

- Read `specs/001-detector-tshirts.md` before changing dataset preparation, training, or counting. Update the specification and acceptance criteria when scope changes.
- Keep `uv.lock` synchronized with `pyproject.toml` using `uv lock`, then verify with `uv sync --locked`. Do not install project packages with `uv pip install` because `uv run` may replace them from the lock file.
- On Windows, retain `torch==2.5.1+cu121` and `torchvision==0.20.1+cu121` from the explicit PyTorch index. Check `torch.cuda.is_available()` before training or inference.
- Annotated images, databases, and trained weights remain local. The user explicitly approved publishing `data/test.mp4` with the README; do not publish other data or weights without explicit direction.
- Count T-shirts only when a tracked detection crosses the configured white line. Show a per-run total separately from the cumulative SQLite total. Use `output/training/polera/weights/best.pt`, never the base COCO detector.
- Validation frames from the same video do not establish generalization. Record metrics and check false positives, false negatives, and repeated crossings before production use.
