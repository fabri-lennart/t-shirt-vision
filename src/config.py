from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SOURCE = PROJECT_ROOT / "data" / "test.mp4"
DB_PATH = PROJECT_ROOT / "output" / "production.db"
MODEL_PATH = PROJECT_ROOT / "output" / "training" / "polera" / "weights" / "best.pt"
LINE_Y = 350
