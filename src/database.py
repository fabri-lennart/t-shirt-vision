import sqlite3
from pathlib import Path

from config import DB_PATH


def init_db(db_path: Path = DB_PATH) -> int:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS t_shirt_counts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
            "line_name TEXT NOT NULL)"
        )
        return connection.execute("SELECT COUNT(*) FROM t_shirt_counts").fetchone()[0]


def save_count(line_name: str, db_path: Path = DB_PATH) -> int:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO t_shirt_counts (line_name) VALUES (?)", (line_name,)
        )
        return connection.execute("SELECT COUNT(*) FROM t_shirt_counts").fetchone()[0]
