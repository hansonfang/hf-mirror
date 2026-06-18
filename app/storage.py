import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("hf-downloader")

DEFAULT_ENDPOINT = "https://huggingface.co"
MAX_ENDPOINT_HISTORY = 8

_TASK_COLUMNS = (
    "id",
    "repo_id",
    "repo_type",
    "filename",
    "hf_endpoint",
    "local_path",
    "status",
    "progress",
    "downloaded_bytes",
    "total_bytes",
    "current_file",
    "message",
    "created_at",
    "started_at",
    "finished_at",
)


class Storage:
    def __init__(self, work_dir: Path) -> None:
        config_dir = work_dir / ".config"
        self.config_path = config_dir / "settings.json"
        self.db_path = config_dir / "hf.db"
        self._settings_lock = threading.Lock()
        self._tasks_lock = threading.Lock()
        config_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_legacy_tasks_json()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    repo_type TEXT NOT NULL,
                    filename TEXT,
                    hf_endpoint TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    downloaded_bytes INTEGER NOT NULL DEFAULT 0,
                    total_bytes INTEGER,
                    current_file TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);
                """
            )

    def _read_settings_raw(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"hf_endpoint": DEFAULT_ENDPOINT, "endpoint_history": []}

        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("settings.json 损坏，使用默认配置: %s", exc)
            return {"hf_endpoint": DEFAULT_ENDPOINT, "endpoint_history": []}

        if not isinstance(data, dict):
            return {"hf_endpoint": DEFAULT_ENDPOINT, "endpoint_history": []}
        return data

    @staticmethod
    def _normalize_history(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        seen: set[str] = set()
        history: list[str] = []
        for item in raw:
            value = str(item).strip()
            if value and value not in seen:
                seen.add(value)
                history.append(value)
        return history[:MAX_ENDPOINT_HISTORY]

    def _history_from_tasks(self) -> list[str]:
        seen: set[str] = set()
        history: list[str] = []
        for task in self.list_tasks():
            endpoint = str(task.get("hf_endpoint", "")).strip()
            if endpoint and endpoint not in seen:
                seen.add(endpoint)
                history.append(endpoint)
        return history[:MAX_ENDPOINT_HISTORY]

    def _write_settings(self, data: dict[str, Any]) -> None:
        payload = {
            "hf_endpoint": str(data.get("hf_endpoint", DEFAULT_ENDPOINT)).strip() or DEFAULT_ENDPOINT,
            "endpoint_history": self._normalize_history(data.get("endpoint_history")),
        }
        self.config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_settings(self) -> dict[str, Any]:
        with self._settings_lock:
            data = self._read_settings_raw()
            endpoint = str(data.get("hf_endpoint", DEFAULT_ENDPOINT)).strip() or DEFAULT_ENDPOINT
            history = self._normalize_history(data.get("endpoint_history"))
            if not history:
                history = self._history_from_tasks()
                if history:
                    data["hf_endpoint"] = endpoint
                    data["endpoint_history"] = history
                    self._write_settings(data)
            if endpoint not in history:
                history = [endpoint, *[item for item in history if item != endpoint]]
                history = history[:MAX_ENDPOINT_HISTORY]
            return {"hf_endpoint": endpoint, "endpoint_history": history}

    def get_hf_endpoint(self) -> str:
        return self.get_settings()["hf_endpoint"]

    def set_hf_endpoint(self, endpoint: str) -> dict[str, Any]:
        with self._settings_lock:
            data = self._read_settings_raw()
            history = self._normalize_history(data.get("endpoint_history"))
            history = [endpoint, *[item for item in history if item != endpoint]]
            history = history[:MAX_ENDPOINT_HISTORY]
            data["hf_endpoint"] = endpoint
            data["endpoint_history"] = history
            self._write_settings(data)
            return {"hf_endpoint": endpoint, "endpoint_history": history}

    def _migrate_legacy_tasks_json(self) -> None:
        tasks_json = self.config_path.parent / "tasks.json"
        if not tasks_json.exists():
            return

        with self._tasks_lock, self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if count > 0:
            return

        try:
            items = json.loads(tasks_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("tasks.json 损坏，跳过迁移: %s", exc)
            return

        if not isinstance(items, list):
            return

        for item in items:
            if isinstance(item, dict) and item.get("id"):
                self.upsert_task(item)
        logger.info("已从 tasks.json 迁移 %d 条任务", len(items))

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._tasks_lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC",
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._tasks_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        return self._row_to_task(row) if row else None

    def upsert_task(self, task: dict[str, Any]) -> None:
        values = {col: task.get(col) for col in _TASK_COLUMNS}
        placeholders = ", ".join("?" for _ in _TASK_COLUMNS)
        columns = ", ".join(_TASK_COLUMNS)
        updates = ", ".join(f"{col} = excluded.{col}" for col in _TASK_COLUMNS if col != "id")

        with self._tasks_lock, self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO tasks ({columns}) VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {updates}
                """,
                tuple(values[col] for col in _TASK_COLUMNS),
            )

    def load_tasks_into(self, target: dict[str, dict[str, Any]]) -> None:
        for task in self.list_tasks():
            target[task["id"]] = task
