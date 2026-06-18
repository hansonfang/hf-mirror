import logging
import os
import time

# 须在 import huggingface_hub 之前设置
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 大文件经镜像 CDN 下载时，默认 10s 读超时过短易触发断点续传失败
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import logging as hf_logging
from huggingface_hub.utils import tqdm as hf_tqdm
from pydantic import BaseModel, Field

from app.storage import Storage

WORK_DIR = Path(os.environ.get("HF_WORK_DIR", "/work"))
CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/root/.cache/huggingface"))

logger = logging.getLogger("hf-downloader")
hf_logging.set_verbosity_info()

app = FastAPI(title="HF Downloader")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()
_progress_state: dict[str, dict[str, Any]] = {}
_settings_lock = threading.Lock()
_storage: Storage | None = None
_cancelled_tasks: set[str] = set()
_cancel_lock = threading.Lock()

T = TypeVar("T")
_DOWNLOAD_RETRY_ATTEMPTS = 5
_TRANSIENT_DOWNLOAD_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    OSError,
)


class DownloadCancelled(Exception):
    """用户主动停止下载"""


def _request_cancel(task_id: str) -> None:
    with _cancel_lock:
        _cancelled_tasks.add(task_id)


def _is_cancelled(task_id: str) -> bool:
    with _cancel_lock:
        return task_id in _cancelled_tasks


def _clear_cancel(task_id: str) -> None:
    with _cancel_lock:
        _cancelled_tasks.discard(task_id)


def _ensure_not_cancelled(task_id: str) -> None:
    if _is_cancelled(task_id):
        raise DownloadCancelled()


def _call_with_download_retry(task_id: str, action: Callable[[], T]) -> T:
    """网络抖动时重试整次下载调用，hub 会从缓存断点续传。"""
    for attempt in range(1, _DOWNLOAD_RETRY_ATTEMPTS + 1):
        try:
            _ensure_not_cancelled(task_id)
            return action()
        except DownloadCancelled:
            raise
        except _TRANSIENT_DOWNLOAD_ERRORS as exc:
            if attempt >= _DOWNLOAD_RETRY_ATTEMPTS:
                raise
            wait_s = min(30, 2**attempt)
            logger.warning(
                "[%s] 下载中断 (%s)，%ds 后重试 (%d/%d)",
                task_id,
                exc,
                wait_s,
                attempt,
                _DOWNLOAD_RETRY_ATTEMPTS,
            )
            _set_task(
                task_id,
                message=f"网络中断，{wait_s}s 后重试 ({attempt}/{_DOWNLOAD_RETRY_ATTEMPTS})…",
            )
            time.sleep(wait_s)
    raise RuntimeError("unreachable")


class _QuietTasksAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/tasks" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietTasksAccessFilter())


class RepoType(str, Enum):
    model = "model"
    dataset = "dataset"


class SettingsUpdate(BaseModel):
    hf_endpoint: str = Field(min_length=1, examples=["https://huggingface.co", "https://hf-mirror.com"])


class DownloadRequest(BaseModel):
    repo_id: str = Field(min_length=1, examples=["gpt2", "unsloth/gemma-4-31B-it-GGUF"])
    repo_type: RepoType = RepoType.model
    filename: str | None = Field(
        default=None,
        examples=["q4_k_s.gguf"],
        description="可选，指定仓库内单个文件的相对路径",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_endpoint(raw: str) -> str:
    value = raw.strip().rstrip("/")
    if not value:
        raise HTTPException(status_code=400, detail="HF Endpoint 不能为空")

    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="HF Endpoint 格式不正确")

    return value


def get_hf_endpoint() -> str:
    assert _storage is not None
    with _settings_lock:
        return _storage.get_hf_endpoint()


def save_hf_endpoint(raw: str) -> dict[str, Any]:
    assert _storage is not None
    endpoint = _normalize_endpoint(raw)
    result = _storage.set_hf_endpoint(endpoint)
    logger.info("HF Endpoint 已更新为 %s", endpoint)
    return result


def _format_bytes(num: int | None) -> str:
    if num is None:
        return "未知"
    if num <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{num} B"


def _repo_subdir(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__")


def _validate_relative_path(path: str) -> None:
    parts = path.strip().split("/")
    if not path.strip() or path.startswith("/") or ".." in parts:
        raise HTTPException(status_code=400, detail="文件路径不合法")


def _local_dir_for_repo(repo_id: str) -> Path:
    return WORK_DIR / _repo_subdir(repo_id)


def _set_task(task_id: str, **fields: Any) -> None:
    should_persist = "status" in fields
    with _tasks_lock:
        _tasks[task_id].update(fields)
        snapshot = dict(_tasks[task_id]) if should_persist else None
    if should_persist and snapshot is not None:
        assert _storage is not None
        _storage.upsert_task(snapshot)


def _estimate_total_bytes(
    repo_id: str,
    repo_type: RepoType,
    filename: str | None,
    endpoint: str,
) -> int | None:
    api = HfApi(endpoint=endpoint)
    try:
        if filename:
            info = api.get_paths_info(repo_id, [filename], repo_type=repo_type.value)
            return sum(item.size or 0 for item in info)

        paths = api.list_repo_files(repo_id, repo_type=repo_type.value)
        if not paths:
            return None
        info = api.get_paths_info(repo_id, paths, repo_type=repo_type.value)
        return sum(item.size or 0 for item in info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法预估下载大小 %s: %s", repo_id, exc)
        return None


def _push_progress(task_id: str) -> None:
    _ensure_not_cancelled(task_id)
    state = _progress_state.get(task_id)
    if not state:
        return

    completed = int(state["completed_bytes"])
    current = int(state.get("current_bytes", 0))
    downloaded = completed + current
    total = state.get("total_bytes")
    current_file = state.get("current_file") or ""

    if total and total > 0:
        progress = min(100, int(downloaded / total * 100))
        message = (
            f"{current_file} · {progress}% "
            f"({_format_bytes(downloaded)} / {_format_bytes(total)})"
        ).strip()
    elif current_file:
        message = f"{current_file} · {_format_bytes(downloaded)}"
        progress = 0
    else:
        message = "正在下载…"
        progress = 0

    _set_task(
        task_id,
        progress=progress,
        downloaded_bytes=downloaded,
        total_bytes=total,
        current_file=current_file,
        message=message,
    )


def _bar_label(bar: Any) -> str:
    return str(getattr(bar, "desc", "") or getattr(bar, "name", "") or "")


def _create_progress_tqdm(task_id: str) -> type:
    state = _progress_state[task_id]
    base_tqdm = hf_tqdm

    class TaskProgressBar(base_tqdm):
        def _sync_progress(self) -> None:
            # snapshot_download 通过 refresh() 累加 total，通过 update() 累加 n
            state["current_bytes"] = int(self.n)
            bar_total = int(self.total or 0)
            if bar_total > 0:
                state["total_bytes"] = bar_total
            label = _bar_label(self)
            if label:
                state["current_file"] = label
            _push_progress(task_id)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._sync_progress()

        def update(self, n: float = 1) -> bool | None:
            result = super().update(n)
            self._sync_progress()
            return result

        def refresh(self, nolock: bool = False, lock_args: Any = None) -> None:
            super().refresh(nolock=nolock, lock_args=lock_args)
            self._sync_progress()

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            super().set_description(desc, refresh=refresh)
            if desc:
                state["current_file"] = str(desc)
            self._sync_progress()

        def close(self) -> None:
            try:
                super().close()
            finally:
                state["completed_bytes"] = int(state["completed_bytes"]) + int(self.n)
                state["current_bytes"] = 0
                _push_progress(task_id)

    return TaskProgressBar


def _run_download(
    task_id: str,
    repo_id: str,
    repo_type: RepoType,
    filename: str | None,
    endpoint: str,
) -> None:
    target_dir = _local_dir_for_repo(repo_id)
    mode = f"单文件 {filename}" if filename else "整仓"
    logger.info("[%s] 开始下载 %s (%s) via %s -> %s", task_id, repo_id, mode, endpoint, target_dir)

    try:
        _ensure_not_cancelled(task_id)
    except DownloadCancelled:
        _set_task(task_id, status="cancelled", message="已停止", finished_at=_now_iso())
        _clear_cancel(task_id)
        return

    total_bytes = _estimate_total_bytes(repo_id, repo_type, filename, endpoint)
    _progress_state[task_id] = {
        "completed_bytes": 0,
        "current_bytes": 0,
        "current_file": "",
        "total_bytes": total_bytes,
    }
    _set_task(
        task_id,
        status="running",
        progress=0,
        downloaded_bytes=0,
        total_bytes=total_bytes,
        current_file="",
        message="正在准备下载…",
        started_at=_now_iso(),
    )

    tqdm_class = _create_progress_tqdm(task_id)

    try:
        _ensure_not_cancelled(task_id)
        if filename:
            saved = _call_with_download_retry(
                task_id,
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type.value,
                    local_dir=str(target_dir),
                    cache_dir=str(CACHE_DIR),
                    endpoint=endpoint,
                    tqdm_class=tqdm_class,
                ),
            )
            local_path = saved
            message = f"下载完成：{saved}"
        else:
            _call_with_download_retry(
                task_id,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type.value,
                    local_dir=str(target_dir),
                    cache_dir=str(CACHE_DIR),
                    endpoint=endpoint,
                    tqdm_class=tqdm_class,
                    max_workers=1,
                ),
            )
            local_path = str(target_dir)
            message = f"下载完成：{target_dir}"

        if _is_cancelled(task_id):
            return

        logger.info("[%s] 下载完成 %s", task_id, local_path)
        _set_task(
            task_id,
            status="done",
            progress=100,
            message=message,
            finished_at=_now_iso(),
            local_path=local_path,
        )
    except DownloadCancelled:
        logger.info("[%s] 下载已停止 %s", task_id, repo_id)
        _set_task(
            task_id,
            status="cancelled",
            message="已停止",
            finished_at=_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001 - 展示给用户
        if _is_cancelled(task_id):
            _set_task(
                task_id,
                status="cancelled",
                message="已停止",
                finished_at=_now_iso(),
            )
            return
        logger.exception("[%s] 下载失败 %s: %s", task_id, repo_id, exc)
        _set_task(
            task_id,
            status="failed",
            message=str(exc),
            finished_at=_now_iso(),
        )
    finally:
        _clear_cancel(task_id)
        _progress_state.pop(task_id, None)


@app.on_event("startup")
def _configure_logging() -> None:
    global _storage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    _storage = Storage(WORK_DIR)
    _storage.load_tasks_into(_tasks)
    logger.info("HF Endpoint=%s, 输出目录=%s", get_hf_endpoint(), WORK_DIR)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "endpoint": get_hf_endpoint(),
        "download_dir": str(WORK_DIR),
    }


@app.get("/api/settings")
def read_settings() -> dict[str, Any]:
    assert _storage is not None
    return _storage.get_settings()


@app.put("/api/settings")
def update_settings(body: SettingsUpdate) -> dict[str, Any]:
    with _settings_lock:
        return save_hf_endpoint(body.hf_endpoint)


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    with _tasks_lock:
        items = list(_tasks.values())
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str) -> dict[str, Any]:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task["status"] not in {"pending", "running"}:
            raise HTTPException(status_code=400, detail="任务无法停止")

    _request_cancel(task_id)
    _set_task(
        task_id,
        status="cancelled",
        message="正在停止…",
        finished_at=_now_iso(),
    )
    with _tasks_lock:
        return dict(_tasks[task_id])


@app.post("/api/download")
def start_download(body: DownloadRequest) -> dict[str, Any]:
    repo_id = body.repo_id.strip()
    filename = body.filename.strip() if body.filename else None
    if filename:
        _validate_relative_path(filename)

    endpoint = get_hf_endpoint()
    target_dir = _local_dir_for_repo(repo_id)
    local_path = str(target_dir / filename) if filename else str(target_dir)

    task_id = uuid.uuid4().hex[:12]
    task = {
        "id": task_id,
        "repo_id": repo_id,
        "repo_type": body.repo_type.value,
        "filename": filename,
        "hf_endpoint": endpoint,
        "local_path": local_path,
        "status": "pending",
        "progress": 0,
        "downloaded_bytes": 0,
        "total_bytes": None,
        "current_file": "",
        "message": "等待开始",
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
    }

    with _tasks_lock:
        _tasks[task_id] = task
    assert _storage is not None
    _storage.upsert_task(task)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "创建下载任务 %s: repo=%s filename=%s endpoint=%s",
        task_id,
        repo_id,
        filename or "(整仓)",
        endpoint,
    )

    thread = threading.Thread(
        target=_run_download,
        args=(task_id, repo_id, body.repo_type, filename, endpoint),
        daemon=True,
    )
    thread.start()

    return task
