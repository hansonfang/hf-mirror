#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PORT="${PORT:-8080}"

cd "${ROOT_DIR}"

if command -v uv >/dev/null 2>&1; then
  uv venv "${VENV_DIR}" --python python3
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  uv pip install -r requirements.txt
else
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  pip install -q -r requirements.txt
fi

mkdir -p "${ROOT_DIR}/data" "${ROOT_DIR}/.cache/huggingface"

if lsof -i ":${PORT}" >/dev/null 2>&1; then
  PORT=8081
  echo "端口 8080 已被占用，改用 ${PORT}"
fi

export PYTHONPATH="${ROOT_DIR}"
export HF_WORK_DIR="${ROOT_DIR}/data"
export HF_CACHE_DIR="${ROOT_DIR}/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

echo "本地调试: http://127.0.0.1:${PORT}"
echo "输出目录: ${HF_WORK_DIR}"
echo "缓存目录: ${HF_CACHE_DIR}"

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --reload
