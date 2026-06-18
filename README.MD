
**HF CLI Docker**

这个仓库提供一个 Hugging Face 下载 Web 界面镜像，文件是 [`Dockerfile.huggingface-cli`](Dockerfile.huggingface-cli)。

容器启动后会常驻运行 Web 服务。HF Endpoint 可在页面配置，默认 `https://huggingface.co`，也可改为 `https://hf-mirror.com` 等镜像地址。

## 容器配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 端口 | `8080` | 映射到宿主机后浏览器访问 |
| 输出目录 | 宿主机目录 → `/work` | 下载的模型/数据集保存在这里 |
| 缓存目录 | 宿主机目录 → `/root/.cache/huggingface` | 断点续传缓存，建议挂载 |
| 启动命令 | 留空 | 镜像默认启动 Web 服务 |

浏览器打开：`http://<宿主机IP>:8080`

## 本地调试（无需构建镜像）

适合改页面或接口时快速验证，支持热重载：

```bash
chmod +x scripts/dev.sh
./scripts/dev.sh
```

浏览器打开：`http://127.0.0.1:8080`

| 环境变量 | 本地默认值 | 说明 |
|----------|------------|------|
| `HF_WORK_DIR` | `./data` | 下载输出与 Endpoint 配置 |
| `HF_CACHE_DIR` | `./.cache/huggingface` | HF 缓存 |

手动启动（等价于上述脚本）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data .cache/huggingface

export PYTHONPATH="$PWD"
export HF_WORK_DIR="$PWD/data"
export HF_CACHE_DIR="$PWD/.cache/huggingface"
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## 本地构建（Docker）

```bash
docker build -f Dockerfile.huggingface-cli -t hf-cli .
docker run --rm -p 8080:8080 \
  -v "$PWD/data:/work" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  hf-cli
```

## 导出可导入的镜像 tar（ARM 设备）

部分平台只认 `docker save` 格式，且目标设备为 **linux/arm64**。在 macOS 上可用 Apple 的 `container` CLI 构建，再用 `skopeo` 转成 docker 格式 tar：

```bash
# 首次使用需创建 container machine
container machine create alpine:3.22 --name hf-builder --set-default --cpus 4 --memory 4G

# 构建 arm64 镜像
container build -f Dockerfile.huggingface-cli -t hf-cli:arm64 --platform linux/arm64 .

# 导出 OCI tar
container image save --platform linux/arm64 -o hf-cli-arm64-oci.tar hf-cli:arm64

# 转为 docker save 格式（平台导入用）
skopeo copy --override-os linux --override-arch arm64 \
  oci-archive:hf-cli-arm64-oci.tar \
  docker-archive:hf-cli-arm64-docker.tar:hf-cli:latest
```

产物 `hf-cli-arm64-docker.tar` 可在目标环境执行：

```bash
docker load -i hf-cli-arm64-docker.tar
```

若目标设备为 x86_64，将上述命令中的 `arm64` 改为 `amd64`，输出文件名改为 `hf-cli-amd64-docker.tar` 即可。

## Web 页面使用

1. 导入镜像并配置端口 `8080`、挂载 `/work` 与缓存目录
2. 启动容器，**启动命令留空**
3. 浏览器打开 Web 页面，先配置 **HF Endpoint** 并保存
4. 填写仓库 ID，可选填写文件路径，点击「开始下载」
5. 文件会保存到 `/work/<仓库ID>/`（`/` 会替换为 `__`）

Endpoint 配置保存在 `/work/.config/settings.json`，任务记录保存在 `/work/.config/hf.db`（SQLite），挂载 `/work` 后重启仍有效。

示例：

| 场景 | 仓库 ID | 文件路径 | 保存位置 |
|------|---------|----------|----------|
| 整仓下载 | `gpt2` | 留空 | `/work/gpt2/` |
| 单文件下载 | `unsloth/gemma-4-31B-it-GGUF` | `q4_k_s.gguf` | `/work/unsloth__gemma-4-31B-it-GGUF/q4_k_s.gguf` |

## 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `401 Unauthorized` + `cas-server.xethub.hf.co` | 大文件走 XET 协议，镜像站无法鉴权 | 镜像已默认 `HF_HUB_DISABLE_XET=1`，需**重新构建/导入镜像** |
| `308 Redirect` + `hf-mirror.com` | hub 库对 308 处理不完整 | 换 `huggingface.co` 或 `hf-cdn.sufy.com` |
| 容器 `exit 137` / OOM | 内存不足 | 容器内存 ≥ **2GB**（大 GGUF 建议 4GB） |
| 下载中途 timeout / SSL EOF / 突然卡死 | 镜像 CDN 慢或连接静默中断 | 已默认 `HF_HUB_DOWNLOAD_TIMEOUT=60` 并自动重试；仍失败可换 endpoint 或重点下载 |
| 进度 100% 但一直「下载中」 | 字节已传完，hub 仍在 cas-bridge 续传/校验；或同时下多个大 GGUF 争抢带宽 | 页面会显示「续传/校验中」；大文件建议**一次只下一个**；可换 `huggingface.co` endpoint |
