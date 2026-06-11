# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

**片刻 (Pianke)** 是一款本地运行的照片擂台式选片工具。它把一次拍摄中相似的连拍自动归入「同一个瞬间」的组，再通过左右 A/B PK 让用户快速挑出最满意的一张。后端 Flask + 原生 JS 前端，服务只监听 `127.0.0.1`，照片不出本地（除「土豪模式」外）。

## 常用命令

工具链版本与所有命令统一由 **mise** 管理（`mise.toml`），底层依赖走 **uv**。一律用 `mise run <task>`，不要直接调底层命令。`[tools]` 钉死了 Python 3.11.15 + uv 0.11.20；依赖在 `pyproject.toml` 声明、`uv.lock` 锁定。

```bash
mise install              # 装锁定版本的 Python + uv（首次）
mise run install          # uv sync：按 uv.lock 把全部依赖（含 dev/pytest）同步到 .venv
mise run install-gpu-win  # 仅 Windows+NVIDIA：在 install 后装 cu128 版 torch（覆盖 CPU 版）
mise run run              # 启动服务（默认端口 5057，自动开浏览器）
mise run run -- --port 8080 --no-browser   # 透传参数给 app.py（--runtime auto|cpu|cuda|mps）
mise run test             # 全部测试
mise run test -- tests/test_quality.py     # 单文件
mise run test -- tests/test_app_runtime.py::test_app_main_sets_runtime_env   # 单用例
mise run diagnose -- <folder> [--debug] [--limit N]   # 分组诊断，不启 UI
mise run compress         # 压缩指定目录 JPG（参数在脚本顶部常量）
```

测试通过 `tests/conftest.py` 把仓库根目录加进 `sys.path`，所以以 `import app`、`from pic_selecter import ...` 的方式导入。注意 `test_vision_gpu.py` 需要 `torch`（已是主依赖）。

> 非开发者的一键启动走 `启动_macOS.command` / `启动_Windows.bat`（背后是 `scripts/launcher.py`，含版本检查与**按模式**用 pip 装依赖）。它有自己一套硬编码的包列表（`CORE_PACKAGES` / `VISION_*` 等），**不读 `pyproject.toml`**，与上面的 mise/uv 开发流程相互独立——改依赖时两边都要顾及。

### 依赖与已知陷阱

- **依赖来源**：全在 `pyproject.toml` 的 `[project.dependencies]`；改完跑 `mise run install`（即 `uv sync`）重新锁定/同步。
- **OpenCV 三包冲突（已根治）**：`insightface` / `pyiqa` 会把 `opencv-python(-headless)` 当传递依赖拉进来，与本项目要的 `opencv-contrib-python` 共存会覆盖 `cv2.saliency`、expert 模式跑不起来。已用 `pyproject.toml` 里 `[tool.uv] override-dependencies` 的「marker 永假」写法把这两个包从依赖树彻底剔除——**解析阶段就消失，无需任何事后 uninstall/reinstall**。别再把它们加回依赖。
- **Windows + NVIDIA**：`pyproject.toml` 里 `torch/torchvision` 走 PyPI（默认 CPU 版），避免污染通用 lock；GPU 版由 `mise run install-gpu-win` 单独从 `download.pytorch.org/whl/cu128` 装。
- **Python 上界**：`requires-python = ">=3.11,<3.14"`，上界对齐 PyTorch wheel 生态（暂未覆盖 3.14）。

## 架构总览

### 三种引擎（engine / mode）

代码里用 `engine` 字段贯穿全流程，取值 `fast` / `expert` / `tycoon`（对应 README 的极速 / 专家 / 土豪）。三者只在「特征提取 + 初筛 + 分组」三步上分流，PK 擂台与归档逻辑共用：

| engine | 分组 | 初筛废片 | 联网 | 关键模块 |
| :-- | :-- | :-- | :-- | :-- |
| `fast` | 纯传统 CV（pHash+ORB+直方图） | numpy/cv2 锐度·曝光 | 否 | `fast_clustering.py`, `fast_quality.py` |
| `expert` | DINOv2 语义 + InsightFace 人脸 | 同上 + 人脸/美学模型 | 仅首次下权重 | `vision.py`, `clustering.py`, `quality.py` |
| `tycoon` | DINOv2 + InsightFace | 视觉 LLM（火山 Ark） | 每张图都联网 | `vision.py`, `clustering.py`, `llm_judge.py` |

`app.py::_require_engine()` 在任务启动时**硬校验**当前 engine 的全部依赖并预热模型——**设计原则是不静默降级**，依赖缺失或模型加载失败立刻抛异常（见 `vision.py` 文件头）。改特征/分组逻辑时要保持这条原则。

### 单进程任务模型

`app.py` 是单文件后端（~3700 行），核心是一套围绕全局会话的状态机：

- **`SessionState` / `GroupState` / `JobState`**（`app.py` 顶部 dataclass）：会话、每个相似组的 PK 进度、后台分析任务的进度。
- **后台任务**：`/api/start` 起一个线程跑 `_run_job()`，前端轮询 `/api/job` 看进度。`_run_job` 依次调 `grouper.compute_infos()`（按 engine 提特征）→ 初筛 → `cluster`/`fast_cluster` 分组 → 构建 PK 会话。
- **PK 状态机**：`advance()` / `kick_side()` / `reopen_group()` 驱动两两对决；`/api/choose`、`/api/kick`、`/api/undo`、`/api/skip_group` 是其 HTTP 入口。
- **持久化**：进度写到照片目录下的 `.pic_selecter_state.json`，`load_state()` + `_migrate_state()` 支持中断后续做、跨版本迁移。

### 输出目录约定（单文件夹单会话）

所有产物都落在**用户指定的照片目录**下，不污染系统：`winners/`、`losers/`、`.pic_selecter_state.json`、`_pic_selecter/`（内含 `thumbs/` 缩略图缓存、`log.txt`、`skipped.log`）。归档有「移动」（默认）/「复制」两种模式，反悔时移动模式会把文件无损搬回原位。

### RAW 处理

`grouper.py` 区分 `IMAGE_EXTS`（PIL 直接解码）和 `RAW_EXTS`（靠 `rawpy` 提取内嵌 JPEG 预览，毫秒级，不做 demosaic）。RAW+JPG 同名配对时优先用 JPG 分析，归档时 RAW 及 `.xmp` 伴随文件跟着一起搬（见 `_transfer_main_with_companions`）。分析统一缩到长边 `ANALYSIS_MAX_SIDE=2048`（避免大图喂 CLIP 时 MPS OOM）。

### 安全模型

- `app.py::_security_check()`（`@app.before_request`）做严格来源校验，防外部网站偷调本地接口。
- 设 `PIC_SELECTER_TOKEN` 可加额外鉴权（`SCRIPT_TOKEN`）。
- 土豪模式的 `ARK_API_KEY` 仅本地存于 `~/.config/pic_selecter/ark_key`，可由网页录入（`/api/ark_key`）或环境变量注入。

### 前端

`static/` 下是原生三件套：`index.html` + `app.js`（~2800 行单文件）+ `style.css`。无构建步骤，Flask 直接 `send_from_directory`。

### 相机水印

独立子系统：`watermark.py` 实现 11 个 EXIF 信息水印样式，自动匹配相机品牌 Logo（`assets/` 下）。HTTP 入口 `/api/watermark/*`，输出到 `winners/watermarked_<时间戳>/`。

## 关键环境变量

- `PIC_SELECTER_PORT` / `--port`：服务端口（默认 5057）。
- `PIC_SELECTER_RUNTIME` / `--runtime`：`auto|cpu|cuda|mps`，影响 torch / onnxruntime 设备选择（`vision.py`）。
- `PIC_SELECTER_TOKEN`：开启接口额外鉴权。
- `ARK_API_KEY`：土豪模式 LLM Key。
- `PIANKE_NO_MIRROR=1`：禁用 PyPI / 模型国内镜像，走官方源（启动器读取）。

## 约定

- 依赖在 `pyproject.toml` 声明、`uv.lock` 锁定；改动后 `mise run install`（`uv sync`）同步并更新锁。
- `[tools]` 里工具版本必须钉死具体版本号（禁止 `latest`），保证可复现。
- 新增工具链 / 命令一律沉淀到 `mise.toml` 的 `[tools]` / `[tasks.*]`，不要散落到零散脚本或 README。
