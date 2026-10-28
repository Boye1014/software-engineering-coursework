
# Vision Lab：本地目标检测系统

这是一个面向软件工程课程展示的本地 Web 目标检测系统。项目使用 Ultralytics YOLO11m 预训练模型，在浏览器中完成图片检测和视频逐帧检测，不需要额外训练即可运行。

## 功能

- 图片上传、预览、目标检测和结果下载
- 视频上传、逐帧检测、实时处理进度和带框视频输出
- 界面内切换检测模型（COCO 预训练 80 类 / 自训 4 类），可选类别随模型自动变化
- 置信度阈值、NMS / IoU 阈值调节
- 12 个核心类别筛选：`person`、`bicycle`、`car`、`motorcycle`、`bus`、`truck`、`bird`、`cat`、`dog`、`bottle`、`laptop`、`cell phone`
- 类别列表单击选择，已选类别单独展示并可逐个移除
- 未手动选择类别时，默认检测上述 12 个核心类别
- 显示检测类别、置信度和边界框
- SQLite 历史记录，支持查看、下载和删除结果
- 自动选择 NVIDIA CUDA、Apple MPS 或 CPU
- 上传文件类型和大小校验

## 技术栈

- Python 3
- Flask
- Ultralytics YOLO11m
- OpenCV
- SQLite
- HTML / CSS / JavaScript

## 训练流程（YOLO11n + VOC2007）

系统默认用 COCO 预训练的 YOLO11m 做推理，开箱即可用。此外项目还提供一条完整的
本地训练链，用于演示"数据准备 → 训练 → 验证 → 部署"的全流程：

```bash
# 1) 构建数据集：下载 PASCAL VOC2007，过滤出 person/bicycle/car/bus 四类
python training/prepare_data.py

# 2) 训练 YOLO11n（100 epoch，RTX 4070 约 100 分钟）
python training/train.py

# 3) 启动系统，在界面「检测模型」下拉里切到自训权重即可
python app.py
```

也可以直接用环境变量指定启动模型：

```bash
YOLO_MODEL=models/traffic_yolo11n.pt python app.py
```

自训模型在 VOC2007 四类验证集上的结果：**mAP@0.5 = 0.857，mAP@0.5:0.95 = 0.653**，
权重仅 5.2MB，CPU 单帧 33ms。切到它之后系统类别筛选区会自动只显示这 4 个类别，
其余功能无需改动。

如果想覆盖界面全部 12 个核心类别，可以训练 12 类版本（COCO2017-val + VOC2007 合并，
7521 张图）：

```bash
python training/prepare_coco12.py
python training/train.py --data datasets/traffic12/data.yaml \
    --epochs 100 --batch 16 --patience 100 --name traffic12_yolo11n
```

结果 **mAP@0.5 = 0.5912，mAP@0.5:0.95 = 0.4312**，权重 5.3MB。同一批验证图上
与官方权重对比：官方 yolo11n 为 0.7103、官方 yolo11m 为 0.8093——同样的 n 架构、
同样的推理速度，自训版低 12 个点，差距来自训练数据量（11.8 万 vs 6800 张）。

数据集构建细节、训练参数、分类别指标、三模型对比和踩坑记录见
[`training/README.md`](training/README.md) 与 [`训练.md`](训练.md)。

> 说明：`app.py` 主体仍是预训练模型推理；`training/` 目录是独立补充的训练链，
> 不运行训练也不影响系统启动。

## 运行环境

建议使用 Python 3.10 或更高版本。首次运行需要准备：

1. Python 3
2. 一份 `yolo11m.pt` 权重文件，并放在项目根目录
3. 能够访问 Python 包源的网络环境

模型权重没有直接放入 Git 历史，而是作为 GitHub Release 附件提供：

- [下载 yolo11m.pt（Release）](https://github.com/Boye1014/software-engineering-coursework/releases/latest/download/yolo11m.pt)

下载后请将 `yolo11m.pt` 放到项目根目录，与 `app.py` 同级。

## 快速开始

### macOS / Linux

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 app.py
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

如果 PowerShell 不允许激活虚拟环境，可以先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

启动后打开浏览器：

```text
http://127.0.0.1:5000
```

## 运行设备

程序默认自动选择运行设备：

- Apple 芯片 Mac：优先使用 `mps`
- 安装了 CUDA 版 PyTorch 的 NVIDIA 电脑：优先使用 CUDA
- 其他情况：使用 CPU

可以通过环境变量手动指定设备：

```bash
YOLO_DEVICE=mps python3 app.py
YOLO_DEVICE=cpu python3 app.py
```

启动后也可以访问接口查看状态：

```text
http://127.0.0.1:5000/api/health
```

## 使用说明

1. 在“图片检测”或“视频检测”标签页中选择文件。
2. 在“检测模型”下拉里选择要用的权重（默认 COCO 预训练模型；选择自训权重后类别列表会自动变成该模型支持的类别）。
3. 根据需要调整置信度和 NMS / IoU 阈值。
4. 可以单击类别进行筛选；不选择类别时检测模型支持的全部核心类别。
5. 点击开始检测。
6. 视频检测会显示实际的处理百分比和已处理帧数。
7. 检测完成后可以预览、下载结果，或在历史记录中再次查看（记录里会标注所用模型）。

支持的文件类型：

- 图片：JPG、JPEG、PNG、BMP、WEBP
- 视频：MP4、MOV、AVI、MKV、WEBM
- 单个文件大小上限：512 MB

## 测试

项目带一套自动化测试，共 **178 个用例**，本机（RTX 4070 Laptop）全量运行约 **33 秒**。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

测试默认使用 `yolo11n.pt`（5.4MB）而不是 `yolo11m.pt`，单个用例的推理开销从 110ms
降到 30ms；想用别的权重可以 `YOLO_MODEL=yolo11m.pt python -m pytest`。

| 模块 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `test_unit_functions.py` | 59 | 置信度夹紧、扩展名白名单、模型白名单、类别筛选、路径防护、数据库迁移 |
| `test_api_pages.py` | 22 | 首页渲染、`/api/health`、`/api/classes`、静态资源、404 处理 |
| `test_api_detect_image.py` | 25 | 图片检测主流程、类别筛选、参数边界、多模型切换、临时文件清理 |
| `test_api_detect_video.py` | 10 | 异步任务提交、进度单调、产出可播、损坏视频转 error |
| `test_api_history.py` | 9 | 记录写入、倒序、50 条上限、删除联动清理结果文件 |
| `test_security_edges.py` | 23 | 非法文件类型、超大上传、文件名注入、路径穿越、模型标识白名单 |
| `test_frontend_static.py` | 9 | JS 引用的元素 ID / 类名与 HTML、CSS 的一致性 |
| `test_training_scripts.py` | 21 | VOC→YOLO 标注转换、设备选择、产物收集、指标抽取 |

隔离策略：每个用例使用独立的临时 `uploads/`、`results/` 和 SQLite 文件，
`JOBS` 任务表在用例前后清空，因此测试不会污染开发数据。

写测试的过程暴露了两个真实缺陷，已一并修复：

1. **`detections_from_result` 的类别兜底失效**：类别 ID 不在 `names` 里时先抛
   `KeyError`，写在后面的 `.get()` 兜底永远执行不到。
2. **结果文件缺失返回 302 而不是 404**：`<img>` / `<video>` 会拿到一页 HTML，
   前端表现为静默失败；现在 404 处理会区分页面路径与 `/api/`、`/media/`、`/download/`。

完整用例清单、覆盖边界和未覆盖项见 [`tests/README.md`](tests/README.md)。

## 项目结构

```text
software/
├── app.py                 # Flask 后端、模型推理、视频处理和数据库接口
├── templates/index.html   # 页面结构
├── static/style.css       # 页面样式
├── static/app.js          # 上传、进度轮询、交互和历史记录
├── requirements.txt       # Python 依赖
├── requirements-dev.txt   # 测试依赖（pytest）
├── pytest.ini             # pytest 配置
├── tests/                 # 自动化测试，178 个用例
│   ├── conftest.py        # 公共夹具：临时目录、独立数据库、测试素材
│   ├── README.md          # 用例清单与覆盖说明
│   └── test_*.py          # 按模块划分的 8 个测试文件
├── yolo11m.pt             # 从 GitHub Release 下载后放在根目录
├── test_images/           # 图片测试素材及来源说明
├── test_videos/           # 视频测试素材及来源说明
├── test_yolo.py           # 单张图片独立测试脚本
├── 训练.md                 # 训练与系统改造的工作记录、验证清单和遗留问题
├── training/              # 训练链：数据集构建、训练脚本、产物与说明
│   ├── prepare_data.py    # VOC2007 下载 + XML 转 YOLO + 四类过滤 + 切分
│   ├── prepare_coco12.py  # COCO2017-val + VOC2007 合并，构建 12 类数据集
│   ├── train.py           # YOLO11n 训练、验证、指标导出
│   ├── compare_models.py  # 三模型同台评测（标签 ID 重映射回 COCO）
│   ├── README.md          # 训练流程与结果分析
│   └── artifacts/         # 训练曲线、混淆矩阵、PR 曲线等
├── models/                # 自训权重与指标 JSON（权重本身不进仓库）
├── datasets/              # 数据集缓存与构建结果，运行时自动生成
├── uploads/               # 临时上传目录，运行时自动创建
├── results/               # 检测结果目录，运行时自动创建
└── detections.db          # SQLite 数据库，运行时自动创建
```

## 组员部署注意事项

- 不要复制其他电脑的 `.venv`，每台电脑都要根据 `requirements.txt` 重新创建虚拟环境。
- `yolo11m.pt` 必须放在项目根目录，否则程序无法加载模型。
- `detections.db`、`uploads/`、`results/`、`runs/`、`datasets/` 属于本地运行数据，不需要提交到版本库。
- 所有 `.pt` 权重都不进仓库（体积太大），需要时从 Release 下载或运行 `training/` 自行训练。
- 如果 NVIDIA 电脑启动后仍显示 CPU，需要安装与显卡驱动匹配的 CUDA 版 PyTorch。
- 改完代码建议先跑一遍 `python -m pytest`（约 30 秒），确认没有破坏已有功能。

## 测试素材

项目内的 `test_images/` 和 `test_videos/` 文件夹包含可直接上传测试的素材，来源和许可说明分别见：

- [`test_images/SOURCES.md`](test_images/SOURCES.md)
- [`test_videos/SOURCES.md`](test_videos/SOURCES.md)

## 说明

系统主体使用预训练模型进行推理，`training/` 目录提供了一条可复现的本地训练链
（VOC2007 四类子集 + YOLO11n），两者相互独立：不运行训练也不影响系统使用。
模型权重、第三方测试素材和相关依赖的使用应遵守各自的许可证及使用条款。
