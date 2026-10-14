
# Vision Lab：本地目标检测系统

这是一个面向软件工程课程展示的本地 Web 目标检测系统。项目使用 Ultralytics YOLO11m 预训练模型，在浏览器中完成图片检测和视频逐帧检测，不需要额外训练即可运行。

## 功能

- 图片上传、预览、目标检测和结果下载
- 视频上传、逐帧检测、实时处理进度和带框视频输出
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
2. 根据需要调整置信度和 NMS / IoU 阈值。
3. 可以单击类别进行筛选；不选择类别时检测默认的 12 个核心类别。
4. 点击开始检测。
5. 视频检测会显示实际的处理百分比和已处理帧数。
6. 检测完成后可以预览、下载结果，或在历史记录中再次查看。

支持的文件类型：

- 图片：JPG、JPEG、PNG、BMP、WEBP
- 视频：MP4、MOV、AVI、MKV、WEBM
- 单个文件大小上限：512 MB

## 项目结构

```text
software/
├── app.py                 # Flask 后端、模型推理、视频处理和数据库接口
├── templates/index.html   # 页面结构
├── static/style.css       # 页面样式
├── static/app.js          # 上传、进度轮询、交互和历史记录
├── requirements.txt       # Python 依赖
├── yolo11m.pt             # 从 GitHub Release 下载后放在根目录
├── test_images/           # 图片测试素材及来源说明
├── test_videos/           # 视频测试素材及来源说明
├── test_yolo.py           # 单张图片独立测试脚本
├── uploads/               # 临时上传目录，运行时自动创建
├── results/               # 检测结果目录，运行时自动创建
└── detections.db          # SQLite 数据库，运行时自动创建
```

## 组员部署注意事项

- 不要复制其他电脑的 `.venv`，每台电脑都要根据 `requirements.txt` 重新创建虚拟环境。
- `yolo11m.pt` 必须放在项目根目录，否则程序无法加载模型。
- `detections.db`、`uploads/`、`results/` 和 `runs/` 属于本地运行数据，不需要提交到版本库。
- 如果 NVIDIA 电脑启动后仍显示 CPU，需要安装与显卡驱动匹配的 CUDA 版 PyTorch。

## 测试素材

项目内的 `test_images/` 和 `test_videos/` 文件夹包含可直接上传测试的素材，来源和许可说明分别见：

- [`test_images/SOURCES.md`](test_images/SOURCES.md)
- [`test_videos/SOURCES.md`](test_videos/SOURCES.md)

## 说明

本项目使用预训练模型进行推理，不包含自定义数据集训练流程。模型权重、第三方测试素材和相关依赖的使用应遵守各自的许可证及使用条款。
