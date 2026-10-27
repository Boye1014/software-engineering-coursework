# 训练链：VOC2007 四类子集 + YOLO11n

系统主体（`app.py`）用的是 COCO 预训练的 YOLO11m 做推理，开箱就能得到不错的效果。
但实验要求是「轻量级模型 + 本地 CPU/GPU **完成训练和推理**」，只做推理拿不到
训练环节的分数。所以这一节单独补一条**最小但完整**的训练链：

```
数据准备 → 训练 → 验证 → 产出可部署权重 → 接入 Web 系统
```

## 1. 数据集选择

选用 **PASCAL VOC2007 trainval**（公开数据集，官方 5011 张带标注图片）。

| 项目 | 说明 |
| --- | --- |
| 来源 | PASCAL VOC2007，由 ultralytics 官方镜像分发（425MB） |
| 任务 | 通用目标检测，保留 4 个交通/街道相关类别 |
| 保留类别 | `person` `bicycle` `car` `bus` |
| 过滤规则 | 只保留上述 4 类；`difficult=1` 的 1259 个框按 VOC 惯例跳过 |
| 划分 | trainval 按 9:1 用固定种子（seed=0）切分，保证可复现 |

> 为什么选 VOC 而不是自己拍照片标注：VOC 有标准标注、公开可复现，
> 答辩时能拿出 mAP 这类可对比的客观指标；自建小数据集只有几十张，
> 指标没有说服力。

构建后的规模：

| 划分 | 图片 | 目标框 |
| --- | --- | --- |
| train | 4510 | 5899 |
| val | 501 | 623 |

类别分布（目标框数）：

| 类别 | ID | 框数 |
| --- | --- | --- |
| person | 0 | 4690 |
| car | 2 | 1250 |
| bicycle | 1 | 353 |
| bus | 3 | 229 |
| 合计 | | 6522 |

两点需要交代清楚：

1. **类别极度不均衡**。VOC 里 person 占绝对多数，bus 只有 229 个框。
   不过实测下来精度最低的反而是样本最多的 person——原因见第 4 节的分析，
   类别的"难度"（目标大小、遮挡程度）比样本数量更决定最终 mAP。
2. **保留了负样本**。4510 张训练图里有 2130 张不含这 4 类目标（只含
   aeroplane / cow / dog 等其它类）。它们以空标签的形式保留，充当背景图，
   有助于压低误检率。

### 类别 ID 为什么这样排

新类别 ID 刻意和 COCO 前四类对齐：

```
0 person   1 bicycle   2 car   3 bus
```

`app.py` 里用 `CORE_CLASS_IDS` 过滤界面上的候选类别，而该集合包含 0/1/2/3，
所以模型换成自训权重后，**系统一行代码都不用改**，界面会自动只剩这 4 个类别。

## 2. 运行方式

```bash
# 1) 构建数据集（首次会下载 425MB，支持断点续传）
python training/prepare_data.py

# 2) 训练（默认 100 epoch）
python training/train.py

# 训练参数可调，例如先跑通链路再上全量
python training/train.py --epochs 3 --batch 16
```

`prepare_data.py` 做的事：下载 → 解压 → 逐张解析 VOC 的 XML 标注 →
归一化成 YOLO 的 `(class cx cy w h)` → 过滤 4 类 → 切分 → 生成 `data.yaml`。
其中图片用硬链接而非复制，省掉约 700MB 冗余磁盘占用。

`train.py` 做的事：训练 → 取出最优权重复制到 `models/` →
在验证集上复核 → 导出指标 JSON 和曲线图 → 打印切换命令。

## 3. 训练配置

| 参数 | 取值 |
| --- | --- |
| 基础权重 | `yolo11n.pt`（COCO 预训练，迁移学习起点） |
| 模型规模 | 2.6M 参数 / 6.4 GFLOPs / 5.4MB |
| 输入尺寸 | 640×640 |
| batch | 32 |
| epoch | 100（早停 patience=20） |
| 初始学习率 | 0.01（SGD） |
| 随机种子 | 0（`deterministic=True`） |
| 设备 | 自动选择 CUDA / MPS / CPU |
| 数据增强 | ultralytics 默认策略（mosaic、HSV、翻转等），最后 10 epoch 关闭 mosaic |

## 4. 训练结果

训练环境：RTX 4070 Laptop（8GB），batch 32，100 epoch 跑满，耗时 **6000 秒（约 100 分钟）**。

整体指标（VOC2007 四类验证集，501 张图 / 623 个框）：

| 指标 | 数值 |
| --- | --- |
| mAP@0.5 | **0.8565** |
| mAP@0.5:0.95 | **0.6532** |
| mAP@0.75 | 0.7357 |
| 精确率 Precision | 0.8885 |
| 召回率 Recall | 0.7672 |

分类别指标：

| 类别 | 验证集框数 | Precision | Recall | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| person | 451 | 0.8621 | 0.7339 | 0.5375 |
| bicycle | 36 | 0.9071 | 0.8142 | 0.7224 |
| car | 109 | 0.8219 | 0.7798 | 0.6629 |
| bus | 27 | 0.9631 | 0.7407 | 0.6899 |

训练曲线（`results.png`）显示：第 60~70 轮后 mAP@0.5 稳定在 0.84~0.86 之间不再上升，
而 train / val 的 box、cls、dfl 三项损失仍在缓慢下降，没有出现过拟合迹象，
说明在这个数据规模上模型容量已经够用，继续加 epoch 收益很小。

### 一个反直觉的结论

按直觉「样本越少精度越低」，但这组数据恰好相反：

- `person` 有 4690 个训练框，mAP@0.5:0.95 却最低（0.5375）；
- `bus` 只有 229 个训练框，精度最高（P=0.9631，mAP@0.5:0.95=0.6899）。

原因是 VOC 里的 person 大量出现在合影、人群、远景中，目标小而密集、相互遮挡，
定位精度（IoU 要求从 0.5 到 0.95）很难做高；而 bus 目标大、外观统一、背景干净，
即使样本少也容易学。**所以对检测任务来说，类别的"难易程度"比"样本数量"更影响最终 mAP**，
这也解释了为什么 mAP@0.5（0.857）与 mAP@0.5:0.95（0.653）之间有 20 个点的落差——
模型"找得到"目标，但"框得不够紧"。

### 速度与体积对比

同一张测试图（768×432，imgsz=640，各测 8 次取最优）：

| 模型 | 参数量 | 体积 | CPU 单帧 | GPU 单帧 |
| --- | --- | --- | --- | --- |
| 自训 YOLO11n（4 类） | 2.6M | 5.2MB | **33 ms** | **11 ms** |
| 官方 YOLO11n（80 类） | 2.6M | 5.4MB | 36 ms | 11 ms |
| 官方 YOLO11m（80 类，系统默认） | 20.1M | 38.8MB | 129 ms | 21 ms |

结论：换成 n 之后**体积缩小 7.5 倍、CPU 推理快近 4 倍**，GPU 上差距被调度开销吃掉大半。
这也是题目强调"轻量级模型"的实际意义——没有独立显卡的机器上，m 和 n 是两个体验。

## 5. 产物清单

| 路径 | 内容 |
| --- | --- |
| `models/traffic_yolo11n.pt` | 训练得到的最优权重（5.2MB） |
| `models/traffic_yolo11n.metrics.json` | 训练配置 + 整体/分类别指标（纯 JSON，便于写报告） |
| `training/artifacts/traffic_yolo11n/results.png` | 损失与 mAP 训练曲线 |
| `training/artifacts/traffic_yolo11n/confusion_matrix.png` | 混淆矩阵 |
| `training/artifacts/traffic_yolo11n/BoxPR_curve.png` | 各类 PR 曲线 |
| `training/artifacts/traffic_yolo11n/results.csv` | 逐 epoch 原始指标 |
| `training/artifacts/traffic_yolo11n/train_log.txt` | 精简后的训练日志（含每轮分类别指标） |
| `training/artifacts/traffic_yolo11n/val_batch0_pred.jpg` | 验证集预测可视化 |
| `runs/train/traffic_yolo11n/` | ultralytics 原始输出目录（含更多预测图） |

## 6. 把自训模型接进 Web 系统

```bash
# macOS / Linux
YOLO_MODEL=models/traffic_yolo11n.pt python app.py

# Windows PowerShell
$env:YOLO_MODEL="models/traffic_yolo11n.pt"; python app.py
```

启动后首页左下的「ACTIVE MODEL」会显示 `traffic_yolo11n.pt`，
类别筛选区自动只剩 person / bicycle / car / bus 四项。
图片检测、视频逐帧、阈值调节、历史记录、结果下载等功能全部照常工作。

## 7. 局限与说明

- **只认这 4 类**。自训权重是 4 类模型，检测不到 `dog` `laptop` 等类别，
  这是任务裁剪的正常结果，不是模型缺陷。
- **mAP 不能和 COCO 官方数字直接比**。数据集、类别数、验证集都不同。
  本项目报告的 mAP 只针对上面这张 VOC2007 四类验证集。
- **小目标与密集目标偏弱**。人群、远景中的 person 定位精度明显低于车辆，
  mAP@0.5:0.95 只有 0.54。若实际场景以人群计数为主，需要提高输入分辨率
  （imgsz=960/1280）或改用更大的模型。
- **训练集是 VOC2007 单一年份**。想要更强的泛化，可以把 VOC2012 一起并入，
  或改用 VisDrone 这类更贴近实际场景的数据集。

---

## 8. 扩展：12 类模型（COCO2017-val + VOC2007）

上面 1~7 节是 4 类的版本。由于界面展示 12 个核心类别，另提供一条 12 类训练链，
让自训模型的类别范围与界面一致。

```bash
python training/prepare_coco12.py    # 下载 COCO val2017(778MB)+标注(242MB)，与本地 VOC 合并
python training/train.py --data datasets/traffic12/data.yaml \
    --epochs 100 --batch 16 --patience 100 --name traffic12_yolo11n
python training/compare_models.py --name traffic12_yolo11n   # 三模型同台评测
```

数据集：COCO val2017 与 VOC2007 合并，共 **7521 张图 / 25164 个框**（train 6770 / val 751），
类别 ID 连续编号 0~11。

结果：**mAP@0.5 = 0.5912，mAP@0.5:0.95 = 0.4312**，权重 5.3MB，CPU 单帧 27.7ms。

同一批 751 张图上的横向对比：

| 模型 | 类别数 | mAP@0.5 | mAP@0.5:0.95 | 体积 | CPU |
| --- | --- | --- | --- | --- | --- |
| 自训 traffic12_yolo11n | 12 | 0.5912 | 0.4312 | 5.3MB | 27.7ms |
| 官方 yolo11n | 80 | 0.7103 | 0.5474 | 5.4MB | 29.9ms |
| 官方 yolo11m | 80 | 0.8093 | 0.6486 | 38.8MB | 112.5ms |

同样的 n 架构、同样的速度，自训版 mAP@0.5 低 12 个点，差距来自训练数据量
（11.8 万张 vs 6800 张）。详细分析和踩坑记录见根目录 [`训练.md`](../训练.md) 第八节。

**注意**：换检测头后指标会先崩后升（0.52 → 0.12 → 0.59），训练时务必用
`--patience 100` 关闭早停，否则会在第 20 轮左右误停，保存下未收敛的权重。
