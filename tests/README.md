# 测试说明

本目录是「设备端目标检测系统」的自动化测试套件，覆盖后端接口、前端静态资源、
训练链脚本三部分。共 **178 个用例**，在本机（RTX 4070 Laptop / Python 3.13）
全量运行约 **33 秒**。

## 运行方式

```bash
# 安装测试依赖（pytest）
python -m pip install -r requirements-dev.txt

# 全量运行
python -m pytest

# 只跑某个模块
python -m pytest tests/test_api_detect_image.py -v

# 只跑不加载模型的快速用例（约 1 秒）
python -m pytest tests/test_unit_functions.py tests/test_frontend_static.py \
    tests/test_training_scripts.py

# 显示每个用例的执行耗时，定位慢用例
python -m pytest --durations=10
```

> 测试默认使用 `yolo11n.pt`（5.4MB）而不是 `yolo11m.pt`（39MB），
> 单个用例的推理开销从 110ms 降到 30ms。想换模型可以设环境变量：
> `YOLO_MODEL=yolo11m.pt python -m pytest`。

## 隔离策略

三条硬约束写在 `conftest.py` 里，保证测试不会污染开发数据：

| 隔离项 | 做法 |
| --- | --- |
| 上传目录 `uploads/` | 每个用例指向独立的 `tmp_path/uploads` |
| 结果目录 `results/` | 每个用例指向独立的 `tmp_path/results` |
| 数据库 `detections.db` | 每个用例新建独立的 SQLite 文件并建表 |
| 视频任务表 `JOBS` | 每个用例开始与结束都清空，异步状态不串场 |

`test_api_detect_image.py::test_app_module_database_path_is_temp` 专门守住这条底线：
一旦夹具失效、测试开始写真实的 `detections.db`，这个用例会立刻失败。

## 用例清单

### 1. `test_unit_functions.py` — 纯函数与模型元数据（59 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_clamp_float`（9） | 置信度/IoU 收敛：正常值、缺省、空串、非数字、上下越界 |
| `test_clamp_float_custom_range` | 自定义区间同样生效 |
| `test_media_type_image_extensions`（6） | jpg/jpeg/png/bmp/webp，含大写扩展名 |
| `test_media_type_video_extensions`（5） | mp4/mov/avi/mkv/webm，含大写扩展名 |
| `test_media_type_rejects_unknown`（6） | txt/gif/pdf/exe/无扩展名/双扩展名一律拒绝 |
| `test_media_type_with_directory_prefix` | 带目录前缀不影响扩展名判定 |
| `test_available_models_structure` | 权重枚举结果字段完整、文件真实存在 |
| `test_available_models_no_duplicate_ids` | 模型标识不重复 |
| `test_available_models_id_is_relative_to_project` | 标识是项目内相对路径，不泄漏绝对路径 |
| `test_resolve_model_defaults_to_first` | 不传标识时取默认权重 |
| `test_resolve_model_accepts_known_id` | 白名单内的标识可解析 |
| `test_resolve_model_rejects_unknown`（5） | 路径穿越、绝对路径、不存在的权重全部拒绝 |
| `test_model_names_filters_coco_to_core_classes` | 80 类模型被过滤成 12 个核心类 |
| `test_model_names_keeps_small_model_classes` | 12 类自训模型全量展示 |
| `test_model_names_four_class_model` | 4 类模型只输出 person/bicycle/car/bus |
| `test_selected_class_ids_*`（5） | 空值→None、排序去重、非法数字、越界 ID、非核心 COCO 类 |
| `test_selected_class_names_*`（2） | ID→名称映射、None→空列表 |
| `test_detections_from_result_*`（4） | 置信度保留 4 位、坐标保留 2 位、空框、未知类别兜底 |
| `test_safe_result_path_*`（3） | 命中已有文件、缺失文件 404、路径穿越被剥掉 |
| `test_init_database_*`（3） | 建表、旧库自动加 `model` 列、重复调用幂等 |

### 2. `test_api_pages.py` — 页面与元数据接口（22 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_index_returns_html` | 首页 200 且是 HTML |
| `test_index_renders_all_core_classes` | 类别按钮恰好 12 个，含 cat/dog/laptop/cell phone |
| `test_index_renders_model_selector` | 模型下拉选项都在白名单内 |
| `test_index_shows_active_model_and_device` | 当前模型与推理设备都渲染出来 |
| `test_index_contains_form_fields` | 表单、conf/iou 字段、类别区存在 |
| `test_health_reports_status_and_models` | `/api/health` 返回 status/model/device/models |
| `test_health_does_not_leak_absolute_paths` | 接口不暴露本机绝对路径 |
| `test_classes_default_model_returns_core_classes` | 不传 model 返回 12 个核心类 |
| `test_classes_for_coco_model_is_filtered` | COCO 模型返回 12 类 |
| `test_classes_for_self_trained_12` / `_4` | 自训模型按实际类别数返回 |
| `test_classes_rejects_unknown_model`（4） | 未知模型标识返回 400 |
| `test_static_assets_served`（2） | style.css / app.js 可访问 |
| `test_unknown_api_returns_json_404` | 未知 API 返回 JSON 错误 |
| `test_unknown_page_redirects_home` | 未知页面 302 回首页 |
| `test_detect_status_unknown_job` | 查询不存在的任务返回 404 |
| `test_media_missing_file` / `test_download_missing_file` | 结果文件缺失返回 404（不是回首页的 302） |

### 3. `test_api_detect_image.py` — 图片检测（25 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_detect_image_returns_detections` | 主流程：200、结果结构、耗时、结果/下载链接 |
| `test_detect_image_detection_fields` | 每个框的类别、置信度区间、坐标合法性（右下 > 左上） |
| `test_detect_image_finds_expected_classes` | 人+自行车图能识别出 person 或 bicycle |
| `test_detect_cat_image` | 猫图识别出 cat |
| `test_detect_people_bus_image` | 人群图至少 3 个目标且含 person |
| `test_detect_result_file_is_written_and_served` | 输出是真 JPEG（`FF D8` 头）、可预览、可下载 |
| `test_class_filter_only_person` | 只选 person 时结果里不含其它类 |
| `test_class_filter_multiple_classes` | 多选 person+bicycle 生效 |
| `test_class_filter_only_car` | 只选 car 生效 |
| `test_no_class_filter_uses_all_core_classes` | 不筛选时默认 12 类 |
| `test_invalid_class_id_falls_back_to_all` | 越界类别 ID 被忽略，回落到默认 |
| `test_non_numeric_class_id_is_ignored` | 非数字类别被忽略 |
| `test_higher_conf_never_returns_more` | 置信度提高，检出数单调不增 |
| `test_non_numeric_conf_uses_default` | conf/iou 传非数字不报错 |
| `test_out_of_range_conf_is_clamped` | conf=5 被夹紧，接口仍成功 |
| `test_detect_with_each_model`（3） | 三个权重都能跑通并回填 model_id |
| `test_detect_with_four_class_model_only_outputs_its_classes` | 4 类模型不会输出其它类 |
| `test_detect_with_twelve_class_model_exposes_twelve_classes` | 12 类模型暴露 12 个类别 |
| `test_detect_with_unknown_model_returns_400` | 伪造模型标识返回 400 |
| `test_detect_response_includes_model_label` | 响应带模型名，供历史记录展示 |
| `test_upload_temp_file_is_removed` | 上传的临时文件被清理 |
| `test_result_file_is_kept` | 结果文件保留且非空 |
| `test_app_module_database_path_is_temp` | 守住隔离底线（不写真实数据库） |

### 4. `test_api_detect_video.py` — 视频异步任务（10 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_video_submission_is_accepted` | 立即返回 202 + queued，不阻塞请求 |
| `test_video_job_completes` | 轮询到 complete，progress=100，帧数合理 |
| `test_video_result_is_playable` | 输出视频非空且可下载 |
| `test_video_progress_is_monotonic` | 进度单调不减且始终在 0~100 |
| `test_video_job_with_class_filter` | 任务级类别筛选生效 |
| `test_video_job_with_self_trained_model` | 视频用自训权重也能跑 |
| `test_video_rejects_unknown_model` | 伪造模型标识返回 400 |
| `test_video_upload_temp_file_removed` | 任务结束后清理上传文件 |
| `test_corrupt_video_reports_error` | 扩展名合法但内容损坏 → 任务转 error，不产生结果文件 |
| `test_jobs_registry_is_cleared_between_tests` | 验证夹具隔离 |

### 5. `test_api_history.py` — 历史记录（9 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_history_empty_initially` | 空库返回 `[]` |
| `test_history_records_image_detection` | 图片记录字段与检测响应一致（含 model） |
| `test_history_records_video_detection` | 视频记录写入，media_type=video |
| `test_history_is_newest_first` | 按 created_at 倒序 |
| `test_history_limits_to_fifty` | 超过 50 条只返回最近 50 条 |
| `test_delete_history_record` | 删除记录同时清理结果文件 |
| `test_delete_missing_record_returns_404` | 删除不存在的记录返回 404 |
| `test_delete_is_idempotent_only_first_time` | 重复删除第二次返回 404 |
| `test_history_survives_missing_result_file` | 结果文件被手工删掉后列表仍可用 |

### 6. `test_security_edges.py` — 安全与异常输入（23 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_detect_without_file_returns_400` | 未选文件返回 400 |
| `test_detect_with_empty_filename_returns_400` | 空文件名返回 400 |
| `test_detect_rejects_unsupported_extension`（5） | txt/py/zip/gif/exe 全部拒绝 |
| `test_detect_rejects_double_extension_smuggling` | 伪装成 jpg 的非图片 → 500 并给出可读错误，不崩溃 |
| `test_oversized_upload_returns_413` | 超过上限返回 413 |
| `test_malicious_filename_is_sanitized` | 文件名带 `../` 不产生越界写入 |
| `test_saved_original_name_has_no_directory` | 入库文件名不含路径分隔符 |
| `test_path_traversal_is_blocked`（4） | `/media`、`/download` 的各种编码穿越都拿不到源码 |
| `test_media_only_serves_existing_result` | 只服务 results 目录内的文件 |
| `test_model_id_whitelist`（5） | 模型标识白名单挡住绝对路径与通配符 |
| `test_classes_endpoint_never_returns_traceback` | 错误响应不泄漏调用栈 |
| `test_two_image_requests_do_not_interfere` | 连续两次检测的 id 与结果文件互不覆盖 |

### 7. `test_frontend_static.py` — 前端一致性（9 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_frontend_files_exist` | 模板/脚本/样式都存在且非空 |
| `test_every_element_id_used_by_js_exists_in_html` | **JS 里 46 个 `getElementById` 全部能在 HTML 找到** |
| `test_every_css_class_used_by_js_exists_in_stylesheet` | JS 动态拼的类名都有对应样式 |
| `test_model_selector_is_wired` | 模型下拉以 `model` 提交，切换时请求 `/api/classes` |
| `test_class_picker_renders_from_template` | 类别区由模板渲染，带 `data-class` |
| `test_js_has_no_placeholder_leftovers` | 没有 TODO/FIXME 残留 |
| `test_js_is_wrapped_in_iife` | 脚本以 IIFE 包裹并正确闭合 |
| `test_template_has_no_leaked_debug_output` | 模板无调试输出 |
| `test_css_defines_variables_and_balanced_braces` | 使用 CSS 变量、大括号成对 |

### 8. `test_training_scripts.py` — 训练链纯逻辑（21 个）

| 测试项 | 验证内容 |
| --- | --- |
| `test_convert_annotation_maps_class_ids` | VOC 类别名 → YOLO 类别 ID 的映射 |
| `test_convert_annotation_normalizes_coordinates` | 归一化中心点与宽高计算正确 |
| `test_convert_annotation_skips_difficult` | `difficult=1` 的框跳过并计数 |
| `test_convert_annotation_ignores_other_classes` | 非目标类别（dog/train）不写入标签 |
| `test_convert_annotation_drops_degenerate_box` | 宽高不足 1 像素的框丢弃 |
| `test_convert_annotation_clamps_out_of_bounds` | 越界坐标裁剪到图像范围 |
| `test_convert_annotation_handles_broken_xml` | 损坏 XML 返回空而不是抛异常 |
| `test_convert_annotation_without_size_returns_empty` | 缺少尺寸信息时安全返回 |
| `test_convert_annotation_updates_stats` | 类别计数器正确累加 |
| `test_link_or_copy_creates_link` / `_is_idempotent`（2） | 硬链接建立、已存在不覆盖 |
| `test_pick_device_*`（2） | 设备选择：显式指定与 auto |
| `test_resolve_data_missing_raises` / `_accepts_existing` （2） | 数据配置路径解析与缺失报错 |
| `test_copy_artifacts_*`（3） | 新命名优先、兼容旧命名、空目录返回空 |
| `test_extract_metrics_*`（3） | 整体指标、分类别指标、缺字段时不崩 |

## 测试发现并修复的问题

写测试的过程本身就是一次代码审查，以下问题由用例暴露并已修复：

| 问题 | 表现 | 修复 |
| --- | --- | --- |
| `detections_from_result` 的类别兜底失效 | 类别 ID 不在 `names` 里时先抛 `KeyError`，后面的 `.get` 兜底永远执行不到 | 改成先按类型取值再兜底（`app.py`） |
| 结果文件缺失返回 302 而不是 404 | `<img src>`、`<video src>` 拿到一页 HTML，前端表现为静默失败 | 404 处理区分页面路径与 `/api/`、`/media/`、`/download/` |

## 已知未覆盖的部分

诚实记录，避免"测试全绿 = 没问题"的错觉：

1. **模型精度不做断言**。用例只验证"能检出、类别对、结构合法"，不断言具体
   mAP 或检出数量。精度回归应由 `training/compare_models.py` 负责，而不是单元测试。
2. **并发压力未测**。`JOBS` 字典只增不减、两个视频任务抢 `MODEL_LOCK` 这两个
   已知问题需要专门的并发用例，当前只有一次"两次连续请求不互相覆盖"的粗测。
3. **音频轨道未测**。视频输出目前只有画面轨，没有音轨（已知遗留问题），
   用例只验证文件可播、可下载。
4. **跨平台路径未测**。Windows 下的硬链接、盘符路径行为与其他系统不同，
   只在 Windows 上跑过。

## 新增用例的约定

- 一个用例只验证一件事，名字写成"断言内容"而不是"被测函数"。
- 需要模型或视频的用例直接用 `client` / `post_image` / `short_clip` 夹具，
  不要自己拼 multipart 请求。
- 不要写依赖真实 `uploads/`、`results/`、`detections.db` 的用例。
- 涉及外部网络的用例（下载数据集）不要放进测试套件。
