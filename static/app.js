(() => {
    const form = document.getElementById("detectForm");
    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    const filePicked = document.getElementById("filePicked");
    const previewWrap = document.getElementById("previewWrap");
    const imagePreview = document.getElementById("imagePreview");
    const videoPreview = document.getElementById("videoPreview");
    const clearFileButton = document.getElementById("clearFile");
    const submitButton = document.getElementById("submitButton");
    const submitText = document.getElementById("submitText");
    const resultEmpty = document.getElementById("resultEmpty");
    const resultContent = document.getElementById("resultContent");
    const loadingState = document.getElementById("loadingState");
    const loadingTitle = document.getElementById("loadingTitle");
    const loadingProgressWrap = document.getElementById("loadingProgressWrap");
    const loadingProgressBar = document.getElementById("loadingProgressBar");
    const loadingProgress = document.getElementById("loadingProgress");
    const loadingFrames = document.getElementById("loadingFrames");
    const resultStatus = document.getElementById("resultStatus");
    const resultMedia = document.getElementById("resultMedia");
    const resultSummary = document.getElementById("resultSummary");
    const detectionList = document.getElementById("detectionList");
    const downloadButton = document.getElementById("downloadButton");
    const newTaskButton = document.getElementById("newTaskButton");
    const historyList = document.getElementById("historyList");
    const historyCount = document.getElementById("historyCount");
    const batchQueue = document.getElementById("batchQueue");
    const batchQueueSummary = document.getElementById("batchQueueSummary");
    const batchJobList = document.getElementById("batchJobList");
    const toast = document.getElementById("toast");
    const classPicker = document.getElementById("classPicker");
    const selectedClasses = document.getElementById("selectedClasses");
    const selectedInputs = document.getElementById("selectedInputs");
    const confInput = document.getElementById("confInput");
    const iouInput = document.getElementById("iouInput");
    const confValue = document.getElementById("confValue");
    const iouValue = document.getElementById("iouValue");
    const tabs = document.querySelectorAll(".tab");

    let activeKind = "image";
    let previewUrl = null;
    let toastTimer = null;
    let progressTimer = null;
    let batchPollTimer = null;
    let batchPollRunning = false;
    let submitCooldownTimer = null;
    let videoSubmitLocked = false;
    const videoQueueItems = [];

    const escapeHtml = (value) => String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

    const showToast = (message) => {
        toast.textContent = message;
        toast.classList.add("show");
        window.clearTimeout(toastTimer);
        toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3200);
    };

    const lockSubmitButtonBriefly = () => {
        videoSubmitLocked = true;
        submitButton.disabled = true;
        window.clearTimeout(submitCooldownTimer);
        submitCooldownTimer = window.setTimeout(() => {
            videoSubmitLocked = false;
            submitButton.disabled = false;
            submitCooldownTimer = null;
        }, 1000);
    };

    const setKind = (kind) => {
        activeKind = kind;
        tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.kind === kind));
        const isImage = kind === "image";
        fileInput.accept = isImage ? "image/*" : "video/*";
        fileInput.multiple = !isImage;
        document.getElementById("dropTitle").textContent = isImage
            ? "拖拽图片到这里，或点击选择"
            : "拖拽多个视频到这里，或点击选择";
        document.getElementById("dropHint").textContent = isImage
            ? "支持 JPG、PNG、WEBP，单个文件最大 512 MB"
            : "支持一次选择多个 MP4、MOV、AVI、MKV、WEBM 视频，每个文件最大 512 MB";
        submitText.textContent = isImage ? "开始图片检测" : "开始视频检测";
        clearSelectedFile();
    };

    const clearSelectedFile = () => {
        fileInput.value = "";
        filePicked.textContent = "";
        previewWrap.classList.add("hidden");
        imagePreview.removeAttribute("src");
        videoPreview.removeAttribute("src");
        imagePreview.style.display = "none";
        videoPreview.style.display = "none";
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    };

    const showPreview = (files) => {
        const file = Array.isArray(files) ? files[0] : files;
        if (!file) return;
        const isImage = file.type.startsWith("image/") || activeKind === "image";
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        const selectedFiles = Array.isArray(files) ? files : [file];
        filePicked.textContent = selectedFiles.length > 1
            ? `已选择 ${selectedFiles.length} 个视频：${selectedFiles.map((item) => item.name).join("、")}`
            : `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        previewWrap.classList.remove("hidden");
        imagePreview.style.display = isImage ? "block" : "none";
        videoPreview.style.display = isImage ? "none" : "block";
        if (isImage) imagePreview.src = previewUrl;
        else videoPreview.src = previewUrl;
    };

    tabs.forEach((tab) => tab.addEventListener("click", () => setKind(tab.dataset.kind)));
    fileInput.addEventListener("change", () => showPreview([...fileInput.files]));
    clearFileButton.addEventListener("click", (event) => {
        event.preventDefault();
        clearSelectedFile();
    });
    ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (event) => {
        const files = [...event.dataTransfer.files];
        if (!files.length) return;
        if (activeKind === "image" && files.length > 1) {
            showToast("图片检测一次请选择一张图片。");
            return;
        }
        fileInput.files = event.dataTransfer.files;
        showPreview(files);
    });

    confInput.addEventListener("input", () => { confValue.value = confInput.value; confValue.textContent = confInput.value; });
    iouInput.addEventListener("input", () => { iouValue.value = iouInput.value; iouValue.textContent = iouInput.value; });

    const selectedClassIds = new Set();
    let classNames = new Map();

    const syncClassNames = () => {
        classNames = new Map(
            [...classPicker.querySelectorAll(".class-option")].map((option) => [
                option.dataset.class,
                option.querySelector("span:nth-child(2)").textContent.trim(),
            ])
        );
    };

    const renderClassPicker = (classes) => {
        classPicker.innerHTML = classes
            .map((item, index) => `<button type="button" class="class-option" data-class="${escapeHtml(item.id)}" title="添加 ${escapeHtml(item.name)}"><span class="class-option-id">${index + 1}</span><span>${escapeHtml(item.name)}</span></button>`)
            .join("");
        syncClassNames();
    };

    const renderClassSelection = () => {
        document.querySelectorAll(".class-option").forEach((option) => {
            option.classList.toggle("selected", selectedClassIds.has(option.dataset.class));
        });
        selectedInputs.innerHTML = [...selectedClassIds]
            .map((id) => `<input type="hidden" name="classes" value="${escapeHtml(id)}">`)
            .join("");
        if (!selectedClassIds.size) {
            selectedClasses.classList.add("empty");
            selectedClasses.innerHTML = "<span>暂无手动筛选，默认检测核心类别</span>";
            return;
        }
        selectedClasses.classList.remove("empty");
        selectedClasses.innerHTML = [...selectedClassIds].map((id) => `
            <span class="selected-chip">${escapeHtml(classNames.get(id) || id)}
                <button type="button" data-remove-class="${escapeHtml(id)}" aria-label="移除 ${escapeHtml(classNames.get(id) || id)}">×</button>
            </span>
        `).join("");
        selectedClasses.querySelectorAll("[data-remove-class]").forEach((button) => {
            button.addEventListener("click", () => {
                selectedClassIds.delete(button.dataset.removeClass);
                renderClassSelection();
            });
        });
    };

    const addClass = (id) => {
        selectedClassIds.add(String(id));
        renderClassSelection();
    };

    classPicker.addEventListener("click", (event) => {
        const option = event.target.closest(".class-option");
        if (option) addClass(option.dataset.class);
    });
    classPicker.addEventListener("keydown", (event) => {
        const option = event.target.closest(".class-option");
        if (option && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            addClass(option.dataset.class);
        }
    });
    const modelSelect = document.getElementById("modelSelect");
    const activeModelLabel = document.getElementById("activeModelLabel");
    const activeModelHint = document.getElementById("activeModelHint");
    const sideModelLabel = document.getElementById("sideModelLabel");
    const modelMeta = document.getElementById("modelMeta");

    syncClassNames();
    modelSelect?.addEventListener("change", async () => {
        selectedClassIds.clear();
        modelSelect.disabled = true;
        try {
            const response = await fetch(`/api/classes?model=${encodeURIComponent(modelSelect.value)}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "无法加载该模型的类别。");
            renderClassPicker(data.classes || []);
            renderClassSelection();
            activeModelLabel.textContent = data.model;
            activeModelHint.textContent = `${data.model_label} · ${data.class_count} 个可选类别`;
            sideModelLabel.textContent = data.model;
            const sizeText = modelSelect.options[modelSelect.selectedIndex]?.textContent || "";
            modelMeta.textContent = sizeText.replace(/^.*?·\s*/, "");
            showToast(`已切换到 ${data.model_label}，可选类别 ${data.class_count} 个。`);
        } catch (error) {
            showToast(error.message || "切换模型失败。");
        } finally {
            modelSelect.disabled = false;
        }
    });

    document.getElementById("clearClasses").addEventListener("click", () => {
        selectedClassIds.clear();
        renderClassSelection();
    });
    renderClassSelection();

    const stopProgressPolling = () => {
        if (progressTimer) {
            window.clearTimeout(progressTimer);
            progressTimer = null;
        }
    };

    const updateVideoProgress = (data) => {
        const processed = Number(data.processed_frames || 0);
        const total = Number(data.total_frames || 0);
        const progress = total
            ? Math.min(100, Math.round((processed / total) * 100))
            : Math.min(99, Number(data.progress || 0));
        loadingProgress.textContent = `${progress}%`;
        loadingProgressBar.style.width = `${progress}%`;
        loadingFrames.textContent = total
            ? `${processed.toLocaleString()} / ${total.toLocaleString()} 帧`
            : data.status === "queued"
                ? `排队序号 ${Number(data.queue_position || 1)}`
                : `${processed.toLocaleString()} 帧已处理`;
        loadingTitle.textContent = data.status === "queued"
            ? "任务已加入处理队列…"
            : "模型正在逐帧处理视频…";
        resultStatus.textContent = data.status === "queued" ? "QUEUED" : "RUNNING";
    };

    const setLoading = (isLoading) => {
        if (!isLoading) stopProgressPolling();
        submitButton.disabled = isLoading;
        resultEmpty.classList.toggle("hidden", isLoading);
        resultContent.classList.add("hidden");
        loadingState.classList.toggle("hidden", !isLoading);
        loadingProgressWrap.classList.toggle("hidden", !isLoading || activeKind !== "video");
        resultStatus.textContent = isLoading ? "RUNNING" : "WAITING";
        if (isLoading) {
            loadingTitle.textContent = activeKind === "image" ? "模型正在分析图片…" : "模型正在逐帧处理视频…";
            if (activeKind === "video") {
                loadingProgress.textContent = "0%";
                loadingProgressBar.style.width = "0%";
                loadingFrames.textContent = "正在读取视频信息…";
            }
        }
    };

    const summaryItem = (label, value) => `<div class="summary-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;

    const renderResult = (data) => {
        setLoading(false);
        resultEmpty.classList.add("hidden");
        resultContent.classList.remove("hidden");
        resultStatus.textContent = "COMPLETE";
        const cacheBust = `?t=${Date.now()}`;
        if (data.media_type === "image") {
            resultMedia.innerHTML = `<img src="${data.result_url}${cacheBust}" alt="检测结果">`;
            resultSummary.innerHTML = [
                summaryItem("检测目标", `${data.detection_count} 个`),
                summaryItem("耗时", `${data.elapsed_seconds}s`),
                summaryItem("设备", data.device),
                summaryItem("模型", data.model || "—"),
            ].join("");
            const detections = data.detections || [];
            detectionList.innerHTML = detections.length
                ? detections.map((item) => `<div class="detection-row"><b>${escapeHtml(item.class_name)}</b><span>${(item.confidence * 100).toFixed(1)}% · [${item.box.map((n) => Math.round(n)).join(", ")}]</span></div>`).join("")
                : `<div class="detection-row"><span>当前阈值下没有检测到目标</span><span>可降低置信度阈值重试</span></div>`;
        } else {
            resultMedia.innerHTML = `<video src="${data.result_url}${cacheBust}" controls></video>`;
            resultSummary.innerHTML = [
                summaryItem("检测框总数", `${data.detection_count}`),
                summaryItem("处理帧数", `${data.processed_frames}`),
                summaryItem("耗时", `${data.elapsed_seconds}s`),
                summaryItem("模型", data.model || "—"),
            ].join("");
            detectionList.innerHTML = `<div class="detection-row"><b>视频检测完成</b><span>${data.processed_frames} / ${data.total_frames || data.processed_frames} frames</span></div>`;
        }
        downloadButton.href = data.download_url;
        downloadButton.download = true;
        showToast("检测完成，结果已经保存到历史记录。");
    };

    const pollVideoJob = async (jobId) => {
        try {
            const response = await fetch(`/api/detect/${encodeURIComponent(jobId)}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "无法读取视频处理进度。");
            updateVideoProgress(data);
            if (data.status === "complete") {
                renderResult(data);
                await loadHistory();
                return;
            }
            if (data.status === "error") throw new Error(data.error || "视频检测失败。");
            progressTimer = window.setTimeout(() => pollVideoJob(jobId), 350);
        } catch (error) {
            setLoading(false);
            resultStatus.textContent = "ERROR";
            showToast(error.message || "视频检测失败。");
        }
    };

    const renderVideoQueue = () => {
        const counts = { uploading: 0, queued: 0, processing: 0, complete: 0, error: 0 };
        const labels = {
            uploading: "上传中",
            queued: "排队中",
            processing: "处理中",
            complete: "已完成",
            error: "失败",
        };
        videoQueueItems.forEach((item) => { counts[item.status] = (counts[item.status] || 0) + 1; });
        batchQueueSummary.textContent = `${videoQueueItems.length} 个任务 · 上传 ${counts.uploading} · 排队 ${counts.queued} · 处理中 ${counts.processing} · 完成 ${counts.complete} · 失败 ${counts.error}`;
        resultStatus.textContent = counts.uploading
            ? "UPLOADING"
            : counts.processing
                ? `${counts.processing} RUNNING`
                : counts.queued
                    ? `${counts.queued} QUEUED`
                    : videoQueueItems.length ? "BATCH COMPLETE" : "WAITING";
        batchJobList.innerHTML = videoQueueItems.map((item) => {
            const data = item.data || {};
            const total = Number(data.total_frames || 0);
            const processed = Number(data.processed_frames || 0);
            const progress = item.status === "complete"
                ? 100
                : total ? Math.min(100, Math.round(processed / total * 100)) : Number(data.progress || 0);
            const detail = item.status === "uploading"
                ? "正在上传到本机…"
                : item.status === "queued"
                    ? `队列位置 ${Number(data.queue_position || 1)}`
                    : item.status === "processing"
                        ? total ? `${processed.toLocaleString()} / ${total.toLocaleString()} 帧` : "正在读取视频信息…"
                        : item.status === "complete"
                            ? `${data.detection_count || 0} 个检测框 · ${data.elapsed_seconds || 0}s`
                            : item.error || data.error || "任务提交失败";
            const resultLinks = item.status === "complete"
                ? `<span class="batch-job-links"><a href="${escapeHtml(data.result_url)}" target="_blank" rel="noreferrer">预览</a><a href="${escapeHtml(data.download_url)}" download>下载</a></span>`
                : "";
            return `<article class="batch-job">
                <div class="batch-job-top"><strong title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</strong><span class="batch-job-status ${escapeHtml(item.status)}">${labels[item.status] || "等待中"}</span></div>
                <div class="batch-job-track"><span style="width:${progress}%"></span></div>
                <div class="batch-job-bottom"><small>${escapeHtml(detail)}</small>${resultLinks}</div>
            </article>`;
        }).join("");
        batchQueue.classList.toggle("hidden", videoQueueItems.length === 0);
    };

    const pollVideoQueue = async () => {
        batchPollTimer = null;
        batchPollRunning = true;
        const activeItems = videoQueueItems.filter((item) =>
            item.jobId && ["queued", "processing"].includes(item.status)
        );
        await Promise.all(activeItems.map(async (item) => {
            try {
                const response = await fetch(`/api/detect/${encodeURIComponent(item.jobId)}`);
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "无法读取任务状态。");
                item.data = data;
                item.status = data.status;
                if (data.status === "error") item.error = data.error || "视频检测失败";
            } catch (error) {
                item.status = "error";
                item.error = error.message || "视频检测失败";
            }
        }));
        renderVideoQueue();
        batchPollRunning = false;
        const hasOutstandingJobs = videoQueueItems.some((item) =>
            ["uploading", "queued", "processing"].includes(item.status)
        );
        if (hasOutstandingJobs) {
            batchPollTimer = window.setTimeout(pollVideoQueue, 500);
        } else {
            resultStatus.textContent = "BATCH COMPLETE";
            await loadHistory();
        }
    };

    const startVideoQueuePolling = () => {
        if (!batchPollRunning && !batchPollTimer) {
            batchPollTimer = window.setTimeout(pollVideoQueue, 0);
        }
    };

    const submitVideoBatch = async (files, settings) => {
        const items = files.map((file) => ({ file, status: "uploading", data: null, error: "" }));
        videoQueueItems.push(...items);
        batchQueue.classList.remove("hidden");
        resultEmpty.classList.add("hidden");
        resultContent.classList.add("hidden");
        loadingState.classList.add("hidden");
        resultStatus.textContent = "UPLOADING";
        renderVideoQueue();

        await Promise.all(items.map(async (item) => {
            try {
                const payload = new FormData();
                for (const [key, value] of settings.entries()) payload.append(key, value);
                payload.append("file", item.file, item.file.name);
                const response = await fetch("/api/detect", { method: "POST", body: payload });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "任务提交失败。");
                item.jobId = data.job_id;
                item.data = data;
                item.status = data.status || "queued";
            } catch (error) {
                item.status = "error";
                item.error = error.message || "任务提交失败。";
            }
            renderVideoQueue();
            if (item.jobId) startVideoQueuePolling();
        }));

        const submittedCount = items.filter((item) => item.jobId).length;
        const failedCount = items.length - submittedCount;
        showToast(failedCount
            ? `已排队 ${submittedCount} 个视频，${failedCount} 个未能提交。`
            : `已排队 ${submittedCount} 个视频，系统会逐个处理。`);
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const files = [...fileInput.files];
        if (!files.length) {
            showToast("请先选择一个文件。");
            return;
        }
        if (activeKind === "video") {
            if (videoSubmitLocked) return;
            const settings = new FormData(form);
            settings.delete("file");
            lockSubmitButtonBriefly();
            clearSelectedFile();
            submitVideoBatch(files, settings);
            return;
        }
        const payload = new FormData(form);
        setLoading(true);
        try {
            const response = await fetch("/api/detect", { method: "POST", body: payload });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "检测失败，请稍后重试。");
            if (data.media_type === "video" && data.job_id) {
                updateVideoProgress(data);
                pollVideoJob(data.job_id);
                return;
            }
            renderResult(data);
            await loadHistory();
        } catch (error) {
            setLoading(false);
            resultStatus.textContent = "ERROR";
            showToast(error.message || "检测失败。");
        }
    });

    const renderHistory = (items) => {
        historyCount.textContent = items.length;
        if (!items.length) {
            historyList.innerHTML = `<div class="history-empty">还没有检测记录，完成第一次任务后会显示在这里。</div>`;
            return;
        }
        historyList.innerHTML = items.map((item) => {
            const isImage = item.media_type === "image";
            const thumbnail = isImage ? `<img src="${item.result_url}" alt="">` : "▶";
            const classes = item.selected_classes?.length ? item.selected_classes.join(", ") : "全部类别";
            const model = item.model ? ` · ${item.model}` : "";
            return `<article class="history-item">
                <div class="history-thumb">${thumbnail}</div>
                <div class="history-info"><strong title="${escapeHtml(item.original_name)}">${escapeHtml(item.original_name)}</strong><span>${escapeHtml(item.created_at)} · ${escapeHtml(classes)} · ${escapeHtml(item.device)}${escapeHtml(model)}</span></div>
                <div class="history-stat"><b>${item.detection_count}</b> ${isImage ? "个目标" : "个检测框"}</div>
                <div class="history-actions"><a href="${item.result_url}" target="_blank" rel="noreferrer">查看</a><a href="${item.download_url}" download>下载</a><button type="button" data-delete="${item.id}">删除</button></div>
            </article>`;
        }).join("");
        historyList.querySelectorAll("[data-delete]").forEach((button) => {
            button.addEventListener("click", async () => {
                if (!window.confirm("确定删除这条检测记录吗？")) return;
                const response = await fetch(`/api/history/${button.dataset.delete}`, { method: "DELETE" });
                if (response.ok) {
                    showToast("记录已删除。");
                    loadHistory();
                } else showToast("删除失败。");
            });
        });
    };

    const loadHistory = async () => {
        try {
            const response = await fetch("/api/history");
            renderHistory(await response.json());
        } catch (_error) {
            historyList.innerHTML = `<div class="history-empty">历史记录暂时无法读取。</div>`;
        }
    };

    document.getElementById("refreshHistory").addEventListener("click", loadHistory);
    document.getElementById("historyNav").addEventListener("click", () => document.getElementById("historyPanel").scrollIntoView({ behavior: "smooth" }));
    newTaskButton.addEventListener("click", () => {
        clearSelectedFile();
        resultContent.classList.add("hidden");
        resultEmpty.classList.remove("hidden");
        resultStatus.textContent = "WAITING";
        document.getElementById("detectForm").scrollIntoView({ behavior: "smooth" });
    });

    fetch("/api/health").then((response) => response.json()).then((data) => {
        document.getElementById("deviceLabel").textContent = data.device;
    }).catch(() => {});
    loadHistory();
})();
