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

    const setKind = (kind) => {
        activeKind = kind;
        tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.kind === kind));
        const isImage = kind === "image";
        fileInput.accept = isImage ? "image/*" : "video/*";
        document.getElementById("dropTitle").textContent = isImage
            ? "拖拽图片到这里，或点击选择"
            : "拖拽视频到这里，或点击选择";
        document.getElementById("dropHint").textContent = isImage
            ? "支持 JPG、PNG、WEBP，单个文件最大 512 MB"
            : "支持 MP4、MOV、AVI、MKV、WEBM，单个文件最大 512 MB";
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

    const showPreview = (file) => {
        if (!file) return;
        const isImage = file.type.startsWith("image/") || activeKind === "image";
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        filePicked.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        previewWrap.classList.remove("hidden");
        imagePreview.style.display = isImage ? "block" : "none";
        videoPreview.style.display = isImage ? "none" : "block";
        if (isImage) imagePreview.src = previewUrl;
        else videoPreview.src = previewUrl;
    };

    tabs.forEach((tab) => tab.addEventListener("click", () => setKind(tab.dataset.kind)));
    fileInput.addEventListener("change", () => showPreview(fileInput.files[0]));
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
        const file = event.dataTransfer.files[0];
        if (!file) return;
        fileInput.files = event.dataTransfer.files;
        showPreview(file);
    });

    confInput.addEventListener("input", () => { confValue.value = confInput.value; confValue.textContent = confInput.value; });
    iouInput.addEventListener("input", () => { iouValue.value = iouInput.value; iouValue.textContent = iouInput.value; });

    const selectedClassIds = new Set();
    const classNames = new Map(
        [...document.querySelectorAll(".class-option")].map((option) => [option.dataset.class, option.querySelector("span:nth-child(2)").textContent.trim()])
    );

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
            : `${processed.toLocaleString()} 帧已处理`;
        loadingTitle.textContent = data.status === "queued"
            ? "任务已加入处理队列…"
            : "模型正在逐帧处理视频…";
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

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
            showToast("请先选择一个文件。");
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
            return `<article class="history-item">
                <div class="history-thumb">${thumbnail}</div>
                <div class="history-info"><strong title="${escapeHtml(item.original_name)}">${escapeHtml(item.original_name)}</strong><span>${escapeHtml(item.created_at)} · ${escapeHtml(classes)} · ${escapeHtml(item.device)}</span></div>
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
