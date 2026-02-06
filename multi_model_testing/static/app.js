let MODELS_INFO = null;

let selectedDetModels = new Set();
let selectedClsModels = new Set();

let detModelLabelSelections = {};
let clsModelLabelSelections = {};

let currentFiles = [];
let latestResults = [];
let currentResultIndex = 0;
const zoomStates = {}; // cardId -> scale

let currentModalKind = null;   // 'det' or 'cls'
let currentModalModel = null;  // model name last clicked in modal

// ---------- helpers ----------

function updateModelSummaries() {
  const detSummary = document.getElementById("det-selected-summary");
  const clsSummary = document.getElementById("cls-selected-summary");

  detSummary.textContent = selectedDetModels.size
    ? Array.from(selectedDetModels).join(", ")
    : "None selected";

  clsSummary.textContent = selectedClsModels.size
    ? Array.from(selectedClsModels).join(", ")
    : "None selected";
}

function rebuildLabelCards(kind) {
  const container = document.getElementById(kind === "det" ? "det-label-cards" : "cls-label-cards");
  container.innerHTML = "";
  if (!MODELS_INFO) return;

  const selected = kind === "det" ? selectedDetModels : selectedClsModels;
  const labelsDict = kind === "det" ? MODELS_INFO.detection_labels : MODELS_INFO.classification_labels;
  const labelSelections = kind === "det" ? detModelLabelSelections : clsModelLabelSelections;

  if (!selected.size) {
    const p = document.createElement("div");
    p.className = "muted small";
    p.textContent = "No " + (kind === "det" ? "detection" : "classification") + " models selected.";
    container.appendChild(p);
    return;
  }

  selected.forEach(modelName => {
    const labels = labelsDict[modelName] || [];
    if (!labelSelections[modelName]) {
      labelSelections[modelName] = new Set(labels);
    }
    const selectedLabels = labelSelections[modelName];

    const card = document.createElement("div");
    card.className = "model-label-card";

    const header = document.createElement("h4");
    header.textContent = modelName;
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = labels.length + " labels";
    header.appendChild(badge);
    card.appendChild(header);

    const list = document.createElement("div");
    list.className = "labels-list";

    labels.forEach(lbl => {
      const wrap = document.createElement("label");
      wrap.className = "label-checkbox";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = selectedLabels.has(lbl);
      cb.dataset.model = modelName;
      cb.dataset.kind = kind;
      cb.value = lbl;
      cb.addEventListener("change", () => {
        const set = kind === "det" ? detModelLabelSelections : clsModelLabelSelections;
        if (!set[modelName]) set[modelName] = new Set();
        if (cb.checked) set[modelName].add(lbl);
        else set[modelName].delete(lbl);
      });
      wrap.appendChild(cb);
      const span = document.createElement("span");
      span.textContent = lbl;
      wrap.appendChild(span);
      list.appendChild(wrap);
    });

    card.appendChild(list);
    container.appendChild(card);
  });
}

// ---------- help section ----------

function populateHelpSection() {
  if (!MODELS_INFO) return;
  const help = document.getElementById("help-section");
  help.innerHTML = "";

  const detTitle = document.createElement("div");
  detTitle.textContent = "Detection models & labels:";
  detTitle.style.marginBottom = "4px";
  detTitle.style.fontSize = "11px";
  detTitle.style.color = "#e5e7eb";
  help.appendChild(detTitle);

  Object.entries(MODELS_INFO.detection_labels).forEach(([name, labels]) => {
    const block = document.createElement("div");
    block.className = "help-model";
    const t = document.createElement("div");
    t.className = "help-model-title";
    t.textContent = name;
    block.appendChild(t);
    const pillRow = document.createElement("div");
    pillRow.className = "help-label-pills";
    labels.forEach(lbl => {
      const pill = document.createElement("span");
      pill.className = "help-pill";
      pill.textContent = lbl;
      pillRow.appendChild(pill);
    });
    block.appendChild(pillRow);
    help.appendChild(block);
  });

  const clsTitle = document.createElement("div");
  clsTitle.textContent = "Classification models & labels:";
  clsTitle.style.margin = "8px 0 4px 0";
  clsTitle.style.fontSize = "11px";
  clsTitle.style.color = "#e5e7eb";
  help.appendChild(clsTitle);

  Object.entries(MODELS_INFO.classification_labels).forEach(([name, labels]) => {
    const block = document.createElement("div");
    block.className = "help-model";
    const t = document.createElement("div");
    t.className = "help-model-title";
    t.textContent = name;
    block.appendChild(t);
    const pillRow = document.createElement("div");
    pillRow.className = "help-label-pills";
    labels.forEach(lbl => {
      const pill = document.createElement("span");
      pill.className = "help-pill";
      pill.textContent = lbl;
      pillRow.appendChild(pill);
    });
    block.appendChild(pillRow);
    help.appendChild(block);
  });
}

// ---------- drag & drop ----------

function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("image-input");
  const fileLabel = document.getElementById("drop-text-file");

  function setFiles(fileList) {
    currentFiles = Array.from(fileList);
    if (currentFiles.length === 0) {
      fileLabel.style.display = "none";
      return;
    }
    fileLabel.style.display = "block";
    if (currentFiles.length === 1) {
      fileLabel.textContent = "Selected: " + currentFiles[0].name;
    } else {
      fileLabel.textContent = "Selected " + currentFiles.length + " images (first: " + currentFiles[0].name + ")";
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      setFiles(fileInput.files);
    }
  });

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", e => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      setFiles(e.dataTransfer.files);
    }
  });
}

// ---------- models & init ----------

async function loadModels() {
  const res = await fetch("/models");
  MODELS_INFO = await res.json();

  // initialise label selections to "all labels" for each model
  detModelLabelSelections = {};
  clsModelLabelSelections = {};
  MODELS_INFO.detection_models.forEach(name => {
    detModelLabelSelections[name] = new Set(MODELS_INFO.detection_labels[name] || []);
  });
  MODELS_INFO.classification_models.forEach(name => {
    clsModelLabelSelections[name] = new Set(MODELS_INFO.classification_labels[name] || []);
  });

  populateHelpSection();
  updateModelSummaries();
  rebuildLabelCards("det");
  rebuildLabelCards("cls");
}

// ---------- sliders ----------

function initThresholdSliders() {
  const detSlider = document.getElementById("det-thresh");
  const clsSlider = document.getElementById("cls-thresh");
  const detLabel = document.getElementById("det-thresh-label");
  const clsLabel = document.getElementById("cls-thresh-label");

  function updateDet() { detLabel.textContent = (detSlider.value / 100).toFixed(2); }
  function updateCls() { clsLabel.textContent = (clsSlider.value / 100).toFixed(2); }

  detSlider.addEventListener("input", updateDet);
  clsSlider.addEventListener("input", updateCls);
  updateDet();
  updateCls();
}

// ---------- modal: model + label selection ----------

function renderModalLabelsPanel() {
  if (!currentModalKind || !MODELS_INFO) return;

  const labelsList = document.getElementById("modal-labels-list");
  const labelsTitle = document.getElementById("modal-labels-title");

  labelsList.innerHTML = "";

  const isDet = currentModalKind === "det";
  const labelsDict = isDet ? MODELS_INFO.detection_labels : MODELS_INFO.classification_labels;
  const labelSelections = isDet ? detModelLabelSelections : clsModelLabelSelections;
  const selectedSet = isDet ? selectedDetModels : selectedClsModels;

  const selectedArr = Array.from(selectedSet);

  if (selectedArr.length > 0) {
    labelsTitle.textContent = "Labels for selected models";

    selectedArr.forEach(modelName => {
      const labels = labelsDict[modelName] || [];
      if (!labelSelections[modelName]) {
        labelSelections[modelName] = new Set(labels);
      }
      const selectedLabels = labelSelections[modelName];

      const card = document.createElement("div");
      card.className = "model-label-card";
      card.style.marginBottom = "6px";

      const header = document.createElement("h4");
      header.textContent = modelName;
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = labels.length + " labels";
      header.appendChild(badge);
      card.appendChild(header);

      const list = document.createElement("div");
      list.className = "labels-list";

      labels.forEach(lbl => {
        const wrap = document.createElement("label");
        wrap.className = "label-checkbox";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = selectedLabels.has(lbl);
        cb.addEventListener("change", () => {
          if (cb.checked) selectedLabels.add(lbl);
          else selectedLabels.delete(lbl);
          rebuildLabelCards(currentModalKind);
        });
        const span = document.createElement("span");
        span.textContent = lbl;
        wrap.appendChild(cb);
        wrap.appendChild(span);
        list.appendChild(wrap);
      });

      card.appendChild(list);
      labelsList.appendChild(card);
    });
  } else if (currentModalModel) {
    // preview-only mode: user clicked on a model but hasn't selected any yet
    const labels = labelsDict[currentModalModel] || [];
    labelsTitle.textContent = "Labels for " + currentModalModel + " (model not selected)";
    if (!labels.length) {
      const p = document.createElement("div");
      p.className = "muted small";
      p.style.padding = "6px 8px";
      p.textContent = "No labels found.";
      labelsList.appendChild(p);
    } else {
      const list = document.createElement("div");
      list.style.padding = "6px 8px";
      labels.forEach(lbl => {
        const row = document.createElement("div");
        row.className = "muted small";
        row.textContent = "• " + lbl;
        list.appendChild(row);
      });
      labelsList.appendChild(list);
    }
  } else {
    labelsTitle.textContent = "Labels";
    const p = document.createElement("div");
    p.className = "muted small";
    p.style.padding = "6px 8px";
    p.textContent = "Select a model on the left to view labels.";
    labelsList.appendChild(p);
  }
}

function openModelModal(kind) {
  if (!MODELS_INFO) return;

  currentModalKind = kind;
  const modal = document.getElementById("model-modal");
  const title = document.getElementById("modal-title");
  const modelList = document.getElementById("modal-model-list");

  modelList.innerHTML = "";
  currentModalModel = null;

  const isDet = kind === "det";
  const models = isDet ? MODELS_INFO.detection_models : MODELS_INFO.classification_models;
  const selectedSet = isDet ? selectedDetModels : selectedClsModels;
  const labelsDict = isDet ? MODELS_INFO.detection_labels : MODELS_INFO.classification_labels;
  const labelSelections = isDet ? detModelLabelSelections : clsModelLabelSelections;

  title.textContent = isDet ? "Select detection models & labels" : "Select classification models & labels";

  models.forEach(name => {
    const row = document.createElement("div");
    row.className = "modal-model-row";
    row.dataset.model = name;

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedSet.has(name);

    cb.addEventListener("click", e => {
      e.stopPropagation();
      if (cb.checked) {
        selectedSet.add(name);
        // when a model is newly selected, default to ALL labels checked
        if (!labelSelections[name]) {
          labelSelections[name] = new Set(labelsDict[name] || []);
        } else if (labelSelections[name].size === 0) {
          labelsDict[name].forEach(l => labelSelections[name].add(l));
        }
      } else {
        selectedSet.delete(name);
      }
      row.classList.toggle("selected", cb.checked);
      updateModelSummaries();
      rebuildLabelCards(kind);
      renderModalLabelsPanel();
    });

    const label = document.createElement("span");
    label.textContent = name;

    row.addEventListener("click", () => {
      // focus model for preview (if not selected)
      currentModalModel = name;
      document.querySelectorAll(".modal-model-row").forEach(r => {
        const innerCb = r.querySelector("input[type='checkbox']");
        r.classList.toggle("selected", innerCb.checked && r.dataset.model === name);
      });
      renderModalLabelsPanel();
    });

    row.appendChild(cb);
    row.appendChild(label);
    row.classList.toggle("selected", cb.checked && selectedSet.has(name));

    modelList.appendChild(row);
  });

  // default preview: first selected or first model
  const firstSelected = models.find(m => (isDet ? selectedDetModels : selectedClsModels).has(m));
  currentModalModel = firstSelected || models[0] || null;

  renderModalLabelsPanel();

  modal.classList.remove("hidden");
}

function closeModelModal() {
  const modal = document.getElementById("model-modal");
  modal.classList.add("hidden");
  currentModalKind = null;
  currentModalModel = null;
}

function initModalControls() {
  const closeBtn = document.getElementById("modal-close");
  const doneBtn = document.getElementById("modal-done");
  const selectAllBtn = document.getElementById("modal-select-all-labels");
  const clearBtn = document.getElementById("modal-clear-labels");

  closeBtn.addEventListener("click", () => {
    closeModelModal();
    rebuildLabelCards("det");
    rebuildLabelCards("cls");
  });

  doneBtn.addEventListener("click", () => {
    closeModelModal();
    rebuildLabelCards("det");
    rebuildLabelCards("cls");
  });

  selectAllBtn.addEventListener("click", () => {
    if (!currentModalKind || !MODELS_INFO) return;
    const isDet = currentModalKind === "det";
    const labelsDict = isDet ? MODELS_INFO.detection_labels : MODELS_INFO.classification_labels;
    const labelSelections = isDet ? detModelLabelSelections : clsModelLabelSelections;
    const selectedSet = isDet ? selectedDetModels : selectedClsModels;

    selectedSet.forEach(name => {
      labelSelections[name] = new Set(labelsDict[name] || []);
    });

    renderModalLabelsPanel();
    rebuildLabelCards(currentModalKind);
  });

  clearBtn.addEventListener("click", () => {
    if (!currentModalKind) return;
    const isDet = currentModalKind === "det";
    const labelSelections = isDet ? detModelLabelSelections : clsModelLabelSelections;
    const selectedSet = isDet ? selectedDetModels : selectedClsModels;

    selectedSet.forEach(name => {
      labelSelections[name] = new Set();
    });

    renderModalLabelsPanel();
    rebuildLabelCards(currentModalKind);
  });

  document.getElementById("det-select-btn").addEventListener("click", () => openModelModal("det"));
  document.getElementById("cls-select-btn").addEventListener("click", () => openModelModal("cls"));
}

// ---------- zoom & pan per card ----------

function initZoomForCard(cardId) {
  const wrapper = document.getElementById("zoom-wrapper-" + cardId);
  const inner = document.getElementById("zoom-inner-" + cardId);
  const label = document.getElementById("zoom-label-" + cardId);

  if (!wrapper || !inner || !label) return;

  let scale = 1.0;
  zoomStates[cardId] = scale;

  function setZoom(newScale) {
    scale = Math.min(3, Math.max(0.5, newScale));
    zoomStates[cardId] = scale;
    inner.style.transform = "scale(" + scale + ")";
    label.textContent = Math.round(scale * 100) + "% — Ctrl + scroll to zoom, drag to pan";
  }

  wrapper.addEventListener("wheel", e => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9; // down => zoom in
    setZoom(scale * factor);
  }, { passive: false });

  let isPanning = false;
  let startX = 0;
  let startY = 0;
  let scrollLeftStart = 0;
  let scrollTopStart = 0;

  wrapper.addEventListener("mousedown", e => {
    isPanning = true;
    wrapper.classList.add("panning");
    startX = e.pageX - wrapper.offsetLeft;
    startY = e.pageY - wrapper.offsetTop;
    scrollLeftStart = wrapper.scrollLeft;
    scrollTopStart = wrapper.scrollTop;
  });

  window.addEventListener("mouseup", () => {
    isPanning = false;
    wrapper.classList.remove("panning");
  });

  wrapper.addEventListener("mousemove", e => {
    if (!isPanning) return;
    e.preventDefault();
    const x = e.pageX - wrapper.offsetLeft;
    const y = e.pageY - wrapper.offsetTop;
    const walkX = (startX - x);
    const walkY = (startY - y);
    wrapper.scrollLeft = scrollLeftStart + walkX;
    wrapper.scrollTop = scrollTopStart + walkY;
  });

  setZoom(1.0);
}

// ---------- EXPORT HELPERS ----------

function sanitizeFileName(name) {
  return name.replace(/[^a-zA-Z0-9_\-\.]/g, "_");
}

async function captureCardAsPng(cardElement) {
  if (!window.html2canvas) {
    alert("html2canvas not loaded; cannot export.");
    return null;
  }
  const canvas = await html2canvas(cardElement, {
    backgroundColor: "#020617",
    scale: 2,
    useCORS: true,
  });
  return canvas.toDataURL("image/png");
}

function dataURLToBlob(dataURL) {
  const parts = dataURL.split(",");
  const mime = parts[0].match(/:(.*?);/)[1];
  const bstr = atob(parts[1]);
  const u8arr = new Uint8Array(bstr.length);
  for (let i = 0; i < bstr.length; i++) {
    u8arr[i] = bstr.charCodeAt(i);
  }
  return new Blob([u8arr], { type: mime });
}

function downloadDataUrl(dataUrl, filename) {
  const blob = dataURLToBlob(dataUrl);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function downloadCurrentResultImageCard() {
  if (!latestResults.length) return;
  const card = document.getElementById("result-card");
  if (!card) return;

  const dataUrl = await captureCardAsPng(card);
  if (!dataUrl) return;

  const result = latestResults[currentResultIndex];
  const base = sanitizeFileName((result.filename || `image_${currentResultIndex + 1}`));
  downloadDataUrl(dataUrl, base.replace(/\.[^.]+$/, "") + "_result_card.png");
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function downloadAllResultCards() {
  if (!latestResults.length) return;

  const originalIndex = currentResultIndex;

  for (let i = 0; i < latestResults.length; i++) {
    currentResultIndex = i;
    renderCurrentResult();
    await wait(200);

    const card = document.getElementById("result-card");
    if (!card) continue;
    const dataUrl = await captureCardAsPng(card);
    if (!dataUrl) continue;

    const result = latestResults[i];
    const base = sanitizeFileName((result.filename || `image_${i + 1}`));
    downloadDataUrl(dataUrl, base.replace(/\.[^.]+$/, "") + "_result_card.png");
  }

  currentResultIndex = originalIndex;
  renderCurrentResult();
}

function downloadAnnotatedCurrent() {
  if (!latestResults.length) return;
  const result = latestResults[currentResultIndex];
  if (!result.annotated_image_base64) return;
  const base = sanitizeFileName(result.filename || `image_${currentResultIndex + 1}`);
  const dataUrl = "data:image/png;base64," + result.annotated_image_base64;
  downloadDataUrl(dataUrl, base.replace(/\.[^.]+$/, "") + "_annotated.png");
}

function downloadCropsCurrent() {
  if (!latestResults.length) return;
  const result = latestResults[currentResultIndex];
  const base = sanitizeFileName(result.filename || `image_${currentResultIndex + 1}`);

  if (!result.detections || !result.detections.length) {
    alert("No detections / crops for this image.");
    return;
  }

  result.detections.forEach((det, idx) => {
    if (!det.crop_image_base64) return;
    const dataUrl = "data:image/png;base64," + det.crop_image_base64;
    const labelSafe = det.class_name ? sanitizeFileName(det.class_name) : "crop";
    const filename = base.replace(/\.[^.]+$/, "") + `_crop_${idx + 1}_${labelSafe}.png`;
    downloadDataUrl(dataUrl, filename);
  });
}

// ---------- rendering results ----------

function renderCurrentResult() {
  const resultsContainer = document.getElementById("results-container");
  resultsContainer.innerHTML = "";

  if (!latestResults.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "No results yet. Run inference first.";
    resultsContainer.appendChild(p);
    return;
  }

  if (currentResultIndex < 0) currentResultIndex = 0;
  if (currentResultIndex >= latestResults.length) currentResultIndex = latestResults.length - 1;

  const result = latestResults[currentResultIndex];
  const idx = currentResultIndex;
  const total = latestResults.length;
  const cardId = "img" + idx;

  const card = document.createElement("div");
  card.className = "image-card";
  card.id = "result-card";

  const header = document.createElement("div");
  header.className = "image-card-header";

  const left = document.createElement("div");
  left.className = "image-card-header-left";
  const hLine = document.createElement("span");
  hLine.className = "muted";
  hLine.textContent = "Annotated image — " + result.filename;
  left.appendChild(hLine);

  const zoomLabel = document.createElement("span");
  zoomLabel.className = "zoom-indicator";
  zoomLabel.id = "zoom-label-" + cardId;
  zoomLabel.textContent = "100% — Ctrl + scroll to zoom, drag to pan";
  left.appendChild(zoomLabel);

  header.appendChild(left);

  const nav = document.createElement("div");
  nav.className = "image-nav";

  const prevBtn = document.createElement("button");
  prevBtn.className = "btn btn-small";
  prevBtn.textContent = "◀";
  prevBtn.title = "Previous image";
  prevBtn.disabled = (idx === 0);
  prevBtn.onclick = () => {
    if (currentResultIndex > 0) {
      currentResultIndex--;
      renderCurrentResult();
    }
  };

  const idxSpan = document.createElement("span");
  idxSpan.className = "image-nav-index";
  idxSpan.textContent = (idx + 1) + " / " + total;

  const nextBtn = document.createElement("button");
  nextBtn.className = "btn btn-small";
  nextBtn.textContent = "▶";
  nextBtn.title = "Next image";
  nextBtn.disabled = (idx === total - 1);
  nextBtn.onclick = () => {
    if (currentResultIndex < latestResults.length - 1) {
      currentResultIndex++;
      renderCurrentResult();
    }
  };

  const saveAnnBtn = document.createElement("button");
  saveAnnBtn.className = "btn btn-small";
  saveAnnBtn.textContent = "🖼";
  saveAnnBtn.title = "Download annotated image";
  saveAnnBtn.onclick = () => downloadAnnotatedCurrent();

  const saveCropsBtn = document.createElement("button");
  saveCropsBtn.className = "btn btn-small";
  saveCropsBtn.textContent = "🧩";
  saveCropsBtn.title = "Download crop images";
  saveCropsBtn.onclick = () => downloadCropsCurrent();

  const saveCardBtn = document.createElement("button");
  saveCardBtn.className = "btn btn-small";
  saveCardBtn.textContent = "💾";
  saveCardBtn.title = "Download result card screenshot";
  saveCardBtn.onclick = () => downloadCurrentResultImageCard();

  const saveAllCardsBtn = document.createElement("button");
  saveAllCardsBtn.className = "btn btn-small";
  saveAllCardsBtn.textContent = "📥";
  saveAllCardsBtn.title = "Download all result cards";
  saveAllCardsBtn.onclick = () => downloadAllResultCards();

  nav.appendChild(prevBtn);
  nav.appendChild(idxSpan);
  nav.appendChild(nextBtn);
  nav.appendChild(saveAnnBtn);
  nav.appendChild(saveCropsBtn);
  nav.appendChild(saveCardBtn);
  nav.appendChild(saveAllCardsBtn);

  header.appendChild(nav);
  card.appendChild(header);

  const wrapper = document.createElement("div");
  wrapper.className = "image-zoom-wrapper";
  wrapper.id = "zoom-wrapper-" + cardId;
  const inner = document.createElement("div");
  inner.className = "image-zoom-inner";
  inner.id = "zoom-inner-" + cardId;
  const img = document.createElement("img");
  img.className = "result-img";
  img.src = "data:image/png;base64," + result.annotated_image_base64;
  inner.appendChild(img);
  wrapper.appendChild(inner);
  card.appendChild(wrapper);

  const cropHeader = document.createElement("div");
  cropHeader.className = "crop-header";
  const chLeft = document.createElement("span");
  chLeft.className = "muted";
  chLeft.textContent = "Cropped detections";
  const chRight = document.createElement("span");
  chRight.className = "tag";
  chRight.textContent = result.detections.length + " detections";
  cropHeader.appendChild(chLeft);
  cropHeader.appendChild(chRight);
  card.appendChild(cropHeader);

  const grid = document.createElement("div");
  grid.className = "crop-grid";

  if (!result.detections.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "No detections for this image (after filters).";
    grid.appendChild(p);
  } else {
    result.detections.forEach(det => {
      const detCard = document.createElement("div");
      detCard.className = "det-card";

      if (det.crop_image_base64) {
        const imgWrap = document.createElement("div");
        imgWrap.className = "det-card-img-wrapper";
        const cropImg = document.createElement("img");
        cropImg.src = "data:image/png;base64," + det.crop_image_base64;
        imgWrap.appendChild(cropImg);
        detCard.appendChild(imgWrap);
      }

      const title = document.createElement("div");
      title.className = "det-title";
      title.textContent = det.class_name;
      detCard.appendChild(title);

      const detScore = (det.score * 100).toFixed(1);
      const detLine = document.createElement("div");
      detLine.className = "muted";
      detLine.textContent = "Detection: " + detScore + "%";
      detCard.appendChild(detLine);

      const detBar = document.createElement("div");
      detBar.className = "progress";
      const detInner = document.createElement("div");
      detInner.className = "progress-inner";
      detInner.style.width = detScore + "%";
      detBar.appendChild(detInner);
      detCard.appendChild(detBar);

      if (det.classifications && det.classifications.length) {
        const clsHeader = document.createElement("div");
        clsHeader.className = "muted";
        clsHeader.style.marginTop = "4px";
        clsHeader.textContent = "Classification details:";
        detCard.appendChild(clsHeader);

        det.classifications.forEach(cls => {
          const clsLine = document.createElement("div");
          clsLine.className = "muted";

          if (cls.label) {
            const clsScore = (cls.score * 100).toFixed(1);
            clsLine.textContent =
              cls.model_name + ": " + cls.label + " (" + clsScore + "%)";
            detCard.appendChild(clsLine);

            const clsBar = document.createElement("div");
            clsBar.className = "progress";
            const clsInner = document.createElement("div");
            clsInner.className = "progress-inner";
            clsInner.style.width = clsScore + "%";
            clsBar.appendChild(clsInner);
            detCard.appendChild(clsBar);
          } else if (cls.score !== null && cls.score !== undefined) {
            const clsScore = (cls.score * 100).toFixed(1);
            clsLine.textContent =
              cls.model_name + ": below threshold (" + clsScore + "%)";
            detCard.appendChild(clsLine);

            const clsBar = document.createElement("div");
            clsBar.className = "progress";
            const clsInner = document.createElement("div");
            clsInner.className = "progress-inner";
            clsInner.style.width = clsScore + "%";
            clsBar.appendChild(clsInner);
            detCard.appendChild(clsBar);
          } else {
            clsLine.textContent = cls.model_name + ": no result";
            detCard.appendChild(clsLine);
          }
        });
      }

      grid.appendChild(detCard);
    });
  }

  card.appendChild(grid);
  resultsContainer.appendChild(card);

  initZoomForCard(cardId);
}

// ---------- run / clear ----------

function getSelectedDetLabels() {
  const labels = [];
  selectedDetModels.forEach(model => {
    const set = detModelLabelSelections[model];
    if (set) labels.push(...set);
  });
  return Array.from(new Set(labels));
}

function getSelectedClsLabels() {
  const labels = [];
  selectedClsModels.forEach(model => {
    const set = clsModelLabelSelections[model];
    if (set) labels.push(...set);
  });
  return Array.from(new Set(labels));
}

async function runInference() {
  const status = document.getElementById("status-text");
  const resultsContainer = document.getElementById("results-container");
  resultsContainer.innerHTML = "";

  if (!currentFiles.length) {
    alert("Select or drop at least one image.");
    return;
  }

  const detModels = Array.from(selectedDetModels);
  const clsModels = Array.from(selectedClsModels);

  if (!detModels.length && !clsModels.length) {
    alert("Select at least one detection or classification model.");
    return;
  }

  const detLabels = getSelectedDetLabels();
  const clsLabels = getSelectedClsLabels();

  const detThresh = document.getElementById("det-thresh").value / 100;
  const clsThresh = document.getElementById("cls-thresh").value / 100;

  const formData = new FormData();
  currentFiles.forEach(f => formData.append("files", f));
  formData.append("detection_models", detModels.join(","));
  formData.append("classification_models", clsModels.join(","));
  formData.append("det_thresh", detThresh.toString());
  formData.append("cls_thresh", clsThresh.toString());
  formData.append("det_labels", detLabels.join(","));
  formData.append("cls_labels", clsLabels.join(","));

  let mode;
  if (detModels.length && clsModels.length) {
    mode = "detection + classification";
  } else if (detModels.length) {
    mode = "detection-only";
  } else {
    mode = "classification-only";
  }

  status.textContent = `Running (${mode}) on ${currentFiles.length} image(s)...`;
  const res = await fetch("/predict", { method: "POST", body: formData });

  if (!res.ok) {
    const txt = await res.text();
    alert("Error: " + txt);
    status.textContent = "Error";
    return;
  }

  const data = await res.json();
  latestResults = data.results || [];
  currentResultIndex = 0;

  status.textContent = `Done (${mode}) — ${latestResults.length} image(s)`;
  renderCurrentResult();
}

function clearAll() {
  currentFiles = [];
  latestResults = [];
  currentResultIndex = 0;
  document.getElementById("image-input").value = "";
  document.getElementById("results-container").innerHTML = "";
  document.getElementById("status-text").textContent = "Idle";
  const fileLabel = document.getElementById("drop-text-file");
  fileLabel.style.display = "none";
}

// ---------- boot ----------

function initApp() {
  document.getElementById("run-btn").addEventListener("click", runInference);
  document.getElementById("clear-btn").addEventListener("click", clearAll);

  initDropzone();
  initThresholdSliders();
  initModalControls();
  loadModels();
  renderCurrentResult();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
