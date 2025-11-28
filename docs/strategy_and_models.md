# 解析策略与高级模型选型指南

ZenParse 的 PDF 解析分两层：

- **基础解析策略**：选择底层引擎（`pdfplumber` / `unstructured` / `hybrid_table` 等），负责"把 PDF 拆成文本 + 表格元素"。

- **高级表格检测（可选）**：在 `hybrid_table` 策略下，对 **低质量表格 / 缺表页面** 走更强的检测模型（DocLayout-YOLO 或 Detectron2），提升表格定位和结构质量。

本指南说明如何在 `config.yaml` 中配置这些策略，以及基础/高级模型的优劣和取舍。

---

## 1. 基础解析策略（`pdf_parsing.strategies`）

`config.yaml` 中：

```yaml
pdf_parsing:
  # 解析策略：
  # - auto         自动按页面"文本/图像比例"选择解析引擎
  # - pdfplumber   只使用 pdfplumber（速度快，对电子版财报友好）
  # - unstructured 使用 unstructured 高分辨率解析（含 OCR，适合扫描版，较慢）
  # - hybrid_table 优先做表格检测 + 表格区域外文本提取
  strategies:
    - hybrid_table
```

### 策略说明

**`pdfplumber`**
- **特点**：速度快、依赖轻，对数字版财报文本/表格支持很好。
- **适用**：你现在的大部分上市公司年报（电子版 PDF）。

**`unstructured`**
- **特点**：支持 OCR、复杂布局，但依赖重、速度慢。
- **适用**：纯扫描版 PDF 或极复杂布局。

**`hybrid_table`**
- **特点**：先用表格处理器抽取表格（可选高级模型），再在"表格以外区域"提取文本。
- **适用**：对表格结构质量要求较高的场景（你当前配置即为此模式）。

**`auto`**
- **特点**：先用 pdfplumber 粗判"数字/扫描/混合"，再选择合适引擎。
- **适用**：一个目录里混合了数字/扫描版，且不想手动区分。

### 建议

- **绝大多数数字财报**：`pdf_parsing.strategies: [pdfplumber]` 或 `[hybrid_table]`。
- **有一部分扫描/图片版**：`[auto]` 或 `[hybrid_table]`。
- **不在意表格结构，只要文本**：`[pdfplumber]` 即可。

---

## 2. 高级表格检测（`table_extraction`）

在 `hybrid_table` 策略下，`ChineseFinancialPDFParser` 会调用 `HybridTableProcessor` 进行表格抽取。

### HybridTableProcessor 的工作流程（质量分级策略）

1. **先用 pdfplumber 全文扫描**，抽取所有表格并计算质量分数。
2. **如果**：
   - 存在表格，但质量分数 **低于阈值**（`quality_threshold`），或
   - 整份 PDF 没抽到表格（缺表页）
   → 则认为"有问题"，触发高级模型做补充检测。
3. **高级模型**（DocLayout-YOLO / Detectron2）只在上述"有问题"的情况下运行，避免对所有页面都跑重模型。
4. **高级模型检测出表格 bbox 后**，用 pdfplumber 在该 bbox 内重新抽取结构化表格。
5. **将 pdfplumber 和高级模型结果按 IoU 合并**，保留质量更高的版本。

---

## 3. `table_extraction` 配置说明

当前实现对应的 `config.yaml` 示例：

```yaml
table_extraction:
  advanced_model: doclayout_yolo       # doclayout_yolo / detectron2 / none
  model_path: null                     # 自定义权重路径，null 表示从 HuggingFace 自动下载
  model_repo: "juliozhao/DocLayout-YOLO-DocStructBench"  # HuggingFace 模型仓库
  model_filename: "doclayout_yolo_docstructbench_imgsz1024.pt"  # 模型文件名
  # 其他可选模型：
  # model_repo: "juliozhao/DocLayout-YOLO-DocLayNet"
  # model_repo: "juliozhao/DocLayout-YOLO-D4LA"
  # model_repo: "juliozhao/DocLayout-YOLO-DocSynth300K-pretrain"
  quality_threshold: 0.65              # 表格质量低于此值时，触发高级模型
  detection_conf: 0.30                 # YOLO 置信度阈值
  detection_iou: 0.50                  # YOLO NMS IoU 阈值
  bbox_padding_ratio: 0.02             # YOLO 框裁剪前向外扩张比例
  merge_iou_threshold: 0.55            # 表格去重合并 IoU 阈值
  render_dpi: 150                      # 页面渲染 DPI（用于生成检测输入图像）
  skip_ocr_for_digital: true           # 数字 PDF 下，文本解析阶段跳过 OCR 兜底
  ocr_trigger_char_threshold: 10       # bbox 内字符数低于此，将被标记为潜在 OCR 候选
```

### 模型路径配置说明

**自动下载（推荐）**：
- 设置 `model_path: null`，代码会自动从 HuggingFace 下载模型
- 首次运行会下载到 `~/.cache/zenparse/models/`，后续运行直接使用缓存
- 需要安装 `huggingface_hub`：`pip install huggingface_hub`

**手动指定本地路径**：
- 设置 `model_path: "./models/doclayout_yolo.pt"` 使用本地模型文件
- 适用于已手动下载模型或使用自定义权重

### 3.1 `advanced_model`

- **`doclayout_yolo`**：使用 DocLayout-YOLO（ultralytics YOLO）做表格检测。
- **`detectron2`**：使用 detectron2 量化模型（通过 unstructured_inference 的接口）。
- **`none`**：完全禁用高级模型，只使用 pdfplumber。

### 3.2 质量触发阈值（`quality_threshold`）

用于 `_is_low_quality_table`：
- `table.quality_score < quality_threshold` 认为是"低质量表格"。
- 或 pdfplumber 完全没抽到表格 → 视为"缺表"，触发高级模型。

**调优建议**：
- **调大**（如 0.7）：更敏感，更多表格页会触发高级模型。
- **调小**（如 0.5）：更保守，只有很差的表格才触发。

### 3.3 检测阈值（`detection_conf`, `detection_iou`）

- **`detection_conf`**：YOLO 检测结果最低置信度（0.25–0.4 较常用）。
- **`detection_iou`**：YOLO 内部 NMS 的 IoU 阈值。

这两个参数只影响 YOLO 检测本身，不影响表格质量分数。

### 3.4 合并阈值（`merge_iou_threshold`）

用于 `_merge_tables` 中判断 pdfplumber 表格与 YOLO 表格是否"重合"：
- IoU > `merge_iou_threshold` → 视为同一张表，只保留质量更高的一个。
- 默认 0.55，一般能兼顾"重合"和"相邻两张表"。

### 3.5 bbox 扩张与渲染 DPI

- **`bbox_padding_ratio`**：避免 YOLO 框太紧导致边缘字符被裁掉，默认 2%。
- **`render_dpi`**：`page.to_image` 使用的 DPI，DPI 越高图像越清晰但越慢；150 是一个折中。

### 3.6 OCR 相关（当前实现的状态）

**`skip_ocr_for_digital`**：
- 用在 **文本解析** 阶段：如果 PDF 被判定为数字版且此项为 `true`，则 `_extract_text_excluding_tables` 里不会在 pdfplumber 失败时启用 hi_res OCR。

**`ocr_trigger_char_threshold`**：
- 在 YOLO 表格抽取里，只是作为 **标记字段**（`metadata.needs_ocr = True`），表示这个表格内字符数很少，未来可以作为 OCR 候选。
- 当前代码尚未在表格层面自动触发 OCR，只是标记；真正的 OCR 兜底仍在文本解析 `_extract_text_excluding_tables` 中按页面级别执行。

---

## 4. 基础模型 vs 高级模型：优势与取舍

### 4.1 pdfplumber（基础）

**优点**
- 速度快，内存占用小。
- 对数字版财报文本/普通表格支持稳定。

**缺点**
- 对无线框表格、严重错位的复杂表格、扫描图像上的表格识别能力较弱。

**适合**
- 你的大部分电子版财报。
- 对表格结构没有"极端严格"要求的批量任务。

### 4.2 DocLayout-YOLO（高级）

**优点**
- 在复杂表格、无线框表格上有更好的区域定位能力。
- 支持 M2（MPS）/CUDA 加速，相对 Detectron2 更轻。
- 能弥补 pdfplumber 对"问题页"的漏检。
- **支持自动从 HuggingFace 下载**，无需手动配置。

**缺点**
- 需要首次下载模型（约几百MB），需要网络连接。
- 仅在 `hybrid_table` 策略下启用。

**适合**
- 数字财报中"少量难页"的补救（当前质量分级策略正是如此）。
- 不希望所有页都跑重模型，只对有问题的页做增强。

### 4.3 Detectron2（高级）

**优点**
- 精度高，在复杂布局上表现好。

**缺点**
- 依赖重，加载+推理慢。
- 对 Mac M 系列的支持不如 YOLO/MPS + ONNX 轻量方案。

**适合**
- 小批量、高精度需求的实验性场景。

---

## 5. 推荐配置示例

### 5.1 数字财报，注重速度

```yaml
pdf_parsing:
  strategies:
    - pdfplumber

table_extraction:
  advanced_model: none
```

### 5.2 数字财报，注重表格质量（当前推荐）

```yaml
pdf_parsing:
  strategies:
    - hybrid_table

table_extraction:
  advanced_model: doclayout_yolo
  model_path: null  # 自动从 HuggingFace 下载
  model_repo: "juliozhao/DocLayout-YOLO-DocStructBench"
  model_filename: "doclayout_yolo_docstructbench_imgsz1024.pt"
  quality_threshold: 0.65
  detection_conf: 0.30
  detection_iou: 0.50
  merge_iou_threshold: 0.55
  skip_ocr_for_digital: true
```

### 5.3 混合/扫描版，允许适度变慢

```yaml
pdf_parsing:
  strategies:
    - hybrid_table

table_extraction:
  advanced_model: doclayout_yolo
  model_path: null  # 自动从 HuggingFace 下载
  model_repo: "juliozhao/DocLayout-YOLO-DocStructBench"
  model_filename: "doclayout_yolo_docstructbench_imgsz1024.pt"
  quality_threshold: 0.50  # 降低阈值，更积极触发
  detection_conf: 0.30
  detection_iou: 0.50
  skip_ocr_for_digital: false  # 允许文本阶段触发 OCR
```

---

## 6. 当前实现的小结

- ✅ **质量分级已实现**：pdfplumber → 质量评估 → 低质/缺表才跑高级模型。
- ✅ **高级模型已接入**：DocLayout-YOLO / Detectron2 的分支已接好，由 `advanced_model` 控制。
- ✅ **HuggingFace 自动下载已实现**：设置 `model_path: null` 即可自动下载并缓存模型。
- ✅ **配置已生效**：`table_extraction` 中的阈值和行为与本文档保持一致。
- ⚠️ **OCR 在文本层面按需触发**，表格层面目前只做"可能需要 OCR"的标记，不自动跑。

