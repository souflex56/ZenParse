# ZenParse - 中文财报 PDF 智能分块器

ZenParse 是一个专门面向 **中文上市公司财报 PDF** 的预处理工具，用来把复杂的 PDF 文档解析成 **结构化的「父子分块 + 表格组」JSON 数据**，方便后续做向量检索、问答系统、风控分析等。

核心目标：**尽量不丢信息，又切成适合模型使用的小块**。

---

## 适用场景

- 为中文年报/半年报/季报搭建 **向量检索 / RAG 知识库**
- 需要从 PDF 中 **精准保留表格 + 标题 + 注释** 的分析类应用
- 对财报进行 **高质量切分和打分**，过滤掉噪声内容

---

## 功能特点（重点）

- **表格上下文识别（TableGroup）**
  - 自动识别表格的 **标题、单位说明、注释**
  - 把「标题 + 表格主体 + 注释」组合成一个 `TableGroup`
  - 尝试将表格内容转为 **标准 Markdown 表格**，保留结构，便于前端展示或检索
  
- **混合表格处理与高级检测（可选）**
  - 默认使用 `pdfplumber` 快速提取表格
  - 在 `hybrid_table` 策略下，对"低质量表格/缺表页面"按需触发 DocLayout-YOLO 或 Detectron2 做高级表格检测
  - 自动合并 pdfplumber 与高级模型的结果，保留质量更高的表格
  - 支持从 HuggingFace 自动下载模型，首次运行自动完成配置。
  - 详细配置说明和选型建议请参考 [`docs/strategy_and_models.md`](docs/strategy_and_models.md)。

- **父子分块架构（SmartChunker）**
  - **父块（ParentChunk）**：尽量包含完整语境的一大段（保持上下文完整）
  - **子块（ChildChunk）**：在父块内部按滑动窗口进一步细分（方便精准召回）
  - 支持语义感知、固定大小、滑动窗口等策略，自动处理重叠和边界
  
- **会计准则自适应模块（可选 Standards Adapter）**
  - 针对 2017–2021 年新收入/金融工具/租赁准则落地后，新旧报表之间“科目改名、口径变了、数字不好比”的问题，提供一层自动适配
  - 能识别当前报表是按新准则还是旧准则披露，并把“叫法不同但本质相同”的项目对齐（例如预收款项 vs 合同负债、应收票据 vs 应收款项融资）
  - 提供统一新旧准则口径的“标准化指标”（附带简短说明），方便做 5–10 年跨期分析或喂给大模型使用；详情见 `docs/standards_adapter.md`

- **质量评估与财报特征**
  - 每个分块都有 `quality_score`、信息密度、连贯性等指标
  - 自动检测是否包含财务数据，识别会计科目、时间期间等关键字段
  - 提供整体统计信息（来自 `ProcessingStats`），统计页数、块数、质量分布、处理耗时等

- **多引擎 PDF 解析**
  - 支持 `unstructured` / `pdfplumber` / `pymupdf` 等引擎
  - 通过 `engine_checker` 自动检测可用引擎，并按优先级选择
  - 默认配置使用 `pdfplumber`，稳定、部署简单

- **工程化能力**
  - 使用 `loguru` 做结构化日志，默认日志输出到 `logs/`
  - 简单的 `config.yaml` 配置，开箱即用
  - 提供 `runner.py` 命令行工具，支持批量处理和文件列表
  - 支持跳过已处理文件（默认不覆盖已有输出），单个文件失败不会影响整体批处理

---

## 目录结构（核心部分）

- `zenparse/`
  - `pipeline.py`：端到端管线（PDF → 元素 → 表格组 → 父子分块）
  - `data/pdf_parser.py`：PDF 解析（多引擎，中文财报优化）
  - `data/context_identifier.py`：表格上下文识别，生成 `TableGroup`
  - `data/smart_chunker.py`：智能分块器，生成父块/子块
  - `data/models.py`：数据模型定义（`DocumentElement`、`Chunk`、`TableGroup` 等）
  - `core/logger.py`：统一日志系统
  - `core/engine_checker.py`：解析引擎检测工具
- `runner.py`：命令行批量处理工具
- `config.yaml`：默认配置
- `input/`：示例输入目录
- `output/`：分块结果 JSON 输出目录

---

## 快速开始

### 1. 环境准备

```bash
git clone <your-repo-url>
cd ZenParse

python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

建议使用 Python 3.10+（最低 3.8），并在有一定内存的环境下运行（处理大体量 PDF 时更稳）。

### 2. 准备 PDF 文件

默认示例目录：

- `input/pdfs/`：放待处理的 PDF 文件
- `input/pdf_file_list/pdf_list_test.txt`：示例文件名列表（每行一个）

你也可以使用自己的目录，只要在命令行参数中指定即可。

### 3. 使用命令行批量处理

**（1）处理一个目录下所有 PDF**

```bash
python runner.py -i input/pdfs -o output
```

- 会扫描 `input/pdfs` 目录下的所有 `.pdf`
- 对每个 PDF 生成 `<原文件名>_chunks.json` 到 `output/` 目录

**（2）处理单个 PDF 文件**

```bash
python runner.py -i input/pdfs/某公司年报.pdf -o output
```

**（3）使用文件列表（推荐批量精确控制）**

```bash
python runner.py \
  --file-list input/pdf_file_list/pdf_list_test.txt \
  --search-dirs input/pdfs input/pdfs_raw \
  --output output
```

- `--file-list`：指定一个文本文件，每行一个文件名（支持 `#` 注释）
- `--search-dirs`：在这些目录中搜索对应的 PDF 文件

**（4）直接指定多个文件名**

```bash
python runner.py \
  --files 2020年年报.pdf,2019年年报.pdf \
  --search-dirs input/pdfs input/pdfs_raw \
  --output output
```

**（5）避免覆盖输出（添加时间戳）**

```bash
python runner.py -i input/pdfs -o output --timestamp
```

输出文件会变成：`xxx_chunks_20251125_130003.json` 这种格式。

---

## 命令行参数

`runner.py` 支持丰富的命令行参数：

| 参数 | 简写 | 说明 |
| :--- | :--- | :--- |
| `--input` | `-i` | 输入 PDF 文件或目录，可重复使用多次 |
| `--output` | `-o` | 输出 JSON 目录（默认 `output/`） |
| `--config` | `-c` | 指定配置文件路径（默认 `config.yaml`） |
| `--timestamp` | `-t` | 在输出文件名后添加时间戳（避免覆盖已有文件） |
| `--file-list` | `-l` | 从文本文件读取待处理文件名列表（每行一个） |
| `--files` | `-f` | 逗号分隔的文件名列表（如 `a.pdf,b.pdf`） |
| `--search-dirs` | `-s` | 搜索 PDF 的目录，可提供多个目录 |

---

## 配置说明（`config.yaml`）

示例：

```yaml
pdf_parsing:
  # 解析策略：
  # - auto         自动按优先级选择可用引擎
  # - pdfplumber   只使用 pdfplumber（速度快、对电子版财报效果好）
  # - unstructured 使用 unstructured 的高分辨率解析（适合复杂/扫描版，但较慢）
  # - hybrid_table 启用高级表格检测与结构化（最慢，仅在特别需要表格结构时建议开启）
  strategies:
    - pdfplumber

context:
  context_before: 10
  context_after: 3

chunking:
  parent_size: 4000
  child_size: 1200
  overlap: 200
  min_chunk_size: 50
  min_quality_score: 0.3
```

关键字段解释：

- `pdf_parsing.strategies`
  - 支持：`auto` / `pdfplumber` / `unstructured` / `hybrid_table`
  - 建议：一般用 `pdfplumber`；需要更强表格/扫描版支持时可以尝试 `unstructured` 或 `hybrid_table`（更占资源）
- `context.context_before` / `context.context_after`
  - 表格上下文搜索范围（前后多少个元素），主要用于找到「表标题」「单位说明」「注释」
- `chunking.parent_size`
  - 父块目标大小（字符数），越大上下文越完整，但后续召回粒度会变粗
- `chunking.child_size` / `chunking.overlap`
  - 子块大小与重叠大小，用于滑动窗口切分  
  - 一般来说：父块偏大、子块偏小且有适当重叠，是比较稳妥的配置
- `min_chunk_size` / `min_quality_score`
  - 过滤过短或质量过低的分块，减少噪声

### 表格抽取与高级模型配置（`table_extraction`）

```yaml
table_extraction:
  advanced_model: doclayout_yolo       # doclayout_yolo / detectron2 / none
  model_path: null                     # 自定义权重路径，null 表示从 HuggingFace 自动下载
  model_repo: "juliozhao/DocLayout-YOLO-DocStructBench"  # HuggingFace 模型仓库
  model_filename: "doclayout_yolo_docstructbench_imgsz1024.pt"  # 模型文件名
  quality_threshold: 0.65              # 表格质量低于此值触发高级检测
  detection_conf: 0.30                 # YOLO 置信度阈值
  detection_iou: 0.50                  # YOLO NMS IoU 阈值
  merge_iou_threshold: 0.55            # 表格去重合并 IoU 阈值
  bbox_padding_ratio: 0.02             # YOLO 框裁剪前的扩张比例
  render_dpi: 150                      # 检测用页面渲染 DPI
  skip_ocr_for_digital: true           # 数字 PDF 下跳过 OCR 兜底
  ocr_trigger_char_threshold: 10       # bbox 内字符数低于此值时认为"文本过少"
```

**建议**：
- **大部分数字财报**：`advanced_model: doclayout_yolo` + 合理设置 `quality_threshold`，`model_path: null` 自动下载模型。
- **若只追求速度**：`advanced_model: none`，只用 pdfplumber。
- **若有扫描版/难页**，可将 `skip_ocr_for_digital` 设为 `false`，允许文本阶段触发 OCR 兜底。

详细配置说明和选型建议请参考 [`docs/strategy_and_models.md`](docs/strategy_and_models.md)。

---

## 输出结果说明

每个 PDF 会对应一个 JSON 文件，例如：

```text
output/2020-01-21__...__年度报告_chunks.json
```

结构大致如下（简化示意）：

```json
{
  "source": "xxx.pdf",
  "overview": {
    "source_file": "xxx.pdf",
    "counts": {
      "parents": 12,
      "children": 48,
      "table_groups": 5,
      "pages": [[1, 20]],
      "page_count": 20
    },
    "quality": {
      "high": 30,
      "medium": 20,
      "low": 10
    },
    "avg_child_per_parent": 4.0,
    "parent_previews": [
      {
        "chunk_id": "abcd1234",
        "chars": 3500,
        "preview": "本公司董事会、监事会及全体董事、监事、高级管理人员保证本报告内容真实、准确、完整...",
        "page_numbers": [1, 2],
        "child_count": 4
      }
    ]
  },
  "parents": [
    {
      "chunk_id": "abcd1234",
      "content": "……",
      "chunk_type": "text_group",
      "child_ids": ["efgh5678", "..."],
      "metadata": {
        "page_numbers": [1, 2],
        "char_count": 3521,
        "report_type": "annual_report",
        "fiscal_year": 2019
      },
      "contains_financial_data": true,
      "financial_indicators": ["营业收入", "净利润"],
      "quality_score": 0.87
    }
  ],
  "children": [
    {
      "chunk_id": "efgh5678",
      "parent_id": "abcd1234",
      "content": "……",
      "start_char": 0,
      "end_char": 1200,
      "page_number": 1,
      "contains_financial_data": true
    }
  ]
}
```

可以直接把 `parents` 和 `children` 用于向量化，`parent_id` / `child_ids` 则用于做父子联动召回。  
表格相关内容会以 `chunk_type="table_group"` 的形式出现在分块中，同时在 `overview.counts.table_groups` 中统计表格组数量。

---

## 在代码中使用 ZenPipeline

如果你希望在自己的 Python 代码中直接调用：

```python
from zenparse import ZenPipeline

pipeline = ZenPipeline("config.yaml")
parents, children, table_groups = pipeline.process(
    "input/pdfs/2020-01-21__某公司__年度报告.pdf"
)

print(len(parents), len(children), len(table_groups))
```

在此基础上，你可以：

- 将 `parents` / `children` 向量化，构建检索库
- 对 `table_groups` 做进一步结构化处理或可视化
- 使用 `metadata` 中的页码、财报年份、公司代码等进行过滤

---

## 注意事项

- 目前优化重点是 **中文 A 股财报类 PDF**，对其他类型文档也能工作，但效果可能不如财报场景
- 使用 `unstructured` / `detectron2` 等高级特性时，对环境和依赖版本有一定要求，推荐先在小规模样本上测试
- PDF 质量本身（扫描清晰度、排版混乱程度）会直接影响解析效果

# 声明
本项目的开发者并非会计专业出身，所有财务相关理解主要来自个人投资过程中的阅读和自学（包括但不限于上述书籍和公开资料）。  
代码和文档难免存在理解偏差或不严谨之处，如你在使用中发现任何错误或有更好的口径建议，非常欢迎指正：

- 邮件：`souflex@163.com`
- 或在仓库中提交 issue

你的反馈会直接帮助这套工具变得更可靠，在此提前致谢。
