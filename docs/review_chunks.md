# 分块结果审阅指南（preview_chunks 使用说明）

本指南介绍如何使用 `scripts/preview_chunks.py` 将 `_chunks.json` 结果导出为 Excel，辅助人工检查分块质量和表格提取情况。

> 假设你已经用 `runner.py` 跑出了形如  
> `output/2020-01-21__...__年度报告_chunks_20251128_194530.json` 的结果文件。

---

## 1. 准备工作

### 1.1 先跑出分块结果

示例命令：

```bash
python runner.py \
  -i input/pdfs/2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告.pdf \
  -o output \
  --timestamp
```

完成后，在 `output/` 目录中会看到：

- `2020-01-21__...__年度报告_chunks_20251128_194530.json`

后续审阅都基于这个 JSON 文件。

### 1.2 审阅脚本位置

- 路径：`scripts/preview_chunks.py`
- 依赖：`pandas`（导出 Excel 时需要，通常在本项目的 `requirements.txt` 中已经包含）
  - 如果缺少 `openpyxl`，脚本在导出时报错会提示安装。

---

## 2. 基本用法

### 2.1 导出为 Excel（推荐）

导出包含父块 + 子块的完整审阅表：

```bash
python scripts/preview_chunks.py \
  output/2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告_chunks_20251128_194530.json \
  --show-children \
  --limit 0 \
  --xlsx output/review.xlsx
```

关键参数说明：

- `json_path`：`runner.py` 生成的 `_chunks.json` 路径
- `--show-children`：把子块（children）也导出（不加则只导出父块）
- `--limit 0`：不截断，导出全部分块（>0 则只导出前 N 条）
- `--xlsx output/review.xlsx`：导出为 Excel 文件，父子分两个 sheet

运行成功后，会在 `output/` 下生成：

- `review.xlsx`：用于人工审阅的 Excel 文件

### 2.2 终端快速预览（可选）

如果只想在终端快速看前几条：

```bash
python scripts/preview_chunks.py \
  output/某公司年报_chunks_2025xxxx_xxxxxx.json \
  --show-children \
  --limit 50
```

终端会按“父块 / 子块分别排序”打印若干行，包含页码、位置、质量分和内容片段。

---

## 3. Excel 结构说明

`review.xlsx` 默认包含两个工作表：

- `parents`：父层分块（包括 `text_group` / `table_group` / `section`）
- `children`：子层分块（`chunk_type` 基本为 `child`）

每个 sheet 的主要列含义如下：

- `idx`：在当前 sheet 内的序号（导出时重新编号）
- `layer`：`parent` 或 `child`
- `chunk_id`：分块的唯一 ID
- `chunk_type`：
  - `text_group`：文本父块
  - `table_group`：表格父块
  - `section`：章节结构节点（纯标题）
  - `child`：子块
- `is_table`：是否标记为表格块
- `entry_page`：此块最主要关联的页码（综合 `page_number` 和 `metadata.page_numbers` 得出）
- `page_number`：主页码（如果有）
- `page_numbers`：元数据里记录的所有页码（逗号分隔）
- `bbox`：块在 PDF 中的坐标（如果有，格式为 `[x0, y0, x1, y1]`）
- `y0`：bbox 的顶部 y 坐标，用于在同一页内判断上下顺序
- `position`：在父块内的相对位置（子块用于排序）
- `quality_score`：质量分（0–1），越高说明分块更完整、文本质量更好
- `parent_id`：父块的 `chunk_id`（children 用这个关联回父层）
- `child_count`：父块下挂的子块数量（parents 用）
- `section_title`：该分块挂载的章节标题（如“第三节 公司业务概要”）
- `section_path`：完整章节路径（如“第三节 公司业务概要 / 5、投资收益”）
- `content_len`：内容长度（字符数）
- `content`：本块的完整文本内容（包括表格 markdown）

---

## 4. 推荐审阅流程

下面是一套推荐的人工审阅流程，可以按自己的习惯微调。

### 4.1 先通览父块（`parents` sheet）

1. 打开 `review.xlsx` 的 `parents` 工作表。
2. 在 Excel 中按以下顺序排序：
   - 第一优先：`entry_page` 升序；
   - 第二优先：`y0` 升序（同一页内从上到下）。
3. 固定表头行，方便滚动浏览。

重点检查：

- 每一页的第一个父块是否从正确位置开始（尤其是目录页、各章节起始页）。
- `chunk_type == table_group` 的表格块：
  - 内容是否只包含表格及其相关说明；
  - 是否仍混入整页目录或正文（这是这套优化重点要消掉的问题）。
- `section_title` / `section_path` 是否符合直觉（如“第四节 经营情况讨论与分析”的块都挂在正确章节下）。

如果发现典型问题（例如目录仍然混入表格父块），可以在 JSON 中定位对应 `chunk_id`，再回到代码侧调整清洗/抽取逻辑。

### 4.2 抽查子块（`children` sheet）

1. 切到 `children` 工作表。
2. 同样按 `entry_page` 和 `position` 排序。
3. 过滤出：
   - `chunk_type == child`；
   - 以及你关心的章节（可按 `section_title` / `section_path` 过滤）。

检查要点：

- 子块之间拼接后，是否能无缝还原父块内容（边界是否切得太碎或太粗）。
- 子块文本是否大量截断在句子中间（可结合 `quality_score` 看低分段）。
- 某些关键段落（风险提示、重要事项、会计政策）是否被保留且切块合理。

### 4.3 专门审阅表格块

如果只想集中看表格：

1. 在 `parents` sheet 中筛选：
   - `chunk_type == table_group`；
   - 或 `is_table == TRUE`。
2. 根据 `entry_page` 排序。
3. 针对每个表格，打开 PDF 对应页，检查：
   - 标题、单位说明、注释是否都被带进了 `TableGroup` 的 `content` 中；
   - 表格本身是否有明显漏行、重复或错位。

---

## 5. 常见问题与定位方式

### 5.1 目录仍然出现在表格块中

- 现象：某个 `table_group` 的 `content` 里开头是目录或大段正文。
- 原因：极端情况下表格检测 / bbox 定位仍然不准，或者该块本身多次跨页。
- 定位方式：
  - 在 Excel 里找该行的 `chunk_id`，记下；
  - 在 `_chunks.json` 中搜索这个 `chunk_id`，排查它的 `metadata`、`page_numbers`、`bbox`；
  - 如需调参，可从：
    - `table_processor.py` 中的 bbox 检测 / 清洗逻辑；
    - `context_identifier.py` 中 `_trim_non_table_noise` 规则 入手。

### 5.2 某些章节被切得太碎

- 现象：`children` 中同一段话被切成很多很短的子块。
- 关注字段：
  - `content_len` 很小且 `quality_score` 偏低；
  - `position` 相邻的多个子块 `content` 接起来读不顺。
- 调整方向：
  - 在 `smart_chunker.py` 中调 `child_size`、`overlap`，以及滑窗策略对句号等标点的处理；
  - 或在 `SmartChunker` 的质量过滤阈值中放宽/收紧 `min_chunk_size`。

---

## 6. 小结

- `runner.py` 负责“跑出结果”；  
- `scripts/preview_chunks.py` 负责“把结果变成易于人工审阅的 Excel”；  
- `docs/review_chunks.md`（本文）总结了一套推荐的审阅流程。

借助 `review.xlsx` 的 `parents` / `children` 两个 sheet，你可以：

- 快速定位表格提取是否干净；
- 检查文本分块是否合理；
- 评估章节结构挂载是否符合直觉。

如果在某一类报表上发现持续性的模式问题（例如某类表格总有相同噪声），建议在 Excel 中记下典型样例的 `chunk_id` 和页码，然后针对性地调整对应模块的规则。

