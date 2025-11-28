"""
核心管线：PDF -> 元素 -> 分块

保持原有 core/data 目录结构，实现零代码修改的迁移体验。
"""

from typing import Dict, List, Tuple, Union
import os
import re
import yaml

from .core.logger import get_logger
from .core.exceptions import ZenParseError
from .data.pdf_parser import ChineseFinancialPDFParser
from .data.context_identifier import ChineseFinancialContextIdentifier
from .data.smart_chunker import ChineseFinancialChunker
from .data.models import ParentChunk, ChildChunk, TableGroup
from .data.standards_detector import StandardsDetector

# 页码/页眉页脚清洗正则
PAGE_MARKER_RE = re.compile(
    r"""
    ^(?:第?\s*\d+\s*页(?:\s*/\s*共?\s*\d+\s*页)?   # 第 3 页 / 共 10 页
    |page\s*\d+(?:\s*of\s*\d+)?                  # page 3 of 10
    |\d+\s*/\s*\d+                               # 1/10
    |-?\s*\d+\s*-?                               # - 3 - 或孤立页码
    |[ivxlcdm]+\s*/\s*[ivxlcdm]+)                # 罗马页码  i/x
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
NUMERIC_LINE_RE = re.compile(r"^[\d\s/.-]{1,15}$")


def _strip_page_markers(text: str) -> str:
    """移除文本中的页码、页眉页脚等整行干扰信息"""
    if not text:
        return text

    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        # 仅对短行的页码样式做过滤，避免误删正文数字（如 211/985）
        if len(stripped) <= 15 and (PAGE_MARKER_RE.match(stripped) or NUMERIC_LINE_RE.match(stripped)):
            continue
        kept.append(line)

    return "\n".join(kept)


class ZenPipeline:
    """整合解析、上下文识别与分块的端到端管线"""

    def __init__(self, config: Union[str, Dict] = "config.yaml"):
        self.logger = get_logger("ZenPipeline")
        self.config = self._load_config(config)

        parser_config = self._component_config("pdf_parsing")
        context_config = self._component_config("context")
        chunking_config = self._component_config("chunking")

        try:
            self.parser = ChineseFinancialPDFParser(parser_config)
            self.context_identifier = ChineseFinancialContextIdentifier(context_config)
            self.chunker = ChineseFinancialChunker(chunking_config)
            self.standards_detector = StandardsDetector(self.config)
            self.last_standards_profile = None
        except Exception as exc:
            raise ZenParseError(f"组件初始化失败: {exc}") from exc

    def _load_config(self, config_source: Union[str, Dict]) -> Dict:
        """加载配置（支持 YAML 路径或字典）"""
        if isinstance(config_source, dict):
            return config_source

        if isinstance(config_source, str) and os.path.exists(config_source):
            with open(config_source, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            return loaded

        self.logger.warning(f"未找到配置文件或配置为空: {config_source}，使用默认配置")
        return {}

    def _component_config(self, key: str) -> Dict:
        """
        合并顶层配置和子配置，避免调用方手动平铺字段
        """
        if not isinstance(self.config, dict):
            return {}

        section = self.config.get(key, {})
        if isinstance(section, dict):
            merged = {**self.config, **section}
            return merged

        return dict(self.config)

    def _infer_fiscal_year(
        self,
        parent_chunks: List[ParentChunk],
        child_chunks: List[ChildChunk],
        pdf_path: str,
    ) -> Union[int, None]:
        """从分块或文件名推断财年"""
        for chunk in parent_chunks + child_chunks:
            if getattr(chunk, "metadata", None) and getattr(chunk.metadata, "fiscal_year", None):
                try:
                    return int(chunk.metadata.fiscal_year)
                except (TypeError, ValueError):
                    continue

        # 文件名兜底：匹配 20xx
        m = re.search(r"(20\\d{2})", os.path.basename(pdf_path))
        if m:
            try:
                return int(m.group(1))
            except (TypeError, ValueError):
                return None
        return None

    def process(self, pdf_path: str, source_ref: str = None) -> Tuple[List[ParentChunk], List[ChildChunk], List[TableGroup]]:
        """处理单个 PDF，返回父子分块与表格组"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        self.logger.info(f"Step 1: 解析 PDF - {pdf_path}")
        # 传递 source_ref 给 parser (如果 parser 需要记录 source_ref 到 Element)
        # 目前 Element 还没有 source_ref，暂时只传 pdf_path
        elements = self.parser.parse(pdf_path)

        # 清洗页码/页眉页脚，避免干扰 chunk content（保留 metadata.page_numbers）
        self.logger.info("Step 1.5: 清洗页码与页眉页脚")
        for elem in elements:
            content = getattr(elem, "content", "")
            if content:
                elem.content = _strip_page_markers(content)

        self.logger.info("Step 2: 识别上下文与表格组")
        table_groups = self.context_identifier.identify_table_groups(elements)

        self.logger.info("Step 3: 智能分块")
        # 从文件名提取 display_name
        display_name = os.path.basename(pdf_path)
        parent_chunks, child_chunks = self.chunker.create_chunks(
            elements=elements, 
            table_groups=table_groups, 
            source_ref=source_ref,
            display_name=display_name
        )

        # 准则判定（不改变返回结构，将结果保存在实例属性中）
        fiscal_year = self._infer_fiscal_year(parent_chunks, child_chunks, pdf_path)
        self.last_standards_profile = self.standards_detector.detect(elements, fiscal_year)

        return parent_chunks, child_chunks, table_groups
