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

    def process(self, pdf_path: str) -> Tuple[List[ParentChunk], List[ChildChunk], List[TableGroup]]:
        """处理单个 PDF，返回父子分块与表格组"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        self.logger.info(f"Step 1: 解析 PDF - {pdf_path}")
        elements = self.parser.parse(pdf_path)

        self.logger.info("Step 2: 识别上下文与表格组")
        table_groups = self.context_identifier.identify_table_groups(elements)

        self.logger.info("Step 3: 智能分块")
        parent_chunks, child_chunks = self.chunker.create_chunks(
            elements=elements, table_groups=table_groups, source_file=pdf_path
        )

        # 准则判定（不改变返回结构，将结果保存在实例属性中）
        fiscal_year = self._infer_fiscal_year(parent_chunks, child_chunks, pdf_path)
        self.last_standards_profile = self.standards_detector.detect(elements, fiscal_year)

        return parent_chunks, child_chunks, table_groups
