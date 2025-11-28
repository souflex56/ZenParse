"""
数据模型定义 - 中文财报PDF分块

定义文档元素、分块和处理结果的数据结构。
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import uuid
from datetime import datetime


class BaseModel:
    """提供通用的序列化/反序列化能力"""

    def to_dict(self) -> Dict[str, Any]:
        def convert(value):
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, BaseModel):
                return convert(asdict(value))
            if is_dataclass(value):
                return convert(asdict(value))
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """简单 from_dict，复杂场景可在子类重写"""
        return cls(**data)


class ElementType(Enum):
    """文档元素类型"""
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    SUBTITLE = "subtitle"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    IMAGE = "image"
    FORMULA = "formula"


class ChunkType(Enum):
    """分块类型"""
    PARENT = "parent"
    CHILD = "child"
    TABLE_GROUP = "table_group"
    TEXT_GROUP = "text_group"
    SECTION = "section"


class StandardFlag(str, Enum):
    """会计准则执行状态"""
    OLD = "old"
    NEW = "new"
    UNKNOWN = "unknown"


class TableType(Enum):
    """表格类型 - 针对财报"""
    FINANCIAL_STATEMENT = "financial_statement"  # 财务报表
    BALANCE_SHEET = "balance_sheet"              # 资产负债表
    INCOME_STATEMENT = "income_statement"        # 利润表
    CASH_FLOW = "cash_flow"                      # 现金流量表
    NOTES = "notes"                               # 附注表格
    SUMMARY = "summary"                           # 汇总表
    DETAIL = "detail"                             # 明细表
    OTHER = "other"                               # 其他


@dataclass
class DocumentElement(BaseModel):
    """文档元素 - 最小解析单位"""
    element_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    element_type: ElementType = ElementType.TEXT
    content: str = ""
    page_number: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    
    # 质量和置信度
    quality_score: float = 1.0
    extraction_confidence: float = 1.0
    
    # 语义角色
    context_role: Optional[str] = None  # title, note, caption, description
    semantic_type: Optional[str] = None  # 财务术语类型
    
    # 中文特性
    contains_chinese: bool = True
    chinese_ratio: float = 0.0  # 中文字符比例
    
    # 财报特性
    fiscal_period: Optional[str] = None  # 财务期间
    company_name: Optional[str] = None   # 公司名称
    report_section: Optional[str] = None # 报告章节
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.content:
            self.chinese_ratio = self._calculate_chinese_ratio(self.content)
            self.contains_chinese = self.chinese_ratio > 0.1
    
    def _calculate_chinese_ratio(self, text: str) -> float:
        """计算中文字符比例"""
        if not text:
            return 0.0
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_chars / len(text) if text else 0.0


@dataclass
class TableGroup(BaseModel):
    """表格组 - 表格+上下文"""
    group_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    table: DocumentElement = None
    title: Optional[DocumentElement] = None
    note: Optional[DocumentElement] = None
    caption: Optional[DocumentElement] = None
    
    # 表格类型识别
    table_type: TableType = TableType.OTHER
    
    # 统一内容
    unified_content: str = ""
    unified_markdown: str = ""
    row_header_paths: List[str] = field(default_factory=list)  # 行级层级路径（用于表头/科目层级保留）
    
    # 财报特定信息
    accounting_items: List[str] = field(default_factory=list)  # 会计科目
    time_periods: List[str] = field(default_factory=list)      # 时间期间
    monetary_unit: Optional[str] = None                        # 货币单位
    
    # 质量评估
    completeness_score: float = 1.0
    structure_score: float = 1.0
    context_quality: float = 1.0
    
    def calculate_group_quality(self) -> float:
        """计算组质量分数"""
        weights = {
            'completeness': 0.3,
            'structure': 0.3,
            'context': 0.2,
            'table_quality': 0.2
        }
        
        scores = [
            self.completeness_score * weights['completeness'],
            self.structure_score * weights['structure'],
            self.context_quality * weights['context'],
            self.table.quality_score * weights['table_quality'] if self.table else 0
        ]
        
        return sum(scores)


@dataclass
class TableMetadata(BaseModel):
    """表格元数据"""
    table_dimensions: str = ""  # e.g., "10x5" (rows x columns)
    has_header: bool = False
    table_type: str = ""  # financial_statement, summary, detail等
    numeric_summary: Dict[str, Any] = field(default_factory=dict)  # 数值统计信息
    high_value: bool = False  # 是否为高价值表格
    extraction_confidence: float = 0.0  # 提取置信度
    row_count: int = 0
    column_count: int = 0
    contains_financial_data: bool = False
    accounting_items: List[str] = field(default_factory=list)
    time_periods: List[str] = field(default_factory=list)
    bbox_estimated: bool = False  # bbox 是否为估算值


@dataclass
class ChunkMetadata(BaseModel):
    """分块元数据"""
    source_ref: Optional[str] = None  # 源文件引用ID，通过 sources catalog 查找文件信息
    page_numbers: List[int] = field(default_factory=list)
    extraction_method: str = ""  # pdfplumber, unstructured, text_semantic等
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 财报元数据
    report_type: Optional[str] = None      # 年报、季报等
    fiscal_year: Optional[int] = None      # 财年
    fiscal_quarter: Optional[int] = None   # 季度
    company_code: Optional[str] = None     # 公司代码
    company_name: Optional[str] = None     # 公司名称
    industry: Optional[str] = None         # 行业

    # 结构化章节元数据
    section_level: int = 0                 # 章节层级（1=节/章，2=一、，3=（一），4=1.）
    section_path: List[str] = field(default_factory=list)  # 章节路径
    section_title: str = ""                # 当前章节标题
    is_complete_section: bool = False      # 是否为完整章节（未被截断）
    
    # 分块统计
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    chinese_char_count: int = 0
    
    # 质量指标
    readability_score: float = 0.0
    information_density: float = 0.0
    coherence_score: float = 0.0
    # 准则摘要（可选，用于检索加权）
    standards_profile_summary: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk(BaseModel):
    """通用分块"""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    chunk_type: ChunkType = ChunkType.TEXT_GROUP
    
    # 层级关系
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    
    # 位置信息
    position: int = 0  # 在父块中的位置
    start_char: int = 0  # 在原文中的起始字符位置
    end_char: int = 0    # 在原文中的结束字符位置
    page_number: Optional[int] = None  # 页码
    bbox: Optional[Tuple[float, float, float, float]] = None  # 边界框 (x0, y0, x1, y1)
    element_type: Optional[str] = None  # 元素类型（来自原始DocumentElement）
    extraction_confidence: float = 0.0  # 提取置信度
    
    # 表格相关
    is_table: bool = False  # 是否为表格块
    table_structure_preserved: bool = False  # 表格结构是否保留
    table_metadata: Optional[TableMetadata] = None  # 表格元数据
    
    # 元数据
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    
    # 质量评分
    quality_score: float = 1.0
    relevance_score: float = 1.0
    
    # 语义信息
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)  # 实体识别结果
    topics: List[str] = field(default_factory=list)
    
    # 财报特定
    contains_financial_data: bool = False
    financial_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.content:
            self.metadata.char_count = len(self.content)
            self.metadata.chinese_char_count = sum(
                1 for char in self.content if '\u4e00' <= char <= '\u9fff'
            )
            # 简单的词数统计
            self.metadata.word_count = len(self.content.split())
            # 句子数统计（基于中文句号和英文句号）
            self.metadata.sentence_count = (
                self.content.count('。') + 
                self.content.count('.') + 
                self.content.count('！') + 
                self.content.count('？')
            )
    
    def calculate_information_density(self) -> float:
        """计算信息密度"""
        if not self.content:
            return 0.0
        
        # 基于关键词、实体和财务指标的密度
        total_info_items = (
            len(self.keywords) + 
            len(self.entities) + 
            len(self.financial_indicators)
        )
        
        # 归一化到字符长度
        density = total_info_items / (len(self.content) / 100) if self.content else 0
        return min(1.0, density / 10)  # 假设每100字符10个信息项为满分
    
    def calculate_coherence_score(self) -> float:
        """
        计算文本连贯性分数
        
        基于三个维度:
        1. 衔接词检测 (40%) - 检测逻辑连接词
        2. 主题词重叠 (30%) - 相邻句子的关键词重叠
        3. 结构完整性 (30%) - 句子数量和段落结构
        
        Returns:
            float: 0.0-1.0 之间的连贯性分数
        """
        if not self.content or len(self.content.strip()) < 20:
            return 0.0
        
        # 1. 衔接词检测 (权重 40%)
        connector_score = self._calculate_connector_score()
        
        # 2. 主题词重叠 (权重 30%)
        topic_overlap_score = self._calculate_topic_overlap_score()
        
        # 3. 结构完整性 (权重 30%)
        structure_score = self._calculate_structure_score()
        
        # 加权计算
        coherence = (
            connector_score * 0.4 +
            topic_overlap_score * 0.3 +
            structure_score * 0.3
        )
        
        return min(1.0, max(0.0, coherence))
    
    def _calculate_connector_score(self) -> float:
        """计算衔接词分数"""
        import re
        
        # 中文衔接词分类
        connectors = {
            # 因果关系
            'causal': ['因此', '所以', '由于', '因为', '故', '以致', '导致', '使得', '造成'],
            # 转折关系
            'adversative': ['但是', '然而', '不过', '可是', '虽然', '尽管', '虽说', '却'],
            # 递进关系
            'progressive': ['同时', '另外', '此外', '而且', '并且', '还有', '不仅', '更'],
            # 顺序关系
            'sequential': ['首先', '其次', '最后', '然后', '接着', '随后', '第一', '第二'],
            # 总结关系
            'summary': ['总之', '综上', '总的来说', '概括', '总结', '归纳'],
            # 举例关系
            'exemplification': ['例如', '比如', '如', '譬如', '举例', '以...为例']
        }
        
        content = self.content
        total_connectors = 0
        connector_types_found = set()
        
        for conn_type, words in connectors.items():
            for word in words:
                if word in content:
                    total_connectors += content.count(word)
                    connector_types_found.add(conn_type)
        
        # 计算得分
        # 基于衔接词数量（每500字符有1-2个衔接词为佳）
        content_length = len(content)
        expected_connectors = max(1, content_length / 500)
        quantity_score = min(1.0, total_connectors / expected_connectors)
        
        # 衔接词类型多样性加分
        diversity_bonus = min(0.3, len(connector_types_found) * 0.1)
        
        return min(1.0, quantity_score + diversity_bonus)
    
    def _calculate_topic_overlap_score(self) -> float:
        """计算相邻句子的主题词重叠分数"""
        import re
        
        # 按句子分割
        sentences = re.split(r'[。！？.!?]', self.content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return 0.5  # 单句给中等分数
        
        # 提取每个句子的关键词（简单实现：提取2字以上的中文词）
        def extract_keywords(text: str) -> set:
            # 提取中文词语（2-4个字的连续中文）
            words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
            # 过滤常见停用词
            stopwords = {'这个', '那个', '什么', '如何', '怎么', '可以', '能够', '已经', 
                        '进行', '通过', '根据', '按照', '对于', '关于', '由于'}
            return {w for w in words if w not in stopwords}
        
        # 计算相邻句子的重叠
        overlaps = []
        for i in range(len(sentences) - 1):
            keywords1 = extract_keywords(sentences[i])
            keywords2 = extract_keywords(sentences[i + 1])
            
            if keywords1 and keywords2:
                intersection = keywords1 & keywords2
                union = keywords1 | keywords2
                overlap = len(intersection) / len(union) if union else 0
                overlaps.append(overlap)
        
        if not overlaps:
            return 0.3
        
        # 平均重叠率
        avg_overlap = sum(overlaps) / len(overlaps)
        
        # 期望重叠率约 0.1-0.3 为佳（太高说明重复，太低说明跳跃）
        if avg_overlap < 0.05:
            return avg_overlap * 4  # 太低惩罚
        elif avg_overlap > 0.5:
            return 0.7  # 太高略微惩罚
        else:
            return min(1.0, avg_overlap * 2 + 0.3)
    
    def _calculate_structure_score(self) -> float:
        """计算结构完整性分数"""
        import re
        
        content = self.content
        
        # 句子数量
        sentences = re.split(r'[。！？.!?]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # 段落数量（基于换行）
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # 平均句子长度
        avg_sentence_length = len(content) / sentence_count if sentence_count > 0 else 0
        
        score = 0.0
        
        # 1. 句子数量合理性（3-15句为佳）
        if 3 <= sentence_count <= 15:
            score += 0.4
        elif sentence_count < 3:
            score += sentence_count * 0.1
        else:
            score += 0.3  # 太多句子略微降分
        
        # 2. 句子长度合理性（20-80字为佳）
        if 20 <= avg_sentence_length <= 80:
            score += 0.3
        elif avg_sentence_length < 20:
            score += avg_sentence_length / 20 * 0.2
        else:
            score += 0.2
        
        # 3. 段落结构（有多个段落说明结构良好）
        if paragraph_count >= 2:
            score += 0.3
        else:
            score += 0.15
        
        return min(1.0, score)


ParentChunk = Chunk
ChildChunk = Chunk


@dataclass
class ProcessingStats(BaseModel):
    """处理统计信息"""
    total_pages: int = 0
    total_elements: int = 0
    total_tables: int = 0
    total_text_blocks: int = 0
    
    # 分块统计
    parent_chunks_count: int = 0
    child_chunks_count: int = 0
    table_groups_count: int = 0
    
    # 质量统计
    high_quality_chunks: int = 0  # quality_score > 0.8
    medium_quality_chunks: int = 0  # 0.5 < quality_score <= 0.8
    low_quality_chunks: int = 0    # quality_score <= 0.5
    
    # 性能统计
    processing_time_seconds: float = 0.0
    pdf_parsing_time: float = 0.0
    context_identification_time: float = 0.0
    chunking_time: float = 0.0
    
    # 错误统计
    parsing_errors: int = 0
    extraction_warnings: int = 0
    
    # 财报特定统计
    financial_statements_count: int = 0
    accounting_items_extracted: int = 0
    time_periods_identified: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            "文档统计": {
                "总页数": self.total_pages,
                "总元素数": self.total_elements,
                "表格数": self.total_tables,
                "文本块数": self.total_text_blocks
            },
            "分块统计": {
                "父块数": self.parent_chunks_count,
                "子块数": self.child_chunks_count,
                "表格组数": self.table_groups_count,
                "平均子块/父块": self.child_chunks_count / self.parent_chunks_count if self.parent_chunks_count > 0 else 0
            },
            "质量分布": {
                "高质量": self.high_quality_chunks,
                "中等质量": self.medium_quality_chunks,
                "低质量": self.low_quality_chunks,
                "高质量占比": f"{self.high_quality_chunks / (self.parent_chunks_count + self.child_chunks_count) * 100:.1f}%" if (self.parent_chunks_count + self.child_chunks_count) > 0 else "0%"
            },
            "性能指标": {
                "总处理时间": f"{self.processing_time_seconds:.2f}秒",
                "PDF解析": f"{self.pdf_parsing_time:.2f}秒",
                "上下文识别": f"{self.context_identification_time:.2f}秒",
                "分块处理": f"{self.chunking_time:.2f}秒"
            },
            "财报信息": {
                "财务报表数": self.financial_statements_count,
                "会计科目数": self.accounting_items_extracted,
                "时间期间数": self.time_periods_identified
            }
        }


@dataclass
class StandardsProfile(BaseModel):
    """会计准则执行情况（报表级）"""
    revenue: StandardFlag = StandardFlag.UNKNOWN
    financial_instruments: StandardFlag = StandardFlag.UNKNOWN
    lease: StandardFlag = StandardFlag.UNKNOWN
    decided_by: str = "unknown"  # explicit_note / table_keywords / year_threshold / mixed


@dataclass
class ChunkingResult(BaseModel):
    """分块处理结果"""
    parent_chunks: List[Chunk] = field(default_factory=list)
    child_chunks: List[Chunk] = field(default_factory=list)
    table_groups: List[TableGroup] = field(default_factory=list)
    
    # 统计信息
    statistics: ProcessingStats = field(default_factory=ProcessingStats)
    
    # 文档信息
    source_file: str = ""
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 来源目录 (Source Catalog)
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return (
            len(self.parent_chunks) > 0 or 
            len(self.child_chunks) > 0 or 
            len(self.table_groups) > 0
        ) and len(self.errors) == 0
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """根据ID获取分块"""
        for chunk in self.parent_chunks + self.child_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_children_of_parent(self, parent_id: str) -> List[Chunk]:
        """获取父块的所有子块"""
        return [
            chunk for chunk in self.child_chunks 
            if chunk.parent_id == parent_id
        ]


# =============================================================================
# 训练相关数据模型
# =============================================================================

@dataclass
class PreferencePair(BaseModel):
    """偏好对数据结构 - 用于DPO/RLHF训练"""
    question: str                                           # 问题/提示
    positive_response: str                                  # 正样本回答（高质量）
    negative_response: str                                  # 负样本回答（低质量）
    strategy_used: Any                                      # 使用的策略（StrategyType）
    quality_scores: Dict[str, float]                        # 质量评分
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    # 可选字段
    positive_score: Optional[float] = None                  # 正样本得分
    negative_score: Optional[float] = None                  # 负样本得分
    quality_gap: Optional[float] = None                     # 质量差距
    
    def __post_init__(self):
        """初始化后处理"""
        # 计算质量差距
        if 'positive_score' in self.quality_scores and 'negative_score' in self.quality_scores:
            self.positive_score = self.quality_scores['positive_score']
            self.negative_score = self.quality_scores['negative_score']
            self.quality_gap = self.positive_score - self.negative_score
