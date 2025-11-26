"""
智能分块器 - 父子分块架构

实现语义感知的父子分块策略，保持上下文完整性。
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple

from .models import (
    DocumentElement, 
    TableGroup, 
    Chunk, 
    ChunkType,
    ChunkMetadata,
    ElementType,
    TableMetadata
)


class SmartChunker:
    """智能分块器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化分块器"""
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 分块参数
        self.parent_size = self.config.get('parent_size', 3000)  # 父块大小
        self.child_size = self.config.get('child_size', 1000)   # 子块大小
        self.overlap = self.config.get('overlap', 200)          # 重叠大小
        
        # 质量阈值
        self.min_chunk_size = self.config.get('min_chunk_size', 50)
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        
        # 分块策略
        self.chunking_strategy = self.config.get('strategy', 'semantic')  # semantic, fixed, sliding
        
        self.logger.info(f"分块器初始化: 父块={self.parent_size}, "
                        f"子块={self.child_size}, 重叠={self.overlap}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # 防止日志向上传播导致重复
            logger.propagate = False
        
        return logger
    
    def create_chunks(
        self,
        elements: List[DocumentElement],
        table_groups: List[TableGroup],
        source_file: str = ""
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """创建父子分块"""
        # 保存源文件路径
        self.source_file = source_file
        parent_chunks = []
        child_chunks = []
        
        # 1. 处理表格组（保持完整性）
        table_parents, table_children = self._process_table_groups(table_groups)
        parent_chunks.extend(table_parents)
        child_chunks.extend(table_children)
        
        # 2. 获取非表格元素
        table_element_ids = set()
        for group in table_groups:
            if group.table:
                table_element_ids.add(group.table.element_id)
            if group.title:
                table_element_ids.add(group.title.element_id)
            if group.note:
                table_element_ids.add(group.note.element_id)
            if group.caption:
                table_element_ids.add(group.caption.element_id)
        
        text_elements = [
            elem for elem in elements 
            if elem.element_id not in table_element_ids
        ]
        
        # 3. 处理文本元素
        text_parents, text_children = self._process_text_elements(text_elements)
        parent_chunks.extend(text_parents)
        child_chunks.extend(text_children)
        
        # 4. 建立父子关系
        self._establish_relationships(parent_chunks, child_chunks)
        
        # 5. 质量过滤
        parent_chunks = self._filter_low_quality_chunks(parent_chunks, chunk_type="parent")
        child_chunks = self._filter_low_quality_chunks(child_chunks, chunk_type="child")
        
        # 6. 计算质量指标（信息密度、连贯性分数）
        self._calculate_quality_metrics(parent_chunks, child_chunks)
        
        self.logger.info(f"分块完成: {len(parent_chunks)}个父块, "
                        f"{len(child_chunks)}个子块")
        
        return parent_chunks, child_chunks
    
    def _process_table_groups(
        self, 
        table_groups: List[TableGroup]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """处理表格组"""
        parent_chunks = []
        child_chunks = []
        
        for group in table_groups:
            # 创建父块（完整表格组）
            parent_chunk = self._create_table_parent_chunk(group)
            parent_chunks.append(parent_chunk)
            
            # 如果表格内容过大，创建子块
            if len(group.unified_content) > self.child_size:
                children = self._create_table_child_chunks(group, parent_chunk.chunk_id)
                child_chunks.extend(children)
                parent_chunk.child_ids = [c.chunk_id for c in children]
            else:
                # 小表格直接作为子块
                # 检查财务数据并提取指标
                contains_financial, financial_indicators = self._detect_financial_content_with_indicators(group.unified_content)
                
                # 创建表格元数据
                table_meta = TableMetadata(
                    table_dimensions=f"{len(group.table.rows) if group.table and hasattr(group.table, 'rows') else 0}x{len(group.table.rows[0]) if group.table and hasattr(group.table, 'rows') and group.table.rows else 0}",
                    has_header=True if group.title else False,
                    table_type=group.table_type.value if group.table_type else 'other',
                    high_value=group.table.quality_score > 0.9 if group.table else False,
                    extraction_confidence=group.table.extraction_confidence if group.table and hasattr(group.table, 'extraction_confidence') else 0.95,
                    row_count=len(group.table.rows) if group.table and hasattr(group.table, 'rows') else 0,
                    column_count=len(group.table.rows[0]) if group.table and hasattr(group.table, 'rows') and group.table.rows else 0,
                    contains_financial_data=contains_financial,
                    accounting_items=financial_indicators,
                    time_periods=group.time_periods if hasattr(group, 'time_periods') else [],
                    numeric_summary=self._calculate_numeric_summary(group)
                )
                
                # 创建带有extraction_method的元数据
                chunk_metadata = ChunkMetadata(
                    source_file=getattr(self, 'source_file', ''),
                    page_numbers=[group.table.page_number] if group.table and group.table.page_number else [],
                    extraction_method=group.table.extraction_method if group.table and hasattr(group.table, 'extraction_method') else 'table_group'
                )
                
                child = Chunk(
                    content=group.unified_content,
                    chunk_type=ChunkType.TABLE_GROUP,
                    parent_id=parent_chunk.chunk_id,
                    quality_score=group.calculate_group_quality(),
                    contains_financial_data=contains_financial,
                    financial_indicators=financial_indicators,
                    start_char=0,  # 表格子块位置
                    end_char=len(group.unified_content),
                    # 表格相关字段
                    is_table=True,
                    table_structure_preserved=True,
                    table_metadata=table_meta,
                    metadata=chunk_metadata,
                    # 添加坐标信息
                    page_number=group.table.page_number if group.table else None,
                    bbox=group.table.bbox if group.table else None,
                    element_type=str(group.table.element_type.value) if group.table and hasattr(group.table.element_type, 'value') else 'table',
                    extraction_confidence=group.table.extraction_confidence if group.table else 0.95
                )
                child_chunks.append(child)
                parent_chunk.child_ids = [child.chunk_id]
        
        return parent_chunks, child_chunks
    
    def _create_table_parent_chunk(self, group: TableGroup) -> Chunk:
        """创建表格父块"""
        # 获取页码信息
        page_number = group.table.page_number if group.table else None
        
        # 构建元数据 - 确保page_numbers字段正确
        metadata = ChunkMetadata(
            page_numbers=[page_number] if page_number is not None else [],
            extraction_method=group.table.extraction_method if group.table and hasattr(group.table, 'extraction_method') else 'table_group',
            report_type='financial_report',
            source_file=getattr(self, 'source_file', '')
        )
        
        # 添加财报特定信息
        if group.accounting_items:
            metadata.industry = 'financial'
        if group.time_periods:
            # 尝试解析年份
            for period in group.time_periods:
                year_match = re.search(r'(\d{4})', period)
                if year_match:
                    metadata.fiscal_year = int(year_match.group(1))
                    break
        
        # 添加表格上下文信息到元数据
        if group.title and group.title.content:
            metadata.table_title = group.title.content
        if group.note and group.note.content:
            metadata.table_note = group.note.content
        if group.caption and group.caption.content:
            metadata.table_caption = group.caption.content
        
        # 获取或估算 bbox
        bbox, bbox_estimated = self._get_or_estimate_bbox(group, page_number)
        
        # 计算表格维度（从内容推断）
        row_count, col_count = self._estimate_table_dimensions(group)
        
        # 清理会计科目（移除开头标点、过滤不完整项）
        cleaned_accounting_items = self._clean_accounting_items(group.accounting_items) if group.accounting_items else []
        
        # 创建表格元数据
        table_meta = TableMetadata(
            table_dimensions=f"{row_count}x{col_count}",
            has_header=True if group.title else False,
            table_type=group.table_type.value if group.table_type else 'other',
            high_value=group.table.quality_score > 0.9 if group.table else False,
            extraction_confidence=group.table.extraction_confidence if group.table and hasattr(group.table, 'extraction_confidence') else 0.95,
            row_count=row_count,
            column_count=col_count,
            contains_financial_data=True,
            accounting_items=cleaned_accounting_items,
            time_periods=group.time_periods if group.time_periods else [],
            numeric_summary=self._calculate_numeric_summary(group),
            bbox_estimated=bbox_estimated
        )
        
        # 清理 unified_markdown 中残留的 * 前缀
        cleaned_content = self._clean_content_markers(group.unified_markdown)
        
        chunk = Chunk(
            content=cleaned_content,
            chunk_type=ChunkType.TABLE_GROUP,
            metadata=metadata,
            quality_score=group.calculate_group_quality(),
            contains_financial_data=True,
            financial_indicators=cleaned_accounting_items,
            # 表格相关字段
            is_table=True,
            table_structure_preserved=True,
            table_metadata=table_meta,
            # 添加坐标信息作为Chunk的直接属性
            page_number=page_number,
            bbox=bbox,
            element_type='table',  # 明确设置为table
            extraction_confidence=group.table.extraction_confidence if group.table else 0.95,
            # 添加起止位置
            start_char=0,
            end_char=len(cleaned_content) if cleaned_content else 0
        )
        
        # 添加关键词
        chunk.keywords = self._extract_keywords(group.unified_content)
        
        return chunk
    
    def _estimate_table_dimensions(self, group: TableGroup) -> Tuple[int, int]:
        """
        从表格内容估算行列数
        
        Returns:
            Tuple[row_count, col_count]
        """
        # 优先使用 rows 属性
        if group.table and hasattr(group.table, 'rows') and group.table.rows:
            rows = group.table.rows
            return len(rows), len(rows[0]) if rows[0] else 0
        
        # 从 content 推断
        content = group.unified_content or (group.table.content if group.table else "")
        if not content:
            return 0, 0
        
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        row_count = 0
        col_count = 0
        
        for line in lines:
            # 跳过标题行（【标题】、【说明】等）
            if line.startswith('【') and '】' in line[:10]:
                continue
            # 跳过分隔线
            if re.match(r'^[-\s|:]+$', line):
                continue
            
            row_count += 1
            
            # 估算列数
            if '|' in line:
                # Markdown 格式
                cells = [c for c in line.split('|') if c.strip()]
                col_count = max(col_count, len(cells))
            elif '\t' in line:
                cells = line.split('\t')
                col_count = max(col_count, len(cells))
            elif re.search(r'\s{2,}', line):
                cells = re.split(r'\s{2,}', line)
                col_count = max(col_count, len(cells))
        
        return row_count, col_count
    
    def _clean_accounting_items(self, items: List[str]) -> List[str]:
        """
        清理会计科目列表
        - 移除开头的标点符号
        - 过滤过短或不完整的项
        """
        if not items:
            return []
        
        cleaned = []
        for item in items:
            if not item:
                continue
            
            # 移除开头的标点
            item = re.sub(r'^[、,，。；：\s]+', '', item)
            
            # 移除结尾的标点
            item = re.sub(r'[、,，。；：\s]+$', '', item)
            
            # 过滤过短的项（小于2个字符）
            if len(item) < 2:
                continue
            
            # 过滤明显不完整的项（如只有"他长期"而不是"其他长期"）
            if item.startswith('他') and len(item) < 4:
                continue
            
            cleaned.append(item)
        
        # 去重
        return list(dict.fromkeys(cleaned))
    
    def _clean_content_markers(self, content: str) -> str:
        """
        清理内容中残留的标记符号
        - 移除开头的孤立 * 前缀（保留 markdown 粗体 **）
        - 移除页码+星号模式 (如 "9。*")
        """
        if not content:
            return content
        
        cleaned = content
        
        # 移除 "数字。*" 模式（页码+星号）
        cleaned = re.sub(r'\d+。\*\s*', '', cleaned)
        cleaned = re.sub(r'\d+\.\*\s*', '', cleaned)
        
        # 移除开头的孤立 * 前缀（不移除 ** 开头的 markdown 粗体）
        # 只有当 * 后面不是另一个 * 时才移除
        cleaned = re.sub(r'^\*(?!\*)', '', cleaned)
        
        # 移除行首的孤立 * 标记（保留 markdown 粗体 **）
        cleaned = re.sub(r'^(\*(?!\*))', '', cleaned, flags=re.MULTILINE)
        
        # 移除行尾孤立的 *（保留 markdown 粗体结尾 **）
        cleaned = re.sub(r'(?<!\*)\*\s*$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
    
    def _get_or_estimate_bbox(
        self, 
        group: TableGroup, 
        page_number: Optional[int]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], bool]:
        """
        获取或估算表格的边界框
        
        Args:
            group: 表格组
            page_number: 页码
            
        Returns:
            Tuple[bbox, is_estimated]: bbox 坐标和是否为估算值
        """
        # 1. 如果已有精确 bbox，直接返回
        if group.table and group.table.bbox:
            return (group.table.bbox, False)
        
        # 2. 否则进行估算
        # 基于表格内容估算大小
        content = group.unified_content or (group.table.content if group.table else "")
        if not content:
            # 默认值：A4 页面中心区域
            return ((50.0, 100.0, 545.0, 742.0), True)
        
        # 分析内容行数
        lines = content.split('\n')
        line_count = len([l for l in lines if l.strip()])
        
        # 估算每行高度（约 20 点）和宽度
        estimated_height = min(600, line_count * 20)
        
        # 检测最长行来估算宽度
        max_line_length = max((len(l) for l in lines if l.strip()), default=50)
        # 每个字符约 8 点宽度
        estimated_width = min(500, max_line_length * 8)
        
        # 假设表格居中放置
        page_width = 595.0  # A4 宽度
        page_height = 842.0  # A4 高度
        
        x0 = (page_width - estimated_width) / 2
        y0 = 80.0  # 顶部边距
        x1 = x0 + estimated_width
        y1 = y0 + estimated_height
        
        bbox = (x0, y0, x1, y1)
        self.logger.debug(f"估算 bbox: {bbox} (行数={line_count}, 最长行={max_line_length})")
        
        return (bbox, True)
    
    def _create_table_child_chunks(
        self, 
        group: TableGroup,
        parent_id: str
    ) -> List[Chunk]:
        """创建表格子块"""
        children = []
        
        # 对表格内容进行智能分割
        if group.table and group.table.content:
            # 尝试按行分割表格
            rows = group.table.content.split('\n')
            
            current_chunk = []
            current_size = 0
            
            for row in rows:
                row_size = len(row)
                
                if current_size + row_size > self.child_size and current_chunk:
                    # 创建子块
                    content = '\n'.join(current_chunk)
                    # 检查财务数据并提取指标
                    contains_financial, financial_indicators = self._detect_financial_content_with_indicators(content)
                    # 创建元数据
                    child_metadata = ChunkMetadata(
                        extraction_method='table_split',
                        page_numbers=[group.table.page_number] if group.table and group.table.page_number else [],
                        source_file=getattr(self, 'source_file', '')
                    )
                    
                    # 创建表格元数据
                    table_meta = TableMetadata(
                        table_dimensions="",  # 分割后的子块没有独立的维度
                        has_header=False,  # 子块通常不含表头
                        table_type=group.table_type.value if group.table_type else 'other',
                        high_value=group.table.quality_score > 0.9 if group.table else False,
                        extraction_confidence=group.table.extraction_confidence if group.table and hasattr(group.table, 'extraction_confidence') else 0.95,
                        row_count=len(current_chunk),
                        column_count=0,  # 分割后难以确定列数
                        contains_financial_data=contains_financial,
                        accounting_items=financial_indicators[:5] if financial_indicators else [],
                        time_periods=[],
                        numeric_summary={}
                    )
                    
                    child = Chunk(
                        content=content,
                        chunk_type=ChunkType.TABLE_GROUP,
                        parent_id=parent_id,
                        position=len(children),
                        quality_score=group.table.quality_score if group.table else 1.0,
                        contains_financial_data=contains_financial,
                        financial_indicators=financial_indicators,
                        start_char=len(children) * 500,  # 简单估算位置
                        end_char=(len(children) + 1) * 500 + len(content),
                        # 表格相关字段
                        is_table=True,
                        table_structure_preserved=True,
                        table_metadata=table_meta,
                        # 添加坐标信息
                        page_number=group.table.page_number if group.table else None,
                        bbox=group.table.bbox if group.table else None,
                        element_type='table',  # 明确设置为table
                        extraction_confidence=group.table.extraction_confidence if group.table else 0.95,
                        metadata=child_metadata
                    )
                    children.append(child)
                    
                    # 重置
                    current_chunk = [row] if row_size < self.child_size else []
                    current_size = row_size if row_size < self.child_size else 0
                else:
                    current_chunk.append(row)
                    current_size += row_size
            
            # 处理剩余内容
            if current_chunk:
                content = '\n'.join(current_chunk)
                # 检查财务数据并提取指标
                contains_financial, financial_indicators = self._detect_financial_content_with_indicators(content)
                
                # 创建元数据
                child_metadata = ChunkMetadata(
                    extraction_method='table_split',
                    page_numbers=[group.table.page_number] if group.table and group.table.page_number else []
                )
                
                # 创建表格元数据
                table_meta = TableMetadata(
                    table_dimensions="",  # 分割后的子块没有独立的维度
                    has_header=False,  # 子块通常不含表头
                    table_type=group.table_type.value if group.table_type else 'other',
                    high_value=group.table.quality_score > 0.9 if group.table else False,
                    extraction_confidence=group.table.extraction_confidence if group.table and hasattr(group.table, 'extraction_confidence') else 0.95,
                    row_count=len(current_chunk),
                    column_count=0,  # 分割后难以确定列数
                    contains_financial_data=contains_financial,
                    accounting_items=financial_indicators[:5] if financial_indicators else [],
                    time_periods=[],
                    numeric_summary={}
                )
                
                child = Chunk(
                    content=content,
                    chunk_type=ChunkType.TABLE_GROUP,
                    parent_id=parent_id,
                    position=len(children),
                    quality_score=group.table.quality_score if group.table else 1.0,
                    contains_financial_data=contains_financial,
                    financial_indicators=financial_indicators,
                    start_char=len(children) * 500,  # 简单估算位置
                    end_char=(len(children) + 1) * 500 + len(content),
                    # 表格相关字段
                    is_table=True,
                    table_structure_preserved=True,
                    table_metadata=table_meta,
                    # 添加坐标信息
                    page_number=group.table.page_number if group.table else None,
                    bbox=group.table.bbox if group.table else None,
                    element_type='table',  # 明确设置为table
                    extraction_confidence=group.table.extraction_confidence if group.table else 0.95,
                    metadata=child_metadata
                )
                children.append(child)
        
        return children
    
    def _process_text_elements(
        self,
        elements: List[DocumentElement]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """处理文本元素"""
        if not elements:
            return [], []
        
        if self.chunking_strategy == 'semantic':
            return self._semantic_chunking(elements)
        elif self.chunking_strategy == 'sliding':
            return self._sliding_window_chunking(elements)
        else:
            return self._fixed_size_chunking(elements)
    
    def _semantic_chunking(
        self,
        elements: List[DocumentElement]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """语义分块"""
        parent_chunks = []
        child_chunks = []
        
        # 按页面和语义相关性分组
        element_groups = self._group_by_semantic_relevance(elements)
        
        for group in element_groups:
            # 合并元素内容
            combined_content = '\n\n'.join([elem.content for elem in group])
            
            if len(combined_content) <= self.parent_size:
                # 直接创建父块
                parent = self._create_text_parent_chunk(combined_content, group)
                parent_chunks.append(parent)
                
                # 创建子块
                children = self._create_text_child_chunks(combined_content, parent.chunk_id, group)
                child_chunks.extend(children)
                parent.child_ids = [c.chunk_id for c in children]
            else:
                # 需要分割成多个父块
                sub_groups = self._split_large_group(group, self.parent_size)
                
                for sub_group in sub_groups:
                    sub_content = '\n\n'.join([elem.content for elem in sub_group])
                    parent = self._create_text_parent_chunk(sub_content, sub_group)
                    parent_chunks.append(parent)
                    
                    children = self._create_text_child_chunks(sub_content, parent.chunk_id, sub_group)
                    child_chunks.extend(children)
                    parent.child_ids = [c.chunk_id for c in children]
        
        return parent_chunks, child_chunks
    
    def _group_by_semantic_relevance(
        self,
        elements: List[DocumentElement]
    ) -> List[List[DocumentElement]]:
        """按语义相关性分组"""
        groups = []
        current_group = []
        current_size = 0
        current_page = None
        
        for elem in elements:
            elem_size = len(elem.content)
            
            # 判断是否需要新组
            need_new_group = False
            
            # 页面变化
            if current_page is not None and abs(elem.page_number - current_page) > 1:
                need_new_group = True
            
            # 大小超限
            if current_size + elem_size > self.parent_size:
                need_new_group = True
            
            # 标题元素开始新组
            if elem.element_type == ElementType.TITLE and current_group:
                need_new_group = True
            
            if need_new_group and current_group:
                groups.append(current_group)
                current_group = [elem]
                current_size = elem_size
                current_page = elem.page_number
            else:
                current_group.append(elem)
                current_size += elem_size
                current_page = elem.page_number
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _split_large_group(
        self,
        group: List[DocumentElement],
        max_size: int
    ) -> List[List[DocumentElement]]:
        """分割大组"""
        sub_groups = []
        current_sub = []
        current_size = 0
        
        for elem in group:
            elem_size = len(elem.content)
            
            if current_size + elem_size > max_size and current_sub:
                sub_groups.append(current_sub)
                current_sub = [elem]
                current_size = elem_size
            else:
                current_sub.append(elem)
                current_size += elem_size
        
        if current_sub:
            sub_groups.append(current_sub)
        
        return sub_groups
    
    def _sliding_window_chunking(
        self,
        elements: List[DocumentElement]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """滑动窗口分块"""
        # 合并所有元素内容
        full_content = '\n\n'.join([elem.content for elem in elements])
        
        parent_chunks = []
        child_chunks = []
        
        # 创建父块
        parent_start = 0
        while parent_start < len(full_content):
            parent_end = min(parent_start + self.parent_size, len(full_content))
            
            # 尝试在句子边界断开
            if parent_end < len(full_content):
                last_period = full_content.rfind('。', parent_start, parent_end)
                if last_period > parent_start + self.parent_size // 2:
                    parent_end = last_period + 1
            
            parent_content = full_content[parent_start:parent_end]
            
            # 创建父块
            # 检查财务数据并提取指标
            contains_financial, financial_indicators = self._detect_financial_content_with_indicators(parent_content)
            parent = Chunk(
                content=parent_content,
                chunk_type=ChunkType.TEXT_GROUP,
                start_char=parent_start,
                end_char=parent_end,
                quality_score=self._calculate_content_quality(parent_content),
                contains_financial_data=contains_financial,
                financial_indicators=financial_indicators
            )
            parent_chunks.append(parent)
            
            # 创建子块
            children = self._create_sliding_child_chunks(
                parent_content,
                parent.chunk_id,
                parent_start
            )
            child_chunks.extend(children)
            parent.child_ids = [c.chunk_id for c in children]
            
            # 移动到下一个父块
            parent_start = parent_end - self.overlap if parent_end < len(full_content) else parent_end
        
        return parent_chunks, child_chunks
    
    def _fixed_size_chunking(
        self,
        elements: List[DocumentElement]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """固定大小分块"""
        # 简单实现，使用滑动窗口的逻辑
        return self._sliding_window_chunking(elements)
    
    def _create_text_parent_chunk(
        self,
        content: str,
        elements: List[DocumentElement]
    ) -> Chunk:
        """创建文本父块"""
        # 提取页面信息
        page_numbers = list(set(elem.page_number for elem in elements))
        
        # 构建元数据
        metadata = ChunkMetadata(
            page_numbers=sorted(page_numbers),
            extraction_method='text_semantic',
            source_file=getattr(self, 'source_file', '')
        )
        
        # 检查是否包含财务信息并提取指标
        contains_financial, financial_indicators = self._detect_financial_content_with_indicators(content)
        
        # 计算文档位置（基于元素索引）
        start_index = elements[0].index if elements and hasattr(elements[0], 'index') else 0
        end_index = elements[-1].index if elements and hasattr(elements[-1], 'index') else start_index + len(content)
        
        # 合并多个元素的bbox（取最小和最大坐标）
        bbox = None
        if elements and any(elem.bbox for elem in elements):
            valid_bboxes = [elem.bbox for elem in elements if elem.bbox]
            if valid_bboxes:
                x0 = min(bbox[0] for bbox in valid_bboxes)
                y0 = min(bbox[1] for bbox in valid_bboxes)
                x1 = max(bbox[2] for bbox in valid_bboxes)
                y1 = max(bbox[3] for bbox in valid_bboxes)
                bbox = (x0, y0, x1, y1)
        
        # 使用第一个元素的页码作为主页码
        page_number = elements[0].page_number if elements else None
        
        # 获取主要元素类型
        element_type = str(elements[0].element_type.value) if elements and hasattr(elements[0].element_type, 'value') else 'text'
        
        # 计算平均置信度
        avg_confidence = 0.0
        if elements:
            confidences = [elem.extraction_confidence for elem in elements if hasattr(elem, 'extraction_confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.95
        
        # 清理内容中的标记符号
        cleaned_content = self._clean_content_markers(content)
        
        chunk = Chunk(
            content=cleaned_content,
            chunk_type=ChunkType.TEXT_GROUP,
            metadata=metadata,
            quality_score=self._calculate_content_quality(content),
            contains_financial_data=contains_financial,
            financial_indicators=financial_indicators,
            start_char=start_index,
            end_char=end_index,
            # 添加坐标信息
            page_number=page_number,
            bbox=bbox,
            element_type=element_type,
            extraction_confidence=avg_confidence,
            # 文本块不是表格
            is_table=False,
            table_structure_preserved=False,
            table_metadata=None
        )
        
        # 提取关键词
        chunk.keywords = self._extract_keywords(content)
        
        return chunk
    
    def _create_text_child_chunks(
        self,
        content: str,
        parent_id: str,
        elements: Optional[List[DocumentElement]] = None
    ) -> List[Chunk]:
        """创建文本子块"""
        return self._create_sliding_child_chunks(content, parent_id, 0, elements)
    
    def _create_sliding_child_chunks(
        self,
        content: str,
        parent_id: str,
        global_offset: int = 0,
        elements: Optional[List[DocumentElement]] = None
    ) -> List[Chunk]:
        """使用滑动窗口创建子块"""
        children = []
        
        start = 0
        position = 0
        
        while start < len(content):
            # 如果不是第一个子块，尝试在句子边界开始
            if start > 0:
                # 在 overlap 区域内找到下一个句子开头
                search_end = min(start + self.overlap, len(content))
                for punct in ['。', '！', '？', '.', '!', '?']:
                    next_punct = content.find(punct, start, search_end)
                    if next_punct != -1 and next_punct + 1 < len(content):
                        start = next_punct + 1
                        # 跳过可能的空白和换行
                        while start < len(content) and content[start] in ' \n\t':
                            start += 1
                        break
            
            end = min(start + self.child_size, len(content))
            
            # 尝试在句子边界断开
            if end < len(content):
                # 查找句号、问号、感叹号
                for punct in ['。', '！', '？', '.', '!', '?']:
                    last_punct = content.rfind(punct, start, end)
                    if last_punct > start + self.child_size // 2:
                        end = last_punct + 1
                        break
            
            child_content = content[start:end]
            
            # 质量检查
            if len(child_content.strip()) >= self.min_chunk_size:
                # 检查子块是否包含财务数据并提取指标
                child_financial_data, child_financial_indicators = self._detect_financial_content_with_indicators(child_content)
                
                # 尝试从elements获取坐标信息
                page_number = None
                bbox = None
                element_type = 'text'
                extraction_confidence = 0.95
                
                if elements:
                    # 使用第一个元素的信息
                    page_number = elements[0].page_number if elements else None
                    bbox = elements[0].bbox if elements and elements[0].bbox else None
                    element_type = str(elements[0].element_type.value) if elements and hasattr(elements[0].element_type, 'value') else 'text'
                    extraction_confidence = elements[0].extraction_confidence if elements and hasattr(elements[0], 'extraction_confidence') else 0.95
                
                # 创建元数据
                child_metadata = ChunkMetadata(
                    extraction_method='text_semantic',
                    page_numbers=[page_number] if page_number else [],
                    source_file=getattr(self, 'source_file', '')
                )
                
                child = Chunk(
                    content=child_content,
                    chunk_type=ChunkType.CHILD,
                    parent_id=parent_id,
                    position=position,
                    start_char=global_offset + start,
                    end_char=global_offset + end,
                    quality_score=self._calculate_content_quality(child_content),
                    contains_financial_data=child_financial_data,
                    financial_indicators=child_financial_indicators,
                    # 添加坐标信息
                    page_number=page_number,
                    bbox=bbox,
                    element_type=element_type,
                    extraction_confidence=extraction_confidence,
                    metadata=child_metadata,
                    # 文本块不是表格
                    is_table=False,
                    table_structure_preserved=False,
                    table_metadata=None
                )
                
                # 提取关键词
                child.keywords = self._extract_keywords(child_content)
                
                children.append(child)
                position += 1
            
            # 移动窗口
            if end < len(content):
                start = end - self.overlap
            else:
                start = end
        
        return children
    
    def _establish_relationships(
        self,
        parent_chunks: List[Chunk],  # noqa: F841
        child_chunks: List[Chunk]
    ):
        """建立父子和兄弟关系"""
        # 按父块分组子块
        children_by_parent = {}
        for child in child_chunks:
            if child.parent_id:
                if child.parent_id not in children_by_parent:
                    children_by_parent[child.parent_id] = []
                children_by_parent[child.parent_id].append(child)
        
        # 设置兄弟关系
        for children in children_by_parent.values():
            # 按位置排序
            children.sort(key=lambda x: x.position)
            
            for child in children:
                sibling_ids = [c.chunk_id for c in children if c.chunk_id != child.chunk_id]
                child.sibling_ids = sibling_ids
    
    def _filter_low_quality_chunks(
        self, 
        chunks: List[Chunk], 
        chunk_type: str = "chunk"
    ) -> List[Chunk]:
        """
        过滤低质量分块
        
        Args:
            chunks: 待过滤的分块列表
            chunk_type: 分块类型标识（用于日志），如 "parent" 或 "child"
        """
        filtered = []
        
        for chunk in chunks:
            # 质量分数检查
            if chunk.quality_score < self.min_quality_score:
                self.logger.debug(f"过滤低质量分块: {chunk.chunk_id}, "
                                f"质量分={chunk.quality_score:.2f}")
                continue
            
            # 内容长度检查
            if len(chunk.content.strip()) < self.min_chunk_size:
                self.logger.debug(f"过滤过短分块: {chunk.chunk_id}, "
                                f"长度={len(chunk.content)}")
                continue
            
            filtered.append(chunk)
        
        self.logger.info(f"质量过滤 [{chunk_type}]: {len(chunks)} -> {len(filtered)}")
        return filtered
    
    def _calculate_quality_metrics(
        self,
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk]
    ):
        """计算所有分块的质量指标"""
        all_chunks = parent_chunks + child_chunks
        
        for chunk in all_chunks:
            # 计算信息密度
            chunk.metadata.information_density = chunk.calculate_information_density()
            
            # 计算连贯性分数
            chunk.metadata.coherence_score = chunk.calculate_coherence_score()
        
        self.logger.debug(f"计算了 {len(all_chunks)} 个分块的质量指标")
    
    def _calculate_content_quality(self, content: str) -> float:
        """计算内容质量"""
        if not content:
            return 0.0
        
        # 长度分数
        length_score = min(1.0, len(content) / 500)
        
        # 中文比例
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / len(content) if content else 0
        
        # 信息密度（基于标点符号）
        punctuation_count = sum(1 for c in content if c in '，。！？；：""''（）')
        density_score = min(1.0, punctuation_count / (len(content) / 50))
        
        # 综合评分
        return (length_score * 0.3 + chinese_ratio * 0.4 + density_score * 0.3)
    
    def _detect_financial_content_with_indicators(self, content: str) -> Tuple[bool, List[str]]:
        """检测内容是否包含财务信息并返回具体指标"""
        # 扩展财务关键词（包含更多财务相关词汇）
        financial_keywords = [
            # 基础财务概念
            '资产', '负债', '权益', '收入', '成本', '费用', '利润', '现金', '投资',
            # 会计科目
            '应收', '应付', '预收', '预付', '存货', '固定资产', '无形资产', '商誉',
            '营业收入', '营业成本', '销售费用', '管理费用', '研发费用', '财务费用',
            '净利润', '毛利润', '营业利润', '利润总额', '所得税', '净资产',
            # 财务报表
            '资产负债表', '利润表', '现金流量表', '股东权益', '合并', '母公司',
            # 财务指标
            '资产负债率', '流动比率', '速动比率', '毛利率', '净利率', 'ROE', 'ROA',
            # 业务相关
            '营业', '经营', '投资活动', '筹资活动', '股本', '资本公积', '未分配利润',
            # 数据相关
            '会计数据', '财务数据', '主要会计', '财务指标', '经营业绩', '财务状况',
            # 新增：股权和分配相关
            '股权', '股份', '股东', '分配', '分红', '红利', '派发', '转增', '配股',
            '每股', '总股本', '流通股', '限售股', '回购', '增发',
            # 新增：审计和报告相关
            '审计', '报告', '财务报告', '年度报告', '审计报告', '内控', '披露',
            # 新增：税务相关
            '税收', '税费', '税率', '增值税', '企业所得税', '关税',
            # 新增：其他财务概念
            '折旧', '摊销', '减值', '公允价值', '账面价值', '市值', '估值',
            # 投资相关
            '投资者', '融资', '借款', '债务', '价值', '证券', '债券'
        ]
        
        # 收集找到的财务指标
        found_indicators = []
        for kw in financial_keywords:
            if kw in content:
                found_indicators.append(kw)
        
        contains_financial = len(found_indicators) > 0
        
        # 额外检查：包含金额数字格式也算财务数据
        if not contains_financial:
            # 检查是否包含财务数字格式（如：123,456.78 万元）
            amount_patterns = [
                r'\d{1,3}(,\d{3})*\.?\d*\s*万元',
                r'\d{1,3}(,\d{3})*\.?\d*\s*亿元',  
                r'\d{1,3}(,\d{3})*\.?\d*\s*千万',
                r'\d{1,3}(,\d{3})*\.?\d*\s*元',
                r'\d+\.\d+%',  # 百分比
                r'￥\s*\d+',   # 人民币符号
                r'\$\s*\d+'    # 美元符号
            ]
            for pattern in amount_patterns:
                if re.search(pattern, content):
                    contains_financial = True
                    found_indicators.append('金额数据')
                    break
        
        # 去重并限制数量
        found_indicators = list(dict.fromkeys(found_indicators))[:10]  # 最多返回10个指标
        
        return contains_financial, found_indicators
    
    def _detect_financial_content(self, content: str) -> bool:
        """检测内容是否包含财务信息"""
        contains_financial, _ = self._detect_financial_content_with_indicators(content)
        return contains_financial
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """提取关键词（简单实现）"""
        keywords = []
        
        # 财务关键词
        financial_terms = [
            '资产', '负债', '权益', '收入', '成本', '利润',
            '现金流', '投资', '融资', '经营', '财务', '报表'
        ]
        
        for term in financial_terms:
            if term in content:
                keywords.append(term)
        
        # 提取数字相关
        numbers = re.findall(r'\d+\.?\d*[万亿百千]?元', content)
        keywords.extend(numbers[:3])  # 只取前3个
        
        return keywords[:max_keywords]
    
    def _calculate_numeric_summary(self, table_group: TableGroup) -> Dict[str, Any]:
        """计算表格数值统计摘要"""
        summary = {
            'total_values': 0,
            'numeric_values': 0,
            'min_value': None,
            'max_value': None,
            'has_negative_values': False,
            'has_percentage': False,
            'currency_detected': False
        }
        
        if not table_group or not table_group.table:
            return summary
        
        numeric_values = []
        
        # 分析表格内容
        if hasattr(table_group.table, 'rows'):
            for row in table_group.table.rows:
                for cell in row:
                    if isinstance(cell, str):
                        # 检测百分比
                        if '%' in cell or '％' in cell:
                            summary['has_percentage'] = True
                        
                        # 检测货币符号
                        if any(symbol in cell for symbol in ['¥', '$', '€', '￥', '元', '万元', '亿元']):
                            summary['currency_detected'] = True
                        
                        # 提取数值
                        numbers = re.findall(r'-?[\d,]+\.?\d*', cell.replace(',', ''))
                        for num_str in numbers:
                            try:
                                num = float(num_str)
                                numeric_values.append(num)
                                if num < 0:
                                    summary['has_negative_values'] = True
                            except ValueError:
                                pass
        
        summary['total_values'] = len(numeric_values)
        summary['numeric_values'] = len(numeric_values)
        
        if numeric_values:
            summary['min_value'] = min(numeric_values)
            summary['max_value'] = max(numeric_values)
        
        return summary


class ChineseFinancialChunker(SmartChunker):
    """中文财报专用分块器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化财报分块器"""
        super().__init__(config)
        
        # 财报特定参数
        self.report_sections = [
            '资产负债表', '利润表', '现金流量表',
            '股东权益变动表', '财务报表附注'
        ]
        
        # 调整分块大小（财报通常需要更大的上下文）
        self.parent_size = self.config.get('parent_size', 4000)
        self.child_size = self.config.get('child_size', 1200)
        
        self.logger.info("中文财报分块器初始化完成")
    
    def create_chunks(
        self,
        elements: List[DocumentElement],
        table_groups: List[TableGroup],
        source_file: str = ""
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """创建财报分块（重写）"""
        # 保存源文件路径
        self.source_file = source_file
        # 预处理：按报告章节分组
        section_groups = self._group_by_report_section(elements)
        
        parent_chunks = []
        child_chunks = []
        
        # 1. 优先处理重要财务报表
        important_tables = [
            g for g in table_groups 
            if g.table_type.value in ['balance_sheet', 'income_statement', 'cash_flow']
        ]
        
        if important_tables:
            table_parents, table_children = self._process_important_tables(important_tables)
            parent_chunks.extend(table_parents)
            child_chunks.extend(table_children)
        
        # 2. 处理其他表格
        other_tables = [
            g for g in table_groups 
            if g not in important_tables
        ]
        
        if other_tables:
            other_parents, other_children = self._process_table_groups(other_tables)
            parent_chunks.extend(other_parents)
            child_chunks.extend(other_children)
        
        # 3. 按章节处理文本
        for section_name, section_elements in section_groups.items():
            section_parents, section_children = self._process_section(
                section_name, 
                section_elements,
                table_groups
            )
            parent_chunks.extend(section_parents)
            child_chunks.extend(section_children)
        
        # 4. 建立关系和质量控制
        self._establish_relationships(parent_chunks, child_chunks)
        parent_chunks = self._filter_low_quality_chunks(parent_chunks, chunk_type="parent")
        child_chunks = self._filter_low_quality_chunks(child_chunks, chunk_type="child")
        
        # 5. 添加财报特定元数据
        self._enrich_financial_metadata(parent_chunks, child_chunks)
        
        return parent_chunks, child_chunks
    
    def _group_by_report_section(
        self,
        elements: List[DocumentElement]
    ) -> Dict[str, List[DocumentElement]]:
        """按报告章节分组"""
        section_groups = {'其他': []}
        current_section = '其他'
        
        for elem in elements:
            # 检查是否为章节标题
            for section in self.report_sections:
                if section in elem.content[:100]:
                    current_section = section
                    if section not in section_groups:
                        section_groups[section] = []
                    break
            
            # 添加到当前章节
            if current_section not in section_groups:
                section_groups[current_section] = []
            section_groups[current_section].append(elem)
        
        # 移除空章节
        return {k: v for k, v in section_groups.items() if v}
    
    def _process_important_tables(
        self,
        table_groups: List[TableGroup]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """处理重要财务报表（保持完整）"""
        parent_chunks = []
        child_chunks = []
        
        for group in table_groups:
            # 重要报表作为单独的父块
            parent = self._create_financial_statement_chunk(group)
            parent_chunks.append(parent)
            
            # 创建结构化子块
            children = self._create_structured_table_chunks(group, parent.chunk_id)
            child_chunks.extend(children)
            parent.child_ids = [c.chunk_id for c in children]
        
        return parent_chunks, child_chunks
    
    def _create_financial_statement_chunk(self, group: TableGroup) -> Chunk:
        """创建财务报表分块"""
        chunk = self._create_table_parent_chunk(group)
        
        # 添加财报特定信息
        chunk.metadata.report_type = '财务报表'
        chunk.topics = [group.table_type.value]
        
        # 提取实体
        if group.accounting_items:
            chunk.entities = [
                {'type': 'accounting_item', 'value': item}
                for item in group.accounting_items[:10]  # 限制数量
            ]
        
        return chunk
    
    def _create_structured_table_chunks(
        self,
        group: TableGroup,
        parent_id: str
    ) -> List[Chunk]:
        """创建结构化表格子块"""
        children = []
        
        # 按会计科目或时间期间分割
        if group.accounting_items:
            # 按会计科目分组创建子块
            item_chunks = self._chunk_by_accounting_items(group, parent_id)
            children.extend(item_chunks)
        elif group.time_periods:
            # 按时间期间分组创建子块
            period_chunks = self._chunk_by_time_periods(group, parent_id)
            children.extend(period_chunks)
        else:
            # 使用默认分割
            children = self._create_table_child_chunks(group, parent_id)
        
        return children
    
    def _chunk_by_accounting_items(
        self,
        group: TableGroup,
        parent_id: str
    ) -> List[Chunk]:
        """按会计科目分块"""
        # 简化实现：将表格内容按科目关键词分段
        chunks = []
        content = group.table.content if group.table else ""
        
        if not content:
            return chunks
        
        # 按行分割
        lines = content.split('\n')
        current_item = None
        current_lines = []
        
        for line in lines:
            # 检查是否包含新的会计科目
            new_item = None
            for item in group.accounting_items:
                if item in line:
                    new_item = item
                    break
            
            if new_item and new_item != current_item:
                # 保存当前块
                if current_lines:
                    chunk_content = '\n'.join(current_lines)
                    # 检查财务数据并提取更多指标
                    contains_financial, financial_indicators = self._detect_financial_content_with_indicators(chunk_content)
                    # 添加当前会计科目
                    if current_item and current_item not in financial_indicators:
                        financial_indicators.append(current_item)
                    
                    # 创建表格元数据
                    table_meta = TableMetadata(
                        table_dimensions="",
                        has_header=False,
                        table_type=group.table_type.value if group.table_type else 'financial_statement',
                        high_value=group.table.quality_score > 0.9 if group.table else True,
                        extraction_confidence=group.table.extraction_confidence if group.table and hasattr(group.table, 'extraction_confidence') else 0.95,
                        row_count=len(current_lines),
                        column_count=0,
                        contains_financial_data=True,
                        accounting_items=financial_indicators[:5] if financial_indicators else [],
                        time_periods=group.time_periods[:3] if hasattr(group, 'time_periods') and group.time_periods else [],
                        numeric_summary={}
                    )
                    
                    # 创建元数据
                    chunk_metadata = ChunkMetadata(
                        extraction_method='table_split',
                        page_numbers=[group.table.page_number] if group.table and group.table.page_number else [],
                        source_file=getattr(self, 'source_file', '')
                    )
                    
                    chunk = Chunk(
                        content=chunk_content,
                        chunk_type=ChunkType.TABLE_GROUP,
                        parent_id=parent_id,
                        position=len(chunks),
                        quality_score=group.table.quality_score,
                        financial_indicators=financial_indicators,
                        contains_financial_data=True,  # 表格子块通常包含财务数据
                        start_char=len(chunks) * 300,  # 估算位置
                        end_char=(len(chunks) + 1) * 300 + len(chunk_content),
                        # 表格相关字段
                        is_table=True,
                        table_structure_preserved=True,
                        table_metadata=table_meta,
                        metadata=chunk_metadata,
                        # 添加坐标信息
                        page_number=group.table.page_number if group.table else None,
                        bbox=group.table.bbox if group.table else None,
                        element_type='table',
                        extraction_confidence=group.table.extraction_confidence if group.table else 0.95
                    )
                    chunks.append(chunk)
                
                # 开始新块
                current_item = new_item
                current_lines = [line]
            else:
                current_lines.append(line)
        
        # 处理最后一块
        if current_lines:
            chunk_content = '\n'.join(current_lines)
            # 检查财务数据并提取更多指标
            contains_financial, financial_indicators = self._detect_financial_content_with_indicators(chunk_content)
            # 添加当前会计科目
            if current_item and current_item not in financial_indicators:
                financial_indicators.append(current_item)
            
            # 创建表格元数据
            table_meta = TableMetadata(
                table_dimensions="",
                has_header=False,
                table_type=group.table_type.value if group.table_type else 'financial_statement',
                high_value=group.table.quality_score > 0.9 if group.table else True,
                extraction_confidence=group.table.extraction_confidence if group.table and hasattr(group.table, 'extraction_confidence') else 0.95,
                row_count=len(current_lines),
                column_count=0,
                contains_financial_data=True,
                accounting_items=financial_indicators[:5] if financial_indicators else [],
                time_periods=group.time_periods[:3] if hasattr(group, 'time_periods') and group.time_periods else [],
                numeric_summary={}
            )
            
            # 创建元数据
            chunk_metadata = ChunkMetadata(
                extraction_method='table_split',
                page_numbers=[group.table.page_number] if group.table and group.table.page_number else [],
                source_file=getattr(self, 'source_file', '')
            )
            
            chunk = Chunk(
                content=chunk_content,
                chunk_type=ChunkType.TABLE_GROUP,
                parent_id=parent_id,
                position=len(chunks),
                quality_score=group.table.quality_score,
                financial_indicators=financial_indicators,
                contains_financial_data=True,  # 表格子块通常包含财务数据
                start_char=len(chunks) * 300,  # 估算位置
                end_char=(len(chunks) + 1) * 300 + len(chunk_content),
                # 表格相关字段
                is_table=True,
                table_structure_preserved=True,
                table_metadata=table_meta,
                metadata=chunk_metadata,
                # 添加坐标信息
                page_number=group.table.page_number if group.table else None,
                bbox=group.table.bbox if group.table else None,
                element_type='table',
                extraction_confidence=group.table.extraction_confidence if group.table else 0.95
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_time_periods(
        self,
        group: TableGroup,
        parent_id: str
    ) -> List[Chunk]:
        """按时间期间分块"""
        # 类似按会计科目分块的逻辑
        chunks = []
        
        # 简化实现：均匀分割
        if group.table and group.table.content:
            content = group.table.content
            period_count = len(group.time_periods)
            
            if period_count > 0:
                lines = content.split('\n')
                lines_per_period = max(1, len(lines) // period_count)
                
                for i, period in enumerate(group.time_periods):
                    start_idx = i * lines_per_period
                    end_idx = start_idx + lines_per_period if i < period_count - 1 else len(lines)
                    
                    chunk_lines = lines[start_idx:end_idx]
                    chunk_content = '\n'.join(chunk_lines)
                    
                    # 检查财务数据并提取指标
                    contains_financial, financial_indicators = self._detect_financial_content_with_indicators(chunk_content)
                    
                    chunk = Chunk(
                        content=chunk_content,
                        chunk_type=ChunkType.TABLE_GROUP,
                        parent_id=parent_id,
                        position=i,
                        quality_score=group.table.quality_score,
                        contains_financial_data=contains_financial,
                        financial_indicators=financial_indicators,
                        start_char=i * 300,  # 估算位置
                        end_char=(i + 1) * 300 + len(chunk_content),
                        # 添加坐标信息
                        page_number=group.table.page_number if group.table else None,
                        bbox=group.table.bbox if group.table else None,
                        element_type='table',
                        extraction_confidence=group.table.extraction_confidence if group.table else 0.95
                    )
                    
                    # 添加时间期间信息
                    chunk.metadata.fiscal_year = self._extract_year_from_period(period)
                    chunks.append(chunk)
        
        return chunks if chunks else self._create_table_child_chunks(group, parent_id)
    
    def _extract_year_from_period(self, period: str) -> Optional[int]:
        """从期间字符串提取年份"""
        year_match = re.search(r'(\d{4})', period)
        if year_match:
            return int(year_match.group(1))
        return None
    
    def _process_section(
        self,
        section_name: str,
        elements: List[DocumentElement],
        table_groups: List[TableGroup]
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """处理报告章节"""
        # 过滤已处理的表格元素
        table_element_ids = set()
        for group in table_groups:
            if group.table:
                table_element_ids.add(group.table.element_id)
        
        text_elements = [e for e in elements if e.element_id not in table_element_ids]
        
        if not text_elements:
            return [], []
        
        # 使用语义分块
        parent_chunks, child_chunks = self._semantic_chunking(text_elements)
        
        # 添加章节信息
        for chunk in parent_chunks + child_chunks:
            chunk.metadata.report_type = section_name
            chunk.topics.append(section_name)
        
        return parent_chunks, child_chunks
    
    def _enrich_financial_metadata(
        self,
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk]
    ):
        """增强财报元数据"""
        all_chunks = parent_chunks + child_chunks
        
        for chunk in all_chunks:
            # 提取公司名称 - 扩展支持更多公司类型，支持括号内容和特殊格式
            company_pattern = r'([\u4e00-\u9fff]+(?:股份|集团|控股|科技|技术|金融|银行|证券|保险|地产|能源|医药|电子|通信|汽车|钢铁|化工|实业|投资|建设|贸易|物流|文化|教育|农业|航空|铁路|发展|企业)(?:（[^）]*）)?(?:有限|有限责任)?(?:股份)?(?:公司|企业))'
            company_match = re.search(company_pattern, chunk.content)
            if company_match:
                chunk.metadata.company_code = company_match.group(1)
            
            # 提取股票代码（6位数字）
            if not chunk.metadata.company_code or not chunk.metadata.company_code.isdigit():
                stock_code_patterns = [
                    r'(?:股票代码|证券代码)[：:\s]*(\d{6})',  # 股票代码：300617
                    r'[（(](\d{6})[)）]',                    # (300617) 或 （300617）
                ]
                for pattern in stock_code_patterns:
                    code_match = re.search(pattern, chunk.content)
                    if code_match:
                        code = code_match.group(1)
                        # 验证是有效的股票代码范围
                        if code.startswith(('0', '3', '6', '8')):
                            # 同时存储股票代码
                            if not hasattr(chunk.metadata, 'stock_code'):
                                chunk.metadata.stock_code = code
                            break
            
            # 提取年份
            if not chunk.metadata.fiscal_year:
                year_match = re.search(r'(\d{4})年', chunk.content)
                if year_match:
                    chunk.metadata.fiscal_year = int(year_match.group(1))
            
            # 计算信息密度
            chunk.metadata.information_density = chunk.calculate_information_density()
            
            # 计算连贯性分数
            chunk.metadata.coherence_score = chunk.calculate_coherence_score()