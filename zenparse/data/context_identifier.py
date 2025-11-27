"""
表格上下文识别器 - 中文财报优化

识别表格的标题、注释和相关上下文，解决表格与说明分离的核心问题。
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from .models import DocumentElement, TableGroup, ElementType, TableType
from .accounting_domain import (
    STANDARD_ACCOUNTING_ITEMS,
    BROAD_ACCOUNTING_CATEGORIES,
    COMMON_ACCOUNTING_ITEMS,
)


class TableContextIdentifier:
    """表格上下文识别器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化识别器"""
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 上下文搜索范围 - 增加搜索范围以找到更远的标题
        self.context_before = self.config.get('context_before', 10)  # 向前查找10个元素，以捕获多级标题
        self.context_after = self.config.get('context_after', 3)   # 向后查找3个元素
        self.max_title_distance = self.config.get('max_title_distance', 200)  # 增加标题最大字符距离
        
        # 表头/科目层级保留开关与限制
        table_cfg = {}
        if isinstance(self.config, dict):
            table_cfg = self.config.get('table', {}) or self.config.get('chunking', {}).get('table', {})
        self.include_header_hierarchy = table_cfg.get('include_header_hierarchy', True)
        self.hierarchy_max_depth = table_cfg.get('hierarchy_max_depth', 4)
        self.hierarchy_max_length = table_cfg.get('hierarchy_max_length', 50)
        self.skip_numeric_rows = table_cfg.get('skip_numeric_rows', True)
        
        # 财报特定的标题模式（基于你的例子）
        self.financial_title_patterns = [
            r'[一二三四五六七八九十\d]+、\s*.*(?:会计数据|财务指标|主要.*表)',
            r'\([一二三四五六七八九十\d]+\)\s*.*(?:会计数据|财务指标)',
            r'.*(?:近.年|年度).*(?:会计数据|财务指标)',
            r'主要会计数据.*',
            r'主要财务指标.*',
            r'第.+节\s+.*',  # 第X节 标题
            r'[一二三四五六七八九十]+、\s+\S+',  # 一、标题
            r'\([一二三四五六七八九十]+\)\s*\S+'  # (一) 标题
        ]
        
        # 标题关键词（增强版）
        self.title_keywords = [
            '表', '图表', 'Table', 'Figure', '如下', '以下',
            '明细', '清单', '汇总', '统计', '一览',
            '会计数据', '财务指标', '资产负债', '利润', '现金流',
            '股东权益', '营业收入', '净利润'
        ]
        
        # 注释关键词
        self.note_keywords = [
            '注:', '注：', '注释:', '说明:', '备注:',
            'Note:', '说明：', '备注：', '附注：',
            '*', '※', '★'
        ]
        
        # 单位关键词（增强版）
        self.unit_keywords = [
            '单位:', '单位：', '币种:', '币种：',
            '单位：元', '单位：万元', '单位：千元', '单位：百万元',
            '币种：人民币', '币种：美元',
            '人民币', '美元', '万元', '百万', '千元',
            '元', '股', '%', '‰'
        ]
    
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
    
    def identify_table_groups(self, elements: List[DocumentElement]) -> List[TableGroup]:
        """识别表格及其上下文"""
        table_groups = []
        processed_indices = set()  # 记录已处理的元素索引
        
        for i, elem in enumerate(elements):
            if i in processed_indices:
                continue
                
            if elem.element_type != ElementType.TABLE:
                continue
            
            # 创建表格组
            group = TableGroup(table=elem)
            
            # 查找上下文
            title_idx = self._find_title(elements, i, processed_indices)
            if title_idx is not None:
                group.title = elements[title_idx]
                processed_indices.add(title_idx)
                elements[title_idx].context_role = "title"
            
            note_idx = self._find_note(elements, i, processed_indices)
            if note_idx is not None:
                group.note = elements[note_idx]
                processed_indices.add(note_idx)
                elements[note_idx].context_role = "note"
            
            caption_idx = self._find_caption(elements, i, processed_indices)
            if caption_idx is not None:
                group.caption = elements[caption_idx]
                processed_indices.add(caption_idx)
                elements[caption_idx].context_role = "caption"
            
            # 生成统一内容
            group.unified_content = self._unify_group_content(group)
            group.unified_markdown = self._generate_markdown(group)
            
            # 构建行级表头/科目层级路径
            if self.include_header_hierarchy:
                self._populate_row_header_paths(group)
            
            # 计算质量分数
            group.completeness_score = self._calculate_completeness(group)
            group.structure_score = elem.quality_score
            group.context_quality = self._calculate_context_quality(group)
            
            processed_indices.add(i)
            table_groups.append(group)
            
            self.logger.info(f"识别表格组 {group.group_id}: "
                           f"标题={group.title is not None}, "
                           f"注释={group.note is not None}, "
                           f"说明={group.caption is not None}")
        
        self.logger.info(f"共识别 {len(table_groups)} 个表格组")
        return table_groups
    
    def _find_title(self, elements: List[DocumentElement], table_idx: int, 
                   processed: set) -> Optional[int]:
        """查找表格标题 - 支持多级标题识别"""
        table_elem = elements[table_idx]
        title_candidates = []
        
        # 向前查找可能的标题元素
        for j in range(max(0, table_idx - self.context_before), table_idx):
            if j in processed:
                continue
                
            elem = elements[j]
            
            # 跳过其他表格
            if elem.element_type == ElementType.TABLE:
                continue
            
            # 检查是否匹配财报标题模式
            content = elem.content.strip() if elem.content else ""
            
            if not content:
                continue
            
            for pattern in self.financial_title_patterns:
                if re.search(pattern, content):
                    # 计算匹配分数
                    score = self._calculate_title_score(elem, table_elem, table_idx - j)
                    title_candidates.append((j, score, 'pattern'))
                    break
            
            # 检查是否包含单位/币种信息
            if any(keyword in content for keyword in self.unit_keywords):
                score = self._calculate_title_score(elem, table_elem, table_idx - j) * 0.8
                title_candidates.append((j, score, 'unit'))
            
            # 检查是否包含标题关键词
            elif any(keyword in content for keyword in self.title_keywords):
                score = self._calculate_title_score(elem, table_elem, table_idx - j) * 0.7
                title_candidates.append((j, score, 'keyword'))
        
        # 选择最佳候选（支持多个标题）
        if title_candidates:
            # 按分数排序
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 返回最高分的标题
            best_idx, best_score, match_type = title_candidates[0]
            
            # 如果是多级标题（如"七、" 和 "(一)"），尝试合并
            if match_type == 'pattern' and best_score >= 0.5:
                # 检查是否有相邻的上级标题
                for idx, score, mtype in title_candidates[1:]:
                    if abs(idx - best_idx) <= 2 and mtype == 'pattern':
                        # 合并多级标题内容
                        self._merge_multi_level_titles(elements, best_idx, idx)
                        break
                
                return best_idx
        
        return None
    
    def _find_note(self, elements: List[DocumentElement], table_idx: int,
                  processed: set) -> Optional[int]:
        """查找表格注释"""
        # 向后查找
        for j in range(table_idx + 1, min(len(elements), table_idx + self.context_after + 1)):
            if j in processed:
                continue
                
            elem = elements[j]
            if self._is_table_note(elem):
                return j
        
        return None
    
    def _find_caption(self, elements: List[DocumentElement], table_idx: int,
                     processed: set) -> Optional[int]:
        """查找表格说明"""
        # 可以在表格前后查找
        search_range = list(range(max(0, table_idx - 1), table_idx)) + \
                      list(range(table_idx + 1, min(len(elements), table_idx + 2)))
        
        for j in search_range:
            if j in processed:
                continue
                
            elem = elements[j]
            if self._is_table_caption(elem):
                return j
        
        return None
    
    def _is_table_title(self, elem: DocumentElement, table_elem: DocumentElement) -> bool:
        """判断是否为表格标题"""
        if elem.element_type == ElementType.TABLE:
            return False
        
        content = elem.content.lower() if elem.content else ""
        
        # 长度限制
        if len(content) > 200:
            return False
        
        # 关键词匹配
        has_keyword = any(kw in content for kw in self.title_keywords)
        
        # 位置关系（标题通常在表格上方）
        if elem.page_number > table_elem.page_number:
            return False
        
        # 检查是否为编号标题
        is_numbered = bool(re.match(r'^[\d一二三四五六七八九十]+[\.、]', content))
        
        return has_keyword or is_numbered
    
    def _is_table_note(self, elem: DocumentElement) -> bool:
        """判断是否为表格注释"""
        if elem.element_type == ElementType.TABLE:
            return False
        
        content = elem.content if elem.content else ""
        
        # 长度限制
        if len(content) > 500:
            return False
        
        # 关键词匹配
        return any(kw in content for kw in self.note_keywords)
    
    def _merge_multi_level_titles(self, elements: List[DocumentElement], idx1: int, idx2: int):
        """合并多级标题内容"""
        # 确保idx1是较小的索引
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # 合并内容
        elem1 = elements[idx1]
        elem2 = elements[idx2]
        
        # 将两个标题内容合并，保持层级关系
        merged_content = f"{elem1.content.strip()}\n{elem2.content.strip()}"
        elem1.content = merged_content
        
        # 标记已合并
        elem1.metadata['is_multi_level_title'] = True
        elem1.metadata['merged_with'] = idx2
    
    def _is_table_caption(self, elem: DocumentElement) -> bool:
        """判断是否为表格说明"""
        if elem.element_type == ElementType.TABLE:
            return False
        
        content = elem.content if elem.content else ""
        
        # 单位说明
        if any(kw in content for kw in self.unit_keywords):
            return True
        
        # 短说明文本
        if 20 < len(content) < 100:
            # 检查是否包含括号说明
            if '(' in content or '（' in content:
                return True
        
        return False
    
    def _unify_group_content(self, group: TableGroup) -> str:
        """统一表格组内容 - 保持清晰的结构"""
        parts = []
        
        # 1. 标题部分
        if group.title:
            title_content = group.title.content.strip()
            parts.append(f"【标题】{title_content}")
        
        # 2. 说明部分（如单位、币种等）
        if group.caption:
            caption_content = group.caption.content.strip()
            parts.append(f"【说明】{caption_content}")
        
        # 3. 表格主体内容
        if group.table:
            table_content = group.table.content
            # 如果表格内容是markdown格式，保持原样
            if table_content.startswith('|') or '\n|' in table_content:
                parts.append(f"【表格内容】\n{table_content}")
            else:
                # 尝试清理和格式化
                cleaned_content = self._clean_table_content(table_content)
                parts.append(f"【表格内容】\n{cleaned_content}")
        
        # 4. 注释部分
        if group.note:
            note_content = group.note.content.strip()
            parts.append(f"【注释】{note_content}")
        
        return '\n\n'.join(parts)

    # =========================
    # 表头/科目层级处理
    # =========================
    def _populate_row_header_paths(self, group: TableGroup):
        """基于 raw_data 或内容行生成行级层级路径"""
        if not group.table:
            return
        
        paths: List[str] = []
        
        # 优先使用 raw_data（来自 table_processor 注入的 metadata）
        raw_data = None
        if isinstance(getattr(group.table, 'metadata', None), dict):
            raw_data = group.table.metadata.get('raw_data')
        
        if raw_data and isinstance(raw_data, list):
            paths = self._build_row_header_paths(raw_data)
        
        # 回退：用内容行作为单列表格进行层级识别
        if not paths and group.table.content:
            lines = [line for line in group.table.content.split('\n') if line.strip()]
            pseudo_raw = [[line] for line in lines]
            paths = self._build_row_header_paths(pseudo_raw)
        
        if paths:
            group.row_header_paths = paths
            # 便于后续阶段直接读取
            if isinstance(getattr(group.table, 'metadata', None), dict):
                group.table.metadata['row_header_paths'] = paths
    
    def _build_row_header_paths(self, raw_data: List[List[Any]]) -> List[str]:
        """从原始表格数据构建每行的层级路径"""
        paths: List[str] = []
        stack: List[Tuple[int, str]] = []
        
        for row in raw_data:
            cell = ""
            if row:
                cell = str(row[0] or "").rstrip()
            
            # 处理空首列但有数据的行（合并单元格延续）
            if not cell and row and any(str(c or "").strip() for c in row[1:]):
                paths.append(" / ".join(name for _, name in stack) if stack else "")
                continue
            
            # 跳过全空行
            if not cell:
                paths.append(" / ".join(name for _, name in stack) if stack else "")
                continue
            
            # 跳过纯数字/百分比行
            if self.skip_numeric_rows and re.match(r'^[\d,.\-%\s]+$', cell):
                paths.append(" / ".join(name for _, name in stack) if stack else "")
                continue
            
            level, normalized = self._detect_row_level(cell)
            name = self._clean_row_name(normalized)
            if not name:
                paths.append(" / ".join(name for _, name in stack) if stack else "")
                continue
            
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
            
            trimmed_path = " / ".join(name for _, name in stack[: self.hierarchy_max_depth])
            paths.append(trimmed_path[: self.hierarchy_max_length])
        
        return paths
    
    def _detect_row_level(self, cell_text: str) -> Tuple[int, str]:
        """检测单元格层级，返回(level, 去掉缩进/编号的文本)"""
        text = cell_text or ""
        # 计算缩进（包含全角空格）
        leading = len(text) - len(text.lstrip(' \t　'))
        level_from_indent = leading // 2
        stripped = text.lstrip(' \t　')
        
        level = level_from_indent
        
        if re.match(r'^[一二三四五六七八九十]+、', stripped):
            level = max(level, 1)
        if re.match(r'^[（(][一二三四五六七八九十]+[)）]', stripped):
            level = max(level, 2)
        if re.match(r'^\d+[、.．]', stripped):
            level = max(level, 3)
        if stripped.startswith(('-', '•', '·', '└', '├', '─')):
            level = max(level, level_from_indent + 1)
            stripped = stripped[1:].strip()
        
        return level, stripped
    
    def _clean_row_name(self, text: str) -> str:
        """清理行名称中的编号/符号"""
        name = re.sub(r'^[一二三四五六七八九十]+、', '', text)
        name = re.sub(r'^[（(][一二三四五六七八九十]+[)）]', '', name)
        name = re.sub(r'^\d+[、.．]', '', name)
        name = name.lstrip('-•·└├─').strip()
        return name
    
    def _clean_table_content(self, content: str) -> str:
        """清理表格内容，去除乱码和噪声"""
        if not content:
            return content
        
        cleaned = content
        
        # 移除页眉/页脚等重复噪声（防止混入表格内容）
        header_pattern = r'^\*?[^\n]{0,80}年度报告全文[\s\n]*'
        cleaned = re.sub(header_pattern, '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\d+\s*$', '', cleaned, flags=re.MULTILINE)  # 纯页码
        cleaned = re.sub(r'^\d+\s*[/／]\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)  # 1/100
        cleaned = re.sub(r'^第\s*\d+\s*页\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^共\s*\d+\s*页\s*$', '', cleaned, flags=re.MULTILINE)
        
        # 移除行首的孤立 * 标记（PDF 格式伪影）
        cleaned = re.sub(r'^\*(?=[\u4e00-\u9fff□√一二三四五六七八九十（\-0-9])', '', cleaned, flags=re.MULTILINE)
        
        # 移除连续的特殊字符（保留中文、英文、数字、常见标点和括号）
        cleaned = re.sub('[^\\u4e00-\\u9fff\\u0020-\\u007e\\n\\r\\t，。！？；：""''（）【】\\[\\]|%-]+', ' ', cleaned)
        
        # 移除多余的空格
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # 移除多余的换行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def _generate_markdown(self, group: TableGroup) -> str:
        """生成Markdown格式内容"""
        parts = []
        
        if group.title:
            parts.append(f"### {group.title.content}\n")
        
        if group.caption:
            parts.append(f"*{group.caption.content}*\n")
        
        if group.table:
            # 格式化表格内容为标准 Markdown
            formatted_content = self._format_table_content(group.table.content)
            parts.append(formatted_content)
        
        if group.note:
            parts.append(f"\n> 注：{group.note.content}")
        
        return '\n'.join(parts)
    
    def _format_table_content(self, content: str) -> str:
        """
        将表格内容格式化为标准 Markdown 表格格式
        
        处理多种输入格式:
        - pipe 分隔 (| cell | cell |)
        - tab 分隔
        - 多空格分隔
        - 混合格式（文本 + 表格）
        """
        if not content:
            return ""
        
        # 1. 清理特殊标记（如 *...* 格式）
        content = self._clean_special_markers(content)
        
        # 2. 分离文本部分和表格部分
        text_part, table_part = self._separate_text_and_table(content)
        
        # 3. 格式化表格部分
        formatted_table = self._format_table_rows(table_part) if table_part else ""
        
        # 4. 组合结果
        result_parts = []
        if text_part.strip():
            result_parts.append(text_part.strip())
        if formatted_table:
            result_parts.append(formatted_table)
        
        return "\n\n".join(result_parts) if result_parts else content
    
    def _clean_special_markers(self, content: str) -> str:
        """清理特殊标记"""
        # 移除 *...* 包裹标记（常见于 PDF 提取）
        content = re.sub(r'^\*(.+?)\*$', r'\1', content, flags=re.DOTALL)
        content = re.sub(r'^\*', '', content)
        content = re.sub(r'\*$', '', content)
        
        # 移除文本中的 *...* 标记（行尾的页码+星号）
        content = re.sub(r'\d+。\*\s*\n', '\n', content)
        content = re.sub(r'\d+。\*\s*$', '', content)
        
        # 移除孤立的星号
        content = re.sub(r'\s*\*\s*$', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _separate_text_and_table(self, content: str) -> Tuple[str, str]:
        """
        分离文本内容和表格内容
        
        Returns:
            Tuple[text_part, table_part]
        """
        lines = content.split('\n')
        text_lines = []
        table_lines = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # 检测是否是表格行（包含 | 且不是纯文本）
            is_table_line = self._is_table_line(stripped)
            
            if is_table_line:
                in_table = True
                table_lines.append(stripped)
            elif in_table and self._could_be_table_continuation(stripped):
                # 可能是表格的延续行（如跨行单元格）
                table_lines.append(stripped)
            else:
                if not in_table:
                    text_lines.append(stripped)
                else:
                    # 已经进入表格后遇到非表格行，可能是表格结束
                    # 保留到文本部分
                    text_lines.append(stripped)
        
        return '\n'.join(text_lines), '\n'.join(table_lines)
    
    def _is_table_line(self, line: str) -> bool:
        """判断是否为表格行"""
        # 包含多个 | 分隔符
        if line.count('|') >= 2:
            return True
        
        # markdown 分隔线
        if re.match(r'^[\|\s\-:]+$', line) and '---' in line:
            return True
        
        return False
    
    def _could_be_table_continuation(self, line: str) -> bool:
        """判断是否可能是表格的延续行"""
        # 如果行很短且像是单元格内容，可能是跨行内容
        if len(line) < 50 and not any(c in line for c in ['。', '！', '？']):
            return True
        return False
    
    def _format_table_rows(self, table_content: str) -> str:
        """将表格内容格式化为标准 Markdown"""
        if not table_content:
            return ""
        
        lines = table_content.split('\n')
        parsed_rows = []
        has_separator = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过已有的分隔线
            if re.match(r'^[\|\s\-:]+$', line) and '---' in line:
                has_separator = True
                continue
            
            # 解析 pipe 分隔的行
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                # 移除首尾空元素
                while cells and cells[0] == '':
                    cells = cells[1:]
                while cells and cells[-1] == '':
                    cells = cells[:-1]
                if cells:
                    # 清理：移除连续的空单元格，保留有内容的单元格
                    cleaned_cells = self._clean_empty_cells(cells)
                    if cleaned_cells:
                        parsed_rows.append(cleaned_cells)
            elif '\t' in line:
                cells = [cell.strip() for cell in line.split('\t')]
                if cells:
                    parsed_rows.append(cells)
            elif re.search(r'\s{2,}', line):
                cells = [cell.strip() for cell in re.split(r'\s{2,}', line)]
                if cells and len(cells) > 1:  # 只有多列才算表格行
                    parsed_rows.append(cells)
        
        if not parsed_rows:
            return table_content
        
        # 如果已经有正确的 markdown 格式，也进行空单元格清理
        if has_separator:
            return self._clean_markdown_table(table_content)
        
        # 标准化列数
        max_cols = max(len(row) for row in parsed_rows) if parsed_rows else 0
        for row in parsed_rows:
            while len(row) < max_cols:
                row.append("")
        
        # 生成标准 Markdown 表格
        result_lines = []
        
        if parsed_rows:
            # 第一行作为表头
            headers = parsed_rows[0]
            result_lines.append("| " + " | ".join(headers) + " |")
            result_lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
            
            # 其余行作为数据
            for row in parsed_rows[1:]:
                result_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(result_lines)
    
    def _clean_empty_cells(self, cells: List[str]) -> List[str]:
        """清理连续的空单元格，合并相邻的空格"""
        if not cells:
            return cells
        
        cleaned = []
        prev_empty = False
        
        for cell in cells:
            is_empty = cell.strip() == ''
            if is_empty:
                if not prev_empty:
                    # 第一个空单元格保留（可能是格式需要）
                    cleaned.append('')
                prev_empty = True
            else:
                cleaned.append(cell)
                prev_empty = False
        
        # 如果清理后只剩空单元格，返回原始
        if all(c == '' for c in cleaned):
            return cells
        
        return cleaned
    
    def _clean_markdown_table(self, content: str) -> str:
        """清理已有的 Markdown 表格，移除过多的空单元格"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if '|' in line:
                # 移除连续的空单元格 |  |  |  | -> |  |
                line = re.sub(r'(\|\s*){3,}', '| ', line)
                # 清理行尾多余的空单元格
                line = re.sub(r'(\|\s*)+\|$', ' |', line)
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_completeness(self, group: TableGroup) -> float:
        """计算完整性分数"""
        score = 0.5  # 基础分
        
        if group.title:
            score += 0.2
        if group.note:
            score += 0.15
        if group.caption:
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_context_quality(self, group: TableGroup) -> float:
        """计算上下文质量"""
        scores = []
        
        if group.title:
            scores.append(group.title.quality_score)
        if group.note:
            scores.append(group.note.quality_score)
        if group.caption:
            scores.append(group.caption.quality_score)
        
        if not scores:
            return 0.5
        
        return sum(scores) / len(scores)
    
    def _calculate_title_score(self, elem: DocumentElement, table_elem: DocumentElement, distance: int) -> float:
        """计算元素作为表格标题的得分"""
        score = 0.0
        content = elem.content.strip() if elem.content else ""
        
        # 1. 长度合适性 (标题通常较短)
        if 5 <= len(content) <= 100:
            score += 0.2
        elif len(content) > 100:
            return 0  # 太长不可能是标题
        
        # 2. 包含标题关键词
        if any(kw in content for kw in self.title_keywords):
            score += 0.3
        
        # 3. 编号模式匹配 (如: 表1, 七、, (一) 等)
        number_patterns = [
            r'^表\s*[\d一二三四五六七八九十]+',
            r'^[\d一二三四五六七八九十]+[\.、\s]',
            r'^[（\(][\d一二三四五六七八九十]+[）\)]',
            r'^附表\s*[\d一二三四五六七八九十]+',
            r'^图表\s*[\d一二三四五六七八九十]+'
        ]
        
        for pattern in number_patterns:
            if re.match(pattern, content):
                score += 0.4
                break
        
        # 4. 距离惩罚 (越远分数越低)
        distance_penalty = max(0, 1 - (distance - 1) * 0.2)
        score *= distance_penalty
        
        # 5. 页面位置检查
        if elem.page_number == table_elem.page_number:
            score += 0.1
        elif elem.page_number < table_elem.page_number:
            # 标题可能在前一页
            score += 0.05
        
        return min(1.0, score)


class ChineseFinancialContextIdentifier(TableContextIdentifier):
    """中文财报表格上下文识别器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化中文财报识别器"""
        super().__init__(config)
        
        # 表头识别模式
        self.table_header_patterns = [
            r'第[一二三四]季度',  # 季度表头
            r'\d{4}年\d{1,2}月',  # 年月表头
            r'\d{4}年度?',  # 年度表头
            r'[一二三四五六七八九十\d]+[-\s]?[一二三四五六七八九十\d]+月份?',  # 月份范围
            r'（[1-4]-[36912]月份?）',  # (1-3月份)
            r'项目',  # 通用项目列
            r'科目',  # 会计科目列
        ]
        
        # 中文财报特定标题模式 - 增强模式识别
        self.financial_title_patterns = [
            r'^表\s*[\d一二三四五六七八九十]+[-：:]?\s*(.+)',  # 表1: 资产负债表
            r'^附表\s*[\d一二三四五六七八九十]+[-：:]?\s*(.+)',  # 附表1: 明细
            r'^图表\s*[\d一二三四五六七八九十]+[-：:]?\s*(.+)',  # 图表1: 趋势
            r'^[\(（][\d一二三四五六七八九十]+[\)）]\s*(.+)',  # (1) 资产明细
            r'^[\d一二三四五六七八九十]+[\.、]\s*(.+)',  # 1. 资产负债表
            r'^第?[一二三四五六七八九十]+[节章部分]\s*(.+)',  # 第一节 财务报表
            r'^[一二三四五六七八九十]+、\s*(.+)',  # 七、近三年主要会计数据
            r'^（[一二三四五六七八九十]+）\s*(.+)',  # （一）主要会计数据
        ]
        
        # 财务报表类型识别
        self.statement_keywords = {
            TableType.BALANCE_SHEET: ['资产负债表', '资产和负债', '财务状况'],
            TableType.INCOME_STATEMENT: ['利润表', '损益表', '经营成果', '收入'],
            TableType.CASH_FLOW: ['现金流量表', '现金流', '现金及现金等价物'],
            TableType.FINANCIAL_STATEMENT: ['财务报表', '合并报表', '财务数据']
        }
        
        # 会计科目识别
        self.accounting_items = BROAD_ACCOUNTING_CATEGORIES
        
        # 时间期间模式
        self.period_patterns = [
            r'(\d{4})年度',
            r'(\d{4})年(\d{1,2})月',
            r'第([一二三四])季度',
            r'(\d{4})年?第([一二三四])季度',
            r'(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})'
        ]
        
        # 货币单位模式
        self.currency_patterns = [
            r'单位[：:]\s*(.*元)',
            r'币种[：:]\s*(.+)',
            r'人民币(.*元)',
            r'\((.+元)\)',
            r'（(.+元)）'
        ]
    
    def identify_table_groups(self, elements: List[DocumentElement]) -> List[TableGroup]:
        """识别财报表格组"""
        # 调用父类方法
        table_groups = super().identify_table_groups(elements)
        
        # 财报特定处理
        for i, group in enumerate(table_groups):
            # 查找分离的表头
            self._find_separated_headers(group, elements, i)
            
            # 识别表格类型
            group.table_type = self._identify_table_type(group)
            
            # 提取会计科目
            group.accounting_items = self._extract_accounting_items(group)
            
            # 提取时间期间
            group.time_periods = self._extract_time_periods(group)
            
            # 提取货币单位
            group.monetary_unit = self._extract_monetary_unit(group)
            
            # 重新生成统一内容（包含表头）
            if hasattr(group, 'headers') and group.headers:
                group.unified_content = self._unify_group_content_with_headers(group)
                group.unified_markdown = self._generate_markdown_with_headers(group)
            
            self.logger.info(f"财报表格 {group.group_id}: "
                           f"类型={group.table_type.value}, "
                           f"科目数={len(group.accounting_items)}, "
                           f"期间数={len(group.time_periods)}, "
                           f"表头={hasattr(group, 'headers') and group.headers is not None}")
        
        return table_groups
    
    def _find_separated_headers(self, group: TableGroup, elements: List[DocumentElement], table_idx: int):
        """查找分离的表头"""
        if not group.table:
            return
        
        # 检查表格内容是否缺少明显的表头
        table_content = group.table.content
        has_header_in_table = self._check_has_header(table_content)
        
        if not has_header_in_table:
            # 向前搜索可能的表头
            for j in range(max(0, table_idx - 3), table_idx):
                elem = elements[j]
                if self._is_table_header(elem):
                    group.headers = elem.content
                    self.logger.info(f"找到分离的表头: {elem.content[:50]}...")
                    break
    
    def _check_has_header(self, content: str) -> bool:
        """检查表格内容是否已包含表头"""
        if not content:
            return False
        
        # 检查前几行是否包含典型的表头模式
        lines = content.split('\n')[:3]  # 只检查前3行
        for line in lines:
            for pattern in self.table_header_patterns:
                if re.search(pattern, line):
                    return True
        return False
    
    def _is_table_header(self, elem: DocumentElement) -> bool:
        """判断元素是否为表头"""
        if elem.element_type == ElementType.TABLE:
            return False
        
        content = elem.content.strip() if elem.content else ""
        
        # 长度限制
        if len(content) > 300 or len(content) < 5:
            return False
        
        # 末尾是句号/问号/感叹号，极大概率是句子而非表头
        if content and content[-1] in ['。', '！', '？', '!', '?']:
            return False
        
        has_separators = any(sep in content for sep in ['|', '\t', '    '])
        # 长文本且没有列分隔符，通常是叙述性文本
        if len(content) > 50 and not has_separators:
            return False
        
        # 计算表头特征得分
        score = 0
        
        # 包含多个表头关键词
        header_count = sum(1 for pattern in self.table_header_patterns 
                          if re.search(pattern, content))
        if header_count >= 2:
            score += 0.5
        elif header_count == 1:
            score += 0.3
        
        # 包含分隔符（表明是列名）
        if has_separators:  # 4个空格
            score += 0.3
        
        # 包含括号说明（如单位）
        if '（' in content or '(' in content:
            score += 0.2
        
        return score >= 0.5
    
    def _unify_group_content_with_headers(self, group: TableGroup) -> str:
        """统一表格组内容（包含独立的表头）"""
        parts = []
        
        # 1. 标题部分
        if group.title:
            title_content = group.title.content.strip()
            parts.append(f"【标题】{title_content}")
        
        # 2. 说明部分
        if group.caption:
            caption_content = group.caption.content.strip()
            parts.append(f"【说明】{caption_content}")
        
        # 3. 表头部分（如果存在独立的表头）
        if hasattr(group, 'headers') and group.headers:
            parts.append(f"【表头】{group.headers}")
        
        # 4. 表格主体内容
        if group.table:
            table_content = group.table.content
            cleaned_content = self._clean_table_content(table_content)
            parts.append(f"【表格内容】\n{cleaned_content}")
        
        # 5. 注释部分
        if group.note:
            note_content = group.note.content.strip()
            parts.append(f"【注释】{note_content}")
        
        return '\n\n'.join(parts)
    
    def _generate_markdown_with_headers(self, group: TableGroup) -> str:
        """生成包含表头的Markdown格式"""
        parts = []
        
        if group.title:
            parts.append(f"### {group.title.content}\n")
        
        if group.caption:
            parts.append(f"*{group.caption.content}*\n")
        
        if hasattr(group, 'headers') and group.headers:
            parts.append(f"**表头:** {group.headers}\n")
        
        if group.table:
            parts.append(group.table.content)
        
        if group.note:
            parts.append(f"\n> 注：{group.note.content}")
        
        return '\n'.join(parts)
    
    def _identify_table_type(self, group: TableGroup) -> TableType:
        """识别财务报表类型"""
        # 检查标题和内容
        check_content = ""
        if group.title:
            check_content += group.title.content + " "
        if group.table:
            check_content += group.table.content[:500]  # 只检查前500字符
        
        check_content = check_content.lower()
        
        # 匹配报表类型
        for table_type, keywords in self.statement_keywords.items():
            if any(kw in check_content for kw in keywords):
                return table_type
        
        # 基于内容特征判断
        if '资产' in check_content and '负债' in check_content:
            return TableType.BALANCE_SHEET
        elif '收入' in check_content and ('成本' in check_content or '费用' in check_content):
            return TableType.INCOME_STATEMENT
        elif '现金' in check_content and '流量' in check_content:
            return TableType.CASH_FLOW
        
        return TableType.OTHER
    
    def _extract_accounting_items(self, group: TableGroup) -> List[str]:
        """提取会计科目"""
        if not group.table:
            return []
        
        content = group.table.content
        candidates = set()
        
        # 基础词表命中（如 资产 / 负债 / 收入 / 利润等）
        for item in self.accounting_items:
            if item in content:
                candidates.add(item)
        # 精细词表命中（完整会计科目，用于补充发现）
        for item in STANDARD_ACCOUNTING_ITEMS:
            if item in content:
                candidates.add(item)
        
        # 常见会计科目词表（过滤碎片时优先保留）
        common_items = set(COMMON_ACCOUNTING_ITEMS) | set(STANDARD_ACCOUNTING_ITEMS)
        
        # 结构化匹配常见科目，限制长度避免截取句子
        specific_patterns = [
            r'(交易性金融资产)',
            r'(衍生金融资产)',
            r'(交易性金融负债)',
            r'(衍生金融负债)',
            r'(其他应[收付][^，。\s]{0,6})',
            r'(应收[^，。\s]{1,8})',
            r'(应付[^，。\s]{1,8})',
            r'(预[收付][^，。\s]{1,8})',
            r'([^，。\s]{2,8}资产)',
            r'([^，。\s]{2,8}负债)'
        ]
        
        for pattern in specific_patterns:
            matches = re.findall(pattern, content)
            candidates.update(matches)
        
        # 噪声关键词（出现即视为叙述性文本而非科目）
        noise_keywords = [
            '如果', '超过', '可能', '导致', '标准', '指标', '衡量',
            '缺陷', '损失', '报告', '错报', '单独', '连同'
        ]
        # 合法前缀：支持生成型科目（如 应收/应付/预收/长期等）
        allowed_prefixes = (
            '应收', '应付', '预收', '预付', '其他应收', '其他应付',
            '长期', '短期', '交易性', '衍生', '租赁', '合同', '递延',
            '固定', '无形', '在建', '资产', '负债', '收入', '成本', '利润', '现金', '货币'
        )
        punctuation_pattern = re.compile(r'[，。、；：,:%（）()\-]')
        digit_or_ascii = re.compile(r'[A-Za-z0-9]')
        
        filtered: List[str] = []
        for item in candidates:
            token = item.strip()
            if len(token) < 2 or len(token) > 15:
                continue
            if punctuation_pattern.search(token):
                continue
            if digit_or_ascii.search(token):
                continue
            if any(noise in token for noise in noise_keywords):
                continue
            if token[-1] in {'和', '或', '的', '则', '为', '及'}:
                continue
            
            # 只保留：在常见科目词表中，或有明确科目前缀
            if token in common_items or token.startswith(allowed_prefixes):
                filtered.append(token)
        
        return list(set(filtered))
    
    def _extract_time_periods(self, group: TableGroup) -> List[str]:
        """提取时间期间"""
        periods = []
        
        # 检查所有相关内容
        check_contents = []
        if group.title:
            check_contents.append(group.title.content)
        if group.caption:
            check_contents.append(group.caption.content)
        if group.table:
            check_contents.append(group.table.content[:500])
        
        for content in check_contents:
            for pattern in self.period_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        period = '-'.join(str(m) for m in match if m)
                    else:
                        period = str(match)
                    periods.append(period)
        
        # 去重并排序
        return sorted(list(set(periods)))
    
    def _extract_monetary_unit(self, group: TableGroup) -> Optional[str]:
        """提取货币单位"""
        # 优先从说明中查找
        if group.caption:
            for pattern in self.currency_patterns:
                match = re.search(pattern, group.caption.content)
                if match:
                    return match.group(1)
        
        # 从标题中查找
        if group.title:
            for pattern in self.currency_patterns:
                match = re.search(pattern, group.title.content)
                if match:
                    return match.group(1)
        
        # 从表格内容推断
        if group.table and group.table.content:
            content_sample = group.table.content[:200]
            if '万元' in content_sample:
                return '人民币万元'
            elif '千元' in content_sample:
                return '人民币千元'
            elif '百万' in content_sample:
                return '人民币百万元'
            elif '元' in content_sample:
                return '人民币元'
        
        return None
    
    def _is_table_title(self, elem: DocumentElement, table_elem: DocumentElement) -> bool:
        """判断是否为财报表格标题（重写）"""
        if elem.element_type == ElementType.TABLE:
            return False
        
        content = elem.content.strip() if elem.content else ""
        
        # 长度限制
        if len(content) > 200 or len(content) < 2:
            return False
        
        # 财报特定标题模式 - 优先级最高
        for pattern in self.financial_title_patterns:
            if re.match(pattern, content):
                return True
        
        # 检查是否包含财务报表名称
        financial_table_names = [
            '会计数据', '财务指标', '主要财务数据', '分季度财务数据',
            '营业收入', '净利润', '现金流量', '股东权益',
            '资产负债', '利润表', '现金流量表'
        ]
        
        for name in financial_table_names:
            if name in content:
                return True
        
        # 包含报表关键词
        if any(kw in content for kw in ['表', '明细', '汇总', '清单', '一览']):
            # 且不是纯数字
            if not content.replace(' ', '').isdigit():
                return True
        
        # 调用父类方法
        return super()._is_table_title(elem, table_elem)
