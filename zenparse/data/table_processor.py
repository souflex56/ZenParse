"""
表格专用处理器 - 保留结构和坐标信息

专门用于处理PDF中的表格，保留完整的行列结构和位置信息。
结合pdfplumber和unstructured-inference的优势。
"""

import re
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from .models import DocumentElement, ElementType
from ..core.logger import get_logger
from ..core.enhanced_device_manager import EnhancedDeviceManager
from .engine_checker import (
    get_pdf_engines,
    is_engine_available,
    get_available_engines
)

# 条件导入（基于引擎可用性）
if is_engine_available('pdfplumber'):
    try:
        import pdfplumber
    except ImportError:
        pass

if is_engine_available('unstructured'):
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.cleaners.core import clean_extra_whitespace
    except (ImportError, AttributeError):
        pass

# 可选的高级功能
try:
    from unstructured_inference.inference.layout import DocumentLayout
    from unstructured_inference.models.base import get_model
    UNSTRUCTURED_INFERENCE_AVAILABLE = True
except (ImportError, AttributeError):
    UNSTRUCTURED_INFERENCE_AVAILABLE = False

if is_engine_available('pandas'):
    try:
        import pandas as pd
    except ImportError:
        pass


@dataclass
class TableElement:
    """表格元素数据类"""
    content: str  # 表格内容（结构化格式）
    raw_data: List[List[Any]]  # 原始表格数据
    page_number: int  # 页码
    bbox: Optional[Tuple[float, float, float, float]] = None  # 边界框 (x0, y0, x1, y1)
    table_index: int = 0  # 表格索引
    rows: int = 0  # 行数
    cols: int = 0  # 列数
    has_header: bool = False  # 是否有表头
    quality_score: float = 0.0  # 质量分数
    extraction_method: str = "unknown"  # 提取方法
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        # 如果有 raw_data，使用结构化数据生成 markdown
        if self.raw_data and len(self.raw_data) > 0:
            return self._raw_data_to_markdown()
        
        # 否则，尝试解析 content 生成 markdown
        if self.content:
            return self._parse_content_to_markdown()
        
        return ""
    
    def _raw_data_to_markdown(self) -> str:
        """将 raw_data 转换为 Markdown 格式"""
        lines = ["【表格内容】"]
        
        # 添加表头
        if self.raw_data and self.raw_data[0]:
            headers = [str(cell).strip() if cell else "" for cell in self.raw_data[0]]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
        
        # 添加数据行
        for row in self.raw_data[1:]:
            if row:
                cells = [str(cell).strip() if cell else "" for cell in row]
                lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join(lines)
    
    def _parse_content_to_markdown(self) -> str:
        """解析 content 内容并转换为 Markdown 格式"""
        if not self.content:
            return ""
        
        lines = self.content.split('\n')
        parsed_rows = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过已有的 markdown 分隔线
            if re.match(r'^[\|\s\-:]+$', line) and '---' in line:
                continue
            
            # 检测是否已经是 pipe 分隔格式
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                # 移除首尾空元素
                while cells and cells[0] == '':
                    cells = cells[1:]
                while cells and cells[-1] == '':
                    cells = cells[:-1]
                if cells:
                    parsed_rows.append(cells)
            # Tab 分隔
            elif '\t' in line:
                cells = [cell.strip() for cell in line.split('\t')]
                if cells:
                    parsed_rows.append(cells)
            # 多空格分隔
            elif re.search(r'\s{2,}', line):
                cells = [cell.strip() for cell in re.split(r'\s{2,}', line)]
                if cells:
                    parsed_rows.append(cells)
            else:
                # 单行文本，作为单列处理
                parsed_rows.append([line])
        
        if not parsed_rows:
            return self.content
        
        # 标准化列数（以最多列数为准）
        max_cols = max(len(row) for row in parsed_rows) if parsed_rows else 0
        for i, row in enumerate(parsed_rows):
            while len(row) < max_cols:
                row.append("")
        
        # 生成 markdown
        result_lines = ["【表格内容】"]
        
        if parsed_rows:
            # 第一行作为表头
            headers = parsed_rows[0]
            result_lines.append("| " + " | ".join(headers) + " |")
            result_lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
            
            # 其余行作为数据
            for row in parsed_rows[1:]:
                result_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(result_lines)
    
    def to_dataframe(self) -> Optional['pd.DataFrame']:
        """转换为pandas DataFrame"""
        if not is_engine_available('pandas') or not self.raw_data:
            return None
        
        try:
            if self.has_header and len(self.raw_data) > 1:
                df = pd.DataFrame(self.raw_data[1:], columns=self.raw_data[0])
            else:
                df = pd.DataFrame(self.raw_data)
            return df
        except Exception as e:
            print(f"无法转换为DataFrame: {e}")
            return None


class HybridTableProcessor:
    """混合表格处理器 - 使用统一工具"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化表格处理器"""
        self.config = config or {}
        
        # 直接使用已有工具，删除重复的初始化代码
        self.logger = get_logger(self.__class__.__name__)
        self.device_manager = EnhancedDeviceManager()
        
        # 使用共享的引擎检查结果（避免重复检查）
        self.available_engines = get_pdf_engines()
        
        # 简化的初始化日志
        self.logger.info("混合表格处理器初始化完成")
        
        # 财务表格关键词
        self.financial_keywords = [
            # --- 一级：基础财务 ---
            '资产','负债','权益','收入','成本','利润','现金','费用',
            '营业','净利润','净资产','股东','会计数据','财务指标',
            '主要指标','同比','环比','毛利','税前','税后','收益','支出','总额',
            '流动','非流动','应收','应付','存货','折旧','摊销','资本支出','每股收益',
            '合并','母公司','子公司','分部','附注','注释','说明','披露',

            # --- 二级：常见附注科目 ---
            '货币资金','现金等价物','短期借款','长期借款','应收账款','应付账款',
            '其他应收款','预付款项','存货','固定资产','无形资产','长期股权投资',
            '投资性房地产','金融资产','金融负债','租赁负债','递延所得税',
            '应交税费','资本公积','盈余公积','利润分配','未分配利润',
            '政府补助','资产减值','商誉','或有事项','承诺事项','股权激励'
        ]
        
        # 高价值表格模式
        self.high_value_table_patterns = [
            # --- 主报表 ---
            r'资产负债表', r'利润表', r'现金流量表', r'股东权益变动表',

            # --- 财务指标与摘要 ---
            r'主要会计数据', r'主要财务指标', r'财务摘要', r'关键财务数据',

            # --- 附注类核心表格 ---
            r'附注', r'财务报表附注', r'注释', r'会计政策', r'会计估计',

            # --- 科目类附注（细粒度财务科目） ---
            r'货币资金', r'现金及现金等价物', r'短期借款', r'长期借款',
            r'应收账款', r'应付账款', r'其他应收款', r'预付款项',
            r'存货', r'固定资产', r'无形资产', r'长期股权投资',
            r'投资性房地产', r'金融资产', r'递延所得税',
            r'资本公积', r'盈余公积', r'利润分配', r'未分配利润',

            # --- 披露类附注（高价值表格） ---
            r'政府补助', r'资产减值', r'商誉', r'或有事项',
            r'承诺事项', r'股权激励', r'关联交易', r'关联方',

            # --- 其他结构性披露 ---
            r'分部信息', r'分部报告', r'股本结构', r'每股收益',
            r'会计差错更正', r'期末余额'
        ]
        
        # 初始化高级模型（如果可用）
        self.table_detection_model = None
        self._init_advanced_models()
    
    def _init_advanced_models(self):
        """初始化高级模型 - """
        if is_engine_available('unstructured') and UNSTRUCTURED_INFERENCE_AVAILABLE:
            try:
                self.logger.info("尝试加载高级表格检测模型...")
                
                # 为 Mac M2 Max 选择最适合的模型
                # 使用 Detectron2 量化模型，专门针对表格检测优化
                model_name = "detectron2_quantized"
                
                self.logger.info(f"加载模型: {model_name}")
                self.table_detection_model = get_model(model_name)
                
                # 设置模型配置
                if hasattr(self.table_detection_model, 'confidence_threshold'):
                    # 优先读取配置中的阈值，缺失时回退到经验默认值
                    threshold = self._resolve_detection_threshold()
                    self.table_detection_model.confidence_threshold = threshold
                    self.logger.info(f"设置表格检测置信度阈值: {threshold}")
                
                self.logger.info("✅ 高级表格检测模型加载成功")
                
            except Exception as e:
                self.logger.warning(f"高级模型加载失败，使用基础模式: {e}")
                self.table_detection_model = None
        else:
            self.logger.info("使用基础表格检测策略")
            self.table_detection_model = None
    
    @staticmethod
    def _get_nested_config_value(config: Dict[str, Any], path: Tuple[str, ...]) -> Optional[Any]:
        """按层级路径安全获取配置值"""
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current
    
    def _resolve_detection_threshold(self) -> float:
        """
        解析表格检测阈值，遵循配置优先级：
        data.table_extraction.table_detection_threshold
        → data.document_processing.quality_filtering.min_table_confidence
        → 默认值 0.3
        """
        candidate_paths = [
            ('data', 'table_extraction', 'table_detection_threshold'),
            ('table_extraction', 'table_detection_threshold'),
            ('data', 'document_processing', 'table_extraction', 'table_detection_threshold'),
            ('document_processing', 'table_extraction', 'table_detection_threshold'),
            ('data', 'document_processing', 'quality_filtering', 'min_table_confidence'),
            ('document_processing', 'quality_filtering', 'min_table_confidence'),
        ]
        
        for path in candidate_paths:
            value = self._get_nested_config_value(self.config, path)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                joined_path = ".".join(path)
                self.logger.warning(f"表格检测阈值配置无效 ({joined_path}={value})，尝试下一个候选值")
        
        return 0.3
    
    def extract_tables(self, pdf_path: str) -> List[TableElement]:
        """从PDF中提取所有表格"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        self.logger.info(f"开始提取表格: {pdf_path}")
        
        # 策略1：优先使用高级模型（如果可用）
        if self.table_detection_model:
            self.logger.info("使用高级模型进行表格检测")
            tables = self._extract_with_advanced_model(pdf_path)
        else:
            tables = []
        
        # 策略2：如果高级模型不可用或效果不好，使用pdfplumber
        if not tables and is_engine_available('pdfplumber'):
            self.logger.info("使用pdfplumber提取表格")
            tables = self._extract_with_pdfplumber(pdf_path)
        
        # 策略3：如果pdfplumber效果不好，尝试unstructured
        if not tables and is_engine_available('unstructured'):
            self.logger.info("尝试使用unstructured提取表格")
            tables.extend(self._extract_with_unstructured(pdf_path))
        
        # 后处理：增强表格质量
        tables = self._enhance_tables(tables)
        
        self.logger.info(f"成功提取 {len(tables)} 个表格")
        return tables
    
    def _extract_with_advanced_model(self, pdf_path: str) -> List[TableElement]:
        """使用高级模型进行表格检测和提取"""
        if not self.table_detection_model:
            return []
        
        tables = []
        
        try:
            self.logger.info("使用 Detectron2 量化模型进行表格检测...")
            
            # 使用 DocumentLayout 进行布局分析
            from unstructured_inference.inference.layout import DocumentLayout
            
            # 加载PDF并进行布局分析
            try:
                # DocumentLayout.from_file 可能返回不同类型的结果
                layout_result = DocumentLayout.from_file(pdf_path)
                
                # 检查返回值类型
                if isinstance(layout_result, list):
                    # 如果返回的是列表，每个元素是一个页面的布局
                    self.logger.info(f"DocumentLayout返回了列表，共{len(layout_result)}页")
                    pages = layout_result
                elif hasattr(layout_result, 'pages'):
                    # 如果返回的是对象，有pages属性
                    self.logger.info(f"DocumentLayout返回了对象，有pages属性")
                    pages = layout_result.pages
                else:
                    # 其他情况
                    self.logger.warning(f"DocumentLayout返回了未知类型: {type(layout_result)}")
                    pages = []
                
            except Exception as e:
                self.logger.error(f"DocumentLayout.from_file失败: {e}")
                return []
            
            # 处理每一页
            for page_num, page_layout in enumerate(pages, 1):
                self.logger.debug(f"处理第 {page_num} 页的布局")
                
                # 记录页面元素总数
                total_elements = len(page_layout.elements) if hasattr(page_layout, 'elements') else 0
                self.logger.info(f"页面 {page_num} 共有 {total_elements} 个元素")
                
                # 记录各类型元素
                element_types = {}
                if hasattr(page_layout, 'elements'):
                    for elem in page_layout.elements:
                        elem_type = 'unknown'
                        if hasattr(elem, 'type'):
                            elem_type = str(elem.type)
                        elif hasattr(elem, 'category'):
                            elem_type = str(elem.category)
                        element_types[elem_type] = element_types.get(elem_type, 0) + 1
                
                self.logger.info(f"元素类型分布: {element_types}")
                
                # 查找表格元素
                table_elements = []
                for element in page_layout.elements:
                    # 检查元素类型（不区分大小写）
                    if hasattr(element, 'type'):
                        element_type = str(element.type).lower()
                        self.logger.debug(f"检查元素: type={element.type}, lower={element_type}")
                        if element_type == 'table':
                            table_elements.append(element)
                            confidence = getattr(element, 'prob', getattr(element, 'confidence', 'N/A'))
                            self.logger.info(f"✅ 找到表格元素: type={element.type}, confidence={confidence}")
                    # 检查元素类别
                    elif hasattr(element, 'category') and 'table' in str(element.category).lower():
                        table_elements.append(element)
                        confidence = getattr(element, 'prob', getattr(element, 'confidence', 'N/A'))
                        self.logger.debug(f"找到表格元素: category={element.category}, confidence={confidence}")
                    
                    # 额外检查：置信度较低但可能是表格的元素
                    elif hasattr(element, 'prob') and element.prob > 0.2:
                        # 检查是否有表格特征
                        element_text = str(element)
                        if '|' in element_text or '\t' in element_text or element_text.count(' ') > 10:
                            self.logger.debug(f"发现潜在表格元素（低置信度）: prob={element.prob}")
                            table_elements.append(element)
                
                self.logger.debug(f"页面 {page_num} 找到 {len(table_elements)} 个表格元素")
                
                # 处理找到的表格
                for table_idx, table_elem in enumerate(table_elements):
                    try:
                        # 提取表格内容
                        table_content = str(table_elem)
                        self.logger.info(f"表格{table_idx}内容长度: {len(table_content)}, 前100字符: {table_content[:100] if table_content else 'EMPTY'}")
                        if not table_content.strip():
                            self.logger.warning(f"表格{table_idx}内容为空，跳过")
                            continue
                        
                        # 解析表格结构
                        raw_data = self._parse_table_structure(table_content)
                        if not raw_data or len(raw_data) < 2:
                            continue
                        
                        # 获取边界框
                        bbox = None
                        if hasattr(table_elem, 'bbox'):
                            bbox = table_elem.bbox
                        elif hasattr(table_elem, 'coordinates'):
                            coords = table_elem.coordinates
                            if coords and hasattr(coords, 'points'):
                                points = coords.points
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                        
                        # 创建表格元素
                        table_element = TableElement(
                            content=self._format_table_as_markdown(raw_data),
                            raw_data=raw_data,
                            page_number=page_num,
                            bbox=bbox,
                            table_index=table_idx,
                            rows=len(raw_data),
                            cols=len(raw_data[0]) if raw_data[0] else 0,
                            has_header=self._detect_header(raw_data),
                            quality_score=self._calculate_table_quality(raw_data),
                            extraction_method='detectron2_advanced',
                            metadata={
                                'source_file': str(pdf_path),
                                'extraction_time': time.time(),
                                'model_name': 'detectron2_quantized',
                                'contains_financial_data': self._contains_financial_data(raw_data),
                                'confidence': getattr(table_elem, 'confidence', 0.8)
                            }
                        )
                        
                        tables.append(table_element)
                        self.logger.debug(f"高级模型提取表格: 第{page_num}页, 索引{table_idx}, "
                                        f"尺寸{table_element.rows}x{table_element.cols}")
                        
                    except Exception as e:
                        self.logger.error(f"处理表格元素失败: {e}")
                        import traceback
                        self.logger.error(f"错误详情: {traceback.format_exc()}")
                        continue
            
            self.logger.info(f"高级模型成功检测到 {len(tables)} 个表格")
            
        except Exception as e:
            self.logger.error(f"高级模型表格检测失败: {e}")
            # 如果高级模型失败，返回空列表，让后续策略接管
        
        return tables
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[TableElement]:
        """使用pdfplumber提取表格并保留完整信息"""
        if not is_engine_available('pdfplumber'):
            return []
        
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    self.logger.debug(f"处理第 {page_num} 页")
                    
                    # 提取页面中的所有表格
                    page_tables = page.extract_tables()
                    
                    for table_idx, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        # 清理表格数据
                        cleaned_data = self._clean_table_data(table_data)
                        if not cleaned_data:
                            continue
                        
                        # 获取表格边界框（返回 bbox 和是否为估算值）
                        table_bbox, bbox_estimated = self._find_table_bbox(page, cleaned_data)
                        
                        # 创建表格元素
                        table_elem = TableElement(
                            content=self._format_table_as_markdown(cleaned_data),
                            raw_data=cleaned_data,
                            page_number=page_num,
                            bbox=table_bbox,
                            table_index=table_idx,
                            rows=len(cleaned_data),
                            cols=len(cleaned_data[0]) if cleaned_data[0] else 0,
                            has_header=self._detect_header(cleaned_data),
                            quality_score=self._calculate_table_quality(cleaned_data),
                            extraction_method='pdfplumber',
                            metadata={
                                'source_file': str(pdf_path),
                                'extraction_time': time.time(),
                                'contains_financial_data': self._contains_financial_data(cleaned_data),
                                'bbox_estimated': bbox_estimated
                            }
                        )
                        
                        tables.append(table_elem)
                        self.logger.debug(f"提取表格: 第{page_num}页, 索引{table_idx}, "
                                        f"尺寸{table_elem.rows}x{table_elem.cols}")
        
        except Exception as e:
            self.logger.error(f"pdfplumber提取失败: {e}")
        
        return tables
    
    def _extract_with_unstructured(self, pdf_path: str) -> List[TableElement]:
        """使用unstructured提取表格（带坐标）"""
        if not is_engine_available('unstructured'):
            return []
        
        tables = []
        
        try:
            # 使用hi_res策略并启用坐标
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["chi_sim", "eng"],
                include_page_breaks=True,
                include_metadata=True
            )
            
            for elem in elements:
                # 只处理表格元素
                if hasattr(elem, 'category') and 'table' in elem.category.lower():
                    content = str(elem).strip()
                    if not content:
                        continue
                    
                    # 提取元数据和坐标
                    page_number = 1
                    bbox = None
                    
                    if hasattr(elem, 'metadata') and elem.metadata:
                        if hasattr(elem.metadata, 'page_number'):
                            page_number = elem.metadata.page_number
                        
                        # 提取坐标信息
                        if hasattr(elem.metadata, 'coordinates') and elem.metadata.coordinates:
                            coords = elem.metadata.coordinates
                            if hasattr(coords, 'points') and coords.points:
                                try:
                                    points = coords.points
                                    x_coords = [p[0] for p in points]
                                    y_coords = [p[1] for p in points]
                                    bbox = (
                                        min(x_coords), min(y_coords),
                                        max(x_coords), max(y_coords)
                                    )
                                except Exception as e:
                                    self.logger.debug(f"坐标解析失败: {e}")
                    
                    # 尝试解析表格结构
                    raw_data = self._parse_table_structure(content)
                    
                    table_elem = TableElement(
                        content=content,
                        raw_data=raw_data,
                        page_number=page_number,
                        bbox=bbox,
                        rows=len(raw_data) if raw_data else 0,
                        cols=len(raw_data[0]) if raw_data and raw_data[0] else 0,
                        has_header=self._detect_header(raw_data) if raw_data else False,
                        quality_score=self._calculate_content_quality(content),
                        extraction_method='unstructured',
                        metadata={
                            'source_file': str(pdf_path),
                            'element_type': elem.category if hasattr(elem, 'category') else 'table'
                        }
                    )
                    
                    tables.append(table_elem)
        
        except Exception as e:
            self.logger.error(f"unstructured提取失败: {e}")
        
        return tables
    
    def _clean_table_data(self, table_data: List[List]) -> List[List]:
        """清理表格数据"""
        cleaned = []
        
        for row in table_data:
            if not row or all(cell is None or str(cell).strip() == '' for cell in row):
                continue  # 跳过空行
            
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # 清理单元格内容
                    cell_str = str(cell).strip()
                    # 移除多余的空白字符
                    cell_str = re.sub(r'\s+', ' ', cell_str)
                    cleaned_row.append(cell_str)
            
            cleaned.append(cleaned_row)
        
        return cleaned
    
    def _find_table_bbox(self, page, table_data) -> Tuple[Tuple[float, float, float, float], bool]:
        """
        查找表格的边界框
        
        Returns:
            Tuple[bbox, is_estimated]: bbox 坐标和是否为估算值的标记
        """
        try:
            # 方法1：通过page.find_tables()获取精确表格位置
            page_tables = page.find_tables()
            if page_tables:
                for table in page_tables:
                    if hasattr(table, 'bbox') and table.bbox:
                        self.logger.debug(f"通过 find_tables 获取精确 bbox: {table.bbox}")
                        return (table.bbox, False)  # 精确值
            
            # 方法2：通过第一个单元格内容查找位置（部分估算）
            if table_data and table_data[0] and table_data[0][0]:
                first_cell = str(table_data[0][0]).strip()
                if first_cell and len(first_cell) > 1:  # 确保有足够内容匹配
                    try:
                        words = page.extract_words()
                        for word in words:
                            if first_cell[:min(10, len(first_cell))] in word.get('text', ''):
                                # 基于第一个单词和表格维度估算边界
                                x0, y0 = word['x0'], word['top']
                                # 根据表格行列数估算大小
                                num_cols = len(table_data[0]) if table_data[0] else 1
                                num_rows = len(table_data)
                                # 估算每列宽度（基于页面宽度）
                                page_content_width = page.width - 100  # 留出边距
                                estimated_col_width = min(100, page_content_width / num_cols)
                                estimated_row_height = 25  # 每行约25点高度
                                estimated_width = num_cols * estimated_col_width
                                estimated_height = num_rows * estimated_row_height
                                # 确保不超出页面
                                x1 = min(x0 + estimated_width, page.width - 30)
                                y1 = min(y0 + estimated_height, page.height - 30)
                                bbox = (x0, y0, x1, y1)
                                self.logger.debug(f"通过单元格内容估算 bbox: {bbox}")
                                return (bbox, True)  # 估算值
                    except Exception as e:
                        self.logger.debug(f"单元格定位失败: {e}")
            
            # 方法3：智能估算（基于表格维度和页面布局）
            num_cols = len(table_data[0]) if table_data and table_data[0] else 5
            num_rows = len(table_data) if table_data else 10
            
            # 估算表格在页面中的位置（居中偏上）
            page_width = getattr(page, 'width', 595)  # A4 默认宽度
            page_height = getattr(page, 'height', 842)  # A4 默认高度
            
            # 计算表格尺寸
            table_width = min(page_width - 80, num_cols * 80)  # 每列约80点
            table_height = min(page_height * 0.6, num_rows * 25)  # 每行约25点
            
            # 居中放置
            x0 = (page_width - table_width) / 2
            y0 = 80  # 顶部留出边距
            x1 = x0 + table_width
            y1 = y0 + table_height
            
            bbox = (x0, y0, x1, y1)
            self.logger.debug(f"使用智能估算 bbox: {bbox} (rows={num_rows}, cols={num_cols})")
            return (bbox, True)  # 估算值
            
        except Exception as e:
            self.logger.warning(f"bbox 提取失败，使用默认值: {e}")
            # 最终兜底：使用页面大部分区域
            page_width = getattr(page, 'width', 595)
            page_height = getattr(page, 'height', 842)
            default_bbox = (40, 60, page_width - 40, page_height - 60)
            return (default_bbox, True)
    
    def _format_table_as_markdown(self, table_data: List[List]) -> str:
        """将表格格式化为Markdown"""
        if not table_data:
            return ""
        
        lines = ["【表格内容】"]
        
        # 添加表头
        if table_data[0]:
            headers = table_data[0]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
        
        # 添加数据行
        for row in table_data[1:]:
            if row:
                lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _detect_header(self, table_data: List[List]) -> bool:
        """检测表格是否有表头"""
        if not table_data or len(table_data) < 2:
            return False
        
        first_row = table_data[0]
        second_row = table_data[1] if len(table_data) > 1 else []
        
        # 检查第一行是否主要是文本，第二行是否包含数字
        first_row_text_count = sum(1 for cell in first_row if cell and not re.match(r'^[\d.,%-]+$', str(cell)))
        second_row_num_count = sum(1 for cell in second_row if cell and re.match(r'^[\d.,%-]+$', str(cell)))
        
        return first_row_text_count > len(first_row) * 0.5 and second_row_num_count > len(second_row) * 0.3
    
    def _calculate_table_quality(self, table_data: List[List]) -> float:
        """计算表格质量分数"""
        if not table_data:
            return 0.0
        
        base_score = 0.7  # 表格基础分数
        
        # 维度合理性
        rows = len(table_data)
        cols = len(table_data[0]) if table_data[0] else 0
        
        if 2 <= rows <= 50 and 2 <= cols <= 15:
            base_score += 0.1
        
        # 数据完整性（非空单元格比例）
        total_cells = rows * cols if cols > 0 else 0
        non_empty_cells = sum(
            1 for row in table_data for cell in row 
            if cell and str(cell).strip()
        )
        
        if total_cells > 0:
            completeness = non_empty_cells / total_cells
            base_score += completeness * 0.1
        
        # 财务数据奖励
        financial_content = self._contains_financial_data(table_data)
        if financial_content:
            base_score += 0.15
        
        # 结构规整性（每行列数一致）
        col_counts = [len(row) for row in table_data if row]
        if col_counts and len(set(col_counts)) == 1:
            base_score += 0.05
        
        return min(1.0, base_score)
    
    def _calculate_content_quality(self, content: str) -> float:
        """计算内容质量分数"""
        if not content:
            return 0.0
        
        base_score = 0.5
        
        # 长度合理性
        if 50 < len(content) < 5000:
            base_score += 0.1
        
        # 包含数字（表格通常包含数值）
        numbers = re.findall(r'\d+\.?\d*', content)
        if len(numbers) > 5:
            base_score += 0.15
        
        # 包含财务关键词
        keyword_count = sum(1 for kw in self.financial_keywords if kw in content)
        base_score += min(0.2, keyword_count * 0.05)
        
        # 包含表格标记（如果是Markdown格式）
        if '|' in content or '【表格内容】' in content:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _contains_financial_data(self, table_data: List[List]) -> bool:
        """检查表格是否包含财务数据"""
        if not table_data:
            return False
        
        # 将表格转换为文本
        text = ' '.join(
            str(cell) for row in table_data for cell in row if cell
        )
        
        # 检查财务关键词
        for keyword in self.financial_keywords:
            if keyword in text:
                return True
        
        # 检查高价值表格模式
        for pattern in self.high_value_table_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _parse_table_structure(self, content: str) -> List[List]:
        """从文本内容解析表格结构"""
        rows = []
        
        # 尝试按行分割
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试多种分隔符
            if '|' in line:
                # Markdown格式
                cells = [cell.strip() for cell in line.split('|')]
                # 移除首尾空元素
                if cells and cells[0] == '':
                    cells = cells[1:]
                if cells and cells[-1] == '':
                    cells = cells[:-1]
                rows.append(cells)
            elif '\t' in line:
                # Tab分隔
                cells = [cell.strip() for cell in line.split('\t')]
                rows.append(cells)
            elif re.search(r'\s{2,}', line):
                # 多空格分隔
                cells = [cell.strip() for cell in re.split(r'\s{2,}', line)]
                rows.append(cells)
        
        # 过滤掉分隔线
        rows = [row for row in rows if not all(
            cell in ['', '-', '---', '----'] or cell.strip('-') == '' 
            for cell in row
        )]
        
        return rows
    
    def _enhance_tables(self, tables: List[TableElement]) -> List[TableElement]:
        """增强表格质量和元数据"""
        enhanced = []
        
        for table in tables:
            # 识别财务报表类型
            table_type = self._identify_table_type(table)
            if table_type:
                table.metadata['table_type'] = table_type
                table.quality_score = min(1.0, table.quality_score + 0.1)
            
            # 提取数值信息
            numeric_data = self._extract_numeric_data(table)
            if numeric_data:
                table.metadata['numeric_summary'] = numeric_data
            
            # 标记高价值表格
            if table.quality_score > 0.8 or table_type:
                table.metadata['high_value'] = True
            
            enhanced.append(table)
        
        return enhanced
    
    def _identify_table_type(self, table: TableElement) -> Optional[str]:
        """识别表格类型"""
        content = table.content
        
        table_types = {
            '资产负债表': ['资产', '负债', '所有者权益', '流动资产', '非流动资产'],
            '利润表': ['营业收入', '营业成本', '净利润', '营业利润', '利润总额'],
            '现金流量表': ['经营活动', '投资活动', '筹资活动', '现金流量', '现金及现金等价物'],
            '股东权益变动表': ['股本', '资本公积', '盈余公积', '未分配利润', '股东权益'],
            '主要财务指标': ['每股收益', '净资产收益率', '资产负债率', '流动比率']
        }
        
        for table_type, keywords in table_types.items():
            match_count = sum(1 for kw in keywords if kw in content)
            if match_count >= 2:  # 至少匹配2个关键词
                return table_type
        
        return None
    
    def _extract_numeric_data(self, table: TableElement) -> Dict[str, Any]:
        """提取表格中的数值数据"""
        if not table.raw_data:
            return {}
        
        numeric_values = []
        
        for row in table.raw_data:
            for cell in row:
                if cell:
                    # 提取数值（支持千分位、百分比、负数）
                    numbers = re.findall(
                        r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?',
                        str(cell)
                    )
                    for num in numbers:
                        # 清理数值
                        clean_num = num.replace(',', '').replace('%', '')
                        try:
                            value = float(clean_num)
                            numeric_values.append(value)
                        except ValueError:
                            continue
        
        if numeric_values:
            return {
                'count': len(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'has_negative': any(v < 0 for v in numeric_values),
                'has_percentage': '%' in table.content
            }
        
        return {}
    
    def convert_to_document_elements(self, tables: List[TableElement]) -> List[DocumentElement]:
        """将表格元素转换为文档元素"""
        elements = []
        
        for table in tables:
            doc_elem = DocumentElement(
                element_type=ElementType.TABLE,
                content=table.to_markdown(),
                page_number=table.page_number,
                bbox=table.bbox,
                quality_score=table.quality_score,
                extraction_confidence=0.95 if table.extraction_method == 'pdfplumber' else 0.85,
                metadata={
                    **table.metadata,
                    'table_dimensions': f"{table.rows}x{table.cols}",
                    'has_header': table.has_header,
                    'extraction_method': table.extraction_method,
                    'element_type': 'Table',
                    'source': f'{table.extraction_method}_table'
                }
            )
            
            elements.append(doc_elem)
        
        return elements


class TableQualityAnalyzer:
    """表格质量分析器"""
    
    @staticmethod
    def analyze_table_quality(table: TableElement) -> Dict[str, Any]:
        """分析表格质量"""
        analysis = {
            'structure_quality': TableQualityAnalyzer._analyze_structure(table),
            'content_quality': TableQualityAnalyzer._analyze_content(table),
            'financial_relevance': TableQualityAnalyzer._analyze_financial_relevance(table),
            'extraction_quality': TableQualityAnalyzer._analyze_extraction_quality(table),
            'overall_score': table.quality_score
        }
        
        # 计算综合建议
        if analysis['overall_score'] < 0.6:
            analysis['recommendation'] = '质量较低，建议重新提取或手动校验'
        elif analysis['overall_score'] < 0.8:
            analysis['recommendation'] = '质量中等，可能需要部分校验'
        else:
            analysis['recommendation'] = '质量良好，可直接使用'
        
        return analysis
    
    @staticmethod
    def _analyze_structure(table: TableElement) -> Dict[str, Any]:
        """分析表格结构质量"""
        return {
            'has_coordinates': table.bbox is not None,
            'dimension_reasonable': 2 <= table.rows <= 100 and 2 <= table.cols <= 20,
            'has_header': table.has_header,
            'structure_preserved': '【表格内容】' in table.content or '|' in table.content
        }
    
    @staticmethod
    def _analyze_content(table: TableElement) -> Dict[str, Any]:
        """分析表格内容质量"""
        if not table.raw_data:
            return {'data_available': False}
        
        # 计算非空单元格比例
        total_cells = table.rows * table.cols if table.cols > 0 else 0
        non_empty = sum(
            1 for row in table.raw_data for cell in row 
            if cell and str(cell).strip()
        )
        
        return {
            'data_available': True,
            'completeness': non_empty / total_cells if total_cells > 0 else 0,
            'has_numeric_data': bool(re.findall(r'\d+\.?\d*', table.content)),
            'content_length': len(table.content)
        }
    
    @staticmethod
    def _analyze_financial_relevance(table: TableElement) -> float:
        """分析财务相关性"""
        if 'table_type' in table.metadata:
            return 1.0  # 已识别的财务报表
        
        if table.metadata.get('contains_financial_data'):
            return 0.8
        
        # 计算财务关键词密度
        financial_keywords = ['资产', '负债', '收入', '利润', '现金', '股东']
        keyword_count = sum(1 for kw in financial_keywords if kw in table.content)
        
        return min(1.0, keyword_count * 0.2)
    
    @staticmethod
    def _analyze_extraction_quality(table: TableElement) -> Dict[str, Any]:
        """分析提取质量"""
        return {
            'method': table.extraction_method,
            'confidence': 0.95 if table.extraction_method == 'pdfplumber' else 0.85,
            'has_raw_data': bool(table.raw_data),
            'page_number_valid': table.page_number > 0
        }
