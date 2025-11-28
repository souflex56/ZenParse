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
        if not self.raw_data:
            return self.content
        
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
        table_cfg = self.config.get("table_extraction", {})
        self.advanced_model = table_cfg.get("advanced_model", "detectron2")
        self.detection_conf, self.detection_iou = self._resolve_detection_conf_iou()
        self.table_quality_threshold = self._resolve_table_quality_threshold()
        self.merge_iou_threshold = self._resolve_merge_iou_threshold()
        self.bbox_padding_ratio = self._resolve_bbox_padding_ratio()
        self.render_dpi = self._resolve_render_dpi()
        self.ocr_trigger_char_threshold = self._resolve_ocr_trigger_char_threshold()
        self.skip_ocr_for_digital = bool(table_cfg.get("skip_ocr_for_digital", False))
        self._yolo_device = None
        
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
        model_choice = (self.advanced_model or "detectron2").lower()
        if model_choice == "doclayout_yolo":
            try:
                from ultralytics import YOLO  # type: ignore
                self.logger.info("加载 DocLayout-YOLO 模型...")
                table_cfg = self.config.get("table_extraction", {})
                model_path = table_cfg.get("model_path")
                
                # 如果未指定路径，尝试从 HuggingFace 下载
                if not model_path:
                    try:
                        from huggingface_hub import hf_hub_download
                        from pathlib import Path
                        import os
                        import shutil
                        
                        # 默认使用 DocStructBench 模型（通用性好）
                        model_repo = table_cfg.get("model_repo", "juliozhao/DocLayout-YOLO-DocStructBench")
                        model_filename = table_cfg.get("model_filename", "doclayout_yolo_docstructbench_imgsz1024.pt")
                        
                        # 创建模型缓存目录
                        cache_dir = Path.home() / ".cache" / "zenparse" / "models"
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        local_model_path = cache_dir / model_filename
                        
                        # 如果本地不存在，则从 HuggingFace 下载
                        if not local_model_path.exists():
                            self.logger.info(f"从 HuggingFace 下载模型: {model_repo}/{model_filename}")
                            try:
                                downloaded_path = hf_hub_download(
                                    repo_id=model_repo,
                                    filename=model_filename,
                                    local_dir=str(cache_dir),
                                    local_dir_use_symlinks=False
                                )
                                # 确保文件存在
                                if downloaded_path and os.path.exists(downloaded_path):
                                    if os.path.basename(downloaded_path) != model_filename:
                                        # 如果文件名不同，复制到标准名称
                                        shutil.copy2(downloaded_path, str(local_model_path))
                                        model_path = str(local_model_path)
                                    else:
                                        model_path = downloaded_path
                                else:
                                    raise FileNotFoundError(f"下载的模型文件不存在: {downloaded_path}")
                            except Exception as e:
                                self.logger.warning(f"从 HuggingFace 下载失败: {e}")
                                # 尝试其他可能的文件名
                                alternative_filenames = [
                                    "best.pt",
                                    "yolov8_doclayout.pt",
                                    "doclayout_yolo.pt",
                                ]
                                downloaded = False
                                for alt_filename in alternative_filenames:
                                    try:
                                        alt_path = hf_hub_download(
                                            repo_id=model_repo,
                                            filename=alt_filename,
                                            local_dir=str(cache_dir),
                                            local_dir_use_symlinks=False
                                        )
                                        if alt_path and os.path.exists(alt_path):
                                            model_path = alt_path
                                            downloaded = True
                                            self.logger.info(f"成功下载模型: {alt_filename}")
                                            break
                                    except Exception:
                                        continue
                                
                                if not downloaded:
                                    raise FileNotFoundError(f"无法从 {model_repo} 下载模型文件")
                        else:
                            model_path = str(local_model_path)
                            self.logger.info(f"使用缓存的模型: {model_path}")
                    except ImportError:
                        self.logger.warning("huggingface_hub 未安装，无法自动下载模型。请安装: pip install huggingface_hub")
                        raise
                    except Exception as e:
                        self.logger.warning(f"模型下载/加载失败: {e}")
                        raise
                
                # 加载模型
                self.table_detection_model = YOLO(model_path)
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        self._yolo_device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        self._yolo_device = "mps"
                    else:
                        self._yolo_device = "cpu"
                except Exception:
                    self._yolo_device = "cpu"
                self.logger.info(f"✅ DocLayout-YOLO 加载成功，device={self._yolo_device}, model={model_path}")
            except Exception as e:
                self.logger.warning(f"DocLayout-YOLO 加载失败，回退基础策略: {e}")
                self.table_detection_model = None
        elif model_choice == "detectron2" and is_engine_available('unstructured') and UNSTRUCTURED_INFERENCE_AVAILABLE:
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
    
    def _resolve_detection_conf_iou(self) -> Tuple[float, float]:
        table_cfg = self.config.get("table_extraction", {})
        conf = table_cfg.get("detection_conf", 0.3)
        iou = table_cfg.get("detection_iou", 0.5)
        try:
            return float(conf), float(iou)
        except Exception:
            return 0.3, 0.5
    
    def _resolve_merge_iou_threshold(self) -> float:
        table_cfg = self.config.get("table_extraction", {})
        try:
            return float(table_cfg.get("merge_iou_threshold", 0.55))
        except Exception:
            return 0.55
    
    def _resolve_bbox_padding_ratio(self) -> float:
        table_cfg = self.config.get("table_extraction", {})
        try:
            return float(table_cfg.get("bbox_padding_ratio", 0.02))
        except Exception:
            return 0.02
    
    def _resolve_render_dpi(self) -> int:
        table_cfg = self.config.get("table_extraction", {})
        try:
            return int(table_cfg.get("render_dpi", 150))
        except Exception:
            return 150
    
    def _resolve_ocr_trigger_char_threshold(self) -> int:
        table_cfg = self.config.get("table_extraction", {})
        try:
            return int(table_cfg.get("ocr_trigger_char_threshold", 10))
        except Exception:
            return 10
    
    def _resolve_table_quality_threshold(self) -> float:
        """
        获取表格质量阈值，用于决定是否触发补充检测。
        优先级：
        table_extraction.quality_threshold
        → document_processing.quality_filtering.min_table_quality
        → 默认 0.65
        """
        candidate_paths = [
            ('table_extraction', 'quality_threshold'),
            ('data', 'table_extraction', 'quality_threshold'),
            ('document_processing', 'quality_filtering', 'min_table_quality'),
            ('data', 'document_processing', 'quality_filtering', 'min_table_quality'),
        ]
        for path in candidate_paths:
            value = self._get_nested_config_value(self.config, path)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                joined = ".".join(path)
                self.logger.warning(f"表格质量阈值配置无效 ({joined}={value})，尝试下一个候选值")
        return 0.65
    
    def extract_tables(self, pdf_path: str) -> List[TableElement]:
        """从PDF中提取所有表格（质量分级：先pdfplumber，高质量则直接用，低质再补模型）"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        self.logger.info(f"开始提取表格: {pdf_path}")
        quality_threshold = self._resolve_table_quality_threshold()
        
        tables: List[TableElement] = []
        plumber_tables: List[TableElement] = []
        needs_advanced = False
        
        # 首先用 pdfplumber 覆盖全页（数字版最快）
        if is_engine_available('pdfplumber'):
            self.logger.info("使用pdfplumber提取表格（首选路径）")
            plumber_tables = self._extract_with_pdfplumber(pdf_path)
            tables.extend(plumber_tables)
            
            if plumber_tables:
                low_quality = [
                    t for t in plumber_tables
                    if self._is_low_quality_table(t, quality_threshold)
                ]
                if low_quality:
                    needs_advanced = True
                    self.logger.info(
                        f"检测到 {len(low_quality)} 个低质量表格，触发补充检测（阈值={quality_threshold}）"
                    )
            else:
                # 缺表页触发补充检测
                needs_advanced = True
                self.logger.info("pdfplumber 未发现表格，尝试补充检测")
        
        # 补充：仅在需要时调用高级模型（或其他检测器）
        if needs_advanced:
            model_choice = (self.advanced_model or "").lower()
            if model_choice == "doclayout_yolo" and self.table_detection_model:
                self.logger.info("使用 DocLayout-YOLO 对低质量/缺失页面补充检测")
                advanced_tables = self._extract_with_doclayout_yolo(pdf_path)
                tables = self._merge_tables(plumber_tables, advanced_tables)
            elif self.table_detection_model:
                self.logger.info("使用高级模型对低质量/缺失页面补充检测")
                advanced_tables = self._extract_with_advanced_model(pdf_path)
                tables = self._merge_tables(plumber_tables, advanced_tables)
            elif is_engine_available('unstructured'):
                self.logger.info("表格低质量，使用unstructured补充检测")
                extra_tables = self._extract_with_unstructured(pdf_path)
                tables = self._merge_tables(plumber_tables, extra_tables)
        
        # 最后兜底：仍然没有表格时尝试 unstructured
        if not tables and is_engine_available('unstructured'):
            self.logger.info("尝试使用unstructured提取表格（兜底）")
            tables.extend(self._extract_with_unstructured(pdf_path))
        
        # 后处理：增强表格质量
        tables = self._enhance_tables(tables)
        
        # 页内排序，保持阅读顺序
        tables.sort(key=lambda t: (t.page_number, t.bbox[1] if t.bbox else 0))
        
        self.logger.info(f"成功提取 {len(tables)} 个表格")
        return tables
    
    def _extract_with_advanced_model(self, pdf_path: str) -> List[TableElement]:
        """使用高级模型进行表格检测和提取"""
        if not self.table_detection_model:
            return []
        
        # detectron2 路径（保留原有行为）
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

    def _extract_with_doclayout_yolo(self, pdf_path: str) -> List[TableElement]:
        """使用 DocLayout-YOLO 进行表格检测并裁剪抽取"""
        if not self.table_detection_model or not is_engine_available('pdfplumber'):
            return []
        
        tables: List[TableElement] = []
        try:
            import pdfplumber
        except Exception as e:
            self.logger.error(f"加载pdfplumber失败: {e}")
            return tables
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    # 若 pdfplumber 本身已有表格且质量足够，可跳过；这里仅在调用方触发时补充全页检测
                    try:
                        page_img = page.to_image(resolution=self.render_dpi).original
                    except Exception as e:
                        self.logger.debug(f"页面渲染失败，第{page_index+1}页: {e}")
                        continue
                    
                    results = self.table_detection_model.predict(
                        source=page_img,
                        conf=self.detection_conf,
                        iou=self.detection_iou,
                        device=self._yolo_device or "cpu",
                        verbose=False,
                    )
                    if not results:
                        continue
                    
                    # 获取类别映射，确认 table 类别
                    names = getattr(results[0], "names", {}) or {}
                    table_class_ids = [cid for cid, name in names.items() if str(name).lower() == "table"]
                    if not table_class_ids:
                        # DocLayout-YOLO 默认 table 类别为 5
                        table_class_ids = [5]
                    
                    img_w, img_h = page_img.size
                    for box, cls_id, conf_val in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
                        if int(cls_id) not in table_class_ids:
                            continue
                        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
                        # padding
                        pad_w = (x1 - x0) * self.bbox_padding_ratio
                        pad_h = (y1 - y0) * self.bbox_padding_ratio
                        x0 = max(0.0, x0 - pad_w)
                        y0 = max(0.0, y0 - pad_h)
                        x1 = min(img_w, x1 + pad_w)
                        y1 = min(img_h, y1 + pad_h)
                        
                        pdf_bbox = self._map_bbox_pixels_to_pdf((x0, y0, x1, y1), img_w, img_h, page.width, page.height)
                        raw_data, content = self._extract_table_by_bbox(page, pdf_bbox)
                        
                        rows = len(raw_data) if raw_data else 0
                        cols = len(raw_data[0]) if raw_data and raw_data[0] else 0
                        text_len = 0
                        if raw_data:
                            text_len = len("".join([c for row in raw_data for c in row if c]))
                        elif content:
                            text_len = len(content)
                        quality_score = self._calculate_table_quality(raw_data) if raw_data else self._calculate_content_quality(content)
                        needs_ocr = text_len < self.ocr_trigger_char_threshold
                        tables.append(
                            TableElement(
                                content=content,
                                raw_data=raw_data or [],
                                page_number=page_index + 1,
                                bbox=pdf_bbox,
                                table_index=len(tables),
                                rows=rows,
                                cols=cols,
                                has_header=self._detect_header(raw_data) if raw_data else False,
                                quality_score=quality_score,
                                extraction_method="doclayout_yolo",
                                metadata={
                                    "source_file": str(pdf_path),
                                    "model_name": "doclayout_yolo",
                                    "model_conf": float(conf_val),
                                    "padding_ratio": self.bbox_padding_ratio,
                                    "detection_conf": self.detection_conf,
                                    "detection_iou": self.detection_iou,
                                    "needs_ocr": needs_ocr,
                                },
                            )
                        )
        except Exception as e:
            self.logger.error(f"DocLayout-YOLO 提取失败: {e}")
        
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
                        
                        # 获取表格边界框
                        table_bbox = self._find_table_bbox(page, cleaned_data)
                        
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
                                'contains_financial_data': self._contains_financial_data(cleaned_data)
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
    
    def _is_low_quality_table(self, table: TableElement, threshold: float) -> bool:
        """判定表格是否低质量，需要补充检测"""
        if not table:
            return True
        if table.quality_score < threshold:
            return True
        if not table.raw_data or table.rows < 2 or table.cols < 2:
            return True
        return False
    
    def _merge_tables(
        self,
        base: List[TableElement],
        extra: List[TableElement],
    ) -> List[TableElement]:
        """合并表格列表，避免重复，保留高质量版本"""
        if not base:
            return extra or []
        if not extra:
            return base
        
        merged = list(base)
        for new_table in extra:
            if not new_table:
                continue
            is_duplicate = False
            for idx, old_table in enumerate(list(merged)):
                if old_table.page_number != new_table.page_number:
                    continue
                if old_table.bbox and new_table.bbox:
                    overlap = self._bbox_iou(old_table.bbox, new_table.bbox)
                    if overlap > self.merge_iou_threshold:
                        is_duplicate = True
                        if new_table.quality_score > old_table.quality_score:
                            merged[idx] = new_table
                        break
            if not is_duplicate:
                merged.append(new_table)
        return merged
    
    @staticmethod
    def _bbox_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
        """计算两个bbox的IoU"""
        try:
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter = inter_w * inter_h
            area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            union = area1 + area2 - inter if (area1 + area2 - inter) > 0 else 1e-6
            return inter / union
        except Exception:
            return 0.0
    
    @staticmethod
    def _map_bbox_pixels_to_pdf(bbox_px: Tuple[float, float, float, float], img_w: float, img_h: float,
                                page_w: float, page_h: float) -> Tuple[float, float, float, float]:
        """将像素坐标系的bbox映射到PDF坐标系"""
        if img_w <= 0 or img_h <= 0 or page_w <= 0 or page_h <= 0:
            return bbox_px
        scale_x = img_w / page_w
        scale_y = img_h / page_h
        x0, y0, x1, y1 = bbox_px
        return (x0 / scale_x, y0 / scale_y, x1 / scale_x, y1 / scale_y)
    
    def _extract_table_by_bbox(self, page, bbox: Tuple[float, float, float, float]) -> Tuple[List[List[str]], str]:
        """在指定bbox内用pdfplumber抽取表格内容，返回raw_data和markdown内容"""
        raw_data: List[List[str]] = []
        content = ""
        try:
            cropped = page.crop(bbox)
            tables = cropped.extract_tables()
            if tables:
                cleaned = self._clean_table_data(tables[0])
                raw_data = cleaned
                content = self._format_table_as_markdown(cleaned)
            else:
                text = cropped.extract_text() or ""
                content = text.strip()
        except Exception as e:
            self.logger.debug(f"bbox提取失败: {e}")
        return raw_data, content
    
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
    
    def _find_table_bbox(self, page, table_data) -> Optional[Tuple[float, float, float, float]]:
        """查找表格的边界框"""
        try:
            # 方法1：通过page.find_tables()获取表格位置
            page_tables = page.find_tables()
            if page_tables:
                for table in page_tables:
                    if hasattr(table, 'bbox'):
                        return table.bbox
            
            # 方法2：通过第一个单元格内容查找位置
            if table_data and table_data[0] and table_data[0][0]:
                first_cell = str(table_data[0][0]).strip()
                if first_cell:
                    words = page.extract_words()
                    for word in words:
                        if first_cell in word['text']:
                            # 基于第一个单词估算表格边界
                            x0, y0 = word['x0'], word['top']
                            # 估算表格大小
                            estimated_width = len(table_data[0]) * 100
                            estimated_height = len(table_data) * 25
                            return (x0, y0, x0 + estimated_width, y0 + estimated_height)
            
            # 方法3：使用页面默认区域
            return (50, 100, page.width - 50, page.height - 100)
            
        except Exception as e:
            self.logger.debug(f"无法确定表格边界框: {e}")
            return None
    
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
