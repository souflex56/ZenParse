"""
PDF解析器 - 中文财报优化

两级智能解析策略，优先使用高级解析，失败时降级到稳定方案。
针对中文财报进行了特殊优化。
"""

import re
import logging
import platform
import sys
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .models import DocumentElement, ElementType
from ..core.logger import get_logger
from ..core.enhanced_device_manager import EnhancedDeviceManager
from .engine_checker import (
    get_pdf_engines, 
    is_engine_available, 
    get_available_engines,
    get_primary_engine
)

# 应用兼容性补丁（在导入 unstructured 之前）
try:
    from .pytorch_compat import auto_patch
    auto_patch()
except Exception as e:
    print(f"PyTorch 兼容性补丁加载失败: {e}")

# cv2兼容性问题已通过其他方式处理
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# 条件导入（基于引擎可用性）
if is_engine_available('unstructured'):
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.cleaners.core import clean_extra_whitespace
    except (ImportError, AttributeError):
        pass

if is_engine_available('pdfplumber'):
    try:
        import pdfplumber
    except ImportError:
        pass

if is_engine_available('pymupdf'):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        pass

# 导入表格处理器（如果可用）
try:
    from .table_processor import HybridTableProcessor, TableElement, TableQualityAnalyzer
    TABLE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    TABLE_PROCESSOR_AVAILABLE = False
    print(f"表格处理器不可用: {e}")


class ChinesePDFParser:
    """中文PDF解析器基类 - 简化版本"""
    
    _base_initialized = False
    _strategies_determined = False
    _cached_strategies = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化解析器"""
        self.config = config or {}
        
        # 直接使用已有工具 - KISS原则
        self.logger = get_logger(self.__class__.__name__)
        self.device_manager = EnhancedDeviceManager()
        
        # 使用独立模块的引擎检查（避免循环依赖）
        self.available_engines = get_pdf_engines()
        
        # 只在第一次初始化时输出基础日志
        if not ChinesePDFParser._base_initialized:
            self.logger.info(f"{self.__class__.__name__} ChinesePDFParser 初始化完成")
            available_list = [k for k, v in self.available_engines.items() if v]
            self.logger.info(f"可用引擎: {available_list}")
            
            # 记录设备信息（简化版）
            device_info = self.device_manager.get_optimal_device()
            self.logger.info(f"使用设备: {device_info}")
            
            ChinesePDFParser._base_initialized = True
        
        # 确定解析策略（使用缓存）
        if not ChinesePDFParser._strategies_determined:
            self.parsing_strategies = self._determine_strategies()
            ChinesePDFParser._cached_strategies = self.parsing_strategies
            ChinesePDFParser._strategies_determined = True
        else:
            self.parsing_strategies = ChinesePDFParser._cached_strategies
        
        # 中文特定配置
        self.chinese_punctuation = '，。！？；：""''（）【】《》、—…'
        self.financial_keywords = [
            '资产', '负债', '权益', '收入', '成本', '利润', '现金流',
            '合并', '报表', '附注', '审计', '会计', '财务', '股东'
        ]
        
        # 表格标题模式
        self.table_title_patterns = [
            r'表\s*[\d一二三四五六七八九十]+',
            r'表格\s*[\d一二三四五六七八九十]+',
            r'附表\s*[\d一二三四五六七八九十]+',
            r'Table\s*\d+',
            r'图表\s*[\d一二三四五六七八九十]+',
        ]
        
        # 财务报表识别模式
        self.financial_statement_patterns = {
            '资产负债表': ElementType.TABLE,
            '利润表': ElementType.TABLE,
            '现金流量表': ElementType.TABLE,
            '股东权益变动表': ElementType.TABLE,
            '财务报表附注': ElementType.TEXT
        }
    
    def _determine_strategies(self) -> List[str]:
        """完全按照配置确定解析策略，检查用户指定策略的依赖"""
        # 从配置读取策略列表
        # 支持两种配置路径：pdf_parsing 和 document_processing.pdf_parsing
        pdf_parsing_config = self.config.get('pdf_parsing', {})
        if not pdf_parsing_config:
            pdf_parsing_config = self.config.get('document_processing', {}).get('pdf_parsing', {})
        
        config_strategies = pdf_parsing_config.get(
            'strategies',
            ['hybrid_table', 'unstructured', 'pdfplumber'],
        )
        
        # 只验证策略名称是否合法
        valid_strategy_names = ['auto', 'hybrid_table', 'unstructured', 'pdfplumber', 'pymupdf']
        valid_strategies = []
        
        for strategy in config_strategies:
            if strategy not in valid_strategy_names:
                self.logger.warning(f"未知策略 '{strategy}'，已跳过。支持的策略: {valid_strategy_names}")
                continue
            
            # 检查用户指定策略的依赖是否满足
            if self._check_strategy_dependencies(strategy):
                valid_strategies.append(strategy)
            else:
                self.logger.warning(f"策略 '{strategy}' 的依赖不满足，已跳过")
        
        if not valid_strategies:
            raise ImportError("配置的所有策略依赖都不满足，请安装至少一个: unstructured, pdfplumber, 或 pymupdf")
        
        self.logger.info(f"PDF解析策略优先级(根据配置文件：basic_config.yaml): {valid_strategies}")
        return valid_strategies
    
    def _check_strategy_dependencies(self, strategy: str) -> bool:
        """检查用户指定策略的依赖是否满足"""
        if strategy == 'auto':
            engines = get_pdf_engines()
            has_any = any(
                engines.get(name, False)
                for name in ['pdfplumber', 'unstructured', 'pymupdf']
            )
            if not has_any:
                self.logger.warning(
                    "auto 策略需要至少一个解析引擎，但未检测到可用的 pdfplumber/unstructured/pymupdf"
                )
            return has_any
        
        if strategy == 'hybrid_table':
            # hybrid_table 依赖 unstructured 和 TABLE_PROCESSOR
            has_unstructured = is_engine_available('unstructured')
            has_processor = TABLE_PROCESSOR_AVAILABLE
            
            if not has_unstructured:
                self.logger.warning(f"hybrid_table 策略需要 unstructured 库，但未安装")
            if not has_processor:
                self.logger.warning(f"hybrid_table 策略需要表格处理器，但不可用")
            
            return has_unstructured and has_processor
        
        elif strategy in ['unstructured', 'pdfplumber', 'pymupdf']:
            # 基础策略只需检查对应库是否可用
            available = is_engine_available(strategy)
            if not available:
                self.logger.warning(f"{strategy} 库未安装或不可用")
            return available
        
        else:
            # 未知策略
            return False
    
    def _get_library_versions(self) -> Dict[str, str]:
        from .engine_checker import get_engine_info
        engine_info = get_engine_info()
        return engine_info.get('versions', {})
    
    def _calculate_quality_score(self, content: str) -> float:
        """计算内容质量分数"""
        if not content or len(content.strip()) < 10:
            return 0.0
        
        # 长度分数
        length_score = min(1.0, len(content) / 500)
        
        # 中文内容分数
        chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / len(content) if content else 0
        
        # 财务关键词分数
        keyword_count = sum(1 for kw in self.financial_keywords if kw in content)
        keyword_score = min(1.0, keyword_count / 3)
        
        # 结构完整性分数（有标点、分段等）
        has_punctuation = any(p in content for p in self.chinese_punctuation)
        structure_score = 0.5 if has_punctuation else 0.0
        
        if '\n' in content or '。' in content:
            structure_score += 0.5
        
        # 综合评分
        final_score = (
            length_score * 0.2 +
            chinese_ratio * 0.3 +
            keyword_score * 0.2 +
            structure_score * 0.3
        )
        
        return min(1.0, final_score)
    
    def _identify_element_type(self, content: str, metadata: Dict = None) -> ElementType:
        """识别元素类型"""
        content_lower = content.lower()
        
        # 检查是否为标题
        if len(content) < 100:
            # 检查表格标题模式
            for pattern in self.table_title_patterns:
                if re.search(pattern, content):
                    return ElementType.TITLE
            
            # 检查章节标题
            if re.match(r'^第[一二三四五六七八九十\d]+[章节部分]', content):
                return ElementType.TITLE
            
            # 数字编号标题
            if re.match(r'^\d+\.?\d*\.?\s*\S+', content):
                return ElementType.TITLE
        
        # 检查是否为表格（基于元数据或内容特征）
        if metadata and metadata.get('element_type') == 'Table':
            return ElementType.TABLE
        
        # 检查表格特征（多个制表符或竖线）
        if content.count('\t') > 3 or content.count('|') > 3:
            return ElementType.TABLE
        
        # 检查是否为列表
        if re.match(r'^[\(（]\d+[\)）]', content) or re.match(r'^[•·▪▫◦‣⁃]', content):
            return ElementType.LIST
        
        # 默认为文本
        return ElementType.TEXT


class SmartPDFParser(ChinesePDFParser):
    """智能PDF解析器 - 继承简化的基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化智能解析器"""
        super().__init__(config)  # 复用父类的简化初始化
        
        # 检查配置中是否包含 hybrid_table 策略，只初始化需要的处理器
        # 支持两种配置路径：pdf_parsing 和 document_processing.pdf_parsing
        pdf_parsing_config = self.config.get('pdf_parsing', {})
        if not pdf_parsing_config:
            pdf_parsing_config = self.config.get('document_processing', {}).get('pdf_parsing', {})
        
        # 使用最终确定的解析策略来决定是否初始化混合表格处理器，避免与配置默认值不一致
        strategies_for_init = self.parsing_strategies or pdf_parsing_config.get('strategies', [])
        
        if 'hybrid_table' in strategies_for_init:
            try:
                self.table_processor = HybridTableProcessor(config)
                self.table_analyzer = TableQualityAnalyzer()
                self.logger.info("混合表格处理器已启用")
            except Exception as e:
                self.logger.warning(f"混合表格处理器加载失败，hybrid_table策略将不可用: {e}")
                self.table_processor = None
                self.table_analyzer = None
        else:
            self.table_processor = None
            self.table_analyzer = None
        
        # 记录系统信息（简化版）- 只在第一次初始化时输出
        if not ChinesePDFParser._base_initialized:
            system_info = self.device_manager.get_system_info()
            self.logger.info(f"SmartPDFParser当前系统状态: 优化设备={system_info.get('optimal_device', 'unknown')}")
    
    def parse(self, pdf_path: str) -> List[DocumentElement]:
        """主解析入口 - 两级策略"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        elements = []
        parse_success = False
        
        # 尝试各种解析策略
        for strategy in self.parsing_strategies:
            try:
                # 记录解析开始时间
                start_time = time.time()
                
                self.logger.info(f"尝试使用 {strategy} 解析: {pdf_path}")
                optimal_device = self.device_manager.get_optimal_device()
                self.logger.info(f"  使用库: {strategy}, 设备: {optimal_device}")
                # 根据策略解析PDF
                if strategy == 'auto':
                    elements = self._parse_with_auto(pdf_path)
                elif strategy == 'hybrid_table':
                    if hasattr(self, 'table_processor') and self.table_processor:
                        elements = self._parse_with_hybrid_table(pdf_path)
                    else:
                        raise ImportError("混合表格处理器未初始化（请确认在初始化时已根据最终策略启用，或检查依赖是否可用）")
                elif strategy == 'unstructured':
                    elements = self._parse_with_unstructured(pdf_path)
                elif strategy == 'pdfplumber':
                    elements = self._parse_with_pdfplumber(pdf_path)
                elif strategy == 'pymupdf':
                    elements = self._parse_with_pymupdf(pdf_path)
                else:
                    raise ValueError(f"未知策略: {strategy}")
                
                if elements:
                    # 记录解析时间
                    parse_time = time.time() - start_time
                    
                    parse_success = True
                    self.logger.info(f"{strategy} 成功解析 {len(elements)} 个元素")
                    self.logger.info(f"  耗时: {parse_time:.2f}秒")
                    break
                    
            except Exception as e:
                self.logger.warning(f"{strategy} 解析失败: {str(e)}")
                # 尝试清理内存
                system_info = self.device_manager.get_system_info()
                if system_info.get('available_devices', []):
                    # GPU缓存清理（如果需要）
                    pass  # EnhancedDeviceManager目前不支持clear_gpu_cache
                    self.logger.info("  已清理GPU缓存")
                continue
        
        if not parse_success:
            self.logger.error("所有解析策略都失败了")
            raise RuntimeError("PDF解析失败：所有策略都无法处理该文件")
        
        # 后处理：识别财务报表
        elements = self._identify_financial_statements(elements)
        
        return elements
    
    def _parse_with_auto(self, pdf_path: str) -> List[DocumentElement]:
        """
        自动选择解析策略（基于“数字版 vs 扫描版”判定）：
        - digital_pdf: 首选 pdfplumber，快速稳定
        - scanned_pdf: 首选 unstructured（hi_res + OCR）
        - mixed_pdf: 默认偏向 pdfplumber，必要时再回退到 unstructured
        """
        engines = self.available_engines or get_pdf_engines()
        pdf_type, stats = self._classify_pdf_type(pdf_path)
        
        self.logger.info(
            f"auto 策略: 页面类型分析结果 = {pdf_type}, "
            f"文本页比例={stats.get('text_page_ratio', 0):.2f}, "
            f"扫描页比例={stats.get('scan_page_ratio', 0):.2f}"
        )
        
        # digital：大部分页面都有足够文本 → pdfplumber 优先
        if pdf_type == 'digital' or pdf_type == 'unknown':
            if engines.get('pdfplumber'):
                self.logger.info("auto 策略: 判定为数字版/未知类型，使用 pdfplumber 解析")
                try:
                    return self._parse_with_pdfplumber(pdf_path)
                except Exception as exc:
                    self.logger.warning(f"auto 策略: pdfplumber 解析失败，尝试其他引擎: {exc}")
            # 回退
            if engines.get('unstructured'):
                self.logger.info("auto 策略: 回退使用 unstructured 解析")
                return self._parse_with_unstructured(pdf_path)
            if engines.get('pymupdf'):
                self.logger.info("auto 策略: 回退使用 pymupdf 解析")
                return self._parse_with_pymupdf(pdf_path)
            raise RuntimeError("auto 策略: 没有可用的 PDF 解析引擎")
        
        # scanned：几乎所有页面都是大图像且无文本 → OCR 型解析优先
        if pdf_type == 'scanned':
            if engines.get('unstructured'):
                self.logger.info("auto 策略: 判定为扫描版 PDF，使用 unstructured(hi_res+OCR) 解析")
                try:
                    return self._parse_with_unstructured(pdf_path)
                except Exception as exc:
                    self.logger.warning(f"auto 策略: unstructured 解析失败，尝试其他引擎: {exc}")
            # 回退
            if engines.get('pdfplumber'):
                self.logger.info("auto 策略: 扫描版回退使用 pdfplumber（效果可能有限）")
                return self._parse_with_pdfplumber(pdf_path)
            if engines.get('pymupdf'):
                self.logger.info("auto 策略: 扫描版回退使用 pymupdf 解析")
                return self._parse_with_pymupdf(pdf_path)
            raise RuntimeError("auto 策略: 没有可用的 PDF 解析引擎")
        
        # mixed：存在部分扫描页，但整体仍有不少文本
        if pdf_type == 'mixed':
            # 默认仍然偏向 pdfplumber（性能优先），必要时由调用方再按需调整
            if engines.get('pdfplumber'):
                self.logger.info("auto 策略: 判定为混合 PDF，优先使用 pdfplumber 解析")
                try:
                    return self._parse_with_pdfplumber(pdf_path)
                except Exception as exc:
                    self.logger.warning(f"auto 策略: pdfplumber 解析失败，尝试其他引擎: {exc}")
            if engines.get('unstructured'):
                self.logger.info("auto 策略: 混合 PDF 回退使用 unstructured 解析")
                return self._parse_with_unstructured(pdf_path)
            if engines.get('pymupdf'):
                self.logger.info("auto 策略: 混合 PDF 回退使用 pymupdf 解析")
                return self._parse_with_pymupdf(pdf_path)
            raise RuntimeError("auto 策略: 没有可用的 PDF 解析引擎")

        # 理论上不会到这里，保险起见再兜一次底
        if engines.get('pdfplumber'):
            return self._parse_with_pdfplumber(pdf_path)
        if engines.get('unstructured'):
            return self._parse_with_unstructured(pdf_path)
        if engines.get('pymupdf'):
            return self._parse_with_pymupdf(pdf_path)
        raise RuntimeError("auto 策略: 没有可用的 PDF 解析引擎")

    def _classify_pdf_type(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        基于多特征的“页面类型”分析：
        - text_obj_count / text_chars
        - image_area_ratio
        输出 digital / scanned / mixed / unknown
        """
        stats: Dict[str, Any] = {
            "total_pages": 0,
            "text_page_ratio": 0.0,
            "scan_page_ratio": 0.0,
        }
        
        # 需要 pdfplumber 支持，若不可用则直接返回 unknown
        if not is_engine_available('pdfplumber'):
            self.logger.warning("auto 策略: pdfplumber 不可用，无法进行结构分析，类型标记为 unknown")
            return "unknown", stats
        
        # 阈值可通过配置微调
        pdf_parsing_config = self.config.get('pdf_parsing', {})
        if not pdf_parsing_config:
            pdf_parsing_config = self.config.get('document_processing', {}).get('pdf_parsing', {}) or {}
        
        text_threshold = int(pdf_parsing_config.get('auto_text_threshold', 80))
        sparse_text_threshold = int(pdf_parsing_config.get('auto_sparse_text_threshold', 20))
        image_ratio_threshold = float(pdf_parsing_config.get('auto_image_ratio_threshold', 0.7))
        scanned_ratio_threshold = float(pdf_parsing_config.get('auto_scanned_ratio_threshold', 0.8))
        digital_ratio_threshold = float(pdf_parsing_config.get('auto_digital_ratio_threshold', 0.8))
        
        text_pages = 0
        scan_like_pages = 0
        total_pages = 0
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return "unknown", stats
                
                for page in pdf.pages:
                    width, height = page.width or 1, page.height or 1
                    page_area = max(width * height, 1.0)
                    
                    # 1) 文本对象/字符数（近似 text_obj_count）
                    text_chars = 0
                    try:
                        if hasattr(page, "chars") and page.chars:
                            for ch in page.chars:
                                text_chars += len(ch.get("text", "") or "")
                        else:
                            text = page.extract_text() or ""
                            text_chars = len(text.strip())
                    except Exception:
                        text = page.extract_text() or ""
                        text_chars = len(text.strip())
                    
                    # 2) 图像面积占比（image_area_ratio）
                    image_area = 0.0
                    try:
                        for im in getattr(page, "images", []):
                            x0 = im.get("x0", 0.0)
                            x1 = im.get("x1", 0.0)
                            top = im.get("top", 0.0)
                            bottom = im.get("bottom", 0.0)
                            w = max(x1 - x0, 0.0)
                            h = max(bottom - top, 0.0)
                            image_area += w * h
                    except Exception:
                        pass
                    
                    image_ratio = min(image_area / page_area, 1.0)
                    
                    # 判定当前页类型
                    is_text_page = text_chars >= text_threshold
                    is_scan_like = (text_chars <= sparse_text_threshold and image_ratio >= image_ratio_threshold)
                    
                    if is_text_page:
                        text_pages += 1
                    if is_scan_like:
                        scan_like_pages += 1
        except Exception as exc:
            self.logger.warning(f"auto 策略: 结构分析失败，标记为 unknown: {exc}")
            return "unknown", stats
        
        if total_pages > 0:
            text_ratio = text_pages / total_pages
            scan_ratio = scan_like_pages / total_pages
        else:
            text_ratio = 0.0
            scan_ratio = 0.0
        
        stats["total_pages"] = total_pages
        stats["text_page_ratio"] = text_ratio
        stats["scan_page_ratio"] = scan_ratio
        
        # 判定整体类型
        if scan_ratio >= scanned_ratio_threshold and text_ratio <= (1 - scanned_ratio_threshold):
            pdf_type = "scanned"
        elif text_ratio >= digital_ratio_threshold and scan_ratio <= (1 - digital_ratio_threshold):
            pdf_type = "digital"
        elif total_pages > 0:
            pdf_type = "mixed"
        else:
            pdf_type = "unknown"
        
        self.logger.info(
            f"auto 策略: PDF 类型判定为 {pdf_type} "
            f"(text_ratio={text_ratio:.2f}, scan_ratio={scan_ratio:.2f}, pages={total_pages})"
        )
        return pdf_type, stats
    
    def _parse_with_hybrid_table(self, pdf_path: str) -> List[DocumentElement]:
        """使用混合表格检测策略解析（优先策略）"""
        if not TABLE_PROCESSOR_AVAILABLE:
            raise ImportError("表格处理器不可用")
        
        elements_list = []
        
        # Step 1: 提取所有表格并保留完整结构
        self.logger.info("Step 1: 使用混合策略提取表格...")
        tables = self.table_processor.extract_tables(pdf_path)
        
        # 转换表格为文档元素
        table_elements = self.table_processor.convert_to_document_elements(tables)
        elements_list.extend(table_elements)
        
        # 记录表格提取结果
        self.logger.info(f"  提取了 {len(table_elements)} 个表格")
        for i, table_elem in enumerate(table_elements):
            if table_elem.metadata.get('high_value'):
                self.logger.info(f"    高价值表格 {i+1}: 页{table_elem.page_number}, "
                               f"质量分数{table_elem.quality_score:.2f}")
        
        # Step 2: 提取非表格文本（跳过表格区域）
        self.logger.info("Step 2: 提取非表格文本内容...")
        text_elements = self._extract_text_excluding_tables(pdf_path, table_elements)
        elements_list.extend(text_elements)
        
        self.logger.info(f"  提取了 {len(text_elements)} 个文本元素")
        
        # Step 3: 分析整体质量
        self._analyze_extraction_quality(elements_list)
        
        # 按页码和位置排序
        elements_list.sort(key=lambda x: (
            x.page_number, 
            x.bbox[1] if x.bbox else 0
        ))
        
        return elements_list
    
    def _extract_text_excluding_tables(self, pdf_path: str, table_elements: List[DocumentElement]) -> List[DocumentElement]:
        """提取文本，排除已识别的表格区域"""
        text_elements = []
        
        # 尝试用unstructured提取文本
        if is_engine_available('unstructured'):
            try:
                # 使用auto策略以正确处理各种PDF格式
                partition_elements = partition_pdf(
                    filename=pdf_path,
                    #strategy="auto",  # auto策略可以自动检测并使用OCR
                    strategy="hi_res",  # 高分辨率策略，支持中文OCR
                    infer_table_structure=False,  # 不推断表格，因为表格已单独处理
                    languages=["chi_sim", "eng"],  # 支持中英文
                    include_page_breaks=True,
                    include_metadata=True,
                    coordinates=True  # 获取坐标
                )
                
                for elem in partition_elements:
                    # 跳过表格类型
                    if hasattr(elem, 'category') and 'table' in elem.category.lower():
                        continue
                    
                    content = str(elem).strip()
                    if not content or len(content) < 10:
                        continue
                    
                    # 获取元数据
                    metadata = self._extract_element_metadata(elem)
                    page_number = metadata.get('page_number', 1)
                    bbox = metadata.get('bbox')
                    
                    # 检查是否与表格重叠
                    if self._overlaps_with_tables(bbox, page_number, table_elements):
                        continue
                    
                    # 清理内容
                    content = clean_extra_whitespace(content)
                    
                    # 识别元素类型
                    element_type = self._identify_element_type(content, metadata)
                    
                    doc_elem = DocumentElement(
                        element_type=element_type,
                        content=content,
                        page_number=page_number,
                        bbox=bbox,
                        quality_score=self._calculate_quality_score(content),
                        extraction_confidence=0.85,
                        metadata=metadata
                    )
                    
                    text_elements.append(doc_elem)
                    
            except Exception as e:
                self.logger.warning(f"unstructured文本提取失败: {e}")
        
        # 如果unstructured失败或质量太低，使用pdfplumber作为补充
        # 检查是否需要补充提取（例如某些页面没有文本）
        pages_with_text = set(elem.page_number for elem in text_elements)
        
        if is_engine_available('pdfplumber'):
            self.logger.info(f"使用pdfplumber补充提取文本（已有页面: {pages_with_text}）")
            try:
                import pdfplumber
                
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        # 如果这一页已经有高质量文本，跳过
                        page_text_quality = [elem.quality_score for elem in text_elements 
                                            if elem.page_number == page_num]
                        if page_text_quality and max(page_text_quality) > 0.5:
                            continue
                        # 提取页面文本
                        page_text = page.extract_text()
                        
                        if not page_text or not page_text.strip():
                            continue
                        
                        # 获取表格区域以排除
                        table_bboxes = []
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                # 获取表格的边界框
                                table_settings = page.find_tables()[0] if page.find_tables() else None
                                if table_settings and hasattr(table_settings, 'bbox'):
                                    table_bboxes.append(table_settings.bbox)
                        
                        # 提取非表格文本
                        # 如果有表格，尝试排除表格区域
                        if table_bboxes:
                            # 使用crop来排除表格区域
                            non_table_text = ""
                            page_height = page.height
                            page_width = page.width
                            
                            # 简单策略：提取表格外的文本
                            for bbox in table_bboxes:
                                # 提取表格上方的文本
                                if bbox[1] > 50:  # 如果表格不在页面顶部
                                    top_region = page.within_bbox((0, 0, page_width, bbox[1]))
                                    if top_region:
                                        top_text = top_region.extract_text()
                                        if top_text and top_text.strip():
                                            non_table_text += top_text + "\n"
                                
                                # 提取表格下方的文本
                                if bbox[3] < page_height - 50:  # 如果表格不在页面底部
                                    bottom_region = page.within_bbox((0, bbox[3], page_width, page_height))
                                    if bottom_region:
                                        bottom_text = bottom_region.extract_text()
                                        if bottom_text and bottom_text.strip():
                                            non_table_text += bottom_text + "\n"
                            
                            # 如果没有提取到表格外的文本，使用全部文本
                            if not non_table_text.strip():
                                non_table_text = page_text
                        else:
                            # 没有表格，使用全部文本
                            non_table_text = page_text
                        
                        # 将文本分段
                        paragraphs = [p.strip() for p in non_table_text.split('\n\n') if p.strip()]
                        if not paragraphs:
                            paragraphs = [p.strip() for p in non_table_text.split('\n') if p.strip()]
                        
                        for para_text in paragraphs:
                            if len(para_text) < 10:  # 跳过太短的文本
                                continue
                            
                            doc_elem = DocumentElement(
                                element_type=ElementType.TEXT,
                                content=para_text,
                                page_number=page_num,
                                bbox=None,  # pdfplumber文本没有精确边界框
                                quality_score=self._calculate_quality_score(para_text),
                                extraction_confidence=0.75,
                                metadata={'source': 'pdfplumber_text'}
                            )
                            text_elements.append(doc_elem)
                        
                        if paragraphs:
                            self.logger.info(f"  补充了第{page_num}页的{len(paragraphs)}个段落")
                            
            except Exception as e:
                self.logger.error(f"pdfplumber文本提取失败: {e}")
        
        return text_elements
    
    def _extract_element_metadata(self, elem) -> Dict[str, Any]:
        """提取unstructured元素的元数据和坐标"""
        metadata = {}
        
        # 基本元数据
        if hasattr(elem, 'category'):
            metadata['element_type'] = elem.category
        
        # 获取页码
        if hasattr(elem, 'metadata') and elem.metadata:
            if hasattr(elem.metadata, 'page_number'):
                metadata['page_number'] = elem.metadata.page_number
            
            # 获取坐标信息
            if hasattr(elem.metadata, 'coordinates') and elem.metadata.coordinates:
                coords = elem.metadata.coordinates
                if hasattr(coords, 'points') and coords.points:
                    try:
                        points = coords.points
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        metadata['bbox'] = (
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        )
                    except Exception as e:
                        self.logger.debug(f"坐标解析失败: {e}")
        
        # 默认值
        if 'page_number' not in metadata:
            metadata['page_number'] = 1
        
        metadata['source'] = 'unstructured_text'
        return metadata
    
    def _overlaps_with_tables(self, bbox: Optional[Tuple], page_number: int, 
                             table_elements: List[DocumentElement]) -> bool:
        """检查文本是否与表格重叠"""
        if not bbox:
            return False
        
        x1, y1, x2, y2 = bbox
        
        for table_elem in table_elements:
            if table_elem.page_number != page_number or not table_elem.bbox:
                continue
            
            tx1, ty1, tx2, ty2 = table_elem.bbox
            
            # 矩形重叠检测
            if not (x2 < tx1 or x1 > tx2 or y2 < ty1 or y1 > ty2):
                return True
        
        return False
    
    def _analyze_extraction_quality(self, elements: List[DocumentElement]):
        """分析提取质量"""
        if not elements:
            return
        
        # 统计各类元素
        table_count = sum(1 for e in elements if e.element_type == ElementType.TABLE)
        text_count = sum(1 for e in elements if e.element_type == ElementType.TEXT)
        
        # 统计有坐标的元素
        with_coords = sum(1 for e in elements if e.bbox is not None)
        
        # 计算平均质量分数
        avg_quality = sum(e.quality_score for e in elements) / len(elements)
        
        # 统计高质量元素
        high_quality = sum(1 for e in elements if e.quality_score > 0.8)
        
        self.logger.info(f"提取质量分析:")
        self.logger.info(f"  总元素数: {len(elements)}")
        self.logger.info(f"  表格: {table_count}, 文本: {text_count}")
        self.logger.info(f"  包含坐标: {with_coords}/{len(elements)} ({with_coords/len(elements)*100:.1f}%)")
        self.logger.info(f"  平均质量: {avg_quality:.3f}")
        self.logger.info(f"  高质量元素: {high_quality} ({high_quality/len(elements)*100:.1f}%)")
    
    def _parse_with_unstructured(self, pdf_path: str) -> List[DocumentElement]:
        """使用Unstructured解析（高级功能）"""
        if not is_engine_available('unstructured'):
            raise ImportError("unstructured库不可用")
        
        elements_list = []
        
        try:
            # 在解析前再次应用补丁
            from .pytorch_compat import patch_pytorch_pytree
            patch_pytorch_pytree()
        except:
            pass
        
        # 使用高分辨率策略，支持中文OCR
        partition_elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # 高分辨率策略
            infer_table_structure=True,  # 推断表格结构
            languages=["chi_sim", "eng"],  # 支持简体中文和英文
            include_page_breaks=True
        )
        
        for idx, elem in enumerate(partition_elements):
            # 提取基本信息
            content = str(elem).strip()
            if not content:
                continue
            
            # 获取元数据
            metadata = {}
            if hasattr(elem, 'metadata'):
                if hasattr(elem.metadata, 'page_number'):
                    page_number = elem.metadata.page_number
                else:
                    page_number = 1
                
                if hasattr(elem.metadata, 'coordinates'):
                    coords = elem.metadata.coordinates
                    if coords:
                        bbox = (coords.points[0][0], coords.points[0][1],
                               coords.points[2][0], coords.points[2][1])
                    else:
                        bbox = None
                else:
                    bbox = None
                
                metadata['element_type'] = elem.category
            else:
                page_number = 1
                bbox = None
            
            # 清理内容
            content = clean_extra_whitespace(content)
            
            # 识别元素类型
            element_type = self._identify_element_type(content, metadata)
            
            # 创建文档元素
            doc_elem = DocumentElement(
                element_type=element_type,
                content=content,
                page_number=page_number,
                bbox=bbox,
                quality_score=self._calculate_quality_score(content),
                extraction_confidence=0.9,  # Unstructured通常质量较高
                metadata=metadata
            )
            
            elements_list.append(doc_elem)
        
        return elements_list
    
    def _parse_with_pdfplumber(self, pdf_path: str) -> List[DocumentElement]:
        """使用PDFPlumber解析（稳定备选）"""
        if not is_engine_available('pdfplumber'):
            raise ImportError("pdfplumber库不可用")
        
        elements_list = []
        
        with pdfplumber.open(pdf_path) as pdf:
            self.logger.info(f"PDF共有 {len(pdf.pages)} 页")
            
            for page_num, page in enumerate(pdf.pages, 1):
                self.logger.debug(f"处理第 {page_num} 页")
                page_elements_count = 0
                
                # 首先提取所有文本（包括表格区域的文本）
                full_text = page.extract_text()
                if full_text and full_text.strip():
                    # 移除过多的空白字符
                    full_text = re.sub(r'\n\s*\n\s*\n', '\n\n', full_text)
                    
                    # 分段处理文本
                    paragraphs = self._split_into_paragraphs(full_text)
                    
                    for para_idx, para in enumerate(paragraphs):
                        if len(para.strip()) < 10:
                            continue
                        
                        element_type = self._identify_element_type(para)
                        
                        doc_elem = DocumentElement(
                            element_type=element_type,
                            content=para.strip(),
                            page_number=page_num,
                            quality_score=self._calculate_quality_score(para),
                            extraction_confidence=0.7,
                            metadata={
                                'source': 'pdfplumber_text',
                                'paragraph_index': para_idx
                            }
                        )
                        elements_list.append(doc_elem)
                        page_elements_count += 1
                
                # 然后专门提取结构化表格
                tables = page.extract_tables()
                for table_idx, table_data in enumerate(tables):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # 转换表格为文本
                    content = self._table_to_text(table_data)
                    
                    if content and len(content.strip()) > 50:  # 确保表格有实质内容
                        doc_elem = DocumentElement(
                            element_type=ElementType.TABLE,
                            content=content,
                            page_number=page_num,
                            quality_score=self._calculate_quality_score(content),
                            extraction_confidence=0.8,  # 表格提取置信度更高
                            metadata={
                                'source': 'pdfplumber_table', 
                                'table_index': table_idx,
                                'table_dimensions': f"{len(table_data)}x{len(table_data[0]) if table_data[0] else 0}"
                            }
                        )
                        elements_list.append(doc_elem)
                        page_elements_count += 1
                
                self.logger.debug(f"第 {page_num} 页提取了 {page_elements_count} 个元素")
        
        self.logger.info(f"pdfplumber 总共提取了 {len(elements_list)} 个元素")
        return elements_list
    
    def _parse_with_pymupdf(self, pdf_path: str) -> List[DocumentElement]:
        """使用PyMuPDF解析"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF库不可用")
        
        elements_list = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 提取文本块
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # block格式: (x0, y0, x1, y1, "text", block_no, block_type)
                if len(block) < 5:
                    continue
                
                content = block[4].strip()
                if not content or len(content) < 10:
                    continue
                
                bbox = (block[0], block[1], block[2], block[3])
                
                element_type = self._identify_element_type(content)
                
                doc_elem = DocumentElement(
                    element_type=element_type,
                    content=content,
                    page_number=page_num + 1,
                    bbox=bbox,
                    quality_score=self._calculate_quality_score(content),
                    extraction_confidence=0.6,
                    metadata={'source': 'pymupdf', 'block_no': block[5] if len(block) > 5 else None}
                )
                elements_list.append(doc_elem)
        
        doc.close()
        return elements_list
    
    def _table_to_text(self, table_data: List[List]) -> str:
        """将表格数据转换为文本"""
        if not table_data:
            return ""
        
        lines = []
        
        # 处理表头
        if table_data[0]:
            headers = [str(cell) if cell else "" for cell in table_data[0]]
            lines.append(" | ".join(headers))
            lines.append("-" * 50)  # 分隔线
        
        # 处理数据行
        for row in table_data[1:]:
            if row:
                cells = [str(cell) if cell else "" for cell in row]
                lines.append(" | ".join(cells))
        
        return "\n".join(lines)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        # 基于空行分段
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 如果没有空行，基于中文句号分段
        if len(paragraphs) == 1 and len(text) > 500:
            sentences = re.split(r'。\s*', text)
            # 重组成段落（每3-5句一段）
            paragraphs = []
            current_para = []
            
            for sent in sentences:
                if sent.strip():
                    current_para.append(sent + '。')
                    if len(current_para) >= 3:
                        paragraphs.append(''.join(current_para))
                        current_para = []
            
            if current_para:
                paragraphs.append(''.join(current_para))
        
        return paragraphs
    
    def _identify_financial_statements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """识别财务报表"""
        for elem in elements:
            content_preview = elem.content[:200] if elem.content else ""
            
            # 检查是否包含财务报表关键词
            for statement_name in self.financial_statement_patterns:
                if statement_name in content_preview:
                    elem.metadata['financial_statement'] = statement_name
                    elem.report_section = statement_name
                    
                    # 如果是已知的财务报表，更新类型
                    if '表' in statement_name and elem.element_type != ElementType.TABLE:
                        elem.element_type = ElementType.TABLE
                    
                    break
        
        return elements


class ChineseFinancialPDFParser(SmartPDFParser):
    """财报PDF解析器 - 继承智能解析器，保留关键业务配置"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化财报解析器"""
        super().__init__(config)  # 复用智能解析器
        
        # 保留重要的财报配置 - 不要删除业务逻辑
        self.report_sections = [
            # 前言及摘要类
            '释义',
            '重要提示', 
            '公司简介',
            '会计数据和财务指标摘要',
            
            # 管理层讨论与分析类
            '公司业务概要',
            '经营情况讨论与分析',
            '董事会报告',
            '重要事项',
            
            # 公司治理及股权信息类
            '股份变动及股东情况',
            '普通股股份变动及股东情况',
            '优先股相关情况',
            '董事、监事、高级管理人员和员工情况',
            '公司治理',
            '内部控制',
            '债券相关情况',
            
            # 财务和审计结论
            '财务报告',
            '审计报告',
            '备查文件目录'
        ]
        
        # 会计科目模式（保留关键业务逻辑）
        self.accounting_item_patterns = [
            r'[一二三四五六七八九十\d]+、\s*\S+',  # 一、流动资产
            r'[(（][一二三四五六七八九十\d]+[)）]\s*\S+',  # (一)货币资金
            r'\d+\.\s*\S+',  # 1. 货币资金
            r'[(（]\d+[)）]\s*\S+'  # (1)现金
        ]
        
        # 会计科目关键词库 - 采用"全称+简称+衍生词"策略
        self.accounting_keywords = [
            # 资产负债表 - 流动资产
            '货币资金', '现金', '银行存款', '库存现金', '现金等价物',
            '应收票据', '应收账款', '应收账款融资', '应收款项融资',
            '预付款项', '预付账款', '其他应收款', '长期应收款',
            '存货', '合同资产', '交易性金融资产', '其他流动资产',
            
            # 资产负债表 - 非流动资产
            '长期股权投资', '投资性房地产', '固定资产', '在建工程', '工程物资',
            '无形资产', '开发支出', '长期待摊费用', '商誉',
            '使用权资产', '递延所得税资产', '持有待售资产',
            
            # 资产负债表 - 流动负债
            '短期借款', '应付票据', '应付账款', '预收款项', '预收账款',
            '合同负债', '应付职工薪酬', '应交税费', '应付利息', '应付股利',
            '其他应付款', '吸收存款', '拆入资金', '一年内到期的非流动负债', '租赁负债',
            
            # 资产负债表 - 非流动负债
            '长期借款', '应付债券', '长期应付款', '递延收益',
            '递延所得税负债', '预计负债',
            
            # 资产负债表 - 所有者权益
            '实收资本', '股本', '资本公积', '盈余公积', '未分配利润',
            '库存股', '少数股东权益', '其他权益工具',
            
            # 利润表 - 收入
            '营业收入', '营业总收入', '主营业务收入', '其他业务收入',
            
            # 利润表 - 成本费用
            '营业成本', '主营业务成本', '销售费用', '管理费用', '财务费用', '研发费用',
            
            # 利润表 - 损益
            '营业外收入', '营业外支出', '资产减值损失', '信用减值损失',
            '公允价值变动损益', '投资收益',
            
            # 利润表 - 利润（含全称、简称、衍生词）
            '利润总额', '净利润', '归属于母公司所有者的净利润', '归属于母公司股东的净利润',
            '归属于上市公司股东净利润', '归母净利润', '归母净利', '少数股东损益',
            
            # 利润表 - 每股收益
            '基本每股收益', '稀释每股收益', '每股收益',
            
            # 利润表 - 税费
            '所得税费用', '所得税',
            
            # 利润表 - 其他
            '其他综合收益', '综合收益总额',
            
            # 现金流量表
            '经营活动产生的现金流量', '经营活动现金流',
            '投资活动产生的现金流量', '投资活动现金流',
            '筹资活动产生的现金流量', '筹资活动现金流',
            '汇率变动对现金及现金等价物的影响', '现金及现金等价物净增加额', '现金及现金等价物'
        ]
        
        # 非会计科目关键词库
        self.non_accounting_keywords = [
            # 财报章节名称
            '释义', '重要提示', '公司简介', '公司基本情况', '董事会报告', '重要事项',
            '审计报告', '内部控制', '公司治理', '备查文件目录', '股份变动', '股东情况',
            '经营情况讨论与分析', '公司业务概要', '优先股相关情况', '债券相关情况', '普通股股份变动',
            
            # 描述性概念
            '公允价值', '折旧', '摊销', '减值准备', '预期信用损失', '母公司', '合并报表',
            '关联交易', '利润分配', '股本', '现金流', '财务指标'
        ]
        
        # 财报特定质量阈值
        self.chinese_config = {
            'languages': ['chi_sim', 'eng'],
            'chunking_strategy': 'by_title',
            'quality_threshold': 0.7,  # 财报要求更高质量
            'preserve_table_structure': True,
            'enable_accounting_recognition': True
        }
        
        self.logger.info("财报专用配置已应用")
    
    def parse(self, pdf_path: str) -> List[DocumentElement]:
        """解析中文财报PDF"""
        # 调用父类解析
        elements = super().parse(pdf_path)
        
        # 财报特定后处理
        # 提取报告元数据
        elements = self._extract_report_metadata(elements)
        # 识别会计科目
        elements = self._identify_accounting_items(elements)
        # 提取财务期间
        elements = self._extract_fiscal_periods(elements)
        
        return elements
    
    def _extract_report_metadata(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """提取报告元数据"""
        for elem in elements:
            # 提取公司名称 - 扩展支持更多公司类型，支持括号内容和特殊格式
            company_pattern = r'([\u4e00-\u9fff]+(?:股份|集团|控股|科技|技术|金融|银行|证券|保险|地产|能源|医药|电子|通信|汽车|钢铁|化工|实业|投资|建设|贸易|物流|文化|教育|农业|航空|铁路|发展|企业)(?:（[^）]*）)?(?:有限|有限责任)?(?:股份)?(?:公司|企业))'
            company_match = re.search(company_pattern, elem.content)
            if company_match:
                elem.company_name = company_match.group(1)
            
            # 提取财务期间
            year_pattern = r'(\d{4})\s*年'
            quarter_pattern = r'第([一二三四])\s*季度|(\d)\s*季度'
            
            year_match = re.search(year_pattern, elem.content)
            if year_match:
                elem.fiscal_period = year_match.group(1) + "年"
            
            quarter_match = re.search(quarter_pattern, elem.content)
            if quarter_match:
                quarter = quarter_match.group(1) or quarter_match.group(2)
                if elem.fiscal_period:
                    elem.fiscal_period += f"第{quarter}季度"
            
            # 识别报告章节
            for section in self.report_sections:
                if section in elem.content[:100]:
                    elem.report_section = section
                    break
        
        return elements
    
    def _identify_accounting_items(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """识别会计科目 - 集成智能过滤"""
        for elem in elements:
            # 检查是否包含会计科目模式
            for pattern in self.accounting_item_patterns:
                matches = re.findall(pattern, elem.content)
                if matches:
                    # 使用智能过滤筛选真正的会计科目
                    filtered_matches = self._filter_real_accounting_items(matches, elem)
                    
                    # 只有过滤后仍有结果时，才标记为包含会计科目
                    if filtered_matches:
                        elem.metadata['accounting_items'] = filtered_matches
                        elem.metadata['contains_accounting_items'] = True
                        self.logger.debug(f"识别到会计科目: {filtered_matches}, 章节: {elem.report_section or '未知'}")
                    break
        
        return elements
    
    def _filter_real_accounting_items(self, matches: List[str], elem: DocumentElement) -> List[str]:
        """过滤真正的会计科目 - 基于关键词和章节上下文"""
        if not matches:
            return []
        
        filtered_matches = []
        original_count = len(matches)
        
        # 获取元素所在章节
        report_section = elem.report_section
        
        # 根据章节调整过滤严格程度
        is_financial_section = report_section in ['财务报告', '审计报告']
        is_non_financial_section = report_section in ['公司简介', '公司基本情况', '董事会报告', '重要事项', '释义', '重要提示']
        
        for match in matches:
            should_keep = False
            
            # 1. 检查是否包含会计科目关键词
            for keyword in self.accounting_keywords:
                if keyword in match:
                    should_keep = True
                    break
            
            # 2. 检查是否包含非会计科目关键词
            for non_keyword in self.non_accounting_keywords:
                if non_keyword in match:
                    should_keep = False
                    break
            
            # 3. 根据章节上下文调整过滤策略
            if should_keep:
                # 如果在财务报告章节，优先保留
                if is_financial_section:
                    should_keep = True
                # 如果在非财务章节，提高过滤标准
                elif is_non_financial_section:
                    # 必须明确包含会计科目关键词才保留
                    has_accounting_keyword = any(keyword in match for keyword in self.accounting_keywords)
                    should_keep = has_accounting_keyword
                # 其他章节，保持原有判断
                else:
                    should_keep = True
            
            if should_keep:
                filtered_matches.append(match)
        
        # 记录过滤日志
        filtered_count = len(filtered_matches)
        if original_count != filtered_count:
            self.logger.debug(f"会计科目过滤: {original_count} -> {filtered_count}, "
                            f"章节: {report_section or '未知'}, "
                            f"保留: {filtered_matches}, "
                            f"过滤: {[m for m in matches if m not in filtered_matches]}")
        
        return filtered_matches
    
    def _extract_fiscal_periods(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """提取财务期间信息"""
        period_patterns = [
            r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日',  # 2024年3月31日
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})',  # 2024.3.31
            r'(\d{4})/(\d{1,2})/(\d{1,2})',   # 2024/3/31
            r'(\d{4})-(\d{1,2})-(\d{1,2})'    # 2024-3-31
        ]
        
        for elem in elements:
            for pattern in period_patterns:
                matches = re.findall(pattern, elem.content)
                if matches:
                    elem.metadata['fiscal_dates'] = matches
                    break
        
        return elements
