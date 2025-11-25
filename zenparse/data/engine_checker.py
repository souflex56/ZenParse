"""
PDF引擎检查工具 - 独立模块避免循环依赖

提供PDF解析引擎的可用性检查，使用缓存避免重复检查。
"""
from typing import Dict, List, Optional
import importlib
from ..core.logger import get_logger

# 模块导出声明
__all__ = [
    'get_pdf_engines',
    'is_engine_available', 
    'get_available_engines',
    'get_primary_engine',
    'reset_engine_cache'
]

# 全局缓存，避免重复检查
_ENGINE_CACHE: Optional[Dict[str, bool]] = None
_logger = get_logger(__name__)


def _lazy_module_exists(module_path: str) -> bool:
    """使用 importlib 进行延迟检测，避免重依赖在启动时加载"""
    try:
        return importlib.util.find_spec(module_path) is not None
    except Exception as exc:  # pragma: no cover - 调试辅助
        _logger.debug(f"检查模块 {module_path} 失败: {exc}")
        return False


def check_pdf_engines() -> Dict[str, bool]:
    """检查PDF解析引擎可用性"""
    engines = {}
    
    # 检查unstructured
    engines['unstructured'] = _lazy_module_exists("unstructured.partition.pdf")
    _logger.debug("✅ unstructured 可用" if engines['unstructured'] else "❌ unstructured 不可用")
    
    # 检查pdfplumber
    engines['pdfplumber'] = _lazy_module_exists("pdfplumber")
    _logger.debug("✅ pdfplumber 可用" if engines['pdfplumber'] else "❌ pdfplumber 不可用")
    
    # 检查pymupdf
    engines['pymupdf'] = _lazy_module_exists("fitz")
    _logger.debug("✅ pymupdf 可用" if engines['pymupdf'] else "❌ pymupdf 不可用")
    
    # 检查pandas（用于表格处理）
    engines['pandas'] = _lazy_module_exists("pandas")
    _logger.debug("✅ pandas 可用" if engines['pandas'] else "❌ pandas 不可用")

    # 轻量检测重依赖，避免启动时加载
    engines['torch'] = _lazy_module_exists("torch")
    engines['detectron2'] = _lazy_module_exists("detectron2")
    _logger.debug("✅ torch 可用" if engines['torch'] else "❌ torch 不可用")
    _logger.debug("✅ detectron2 可用" if engines['detectron2'] else "❌ detectron2 不可用")
    
    # 记录可用引擎
    available = [k for k, v in engines.items() if v]
    _logger.info(f"PDF处理引擎检查完成，可用引擎: {available}")
    
    if not any([engines.get('unstructured'), engines.get('pdfplumber'), engines.get('pymupdf')]):
        _logger.warning("⚠️ 没有可用的PDF解析引擎，请安装至少一个: unstructured, pdfplumber, 或 pymupdf")
    
    return engines


def get_pdf_engines() -> Dict[str, bool]:
    """
    获取PDF引擎可用性（带缓存）
    
    Returns:
        Dict[str, bool]: 引擎名称到可用性的映射副本
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is None:
        _ENGINE_CACHE = check_pdf_engines()
    return _ENGINE_CACHE.copy()  # 返回副本，避免被外部修改


def is_engine_available(engine_name: str, strict: bool = False) -> bool:
    """
    检查特定引擎是否可用

    Args:
        engine_name: 引擎名称 ('unstructured', 'pdfplumber', 'pymupdf', 'pandas', 'torch', 'detectron2')
        strict: 是否执行真实 import（默认仅检查模块存在，避免重依赖在启动时加载）

    Returns:
        bool: 引擎是否可用
    """
    module_map = {
        "unstructured": "unstructured.partition.pdf",
        "pdfplumber": "pdfplumber",
        "pymupdf": "fitz",
        "pandas": "pandas",
        "torch": "torch",
        "detectron2": "detectron2",
    }
    module_path = module_map.get(engine_name)
    if not module_path:
        return False

    available = _lazy_module_exists(module_path)
    if not available or not strict:
        return available

    try:
        importlib.import_module(module_path)
        return True
    except Exception as exc:  # pragma: no cover - 调试辅助
        _logger.debug(f"严格检查 {module_path} 失败: {exc}")
        return False


def get_available_engines() -> List[str]:
    """
    获取所有可用引擎列表
    
    Returns:
        List[str]: 可用引擎名称列表
    """
    engines = get_pdf_engines()
    return [k for k, v in engines.items() if v]


def get_primary_engine() -> str:
    """
    获取主要引擎（按优先级）
    
    Returns:
        str: 最优先的可用引擎名称
        
    Raises:
        ImportError: 如果没有可用的PDF解析引擎
    """
    # 引擎优先级顺序
    priority = ['unstructured', 'pdfplumber', 'pymupdf']
    engines = get_pdf_engines()
    
    for engine in priority:
        if engines.get(engine, False):
            _logger.info(f"使用主要PDF引擎: {engine}")
            return engine
    
    raise ImportError(
        "没有可用的PDF解析引擎。请安装至少一个: "
        "pip install unstructured[pdf] 或 pip install pdfplumber 或 pip install pymupdf"
    )


def reset_engine_cache():
    """
    重置引擎缓存（用于调试或重新安装后）
    
    在安装新的PDF处理库后调用此函数以重新检查引擎可用性。
    """
    global _ENGINE_CACHE
    _ENGINE_CACHE = None
    _logger.info("引擎缓存已重置，下次调用时将重新检查")


def get_engine_info() -> Dict[str, any]:
    """
    获取详细的引擎信息（用于调试）
    
    Returns:
        Dict: 包含引擎可用性和版本信息
    """
    info = {
        'engines': get_pdf_engines(),
        'primary': None,
        'versions': {}
    }
    
    try:
        info['primary'] = get_primary_engine()
    except ImportError:
        pass
    
    # 获取版本信息
    if is_engine_available('unstructured'):
        try:
            import unstructured
            info['versions']['unstructured'] = getattr(unstructured, '__version__', 'unknown')
        except:
            pass
    
    if is_engine_available('pdfplumber'):
        try:
            import pdfplumber
            info['versions']['pdfplumber'] = getattr(pdfplumber, '__version__', 'unknown')
        except:
            pass
    
    if is_engine_available('pymupdf'):
        try:
            import fitz
            info['versions']['pymupdf'] = fitz.version[0] if hasattr(fitz, 'version') else 'unknown'
        except:
            pass
    
    return info
