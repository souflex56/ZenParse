"""
ZenParse 异常处理模块

为管线与业务组件提供统一的异常基类，便于序列化和日志记录。
"""

from typing import Optional, Any, Dict


class ZenParseError(Exception):
    """ZenParse 基础异常"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """结构化异常信息，方便落盘或返回 API"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return f"[{self.error_code}] {self.message}"


# 向后兼容，保留旧名称
ZenSeekerException = ZenParseError


class ConfigError(ZenParseError):
    """配置相关异常"""


class ModelError(ZenParseError):
    """模型相关异常"""


class ModelLoadError(ModelError):
    """模型加载异常"""


class ModelInferenceError(ModelError):
    """模型推理异常"""


class DataProcessingError(ZenParseError):
    """数据处理异常"""


class DocumentProcessingError(DataProcessingError):
    """文档处理异常"""


class ChunkingError(DataProcessingError):
    """文档分块异常"""


class StyleAnalysisError(DataProcessingError):
    """风格分析异常"""


class RetrievalError(ZenParseError):
    """检索相关异常"""


class IndexError(RetrievalError):
    """索引异常"""


class SearchError(RetrievalError):
    """搜索异常"""


class RankingError(RetrievalError):
    """重排序异常"""


class GenerationError(ZenParseError):
    """生成相关异常"""


class PromptError(GenerationError):
    """Prompt异常"""


class ReflectionError(GenerationError):
    """反思生成异常"""


class StyleAlignmentError(GenerationError):
    """风格对齐异常"""


class TrainingError(ZenParseError):
    """训练相关异常"""


class SFTTrainingError(TrainingError):
    """SFT训练异常"""


class DPOTrainingError(TrainingError):
    """DPO训练异常"""


class PreferenceGenerationError(TrainingError):
    """偏好对生成异常"""


class EvaluationError(ZenParseError):
    """评估相关异常"""


class MetricError(EvaluationError):
    """指标计算异常"""


class BenchmarkError(EvaluationError):
    """基准测试异常"""


class APIError(ZenParseError):
    """API相关异常"""


class AuthenticationError(APIError):
    """认证异常"""


class RateLimitError(APIError):
    """限流异常"""


class ValidationError(APIError):
    """输入验证异常"""


class DeviceError(ZenParseError):
    """设备相关异常"""


class CUDAError(DeviceError):
    """CUDA异常"""


class MemoryError(DeviceError):
    """内存异常"""


class LazyImportError(ZenParseError):
    """延迟加载依赖失败"""


def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    统一异常处理函数

    Args:
        exception: 异常对象
        context: 上下文信息
        reraise: 是否重新抛出异常

    Returns:
        异常信息字典
    """
    error_info = {
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "context": context or {},
    }

    # 如果是ZenParse异常，获取详细信息
    if isinstance(exception, ZenParseError):
        error_info.update(exception.to_dict())

    from .logger import get_logger

    logger = get_logger(__name__)
    logger.error(f"异常处理: {error_info}")

    if reraise:
        raise exception

    return error_info


class ExceptionContext:
    """异常上下文管理器"""

    def __init__(self, operation: str, reraise: bool = True, default_return=None):
        self.operation = operation
        self.reraise = reraise
        self.default_return = default_return
        self.logger = None

    def __enter__(self):
        from .logger import get_logger

        self.logger = get_logger(__name__)
        self.logger.debug(f"开始执行操作: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_info = {
                "operation": self.operation,
                "exception_type": exc_type.__name__,
                "message": str(exc_val),
            }

            self.logger.error(f"操作失败: {error_info}")

            if not self.reraise:
                return True  # 抑制异常
        else:
            self.logger.debug(f"操作完成: {self.operation}")

        return False  # 正常传播异常


def zen_parse_exception_handler(func):
    """ZenParse异常处理装饰器"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZenParseError:
            raise
        except Exception as e:
            raise ZenParseError(
                message=f"执行 {func.__name__} 时发生未预期的错误: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "original_exception": type(e).__name__,
                },
            ) from e

    return wrapper


# 兼容旧名称，方便复用原有代码
zen_seeker_exception_handler = zen_parse_exception_handler
