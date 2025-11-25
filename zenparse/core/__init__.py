"""
ZenParse 核心模块

提供日志、设备管理和异常处理等基础设施。
"""

from .config_manager import ConfigManager
from .logger import Logger, get_logger
from .enhanced_device_manager import EnhancedDeviceManager as DeviceManager
from .enhanced_device_manager import detect_device_type as get_optimal_device
from .exceptions import (
    ZenParseError,
    ZenSeekerException,  # 向后兼容
    ConfigError,
    ModelError,
    DataProcessingError,
    RetrievalError,
    GenerationError,
)

__all__ = [
    "ConfigManager",
    "Logger",
    "get_logger",
    "DeviceManager",
    "get_optimal_device",
    "ZenParseError",
    "ZenSeekerException",
    "ConfigError",
    "ModelError",
    "DataProcessingError",
    "RetrievalError",
    "GenerationError",
]
