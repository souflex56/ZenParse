"""
日志系统模块

提供统一的日志记录功能，支持结构化日志、多级别输出、文件轮转等功能。
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any, Dict, Union
from datetime import datetime
from loguru import logger


class Logger:
    """ZenParse日志管理器"""
    
    def __init__(
        self,
        name: str = "zenparse",
        level: str = "INFO",
        log_file: Optional[str] = None,
        rotation: str = "1 day",
        retention: str = "7 days",
        format_type: str = "detailed"
    ):
        """
        初始化日志管理器
        
        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径
            rotation: 日志轮转规则
            retention: 日志保留时间
            format_type: 格式类型 (simple/detailed/json)
        """
        self.name = name
        self.level = level.upper()
        self.log_file = log_file
        self.rotation = rotation
        self.retention = retention
        self.format_type = format_type
        
        # 移除默认处理器
        logger.remove()
        
        # 设置格式
        self.formats = {
            "simple": "{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}",
            "detailed": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {name}:{function}:{line} | {extra} - {message}",
            "json": '{"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", "level": "{level}", "module": "{name}", "function": "{function}", "line": {line}, "message": "{message}", "extra": {extra}}'
        }
        
        self._setup_handlers()
        
        # 绑定上下文信息
        self.logger = logger.bind(name=self.name)
    
    def _setup_handlers(self):
        """设置日志处理器"""
        format_str = self.formats.get(self.format_type, self.formats["detailed"])
        
        # 控制台处理器
        logger.add(
            sys.stdout,
            level=self.level,
            format=format_str,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 文件处理器
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                self.log_file,
                level=self.level,
                format=format_str,
                rotation=self.rotation,
                retention=self.retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
                encoding="utf-8"
            )
        
        # 错误日志单独文件
        if self.log_file:
            error_log_file = log_path.parent / f"{log_path.stem}_error.log"
            logger.add(
                str(error_log_file),
                level="ERROR",
                format=format_str,
                rotation=self.rotation,
                retention=self.retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
                encoding="utf-8"
            )
        
        # 性能日志文件
        if self.log_file:
            perf_log_file = log_path.parent / f"{log_path.stem}_performance.log"
            logger.add(
                str(perf_log_file),
                level="INFO",
                format=format_str,
                filter=lambda record: record["extra"].get("category") == "performance",
                rotation=self.rotation,
                retention="30 days",
                compression="zip",
                encoding="utf-8"
            )
    
    def debug(self, message: str, **context):
        """调试日志"""
        self.logger.bind(**context).debug(message)
    
    def info(self, message: str, **context):
        """信息日志"""
        self.logger.bind(**context).info(message)
    
    def warning(self, message: str, **context):
        """警告日志"""
        self.logger.bind(**context).warning(message)
    
    def error(self, message: str, **context):
        """错误日志"""
        self.logger.bind(**context).error(message)
    
    def critical(self, message: str, **context):
        """严重错误日志"""
        self.logger.bind(**context).critical(message)
    
    def exception(self, message: str, **context):
        """异常日志(包含堆栈跟踪)"""
        self.logger.bind(**context).exception(message)
    
    def performance(self, message: str, **metrics):
        """性能日志"""
        context = {"category": "performance", **metrics}
        self.logger.bind(**context).info(message)
    
    def audit(self, action: str, user: Optional[str] = None, **details):
        """审计日志"""
        context = {
            "category": "audit",
            "action": action,
            "user": user or "system",
            "timestamp": datetime.now().isoformat(),
            **details
        }
        self.logger.bind(**context).info(f"审计事件: {action}")
    
    def security(self, event: str, level: str = "INFO", **details):
        """安全日志"""
        context = {
            "category": "security",
            "event": event,
            "security_level": level,
            **details
        }
        log_method = getattr(self.logger.bind(**context), level.lower())
        log_method(f"安全事件: {event}")
    
    def business(self, event: str, **metrics):
        """业务日志"""
        context = {
            "category": "business", 
            "event": event,
            **metrics
        }
        self.logger.bind(**context).info(f"业务事件: {event}")
    
    def structured_log(
        self, 
        level: str, 
        message: str, 
        category: Optional[str] = None,
        **fields
    ):
        """结构化日志"""
        context = {"category": category, **fields} if category else fields
        log_method = getattr(self.logger.bind(**context), level.lower())
        log_method(message)
    
    def set_level(self, level: str):
        """设置日志级别"""
        self.level = level.upper()
        # 重新设置处理器
        logger.remove()
        self._setup_handlers()
        self.logger = logger.bind(name=self.name)


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger: Logger, operation: str):
        """
        初始化性能日志记录器
        
        Args:
            logger: 日志器实例
            operation: 操作名称
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.performance(
            f"开始执行: {self.operation}",
            operation=self.operation,
            start_time=self.start_time.isoformat()
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        status = "failed" if exc_type is not None else "success"
        
        self.logger.performance(
            f"执行完成: {self.operation}",
            operation=self.operation,
            duration_seconds=duration,
            status=status,
            end_time=self.end_time.isoformat(),
            **self.metrics
        )
    
    def add_metric(self, key: str, value: Any):
        """添加性能指标"""
        self.metrics[key] = value
    
    def add_metrics(self, **metrics):
        """添加多个性能指标"""
        self.metrics.update(metrics)


class LoggerContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: Logger, **context):
        """
        初始化日志上下文
        
        Args:
            logger: 日志器实例
            **context: 上下文信息
        """
        self.logger = logger
        self.context = context
        self.bound_logger = None
    
    def __enter__(self):
        self.bound_logger = self.logger.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 全局日志器实例
_global_logger: Optional[Logger] = None


def get_logger(
    name: str = "zenparse",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        Logger实例
    """
    global _global_logger
    
    if _global_logger is None:
        # 从环境变量或配置获取默认值
        default_level = level or os.getenv("ZENPARSE_LOG_LEVEL", "INFO")
        default_log_file = log_file or os.getenv("ZENPARSE_LOG_FILE", "./logs/zenparse.log")
        
        _global_logger = Logger(
            name=name,
            level=default_level,
            log_file=default_log_file
        )
    
    return _global_logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "detailed"
) -> Logger:
    """
    设置全局日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
        format_type: 格式类型
        
    Returns:
        Logger实例
    """
    global _global_logger
    
    _global_logger = Logger(
        name="zenparse",
        level=level,
        log_file=log_file,
        format_type=format_type
    )
    
    return _global_logger


def reset_logger():
    """重置全局日志器"""
    global _global_logger
    _global_logger = None


# 便捷函数
def log_performance(operation: str, logger: Optional[Logger] = None):
    """
    性能日志装饰器
    
    Args:
        operation: 操作名称
        logger: 日志器实例
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_instance = logger or get_logger()
            with PerformanceLogger(log_instance, operation or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_function_call(logger: Optional[Logger] = None, level: str = "DEBUG"):
    """
    函数调用日志装饰器
    
    Args:
        logger: 日志器实例
        level: 日志级别
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_instance = logger or get_logger()
            
            # 记录函数调用
            log_method = getattr(log_instance, level.lower())
            log_method(
                f"调用函数: {func.__name__}",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                log_method(
                    f"函数执行成功: {func.__name__}",
                    function=func.__name__
                )
                return result
            except Exception as e:
                log_instance.error(
                    f"函数执行失败: {func.__name__}",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
                
        return wrapper
    return decorator
