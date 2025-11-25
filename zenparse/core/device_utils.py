"""
设备管理工具

提供GPU/CPU设备检测、资源管理、内存优化等功能。
"""

import os
import torch
import psutil
from typing import Dict, List, Optional, Tuple, Any
from .logger import get_logger
from .exceptions import DeviceError, CUDAError, MemoryError

# 可选依赖
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class DeviceManager:
    """设备资源管理器"""
    
    def __init__(self):
        """初始化设备管理器"""
        self.logger = get_logger(__name__)
        self.device_info = self._detect_devices()
        self.optimal_device = self._determine_optimal_device()
        
        self.logger.info(f"设备管理器初始化完成, 最优设备: {self.optimal_device}")
    
    def _detect_devices(self) -> Dict[str, Any]:
        """检测可用设备"""
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': 0,
            'cuda_devices': [],
            'mps_available': False,  # Apple Silicon MPS 支持
            'mps_device': None,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }
        
        # MPS (Metal Performance Shaders) 设备检测 - Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                device_info['mps_available'] = True
                device_info['mps_device'] = {
                    'type': 'MPS',
                    'name': 'Apple Silicon GPU',
                    'is_built': torch.backends.mps.is_built(),
                    'available': torch.backends.mps.is_available()
                }
                self.logger.info("检测到 Apple Silicon MPS 设备")
            except Exception as e:
                self.logger.debug(f"MPS 检测失败: {str(e)}")
        
        # CUDA设备检测
        if device_info['cuda_available']:
            try:
                device_info['cuda_device_count'] = torch.cuda.device_count()
                device_info['cuda_version'] = torch.version.cuda
                
                # 获取每个GPU的详细信息
                for i in range(device_info['cuda_device_count']):
                    gpu_props = torch.cuda.get_device_properties(i)
                    device_info['cuda_devices'].append({
                        'id': i,
                        'name': gpu_props.name,
                        'total_memory_gb': round(gpu_props.total_memory / (1024**3), 2),
                        'major': gpu_props.major,
                        'minor': gpu_props.minor,
                        'multi_processor_count': gpu_props.multi_processor_count
                    })
                
                # 使用GPUtil获取GPU使用率信息（如果可用）
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        for i, gpu in enumerate(gpus):
                            if i < len(device_info['cuda_devices']):
                                device_info['cuda_devices'][i].update({
                                    'memory_used_gb': round(gpu.memoryUsed / 1024, 2),
                                    'memory_free_gb': round(gpu.memoryFree / 1024, 2),
                                    'memory_util_percent': gpu.memoryUtil * 100,
                                    'gpu_util_percent': gpu.load * 100,
                                    'temperature': gpu.temperature
                                })
                    except Exception as e:
                        self.logger.warning(f"无法获取GPU详细信息: {str(e)}")
                    
            except Exception as e:
                self.logger.error(f"CUDA设备检测失败: {str(e)}")
                device_info['cuda_available'] = False
        
        return device_info
    
    def _determine_optimal_device(self) -> str:
        """确定最优设备"""
        # 优先级: CUDA > MPS > CPU
        if self.device_info['cuda_available']:
            # 如果有 CUDA，继续执行下面的 CUDA 逻辑
            pass
        elif self.device_info.get('mps_available', False):
            # 如果有 MPS (Apple Silicon)，优先使用
            return "mps"
        else:
            return "cpu"
        
        # CUDA 设备选择逻辑（只有 CUDA 可用时才执行到这里）
        
        # 选择内存最多且使用率最低的GPU
        best_gpu = None
        best_score = -1
        
        for gpu in self.device_info['cuda_devices']:
            # 计算评分: 可用内存 * (1 - 使用率)
            memory_free = gpu.get('memory_free_gb', gpu['total_memory_gb'])
            gpu_util = gpu.get('gpu_util_percent', 0) / 100
            score = memory_free * (1 - gpu_util)
            
            if score > best_score:
                best_score = score
                best_gpu = gpu
        
        if best_gpu:
            return f"cuda:{best_gpu['id']}"
        else:
            return "cuda:0" if self.device_info['cuda_device_count'] > 0 else "cpu"
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        # 更新实时信息
        current_info = self.device_info.copy()
        
        # 确定当前使用的设备名称
        if torch.cuda.is_available():
            torch_device = torch.cuda.get_device_name()
        elif self.device_info.get('mps_available', False):
            torch_device = "Apple Silicon GPU (MPS)"
        else:
            torch_device = "CPU"
        
        current_info.update({
            'current_memory_usage': self.get_memory_usage(),
            'optimal_device': self.optimal_device,
            'torch_device': torch_device
        })
        return current_info
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = {
            'cpu_memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
                'percent': psutil.virtual_memory().percent
            }
        }
        
        # MPS 内存信息 (Apple Silicon)
        if self.device_info.get('mps_available', False):
            try:
                # MPS 使用统一内存架构，报告系统内存使用情况
                memory_info['mps_memory'] = {
                    'type': 'Unified Memory',
                    'shared_with_cpu': True,
                    'total_gb': memory_info['cpu_memory']['total_gb'],
                    'available_gb': memory_info['cpu_memory']['available_gb'],
                    'note': 'MPS uses unified memory architecture'
                }
            except Exception as e:
                self.logger.debug(f"获取 MPS 内存信息失败: {str(e)}")
        
        # CUDA GPU内存信息
        if self.device_info['cuda_available']:
            memory_info['gpu_memory'] = {}
            for i in range(self.device_info['cuda_device_count']):
                try:
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    
                    memory_info['gpu_memory'][f'cuda:{i}'] = {
                        'allocated_gb': round(allocated, 2),
                        'cached_gb': round(cached, 2),
                        'total_gb': round(total, 2),
                        'free_gb': round(total - allocated, 2),
                        'utilization_percent': round((allocated / total) * 100, 2)
                    }
                except Exception as e:
                    self.logger.warning(f"获取GPU {i}内存信息失败: {str(e)}")
        
        return memory_info
    
    def clear_gpu_cache(self, device_id: Optional[int] = None):
        """清理GPU缓存"""
        # 清理 MPS 缓存
        if self.device_info.get('mps_available', False):
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    self.logger.info("已清理 MPS 缓存")
            except Exception as e:
                self.logger.warning(f"清理 MPS 缓存失败: {str(e)}")
        
        # 清理 CUDA 缓存
        if not self.device_info['cuda_available']:
            if not self.device_info.get('mps_available', False):
                self.logger.warning("CUDA 和 MPS 均不可用，无法清理GPU缓存")
            return
        
        try:
            if device_id is not None:
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                self.logger.info(f"已清理GPU {device_id}缓存")
            else:
                # 清理所有GPU缓存
                for i in range(self.device_info['cuda_device_count']):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                self.logger.info("已清理所有GPU缓存")
                
        except Exception as e:
            raise CUDAError(f"清理GPU缓存失败: {str(e)}")
    
    def set_memory_fraction(self, fraction: float, device_id: Optional[int] = None):
        """设置GPU内存使用限制"""
        if not self.device_info['cuda_available']:
            raise CUDAError("CUDA不可用")
        
        if not 0 < fraction <= 1:
            raise ValueError("内存比例必须在0到1之间")
        
        try:
            device_id = device_id or 0
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            self.logger.info(f"GPU {device_id}内存限制设置为 {fraction*100:.1f}%")
            
        except Exception as e:
            raise CUDAError(f"设置内存限制失败: {str(e)}")
    
    def optimize_memory(self, 
                       clear_cache: bool = True,
                       set_memory_fraction: Optional[float] = None) -> Dict[str, Any]:
        """内存优化"""
        optimization_results = {
            'actions_taken': [],
            'memory_before': self.get_memory_usage(),
            'memory_after': None
        }
        
        # 清理缓存
        if clear_cache:
            self.clear_gpu_cache()
            optimization_results['actions_taken'].append('cleared_gpu_cache')
        
        # 设置内存限制
        if set_memory_fraction:
            self.set_memory_fraction(set_memory_fraction)
            optimization_results['actions_taken'].append(f'set_memory_fraction_{set_memory_fraction}')
        
        # Python垃圾回收
        import gc
        gc.collect()
        optimization_results['actions_taken'].append('python_gc_collect')
        
        # 获取优化后的内存信息
        optimization_results['memory_after'] = self.get_memory_usage()
        
        self.logger.info(f"内存优化完成: {optimization_results['actions_taken']}")
        return optimization_results
    
    def check_memory_requirements(self, required_gb: float) -> Dict[str, bool]:
        """检查内存需求是否满足"""
        memory_info = self.get_memory_usage()
        results = {}
        
        # 检查CPU内存
        cpu_available = memory_info['cpu_memory']['available_gb']
        results['cpu_sufficient'] = cpu_available >= required_gb
        
        # 检查GPU内存
        if 'gpu_memory' in memory_info:
            for device, gpu_mem in memory_info['gpu_memory'].items():
                gpu_available = gpu_mem['free_gb']
                results[f'{device}_sufficient'] = gpu_available >= required_gb
        
        return results
    
    def get_optimal_batch_size(self, 
                             model_memory_gb: float,
                             sample_memory_mb: float,
                             safety_margin: float = 0.8) -> int:
        """估算最优批次大小"""
        if self.optimal_device == "cpu":
            available_memory_gb = self.get_memory_usage()['cpu_memory']['available_gb']
        else:
            gpu_id = int(self.optimal_device.split(':')[1])
            gpu_memory = self.get_memory_usage()['gpu_memory'][self.optimal_device]
            available_memory_gb = gpu_memory['free_gb']
        
        # 计算可用于batch的内存
        usable_memory_gb = (available_memory_gb - model_memory_gb) * safety_margin
        
        if usable_memory_gb <= 0:
            return 1
        
        # 计算batch size
        batch_size = int((usable_memory_gb * 1024) // sample_memory_mb)
        return max(1, batch_size)
    
    def monitor_resources(self) -> Dict[str, Any]:
        """资源监控"""
        return {
            'timestamp': torch.cuda.Event(enable_timing=True).record(),
            'device_info': self.get_device_info(),
            'memory_usage': self.get_memory_usage(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'disk_usage': {
                'total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
                'used_gb': round(psutil.disk_usage('/').used / (1024**3), 2),
                'free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
                'percent': round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 2)
            }
        }


def get_optimal_device() -> str:
    """获取最优设备"""
    device_manager = DeviceManager()
    
    return device_manager.optimal_device


def get_available_device_types() -> Dict[str, bool]:
    """获取所有可用的设备类型"""
    device_manager = DeviceManager()
    return {
        'cuda': device_manager.device_info.get('cuda_available', False),
        'mps': device_manager.device_info.get('mps_available', False),
        'cpu': True  # CPU 总是可用
    }


def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    device_manager = DeviceManager()
    return device_manager.get_device_info()


def clear_gpu_memory():
    """清理GPU内存"""
    device_manager = DeviceManager()
    device_manager.clear_gpu_cache()


def optimize_memory_usage(
    clear_cache: bool = True,
    memory_fraction: Optional[float] = None
) -> Dict[str, Any]:
    """优化内存使用"""
    device_manager = DeviceManager()
    return device_manager.optimize_memory(
        clear_cache=clear_cache,
        set_memory_fraction=memory_fraction
    )


def check_gpu_availability() -> bool:
    """检查GPU可用性（包括 CUDA 和 MPS）"""
    # 检查 CUDA
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    
    # 检查 MPS (Apple Silicon)
    mps_available = False
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
    
    return cuda_available or mps_available


def get_gpu_memory_info() -> Dict[str, Any]:
    """获取GPU内存信息（包括 CUDA 和 MPS）"""
    if not check_gpu_availability():
        return {}
    
    device_manager = DeviceManager()
    memory_usage = device_manager.get_memory_usage()
    
    gpu_memory = {}
    # 包含 CUDA GPU 内存
    if 'gpu_memory' in memory_usage:
        gpu_memory.update(memory_usage['gpu_memory'])
    
    # 包含 MPS 内存信息
    if 'mps_memory' in memory_usage:
        gpu_memory['mps'] = memory_usage['mps_memory']
    
    return gpu_memory


def estimate_model_memory(model_params: int, precision: str = "float16") -> float:
    """
    估算模型内存需求
    
    Args:
        model_params: 模型参数数量
        precision: 精度类型
        
    Returns:
        内存需求(GB)
    """
    bytes_per_param = {
        'float32': 4,
        'float16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    param_bytes = bytes_per_param.get(precision, 2)
    
    # 模型参数 + 梯度 + 优化器状态 + 缓存
    total_bytes = model_params * param_bytes * 3  # 保守估计3倍
    
    return total_bytes / (1024**3)  # 转换为GB


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        初始化内存监控器
        
        Args:
            device_manager: 设备管理器实例
        """
        self.device_manager = device_manager or DeviceManager()
        self.logger = get_logger(__name__)
        self.initial_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        self.initial_memory = self.device_manager.get_memory_usage()
        self.peak_memory = self.initial_memory.copy()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        final_memory = self.device_manager.get_memory_usage()
        
        # 计算内存变化
        memory_delta = self._calculate_memory_delta(
            self.initial_memory, 
            final_memory
        )
        
        self.logger.info(
            "内存监控结果",
            memory_delta=memory_delta,
            peak_memory=self.peak_memory,
            final_memory=final_memory
        )
    
    def _calculate_memory_delta(self, initial: Dict, final: Dict) -> Dict:
        """计算内存变化"""
        delta = {}
        
        # CPU内存变化
        cpu_initial = initial['cpu_memory']['used_gb']
        cpu_final = final['cpu_memory']['used_gb']
        delta['cpu_delta_gb'] = round(cpu_final - cpu_initial, 2)
        
        # GPU内存变化
        if 'gpu_memory' in initial and 'gpu_memory' in final:
            delta['gpu_deltas'] = {}
            for device in initial['gpu_memory']:
                if device in final['gpu_memory']:
                    gpu_initial = initial['gpu_memory'][device]['allocated_gb']
                    gpu_final = final['gpu_memory'][device]['allocated_gb']
                    delta['gpu_deltas'][device] = round(gpu_final - gpu_initial, 2)
        
        return delta
    
    def update_peak_memory(self):
        """更新峰值内存记录"""
        current_memory = self.device_manager.get_memory_usage()
        
        # 更新CPU峰值
        if (current_memory['cpu_memory']['used_gb'] > 
            self.peak_memory['cpu_memory']['used_gb']):
            self.peak_memory['cpu_memory'] = current_memory['cpu_memory']
        
        # 更新GPU峰值
        if 'gpu_memory' in current_memory and 'gpu_memory' in self.peak_memory:
            for device in current_memory['gpu_memory']:
                if device in self.peak_memory['gpu_memory']:
                    if (current_memory['gpu_memory'][device]['allocated_gb'] >
                        self.peak_memory['gpu_memory'][device]['allocated_gb']):
                        self.peak_memory['gpu_memory'][device] = current_memory['gpu_memory'][device]