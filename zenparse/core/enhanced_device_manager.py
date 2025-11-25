"""
增强设备管理器

支持苹果芯片(MPS)、NVIDIA显卡(CUDA)、CPU等多种设备的自动检测和优化配置
"""

import os
import torch
import psutil
import platform
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .logger import get_logger
from .exceptions import DeviceError, CUDAError, MemoryError

# 可选依赖
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class DeviceType(Enum):
    """设备类型枚举"""
    MPS = "mps"      # Apple Silicon GPU
    CUDA = "cuda"    # NVIDIA GPU
    CPU = "cpu"      # CPU


@dataclass
class DeviceConfig:
    """设备配置"""
    device_type: DeviceType
    torch_dtype: str
    max_memory: str
    batch_size: int
    gradient_accumulation_steps: int
    device_id: Optional[int] = None


@dataclass
class DeviceInfo:
    """设备信息"""
    device_type: DeviceType
    device_name: str
    memory_total: float  # GB
    memory_available: float  # GB
    compute_capability: Optional[str] = None
    device_id: Optional[int] = None


class EnhancedDeviceManager:
    """增强设备管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """单例模式：确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化设备管理器（只在第一次调用时完整初始化）"""
        # 如果已经初始化过，直接返回
        if self._initialized:
            return
        
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # 配置参数
        self.auto_detect = self.config.get('auto_detect', True)
        self.force_device = self.config.get('force_device')
        
        # 设备信息
        self.available_devices = self._detect_all_devices()
        self.optimal_device_info = self._determine_optimal_device()
        self.device_configs = self._load_device_configs()
        
        # 只在第一次初始化时输出日志
        self.logger.info(f"设备管理器初始化完成")
        self.logger.info(f"检测到设备: {[d.device_type.value for d in self.available_devices]}")
        self.logger.info(f"最优设备: {self.optimal_device_info.device_type.value}")
        
        # 标记为已初始化
        self._initialized = True
    
    def _detect_all_devices(self) -> List[DeviceInfo]:
        """检测所有可用设备"""
        devices = []
        
        # 检测CPU
        cpu_info = self._detect_cpu()
        if cpu_info:
            devices.append(cpu_info)
        
        # 检测CUDA设备
        cuda_devices = self._detect_cuda_devices()
        devices.extend(cuda_devices)
        
        # 检测MPS设备 (Apple Silicon)
        mps_info = self._detect_mps()
        if mps_info:
            devices.append(mps_info)
        
        return devices
    
    def _detect_cpu(self) -> Optional[DeviceInfo]:
        """检测CPU信息"""
        try:
            cpu_name = platform.processor()
            if not cpu_name:
                cpu_name = f"{platform.machine()} CPU"
            
            memory = psutil.virtual_memory()
            total_memory = memory.total / (1024**3)  # GB
            available_memory = memory.available / (1024**3)  # GB
            
            return DeviceInfo(
                device_type=DeviceType.CPU,
                device_name=cpu_name,
                memory_total=round(total_memory, 2),
                memory_available=round(available_memory, 2)
            )
        except Exception as e:
            self.logger.warning(f"CPU检测失败: {str(e)}")
            return None
    
    def _detect_cuda_devices(self) -> List[DeviceInfo]:
        """检测CUDA设备"""
        devices = []
        
        if not torch.cuda.is_available():
            return devices
        
        try:
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                try:
                    # 基本信息
                    device_name = torch.cuda.get_device_name(i)
                    device_props = torch.cuda.get_device_properties(i)
                    
                    # 内存信息
                    memory_total = device_props.total_memory / (1024**3)  # GB
                    memory_free, memory_total_check = torch.cuda.mem_get_info(i)
                    memory_available = memory_free / (1024**3)  # GB
                    
                    # 计算能力
                    compute_capability = f"{device_props.major}.{device_props.minor}"
                    
                    device_info = DeviceInfo(
                        device_type=DeviceType.CUDA,
                        device_name=device_name,
                        memory_total=round(memory_total, 2),
                        memory_available=round(memory_available, 2),
                        compute_capability=compute_capability,
                        device_id=i
                    )
                    
                    devices.append(device_info)
                    
                except Exception as e:
                    self.logger.warning(f"CUDA设备 {i} 检测失败: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"CUDA设备检测失败: {str(e)}")
        
        return devices
    
    def _detect_mps(self) -> Optional[DeviceInfo]:
        """检测Apple Silicon MPS设备"""
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            return None
        
        try:
            # Apple Silicon信息
            if platform.system() == 'Darwin':
                # 获取系统信息
                cpu_name = platform.processor()
                if not cpu_name:
                    # 尝试获取更详细的CPU信息
                    import subprocess
                    try:
                        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            cpu_name = result.stdout.strip()
                        else:
                            cpu_name = "Apple Silicon"
                    except:
                        cpu_name = "Apple Silicon"
                
                # 内存信息
                memory = psutil.virtual_memory()
                total_memory = memory.total / (1024**3)  # GB
                available_memory = memory.available / (1024**3)  # GB
                
                return DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_name=f"MPS ({cpu_name})",
                    memory_total=round(total_memory, 2),
                    memory_available=round(available_memory, 2)
                )
                
        except Exception as e:
            self.logger.warning(f"MPS设备检测失败: {str(e)}")
        
        return None
    
    def _determine_optimal_device(self) -> DeviceInfo:
        """确定最优设备"""
        if self.force_device:
            # 强制指定设备
            forced_type = DeviceType(self.force_device)
            for device in self.available_devices:
                if device.device_type == forced_type:
                    self.logger.info(f"使用强制指定设备: {device.device_type.value}")
                    return device
            
            self.logger.warning(f"强制指定的设备 {self.force_device} 不可用，回退到自动选择")
        
        if not self.auto_detect:
            # 默认使用CPU
            for device in self.available_devices:
                if device.device_type == DeviceType.CPU:
                    return device
        
        # 自动选择最优设备
        # 优先级: CUDA > MPS > CPU
        priority_order = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
        
        for device_type in priority_order:
            candidates = [d for d in self.available_devices if d.device_type == device_type]
            if candidates:
                if device_type == DeviceType.CUDA:
                    # 选择显存最大的CUDA设备
                    best_cuda = max(candidates, key=lambda x: x.memory_available)
                    return best_cuda
                else:
                    return candidates[0]
        
        # 如果没有找到任何设备，抛出异常
        raise DeviceError("未找到可用设备")
    
    def _load_device_configs(self) -> Dict[DeviceType, DeviceConfig]:
        """加载设备配置"""
        configs = {}
        
        config_mapping = {
            DeviceType.MPS: self.config.get('mps', {}),
            DeviceType.CUDA: self.config.get('cuda', {}),
            DeviceType.CPU: self.config.get('cpu', {})
        }
        
        default_configs = {
            DeviceType.MPS: {
                'torch_dtype': 'float16',
                'max_memory': '16GB',
                'batch_size': 1,
                'gradient_accumulation_steps': 4
            },
            DeviceType.CUDA: {
                'torch_dtype': 'float16',
                'max_memory': '20GB',
                'batch_size': 2,
                'gradient_accumulation_steps': 2
            },
            DeviceType.CPU: {
                'torch_dtype': 'float32',
                'max_memory': '8GB',
                'batch_size': 1,
                'gradient_accumulation_steps': 8
            }
        }
        
        for device_type, config_dict in config_mapping.items():
            default_config = default_configs[device_type]
            merged_config = {**default_config, **config_dict}
            
            configs[device_type] = DeviceConfig(
                device_type=device_type,
                torch_dtype=merged_config['torch_dtype'],
                max_memory=merged_config['max_memory'],
                batch_size=merged_config['batch_size'],
                gradient_accumulation_steps=merged_config['gradient_accumulation_steps']
            )
        
        return configs
    
    def get_optimal_device(self) -> str:
        """获取最优设备标识符"""
        device_info = self.optimal_device_info
        
        if device_info.device_type == DeviceType.CUDA:
            return f"cuda:{device_info.device_id}" if device_info.device_id is not None else "cuda"
        elif device_info.device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"
    
    def get_device_config(self, device_type: Optional[DeviceType] = None) -> DeviceConfig:
        """获取设备配置"""
        if device_type is None:
            device_type = self.optimal_device_info.device_type
        
        return self.device_configs.get(device_type, self.device_configs[DeviceType.CPU])
    
    def get_torch_dtype(self, device_type: Optional[DeviceType] = None):
        """获取PyTorch数据类型"""
        config = self.get_device_config(device_type)
        dtype_mapping = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16
        }
        return dtype_mapping.get(config.torch_dtype, torch.float32)
    
    def get_memory_limit_bytes(self, device_type: Optional[DeviceType] = None) -> int:
        """获取内存限制(字节)"""
        config = self.get_device_config(device_type)
        memory_str = config.max_memory.lower()
        
        if memory_str.endswith('gb'):
            return int(float(memory_str[:-2]) * 1024**3)
        elif memory_str.endswith('mb'):
            return int(float(memory_str[:-2]) * 1024**2)
        else:
            # 默认为GB
            return int(float(memory_str) * 1024**3)
    
    def optimize_for_device(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """根据设备优化模型配置"""
        device_config = self.get_device_config()
        optimized_config = model_config.copy()
        
        # 设备相关优化
        optimized_config.update({
            'device': self.get_optimal_device(),
            'torch_dtype': self.get_torch_dtype(),
            'batch_size': device_config.batch_size,
            'gradient_accumulation_steps': device_config.gradient_accumulation_steps,
            'max_memory': self.get_memory_limit_bytes()
        })
        
        # 设备特定优化
        if self.optimal_device_info.device_type == DeviceType.MPS:
            # MPS特定优化
            optimized_config.update({
                'use_cache': False,  # MPS可能有缓存问题
                'low_cpu_mem_usage': True
            })
        elif self.optimal_device_info.device_type == DeviceType.CUDA:
            # CUDA特定优化
            optimized_config.update({
                'use_cache': True,
                'use_flash_attention_2': True  # 如果支持
            })
        else:
            # CPU特定优化
            optimized_config.update({
                'use_cache': True,
                'low_cpu_mem_usage': False,
                'torch_dtype': torch.float32  # CPU通常使用float32
            })
        
        return optimized_config
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        system_info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'optimal_device': self.get_optimal_device(),
            'available_devices': []
        }
        
        for device in self.available_devices:
            device_dict = {
                'type': device.device_type.value,
                'name': device.device_name,
                'memory_total_gb': device.memory_total,
                'memory_available_gb': device.memory_available
            }
            
            if device.compute_capability:
                device_dict['compute_capability'] = device.compute_capability
            if device.device_id is not None:
                device_dict['device_id'] = device.device_id
                
            system_info['available_devices'].append(device_dict)
        
        return system_info
    
    def validate_device_compatibility(self, required_memory_gb: float = 4.0) -> Tuple[bool, str]:
        """验证设备兼容性"""
        device_info = self.optimal_device_info
        
        # 检查内存要求
        if device_info.memory_available < required_memory_gb:
            return False, f"设备内存不足: 需要{required_memory_gb}GB, 可用{device_info.memory_available}GB"
        
        # 检查设备特定要求
        if device_info.device_type == DeviceType.CUDA:
            if device_info.compute_capability:
                major_version = float(device_info.compute_capability.split('.')[0])
                if major_version < 6.0:
                    return False, f"CUDA计算能力过低: {device_info.compute_capability} < 6.0"
        
        return True, "设备兼容"
    
    def cleanup(self):
        """清理资源"""
        if self.optimal_device_info.device_type == DeviceType.CUDA:
            try:
                torch.cuda.empty_cache()
                self.logger.info("CUDA缓存已清理")
            except Exception as e:
                self.logger.warning(f"CUDA缓存清理失败: {str(e)}")


# 工厂函数
def create_device_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedDeviceManager:
    """创建设备管理器"""
    return EnhancedDeviceManager(config)


def get_optimal_device_info() -> Dict[str, Any]:
    """获取最优设备信息"""
    manager = EnhancedDeviceManager()
    return manager.get_system_info()


def detect_device_type() -> str:
    """快速检测设备类型"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# 向后兼容函数
def get_device_info() -> Dict[str, Any]:
    """获取设备信息（兼容旧API）"""
    manager = EnhancedDeviceManager()
    return manager.get_system_info()

def get_memory_usage() -> Dict[str, Any]:
    """获取内存使用情况（兼容旧API）"""
    manager = EnhancedDeviceManager()
    system_info = manager.get_system_info()
    return {
        'cpu_memory': {
            'total_gb': system_info['available_devices'][0]['memory_total_gb'],
            'available_gb': system_info['available_devices'][0]['memory_available_gb']
        }
    }