"""
配置管理器

统一管理ZenSeeker系统的所有配置参数，支持多环境配置、配置验证和热重载。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from .exceptions import ConfigError
from .logger import get_logger


@dataclass
class ModelConfig:
    """模型配置"""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-large"
    
    # 量化配置
    quantization_type: str = "4bit"
    compute_dtype: str = "float16"
    use_double_quant: bool = True
    quant_type: str = "nf4"
    
    # 内存优化
    gradient_checkpointing: bool = True
    use_cache: bool = False
    max_memory_mb: int = 20480


@dataclass
class DeviceConfig:
    """设备配置"""
    auto_detect: bool = True
    force_device: Optional[str] = None
    
    # 不同设备的优化配置
    mps: Dict[str, Any] = field(default_factory=lambda: {
        "torch_dtype": "float16",
        "max_memory": "16GB",
        "batch_size": 1,
        "gradient_accumulation_steps": 4
    })
    cuda: Dict[str, Any] = field(default_factory=lambda: {
        "torch_dtype": "float16",
        "max_memory": "20GB", 
        "batch_size": 2,
        "gradient_accumulation_steps": 2
    })
    cpu: Dict[str, Any] = field(default_factory=lambda: {
        "torch_dtype": "float32",
        "max_memory": "8GB",
        "batch_size": 1,
        "gradient_accumulation_steps": 8
    })


@dataclass 
class ModelAPIConfig:
    """模型API配置"""
    provider: str = "local"
    fallback_order: List[str] = field(default_factory=lambda: ["ollama", "zhipu", "silicon"])
    
    # 本地模型配置
    local: Dict[str, Any] = field(default_factory=lambda: {
        "model_name": "Qwen/Qwen2.5-7B-Instruct"
    })
    
    # Ollama配置
    ollama: Dict[str, Any] = field(default_factory=lambda: {
        "base_url": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "timeout": 30
    })
    
    # 智谱AI配置
    zhipu: Dict[str, Any] = field(default_factory=lambda: {
        "api_key": "${ZHIPU_API_KEY}",
        "model": "glm-4",
        "base_url": "https://open.bigmodel.cn/api/paas/v4"
    })
    
    # 硅基流动配置
    silicon: Dict[str, Any] = field(default_factory=lambda: {
        "api_key": "${SILICON_API_KEY}",
        "model": "Qwen/Qwen2.5-7B-Instruct", 
        "base_url": "https://api.siliconflow.cn/v1"
    })


@dataclass
class DataConfig:
    """数据配置"""
    # 路径配置
    dyp_qa_pairs_path: str = "./data/raw/dyp_qa_pairs.json"
    financial_reports_dir: str = "./data/raw/financial_reports/"
    processed_dir: str = "./data/processed/"
    vector_store_path: str = "./data/embeddings/faiss_index/"
    
    # 文档处理配置 - 分块参数
    parent_chunk_size: int = 4000
    child_chunk_size: int = 1000
    child_chunk_overlap: int = 200
    min_chunk_length: int = 50
    max_chunk_length: int = 2000
    
    # 文档处理配置 - 处理策略
    strategy: str = 'semantic'
    languages: List[str] = field(default_factory=lambda: ['chi_sim', 'eng'])
    
    # 文档处理配置 - 解析配置
    enable_caching: bool = True
    save_intermediate: bool = False
    output_dir: str = './output'


@dataclass
class TrainingConfig:
    """训练配置"""
    # SFT配置
    sft_epochs: int = 5
    sft_batch_size: int = 4
    sft_learning_rate: float = 2e-5
    sft_warmup_ratio: float = 0.03
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # DPO配置
    dpo_epochs: int = 3
    dpo_batch_size: int = 2
    dpo_learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    
    # 偏好生成配置
    preference_dataset_size: int = 5000
    hybrid_probability: float = 0.6


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 4096
    
    # 策略开关
    enable_parent_child: bool = True
    enable_multi_path: bool = True
    enable_style_aware: bool = True
    enable_semantic: bool = True
    enable_keyword: bool = True


@dataclass
class GenerationConfig:
    """生成配置"""
    max_length: int = 2048
    min_length: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # 反思配置
    enable_reflection: bool = True
    max_iterations: int = 3
    style_threshold: float = 0.75
    quality_threshold: float = 0.80


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    
    # 限流配置
    requests_per_minute: int = 60
    burst: int = 10


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_file: str = "./logs/zenparse.log"
    
    # 告警配置
    error_threshold: float = 0.05
    latency_threshold: float = 5.0
    memory_threshold: float = 0.85


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "./config/base_config.yaml"
        self.logger = get_logger(__name__)
        
        # 配置对象
        self._config_data: Dict[str, Any] = {}
        self.model_config: Optional[ModelConfig] = None
        self.device_config: Optional[DeviceConfig] = None
        self.model_api_config: Optional[ModelAPIConfig] = None
        self.data_config: Optional[DataConfig] = None
        self.training_config: Optional[TrainingConfig] = None
        self.retrieval_config: Optional[RetrievalConfig] = None
        self.generation_config: Optional[GenerationConfig] = None
        self.api_config: Optional[APIConfig] = None
        self.monitoring_config: Optional[MonitoringConfig] = None
        
        # 加载配置
        self.load_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            config_path = Path(self.config_path)
            
            if not config_path.exists():
                raise ConfigError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            
            # 解析各模块配置
            self._parse_config()
            
            self.logger.info(f"配置加载成功: {config_path}")
            
        except Exception as e:
            raise ConfigError(f"配置加载失败: {str(e)}")
    
    def _parse_config(self) -> None:
        """解析配置数据"""
        try:
            # 模型配置
            model_cfg = self._config_data.get('model', {})
            quantization_cfg = model_cfg.get('quantization', {})
            memory_cfg = model_cfg.get('memory', {})
            
            self.model_config = ModelConfig(
                base_model=model_cfg.get('base_model', "Qwen/Qwen2.5-7B-Instruct"),
                embedding_model=model_cfg.get('embedding_model', "BAAI/bge-m3"),
                reranker_model=model_cfg.get('reranker_model', "BAAI/bge-reranker-large"),
                quantization_type=quantization_cfg.get('type', '4bit'),
                compute_dtype=quantization_cfg.get('compute_dtype', 'float16'),
                use_double_quant=quantization_cfg.get('use_double_quant', True),
                gradient_checkpointing=memory_cfg.get('gradient_checkpointing', True),
                use_cache=memory_cfg.get('use_cache', False),
                max_memory_mb=memory_cfg.get('max_memory_mb', 20480)
            )
            
            # 设备配置
            device_cfg = self._config_data.get('device_config', {})
            self.device_config = DeviceConfig(
                auto_detect=device_cfg.get('auto_detect', True),
                force_device=device_cfg.get('force_device'),
                mps=device_cfg.get('mps', {}),
                cuda=device_cfg.get('cuda', {}),
                cpu=device_cfg.get('cpu', {})
            )
            
            # 模型API配置
            model_api_cfg = self._config_data.get('api_config', {})
            self.model_api_config = ModelAPIConfig(
                provider=model_api_cfg.get('provider', 'local'),
                fallback_order=model_api_cfg.get('fallback_order', ['ollama', 'zhipu', 'silicon']),
                local=model_api_cfg.get('local', {}),
                ollama=model_api_cfg.get('ollama', {}),
                zhipu=model_api_cfg.get('zhipu', {}),
                silicon=model_api_cfg.get('silicon', {})
            )
            
            # 数据配置
            data_cfg = self._config_data.get('data', {})
            doc_processing_cfg = data_cfg.get('document_processing', {})
            
            self.data_config = DataConfig(
                dyp_qa_pairs_path=data_cfg.get('dyp_qa_pairs_path', "./data/raw/dyp_qa_pairs.json"),
                financial_reports_dir=data_cfg.get('financial_reports_dir', "./data/raw/financial_reports/"),
                processed_dir=data_cfg.get('processed_dir', "./data/processed/"),
                vector_store_path=data_cfg.get('vector_store_path', "./data/embeddings/faiss_index/"),
                # 分块参数
                parent_chunk_size=doc_processing_cfg.get('parent_chunk_size', 4000),
                child_chunk_size=doc_processing_cfg.get('child_chunk_size', 1000),
                child_chunk_overlap=doc_processing_cfg.get('child_chunk_overlap', 200),
                min_chunk_length=doc_processing_cfg.get('min_chunk_length', 50),
                max_chunk_length=doc_processing_cfg.get('max_chunk_length', 2000),
                # 处理策略
                strategy=doc_processing_cfg.get('strategy', 'semantic'),
                languages=doc_processing_cfg.get('languages', ['chi_sim', 'eng']),
                # 解析配置
                enable_caching=doc_processing_cfg.get('enable_caching', True),
                save_intermediate=doc_processing_cfg.get('save_intermediate', False),
                output_dir=doc_processing_cfg.get('output_dir', './output')
            )
            
            # 训练配置
            training_cfg = self._config_data.get('training', {})
            sft_cfg = training_cfg.get('sft', {})
            lora_cfg = sft_cfg.get('lora', {})
            dpo_cfg = training_cfg.get('dpo', {})
            pref_gen_cfg = training_cfg.get('preference_generation', {})
            
            self.training_config = TrainingConfig(
                sft_epochs=sft_cfg.get('epochs', 5),
                sft_batch_size=sft_cfg.get('batch_size', 4),
                sft_learning_rate=sft_cfg.get('learning_rate', 2e-5),
                sft_warmup_ratio=sft_cfg.get('warmup_ratio', 0.03),
                lora_r=lora_cfg.get('r', 16),
                lora_alpha=lora_cfg.get('alpha', 32),
                lora_dropout=lora_cfg.get('dropout', 0.1),
                lora_target_modules=lora_cfg.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
                dpo_epochs=dpo_cfg.get('epochs', 3),
                dpo_batch_size=dpo_cfg.get('batch_size', 2),
                dpo_learning_rate=dpo_cfg.get('learning_rate', 5e-6),
                dpo_beta=dpo_cfg.get('beta', 0.1),
                preference_dataset_size=pref_gen_cfg.get('target_dataset_size', 5000),
                hybrid_probability=pref_gen_cfg.get('strategy_fusion', {}).get('hybrid_probability', 0.6)
            )
            
            # 检索配置
            retrieval_cfg = self._config_data.get('retrieval', {})
            strategies_cfg = retrieval_cfg.get('strategies', {})
            
            self.retrieval_config = RetrievalConfig(
                top_k=retrieval_cfg.get('top_k', 10),
                rerank_top_k=retrieval_cfg.get('rerank_top_k', 5),
                similarity_threshold=retrieval_cfg.get('similarity_threshold', 0.7),
                max_context_length=retrieval_cfg.get('max_context_length', 4096),
                enable_parent_child=strategies_cfg.get('enable_parent_child', True),
                enable_multi_path=strategies_cfg.get('enable_multi_path', True),
                enable_style_aware=strategies_cfg.get('enable_style_aware', True),
                enable_semantic=strategies_cfg.get('enable_semantic', True),
                enable_keyword=strategies_cfg.get('enable_keyword', True)
            )
            
            # 生成配置
            generation_cfg = self._config_data.get('generation', {})
            reflection_cfg = generation_cfg.get('reflection', {})
            
            self.generation_config = GenerationConfig(
                max_length=generation_cfg.get('max_length', 2048),
                min_length=generation_cfg.get('min_length', 50),
                temperature=generation_cfg.get('temperature', 0.7),
                top_p=generation_cfg.get('top_p', 0.9),
                top_k=generation_cfg.get('top_k', 50),
                repetition_penalty=generation_cfg.get('repetition_penalty', 1.1),
                enable_reflection=reflection_cfg.get('enable', True),
                max_iterations=reflection_cfg.get('max_iterations', 3),
                style_threshold=reflection_cfg.get('style_threshold', 0.75),
                quality_threshold=reflection_cfg.get('quality_threshold', 0.80)
            )
            
            # API配置
            api_cfg = self._config_data.get('api', {})
            rate_limiting_cfg = api_cfg.get('rate_limiting', {})
            
            self.api_config = APIConfig(
                host=api_cfg.get('host', "0.0.0.0"),
                port=api_cfg.get('port', 8000),
                reload=api_cfg.get('reload', False),
                workers=api_cfg.get('workers', 4),
                requests_per_minute=rate_limiting_cfg.get('requests_per_minute', 60),
                burst=rate_limiting_cfg.get('burst', 10)
            )
            
            # 监控配置
            monitoring_cfg = self._config_data.get('monitoring', {})
            metrics_cfg = monitoring_cfg.get('metrics', {})
            alerts_cfg = monitoring_cfg.get('alerts', {})
            logging_cfg = monitoring_cfg.get('logging', {})
            
            self.monitoring_config = MonitoringConfig(
                enable=monitoring_cfg.get('enable', True),
                metrics_port=metrics_cfg.get('port', 9090),
                log_level=logging_cfg.get('level', 'INFO'),
                log_file=logging_cfg.get('file', './logs/zenparse.log'),
                error_threshold=alerts_cfg.get('error_threshold', 0.05),
                latency_threshold=alerts_cfg.get('latency_threshold', 5.0),
                memory_threshold=alerts_cfg.get('memory_threshold', 0.85)
            )
            
        except Exception as e:
            raise ConfigError(f"配置解析失败: {str(e)}")
    
    def get_config(self, section: str, key: str = None, default=None) -> Any:
        """
        获取配置值
        
        Args:
            section: 配置节
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            section_data = self._config_data.get(section, {})
            
            if key is None:
                return section_data
            
            return section_data.get(key, default)
            
        except Exception as e:
            self.logger.warning(f"获取配置失败 {section}.{key}: {str(e)}")
            return default
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            section: 配置节
            key: 配置键
            value: 新值
        """
        try:
            if section not in self._config_data:
                self._config_data[section] = {}
            
            self._config_data[section][key] = value
            
            # 重新解析配置
            self._parse_config()
            
            self.logger.info(f"配置更新成功: {section}.{key} = {value}")
            
        except Exception as e:
            raise ConfigError(f"配置更新失败: {str(e)}")
    
    def save_config(self, save_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
        """
        try:
            save_path = save_path or self.config_path
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"配置保存成功: {save_path}")
            
        except Exception as e:
            raise ConfigError(f"配置保存失败: {str(e)}")
    
    def reload_config(self) -> None:
        """重新加载配置"""
        self.load_config()
        self.logger.info("配置重新加载完成")
    
    def validate_config(self) -> Dict[str, bool]:
        """
        验证配置的有效性
        
        Returns:
            验证结果
        """
        validation_results = {}
        
        try:
            # 验证路径存在性
            paths_to_check = [
                ('data_config.dyp_qa_pairs_path', self.data_config.dyp_qa_pairs_path),
                ('data_config.financial_reports_dir', self.data_config.financial_reports_dir),
                ('data_config.processed_dir', self.data_config.processed_dir)
            ]
            
            for path_name, path_value in paths_to_check:
                path = Path(path_value)
                if path_name.endswith('_dir'):
                    validation_results[f"path_exists_{path_name}"] = path.exists() and path.is_dir()
                else:
                    validation_results[f"path_exists_{path_name}"] = path.exists() and path.is_file()
            
            # 验证数值范围
            numerical_validations = [
                ('training_config.sft_learning_rate', 0 < self.training_config.sft_learning_rate < 1),
                ('training_config.dpo_learning_rate', 0 < self.training_config.dpo_learning_rate < 1),
                ('retrieval_config.similarity_threshold', 0 < self.retrieval_config.similarity_threshold <= 1),
                ('generation_config.temperature', 0 < self.generation_config.temperature <= 2),
                ('generation_config.top_p', 0 < self.generation_config.top_p <= 1)
            ]
            
            for check_name, is_valid in numerical_validations:
                validation_results[f"range_valid_{check_name}"] = is_valid
            
            # 验证端口可用性
            validation_results['port_valid_api'] = 1024 <= self.api_config.port <= 65535
            validation_results['port_valid_monitoring'] = 1024 <= self.monitoring_config.metrics_port <= 65535
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            validation_results['validation_error'] = False
        
        return validation_results
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'config_path': self.config_path,
            'python_path': os.environ.get('PYTHONPATH', ''),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'hf_home': os.environ.get('HF_HOME', ''),
            'model_cache_dir': os.environ.get('MODEL_CACHE_DIR', ''),
            'data_path': os.environ.get('DATA_PATH', ''),
            'deployment_env': self.get_config('deployment', 'environment', 'development')
        }


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_path)
    
    return _global_config_manager


def reset_config_manager() -> None:
    """重置全局配置管理器"""
    global _global_config_manager
    _global_config_manager = None
