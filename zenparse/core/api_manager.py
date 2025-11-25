"""
API管理器

支持本地模型、Ollama、智谱AI、硅基流动等多种API提供商的统一调用和自动切换
"""

import os
import json
import asyncio
import httpx
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .logger import get_logger
from .exceptions import APIError, AuthenticationError, RateLimitError


class APIProvider(Enum):
    """API提供商枚举"""
    LOCAL = "local"
    OLLAMA = "ollama"
    ZHIPU = "zhipu"
    SILICON = "silicon"


@dataclass
class APIConfig:
    """API配置"""
    provider: APIProvider
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.7
    extra_params: Dict[str, Any] = None


class BaseAPIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = get_logger(__name__)
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class LocalAPIClient(BaseAPIClient):
    """本地模型API客户端"""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
    
    async def _load_model(self):
        """加载本地模型"""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.logger.info(f"加载本地模型: {self.config.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.logger.info("本地模型加载完成")
            
        except Exception as e:
            raise APIError(f"本地模型加载失败: {str(e)}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        await self._load_model()
        
        try:
            # 编码输入
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            
            # 生成参数
            generate_kwargs = {
                "max_length": inputs.shape[1] + kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "do_sample": True,
                "pad_token_id": self._tokenizer.eos_token_id
            }
            
            # 生成文本
            with torch.no_grad():
                outputs = self._model.generate(inputs, **generate_kwargs)
            
            # 解码输出
            generated_text = self._tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            raise APIError(f"本地模型生成失败: {str(e)}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        # 本地模型的流式生成实现
        full_response = await self.generate(prompt, **kwargs)
        
        # 模拟流式输出
        words = full_response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
            await asyncio.sleep(0.05)  # 模拟延迟
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            await self._load_model()
            return True
        except Exception as e:
            self.logger.warning(f"本地模型健康检查失败: {str(e)}")
            return False


class OllamaAPIClient(BaseAPIClient):
    """Ollama API客户端"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        url = f"{self.config.base_url}/api/generate"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "").strip()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise APIError(f"Ollama模型不存在: {self.config.model_name}")
            else:
                raise APIError(f"Ollama API调用失败: {e.response.status_code}")
        except Exception as e:
            raise APIError(f"Ollama连接失败: {str(e)}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        url = f"{self.config.base_url}/api/generate"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            raise APIError(f"Ollama流式生成失败: {str(e)}")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            url = f"{self.config.base_url}/api/tags"
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # 检查模型是否存在
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                return self.config.model_name in model_names
                
        except Exception as e:
            self.logger.warning(f"Ollama健康检查失败: {str(e)}")
            return False


class ZhipuAPIClient(BaseAPIClient):
    """智谱AI API客户端"""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise AuthenticationError("智谱AI API Key未配置")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("智谱AI API Key无效")
            elif e.response.status_code == 429:
                raise RateLimitError("智谱AI API频率限制")
            else:
                raise APIError(f"智谱AI API调用失败: {e.response.status_code}")
        except Exception as e:
            raise APIError(f"智谱AI连接失败: {str(e)}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip() and line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            raise APIError(f"智谱AI流式生成失败: {str(e)}")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 发送简单测试请求
            await self.generate("测试", max_tokens=10)
            return True
        except Exception as e:
            self.logger.warning(f"智谱AI健康检查失败: {str(e)}")
            return False


class SiliconAPIClient(BaseAPIClient):
    """硅基流动API客户端"""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        if not self.config.api_key:
            raise AuthenticationError("硅基流动 API Key未配置")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("硅基流动 API Key无效")
            elif e.response.status_code == 429:
                raise RateLimitError("硅基流动 API频率限制")
            else:
                raise APIError(f"硅基流动 API调用失败: {e.response.status_code}")
        except Exception as e:
            raise APIError(f"硅基流动连接失败: {str(e)}")
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        url = f"{self.config.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip() and line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            raise APIError(f"硅基流动流式生成失败: {str(e)}")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 发送简单测试请求
            await self.generate("测试", max_tokens=10)
            return True
        except Exception as e:
            self.logger.warning(f"硅基流动健康检查失败: {str(e)}")
            return False


class APIManager:
    """API管理器 - 统一调用和自动切换"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化API管理器
        
        Args:
            config: API配置参数
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # 主要提供商和备选顺序
        self.primary_provider = APIProvider(config.get('provider', 'local'))
        self.fallback_order = [APIProvider(p) for p in config.get('fallback_order', [])]
        
        # 初始化API客户端
        self.clients = self._init_clients()
        
        # 当前激活的客户端
        self.active_client = None
        
        self.logger.info(f"API管理器初始化完成, 主要提供商: {self.primary_provider.value}")
    
    def _init_clients(self) -> Dict[APIProvider, BaseAPIClient]:
        """初始化所有API客户端"""
        clients = {}
        
        # 环境变量处理
        def resolve_env_var(value: str) -> str:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, "")
            return value
        
        # 本地模型客户端
        if 'local' in self.config:
            local_config = self.config['local']
            config = APIConfig(
                provider=APIProvider.LOCAL,
                model_name=local_config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')
            )
            clients[APIProvider.LOCAL] = LocalAPIClient(config)
        
        # Ollama客户端
        if 'ollama' in self.config:
            ollama_config = self.config['ollama']
            config = APIConfig(
                provider=APIProvider.OLLAMA,
                model_name=ollama_config.get('model', 'qwen2.5:7b'),
                base_url=ollama_config.get('base_url', 'http://localhost:11434'),
                timeout=ollama_config.get('timeout', 30)
            )
            clients[APIProvider.OLLAMA] = OllamaAPIClient(config)
        
        # 智谱AI客户端
        if 'zhipu' in self.config:
            zhipu_config = self.config['zhipu']
            api_key = resolve_env_var(zhipu_config.get('api_key', ''))
            if api_key:
                config = APIConfig(
                    provider=APIProvider.ZHIPU,
                    model_name=zhipu_config.get('model', 'glm-4'),
                    base_url=zhipu_config.get('base_url', 'https://open.bigmodel.cn/api/paas/v4'),
                    api_key=api_key
                )
                clients[APIProvider.ZHIPU] = ZhipuAPIClient(config)
        
        # 硅基流动客户端
        if 'silicon' in self.config:
            silicon_config = self.config['silicon']
            api_key = resolve_env_var(silicon_config.get('api_key', ''))
            if api_key:
                config = APIConfig(
                    provider=APIProvider.SILICON,
                    model_name=silicon_config.get('model', 'Qwen/Qwen2.5-7B-Instruct'),
                    base_url=silicon_config.get('base_url', 'https://api.siliconflow.cn/v1'),
                    api_key=api_key
                )
                clients[APIProvider.SILICON] = SiliconAPIClient(config)
        
        return clients
    
    async def _get_working_client(self) -> BaseAPIClient:
        """获取可用的API客户端"""
        if self.active_client is not None:
            return self.active_client
        
        # 优先尝试主要提供商
        if self.primary_provider in self.clients:
            client = self.clients[self.primary_provider]
            if await client.health_check():
                self.active_client = client
                self.logger.info(f"使用主要API提供商: {self.primary_provider.value}")
                return client
        
        # 尝试备选提供商
        for provider in self.fallback_order:
            if provider in self.clients:
                client = self.clients[provider]
                if await client.health_check():
                    self.active_client = client
                    self.logger.info(f"切换到备选API提供商: {provider.value}")
                    return client
        
        # 如果没有可用的客户端，抛出异常
        raise APIError("没有可用的API提供商")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        client = await self._get_working_client()
        try:
            return await client.generate(prompt, **kwargs)
        except Exception as e:
            # 如果当前客户端失败，尝试其他客户端
            self.active_client = None
            self.logger.warning(f"API调用失败，尝试切换提供商: {str(e)}")
            
            # 重试一次
            client = await self._get_working_client()
            return await client.generate(prompt, **kwargs)
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        client = await self._get_working_client()
        try:
            async for chunk in client.generate_stream(prompt, **kwargs):
                yield chunk
        except Exception as e:
            # 流式生成失败时，回退到普通生成
            self.logger.warning(f"流式生成失败，回退到普通生成: {str(e)}")
            response = await self.generate(prompt, **kwargs)
            
            # 模拟流式输出
            words = response.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                await asyncio.sleep(0.05)
    
    def get_current_provider(self) -> Optional[str]:
        """获取当前提供商"""
        if self.active_client:
            return self.active_client.config.provider.value
        return None
    
    def get_available_providers(self) -> List[str]:
        """获取可用提供商列表"""
        return [provider.value for provider in self.clients.keys()]
    
    async def health_check_all(self) -> Dict[str, bool]:
        """检查所有提供商的健康状态"""
        results = {}
        
        for provider, client in self.clients.items():
            try:
                is_healthy = await client.health_check()
                results[provider.value] = is_healthy
            except Exception as e:
                self.logger.warning(f"健康检查失败 {provider.value}: {str(e)}")
                results[provider.value] = False
        
        return results


# 工厂函数
def create_api_manager(config: Dict[str, Any]) -> APIManager:
    """创建API管理器"""
    return APIManager(config)


async def test_api_providers(config: Dict[str, Any]) -> Dict[str, Any]:
    """测试所有API提供商"""
    manager = APIManager(config)
    
    test_results = {
        'available_providers': manager.get_available_providers(),
        'health_checks': await manager.health_check_all(),
        'test_generation': None
    }
    
    try:
        # 测试生成
        response = await manager.generate("你好，请简单介绍一下你自己。", max_tokens=100)
        test_results['test_generation'] = {
            'success': True,
            'provider': manager.get_current_provider(),
            'response_length': len(response)
        }
    except Exception as e:
        test_results['test_generation'] = {
            'success': False,
            'error': str(e)
        }
    
    return test_results