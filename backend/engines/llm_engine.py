import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from loguru import logger


class LLMConfig(BaseModel):
    """LLM 配置类"""
    base_url: str = Field(
        default="https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/qwen3-32b-server/v1", 
        description="LLM API 基础 URL"
    )
    model_name: str = Field(default="Qwen/Qwen3-32B", description="模型名称")
    api_key: str = Field(default="dummy_key", description="API 密钥")
    timeout: int = Field(default=600, description="请求超时时间（秒）")
    max_tokens: int = Field(default=4096, description="最大生成 token 数")
    temperature: float = Field(default=0.7, description="生成温度")


class OllamaConfig(LLMConfig):
    """Ollama 配置类（向后兼容）"""
    base_url: str = Field(default="http://localhost:11434/v1", description="Ollama API 基础 URL")
    model_name: str = Field(default="qwen3:0.6b", description="模型名称")
    api_key: str = Field(default="ollama", description="API 密钥（Ollama 不需要真实密钥）")


class OneAINexusConfig(LLMConfig):
    """OneAINexus 配置类"""
    base_url: str = Field(
        default="https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/qwen3-32b-server/v1",
        description="OneAINexus API 基础 URL"
    )
    model_name: str = Field(default="Qwen/Qwen3-32B", description="模型名称")
    api_key: str = Field(default="dummy_key", description="API 密钥")


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色：system, user, assistant")
    content: str = Field(..., description="消息内容")


class LLMEngine:
    """通用 LLM 调用引擎"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 引擎
        
        Args:
            config: LLM 配置，如果为 None 则使用默认 OneAINexus 配置
        """
        import httpx
        
        self.config = config or OneAINexusConfig()
        
        # 创建自定义 HTTP 客户端（禁用 SSL 验证以避免证书问题）
        http_client = httpx.AsyncClient(verify=False)
        
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            http_client=http_client
        )
        logger.info(f"LLM 引擎初始化完成，模型: {self.config.model_name}, URL: {self.config.base_url}")
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        聊天补全
        
        Args:
            messages: 聊天消息列表
            temperature: 生成温度，覆盖默认配置
            max_tokens: 最大 token 数，覆盖默认配置
            stream: 是否流式输出
            
        Returns:
            聊天补全结果
        """
        try:
            # 转换消息格式
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=formatted_messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=stream,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            if stream:
                return response
            else:
                return {
                    "content": response.choices[0].message.content,
                    "role": response.choices[0].message.role,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"聊天补全失败: {str(e)}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天补全
        
        Args:
            messages: 聊天消息列表
            temperature: 生成温度
            max_tokens: 最大 token 数
            
        Yields:
            流式生成的文本片段
        """
        try:
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=formatted_messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"流式聊天补全失败: {str(e)}")
            raise
    
    async def simple_chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        简单聊天接口
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            
        Returns:
            模型回复
        """
        messages = []
        
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        
        messages.append(ChatMessage(role="user", content=prompt))
        
        result = await self.chat_completion(messages)
        return result["content"]
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        生成文本嵌入（如果模型支持）
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        try:
            response = await self.client.embeddings.create(
                model=self.config.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"嵌入生成失败，可能模型不支持嵌入功能: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            是否连接正常
        """
        try:
            # 发送一个简单的测试请求
            test_message = [ChatMessage(role="user", content="Hello")]
            await self.chat_completion(test_message, max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return False
    
    async def list_models(self) -> List[str]:
        """
        列出可用模型
        
        Returns:
            模型列表
        """
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            return []
    
    def update_config(self, **kwargs) -> None:
        """
        更新配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 重新初始化客户端
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        logger.info(f"配置已更新: {kwargs}")


# 向后兼容：OllamaEngine 别名
OllamaEngine = LLMEngine


# 全局实例（可选）
_default_engine: Optional[LLMEngine] = None


def get_default_engine() -> LLMEngine:
    """获取默认的 LLM 引擎实例（默认使用 OneAINexus 配置）"""
    global _default_engine
    if _default_engine is None:
        _default_engine = LLMEngine()
    return _default_engine


async def quick_chat(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    快速聊天接口
    
    Args:
        prompt: 用户输入
        system_prompt: 系统提示词
        
    Returns:
        模型回复
    """
    engine = get_default_engine()
    return await engine.simple_chat(prompt, system_prompt)


# 示例使用
if __name__ == "__main__":
    async def main():
        # 示例1: 使用默认 OneAINexus 配置
        print("=" * 80)
        print("示例1: 使用 OneAINexus 配置（默认）")
        print("=" * 80)
        config = OneAINexusConfig()
        engine = LLMEngine(config)
        
        # 健康检查
        if await engine.health_check():
            logger.info("✅ OneAINexus 连接正常")
            
            # 简单聊天
            response = await engine.simple_chat(
                prompt="你好，请用一句话介绍你自己",
                system_prompt="你是一个有用的AI助手"
            )
            logger.info(f"模型回复: {response}")


            # 流式聊天
            logger.info("\n流式输出:")
            messages = [
                ChatMessage(role="system", content="你是一个有用的AI助手"),
                ChatMessage(role="user", content="请写一首关于春天的短诗")
            ]
            async for chunk in engine.chat_completion_stream(messages):
                print(chunk, end="", flush=True)
            print("\n")

        else:
            logger.error("❌ OneAINexus 连接失败")
        
        # 示例2: 使用 Ollama 本地配置
        print("\n" + "=" * 80)
        print("示例2: 使用 Ollama 本地配置")
        print("=" * 80)
        ollama_config = OllamaConfig(model_name="qwen3:0.6b")
        ollama_engine = LLMEngine(ollama_config)
        
        if await ollama_engine.health_check():
            logger.info("✅ Ollama 连接正常")
            
            # 流式聊天
            logger.info("\n流式输出:")
            messages = [
                ChatMessage(role="system", content="你是一个有用的AI助手"),
                ChatMessage(role="user", content="请写一首关于春天的短诗")
            ]
            
            async for chunk in ollama_engine.chat_completion_stream(messages):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            logger.error("❌ Ollama 连接失败，跳过示例")
        
        # 示例3: 自定义配置
        print("\n" + "=" * 80)
        print("示例3: 自定义配置")
        print("=" * 80)
        custom_config = LLMConfig(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4",
            api_key="your-api-key-here"
        )
        # custom_engine = LLMEngine(custom_config)
        logger.info(f"自定义配置创建成功: {custom_config.model_name}")
    
    # 运行示例
    asyncio.run(main())

