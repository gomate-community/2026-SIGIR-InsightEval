"""
配置文件
管理系统的各种配置参数
"""

import os
from typing import Optional

class Config:
    """系统配置类"""
    
    # Milvus配置
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    MILVUS_TOKEN: Optional[str] = os.getenv("MILVUS_TOKEN", None)
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ir_papers")
    
    # 嵌入模型配置
    XINFERENCE_URL:str =os.getenv("XINFERENCE_URL","http://localhost:9997/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    # OpenAI配置 (用于嵌入服务)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "api-key")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "http://localhost:9997/v1")
    
    # LLM配置 (用于聊天服务)
    # 默认使用 OneAINexus (Qwen3-32B)
    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL", 
        "https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/qwen3-32b-server/v1"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-32B")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "dummy_key")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "600"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Ollama配置 (备用本地配置)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    
    # arXiv配置
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "100"))
    
    # API配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 数据入库配置
    INGESTION_SCHEDULE: str = os.getenv("INGESTION_SCHEDULE", "daily")  # daily, weekly, manual
    
    @classmethod
    def get_milvus_config(cls) -> dict:
        """获取Milvus配置"""
        config = {
            "uri": cls.MILVUS_URI,
            "collection_name": cls.COLLECTION_NAME
        }
        if cls.MILVUS_TOKEN:
            config["token"] = cls.MILVUS_TOKEN
        return config
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """获取LLM配置（默认OneAINexus）"""
        return {
            "base_url": cls.LLM_BASE_URL,
            "model_name": cls.LLM_MODEL,
            "api_key": cls.LLM_API_KEY,
            "timeout": cls.LLM_TIMEOUT,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "temperature": cls.LLM_TEMPERATURE
        }
    
    @classmethod
    def get_ollama_config(cls) -> dict:
        """获取Ollama配置（备用本地配置）"""
        return {
            "base_url": cls.OLLAMA_BASE_URL,
            "model_name": cls.OLLAMA_MODEL,
            "api_key": "ollama",
            "timeout": cls.OLLAMA_TIMEOUT,
            "max_tokens": cls.OLLAMA_MAX_TOKENS,
            "temperature": cls.OLLAMA_TEMPERATURE
        }
    
    @classmethod
    def get_embedding_config(cls) -> dict:
        """获取嵌入模型配置"""
        return {
            "model_name": cls.EMBEDDING_MODEL,
            "dim": cls.EMBEDDING_DIM
        }

# API配置
API_V1_STR = "/api"
PROJECT_NAME = "IR-Trends API"

# CORS配置
BACKEND_CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]

# 数据库配置 (暂时不用)
DATABASE_URL = "sqlite:///./ir_trends.db"

# Redis配置 (暂时不用)
REDIS_URL = "redis://localhost:6379"

# 开发模式
DEBUG = True