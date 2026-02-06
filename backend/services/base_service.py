"""Base Milvus service with common functionality"""
from typing import List, Optional
from abc import ABC, abstractmethod
from loguru import logger
import logging
from pymilvus import MilvusClient, DataType, Function, FunctionType
from openai import OpenAI


class BaseMilvusService(ABC):
    """Milvus服务基类，提供通用的Milvus操作功能"""
    
    def __init__(
        self, 
        uri: str = "http://localhost:19530",
        token: Optional[str] = None,
        collection_name: str = "ir_papers",
        embedding_model: str = "bge-m3",
        openai_api_key: str = "api-key",
        openai_base_url: str = "http://localhost:9997/v1",
    ):
        """
        初始化Milvus服务基类
        
        Args:
            uri: Milvus服务地址
            token: 认证token
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI API基础URL
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.dense_dim = 1024  # bge-m3的维度

        self.client = None
        self.openai_client = None
        self._initialize_client()
        self._initialize_openai_client()

    def _initialize_client(self):
        """初始化Milvus客户端"""
        try:
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
            else:
                self.client = MilvusClient(uri=self.uri)
            logger.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.client = None
    
    def _initialize_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            self.openai_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )
            logger.info(f"Initialized OpenAI client with model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    def _emb_text(self, text: str) -> List[float]:
        """生成文本嵌入"""
        if not self.openai_client:
            return []
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    @abstractmethod
    def _create_collection_schema(self):
        """创建集合schema（子类必须实现）"""
        pass
    
    @abstractmethod
    def _create_index_params(self):
        """创建索引参数（子类必须实现）"""
        pass
    
    async def initialize_collection(self, drop_existing: bool = False) -> bool:
        """
        初始化集合
        
        Args:
            drop_existing: 是否删除已存在的集合
            
        Returns:
            是否成功初始化
        """
        if not self.client:
            logger.warning("Milvus not available, skipping collection initialization")
            return False
        
        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                if drop_existing:
                    self.client.drop_collection(self.collection_name)
                    logger.info(f"Dropped existing collection: {self.collection_name}")
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return True
            
            # 创建schema和索引
            schema = self._create_collection_schema()
            index_params = self._create_index_params()
            
            if schema and index_params:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params
                )
                logger.info(f"Created collection: {self.collection_name}")
                return True
            else:
                logger.error("Failed to create schema or index params")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            return False
    
    async def get_collection_stats(self) -> Optional[dict]:
        """获取集合统计信息"""
        if not self.client:
            return None
        
        try:
            if not self.client.has_collection(self.collection_name):
                return {"error": "Collection does not exist"}
            
            # 获取集合信息
            collection_info = self.client.describe_collection(self.collection_name)
            # 获取实体数量
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "fields": len(collection_info.get("fields", [])),
                "row_count": stats.get("row_count", 0),
                "data_size": stats.get("data_size", 0),
                "collection_info": stats
            }
            
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def _create_bm25_function(self, input_field: str = "full_text", output_field: str = "sparse_bm25") -> Function:
        """
        创建BM25函数（通用方法）
        
        Args:
            input_field: 输入字段名
            output_field: 输出字段名
            
        Returns:
            BM25函数对象
        """
        return Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=[input_field],
            output_field_names=output_field,
        )
    
    def _create_dense_index(self, index_params, field_name: str = "dense_vector", index_name: str = "dense_index"):
        """
        创建密集向量索引（通用方法）
        
        Args:
            index_params: 索引参数对象
            field_name: 字段名
            index_name: 索引名
        """
        index_params.add_index(
            field_name=field_name,
            index_name=index_name,
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
    
    def _create_sparse_index(self, index_params, field_name: str = "sparse_bm25", index_name: str = "sparse_bm25_index"):
        """
        创建BM25稀疏向量索引（通用方法）
        
        Args:
            index_params: 索引参数对象
            field_name: 字段名
            index_name: 索引名
        """
        index_params.add_index(
            field_name=field_name,
            index_name=index_name,
            index_type="SPARSE_WAND",
            metric_type="BM25"
        )

