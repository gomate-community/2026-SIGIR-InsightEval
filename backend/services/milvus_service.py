
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger
import logging
import json
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from openai import OpenAI
from backend.services.arxiv_service import ArxivService

class MilvusService:
    """Milvus向量数据库服务，用于存储和检索IR论文数据"""
    def __init__(
        self, 
        uri: str = "http://localhost:19530",
        token: Optional[str] = None,
        collection_name: str = "ir_papers",
        embedding_model: str = "bge-m3",
        openai_api_key: str = "api-key",
        embedding_base_url: str = "http://localhost:9997/v1",
        max_results: int = 100
    ):
        """
        初始化Milvus服务
        
        Args:
            uri: Milvus服务地址
            token: 认证token
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            openai_api_key: OpenAI API密钥
            embedding_base_url: OpenAI API基础URL
            max_results: 每次查询的最大结果数
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.openai_api_key = openai_api_key
        self.embedding_base_url = embedding_base_url
        self.dense_dim = 1024  # bge-m3的维度

        self.client = None
        self.openai_client = None
        self.arxiv_service = ArxivService(max_results=max_results)
        self._initialize_client()
        self._initialize_openai_client()

    def _initialize_client(self):
        """
        初始化Milvus客户端
        """
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
                base_url=self.embedding_base_url
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
    
    def _create_collection_schema(self):
        """创建集合schema"""
        if not self.client:
            return None
            
        schema = MilvusClient.create_schema(enable_dynamic_field=True)
        analyzer_params = {"type": "english"}
        
        # 核心字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="arxiv_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=10000)
        
        # JSON 元数据字段，存储所有其他字段
        schema.add_field(field_name="metadata", datatype=DataType.JSON, nullable=True, max_length=125536)
        schema.add_field(field_name="hits", datatype=DataType.JSON, nullable=True, max_length=125536)
        schema.add_field(field_name="alphaxiv_detail", datatype=DataType.JSON, nullable=True, max_length=125536)
        schema.add_field(field_name="alphaxiv_overview", datatype=DataType.JSON, nullable=True, max_length=125536)

        # 全文检索字段
        schema.add_field(
            field_name="full_text", 
            datatype=DataType.VARCHAR, 
            max_length=15000,
            enable_analyzer=True,
            analyzer_params=analyzer_params,
            enable_match=True
        )
        
        # 向量字段
        schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
        
        # BM25函数
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["full_text"],
            output_field_names="sparse_bm25",
        )
        schema.add_function(bm25_function)
        
        return schema
    
    def _create_index_params(self):
        """创建索引参数"""

        index_params = self.client.prepare_index_params()
        
        # 密集向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        
        # BM25稀疏向量索引
        index_params.add_index(
            field_name="sparse_bm25",
            index_name="sparse_bm25_index",
            index_type="SPARSE_WAND",
            metric_type="BM25"
        )
        
        return index_params
    
    async def initialize_collection(self, drop_existing: bool = False):
        """初始化集合"""
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
    
    def _prepare_paper_data(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备论文数据用于插入Milvus"""
        if not self.openai_client:
            logging.warning("OpenAI client not available, cannot generate embeddings")
            return []
        data = []
        for paper in papers:
            try:
                full_text = paper.get("full_text")
                if not full_text:
                    # 如果没有full_text，尝试从title和abstract构建
                    title = paper.get("title", "")
                    abstract = paper.get("abstract", "")
                    full_text = f"{title} {abstract}".strip()
                    if not full_text:
                        logging.warning(f"Paper {paper.get('arxiv_id', 'unknown')} has no full_text, skipping")
                        continue
                    paper["full_text"] = full_text
                
                # 生成密集向量嵌入
                dense_vector = self._emb_text(full_text)
                if not dense_vector:
                    logging.warning(f"Failed to generate embedding for paper {paper.get('arxiv_id', 'unknown')}, skipping")
                    continue
                
                paper["dense_vector"] = dense_vector
                data.append(paper)
                
            except Exception as e:
                logging.error(f"Error preparing paper data for {paper.get('arxiv_id', 'unknown')}: {e}")
                continue
        return data
    
    async def _get_existing_paper_ids(self, paper_ids: List[str]) -> set:
        """获取已存在的arxiv_id"""
        if not self.client or not paper_ids:
            return set()
        
        try:
            ids_str = ", ".join([f'"{pid}"' for pid in paper_ids])
            filter_expr = f"arxiv_id in [{ids_str}]"
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["arxiv_id"]
            )
            return {result["arxiv_id"] for result in results}
            
        except Exception as e:
            logging.error(f"Error checking existing papers: {e}")
            return set()
    
    async def insert_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        插入论文数据到Milvus
        
        Returns:
            实际插入的论文数量
        """
        if not self.client:
            logging.warning("Milvus not available, skipping insert")
            return 0
        
        try:
            data = self._prepare_paper_data(papers)
            if not data:
                logging.warning("No data to insert")
                return 0
            # 检查重复数据
            existing_ids = await self._get_existing_paper_ids([d["arxiv_id"] for d in data])
            new_data = [d for d in data if d["arxiv_id"] not in existing_ids]
            if not new_data:
                logging.info("All papers already exist in database")
                return 0
            
            result = self.client.insert(
                collection_name=self.collection_name,
                data=new_data
            )
            inserted_count = len(new_data)
            logging.info(f"Inserted {inserted_count} papers into Milvus")
            return inserted_count
            
        except Exception as e:
            logging.error(f"Error inserting papers: {e}")
            return 0

    async def get_collection_stats(self) -> Optional[Dict[str, Any]]:
        """获取集合统计信息"""
        if  not self.client:
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
                "collection_info":stats
            }
            
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 20,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        混合检索（密集向量 + BM25）
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            filter_expr: 过滤表达式，例如: 'metadata["category"] == "IR"'
            
        Returns:
            查询结果列表
        """
        if not self.client or not self.openai_client:
            logging.warning("Milvus not available, returning empty results")
            return []
        
        try:
            query_vector = self._emb_text(query)
            
            # 密集向量搜索请求
            dense_request = AnnSearchRequest(
                [query_vector], 
                "dense_vector", 
                {"metric_type": "IP", "params": {"nprobe": 10}}, 
                limit=limit,
                expr=filter_expr
            )
            
            # BM25搜索请求
            bm25_request = AnnSearchRequest(
                [query], 
                "sparse_bm25", 
                {"metric_type": "BM25"}, 
                limit=limit,
                expr=filter_expr
            )
            # 执行混合搜索
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, bm25_request],
                ranker=RRFRanker(100),
                limit=limit,
            )
            # 处理结果
            processed_results = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    result = {
                        "entity_id": entity.get("id", ""),
                        "arxiv_id": entity.get("arxiv_id", ""),
                        "distance": hit.get("distance", 0),
                        "title": entity.get("title", ""),
                        "abstract": entity.get("abstract", ""),
                        "metadata": entity.get("metadata", {}),
                        "hits":  entity.get("hits", {}),
                        "score_detail": entity.get("score_detail", {}),
                        "category_detail": entity.get("category_detail", {}),
                        "alphaxiv_detail": entity.get("alphaxiv_detail", {}),
                        "alphaxiv_overview": entity.get("alphaxiv_overview", {}),
                    }
                    
                    processed_results.append(result)
            
            # 按 distance 从小到大排序
            processed_results.sort(key=lambda x: x["distance"],reverse=True)
            
            # 取前 30% 的结果
            if processed_results:
                top_count = max(1, int(len(processed_results) * 0.3))
                processed_results = processed_results[:top_count]
                logging.info(f"Filtered top {top_count} results from {len(results)} total results (30%)")
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Error in hybrid search: {e}")
            return []
    
    async def query_by_metadata(
        self,
        filter_expr: str,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        基于 metadata 字段进行查询
        
        Args:
            filter_expr: 过滤表达式，例如: 'metadata["primary_category"] == "cs.IR"'
            output_fields: 指定输出字段，默认返回所有核心字段（不包括dense_vector和sparse_bm25）
            
        Returns:
            查询结果列表
        """
        if  not self.client:
            logging.warning("Milvus not available, returning empty results")
            return []
        
        try:
            # 如果未指定output_fields，使用默认字段（排除向量字段）
            if output_fields is None:
                output_fields = [
                    "id",
                    "arxiv_id",
                    "title",
                    "abstract",
                    "metadata",
                    "hits",
                    "score_detail",
                    "category_detail",
                    "alphaxiv_detail",
                    "alphaxiv_overview"
                ]
            
            logging.info(f"Querying collection: {self.collection_name}")
            logging.info(f"Filter expression: {filter_expr}")
            logging.info(f"Output fields: {output_fields}")
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
            )
            logging.info(f"Query returned {len(results)} results")
            if results:
                logging.debug(f"First result: {results[0]}")
            else:
                logging.warning("No results returned from query")
            processed_results = []
            for result in results:
                processed_result =  {
                        "entity_id": result.get("id", ""),
                        "arxiv_id": result.get("arxiv_id", ""),
                        "title": result.get("title", ""),
                        "abstract": result.get("abstract", ""),
                        "metadata": result.get("metadata", {}),
                        "hits":  result.get("hits", {}),
                        "score_detail": result.get("score_detail", {}),
                        "category_detail": result.get("category_detail", {}),
                        "alphaxiv_detail": result.get("alphaxiv_detail", {}),
                        "alphaxiv_overview": result.get("alphaxiv_overview", {}),
                    }
                processed_results.append(processed_result)
            return processed_results
            
        except Exception as e:
            logging.error(f"Error in metadata query: {e}")
            logging.error(f"Collection name: {self.collection_name}")
            logging.error(f"Filter expression: {filter_expr}")
            
            # 检查集合是否存在
            try:
                collections = self.client.list_collections()
                logging.error(f"Available collections: {collections}")
                if self.collection_name not in collections:
                    logging.error(f"Collection '{self.collection_name}' does not exist!")
            except Exception as list_error:
                logging.error(f"Failed to list collections: {list_error}")
            
            return []
    
    async def get_by_id(
        self,
        entity_id: int,
        output_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        通过主键id获取实体
        
        Args:
            entity_id: 主键id（INT64）
            output_fields: 指定输出字段，默认返回所有核心字段（不包括dense_vector和sparse_bm25）
            
        Returns:
            查询结果，如果不存在则返回None
        """
        if not self.client:
            logging.warning("Milvus not available, returning None")
            return None
        
        try:
            # 如果未指定output_fields，使用默认字段（排除向量字段）
            if output_fields is None:
                output_fields = [
                    "id",
                    "arxiv_id",
                    "title",
                    "abstract",
                    "metadata",
                    "hits",
                    "score_detail",
                    "category_detail",
                    "alphaxiv_detail",
                    "alphaxiv_overview"
                ]
            
            logging.info(f"Getting entity by id: {entity_id}")
            logging.info(f"Output fields: {output_fields}")
            
            results = self.client.get(
                collection_name=self.collection_name,
                ids=[entity_id],
                output_fields=output_fields
            )
            
            if results and len(results) > 0:
                result = results[0]
                processed_result = {
                    "id": result.get("id"),
                    "entity_id": result.get("id"),
                    "arxiv_id": result.get("arxiv_id", ""),
                    "title": result.get("title", ""),
                    "abstract": result.get("abstract", ""),
                    "metadata": result.get("metadata", {}),
                    "hits": result.get("hits", {}),
                    "score_detail": result.get("score_detail", {}),
                    "category_detail": result.get("category_detail", {}),
                    "alphaxiv_detail": result.get("alphaxiv_detail", {}),
                    "alphaxiv_overview": result.get("alphaxiv_overview", {}),
                }
                logging.info(f"Found entity with id: {entity_id}")
                return processed_result
            else:
                logging.warning(f"Entity with id {entity_id} not found")
                return None
                
        except Exception as e:
            logging.error(f"Error getting entity by id {entity_id}: {e}")
            return None
    
    async def ingest_daily_papers(self, target_date: Optional[datetime] = None) -> int:
        """
        入库指定日期的论文
        
        Args:
            target_date: 目标日期，默认为今天
            
        Returns:
            入库的论文数量
        """
        if target_date is None:
            target_date = datetime.now()
        
        try:
            logger.info(f"Fetching papers for date: {target_date.strftime('%Y-%m-%d')}")
            
            # 使用ArxivService获取指定日期的论文
            papers = await self.arxiv_service.get_papers_by_date(target_date)
            
            if not papers:
                logger.info(f"No papers found for date: {target_date.strftime('%Y-%m-%d')}")
                return 0
            
            logger.info(f"Found {len(papers)} papers, inserting into Milvus...")
            
            # 插入论文到Milvus
            inserted_count = await self.insert_papers(papers)
            
            if inserted_count > 0:
                logger.info(f"Successfully ingested {inserted_count} papers (out of {len(papers)} found)")
                return inserted_count
            else:
                if len(papers) > 0:
                    logger.info(f"All {len(papers)} papers already exist in database")
                else:
                    logger.error("Failed to insert papers into Milvus")
                return 0
                
        except Exception as e:
            logger.error(f"Error ingesting daily papers: {e}")
            return 0
    
    # ========== 报告集合相关方法 ==========
    
    def _create_report_collection_schema(self):
        """创建报告集合schema"""
        if not self.client:
            return None
            
        schema = MilvusClient.create_schema(enable_dynamic_field=True)
        analyzer_params = {"type": "english"}
        
        # 核心字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="report_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="week", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="date_range", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="publish_date", datatype=DataType.VARCHAR, max_length=50)
        
        # JSON 字段存储完整的报告数据
        schema.add_field(field_name="report_data", datatype=DataType.JSON, nullable=True, max_length=125536)
        schema.add_field(field_name="weekly_report", datatype=DataType.JSON, nullable=True, max_length=125536)
        
        # 全文检索字段（用于生成向量）
        schema.add_field(
            field_name="full_text", 
            datatype=DataType.VARCHAR, 
            max_length=15000,
            enable_analyzer=True,
            analyzer_params=analyzer_params,
            enable_match=True
        )
        
        # 向量字段
        schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
        
        # BM25函数
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["full_text"],
            output_field_names="sparse_bm25",
        )
        schema.add_function(bm25_function)
        
        return schema
    
    def _create_report_index_params(self):
        """创建报告集合索引参数"""
        if not self.client:
            return None
        
        index_params = self.client.prepare_index_params()
        
        # 密集向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        
        # BM25稀疏向量索引
        index_params.add_index(
            field_name="sparse_bm25",
            index_name="sparse_bm25_index",
            index_type="SPARSE_WAND",
            metric_type="BM25"
        )
        
        return index_params
    
    async def initialize_report_collection(self, collection_name: str = "ir_reports", drop_existing: bool = False):
        """初始化报告集合"""
        if not self.client:
            logger.warning("Milvus not available, skipping report collection initialization")
            return False
        
        try:
            # 检查集合是否存在
            if self.client.has_collection(collection_name):
                if drop_existing:
                    self.client.drop_collection(collection_name)
                    logger.info(f"Dropped existing report collection: {collection_name}")
                else:
                    logger.info(f"Report collection {collection_name} already exists")
                    return True
            
            # 创建schema和索引
            schema = self._create_report_collection_schema()
            index_params = self._create_report_index_params()
            
            if schema and index_params:
                self.client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params
                )
                logger.info(f"Created report collection: {collection_name}")
                return True
            else:
                logger.error("Failed to create report schema or index params")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing report collection: {e}")
            return False
    
    def _prepare_report_data(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备报告数据用于插入Milvus，生成full_text和向量"""
        if not self.openai_client:
            logging.warning("OpenAI client not available, cannot generate embeddings")
            return []
        
        data = []
        for report in reports:
            try:
                # 从报告中提取文本内容构建 full_text
                full_text_parts = []
                
                # 添加基本信息
                if report.get("week"):
                    full_text_parts.append(f"Week {report['week']}")
                if report.get("date_range"):
                    full_text_parts.append(report["date_range"])
                
                # 从 weekly_report JSON 中提取文本
                weekly_report = report.get("weekly_report", {})
                if isinstance(weekly_report, dict):
                    # 提取 summary
                    summary = weekly_report.get("summary", {})
                    if isinstance(summary, dict):
                        if summary.get("zh"):
                            full_text_parts.append(summary["zh"])
                        if summary.get("en"):
                            full_text_parts.append(summary["en"])
                    
                    # 提取 highlights
                    highlights = weekly_report.get("highlights", [])
                    if isinstance(highlights, list):
                        for highlight in highlights:
                            if isinstance(highlight, dict):
                                if highlight.get("zh"):
                                    full_text_parts.append(highlight["zh"])
                                if highlight.get("en"):
                                    full_text_parts.append(highlight["en"])
                
                # 从 report_data JSON 中提取文本
                report_data = report.get("report_data", {})
                if isinstance(report_data, dict):
                    # 提取 keyInsights
                    key_insights = report_data.get("keyInsights", [])
                    if isinstance(key_insights, list):
                        for insight in key_insights:
                            if isinstance(insight, dict):
                                if insight.get("zh"):
                                    full_text_parts.append(insight["zh"])
                                if insight.get("en"):
                                    full_text_parts.append(insight["en"])
                    
                    # 提取 trendingTopics
                    trending_topics = report_data.get("trendingTopics", [])
                    if isinstance(trending_topics, list):
                        for topic in trending_topics:
                            if isinstance(topic, dict):
                                name = topic.get("name", {})
                                if isinstance(name, dict):
                                    if name.get("zh"):
                                        full_text_parts.append(name["zh"])
                                    if name.get("en"):
                                        full_text_parts.append(name["en"])
                    
                    # 提取 topPapers 的标题和摘要
                    top_papers = report_data.get("topPapers", [])
                    if isinstance(top_papers, list):
                        for paper in top_papers:
                            if isinstance(paper, dict):
                                title = paper.get("title", {})
                                if isinstance(title, dict):
                                    if title.get("zh"):
                                        full_text_parts.append(title["zh"])
                                    if title.get("en"):
                                        full_text_parts.append(title["en"])
                                
                                abstract = paper.get("abstract", {})
                                if isinstance(abstract, dict):
                                    if abstract.get("zh"):
                                        full_text_parts.append(abstract["zh"])
                                    if abstract.get("en"):
                                        full_text_parts.append(abstract["en"])
                
                # 合并所有文本
                full_text = " ".join(full_text_parts)[:300]
                if not full_text:
                    # 如果没有提取到文本，使用基本信息
                    full_text = f"Week {report.get('week', '')} {report.get('date_range', '')}"
                
                # 生成密集向量嵌入
                dense_vector = self._emb_text(full_text)
                if not dense_vector:
                    logging.warning(f"Failed to generate embedding for report {report.get('report_id', 'unknown')}, skipping")
                    continue
                
                # 添加 full_text 和 dense_vector
                report["full_text"] = full_text
                report["dense_vector"] = dense_vector
                data.append(report)
                
            except Exception as e:
                logging.error(f"Error preparing report data for {report.get('report_id', 'unknown')}: {e}")
                continue
        
        return data
    
    async def insert_reports(self, reports: List[Dict[str, Any]], collection_name: str = "ir_reports") -> int:
        """
        插入报告数据到Milvus
        
        Args:
            reports: 报告数据列表
            collection_name: 集合名称
            
        Returns:
            实际插入的报告数量
        """
        if not self.client:
            logging.warning("Milvus not available, skipping insert")
            return 0
        
        try:
            if not reports:
                logging.warning("No reports to insert")
                return 0
            
            # 准备数据（生成 full_text 和向量）
            data = self._prepare_report_data(reports)
            if not data:
                logging.warning("No valid report data to insert")
                return 0
            
            # 检查重复数据
            existing_ids = await self._get_existing_report_ids([r["report_id"] for r in data], collection_name)
            new_reports = [r for r in data if r["report_id"] not in existing_ids]
            
            if not new_reports:
                logging.info("All reports already exist in database")
                return 0
            
            result = self.client.insert(
                collection_name=collection_name,
                data=new_reports
            )
            inserted_count = len(new_reports)
            logging.info(f"Inserted {inserted_count} reports into Milvus")
            return inserted_count
            
        except Exception as e:
            logging.error(f"Error inserting reports: {e}")
            return 0
    
    async def _get_existing_report_ids(self, report_ids: List[str], collection_name: str = "ir_reports") -> set:
        """获取已存在的report_id"""
        if not self.client or not report_ids:
            return set()
        
        try:
            ids_str = ", ".join([f'"{rid}"' for rid in report_ids])
            filter_expr = f"report_id in [{ids_str}]"
            results = self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=["report_id"]
            )
            return {result["report_id"] for result in results}
            
        except Exception as e:
            logging.error(f"Error checking existing reports: {e}")
            return set()
    
    async def query_reports(
        self,
        filter_expr: Optional[str] = None,
        collection_name: str = "ir_reports",
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        查询报告数据
        
        Args:
            filter_expr: 过滤表达式，例如: 'report_id == "70"'
            collection_name: 集合名称
            output_fields: 指定输出字段，默认返回所有字段
            
        Returns:
            查询结果列表
        """
        if not self.client:
            logging.warning("Milvus not available, returning empty results")
            return []
        
        try:
            # 如果未指定output_fields，使用默认字段
            if output_fields is None:
                output_fields = [
                    "id",
                    "report_id",
                    "week",
                    "date_range",
                    "publish_date",
                    "report_data",
                    "weekly_report"
                ]
            
            logging.info(f"Querying report collection: {collection_name}")
            logging.info(f"Filter expression: {filter_expr}")
            
            # 如果没有过滤条件，查询所有记录
            if filter_expr is None:
                filter_expr = "id >= 0"
            
            results = self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
            )
            logging.info(f"Query returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in report query: {e}")
            return []
    
    async def get_report_by_id(
        self,
        report_id: str,
        collection_name: str = "ir_reports"
    ) -> Optional[Dict[str, Any]]:
        """
        通过report_id获取报告
        
        Args:
            report_id: 报告ID
            collection_name: 集合名称
            
        Returns:
            报告数据，如果不存在则返回None
        """
        if not self.client:
            logging.warning("Milvus not available, returning None")
            return None
        
        try:
            filter_expr = f'report_id == "{report_id}"'
            results = await self.query_reports(filter_expr=filter_expr, collection_name=collection_name)
            
            if results and len(results) > 0:
                return results[0]
            else:
                logging.warning(f"Report with id {report_id} not found")
                return None
                
        except Exception as e:
            logging.error(f"Error getting report by id {report_id}: {e}")
            return None