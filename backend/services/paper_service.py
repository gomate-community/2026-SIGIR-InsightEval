"""Paper service for managing paper data in Milvus"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from loguru import logger
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

from backend.services.arxiv_service import ArxivService
from backend.services.base_service import BaseMilvusService


class PaperService(BaseMilvusService):
    """论文服务，用于存储和检索IR论文数据"""

    def __init__(
            self,
            uri: str = "http://localhost:19530",
            token: Optional[str] = None,
            collection_name: str = "ir_papers",
            embedding_model: str = "bge-m3",
            openai_api_key: str = "api-key",
            openai_base_url: str = "http://localhost:9997/v1",
            max_results: int = 100
    ):
        """
        初始化论文服务
        
        Args:
            uri: Milvus服务地址
            token: 认证token
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI API基础URL
            max_results: 每次查询的最大结果数
        """
        super().__init__(
            uri=uri,
            token=token,
            collection_name=collection_name,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url
        )
        self.arxiv_service = ArxivService(max_results=max_results)

    def _create_collection_schema(self):
        """创建论文集合schema"""
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
        schema.add_field(field_name="affiliation_detail", datatype=DataType.JSON, nullable=True, max_length=125536)

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
        bm25_function = self._create_bm25_function("full_text", "sparse_bm25")
        schema.add_function(bm25_function)

        return schema

    def _create_index_params(self):
        """创建论文集合索引参数"""
        if not self.client:
            return None

        index_params = self.client.prepare_index_params()

        # 密集向量索引
        self._create_dense_index(index_params, "dense_vector", "dense_index")

        # BM25稀疏向量索引
        self._create_sparse_index(index_params, "sparse_bm25", "sparse_bm25_index")

        return index_params

    def _prepare_paper_data(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备论文数据用于插入Milvus"""
        if not self.openai_client:
            logger.warning("OpenAI client not available, cannot generate embeddings")
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
                        logger.warning(f"Paper {paper.get('arxiv_id', 'unknown')} has no full_text, skipping")
                        continue
                    paper["full_text"] = full_text

                # 生成密集向量嵌入
                dense_vector = self._emb_text(full_text)
                if not dense_vector:
                    logger.warning(
                        f"Failed to generate embedding for paper {paper.get('arxiv_id', 'unknown')}, skipping")
                    continue

                paper["dense_vector"] = dense_vector
                data.append(paper)

            except Exception as e:
                logger.error(f"Error preparing paper data for {paper.get('arxiv_id', 'unknown')}: {e}")
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
            logger.error(f"Error checking existing papers: {e}")
            return set()

    async def insert_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        插入论文数据到Milvus
        
        Returns:
            实际插入的论文数量
        """
        if not self.client:
            logger.warning("Milvus not available, skipping insert")
            return 0

        try:
            data = self._prepare_paper_data(papers)
            if not data:
                logger.warning("No data to insert")
                return 0
            # 检查重复数据
            existing_ids = await self._get_existing_paper_ids([d["arxiv_id"] for d in data])
            new_data = [d for d in data if d["arxiv_id"] not in existing_ids]
            if not new_data:
                logger.info("All papers already exist in database")
                return 0

            result = self.client.insert(
                collection_name=self.collection_name,
                data=new_data
            )
            inserted_count = len(new_data)
            logger.info(f"Inserted {inserted_count} papers into Milvus")
            return inserted_count

        except Exception as e:
            logger.error(f"Error inserting papers: {e}")
            return 0

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
            logger.warning("Milvus not available, returning empty results")
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
                "alphaxiv_overview",
                "affiliation_detail"
            ]

            # 执行混合搜索
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, bm25_request],
                ranker=RRFRanker(100),
                limit=limit,
                output_fields=output_fields
            )
            # logger.info(results[0])
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
                        "hits": entity.get("hits", {}),
                        "score_detail": entity.get("score_detail", {}),
                        "category_detail": entity.get("category_detail", {}),
                        "alphaxiv_detail": entity.get("alphaxiv_detail", {}),
                        "alphaxiv_overview": entity.get("alphaxiv_overview", {}),
                        "affiliation_detail": entity.get("affiliation_detail", {}),
                    }

                    processed_results.append(result)

            # 按 distance 从小到大排序
            processed_results.sort(key=lambda x: x["distance"], reverse=True)

            # 取前 30% 的结果
            if processed_results:
                top_count = max(1, int(len(processed_results) * 0.3))
                processed_results = processed_results[:top_count]
                logger.info(f"Filtered top {top_count} results from {len(results)} total results (30%)")
            # logger.info(results[0])
            return processed_results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
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
        if not self.client:
            logger.warning("Milvus not available, returning empty results")
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
                    "alphaxiv_overview",
                    "affiliation_detail"
                ]

            logger.info(f"Querying collection: {self.collection_name}")
            logger.info(f"Filter expression: {filter_expr}")
            logger.info(f"Output fields: {output_fields}")

            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
            )
            logger.info(f"Query returned {len(results)} results")
            if results:
                logger.debug(f"First result: {results[0]}")
            else:
                logger.warning("No results returned from query")
            processed_results = []
            for result in results:
                processed_result = {
                    "entity_id": result.get("id", ""),
                    "arxiv_id": result.get("arxiv_id", ""),
                    "title": result.get("title", ""),
                    "abstract": result.get("abstract", ""),
                    "metadata": result.get("metadata", {}),
                    "hits": result.get("hits", {}),
                    "score_detail": result.get("score_detail", {}),
                    "category_detail": result.get("category_detail", {}),
                    "alphaxiv_detail": result.get("alphaxiv_detail", {}),
                    "alphaxiv_overview": result.get("alphaxiv_overview", {}),
                    "affiliation_detail": result.get("affiliation_detail", {}),
                }
                processed_results.append(processed_result)
            processed_results.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
            return processed_results

        except Exception as e:
            logger.error(f"Error in metadata query: {e}")
            logger.error(f"Collection name: {self.collection_name}")
            logger.error(f"Filter expression: {filter_expr}")

            # 检查集合是否存在
            try:
                collections = self.client.list_collections()
                logger.error(f"Available collections: {collections}")
                if self.collection_name not in collections:
                    logger.error(f"Collection '{self.collection_name}' does not exist!")
            except Exception as list_error:
                logger.error(f"Failed to list collections: {list_error}")

            return []

    async def get_paper_detail(
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
            logger.warning("Milvus not available, returning None")
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
                    "alphaxiv_overview",
                    "affiliation_detail"
                ]

            logger.info(f"Getting entity by id: {entity_id}")
            logger.info(f"Output fields: {output_fields}")

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
                    "affiliation_detail": result.get("affiliation_detail", {}),
                }
                logger.info(f"Found entity with id: {entity_id}")
                return processed_result
            else:
                logger.warning(f"Entity with id {entity_id} not found")
                return None

        except Exception as e:
            logger.error(f"Error getting entity by id {entity_id}: {e}")
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
