"""Report service for weekly reports using real data from Milvus"""
import json
import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter
from loguru import logger
from pymilvus import MilvusClient, DataType, Function, FunctionType

from backend.models.schemas import (
    WeeklyReport, BilingualText, TrendingTopic, ReportPaper, CategoryStatistics
)
from backend.services.paper_service import PaperService
from backend.services.base_service import BaseMilvusService
from backend.engines.llm_engine import LLMEngine, LLMConfig, OneAINexusConfig
from backend.config import Config
from backend.prompts.report_prompt import insights_prompt, keywords_prompt, title_prompt, overview_prompt, highlight_prompt, category_summary_prompt
from backend.models.taxonomy import IR_CATEGORIES_MAPPING


class ReportService(BaseMilvusService):
    """Service for handling weekly reports using real data"""
    
    def __init__(
        self, 
        llm_config: Optional[LLMConfig] = None,
        uri: str = None,
        token: Optional[str] = None,
        collection_name: str = "ir_reports",
        embedding_model: str = None,
        openai_api_key: str = None,
        openai_base_url: str = None
    ):
        """
        初始化报告服务
        
        Args:
            llm_config: LLM配置，如果为None则使用默认OneAINexus配置
            uri: Milvus服务地址，如果为None则使用Config中的值
            token: 认证token，如果为None则使用Config中的值
            collection_name: 集合名称，默认为"ir_reports"
            embedding_model: 嵌入模型名称，如果为None则使用Config中的值
            openai_api_key: OpenAI API密钥，如果为None则使用Config中的值
            openai_base_url: OpenAI API基础URL，如果为None则使用Config中的值
        """
        # 初始化基类（Milvus服务）
        super().__init__(
            uri=uri or Config.MILVUS_URI,
            token=token or Config.MILVUS_TOKEN,
            collection_name=collection_name,
            embedding_model=embedding_model or Config.EMBEDDING_MODEL,
            openai_api_key=openai_api_key or Config.OPENAI_API_KEY,
            openai_base_url=openai_base_url or Config.OPENAI_BASE_URL
        )
        
        # 用于查询论文数据的服务
        self.paper_service = PaperService(
            uri=Config.MILVUS_URI,
            token=Config.MILVUS_TOKEN,
            collection_name=Config.COLLECTION_NAME,
            embedding_model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_base_url=Config.OPENAI_BASE_URL,
            max_results=Config.ARXIV_MAX_RESULTS
        )
        
        self.llm_engine = LLMEngine(llm_config or LLMConfig(**Config.get_llm_config()))
        self._reports_cache = {}
    
    def _create_collection_schema(self):
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
        bm25_function = self._create_bm25_function("full_text", "sparse_bm25")
        schema.add_function(bm25_function)
        
        return schema
    
    def _create_index_params(self):
        """创建报告集合索引参数"""
        if not self.client:
            return None
        
        index_params = self.client.prepare_index_params()
        
        # 密集向量索引
        self._create_dense_index(index_params, "dense_vector", "dense_index")
        
        # BM25稀疏向量索引
        self._create_sparse_index(index_params, "sparse_bm25", "sparse_bm25_index")
        
        return index_params
    
    def _prepare_report_data(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备报告数据用于插入Milvus，生成full_text和向量"""
        if not self.openai_client:
            logging.warning("OpenAI client not available, cannot generate embeddings")
            return []
        
        data = []
        for report in reports:
            try:
                # 从报告中提取文本内容构建 full_text（基于 title 和 overview）
                full_text_parts = []
                
                # 添加基本信息
                if report.get("week"):
                    full_text_parts.append(f"Week {report['week']}")
                if report.get("date_range"):
                    full_text_parts.append(report["date_range"])
                
                # 从 weekly_report JSON 中提取 title 和 overview
                weekly_report = report.get("weekly_report", {})
                if isinstance(weekly_report, dict):
                    # 提取 title
                    title = weekly_report.get("title", {})
                    if isinstance(title, dict):
                        if title.get("zh"):
                            full_text_parts.append(title["zh"])
                        if title.get("en"):
                            full_text_parts.append(title["en"])
                    
                    # 提取 overview
                    overview = weekly_report.get("overview", {})
                    if isinstance(overview, dict):
                        if overview.get("zh"):
                            full_text_parts.append(overview["zh"])
                        if overview.get("en"):
                            full_text_parts.append(overview["en"])
                
                # 合并所有文本
                full_text = " ".join(full_text_parts)[:1500]
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
    
    async def _get_existing_report_ids(self, report_ids: List[str]) -> set:
        """获取已存在的report_id"""
        if not self.client or not report_ids:
            return set()
        
        try:
            ids_str = ", ".join([f'"{rid}"' for rid in report_ids])
            filter_expr = f"report_id in [{ids_str}]"
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["report_id"]
            )
            return {result["report_id"] for result in results}
            
        except Exception as e:
            logging.error(f"Error checking existing reports: {e}")
            return set()
    
    async def insert_reports(self, reports: List[Dict[str, Any]]) -> int:
        """
        插入报告数据到Milvus
        
        Args:
            reports: 报告数据列表
            
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
            existing_ids = await self._get_existing_report_ids([r["report_id"] for r in data])
            new_reports = [r for r in data if r["report_id"] not in existing_ids]
            
            if not new_reports:
                logging.info("All reports already exist in database")
                return 0
            
            result = self.client.insert(
                collection_name=self.collection_name,
                data=new_reports
            )
            inserted_count = len(new_reports)
            logging.info(f"Inserted {inserted_count} reports into Milvus")
            return inserted_count
            
        except Exception as e:
            logging.error(f"Error inserting reports: {e}")
            return 0
    
    async def query_reports(
        self,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        查询报告数据
        
        Args:
            filter_expr: 过滤表达式，例如: 'report_id == "70"'
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
                    "weekly_report"
                ]
            
            logging.info(f"Querying report collection: {self.collection_name}")
            logging.info(f"Filter expression: {filter_expr}")
            
            # 如果没有过滤条件，查询所有记录
            if filter_expr is None:
                filter_expr = "id >= 0"
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields,
            )
            logging.info(f"Query returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in report query: {e}")
            return []
    
    
    def _get_week_range(self, week_offset: int = 0) -> Tuple[datetime, datetime, str, str]:
        """
        获取指定周的开始和结束日期
        
        Args:
            week_offset: 周偏移量，0表示本周，-1表示上周，以此类推
            
        Returns:
            (start_date, end_date, week_str, date_range_str)
        """
        today = datetime.now()
        # 计算本周的开始日期（周一）
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        
        # 根据偏移量计算目标周
        target_monday = this_monday + timedelta(weeks=week_offset)
        target_sunday = target_monday + timedelta(days=6)
        
        # 格式化日期字符串
        start_str = target_monday.strftime("%Y-%m-%d")
        end_str = target_sunday.strftime("%Y-%m-%d")
        
        # 生成周标识符（例如：Issue 70）
        week_num = int((today - datetime(2025, 1, 1)).days / 7) + week_offset
        week_str = f"Issue {week_num}"
        
        # 生成日期范围字符串（中文格式：YYYY年MM月DD日 - YYYY年MM月DD日）
        date_range_str = f"{target_monday.strftime('%Y年%m月%d日')} - {target_sunday.strftime('%Y年%m月%d日')}"
        
        return target_monday, target_sunday, week_str, date_range_str
    
    async def _query_weekly_papers(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        查询指定周范围内的论文数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            论文数据列表
        """
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # 构建日期过滤表达式
            filter_expr = f'metadata["date"] >= "{start_str}" && metadata["date"] <= "{end_str}"'
            
            logger.info(f"Querying papers from {start_str} to {end_str}")
            papers = await self.paper_service.query_by_metadata(filter_expr=filter_expr)
            logger.info(f"Found {len(papers)} papers in the date range")
            
            return papers
        except Exception as e:
            logger.error(f"Error querying weekly papers: {e}")
            return []
    
    def _calculate_category_statistics(self, papers: List[Dict[str, Any]]) -> List[CategoryStatistics]:
        """
        计算分类统计（基于category_detail）
        
        Args:
            papers: 论文数据列表
            
        Returns:
            分类统计列表
        """
        category_counter = Counter()
        
        # 统计每个分类的论文数量
        for paper in papers:
            category_detail = paper.get("category_detail", {})
            if category_detail and isinstance(category_detail, dict):
                category_name = category_detail.get("category_name", "")
                if category_name:
                    category_counter[category_name] += 1
        
        total_papers = len(papers)
        if total_papers == 0:
            return []
        
        # 创建分类名称到中英文描述的映射
        category_mapping = {}
        for cat in IR_CATEGORIES_MAPPING:
            category_mapping[cat["category_name"]] = {
                "zh": cat["category_name"],  # 暂时使用英文名，后续可以添加中文翻译
                "en": cat["category_name"],
                "description": cat["category_description"]
            }
        
        # 构建分类统计列表
        category_stats = []
        for category_name, count in category_counter.most_common():
            percentage = round((count / total_papers) * 100, 2)
            
            # 从映射中获取中英文名称
            if category_name in category_mapping:
                category_zh = category_mapping[category_name]["zh"]
                category_en = category_mapping[category_name]["en"]
            else:
                # 如果不在映射中，使用原始名称
                category_zh = category_name
                category_en = category_name
            
            category_stats.append(
                CategoryStatistics(
                    categoryName=BilingualText(zh=category_zh, en=category_en),
                    count=count,
                    percentage=percentage
                )
            )
        
        return category_stats
    
    def _build_category_summary_prompt(self, category_stats: List[CategoryStatistics], language: str = "zh") -> str:
        """
        构建分类总结生成提示词
        
        Args:
            category_stats: 分类统计列表
            language: 语言代码（zh或en）
            
        Returns:
            提示词
        """
        # 构建分类统计数据文本
        stats_text_parts = []
        for stat in category_stats[:10]:  # 只取前10个分类
            if language == "zh":
                stats_text_parts.append(
                    f"{stat.categoryName.zh}: {stat.count}篇 ({stat.percentage}%)"
                )
            else:
                stats_text_parts.append(
                    f"{stat.categoryName.en}: {stat.count} papers ({stat.percentage}%)"
                )
        
        stats_text = "\n".join(stats_text_parts)
        prompt = category_summary_prompt[language].format(category_stats=stats_text)
        return prompt
    
    async def _generate_category_summary(self, category_stats: List[CategoryStatistics]) -> BilingualText:
        """
        生成分类总结（中英文）
        
        Args:
            category_stats: 分类统计列表
            
        Returns:
            分类总结（中英文）
        """
        try:
            # 生成中文总结
            zh_prompt = self._build_category_summary_prompt(category_stats, language="zh")
            zh_response = await self.llm_engine.simple_chat(
                prompt=zh_prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确总结论文分类分布特点。"
            )
            
            # 生成英文总结
            en_prompt = self._build_category_summary_prompt(category_stats, language="en")
            en_response = await self.llm_engine.simple_chat(
                prompt=en_prompt,
                system_prompt="You are a professional expert in the information retrieval field. Please accurately summarize paper category distribution characteristics."
            )
            
            # 解析响应
            zh_summary = self._parse_json_response(zh_response, "summary")
            en_summary = self._parse_json_response(en_response, "summary")
            
            # 如果解析失败，使用默认值
            if not zh_summary or len(zh_summary) == 0:
                top_category = category_stats[0] if category_stats else None
                if top_category:
                    zh_summary_text = f"本周论文主要集中在{top_category.categoryName.zh}领域，占比{top_category.percentage}%。"
                else:
                    zh_summary_text = "本周论文分类分布较为均衡。"
            else:
                zh_summary_text = zh_summary[0]
            
            if not en_summary or len(en_summary) == 0:
                top_category = category_stats[0] if category_stats else None
                if top_category:
                    en_summary_text = f"This week's papers are mainly concentrated in {top_category.categoryName.en}, accounting for {top_category.percentage}%."
                else:
                    en_summary_text = "This week's paper category distribution is relatively balanced."
            else:
                en_summary_text = en_summary[0]
            
            return BilingualText(zh=zh_summary_text, en=en_summary_text)
            
        except Exception as e:
            logger.error(f"Error generating category summary: {e}")
            # 返回默认值
            top_category = category_stats[0] if category_stats else None
            if top_category:
                return BilingualText(
                    zh=f"本周论文主要集中在{top_category.categoryName.zh}领域，占比{top_category.percentage}%。",
                    en=f"This week's papers are mainly concentrated in {top_category.categoryName.en}, accounting for {top_category.percentage}%."
                )
            else:
                return BilingualText(
                    zh="本周论文分类分布较为均衡。",
                    en="This week's paper category distribution is relatively balanced."
                )
    
    def _calculate_trending_topics(self, papers: List[Dict[str, Any]]) -> List[TrendingTopic]:
        """
        计算热点话题（基于category_detail）
        
        Args:
            papers: 论文数据列表
            
        Returns:
            热点话题列表
        """
        category_counter = Counter()
        
        for paper in papers:
            category_detail = paper.get("category_detail", {})
            if category_detail and isinstance(category_detail, dict):
                category_name = category_detail.get("category_name", "")
                if category_name:
                    category_counter[category_name] += 1
        
        # 按数量排序，取前4个
        top_categories = category_counter.most_common(4)
        
        trending_topics = []
        for i, (category_name, count) in enumerate(top_categories):
            # 计算增长率（简化处理，这里使用固定值，实际应该与上周对比）
            growth = f"+{20 + i * 5}%"
            
            # 生成中英文名称（简化处理，实际应该从taxonomy获取）
            category_zh = category_name
            category_en = category_name
            
            # 尝试从category_detail获取描述
            for paper in papers:
                category_detail = paper.get("category_detail", {})
                if category_detail and isinstance(category_detail, dict):
                    if category_detail.get("category_name") == category_name:
                        category_zh = category_detail.get("category_description", category_name)
                        category_en = category_detail.get("category_description", category_name)
                        break
            
            trending_topics.append(
                TrendingTopic(
                    name=BilingualText(zh=category_zh, en=category_en),
                    count=count,
                    growth=growth
                )
            )
        
        return trending_topics
    
    async def _get_top_papers(self, papers: List[Dict[str, Any]], top_n: int = 5) -> List[ReportPaper]:
        """
        获取Top N论文（基于AlphaXiv metrics）
        
        Args:
            papers: 论文数据列表
            top_n: 返回的论文数量
            
        Returns:
            Top论文列表
        """
        scored_papers = []
        
        for paper in papers:
            alphaxiv_detail = paper.get("alphaxiv_detail", {})
            if not alphaxiv_detail or not isinstance(alphaxiv_detail, dict):
                continue
            
            metrics = alphaxiv_detail.get("metrics", {})
            if not metrics or not isinstance(metrics, dict):
                continue
            
            # 计算综合分数
            upvotes = metrics.get("upvotes", 0)
            visits_all = metrics.get("visits_all", 0)
            github_stars = 0
            github_info = alphaxiv_detail.get("github", {})
            if github_info and isinstance(github_info, dict):
                github_stars = github_info.get("stars", 0)
            
            # 综合评分：upvotes * 10 + visits_all / 100 + github_stars * 5
            score = upvotes * 10 + visits_all / 100 + github_stars * 5
            
            scored_papers.append((paper, score))
        
        # 按分数排序
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # 取前N个
        top_papers = scored_papers[:top_n]
        
        result = []
        for idx, (paper, score) in enumerate(top_papers, 1):
            alphaxiv_detail = paper.get("alphaxiv_detail", {})
            category_detail = paper.get("category_detail", {})
            
            # 优先使用alphaxiv_detail中的数据，如果没有则使用paper中的数据
            title = alphaxiv_detail.get("title", "") if alphaxiv_detail else paper.get("title", "")
            if not title:
                title = paper.get("title", "")
            
            authors = []
            if alphaxiv_detail and isinstance(alphaxiv_detail, dict):
                authors = alphaxiv_detail.get("authors", [])
            if not authors or not isinstance(authors, list):
                authors = []
            
            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown"
            
            # 获取机构信息（简化处理）
            institution = "Unknown"
            if authors and len(authors) > 0:
                first_author = authors[0] if isinstance(authors[0], str) else str(authors[0])
                parts = first_author.split()
                institution = parts[-1] if len(parts) > 1 else "Unknown"
            
            arxiv_id = paper.get("arxiv_id", "")
            
            # 获取摘要
            abstract = alphaxiv_detail.get("abstract", "") if alphaxiv_detail else paper.get("abstract", "")
            if not abstract:
                abstract = paper.get("abstract", "")
            
            # 生成highlight（使用LLM结合title和abstract生成）
            highlight = await self._generate_highlight(title, abstract)
            
            # 获取分类信息
            category_name = "Unknown"
            if category_detail and isinstance(category_detail, dict):
                category_name = category_detail.get("category_name", "Unknown")
            
            result.append(
                ReportPaper(
                    id=idx,
                    title=BilingualText(zh=title, en=title),
                    authors=authors_str,
                    institution=institution,
                    arxivId=arxiv_id,
                    highlight=highlight,
                    category=BilingualText(zh=category_name, en=category_name),
                    abstract=BilingualText(zh=abstract, en=abstract)
                )
            )
        
        return result
    
    def _extract_topics_from_papers(self, papers: List[Dict[str, Any]]) -> List[str]:
        """
        从论文中提取话题（基于topics字段）
        
        Args:
            papers: 论文数据列表
            
        Returns:
            话题列表
        """
        all_topics = []
        
        for paper in papers:
            alphaxiv_detail = paper.get("alphaxiv_detail", {})
            if alphaxiv_detail and isinstance(alphaxiv_detail, dict):
                topics = alphaxiv_detail.get("topics", [])
                if topics and isinstance(topics, list):
                    all_topics.extend(topics)
        
        # 统计话题出现次数
        topic_counter = Counter(all_topics)
        
        # 返回出现次数最多的前10个话题
        return [topic for topic, count in topic_counter.most_common(10)]
    
    def _build_insights_prompt(self, papers: List[Dict[str, Any]], language: str = "zh") -> str:
        """
        构建核心洞察生成提示词
        
        Args:
            papers: 论文数据列表
            language: 语言代码（zh或en）
            
        Returns:
            提示词
        """
        # 收集论文摘要和分类信息
        paper_summaries = []
        for i, paper in enumerate(papers[:50], 1):  # 只取前20篇论文
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            category_detail = paper.get("category_detail", {})
            category_name = ""
            if category_detail and isinstance(category_detail, dict):
                category_name = category_detail.get("category_name", "")
            
            paper_summaries.append(
                f"论文{i}: {title}\n分类: {category_name}\n摘要: {abstract[:200]}..."
            )
        
        papers_text = "\n\n".join(paper_summaries)
        prompt=insights_prompt[language].format(papers_text=papers_text)
        return prompt
    
    def _build_keywords_prompt(self, papers: List[Dict[str, Any]], language: str = "zh") -> str:
        """
        构建新兴关键词生成提示词
        
        Args:
            papers: 论文数据列表
            language: 语言代码（zh或en）
            
        Returns:
            提示词
        """
        # 收集论文标题和摘要中的关键词
        all_text = []
        for paper in papers[:30]:  # 只取前30篇论文
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            all_text.append(f"{title} {abstract}")
        
        combined_text = " ".join(all_text)
        prompt = keywords_prompt[language].format(combined_text=combined_text)
        return prompt
    
    def _build_title_prompt(self, papers: List[Dict[str, Any]], week_num: int, date_range: str, language: str = "zh") -> str:
        """
        构建标题生成提示词
        
        Args:
            papers: 论文数据列表
            week_num: 周数
            date_range: 日期范围
            language: 语言代码（zh或en）
            
        Returns:
            提示词
        """
        # 收集论文摘要和分类信息
        paper_summaries = []
        for i, paper in enumerate(papers[:30], 1):  # 只取前30篇论文
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            category_detail = paper.get("category_detail", {})
            category_name = ""
            if category_detail and isinstance(category_detail, dict):
                category_name = category_detail.get("category_name", "")
            
            paper_summaries.append(
                f"论文{i}: {title}\n分类: {category_name}\n摘要: {abstract[:200]}..."
            )
        
        papers_text = "\n\n".join(paper_summaries)
        prompt = title_prompt[language].format(
            papers_text=papers_text,
            week_num=week_num,
            date_range=date_range
        )
        return prompt
    
    def _build_overview_prompt(self, papers: List[Dict[str, Any]], week_num: int, date_range: str, total_papers: int, language: str = "zh") -> str:
        """
        构建概览生成提示词
        
        Args:
            papers: 论文数据列表
            week_num: 周数
            date_range: 日期范围
            total_papers: 论文总数
            language: 语言代码（zh或en）
            
        Returns:
            提示词
        """
        # 收集论文摘要和分类信息
        paper_summaries = []
        for i, paper in enumerate(papers[:50], 1):  # 只取前50篇论文
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            category_detail = paper.get("category_detail", {})
            category_name = ""
            if category_detail and isinstance(category_detail, dict):
                category_name = category_detail.get("category_name", "")
            
            paper_summaries.append(
                f"论文{i}: {title}\n分类: {category_name}\n摘要: {abstract[:200]}..."
            )
        
        papers_text = "\n\n".join(paper_summaries)
        prompt = overview_prompt[language].format(
            papers_text=papers_text,
            week_num=week_num,
            date_range=date_range,
            total_papers=total_papers
        )
        return prompt
    
    def _build_highlight_prompt(self, title: str, abstract: str, language: str = "zh") -> str:
        """
        构建论文亮点生成提示词
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            language: 语言代码（zh或en）
            
        Returns:
            提示词
        """
        prompt = highlight_prompt[language].format(
            title=title,
            abstract=abstract
        )
        return prompt
    
    async def _generate_highlight(self, title: str, abstract: str) -> BilingualText:
        """
        生成论文亮点（中英文）
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            论文亮点（中英文）
        """
        try:
            # 生成中文亮点
            zh_prompt = self._build_highlight_prompt(title, abstract, language="zh")
            zh_response = await self.llm_engine.simple_chat(
                prompt=zh_prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确凝练论文亮点。"
            )
            
            # 生成英文亮点
            en_prompt = self._build_highlight_prompt(title, abstract, language="en")
            en_response = await self.llm_engine.simple_chat(
                prompt=en_prompt,
                system_prompt="You are a professional expert in the information retrieval field. Please accurately refine paper highlights."
            )
            
            # 解析响应
            zh_highlight_list = self._parse_json_response(zh_response, "highlight")
            en_highlight_list = self._parse_json_response(en_response, "highlight")
            
            # 如果解析失败，使用默认值（截取摘要前100字符）
            if not zh_highlight_list or len(zh_highlight_list) == 0:
                zh_highlight = abstract[:100] + "..." if len(abstract) > 100 else abstract
            else:
                zh_highlight = zh_highlight_list[0]
            
            if not en_highlight_list or len(en_highlight_list) == 0:
                en_highlight = abstract[:100] + "..." if len(abstract) > 100 else abstract
            else:
                en_highlight = en_highlight_list[0]
            
            return BilingualText(zh=zh_highlight, en=en_highlight)
            
        except Exception as e:
            logger.error(f"Error generating highlight: {e}")
            # 返回默认值（截取摘要前100字符）
            default_highlight = abstract[:100] + "..." if len(abstract) > 100 else abstract
            return BilingualText(zh=default_highlight, en=default_highlight)
    
    async def _generate_insights(self, papers: List[Dict[str, Any]]) -> List[BilingualText]:
        """
        生成核心洞察（中英文）
        
        Args:
            papers: 论文数据列表
            
        Returns:
            核心洞察列表（中英文）
        """
        try:
            # 生成中文洞察
            zh_prompt = self._build_insights_prompt(papers, language="zh")
            zh_response = await self.llm_engine.simple_chat(
                prompt=zh_prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确分析论文趋势。"
            )
            
            # 生成英文洞察
            en_prompt = self._build_insights_prompt(papers, language="en")
            en_response = await self.llm_engine.simple_chat(
                prompt=en_prompt,
                system_prompt="You are a professional expert in the information retrieval field. Please accurately analyze paper trends."
            )
            
            # 解析响应
            zh_insights = self._parse_json_response(zh_response, "insights")
            en_insights = self._parse_json_response(en_response, "insights")
            
            # 确保数量一致
            min_count = min(len(zh_insights), len(en_insights))
            if min_count == 0:
                # 如果解析失败，返回默认值
                return [
                    BilingualText(
                        zh="本周信息检索领域研究活跃，多篇论文聚焦于检索增强生成和密集检索技术。",
                        en="This week's information retrieval research is active, with multiple papers focusing on retrieval-augmented generation and dense retrieval technologies."
                    )
                ]
            
            # 组合中英文洞察
            insights = []
            for i in range(min_count):
                insights.append(
                    BilingualText(
                        zh=zh_insights[i] if i < len(zh_insights) else "",
                        en=en_insights[i] if i < len(en_insights) else ""
                    )
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            # 返回默认值
            return [
                BilingualText(
                    zh="本周信息检索领域研究活跃，多篇论文聚焦于检索增强生成和密集检索技术。",
                    en="This week's information retrieval research is active, with multiple papers focusing on retrieval-augmented generation and dense retrieval technologies."
                )
            ]
    
    async def _generate_keywords(self, papers: List[Dict[str, Any]]) -> List[BilingualText]:
        """
        生成新兴关键词（中英文）
        
        Args:
            papers: 论文数据列表
            
        Returns:
            新兴关键词列表（中英文）
        """
        try:
            # 生成中文关键词
            zh_prompt = self._build_keywords_prompt(papers, language="zh")
            zh_response = await self.llm_engine.simple_chat(
                prompt=zh_prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确提取新兴关键词。"
            )
            
            # 生成英文关键词
            en_prompt = self._build_keywords_prompt(papers, language="en")
            en_response = await self.llm_engine.simple_chat(
                prompt=en_prompt,
                system_prompt="You are a professional expert in the information retrieval field. Please accurately extract emerging keywords."
            )
            
            # 解析响应
            zh_keywords = self._parse_json_response(zh_response, "keywords")
            en_keywords = self._parse_json_response(en_response, "keywords")
            
            # 确保数量一致
            min_count = min(len(zh_keywords), len(en_keywords))
            if min_count == 0:
                # 如果解析失败，返回默认值
                return [
                    BilingualText(zh="检索增强生成", en="Retrieval-Augmented Generation"),
                    BilingualText(zh="密集检索", en="Dense Retrieval"),
                    BilingualText(zh="神经重排序", en="Neural Re-ranking")
                ]
            
            # 组合中英文关键词
            keywords = []
            for i in range(min_count):
                keywords.append(
                    BilingualText(
                        zh=zh_keywords[i] if i < len(zh_keywords) else "",
                        en=en_keywords[i] if i < len(en_keywords) else ""
                    )
                )
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            # 返回默认值
            return [
                BilingualText(zh="检索增强生成", en="Retrieval-Augmented Generation"),
                BilingualText(zh="密集检索", en="Dense Retrieval"),
                BilingualText(zh="神经重排序", en="Neural Re-ranking")
            ]
    
    async def _generate_title(self, papers: List[Dict[str, Any]], week_num: int, date_range: str) -> BilingualText:
        """
        生成周报标题（中英文）
        
        Args:
            papers: 论文数据列表
            week_num: 周数
            date_range: 日期范围
            
        Returns:
            周报标题（中英文）
        """
        try:
            # 生成中文标题
            zh_prompt = self._build_title_prompt(papers, week_num, date_range, language="zh")
            zh_response = await self.llm_engine.simple_chat(
                prompt=zh_prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确生成周报标题。"
            )
            
            # 生成英文标题
            en_prompt = self._build_title_prompt(papers, week_num, date_range, language="en")
            en_response = await self.llm_engine.simple_chat(
                prompt=en_prompt,
                system_prompt="You are a professional expert in the information retrieval field. Please accurately generate weekly report titles."
            )
            
            # 解析响应
            zh_title = self._parse_json_response(zh_response, "title")
            en_title = self._parse_json_response(en_response, "title")
            
            # 如果解析失败，使用默认值
            if not zh_title or len(zh_title) == 0:
                zh_title_text = f"第{week_num}周: 信息检索研究动态"
            else:
                zh_title_text = zh_title[0]
            
            if not en_title or len(en_title) == 0:
                en_title_text = f"Week {week_num}: Information Retrieval Research Trends"
            else:
                en_title_text = en_title[0]
            
            return BilingualText(zh=zh_title_text, en=en_title_text)
            
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            # 返回默认值
            return BilingualText(
                zh=f"第{week_num}周: 信息检索研究动态",
                en=f"Week {week_num}: Information Retrieval Research Trends"
            )
    
    async def _generate_overview(self, papers: List[Dict[str, Any]], week_num: int, date_range: str, total_papers: int) -> BilingualText:
        """
        生成周报概览（中英文）
        
        Args:
            papers: 论文数据列表
            week_num: 周数
            date_range: 日期范围
            total_papers: 论文总数
            
        Returns:
            周报概览（中英文）
        """
        try:
            # 生成中文概览
            zh_prompt = self._build_overview_prompt(papers, week_num, date_range, total_papers, language="zh")
            zh_response = await self.llm_engine.simple_chat(
                prompt=zh_prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确生成周报概览。"
            )
            
            # 生成英文概览
            en_prompt = self._build_overview_prompt(papers, week_num, date_range, total_papers, language="en")
            en_response = await self.llm_engine.simple_chat(
                prompt=en_prompt,
                system_prompt="You are a professional expert in the information retrieval field. Please accurately generate weekly report overviews."
            )
            
            # 解析响应
            zh_overview = self._parse_json_response(zh_response, "overview")
            en_overview = self._parse_json_response(en_response, "overview")
            
            # 如果解析失败，使用默认值
            if not zh_overview or len(zh_overview) == 0:
                zh_overview_text = f"AI本周摘要: 本周信息检索领域共有{total_papers}篇论文，涵盖了多个重要研究方向和技术突破。"
            else:
                zh_overview_text = zh_overview[0]
            
            if not en_overview or len(en_overview) == 0:
                en_overview_text = f"AI Weekly Summary: This week's information retrieval field has {total_papers} papers covering multiple important research directions and technical breakthroughs."
            else:
                en_overview_text = en_overview[0]
            
            return BilingualText(zh=zh_overview_text, en=en_overview_text)
            
        except Exception as e:
            logger.error(f"Error generating overview: {e}")
            # 返回默认值
            return BilingualText(
                zh=f"AI本周摘要: 本周信息检索领域共有{total_papers}篇论文，涵盖了多个重要研究方向和技术突破。",
                en=f"AI Weekly Summary: This week's information retrieval field has {total_papers} papers covering multiple important research directions and technical breakthroughs."
            )
    
    def _parse_json_response(self, response: str, key: str) -> List[str]:
        """
        解析JSON响应
        
        Args:
            response: LLM响应文本
            key: 要提取的键名
            
        Returns:
            提取的值列表或字符串列表
        """
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                value = result.get(key, [])
                # 如果值是字符串，转换为列表
                if isinstance(value, str):
                    return [value]
                # 如果值是列表，直接返回
                if isinstance(value, list):
                    return value
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
        
        return []
    
    async def get_all_weekly_reports(self) -> List[WeeklyReport]:
        """
        获取所有周报列表（从 Milvus 查询）
        
        Returns:
            周报列表
        """
        try:
            # 从 Milvus 查询所有报告
            results = await self.query_reports()
            
            if not results:
                logger.warning("No reports found in Milvus")
                return []
            
            # 将查询结果转换为 WeeklyReport 对象
            reports = []
            for result in results:
                weekly_report_data = result.get("weekly_report", {})
                if weekly_report_data:
                    try:
                        # 将字典转换为 WeeklyReport 对象
                        report = WeeklyReport(**weekly_report_data)
                        reports.append(report)
                    except Exception as e:
                        logger.error(f"Error parsing report data: {e}")
                        continue
            
            # 按ID降序排序（最新的在前）
            reports.sort(key=lambda x: int(x.id) if x.id.isdigit() else 0, reverse=True)
            
            logger.info(f"Retrieved {len(reports)} reports from Milvus")
            return reports
            
        except Exception as e:
            logger.error(f"Error getting all weekly reports: {e}")
            raise
    
    async def generate_weekly_reports(self, num_weeks: int = 1) -> List[WeeklyReport]:
        """
        生成指定周数的报告
        
        Args:
            num_weeks: 要生成的周数，1表示本周，2表示本周和上一周，以此类推
            
        Returns:
            周报列表，按时间降序排序（最新的在前）
        """
        try:
            if num_weeks < 1:
                logger.warning(f"Invalid num_weeks: {num_weeks}, using default value 1")
                num_weeks = 1
            
            reports = []
            # 生成指定周数的周报
            # num_weeks=1: week_offset=0 (本周)
            # num_weeks=2: week_offset=-1, 0 (本周和上一周)
            # num_weeks=3: week_offset=-2, -1, 0 (本周、上一周和上上周)
            for week_offset in range(-(num_weeks - 1), 1):
                start_date, end_date, week_str, date_range_str = self._get_week_range(week_offset)
                
                # 查询该周的论文
                papers = await self._query_weekly_papers(start_date, end_date)
                
                if not papers:
                    logger.warning(f"No papers found for week_offset {week_offset}, skipping")
                    continue
                
                # 提取话题
                topics = self._extract_topics_from_papers(papers)
                topics_count = len(set(topics))
                
                # 生成摘要（简化处理，实际应该使用LLM生成）
                summary_zh = f"本周信息检索领域共有{len(papers)}篇论文，涵盖{topics_count}个热点话题。"
                summary_en = f"This week's information retrieval field has {len(papers)} papers covering {topics_count} hot topics."
                
                # 生成highlights（基于top categories）
                trending_topics = self._calculate_trending_topics(papers)
                highlights = [
                    BilingualText(zh=topic.name.zh, en=topic.name.en)
                    for topic in trending_topics[:4]
                ]
                
                # 生成报告ID（基于周数）
                week_num = int((datetime.now() - datetime(2025, 1, 1)).days / 7) + week_offset
                report_id = str(week_num)
                
                # 生成发布日期
                publish_date = end_date.strftime("%m-%d")
                
                # 计算热点话题
                trending_topics_list = self._calculate_trending_topics(papers)
                
                # 生成标题和概览
                title = await self._generate_title(papers, week_num, date_range_str)
                overview = await self._generate_overview(papers, week_num, date_range_str, len(papers))
                
                # 生成核心洞察
                key_insights = await self._generate_insights(papers)
                
                # 获取Top论文
                top_papers = await self._get_top_papers(papers, top_n=5)
                
                # 生成新兴关键词
                emerging_keywords = await self._generate_keywords(papers)
                
                # 计算分类统计
                category_statistics = self._calculate_category_statistics(papers)
                
                # 生成分类总结
                category_summary = await self._generate_category_summary(category_statistics)
                
                # 生成完整的报告
                reports.append(
                    WeeklyReport(
                        id=report_id,
                        week=week_str,
                        dateRange=date_range_str,
                        publishDate=publish_date,
                        totalPapers=len(papers),
                        topicsCount=topics_count,
                        title=title,
                        overview=overview,
                        summary=BilingualText(zh=summary_zh, en=summary_en),
                        highlights=highlights,
                        trendingTopics=trending_topics_list,
                        keyInsights=key_insights,
                        topPapers=top_papers,
                        emergingKeywords=emerging_keywords,
                        categoryStatistics=category_statistics,
                        categorySummary=category_summary
                    )
                )
            
            # 按ID降序排序（最新的在前）
            reports.sort(key=lambda x: int(x.id), reverse=True)
            
            logger.info(f"Generated {len(reports)} weekly reports for {num_weeks} weeks")
            return reports
            
        except Exception as e:
            logger.error(f"Error generating weekly reports: {e}")
            raise
    
    async def get_report_detail(self, report_id: str) -> Optional[WeeklyReport]:
        """
        获取报告详情（从 Milvus 查询）
        
        Args:
            report_id: 周报ID
            
        Returns:
            周报对象或None
        """
        try:
            # 从 Milvus 查询指定ID的报告
            filter_expr = f'report_id == "{report_id}"'
            results = await self.query_reports(filter_expr=filter_expr)
            
            if not results or len(results) == 0:
                logger.warning(f"Report {report_id} not found in Milvus")
                return None
            
            result = results[0]
            
            # 从查询结果中获取 weekly_report
            weekly_report_data = result.get("weekly_report", {})
            if not weekly_report_data:
                logger.warning(f"Report data not found for report {report_id}")
                return None
            
            try:
                # 将字典转换为 WeeklyReport 对象
                report = WeeklyReport(**weekly_report_data)
                return report
            except Exception as e:
                logger.error(f"Error parsing report data: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting report detail for {report_id}: {e}")
            raise

