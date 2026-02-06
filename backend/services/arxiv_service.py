import asyncio
import hashlib
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import time
import arxiv
import pytz
from loguru import logger
from backend.services.score_services import PaperScoreService
from backend.services.category_service import PaperCategoryService
from backend.services.alphaxiv_service import AlphaXivService
from backend.services.affiliation_services import AffiliationService

class ArxivService:
    """arXiv论文服务，专门用于信息检索相关论文的获取和处理"""

    def __init__(self, max_results: int = 100):
        """
        初始化arXiv论文服务
        
        Args:
            max_results: 每次查询的最大结果数
        """
        self.max_results = max_results
        self.ir_categories = "cat:cs.IR"
        self.ir_keywords = 'ti:"information retrieval" OR ti:"search engine" OR ti:"text mining" OR ti:"document retrieval" OR ti:"query processing" OR ti:"web search" OR ti:"ranking algorithm" OR ti:"retrieval" OR ti:"RAG" OR ti:"deepresearch"'

        self.client = arxiv.Client()

        # 初始化AI评分服务
        try:
            self.score_service = PaperScoreService()
            logger.info("AI评分服务初始化成功")
        except Exception as e:
            logger.error(f"AI评分服务初始化失败: {e}")
            raise RuntimeError(f"AI评分服务初始化失败，无法继续: {e}")

        # 初始化AI分类服务
        try:
            self.category_service = PaperCategoryService()
            logger.info("AI分类服务初始化成功")
        except Exception as e:
            logger.error(f"AI分类服务初始化失败: {e}")
            raise RuntimeError(f"AI分类服务初始化失败，无法继续: {e}")

        # 初始化AlphaXiv服务
        try:
            self.alphaxiv_service = AlphaXivService()
            logger.info("AlphaXiv服务初始化成功")
        except Exception as e:
            logger.warning(f"AlphaXiv服务初始化失败: {e}，将跳过AlphaXiv数据获取")
            self.alphaxiv_service = None

        # 初始化机构抽取服务
        try:
            self.affiliation_service = AffiliationService()
            logger.info("机构抽取服务初始化成功")
        except Exception as e:
            logger.warning(f"机构抽取服务初始化失败: {e}，将跳过机构信息获取")
            self.affiliation_service = None

    def _format_date_for_query(self, date: datetime) -> str:
        """
        将日期格式化为arXiv API查询格式 (YYYYMMDDHHMM)
        
        Args:
            date: 要格式化的日期
            
        Returns:
            格式化后的日期字符串
        """

        # arXiv使用UTC时间，但需要考虑6小时的时区偏移
        if date.tzinfo is None:
            # 假设输入是本地时间，转换为UTC
            date = pytz.timezone('UTC').localize(date)
        else:
            date = date.astimezone(pytz.timezone('UTC'))

        # arXiv的提交时间通常有6小时的偏移
        date = date - timedelta(hours=6)
        return date.strftime('%Y%m%d%H%M')

    def _build_query(self, target_date: datetime) -> str:
        """
        构建查询字符串，获取指定日期的论文
        
        Args:
            target_date: 目标日期
            
        Returns:
            查询字符串
        """
        # 使用前一天到当天的范围，确保获取到指定日期提交的论文
        start_time = target_date - timedelta(days=1)
        end_time = target_date

        start_str = self._format_date_for_query(start_time)
        end_str = self._format_date_for_query(end_time)

        # 使用submittedDate过滤器和IR相关查询
        query = f"({self.ir_categories} OR {self.ir_keywords}) AND submittedDate:[{start_str} TO {end_str}]"
        return query

    def _build_query_range(self, start_date: datetime, end_date: datetime) -> str:
        """
        构建日期范围查询字符串
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            查询字符串
        """

        start_str = self._format_date_for_query(start_date)
        end_str = self._format_date_for_query(end_date)

        # 使用submittedDate过滤器和IR相关查询
        query = f"({self.ir_categories} OR {self.ir_keywords}) AND submittedDate:[{start_str} TO {end_str}]"
        return query

    def _search_papers(self, query: str) -> List:
        """
        执行论文搜索
        
        Args:
            query: 查询字符串
            
        Returns:
            论文结果列表
        """
        logger.info(f"Searching for papers with query: {query}")
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        return list(self.client.results(search))

    def _generate_paper_id(self, arxiv_url: str) -> str:
        """从arXiv URL生成唯一的paper_id"""
        if "arxiv.org" in arxiv_url:
            parts = arxiv_url.split("/")
            if len(parts) > 0:
                arxiv_id = parts[-1]
                if "v" in arxiv_id:
                    arxiv_id = arxiv_id.split("v")[0]
                return arxiv_id

        return hashlib.md5(arxiv_url.encode()).hexdigest()[:16]
    
    def _extract_arxiv_id(self, entry_id: str) -> Optional[str]:
        """
        从arXiv entry_id中提取arxiv ID（用于AlphaXiv API）
        
        Args:
            entry_id: arXiv entry_id，格式如 "http://arxiv.org/abs/2510.18433v1"
            
        Returns:
            arxiv ID，格式如 "2510.18433"，如果提取失败返回None
        """
        if not entry_id or "arxiv.org" not in entry_id:
            return None
        
        try:
            # 提取URL最后一部分，如 "2510.18433v1"
            parts = entry_id.rstrip("/").split("/")
            if len(parts) > 0:
                arxiv_id_with_version = parts[-1]
                # 去掉版本号（如果有）
                if "v" in arxiv_id_with_version:
                    arxiv_id = arxiv_id_with_version.split("v")[0]
                else:
                    arxiv_id = arxiv_id_with_version
                return arxiv_id
        except Exception as e:
            logger.warning(f"提取arxiv ID失败: {e}")
        
        return None

    async def _format_paper_schema(self, arxiv_papers: List) -> List[Dict[str, Any]]:
        """
        将arXiv论文数据转换为字典格式
        
        Args:
            arxiv_papers: arXiv论文列表
            
        Returns:
            论文字典列表
        """
        papers = []

        for i, paper in enumerate(arxiv_papers):
            # 提取主要分类
            category = "IR"  # 默认为IR
            if hasattr(paper, 'primary_category'):
                if paper.primary_category.startswith('cs.'):
                    category = paper.primary_category.split('.')[-1].upper()

            # 计算评分
            if not self.score_service:
                raise RuntimeError("AI评分服务未初始化")

            try:
                # 准备论文数据
                title = paper.title
                abstract = paper.summary if hasattr(paper, 'summary') else ""
                categories = paper.categories if hasattr(paper, 'categories') else []

                # 尝试使用AI评分
                try:
                    result = await self.score_service.score_paper(title, abstract, categories)
                    score = max(60, min(98, result.get("overall_score", 75)))
                    score_detail = result
                except Exception as ai_error:
                    logger.warning(f"AI评分失败，使用基于规则的评分: {ai_error}")

                    # 使用基于规则的评分作为备选方案
                    paper_data = {
                        'title': title,
                        'abstract': abstract,
                        'categories': categories,
                        'published': paper.published if hasattr(paper, 'published') else None
                    }
                    score = max(60, min(98, self.score_service.get_rule_based_score(paper_data)))
                    score_detail = {
                        "overall_score": score,
                        "reasoning": "基于规则的评分",
                        "fallback": True
                    }

            except Exception as e:
                logger.error(f"评分失败: {e}")
                raise RuntimeError(f"评分失败: {e}")

            if not self.category_service:
                raise RuntimeError("AI分类服务未初始化")
            try:
                # 尝试使用AI分类
                try:
                    category_result = await self.category_service.classify_paper(title, abstract)
                    category_detail = category_result
                except Exception as ai_error:
                    logger.warning(f"AI分类失败，使用默认分类: {ai_error}")
                    
                    # 使用默认分类作为备选方案
                    category_detail = {
                        "category": "Information Retrieval Models",
                        "subcategory": "General IR",
                        "confidence": 0.5,
                        "reasoning": "默认分类（AI分类失败）",
                        "fallback": True
                    }

            except Exception as e:
                logger.error(f"分类失败: {e}")
                # 使用最基本的默认分类
                category_detail = {
                    "category": "Information Retrieval Models",
                    "subcategory": "General IR",
                    "confidence": 0.3,
                    "reasoning": "默认分类（分类服务异常）",
                    "fallback": True
                }

            # 模拟浏览量
            views_count = random.randint(100, 2000)
            views = f"{views_count}{'k' if views_count > 1000 else ''}"
            # 模拟引用数（新论文引用数较少）
            days_since_published = (datetime.now() - paper.published.replace(tzinfo=None)).days
            citations = max(0, random.randint(0, 50) - days_since_published)
            # 模拟评论数
            comments = random.randint(0, 20) if random.random() > 0.5 else random.randint(1, 10)
            # 是否为热门论文（基于AI评分结果）
            trending = random.randint(2, 5) if score > 85 and random.random() > 0.7 else random.randint(1, 3)

            # 提取作者信息（最多显示3个）
            authors = [author.name for author in paper.authors[:3]]
            if len(paper.authors) > 3:
                authors.append("et al.")

            paper_id = self._generate_paper_id(paper.entry_id or "")

            # 获取AlphaXiv数据
            alphaxiv_detail = None
            alphaxiv_overview = None
            
            if self.alphaxiv_service and paper.entry_id:
                try:
                    # 从arxiv URL中提取paper_id（用于alphaxiv API）
                    alphaxiv_paper_id = self._extract_arxiv_id(paper.entry_id)
                    
                    if alphaxiv_paper_id:
                        # 获取论文详情
                        raw_detail = await self.alphaxiv_service.get_paper_alphaxiv_detail(alphaxiv_paper_id)
                        time.sleep(1)
                        if raw_detail:
                            # 格式化论文详情
                            alphaxiv_detail = self.alphaxiv_service.format_paper_detail(raw_detail)
                            # 如果成功获取详情，尝试获取overview
                            if alphaxiv_detail and alphaxiv_detail.get("id"):
                                paper_version_id = alphaxiv_detail.get("id")
                                alphaxiv_overview = await self.alphaxiv_service.get_paper_overview(paper_version_id)
                                time.sleep(1)

                except Exception as e:
                    logger.warning(f"获取AlphaXiv数据失败 (paper_id: {paper_id}): {e}")
                    # 继续处理，不因为AlphaXiv失败而中断

            # 获取机构信息
            affiliation_detail = None
            if self.affiliation_service and paper.pdf_url:
                try:
                    affiliation_result = await self.affiliation_service.extract_affiliations_from_pdf(paper.pdf_url)
                    if affiliation_result and affiliation_result.get("affiliations"):
                        affiliation_detail = {
                            "affiliations": affiliation_result["affiliations"],
                            "timestamp": affiliation_result["timestamp"],
                            "text_length": affiliation_result.get("text_length"),
                            # "markdown_content": affiliation_result.get("markdown_content"),
                            "pdf_minio_info": affiliation_result.get("pdf_minio_info"),
                            "markdown_minio_info": affiliation_result.get("markdown_minio_info")
                        }
                        logger.info(f"成功获取论文 {paper_id} 的机构信息: {len(affiliation_result['affiliations'])} 个作者机构")
                    else:
                        logger.debug(f"论文 {paper_id} 未获取到机构信息")
                except Exception as e:
                    logger.warning(f"获取机构信息失败 (paper_id: {paper_id}): {e}")
                    # 继续处理，不因为机构信息获取失败而中断
            paper_response = {
                # ======原始数据======
                "arxiv_id": paper_id,
                "title": paper.title,
                "abstract": abstract,
                "full_text": f"{title} {abstract}",
                "metadata":{
                    "category": category,
                    "authors": authors,
                    "timestamp": int(paper.published.timestamp()),
                    "date": paper.published.strftime("%Y-%m-%d"),
                    "pdf_url": paper.pdf_url,
                    "arxiv_url": paper.entry_id,
                    "doi": paper.doi if hasattr(paper, 'doi') and paper.doi else None,
                    "journal_ref": paper.journal_ref if hasattr(paper, 'journal_ref') and paper.journal_ref else None,
                    "primary_category": paper.primary_category,
                    "all_categories": paper.categories,
                },
                # =====统计数据======
                "hits":{
                    "views": views,
                    "citations": citations,
                    "comments": comments,
                    "trending": trending,
                },
                # ====预测数据======
                # dict结构
                "score_detail": score_detail,
                # dict结构
                "category_detail": category_detail,
                # ====AlphaXiv数据=====
                # dict结构
                "alphaxiv_detail": alphaxiv_detail,
                # dict结构
                "alphaxiv_overview": alphaxiv_overview,
                # ====机构信息=====
                # dict结构，包含作者机构关系信息
                "affiliation_detail": affiliation_detail
            }
            papers.append(paper_response)
        # 按日期排序
        papers.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
        return papers

    def _parse_date(self, date_input: Union[datetime, str, None]) -> datetime:
        """
        解析日期输入，支持datetime对象和日期字符串
        
        Args:
            date_input: 日期输入，可以是：
                - datetime对象
                - 日期字符串 "YYYYMMDD" (如 "20251108")
                - 日期字符串 "YYYY-MM-DD" (如 "2025-11-08")
                - None (返回当天日期)
                
        Returns:
            datetime对象（时间部分设置为00:00:00）
            
        Raises:
            ValueError: 如果日期格式无法解析
        """
        if date_input is None:
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if isinstance(date_input, datetime):
            return date_input.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if isinstance(date_input, str):
            date_str = date_input.strip()
            
            # 尝试解析 "YYYYMMDD" 格式 (如 "20251108")
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    return datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    raise ValueError(f"无法解析日期格式: {date_str}，期望格式: YYYYMMDD")
            
            # 尝试解析 "YYYY-MM-DD" 格式 (如 "2025-11-08")
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # 尝试其他常见格式
                for fmt in ["%Y/%m/%d", "%Y.%m.%d"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                
                raise ValueError(f"无法解析日期格式: {date_str}，支持的格式: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD")
        
        raise ValueError(f"不支持的日期类型: {type(date_input)}")

    async def get_papers(
            self,
            start_date: Optional[Union[datetime, str]] = None,
            end_date: Optional[Union[datetime, str]] = None,
            search_query: Optional[str] = None,
            category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        统一的论文获取函数，支持获取当天、指定日期或日期范围的论文
        
        Args:
            start_date: 开始日期，可以是：
                - None: 获取今天的论文
                - datetime对象: 指定日期
                - 日期字符串 "YYYYMMDD" (如 "20251108")
                - 日期字符串 "YYYY-MM-DD" (如 "2025-11-08")
            end_date: 结束日期（可选），格式同start_date
                - 如果只提供start_date: 获取指定日期的论文
                - 如果同时提供start_date和end_date: 获取日期范围的论文
            search_query: 搜索关键词（可选），在标题和摘要中搜索
            category_filter: 分类过滤（可选），支持的分类：
                - "all" 或 None: 不过滤
                - "ai", "cv", "nlp", "ro", "lg", "gn", "ir": 过滤指定分类
                
        Returns:
            论文字典列表
            
        Examples:
            # 获取今天的论文
            papers = await service.get_papers()
            
            # 获取指定日期的论文
            papers = await service.get_papers(start_date="2025-11-08")
            papers = await service.get_papers(start_date="20251108")
            papers = await service.get_papers(start_date=datetime(2025, 11, 8))
            
            # 获取日期范围的论文
            papers = await service.get_papers(
                start_date="2025-11-01",
                end_date="2025-11-08",
                search_query="transformer",
                category_filter="ir"
            )
        """
        loop = asyncio.get_event_loop()
        try:
            # 解析日期参数
            parsed_start_date = self._parse_date(start_date)
            parsed_end_date = self._parse_date(end_date) if end_date is not None else None
            
            # 构建查询
            if end_date is None:
                # 只有start_date（或两者都为None，获取今天）
                query = self._build_query(parsed_start_date)
            else:
                # 有start_date和end_date，获取日期范围
                query = self._build_query_range(parsed_start_date, parsed_end_date)
            
            # 如果有搜索关键词，添加到查询中
            if search_query:
                query = f"({query}) AND (ti:{search_query} OR abs:{search_query})"
            
            # 执行搜索
            arxiv_papers = await loop.run_in_executor(None, self._search_papers, query)
            
            # 转换为字典格式（异步）
            papers = await self._format_paper_schema(arxiv_papers)
            # 应用分类过滤
            if category_filter and category_filter != "all":
                category_map = {
                    "ai": "AI",
                    "cv": "CV",
                    "nlp": "NLP",
                    "ro": "RO",
                    "lg": "LG",
                    "gn": "GN",
                    "ir": "IR"
                }
                target_category = category_map.get(category_filter.lower(), category_filter.upper())
                papers = [p for p in papers if p["category"] == target_category]
            
            return papers

        except ValueError as e:
            logger.error(f"日期解析错误: {e}")
            return []
        except Exception as e:
            logger.error(f"获取论文时发生错误: {e}")
            return []

    # 保留向后兼容的便捷方法
    async def get_today_papers(self) -> List[Dict[str, Any]]:
        """
        异步获取今天提交的IR相关论文（向后兼容方法）
        
        Returns:
            今天的论文列表（字典格式）
        """
        return await self.get_papers()

    async def get_papers_by_date(self, target_date: Union[datetime, str]) -> List[Dict[str, Any]]:
        """
        异步获取指定日期的IR相关论文（向后兼容方法）
        
        Args:
            target_date: 目标日期，可以是datetime对象或日期字符串
            
        Returns:
            指定日期的论文列表（字典格式）
        """
        return await self.get_papers(start_date=target_date)

    async def get_papers_by_date_range(
            self,
            start_date: Union[datetime, str],
            end_date: Union[datetime, str],
            search_query: Optional[str] = None,
            category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        异步获取指定日期范围内的IR相关论文（向后兼容方法）
        
        Args:
            start_date: 开始日期，可以是datetime对象或日期字符串
            end_date: 结束日期，可以是datetime对象或日期字符串
            search_query: 搜索关键词
            category_filter: 分类过滤
            
        Returns:
            论文字典列表
        """
        return await self.get_papers(
            start_date=start_date,
            end_date=end_date,
            search_query=search_query,
            category_filter=category_filter
        )
