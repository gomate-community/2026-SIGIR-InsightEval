
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig, OneAINexusConfig
from backend.services.paper_service import PaperService
from backend.utils.json_utils import repair_json_output
from backend.config import Config
class PaperFindService:
    """
    论文查找服务 - 支撑Finder页面功能

    主要功能：
    1. 分析用户查询意图
    2. 搜索相关论文（使用混合检索）
    3. 评估论文相关性（评分+分类）
    4. 提取证据和引用（PDF解析）
    5. 生成搜索结果和证据说明
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        初始化论文查找服务

        Args:
            llm_config: LLM配置，如果为None则使用默认配置
        """
        self.llm_engine = LLMEngine(llm_config)
        self.paper_service = PaperService(
        uri=Config.MILVUS_URI,
        token=Config.MILVUS_TOKEN,
        collection_name=Config.COLLECTION_NAME,
        embedding_model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_base_url=Config.OPENAI_BASE_URL,
        max_results=Config.ARXIV_MAX_RESULTS
    )

    async def find_papers(self, query: str, limit: int = 20):
        """
        查找论文的主入口方法 - 流式返回

        Args:
            query: 用户查询
            limit: 返回论文数量限制

        Yields:
            逐步返回的搜索结果和处理状态
        """
        try:
            logger.info(f"开始处理查询: {query}")

            # 步骤1: 分析用户意图
            yield {"type": "step", "step": "analyzing_intent", "message": "Analyzing your request..."}
            intent_result = await self._analyze_query_intent(query)
            logger.info(f"意图分析完成: {intent_result.get('interpreted_intent', '')}")
            yield {"type": "intent_analyzed", "data": intent_result}

            # 步骤2: 搜索相关论文
            yield {"type": "step", "step": "searching_papers", "message": "Searching for papers..."}
            search_results = await self.paper_service.hybrid_search(
                query=intent_result.get('search_query', query),
                limit=limit * 2  # 多搜索一些用于筛选
            )
            logger.info(f"找到 {len(search_results)} 篇候选论文")
            # logger.info(search_results[0])
            if not search_results:
                yield {"type": "complete", "data": self._create_empty_result(query)}
                return

            yield {"type": "papers_found", "count": len(search_results)}

            # 步骤3: 评估论文相关性和质量
            yield {"type": "step", "step": "evaluating_papers", "message": "Evaluating and ranking papers..."}
            evaluated_papers = await self._evaluate_papers_relevance(
                search_results, query, intent_result
            )
            relevant_count = len([p for p in evaluated_papers if p['relevance'] in ['perfectly', 'relevant']])
            logger.info(f"评估完成，有 {relevant_count} 篇相关论文")
            yield {"type": "papers_evaluated", "count": len(evaluated_papers), "relevant_count": relevant_count}

            # 步骤4: 提取证据和引用
            yield {"type": "step", "step": "extracting_evidence", "message": "Extracting evidence and quotes..."}
            papers_with_evidence = await self._extract_evidence_and_quotes(
                evaluated_papers[:limit], query, intent_result
            )
            logger.info(f"证据提取完成")
            yield {"type": "evidence_extracted"}

            # 步骤5: 整理和返回结果
            result = self._format_search_result(
                query, intent_result, papers_with_evidence, evaluated_papers
            )

            logger.info("论文查找完成")
            yield {"type": "complete", "data": result}

        except Exception as e:
            logger.error(f"论文查找失败: {str(e)}")
            yield {"type": "error", "message": str(e)}

    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        分析用户查询意图

        Args:
            query: 用户查询

        Returns:
            意图分析结果
        """
        try:
            prompt = f"""
你是一位专业的信息检索领域专家，请分析用户的查询意图。

用户查询: "{query}"

请从以下几个方面进行分析：
1. 用户真正想要查找的论文类型或主题
2. 关键技术点或方法
3. 应用场景或研究领域
4. 时间偏好（如果有的话）

请返回JSON格式：
{{
    "interpreted_intent": "用户意图的简洁描述",
    "search_query": "优化后的搜索查询（用于检索论文）",
    "keywords": "unscripted dialogue English/generative retrieval",
    "key_criteria": ["关键标准1", "关键标准2", "关键标准3"],
    "research_focus": "主要研究重点",
    "technical_aspects": ["技术方面1", "技术方面2"]
}}

注意：
- interpreted_intent 应该简洁明了，突出核心需求
- search_query 应该适合用于论文检索，可以添加相关关键词
- key_criteria 应该包含2-4个最重要的判断标准
- 返回有效的JSON格式
/no_think
"""

            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的信息检索领域专家，擅长分析用户查询意图。"
            )
            response=repair_json_output(response)
            # 解析JSON响应
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回默认结果
                logger.warning(f"意图分析JSON解析失败，使用默认结果")
                return {
                    "interpreted_intent": f"查找关于'{query}'的论文",
                    "search_query": query,
                    "key_criteria": ["相关性", "技术质量", "创新性"],
                    "research_focus": "信息检索",
                    "technical_aspects": ["检索算法", "机器学习"]
                }

        except Exception as e:
            logger.error(f"意图分析失败: {str(e)}")
            return {
                "interpreted_intent": f"查找关于'{query}'的论文",
                "search_query": query,
                "key_criteria": ["相关性", "技术质量", "创新性"],
                "research_focus": "信息检索",
                "technical_aspects": ["检索算法", "机器学习"]
            }

    async def _evaluate_papers_relevance(
        self,
        papers: List[Dict[str, Any]],
        original_query: str,
        intent_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        评估论文的相关性（简化版，基于查询匹配）

        Args:
            papers: 论文列表
            original_query: 原始查询
            intent_result: 意图分析结果

        Returns:
            评估后的论文列表
        """
        evaluated_papers = []

        for paper in papers:
            try:
                # 基于查询相关性判断论文相关性（简化版）
                relevance = self._determine_paper_relevance_simple(paper, original_query, intent_result)

                # 生成论文摘要
                summary = await self._generate_paper_summary_simple(original_query, paper)

                # 构建评估后的论文对象
                evaluated_paper = {
                    **paper,
                    "relevance": relevance,
                    "summary": summary,
                    "selected": False,  # 默认未选中
                    "evidenceQuotes": [],  # 将在后续步骤填充
                    "relevantPassages": []  # 将在后续步骤填充
                }

                evaluated_papers.append(evaluated_paper)

            except Exception as e:
                logger.error(f"评估论文失败 {paper.get('arxiv_id', 'unknown')}: {str(e)}")
                # 添加默认评估结果
                evaluated_paper = {
                    **paper,
                    "relevance": "somewhat",
                    "summary": paper.get("abstract", "")[:200] + "...",
                    "selected": False,
                    "evidenceQuotes": [],
                    "relevantPassages": []
                }
                evaluated_papers.append(evaluated_paper)

        return evaluated_papers

    def _determine_paper_relevance_simple(
        self,
        paper: Dict[str, Any],
        query: str,
        intent_result: Dict[str, Any]
    ) -> str:
        """
        确定论文的相关性级别（简化版）

        Args:
            paper: 论文数据
            query: 查询
            intent_result: 意图分析结果

        Returns:
            相关性级别: "perfectly", "relevant", "somewhat"
        """
        try:
            # 检查关键词匹配
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            query_lower = query.lower()

            key_criteria = intent_result.get("key_criteria", [])
            technical_aspects = intent_result.get("technical_aspects", [])

            # 计算关键词匹配度
            query_keywords = set(query_lower.split())
            title_matches = len(query_keywords.intersection(set(title.split())))
            abstract_matches = len(query_keywords.intersection(set(abstract.split())))

            # 检查关键标准匹配
            criteria_matches = 0
            for criteria in key_criteria + technical_aspects:
                if criteria.lower() in title or criteria.lower() in abstract:
                    criteria_matches += 1

            # 基于关键词匹配确定相关性
            if title_matches > 0 or criteria_matches >= 2:
                return "perfectly"
            elif abstract_matches > 0 or criteria_matches >= 1:
                return "relevant"
            else:
                return "somewhat"

        except Exception as e:
            logger.error(f"相关性判断失败: {str(e)}")
            return "somewhat"

    async def _generate_paper_summary_simple(self, user_query: str, paper: Dict[str, Any]) -> str:
        """
        生成论文摘要（基于用户查询的相关性摘要）

        Args:
            user_query: 用户查询问题
            paper: 论文数据

        Returns:
            论文摘要字符串，突出与用户查询的相关性
        """
        try:
            # 构建LLM提示词
            prompt = f"""基于用户查询问题和论文信息，生成一段简洁的摘要，突出论文与用户问题的相关性。

用户查询问题：
{user_query}

论文标题：
{paper.get('title', '')}

论文摘要：
{paper.get('abstract', '')}

请生成一段简短的摘要（50-150字），重点说明这篇论文如何与用户查询相关，突出相关的关键特征、数据集、方法或贡献。摘要应该直接针对用户查询的需求。

摘要格式要求：
- 直接说明论文如何满足用户查询的需求
- 突出相关的关键信息（如数据集特征、方法创新等）
- 语言简洁明了
- 英文生成
"""

            # 调用LLM生成摘要
            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的学术论文分析助手，擅长根据用户需求生成精准的相关性摘要。"
            )

            # 清理响应文本
            summary = response.strip()
            return summary

        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            # 降级方案：返回原始摘要
            abstract = paper.get("abstract", "")
            return abstract[:200] + "..." if len(abstract) > 200 else abstract

    async def _extract_evidence_and_quotes(
        self,
        papers: List[Dict[str, Any]],
        query: str,
        intent_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        从论文摘要中提取证据和引用（简化版）

        Args:
            papers: 论文列表
            query: 查询
            intent_result: 意图分析结果

        Returns:
            包含证据和引用的论文列表
        """
        # 只处理最相关的论文（perfectly和relevant）
        relevant_papers = [p for p in papers if p["relevance"] in ["perfectly", "relevant"]]

        for paper in relevant_papers:
            try:
                # 从摘要中提取相关内容（简化版）
                abstract = paper.get("abstract", "")
                if not abstract:
                    paper["evidenceQuotes"] = []
                    paper["relevantPassages"] = []
                    continue

                # 使用LLM从摘要中提取证据
                evidence_quotes, relevant_passages = await self._extract_relevant_content_from_abstract(
                    abstract, query, intent_result, paper
                )

                paper["evidenceQuotes"] = evidence_quotes
                paper["relevantPassages"] = relevant_passages

            except Exception as e:
                logger.error(f"提取论文证据失败 {paper.get('arxiv_id', 'unknown')}: {str(e)}")
                paper["evidenceQuotes"] = []
                paper["relevantPassages"] = []

        return papers

    async def _extract_relevant_content_from_abstract(
        self,
        abstract: str,
        query: str,
        intent_result: Dict[str, Any],
        paper: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        从摘要中提取相关引用和段落

        Args:
            abstract: 论文摘要
            query: 查询
            intent_result: 意图分析结果
            paper: 论文数据

        Returns:
            (证据引用列表, 相关段落列表)
        """
        try:
            key_criteria = intent_result.get("key_criteria", [])
            technical_aspects = intent_result.get("technical_aspects", [])

            # 构建提取提示
            prompt = f"""
你是一位专业的信息检索领域专家，请从以下论文摘要中提取与查询相关的证据。

查询: "{query}"
论文标题: "{paper.get('title', '')}"
关键标准: {', '.join(key_criteria)}
技术方面: {', '.join(technical_aspects)}

论文摘要:
{abstract}

请提取：
1. 1-2个最相关的证据引用，每个引用包含：
   - criteria: 匹配的关键标准
   - quote: 相关引用内容（50-100字）

2. 1-2个最相关的段落或技术描述

请返回JSON格式：
{{
    "evidence_quotes": [
        {{
            "criteria": "关键标准名称",
            "quote": "引用内容..."
        }}
    ],
    "relevant_passages": [
        "相关段落1...",
        "相关段落2..."
    ]
}}

注意：
- 确保引用内容确实来自摘要内容
- 引用应该支持论文的相关性
- 返回有效的JSON格式
/no_think
"""

            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的信息检索领域专家，擅长从论文摘要中提取关键证据和引用。"
            )
            response=repair_json_output(response)
            # 解析响应
            try:
                result = json.loads(response)
                evidence_quotes = result.get("evidence_quotes", [])
                relevant_passages = result.get("relevant_passages", [])

                return evidence_quotes, relevant_passages

            except json.JSONDecodeError:
                logger.warning("证据提取JSON解析失败，返回空结果")
                return [], []

        except Exception as e:
            logger.error(f"提取相关内容失败: {str(e)}")
            return [], []


    async def _extract_relevant_content(
        self,
        markdown_content: str,
        query: str,
        intent_result: Dict[str, Any],
        paper: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        从markdown内容中提取相关引用和段落

        Args:
            markdown_content: markdown内容
            query: 查询
            intent_result: 意图分析结果
            paper: 论文数据

        Returns:
            (证据引用列表, 相关段落列表)
        """
        try:
            key_criteria = intent_result.get("key_criteria", [])
            technical_aspects = intent_result.get("technical_aspects", [])

            # 构建提取提示
            prompt = f"""
你是一位专业的信息检索领域专家，请从以下论文内容中提取与查询相关的证据。

查询: "{query}"
论文标题: "{paper.get('title', '')}"
关键标准: {', '.join(key_criteria)}
技术方面: {', '.join(technical_aspects)}

论文内容:
{markdown_content[:8000]}  # 限制内容长度

请提取：
1. 2-3个最相关的证据引用，每个引用包含：
   - criteria: 匹配的关键标准
   - quote: 相关引用内容（50-100字）

2. 2-3个最相关的段落或技术描述

请返回JSON格式：
{{
    "evidence_quotes": [
        {{
            "criteria": "关键标准名称",
            "quote": "引用内容..."
        }}
    ],
    "relevant_passages": [
        "相关段落1...",
        "相关段落2..."
    ]
}}

注意：
- 确保引用内容确实来自论文内容
- 引用应该支持论文的相关性
- 返回有效的JSON格式
- JSON中的内容为英文
/no_think
"""

            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的信息检索领域专家，擅长从论文中提取关键证据和引用。"
            )
            response=repair_json_output(response)
            # 解析响应
            try:
                result = json.loads(response)
                evidence_quotes = result.get("evidence_quotes", [])
                relevant_passages = result.get("relevant_passages", [])

                return evidence_quotes, relevant_passages

            except json.JSONDecodeError:
                logger.warning("证据提取JSON解析失败，返回空结果")
                return [], []

        except Exception as e:
            logger.error(f"提取相关内容失败: {str(e)}")
            return [], []


    def _format_search_result(
        self,
        query: str,
        intent_result: Dict[str, Any],
        papers_with_evidence: List[Dict[str, Any]],
        all_evaluated_papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        格式化搜索结果

        Args:
            query: 查询
            intent_result: 意图分析结果
            papers_with_evidence: 包含证据的论文
            all_evaluated_papers: 所有评估过的论文

        Returns:
            格式化的搜索结果
        """
        try:
            # 统计信息
            total_papers = len(all_evaluated_papers)
            perfectly_relevant = len([p for p in all_evaluated_papers if p["relevance"] == "perfectly"])
            relevant = len([p for p in all_evaluated_papers if p["relevance"] == "relevant"])
            somewhat_relevant = len([p for p in all_evaluated_papers if p["relevance"] == "somewhat"])

            # 构建返回结果
            result = {
                "query": query,
                "interpreted_intent": intent_result.get("interpreted_intent", ""),
                "search_stats": {
                    "total_found": total_papers,
                    "perfectly_relevant": perfectly_relevant,
                    "relevant": relevant,
                    "somewhat_relevant": somewhat_relevant,
                    "returned_count": len(papers_with_evidence)
                },
                "key_criteria": intent_result.get("key_criteria", []),
                "papers": papers_with_evidence,
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": 0  # 可以后续添加实际处理时间
            }

            return result

        except Exception as e:
            logger.error(f"格式化搜索结果失败: {str(e)}")
            return self._create_error_result(query, str(e))

    def _create_empty_result(self, query: str) -> Dict[str, Any]:
        """创建空结果"""
        return {
            "query": query,
            "interpreted_intent": f"查找关于'{query}'的论文",
            "search_stats": {
                "total_found": 0,
                "perfectly_relevant": 0,
                "relevant": 0,
                "somewhat_relevant": 0,
                "returned_count": 0
            },
            "key_criteria": [],
            "papers": [],
            "timestamp": datetime.now().isoformat(),
            "error": "未找到相关论文"
        }

    def _create_error_result(self, query: str, error: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "query": query,
            "interpreted_intent": f"查找关于'{query}'的论文",
            "search_stats": {
                "total_found": 0,
                "perfectly_relevant": 0,
                "relevant": 0,
                "somewhat_relevant": 0,
                "returned_count": 0
            },
            "key_criteria": [],
            "papers": [],
            "timestamp": datetime.now().isoformat(),
            "error": error
        }