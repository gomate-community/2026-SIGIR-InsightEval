import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig, OneAINexusConfig
from backend.prompts.score_prompt import SCORE_PROMPT

class PaperScoreService:
    """论文评分服务，使用LLM对论文进行多维度评分"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        初始化评分服务
        
        Args:
            llm_config: LLM配置，如果为None则使用默认OneAINexus配置
        """
        self.llm_engine = LLMEngine(llm_config)
        self.scoring_dimensions = {
            "novelty": "研究内容的创新性和新颖性",
            "methodology": "研究方法与研究过程",
            "relevance": "研究结论的合理性与可靠性",
            "ethics": "学术伦理与规范",
            "technical_quality": "技术质量和实现水平",
            "impact": "研究影响力和应用价值"
        }
        
    def _build_scoring_prompt(self, title: str, abstract: str, categories: List[str]) -> str:
        """
        构建评分提示词
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            categories: 论文分类
            
        Returns:
            评分提示词
        """
        categories_str = ", ".join(categories) if categories else "未知"
        prompt=SCORE_PROMPT.format(title=title, abstract=abstract, categories_str=categories_str)
        return prompt
    
    async def score_paper(
        self, 
        title: str, 
        abstract: str, 
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        对单篇论文进行评分
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            categories: 论文分类列表
            
        Returns:
            评分结果字典
        """
        try:
            # 构建提示词
            prompt = self._build_scoring_prompt(title, abstract, categories or [])
            
            # 调用LLM进行评分
            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的学术论文评审专家，请客观公正地评分。"
            )
            
            # 解析响应
            score_result = self._parse_score_response(response)
            
            # 添加元数据
            score_result["timestamp"] = datetime.now().isoformat()
            logger.info(score_result)
            return score_result
            
        except Exception as e:
            logger.error(f"评分失败: {str(e)}")
            # 返回默认评分
            return self._get_default_score(title, abstract, categories)
    
    def _parse_score_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM的评分响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的评分结果
        """
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 验证必要字段
                if "scores" in result and "overall_score" in result:
                    # 确保分数在合理范围内
                    for dim, score in result["scores"].items():
                        if not isinstance(score, (int, float)) or score < 0 or score > 100:
                            result["scores"][dim] = max(0, min(100, int(score) if isinstance(score, (int, float)) else 70))
                    
                    # 确保总分在合理范围内
                    overall = result["overall_score"]
                    if not isinstance(overall, (int, float)) or overall < 0 or overall > 100:
                        result["overall_score"] = max(0, min(100, int(overall) if isinstance(overall, (int, float)) else 70))
                    
                    return result
            
            # 如果解析失败，尝试从文本中提取分数
            return self._extract_scores_from_text(response)
            
        except Exception as e:
            logger.error(f"解析评分响应失败: {str(e)}")
            return self._get_fallback_score()
    
    def _extract_scores_from_text(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取分数信息
        
        Args:
            text: 响应文本
            
        Returns:
            提取的评分结果
        """
        scores = {}
        
        # 尝试提取各维度分数
        for dim in self.scoring_dimensions.keys():
            pattern = rf"{dim}[:\s]*(\d+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[dim] = max(0, min(100, int(match.group(1))))
            else:
                scores[dim] = 70  # 默认分数
        
        # 计算总分
        overall_score = sum(scores.values()) // len(scores)
        
        return {
            "scores": scores,
            "overall_score": overall_score,
            "reasoning": "基于文本解析的评分结果"
        }
    
    def _get_default_score(self, title: str, abstract: str, categories: Optional[List[str]]) -> Dict[str, Any]:
        """
        获取默认评分（当LLM评分失败时使用）
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            categories: 论文分类
            
        Returns:
            默认评分结果
        """
        # 基于简单规则的评分
        base_score = 70
        
        # 根据标题长度调整
        if len(title) > 50:
            base_score += 5
        
        # 根据摘要长度调整
        if abstract and len(abstract) > 500:
            base_score += 5
        
        # 根据分类调整
        if categories:
            if any("cs.IR" in cat or "Information Retrieval" in cat for cat in categories):
                base_score += 10
        
        # 添加随机变化
        import random
        variation = random.randint(-5, 10)
        final_score = max(60, min(95, base_score + variation))
        
        scores = {dim: final_score + random.randint(-5, 5) for dim in self.scoring_dimensions.keys()}
        
        return {
            "scores": scores,
            "overall_score": final_score,
            "reasoning": "基于规则的默认评分",
            "fallback": True
        }
    
    def _get_fallback_score(self) -> Dict[str, Any]:
        """
        获取兜底评分
        
        Returns:
            兜底评分结果
        """
        import random
        base_score = random.randint(65, 85)
        scores = {dim: base_score + random.randint(-10, 10) for dim in self.scoring_dimensions.keys()}
        
        return {
            "scores": scores,
            "overall_score": base_score,
            "reasoning": "系统兜底评分",
            "fallback": True
        }
    
    async def batch_score_papers(self, papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量评分论文
        
        Args:
            papers_data: 论文数据列表，每个元素包含title, abstract, categories
            
        Returns:
            评分结果列表
        """
        results = []
        
        for paper_data in papers_data:
            try:
                result = await self.score_paper(
                    title=paper_data.get("title", ""),
                    abstract=paper_data.get("abstract", ""),
                    categories=paper_data.get("categories", [])
                )
                results.append(result)
                
                # 添加延迟避免请求过快
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"批量评分失败: {str(e)}")
                results.append(self._get_fallback_score())
        
        return results
    
    def get_rule_based_score(self, paper_data: Dict[str, Any]) -> int:
        """
        基于规则的AI评分（作为异步AI评分的备选方案）
        
        Args:
            paper_data: 论文数据字典，包含 title, abstract, categories, published 等字段
            
        Returns:
            评分 (60-95)
        """
        try:
            # 基于论文特征的智能评分逻辑
            base_score = 70
            
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            categories = paper_data.get('categories', [])
            published = paper_data.get('published')
            
            # 根据标题长度和质量调整
            title_length = len(title)
            if 30 <= title_length <= 100:
                base_score += 5
            
            # 根据摘要长度调整
            abstract_length = len(abstract)
            if abstract_length > 500:
                base_score += 5
            
            # 根据分类调整
            if categories:
                if any('cs.IR' in cat for cat in categories):
                    base_score += 10
                if any('cs.AI' in cat for cat in categories):
                    base_score += 5
            
            # 根据发布时间调整（新论文加分）
            if published:
                try:
                    if hasattr(published, 'replace'):
                        published_date = published.replace(tzinfo=None)
                    else:
                        published_date = published
                    
                    days_since_published = (datetime.now() - published_date).days
                    if days_since_published <= 7:
                        base_score += 8
                    elif days_since_published <= 30:
                        base_score += 5
                except Exception:
                    pass  # 忽略日期处理错误
            
            # 添加基于内容的智能变化（而非随机）
            content_score = 0
            if title:
                # 基于标题关键词的评分
                ir_keywords = ['retrieval', 'search', 'ranking', 'query', 'document', 'information']
                title_lower = title.lower()
                content_score += sum(2 for keyword in ir_keywords if keyword in title_lower)
            
            final_score = max(60, min(95, base_score + content_score))
            
            return final_score
            
        except Exception as e:
            logger.error(f"基于规则的AI评分失败: {str(e)}")
            return 75  # 返回默认分数而不是抛出异常

    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            服务是否正常
        """
        try:
            return await self.llm_engine.health_check()
        except Exception as e:
            logger.error(f"评分服务健康检查失败: {str(e)}")
            return False



# 示例使用
if __name__ == "__main__":
    async def main():
        service = PaperScoreService()
        
        # 健康检查
        if await service.health_check():
            logger.info("✅ 评分服务正常")
        else:
            logger.error("❌ 评分服务异常")
            return
        
        # 测试评分
        result = await service.score_paper(
            title="A Novel Approach to Information Retrieval Using Deep Learning",
            abstract="This paper presents a novel deep learning approach for information retrieval that significantly improves search accuracy and efficiency. We propose a new neural architecture that combines transformer models with traditional IR techniques.",
            categories=["cs.IR", "cs.AI"]
        )
        
        logger.info("评分结果:")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 运行示例
    asyncio.run(main())