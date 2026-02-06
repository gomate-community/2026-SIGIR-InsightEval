import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig, OneAINexusConfig
from backend.models.taxonomy import IR_CATEGORIES_MAPPING
from backend.prompts.category_prompt import CATEGORY_PROMPT
class PaperCategoryService:
    """论文分类服务，使用LLM对论文进行IR分类体系分类"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        初始化分类服务
        
        Args:
            llm_config: LLM配置，如果为None则使用默认OneAINexus配置
        """
        self.llm_engine = LLMEngine(llm_config)
        self.categories_mapping = IR_CATEGORIES_MAPPING
        
    def _build_classification_prompt(self, title: str, abstract: str) -> str:
        """
        构建分类提示词
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            分类提示词
        """
        # 构建分类选项字符串
        categories_info = []
        for i, category in enumerate(self.categories_mapping, 1):
            sub_cats = ", ".join(category["sub_categories"])
            categories_info.append(
                f"{i}. {category['category_name']}: {category['category_description']}\n"
                f"   子类别: {sub_cats}"
            )
        
        categories_str = "\n".join(categories_info)
        prompt=CATEGORY_PROMPT.format(title=title, abstract=abstract,categories_str=categories_str)
        return prompt
    
    async def classify_paper(
        self, 
        title: str, 
        abstract: str
    ) -> Dict[str, Any]:
        """
        对单篇论文进行分类
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            分类结果字典
        """
        try:
            # 构建提示词
            prompt = self._build_classification_prompt(title, abstract)
            
            # 调用LLM进行分类
            response = await self.llm_engine.simple_chat(
                prompt=prompt,
                system_prompt="你是一位专业的信息检索领域专家，请准确分类论文。"
            )
            
            # 解析响应
            classification_result = self._parse_classification_response(response)
            
            # 添加元数据
            classification_result["timestamp"] = datetime.now().isoformat()
            logger.info(f"论文分类结果: {classification_result}")
            return classification_result
            
        except Exception as e:
            logger.error(f"分类失败: {str(e)}")
            # 返回默认分类
            return self._get_default_classification(title, abstract)
    
    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM的分类响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析后的分类结果
        """
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 验证必要字段
                if "category_name" in result and "sub_categories" in result:
                    # 验证分类名称是否有效
                    valid_category = self._validate_category(result["category_name"])
                    if valid_category:
                        result["category_name"] = valid_category["category_name"]
                        result["category_description"] = valid_category["category_description"]
                        
                        # 验证子类别
                        result["sub_categories"] = self._validate_sub_categories(
                            valid_category, result.get("sub_categories", [])
                        )
                        
                        # 确保置信度在合理范围内
                        confidence = result.get("confidence", 70)
                        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                            result["confidence"] = 70
                        
                        return result
            
            # 如果解析失败，尝试从文本中提取分类信息
            return self._extract_classification_from_text(response)
            
        except Exception as e:
            logger.error(f"解析分类响应失败: {str(e)}")
            return self._get_fallback_classification()
    
    def _validate_category(self, category_name: str) -> Optional[Dict[str, Any]]:
        """
        验证分类名称是否有效
        
        Args:
            category_name: 分类名称
            
        Returns:
            有效的分类信息或None
        """
        for category in self.categories_mapping:
            if category["category_name"].lower() == category_name.lower():
                return category
        return None
    
    def _validate_sub_categories(self, category: Dict[str, Any], sub_categories: List[str]) -> List[str]:
        """
        验证子类别是否有效
        
        Args:
            category: 主分类信息
            sub_categories: 子类别列表
            
        Returns:
            有效的子类别列表
        """
        valid_sub_categories = []
        available_subs = category["sub_categories"]
        
        for sub_cat in sub_categories:
            for available_sub in available_subs:
                if available_sub.lower() == sub_cat.lower():
                    valid_sub_categories.append(available_sub)
                    break
        
        # 如果没有有效的子类别，返回第一个子类别
        if not valid_sub_categories and available_subs:
            valid_sub_categories = [available_subs[0]]
        
        return valid_sub_categories[:3]  # 最多返回3个子类别
    
    def _extract_classification_from_text(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取分类信息
        
        Args:
            text: 响应文本
            
        Returns:
            提取的分类结果
        """
        # 尝试匹配分类名称
        for category in self.categories_mapping:
            if category["category_name"].lower() in text.lower():
                return {
                    "category_name": category["category_name"],
                    "category_description": category["category_description"],
                    "sub_categories": category["sub_categories"][:2],  # 取前两个子类别
                    "confidence": 60,
                    "reasoning": "基于文本匹配的分类结果"
                }
        
        # 如果没有匹配到，返回默认分类
        return self._get_fallback_classification()
    
    def _get_default_classification(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        获取默认分类（当LLM分类失败时使用）
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            默认分类结果
        """
        # 基于关键词的简单分类逻辑
        text = (title + " " + abstract).lower()
        
        # 定义关键词映射
        keyword_mapping = {
            "Search & Ranking": ["search", "ranking", "retrieval", "query", "index"],
            "Recommender Systems": ["recommendation", "recommender", "collaborative", "filtering"],
            "Conversational AI & Agents": ["conversation", "dialogue", "chatbot", "agent", "assistant"],
            "Large Language Models & Generative AI": ["llm", "language model", "gpt", "bert", "transformer", "generative"],
            "Machine Learning for IR": ["machine learning", "deep learning", "neural", "embedding"],
            "Multimodal IR": ["multimodal", "image", "video", "audio", "cross-modal"],
            "Knowledge Graphs & Semantic Search": ["knowledge graph", "semantic", "entity", "ontology"],
            "Evaluation & Metrics": ["evaluation", "metric", "benchmark", "dataset"]
        }
        
        # 找到最匹配的分类
        best_category = None
        max_matches = 0
        
        for category_name, keywords in keyword_mapping.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > max_matches:
                max_matches = matches
                best_category = category_name
        
        # 如果没有匹配到关键词，使用默认分类
        if not best_category:
            best_category = "Search & Ranking"
        
        # 找到对应的分类信息
        for category in self.categories_mapping:
            if category["category_name"] == best_category:
                return {
                    "category_name": category["category_name"],
                    "category_description": category["category_description"],
                    "sub_categories": category["sub_categories"][:2],
                    "confidence": min(70, 50 + max_matches * 5),
                    "reasoning": "基于关键词匹配的默认分类",
                    "fallback": True
                }
        
        # 兜底返回第一个分类
        return self._get_fallback_classification()
    
    def _get_fallback_classification(self) -> Dict[str, Any]:
        """
        获取兜底分类
        
        Returns:
            兜底分类结果
        """
        default_category = self.categories_mapping[0]  # 使用第一个分类作为默认
        
        return {
            "category_name": default_category["category_name"],
            "category_description": default_category["category_description"],
            "sub_categories": default_category["sub_categories"][:2],
            "confidence": 50,
            "reasoning": "系统兜底分类",
            "fallback": True
        }
    
    async def batch_classify_papers(self, papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量分类论文
        
        Args:
            papers_data: 论文数据列表，每个元素包含title, abstract
            
        Returns:
            分类结果列表
        """
        results = []
        
        for paper_data in papers_data:
            try:
                result = await self.classify_paper(
                    title=paper_data.get("title", ""),
                    abstract=paper_data.get("abstract", "")
                )
                results.append(result)
                
                # 添加延迟避免请求过快
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"批量分类失败: {str(e)}")
                results.append(self._get_fallback_classification())
        
        return results
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        获取所有分类信息
        
        Returns:
            所有分类的列表
        """
        return self.categories_mapping
    
    def get_category_by_name(self, category_name: str) -> Optional[Dict[str, Any]]:
        """
        根据名称获取分类信息
        
        Args:
            category_name: 分类名称
            
        Returns:
            分类信息或None
        """
        for category in self.categories_mapping:
            if category["category_name"].lower() == category_name.lower():
                return category
        return None
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            服务是否正常
        """
        try:
            return await self.llm_engine.health_check()
        except Exception as e:
            logger.error(f"分类服务健康检查失败: {str(e)}")
            return False