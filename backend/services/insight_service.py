"""
Insight Analysis Service - 论文洞察力分析服务

核心流程:
1. 解析 PDF → 获取 Markdown
2. 提取句子 + 识别引用
3. LLM 筛选观点句 + 搜索证据 + 评分
4. 生成洞察力报告
"""

import re
import json
import tempfile
import os
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig
from backend.models.insight_schemas import (
    AnalyzedSentence, AnalysisResponse, InsightScores,
    SentenceType, InsightLevel, InsightReport, Evidence
)
from backend.prompts.insight_prompt import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    VIEWPOINT_EXTRACTION_PROMPT, SCORING_WITH_EVIDENCE_PROMPT,
    GLOBAL_REPORT_SYSTEM_PROMPT
)
from backend.services.mineru_service import MineruService
from backend.config import Config


class InsightAnalysisService:
    """论文洞察力分析服务"""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_engine = LLMEngine(llm_config)
        self.mineru_service = MineruService()
        # 可选：初始化向量搜索服务（用于证据检索）
        self._paper_service = None
        logger.info("InsightAnalysisService 初始化完成")

    @property
    def paper_service(self):
        """延迟加载 PaperService（避免启动时报错）"""
        if self._paper_service is None:
            try:
                from backend.services.paper_service import PaperService
                self._paper_service = PaperService(
                    uri=Config.MILVUS_URI,
                    token=Config.MILVUS_TOKEN,
                    collection_name=Config.COLLECTION_NAME,
                    embedding_model=Config.EMBEDDING_MODEL,
                    openai_api_key=Config.OPENAI_API_KEY,
                    openai_base_url=Config.OPENAI_BASE_URL,
                    max_results=3
                )
            except Exception as e:
                logger.warning(f"PaperService 初始化失败，证据检索将不可用: {e}")
        return self._paper_service

    # ==================== Step 1: 解析 PDF ====================

    def parse_pdf(self, content: bytes, filename: str) -> str:
        """
        解析 PDF 为 Markdown 文本
        
        优先使用 MinerU，失败时回退到 PyPDF2
        """
        # 尝试 MinerU
        if self.mineru_service.is_available:
            try:
                logger.info(f"使用 MinerU 解析 PDF: {filename}")
                markdown = self.mineru_service.parse_pdf_bytes_to_markdown(content, filename)
                if markdown and len(markdown) > 100:
                    return markdown
            except Exception as e:
                logger.warning(f"MinerU 解析失败: {e}")
        
        # Fallback: PyPDF2
        logger.info(f"使用 PyPDF2 解析 PDF: {filename}")
        return self._parse_with_pypdf2(content)

    def _parse_with_pypdf2(self, content: bytes) -> str:
        """使用 PyPDF2 提取文本（fallback）"""
        import PyPDF2
        import io
        
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PyPDF2 解析失败: {e}")
            raise ValueError(f"无法解析 PDF: {e}")

    # ==================== Step 2: 提取句子 ====================

    def extract_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本切分为句子，并标记引用
        
        Returns:
            List[{text: str, has_citation: bool}]
        """
        # 简单的句子分割（保留常见缩写）
        raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+', text)
        
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s or len(s) < 10:
                continue
            
            # 检测引用标记: [1], [1-3], (Author, 2020), (Author et al., 2020)
            has_citation = bool(re.search(
                r'\[\d+(?:[-–,]\d+)*\]|\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)',
                s
            ))
            
            sentences.append({
                "text": s,
                "has_citation": has_citation
            })
        
        return sentences

    def extract_introduction(self, text: str) -> str:
        """提取 Introduction 部分"""
        # 查找 Introduction 开始
        intro_match = re.search(
            r'(?i)(?:^|\n)#*\s*(?:1\.?\s*)?introduction\b',
            text
        )
        start = intro_match.end() if intro_match else 0
        
        # 查找下一个章节
        next_section = re.search(
            r'(?i)(?:^|\n)#*\s*(?:2\.?\s*)?(?:related\s+work|background|method|approach|preliminary)',
            text[start:]
        )
        end = start + next_section.start() if next_section else min(start + 5000, len(text))
        
        intro = text[start:end].strip()
        
        # 如果太短，取前 3000 字符
        if len(intro) < 100:
            return text[:3000]
        return intro

    # ==================== Step 3: 分析观点句 ====================

    async def analyze_viewpoints(
        self,
        sentences: List[Dict[str, Any]],
        paper_title: str
    ) -> List[AnalyzedSentence]:
        """
        使用 LLM 分析句子，筛选观点句，评分
        """
        if not sentences:
            return []
        
        analyzed = []
        
        # 分批处理（每批 10 句）
        batch_size = 10
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_text = " ".join([s["text"] for s in batch])
            
            try:
                # 调用 LLM 分析
                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=USER_PROMPT_TEMPLATE.format(
                        paper_title=paper_title,
                        text=batch_text
                    ))
                ]
                
                result = await self.llm_engine.chat_completion(messages=messages, temperature=0.1)
                batch_analysis = self._parse_llm_json(result.get("content", ""), as_list=True)
                
                # 处理每个句子
                for idx, s_data in enumerate(batch_analysis):
                    global_idx = i + idx
                    has_citation = batch[idx]["has_citation"] if idx < len(batch) else False
                    
                    sentence = self._build_analyzed_sentence(s_data, global_idx + 1, has_citation)
                    
                    # 对观点句搜索证据并重新评分
                    if sentence.type == SentenceType.VIEWPOINT:
                        evidence = await self._search_evidence(sentence.text)
                        if evidence:
                            sentence.evidence = evidence
                            scores, analysis, level = await self._score_with_evidence(
                                sentence.text, evidence
                            )
                            sentence.scores = scores
                            sentence.analysis = analysis
                            sentence.insightLevel = level
                            sentence.source = f"[{evidence[0].source}]"
                    
                    analyzed.append(sentence)
                    
            except Exception as e:
                logger.error(f"批次 {i} 分析失败: {e}")
                # 添加默认分析结果
                for idx, s in enumerate(batch):
                    analyzed.append(AnalyzedSentence(
                        id=i + idx + 1,
                        text=s["text"],
                        type=SentenceType.CITATION if s["has_citation"] else SentenceType.CONTEXT,
                        insightLevel=InsightLevel.LOW,
                        scores=InsightScores(synthesis=1.0, critical=1.0, abstraction=1.0),
                        analysis="分析失败",
                        evidence=[]
                    ))
        
        return analyzed

    async def _search_evidence(self, viewpoint: str) -> List[Evidence]:
        """搜索支撑材料"""
        if not self.paper_service:
            return []
        
        try:
            results = await self.paper_service.hybrid_search(query=viewpoint, limit=3)
            return [
                Evidence(
                    quote=r.get("abstract", "")[:200] + "..." if r.get("abstract") else "",
                    source=r.get("title", "Unknown"),
                    criteria="Semantic Match"
                )
                for r in results if r.get("abstract")
            ]
        except Exception as e:
            logger.warning(f"证据搜索失败: {e}")
            return []

    async def _score_with_evidence(
        self,
        viewpoint: str,
        evidence: List[Evidence]
    ) -> Tuple[InsightScores, str, InsightLevel]:
        """基于证据对观点句评分"""
        evidence_text = "\n".join([f"- [{e.source}]: {e.quote}" for e in evidence])
        
        messages = [
            ChatMessage(role="user", content=SCORING_WITH_EVIDENCE_PROMPT.format(
                viewpoint=viewpoint,
                evidence=evidence_text or "No direct evidence found."
            ))
        ]
        
        try:
            result = await self.llm_engine.chat_completion(messages=messages, temperature=0.2)
            data = self._parse_llm_json(result.get("content", ""))
            
            scores_data = data.get("scores", {})
            scores = InsightScores(
                synthesis=float(scores_data.get("synthesis", 2.0)),
                critical=float(scores_data.get("critical", 2.0)),
                abstraction=float(scores_data.get("abstraction", 2.0))
            )
            
            analysis = data.get("analysis", "基于证据评估")
            level = InsightLevel(data.get("insightLevel", "medium").lower())
            
            return scores, analysis, level
            
        except Exception as e:
            logger.error(f"评分失败: {e}")
            return (
                InsightScores(synthesis=2.0, critical=2.0, abstraction=2.0),
                "评分失败",
                InsightLevel.MEDIUM
            )

    # ==================== Step 4: 生成报告 ====================

    async def generate_report(
        self,
        viewpoints: List[AnalyzedSentence],
        introduction: str,
        title: str
    ) -> InsightReport:
        """生成全局洞察力报告"""
        # 准备观点摘要
        viewpoint_data = [
            {
                "text": v.text[:100] + "..." if len(v.text) > 100 else v.text,
                "scores": v.scores.model_dump(),
                "level": v.insightLevel.value
            }
            for v in viewpoints
            if v.type == SentenceType.VIEWPOINT
        ][:10]  # 限制数量
        
        messages = [
            ChatMessage(role="user", content=GLOBAL_REPORT_SYSTEM_PROMPT.format(
                title=title,
                viewpoints=json.dumps(viewpoint_data, ensure_ascii=False)
            ) + f"\n\nIntroduction:\n{introduction[:2000]}...")
        ]
        
        try:
            result = await self.llm_engine.chat_completion(messages=messages, temperature=0.4)
            data = self._parse_llm_json(result.get("content", ""))
            
            return InsightReport(
                summary=data.get("summary", "分析完成"),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                overall_score=float(data.get("overall_score", 5.0))
            )
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return InsightReport(
                summary="报告生成失败",
                strengths=[],
                weaknesses=[],
                overall_score=0.0
            )

    # ==================== 主入口方法 ====================

    async def analyze_pdf(self, content: bytes, filename: str) -> AnalysisResponse:
        """
        分析 PDF 论文
        
        完整流程: 解析 → 提取句子 → 分析观点 → 生成报告
        """
        logger.info(f"开始分析 PDF: {filename}")
        
        # Step 1: 解析 PDF
        full_text = self.parse_pdf(content, filename)
        if not full_text:
            raise ValueError("PDF 解析失败")
        
        # Step 2: 提取 Introduction 和句子
        introduction = self.extract_introduction(full_text)
        sentences = self.extract_sentences(introduction)
        
        logger.info(f"提取到 {len(sentences)} 个句子")
        
        # Step 3: 分析观点句
        analyzed = await self.analyze_viewpoints(sentences, filename)
        
        # Step 4: 生成报告
        report = await self.generate_report(analyzed, introduction, filename)
        
        return AnalysisResponse(
            sentences=analyzed,
            overallScore=report.overall_score,
            summary=report.summary,
            paperTitle=filename.replace(".pdf", ""),
            report=report
        )


    # ==================== 辅助方法 ====================

    def _parse_llm_json(self, content: str, as_list: bool = False) -> Any:
        """解析 LLM 返回的 JSON"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON
        pattern = r'\[[\s\S]*\]' if as_list else r'\{[\s\S]*\}'
        match = re.search(pattern, content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return [] if as_list else {}

    def _build_analyzed_sentence(
        self,
        data: Dict[str, Any],
        idx: int,
        has_citation: bool
    ) -> AnalyzedSentence:
        """构建 AnalyzedSentence 对象"""
        scores_data = data.get("scores", {})
        scores = InsightScores(
            synthesis=max(1.0, min(5.0, float(scores_data.get("synthesis", 2.0)))),
            critical=max(1.0, min(5.0, float(scores_data.get("critical", 2.0)))),
            abstraction=max(1.0, min(5.0, float(scores_data.get("abstraction", 2.0))))
        )
        
        # 确定类型
        type_str = data.get("type", "context").lower()
        if type_str not in ["context", "citation", "viewpoint"]:
            type_str = "context"
        
        # 如果有引用但 LLM 说是 context，改为 citation
        if has_citation and type_str == "context":
            type_str = "citation"
        
        # 计算洞察力等级
        avg = (scores.synthesis + scores.critical + scores.abstraction) / 3
        if avg < 2.5:
            level = InsightLevel.LOW
        elif avg <= 4.0:
            level = InsightLevel.MEDIUM
        else:
            level = InsightLevel.HIGH
        
        return AnalyzedSentence(
            id=data.get("id", idx),
            text=data.get("text", ""),
            type=SentenceType(type_str),
            insightLevel=level,
            scores=scores,
            analysis=data.get("analysis", ""),
            source=data.get("source"),
            evidence=[]
        )
