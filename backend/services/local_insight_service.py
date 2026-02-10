"""
Local Insight Analysis Service - 本地引用论文洞察力分析服务

核心流程:
1. 解析 PDF → 提取句子 + 引用标记
2. LLM 筛选观点句 → 解析本地引用 PDF → LLM 提取证据
3. 基于证据对观点句评分
4. 生成洞察力报告

与 insight_service.py 的区别:
- 证据来源: 本地 references/ 文件夹中的 [x].pdf，而非在线搜索
- 输出方式: 流式 SSE 事件，而非一次性返回
"""

import re
import json
import os
from typing import List, Optional, Dict, Any, AsyncGenerator
from loguru import logger

from backend.engines.llm_engine import LLMEngine, ChatMessage, LLMConfig
from backend.models.insight_schemas import (
    InsightScores, InsightLevel, InsightReport, Evidence
)
from backend.models.local_insight_schemas import (
    Step1Result, Step2Result, Step3Result, Step4Result,
    ViewpointWithEvidence, ScoredViewpoint,
)
from backend.prompts.local_insight_prompt import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    VIEWPOINT_FILTER_PROMPT, EVIDENCE_EXTRACTION_PROMPT,
    SCORING_WITH_EVIDENCE_PROMPT, GLOBAL_REPORT_SYSTEM_PROMPT,
)
from backend.services.mineru_service import MineruService


class LocalInsightService:
    """本地引用论文洞察力分析服务"""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_engine = LLMEngine(llm_config)
        self.mineru_service = MineruService()
        logger.info("LocalInsightService 初始化完成")

    # ==================== Step 1: 解析 PDF + 提取句子 ====================

    def parse_pdf(self, content: bytes, filename: str) -> str:
        """解析 PDF 为文本（优先 MinerU，回退 PyPDF2）"""
        if self.mineru_service.is_available:
            try:
                logger.info(f"使用 MinerU 解析 PDF: {filename}")
                markdown = self.mineru_service.parse_pdf_bytes_to_markdown(content, filename)
                if markdown and len(markdown) > 100:
                    return markdown
            except Exception as e:
                logger.warning(f"MinerU 解析失败: {e}")

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

    def parse_pdf_file(self, file_path: str) -> str:
        """从文件路径解析 PDF"""
        with open(file_path, "rb") as f:
            content = f.read()
        return self.parse_pdf(content, os.path.basename(file_path))

    def extract_sentences(self, text: str) -> List[Dict[str, Any]]:
        """切分句子并标记引用"""
        raw_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+', text)

        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s or len(s) < 10:
                continue

            # 检测引用标记: [1], [1-3], [1,2,3], (Author, 2020)
            has_citation = bool(re.search(
                r'\[\d+(?:[-–,]\s*\d+)*\]|\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)',
                s
            ))

            # 提取引用编号
            citation_numbers = self._extract_citation_numbers(s)

            sentences.append({
                "text": s,
                "has_citation": has_citation,
                "citation_numbers": citation_numbers,
            })

        return sentences

    def _extract_citation_numbers(self, text: str) -> List[int]:
        """从文本中提取引用编号，如 [1], [2-5], [1,3,5]"""
        numbers = set()

        # 匹配 [1], [1,2], [1-3] 等模式
        for match in re.finditer(r'\[([\d,\-–\s]+)\]', text):
            inner = match.group(1)
            # 处理范围: 1-3 => 1,2,3
            for part in re.split(r'[,\s]+', inner):
                part = part.strip()
                if '-' in part or '–' in part:
                    bounds = re.split(r'[-–]', part)
                    if len(bounds) == 2:
                        try:
                            start, end = int(bounds[0].strip()), int(bounds[1].strip())
                            numbers.update(range(start, end + 1))
                        except ValueError:
                            pass
                else:
                    try:
                        numbers.add(int(part))
                    except ValueError:
                        pass

        return sorted(numbers)

    def extract_introduction(self, text: str) -> str:
        """提取 Introduction 部分"""
        intro_match = re.search(
            r'(?i)(?:^|\n)#*\s*(?:1\.?\s*)?introduction\b',
            text
        )
        start = intro_match.end() if intro_match else 0

        next_section = re.search(
            r'(?i)(?:^|\n)#*\s*(?:2\.?\s*)?(?:related\s+work|background|method|approach|preliminary)',
            text[start:]
        )
        end = start + next_section.start() if next_section else min(start + 5000, len(text))

        intro = text[start:end].strip()
        if len(intro) < 100:
            return text[:3000]
        return intro

    # ==================== Step 2: 筛选观点句 + 提取证据 ====================

    async def filter_viewpoints(
        self,
        sentences: List[Dict[str, Any]],
        paper_title: str
    ) -> List[Dict[str, Any]]:
        """使用 LLM 筛选观点句"""
        if not sentences:
            return []

        all_results = []
        batch_size = 15

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            sentences_json = json.dumps([
                {"id": i + idx + 1, "text": s["text"], "has_citation": s["has_citation"]}
                for idx, s in enumerate(batch)
            ], ensure_ascii=False)

            try:
                messages = [
                    ChatMessage(role="user", content=VIEWPOINT_FILTER_PROMPT.format(
                        paper_title=paper_title,
                        sentences_json=sentences_json
                    ))
                ]

                result = await self.llm_engine.chat_completion(messages=messages, temperature=0.1)
                batch_results = self._parse_llm_json(result.get("content", ""), as_list=True)

                for item in batch_results:
                    global_idx = i + (item.get("id", 1) - 1)
                    if global_idx < len(sentences):
                        item["original_citation_numbers"] = sentences[global_idx].get("citation_numbers", [])
                        if not item.get("citation_numbers"):
                            item["citation_numbers"] = item["original_citation_numbers"]
                    all_results.append(item)

            except Exception as e:
                logger.error(f"批次 {i} 筛选失败: {e}")
                for idx, s in enumerate(batch):
                    all_results.append({
                        "id": i + idx + 1,
                        "text": s["text"],
                        "type": "citation" if s["has_citation"] else "context",
                        "citation_numbers": s.get("citation_numbers", []),
                        "scores": {"synthesis": 1.0, "critical": 1.0, "abstraction": 1.0},
                        "analysis": "分析失败",
                    })

        return all_results

    async def extract_evidence_from_references(
        self,
        viewpoints: List[Dict[str, Any]],
        references_dir: str,
    ) -> List[ViewpointWithEvidence]:
        """对每个观点句，从对应引用 PDF 中提取证据"""
        results = []

        for vp in viewpoints:
            if vp.get("type") != "viewpoint":
                continue

            vp_id = vp.get("id", 0)
            vp_text = vp.get("text", "")
            citation_nums = vp.get("citation_numbers", []) or vp.get("original_citation_numbers", [])
            all_evidence = []

            # 对每个引用编号，尝试解析对应的 PDF
            for ref_num in citation_nums[:3]:  # 最多处理 3 个引用
                ref_filename = f"[{ref_num}].pdf"
                ref_path = os.path.join(references_dir, ref_filename)

                if not os.path.exists(ref_path):
                    logger.warning(f"引用文件不存在: {ref_path}")
                    continue

                try:
                    # 解析引用 PDF
                    ref_text = self.parse_pdf_file(ref_path)
                    if not ref_text or len(ref_text) < 50:
                        continue

                    # 截取前 3000 字符避免 token 过长
                    ref_text_truncated = ref_text[:3000]

                    # LLM 提取证据
                    evidence_items = await self._extract_evidence_llm(
                        vp_text, ref_text_truncated, ref_filename
                    )
                    for ev in evidence_items:
                        ev["source"] = ref_filename
                    all_evidence.extend(evidence_items)

                except Exception as e:
                    logger.error(f"处理引用 {ref_filename} 失败: {e}")

            results.append(ViewpointWithEvidence(
                id=vp_id,
                text=vp_text,
                citation_numbers=citation_nums,
                evidence=all_evidence,
                analysis=vp.get("analysis", ""),
            ))

        return results

    async def _extract_evidence_llm(
        self, viewpoint: str, ref_text: str, ref_name: str
    ) -> List[Dict[str, Any]]:
        """LLM 从引用论文中提取证据"""
        messages = [
            ChatMessage(role="user", content=EVIDENCE_EXTRACTION_PROMPT.format(
                viewpoint=viewpoint,
                ref_text=ref_text,
                ref_name=ref_name,
            ))
        ]

        try:
            result = await self.llm_engine.chat_completion(messages=messages, temperature=0.1)
            evidence = self._parse_llm_json(result.get("content", ""), as_list=True)
            return evidence if isinstance(evidence, list) else []
        except Exception as e:
            logger.error(f"证据提取失败: {e}")
            return []

    # ==================== Step 3: 评分 ====================

    async def score_viewpoints(
        self, viewpoints: List[ViewpointWithEvidence]
    ) -> List[ScoredViewpoint]:
        """基于证据对观点句评分"""
        scored = []

        for vp in viewpoints:
            evidence_text = "\n".join([
                f"- [{ev.get('source', 'Unknown')}]: {ev.get('quote', '')}"
                for ev in vp.evidence
            ]) if vp.evidence else "No direct evidence found from reference papers."

            messages = [
                ChatMessage(role="user", content=SCORING_WITH_EVIDENCE_PROMPT.format(
                    viewpoint=vp.text,
                    evidence=evidence_text,
                ))
            ]

            try:
                result = await self.llm_engine.chat_completion(messages=messages, temperature=0.2)
                data = self._parse_llm_json(result.get("content", ""))

                scores_data = data.get("scores", {})
                scores = {
                    "synthesis": max(1.0, min(5.0, float(scores_data.get("synthesis", 2.0)))),
                    "critical": max(1.0, min(5.0, float(scores_data.get("critical", 2.0)))),
                    "abstraction": max(1.0, min(5.0, float(scores_data.get("abstraction", 2.0)))),
                }
                avg = (scores["synthesis"] + scores["critical"] + scores["abstraction"]) / 3
                level = "low" if avg < 2.5 else ("high" if avg > 4.0 else "medium")

                scored.append(ScoredViewpoint(
                    id=vp.id,
                    text=vp.text,
                    scores=scores,
                    analysis=data.get("analysis", "基于证据评估"),
                    insight_level=level,
                    evidence=[ev for ev in vp.evidence],
                ))

            except Exception as e:
                logger.error(f"评分失败 (viewpoint {vp.id}): {e}")
                scored.append(ScoredViewpoint(
                    id=vp.id,
                    text=vp.text,
                    scores={"synthesis": 2.0, "critical": 2.0, "abstraction": 2.0},
                    analysis="评分失败",
                    insight_level="medium",
                    evidence=[ev for ev in vp.evidence],
                ))

        return scored

    # ==================== Step 4: 生成报告 ====================

    async def generate_report(
        self,
        scored_viewpoints: List[ScoredViewpoint],
        introduction: str,
        title: str,
    ) -> Step4Result:
        """生成全局洞察力报告"""
        viewpoint_data = [
            {
                "text": sv.text[:100] + "..." if len(sv.text) > 100 else sv.text,
                "scores": sv.scores,
                "level": sv.insight_level,
            }
            for sv in scored_viewpoints
        ][:10]

        messages = [
            ChatMessage(role="user", content=GLOBAL_REPORT_SYSTEM_PROMPT.format(
                title=title,
                viewpoints=json.dumps(viewpoint_data, ensure_ascii=False),
            ) + f"\n\nIntroduction:\n{introduction[:2000]}...")
        ]

        try:
            result = await self.llm_engine.chat_completion(messages=messages, temperature=0.4)
            data = self._parse_llm_json(result.get("content", ""))

            return Step4Result(
                summary=data.get("summary", "分析完成"),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                overall_score=float(data.get("overall_score", 5.0)),
            )

        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return Step4Result(
                summary="报告生成失败",
                strengths=[],
                weaknesses=[],
                overall_score=0.0,
            )

    # ==================== 主入口: 流式分析 ====================

    async def analyze_paper_stream(
        self,
        paper_content: bytes,
        filename: str,
        references_dir: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式分析论文, yield SSE 事件

        Yields:
            {"event": "step1|step2|step3|step4|done|error", "data": {...}}
        """
        logger.info(f"开始流式分析: {filename}, 引用目录: {references_dir}")

        try:
            # === Step 1: 解析 PDF + 提取句子 ===
            yield {"event": "progress", "data": {"step": 1, "message": "正在解析 PDF 文件..."}}

            full_text = self.parse_pdf(paper_content, filename)
            if not full_text:
                yield {"event": "error", "data": {"message": "PDF 解析失败"}}
                return

            introduction = self.extract_introduction(full_text)
            sentences = self.extract_sentences(introduction)

            step1 = Step1Result(
                total_sentences=len(sentences),
                cited_sentences=sum(1 for s in sentences if s["has_citation"]),
                sentences=sentences,
                introduction=introduction[:500],
            )

            yield {"event": "step1", "data": step1.model_dump()}
            logger.info(f"Step 1 完成: {len(sentences)} 个句子")

            # === Step 2: 筛选观点句 + 提取证据 ===
            yield {"event": "progress", "data": {"step": 2, "message": "正在筛选观点句并提取证据..."}}

            filtered = await self.filter_viewpoints(sentences, filename)
            viewpoints = await self.extract_evidence_from_references(filtered, references_dir)

            step2 = Step2Result(
                total_viewpoints=len(viewpoints),
                viewpoints=viewpoints,
            )

            yield {"event": "step2", "data": step2.model_dump()}
            logger.info(f"Step 2 完成: {len(viewpoints)} 个观点句")

            # === Step 3: 评分 ===
            yield {"event": "progress", "data": {"step": 3, "message": "正在对观点句进行评分..."}}

            scored = await self.score_viewpoints(viewpoints)

            avg_score = 0.0
            if scored:
                total = sum(
                    (sv.scores["synthesis"] + sv.scores["critical"] + sv.scores["abstraction"]) / 3
                    for sv in scored
                )
                avg_score = round(total / len(scored), 2)

            step3 = Step3Result(
                scored_viewpoints=scored,
                avg_score=avg_score,
            )

            yield {"event": "step3", "data": step3.model_dump()}
            logger.info(f"Step 3 完成: 平均分 {avg_score}")

            # === Step 4: 生成报告 ===
            yield {"event": "progress", "data": {"step": 4, "message": "正在生成洞察力报告..."}}

            report = await self.generate_report(scored, introduction, filename)

            yield {"event": "step4", "data": report.model_dump()}
            logger.info(f"Step 4 完成: 总评分 {report.overall_score}")

            # === 完成 ===
            yield {"event": "done", "data": {"message": "分析完成", "overall_score": report.overall_score}}

        except Exception as e:
            logger.error(f"流式分析失败: {e}")
            yield {"event": "error", "data": {"message": str(e)}}

    # ==================== 辅助方法 ====================

    def _parse_llm_json(self, content: str, as_list: bool = False) -> Any:
        """解析 LLM 返回的 JSON"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        pattern = r'\[[\s\S]*\]' if as_list else r'\{[\s\S]*\}'
        match = re.search(pattern, content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return [] if as_list else {}
