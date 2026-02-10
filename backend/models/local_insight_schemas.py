"""
Local Insight Analysis Models - SSE 事件模型

用于流式返回本地引用分析结果
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any


class SSEEvent(BaseModel):
    """Server-Sent Event 数据模型"""
    event: str = Field(..., description="事件类型: step1, step2, step3, step4, error, done")
    data: Any = Field(..., description="事件数据（JSON 序列化）")


class Step1Result(BaseModel):
    """Step 1: 句子提取结果"""
    total_sentences: int = Field(..., description="总句子数")
    cited_sentences: int = Field(..., description="带引用的句子数")
    sentences: List[dict] = Field(..., description="句子列表 [{text, has_citation, citation_numbers}]")
    introduction: str = Field("", description="提取的 Introduction 文本")


class ViewpointWithEvidence(BaseModel):
    """观点句 + 证据"""
    id: int = Field(..., description="句子编号")
    text: str = Field(..., description="观点句文本")
    citation_numbers: List[int] = Field(default=[], description="引用编号列表")
    evidence: List[dict] = Field(default=[], description="从引用论文提取的证据")
    analysis: str = Field("", description="分类分析说明")


class Step2Result(BaseModel):
    """Step 2: 观点句 + 证据提取结果"""
    total_viewpoints: int = Field(..., description="观点句总数")
    viewpoints: List[ViewpointWithEvidence] = Field(..., description="观点句列表")


class ScoredViewpoint(BaseModel):
    """评分后的观点句"""
    id: int = Field(..., description="句子编号")
    text: str = Field(..., description="观点句文本")
    scores: dict = Field(..., description="三维评分 {synthesis, critical, abstraction}")
    analysis: str = Field("", description="评分分析")
    insight_level: str = Field("medium", description="洞察力等级")
    evidence: List[dict] = Field(default=[], description="支撑证据")


class Step3Result(BaseModel):
    """Step 3: 评分结果"""
    scored_viewpoints: List[ScoredViewpoint] = Field(..., description="评分后的观点句")
    avg_score: float = Field(0.0, description="平均分")


class Step4Result(BaseModel):
    """Step 4: 洞察力报告"""
    summary: str = Field(..., description="总体评价")
    strengths: List[str] = Field(default=[], description="亮点")
    weaknesses: List[str] = Field(default=[], description="不足")
    overall_score: float = Field(0.0, description="总体评分")
