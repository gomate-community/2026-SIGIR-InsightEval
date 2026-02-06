"""
Insight Analysis Data Models
论文洞察力分析相关的数据模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class SentenceType(str, Enum):
    """句子类型"""
    CONTEXT = "context"       # 背景信息
    CITATION = "citation"     # 引用文献
    VIEWPOINT = "viewpoint"   # 作者观点


class InsightLevel(str, Enum):
    """洞察力等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InsightScores(BaseModel):
    """三维洞察力评分"""
    synthesis: float = Field(..., ge=1.0, le=5.0, description="综合度评分 - 是否连接多个文献发现潜在联系")
    critical: float = Field(..., ge=1.0, le=5.0, description="批判距离评分 - 是否包含转折或局限性分析")
    abstraction: float = Field(..., ge=1.0, le=5.0, description="抽象层级评分 - 是否进行升维思考")



class Evidence(BaseModel):
    """支持证据"""
    quote: str = Field(..., description="证据原文")
    source: str = Field(..., description="证据来源（如论文标题或片段）")
    criteria: str = Field(..., description="匹配的证据标准")


class AnalyzedSentence(BaseModel):
    """分析后的句子"""
    id: int = Field(..., description="句子编号")
    text: str = Field(..., description="原始句子文本")
    type: SentenceType = Field(..., description="句子类型")
    insightLevel: InsightLevel = Field(..., description="洞察力等级")
    scores: InsightScores = Field(..., description="三维评分")
    analysis: str = Field(..., description="AI分析解释")
    source: Optional[str] = Field(None, description="引用来源（如果是citation类型）")
    evidence: Optional[List[Evidence]] = Field(default=[], description="支撑证据")


class InsightReport(BaseModel):
    """全局洞察力报告"""
    summary: str = Field(..., description="总体评价摘要")
    strengths: List[str] = Field(..., description="主要亮点")
    weaknesses: List[str] = Field(..., description="不足之处")
    overall_score: float = Field(..., description="总体评分")


class AnalysisRequest(BaseModel):
    """分析请求"""
    text: Optional[str] = Field(None, description="要分析的文本（Introduction部分）")
    file_path: Optional[str] = Field(None, description="PDF文件路径（后端内部使用）")
    paperTitle: Optional[str] = Field(None, description="论文标题")


class AnalysisResponse(BaseModel):
    """分析响应"""
    sentences: List[AnalyzedSentence] = Field(..., description="分析后的句子列表")
    overallScore: float = Field(..., description="总体评分")
    summary: str = Field(..., description="分析摘要")
    paperTitle: Optional[str] = Field(None, description="论文标题")
    report: Optional[InsightReport] = Field(None, description="详细洞察力报告")


class AnalysisProgress(BaseModel):
    """分析进度"""
    status: Literal["pending", "processing", "completed", "error"] = Field(..., description="分析状态")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="进度百分比")
    message: str = Field("", description="状态消息")
    taskId: Optional[str] = Field(None, description="任务ID")


class PDFUploadResponse(BaseModel):
    """PDF上传响应"""
    success: bool = Field(..., description="是否成功")
    filename: str = Field(..., description="文件名")
    extractedText: Optional[str] = Field(None, description="提取的文本")
    introText: Optional[str] = Field(None, description="提取的Introduction部分")
    tempFilePath: Optional[str] = Field(None, description="临时文件路径")
    message: str = Field(..., description="消息")


class BatchAnalysisRequest(BaseModel):
    """批量分析请求"""
    papers: List[AnalysisRequest] = Field(..., description="要分析的论文列表")


class BatchAnalysisResponse(BaseModel):
    """批量分析响应"""
    results: List[AnalysisResponse] = Field(..., description="分析结果列表")
    totalPapers: int = Field(..., description="总论文数")
    avgScore: float = Field(..., description="平均评分")


class ComparisonResult(BaseModel):
    """论文对比结果"""
    paper1: AnalysisResponse = Field(..., description="论文1分析结果")
    paper2: AnalysisResponse = Field(..., description="论文2分析结果")
    comparison: str = Field(..., description="对比分析")
    winner: Optional[str] = Field(None, description="更优论文")


class SentenceImprovement(BaseModel):
    """句子改进建议"""
    originalText: str = Field(..., description="原始句子")
    improvedText: str = Field(..., description="改进后的句子")
    explanation: str = Field(..., description="改进说明")
    expectedScoreIncrease: float = Field(..., description="预期评分提升")


class ImprovementSuggestions(BaseModel):
    """改进建议响应"""
    suggestions: List[SentenceImprovement] = Field(..., description="改进建议列表")
    overallAdvice: str = Field(..., description="总体建议")
