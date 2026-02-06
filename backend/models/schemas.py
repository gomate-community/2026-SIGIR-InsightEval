from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PaperResponse(BaseModel):
    id: str
    arxiv_id: str
    category: str
    title: str
    date: str
    views: str
    citations: int
    comments: Optional[int] = None
    score: int
    trending: Optional[int] = None
    # 新增的arxiv字段
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    arxiv_url: Optional[str] = None
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    primary_category: Optional[str] = None
    all_categories: Optional[List[str]] = None
    score_detail: Optional[Dict[str, Any]] = None
    category_detail: Optional[Dict[str, Any]] = None
    alphaxiv_detail: Optional[Dict[str, Any]] = None
    alphaxiv_overview: Optional[Dict[str, Any]] = None
    affiliation_detail: Optional[Dict[str, Any]] = None

class EngagementData(BaseModel):
    views: str
    likes: int
    retweets: int

class TrendingTopicResponse(BaseModel):
    id: int
    topic: str
    source: str
    category: str
    language: str
    summary: str
    engagement: EngagementData
    heatScore: int
    tags: List[str]

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

class PapersQueryParams(BaseModel):
    search: Optional[str] = None
    category: Optional[str] = None
    time_range: Optional[str] = "7days"
    page: int = 1
    page_size: int = 20

class PaginationInfo(BaseModel):
    current_page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_prev: bool

class PaginatedPapersResponse(BaseModel):
    papers: List[PaperResponse]
    pagination: PaginationInfo
    time_range: Optional[str] = "7days"
    limit: int = 20
    offset: int = 0

class TopicsQueryParams(BaseModel):
    search: Optional[str] = None
    source: Optional[str] = None
    time_range: Optional[str] = "1week"
    sort_by: Optional[str] = "default"
    limit: int = 20
    offset: int = 0


# Report related schemas
class BilingualText(BaseModel):
    zh: str
    en: str


class ReportPaper(BaseModel):
    id: int
    title: BilingualText
    authors: str
    institution: str
    arxivId: str
    highlight: BilingualText
    category: BilingualText
    abstract: BilingualText


class TrendingTopic(BaseModel):
    name: BilingualText
    count: int
    growth: str


class CategoryStatistics(BaseModel):
    """分类统计信息"""
    categoryName: BilingualText
    count: int
    percentage: float  # 百分比，保留2位小数


class WeeklyReport(BaseModel):
    id: str
    week: str
    dateRange: str
    publishDate: str
    totalPapers: int
    topicsCount: int
    title: BilingualText
    overview: BilingualText
    summary: BilingualText
    highlights: List[BilingualText]
    # 合并 ReportData 的字段
    trendingTopics: List[TrendingTopic]
    keyInsights: List[BilingualText]
    topPapers: List[ReportPaper]
    emergingKeywords: List[BilingualText]
    categoryStatistics: Optional[List[CategoryStatistics]] = None
    categorySummary: Optional[BilingualText] = None


class WeeklyReportsResponse(BaseModel):
    reports: List[WeeklyReport]