"""
Insight Analysis Router - 论文洞察力分析 API 路由
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger

from backend.models.insight_schemas import (
    AnalysisRequest, AnalysisResponse
)
from backend.services.insight_service import InsightAnalysisService
insight_service = InsightAnalysisService()
router = APIRouter(prefix="/api/insight", tags=["Insight Analysis"])


@router.post("/analyze/pdf", response_model=AnalysisResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    """
    上传 PDF 并执行完整的洞察力分析
    
    流程:
    1. 解析 PDF (MinerU / PyPDF2)
    2. 提取 Introduction 部分
    3. LLM 分析观点句
    4. 生成洞察力报告
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    
    try:
        content = await file.read()
        
        if len(content) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="文件过大 (最大 50MB)")
        
        result = await insight_service.analyze_pdf(content, file.filename)
        return result
        
    except ValueError as e:
        logger.error(f"PDF 分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PDF 分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "insight-analysis",
        "version": "2.0.0"
    }
