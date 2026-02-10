"""
Local Insight Analysis Router - 本地引用论文洞察力分析 SSE 路由

提供流式 Server-Sent Events 接口，实时返回 4 个步骤的分析结果
"""

import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.services.local_insight_service import LocalInsightService

local_insight_service = LocalInsightService()
router = APIRouter(prefix="/api/local-insight", tags=["Local Insight Analysis"])


@router.post("/analyze/stream")
async def analyze_stream(
    file: UploadFile = File(...),
    references_dir: str = Form(...),
):
    """
    上传 PDF 并执行流式洞察力分析（SSE）

    - file: 待评估论文 PDF
    - references_dir: 引用论文文件夹路径（包含 [1].pdf, [2].pdf 等）

    返回 text/event-stream，事件类型:
    - progress: 进度更新
    - step1: 句子提取结果
    - step2: 观点句 + 证据
    - step3: 评分结果
    - step4: 洞察力报告
    - done: 分析完成
    - error: 错误
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件过大 (最大 50MB)")

    import os
    if not os.path.isdir(references_dir):
        raise HTTPException(status_code=400, detail=f"引用目录不存在: {references_dir}")

    logger.info(f"开始流式分析: {file.filename}, 引用目录: {references_dir}")

    async def event_generator():
        try:
            async for event in local_insight_service.analyze_paper_stream(
                content, file.filename, references_dir
            ):
                event_type = event.get("event", "message")
                event_data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
        except Exception as e:
            logger.error(f"SSE 流式分析异常: {e}")
            error_data = json.dumps({"message": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "local-insight-analysis",
        "version": "1.0.0",
    }
