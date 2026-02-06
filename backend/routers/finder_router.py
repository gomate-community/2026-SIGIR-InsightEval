from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, AsyncGenerator
import json
from loguru import logger

from backend.services.find_service import PaperFindService
from backend.config import Config

router = APIRouter(prefix="/api/finder", tags=["finder"])

# 初始化查找服务
finder_service = PaperFindService()


async def generate_search_events(query: str) -> AsyncGenerator[str, None]:
    """
    生成搜索事件的流式响应

    Args:
        query: 搜索查询

    Yields:
        SSE格式的事件字符串
    """
    try:
        async for event in finder_service.find_papers(query):
            # 将事件转换为SSE格式
            event_data = json.dumps(event, ensure_ascii=False)
            yield f"data: {event_data}\n\n"

    except Exception as e:
        logger.error(f"流式搜索失败: {str(e)}")
        error_event = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
        yield f"data: {error_event}\n\n"


# @router.on_event("startup")
# async def startup_event():
#     """启动时初始化服务"""
#     try:
#         # 初始化MinIO存储桶
#         if hasattr(finder_service.affiliation_service, 'minio_service'):
#             await finder_service.affiliation_service.minio_service._ensure_default_bucket()
#             logger.info("✅ MinIO存储桶初始化完成")
#     except Exception as e:
#         logger.error(f"❌ 服务初始化失败: {str(e)}")


@router.post("/search")
async def find_papers_stream(request: Dict[str, Any]):
    """
    论文查找接口 - 支持Finder页面的智能搜索功能（流式返回）

    请求体格式:
    {
        "query": "用户查询字符串"
    }

    返回格式: Server-Sent Events 流
    """
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空")

    logger.info(f"收到Finder搜索请求: {query}")

    # 返回流式响应
    return StreamingResponse(
        generate_search_events(query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )
    """
    论文查找接口 - 支持Finder页面的智能搜索功能（流式返回）

    请求体格式:
    {
        "query": "用户查询字符串"
    }

    返回格式: Server-Sent Events 流
    事件类型:
    - {"type": "step", "step": "analyzing_intent", "message": "..."}
    - {"type": "intent_analyzed", "data": {...}}
    - {"type": "step", "step": "searching_papers", "message": "..."}
    - {"type": "papers_found", "count": 10}
    - {"type": "step", "step": "evaluating_papers", "message": "..."}
    - {"type": "papers_evaluated", "count": 10, "relevant_count": 5}
    - {"type": "step", "step": "extracting_evidence", "message": "..."}
    - {"type": "evidence_extracted"}
    - {"type": "complete", "data": {...}}
    - {"type": "error", "message": "..."}
    """
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="查询不能为空")

        logger.info(f"收到Finder搜索请求: {query}")

        # 返回流式响应
        return StreamingResponse(
            generate_search_events(query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Finder搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.get("/health")
async def finder_health_check():
    """
    Finder服务健康检查
    """
    try:
        # 这里可以添加具体的健康检查逻辑
        return {
            "status": "healthy",
            "service": "finder",
            "message": "Finder服务正常运行"
        }
    except Exception as e:
        logger.error(f"Finder健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "finder",
            "error": str(e)
        }
