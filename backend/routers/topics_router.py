from fastapi import APIRouter, HTTPException
from typing import List, Optional

from backend.models.schemas import TrendingTopicResponse
from backend.services.mock_data import MockDataService
router = APIRouter(prefix="/api/topics", tags=["topics"])

# Initialize service
mock_service = MockDataService()


@router.get("", response_model=List[TrendingTopicResponse])
async def get_trending_topics(
        search: Optional[str] = None,
        source: Optional[str] = None,
        time_range: Optional[str] = "1week",
        sort_by: Optional[str] = "default",
        limit: int = 20,
        offset: int = 0
):
    """获取热门话题列表"""
    try:
        topics = await mock_service.get_trending_topics(
            search=search,
            source=source,
            time_range=time_range,
            sort_by=sort_by,
            limit=limit,
            offset=offset
        )
        return topics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))