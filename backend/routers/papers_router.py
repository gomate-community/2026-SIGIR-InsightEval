from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger

from backend.models.schemas import PaperResponse, PaginatedPapersResponse, PapersQueryParams
from backend.services.paper_service import PaperService
from backend.services.alphaxiv_service import AlphaXivService
from backend.services.category_service import PaperCategoryService
from backend.config import Config

router = APIRouter(prefix="/api/papers", tags=["papers"])

paper_milvus_service = PaperService(
        uri=Config.MILVUS_URI,
        token=Config.MILVUS_TOKEN,
        collection_name=Config.COLLECTION_NAME,
        embedding_model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_base_url=Config.OPENAI_BASE_URL,
        max_results=Config.ARXIV_MAX_RESULTS
    )
alphaxiv_service = AlphaXivService()
category_service = PaperCategoryService()

def _build_category_filter(category: str) -> str:
    """构建分类过滤表达式"""
    # 旧的分类映射（保持兼容性）
    legacy_category_map = {
        "ai": "AI",
        "cv": "CV",
        "nlp": "NLP",
        "ro": "RO",
        "lg": "LG",
        "gn": "GN",
        "ir": "IR"
    }

    # 检查是否是旧的分类代码
    if category.lower() in legacy_category_map:
        target_category = legacy_category_map[category.lower()]
        return f'metadata["category"] == "{target_category}"'

    # 新的IR分类体系 - 支持基于 category_detail 的过滤
    # 可以匹配主分类或子分类
    # return f'(metadata["category_detail"]["category_name"] == "{category}")'
    return f'(category_detail["category_name"] == "{category}")'


def _build_time_filter(time_range: str) -> Optional[str]:
    """构建时间过滤表达式"""
    if time_range == "all":
        return None

    now = datetime.now()
    if time_range == "7days":
        cutoff_date = now - timedelta(days=7)
    elif time_range == "30days":
        cutoff_date = now - timedelta(days=30)
    else:
        return None

    # 使用metadata中的date字段进行过滤，格式为YYYY-MM-DD
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    return f'metadata["date"] >= "{cutoff_str}"'


def _convert_to_paper_response(result: Dict[str, Any]) -> PaperResponse:
    """将Milvus查询结果转换为PaperResponse（列表接口使用）"""
    # 适配新的嵌套结构：从 metadata 和 hits 中提取字段
    metadata = result.get("metadata", {})
    hits = result.get("hits", {})
    score_detail = result.get("score_detail", {})
    alphaxiv_detail = result.get("alphaxiv_detail", {})
    alphaxiv_overview = result.get("alphaxiv_overview", {})
    affiliation_detail = result.get("affiliation_detail", {})

    # 获取主键id：优先从id获取，如果没有则从entity_id获取
    entity_id = result.get("id") or result.get("entity_id")
    # logger.info(entity_id)
    if entity_id is None:
        raise ValueError("Missing entity id in result")

    # 计算score：优先从score_detail的overall_score获取，否则使用默认值
    if score_detail and isinstance(score_detail, dict) and "overall_score" in score_detail:
        score = int(score_detail.get("overall_score", 85))
    elif result.get("score"):
        score = int(result.get("score", 85))
    else:
        score = 85
    
    return PaperResponse(
        id=str(entity_id),
        arxiv_id=result.get("arxiv_id", "0"),
        category=metadata.get("category", ""),
        title=result.get("title", ""),
        date=metadata.get("date", ""),
        views=hits.get("views", ""),
        citations=hits.get("citations", 0),
        comments=hits.get("comments"),
        score=score,
        trending=hits.get("trending"),
        authors=metadata.get("authors", []),
        abstract=result.get("abstract", ""),
        pdf_url=metadata.get("pdf_url", ""),
        arxiv_url=metadata.get("arxiv_url", ""),
        doi=metadata.get("doi"),
        journal_ref=metadata.get("journal_ref"),
        primary_category=metadata.get("primary_category", ""),
        all_categories=metadata.get("all_categories", []),
        score_detail=score_detail if score_detail else None,
        category_detail=result.get("category_detail", {}),
        alphaxiv_detail=alphaxiv_detail ,
        alphaxiv_overview=alphaxiv_overview,
        affiliation_detail=affiliation_detail,
    )


@router.post("", response_model=PaginatedPapersResponse)
async def get_trending_papers(
        params: PapersQueryParams,
):
    """获取热门论文列表（支持分页）"""
    print(params)
    try:
        # 计算offset
        offset = (params.page - 1) * params.page_size

        # 如果有搜索查询，使用混合搜索
        if params.search and params.search.strip():
            logger.info(f"Performing hybrid search for query: {params.search}")

            # 构建过滤条件（与metadata查询相同的逻辑）
            filters = []

            if params.category and params.category != "all":
                filters.append(_build_category_filter(params.category))

            time_filter = _build_time_filter(params.time_range)
            if time_filter:
                filters.append(time_filter)

            # 组合过滤条件
            filter_expr = None
            if filters:
                filter_expr = " && ".join(filters)

            logger.info(f"Hybrid search filter expression: {filter_expr}")

            # 获取更多结果以便计算总数和支持分页
            all_results = await paper_milvus_service.hybrid_search(
                query=params.search.strip(),
                limit=1000,  # 获取足够多的结果
                filter_expr=filter_expr
            )
            # 计算总数
            total_count = len(all_results)
            # 应用分页
            results = all_results[offset:offset + params.page_size]
        else:
            # 无搜索查询，使用metadata查询
            logger.info("Performing metadata query")

            # 构建过滤条件
            filters = []

            if params.category and params.category != "all":
                filters.append(_build_category_filter(params.category))

            time_filter = _build_time_filter(params.time_range)
            if time_filter:
                filters.append(time_filter)

            # 组合过滤条件
            if filters:
                filter_expr = " && ".join(filters)
            else:
                filter_expr = "id >= 0"  # 获取所有记录的默认条件

            logger.info(f"Using filter expression: {filter_expr}")
            print(paper_milvus_service.collection_name)
            try:
                # 现在获取当前页的实际数据
                all_results = await paper_milvus_service.query_by_metadata(
                    filter_expr=filter_expr,
                )
            except Exception as e:
                logger.warning(f"Failed to get total count, falling back to estimation: {e}")
                # 如果获取总数失败，回退到原来的方法
                all_results = await paper_milvus_service.query_by_metadata(
                    filter_expr=filter_expr,
                )

            # 估算总数
            total_count = len(all_results)
            # 应用分页
            results = all_results[offset:offset + params.page_size]

        logger.info(total_count)
        # 转换为PaperResponse对象
        papers = [_convert_to_paper_response(result) for result in results]
        # logger.info(papers)
        # 计算分页信息
        total_pages = (total_count + params.page_size - 1) // params.page_size

        logger.info(f"Returning {len(papers)} papers, page {params.page}/{total_pages}, total: {total_count}")

        return {
            "papers": papers,
            "pagination": {
                "current_page": params.page,
                "page_size": params.page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": params.page < total_pages,
                "has_prev": params.page > 1
            }
        }

    except Exception as e:
        logger.error(f"Error in get_trending_papers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch papers: {str(e)}")


@router.get("/stats")
async def get_collection_stats(
):
    """获取论文集合统计信息"""
    try:
        stats = await paper_milvus_service.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/ingest")
async def ingest_daily_papers(
        target_date: Optional[str] = None,
):
    """手动触发论文数据入库"""
    try:
        # 解析日期
        if target_date:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        else:
            date_obj = datetime.now()

        # 执行数据入库
        count = await paper_milvus_service.ingest_daily_papers(target_date=date_obj)

        return {
            "message": f"Successfully ingested {count} papers",
            "date": date_obj.strftime("%Y-%m-%d"),
            "count": count
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Error ingesting papers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest papers: {str(e)}")


@router.post("/initialize")
async def initialize_collection(
        drop_existing: bool = False,
):
    """初始化Milvus集合"""
    try:
        success = await paper_milvus_service.initialize_collection(drop_existing=drop_existing)

        if success:
            return {"message": "Collection initialized successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize collection")

    except Exception as e:
        logger.error(f"Error initializing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize collection: {str(e)}")


@router.post("/detail/{paper_id}")
async def get_paper_detail(
        paper_id: str,
):
    """获取论文详情（通过主键id查询）"""
    try:
        # 将paper_id转换为整数（主键id）
        try:
            entity_id = int(paper_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid paper_id format: {paper_id}. Expected integer.")
        
        # 使用主键id从Milvus查询
        result = await paper_milvus_service.get_paper_detail(
            entity_id=entity_id,
            output_fields=["arxiv_id", "alphaxiv_detail"]
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Paper with id {paper_id} not found")
        
        alphaxiv_detail = result.get("alphaxiv_detail")
        arxiv_id = result.get("arxiv_id", "")
        
        # 如果Milvus中没有alphaxiv_detail或为空，从AlphaXiv API获取
        if not alphaxiv_detail or (isinstance(alphaxiv_detail, dict) and not alphaxiv_detail):
            logger.info(f"alphaxiv_detail not found in Milvus for id {paper_id}, fetching from AlphaXiv API")
            if not arxiv_id:
                raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found: missing arxiv_id")
            raw_data = await alphaxiv_service.get_paper_detail(arxiv_id)
            if not raw_data:
                raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

            # 格式化数据
            formatted_data = alphaxiv_service.format_paper_detail(raw_data)

            if not formatted_data:
                raise HTTPException(status_code=500, detail="Failed to format paper data")
            
            return formatted_data
        else:
            # 返回Milvus中的数据
            logger.info(f"Returning alphaxiv_detail from Milvus for {paper_id}")
            return alphaxiv_detail

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper detail for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get paper detail: {str(e)}")


@router.post("/scores/{paper_id}")
async def get_paper_scores(
        paper_id: str,
):
    """获取论文评分和分类详情（通过主键id查询）"""
    try:
        # 将paper_id转换为整数（主键id）
        try:
            entity_id = int(paper_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid paper_id format: {paper_id}. Expected integer.")
        
        # 使用主键id从Milvus查询 score_detail、category_detail 和 affiliation_detail
        result = await paper_milvus_service.get_paper_detail(
            entity_id=entity_id,
            output_fields=["score_detail", "category_detail", "affiliation_detail"]
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Paper with id {paper_id} not found")
        
        score_detail = result.get("score_detail")
        category_detail = result.get("category_detail")
        affiliation_detail = result.get("affiliation_detail")

        # 移除 timestamp 字段（如果存在）
        response_data = {}

        if score_detail and isinstance(score_detail, dict):
            score_data = {k: v for k, v in score_detail.items() if k != "timestamp"}
            if score_data:
                response_data["score_detail"] = score_data

        if category_detail and isinstance(category_detail, dict):
            category_data = {k: v for k, v in category_detail.items() if k != "timestamp"}
            if category_data:
                response_data["category_detail"] = category_data

        if affiliation_detail and isinstance(affiliation_detail, dict):
            affiliation_data = {k: v for k, v in affiliation_detail.items() if k != "timestamp"}
            if affiliation_data:
                response_data["affiliation_detail"] = affiliation_data
        
        if not response_data:
            raise HTTPException(status_code=404, detail=f"Score and category details not found for paper {paper_id}")
        
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper scores for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get paper scores: {str(e)}")


@router.get("/overview/{paper_version_id}")
async def get_paper_overview(
        paper_version_id: str,
):
    """获取论文概览"""
    try:
        # paper_version_id是UUID格式，无法直接在Milvus中高效查询
        # 先尝试从AlphaXiv API获取，如果失败再尝试其他方式
        # 注意：由于paper_version_id和arxiv_id的映射关系复杂，这里直接从API获取
        # 未来可以考虑在Milvus中建立paper_version_id到arxiv_id的索引以提高查询效率
        overview_data = await alphaxiv_service.get_paper_overview(paper_version_id)

        if not overview_data:
            raise HTTPException(status_code=404, detail=f"Paper overview {paper_version_id} not found")

        return overview_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper overview for {paper_version_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get paper overview: {str(e)}")


@router.get("/categories")
async def get_all_categories(
):
    """获取所有IR分类信息"""
    try:
        categories = category_service.get_all_categories()
        return {
            "categories": categories,
            "total_count": len(categories)
        }
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@router.post("/classify")
async def classify_paper(
        paper_data: Dict[str, str],
):
    """对论文进行分类"""
    try:
        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")

        if not title and not abstract:
            raise HTTPException(status_code=400, detail="Title or abstract is required")

        classification_result = await category_service.classify_paper(title, abstract)
        return classification_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying paper: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to classify paper: {str(e)}")


@router.get("/categories/{category_name}")
async def get_category_by_name(
        category_name: str,
):
    """根据名称获取特定分类信息"""
    try:
        category = category_service.get_category_by_name(category_name)

        if not category:
            raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")

        return category

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting category {category_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get category: {str(e)}")