"""
数据设置脚本
用于初始化 Milvus 集合并入库示例数据
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.paper_service import PaperService
from backend.services.arxiv_service import ArxivService
from backend.config import Config
from loguru import logger

async def setup_milvus_data():
    """设置 Milvus 数据"""
    
    logger.info("Starting Milvus data setup...")
    
    # 初始化服务
    milvus_service = PaperService(
        uri=Config.MILVUS_URI,
        token=Config.MILVUS_TOKEN,
        collection_name=Config.COLLECTION_NAME,
        embedding_model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_base_url=Config.OPENAI_BASE_URL,
        max_results=Config.ARXIV_MAX_RESULTS
    )
    
    arxiv_service = ArxivService(max_results=Config.ARXIV_MAX_RESULTS)
    
    # 步骤1: 初始化集合
    logger.info("Step 1: Initializing collection...")
    try:
        success = await milvus_service.initialize_collection(drop_existing=True)
        if success:
            logger.info("✓ Collection initialized successfully")
        else:
            logger.error("✗ Failed to initialize collection")
            return False
    except Exception as e:
        logger.error(f"✗ Error initializing collection: {e}")
        return False
    
    # 步骤2: 检查集合状态
    logger.info("Step 2: Checking collection stats...")
    try:
        stats = await milvus_service.get_collection_stats()
        if stats:
            logger.info(f"✓ Collection stats: {stats}")
            row_count = stats.get("row_count", 0)
            if row_count > 0:
                logger.info(f"Collection already has {row_count} papers")
        else:
            logger.warning("Could not get collection stats")
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
    
    # 步骤3: 入库最近7天的数据
    logger.info("Step 3: Ingesting recent papers (last 7 days)...")
    total_ingested = 0
    days_to_fetch = 30  # 获取最近7天的数据
    
    for days_ago in range(0,days_to_fetch):
        target_date = datetime.now() - timedelta(days=days_ago)
        date_str = target_date.strftime('%Y-%m-%d')
        logger.info(f"Fetching papers for {date_str}...")
        
        try:
            # 使用 ArxivService 获取指定日期的论文
            papers = await arxiv_service.get_papers(start_date=target_date)
            logger.info(f"Found {len(papers)} papers for {date_str}")
            
            if papers:
                # 使用 MilvusService 插入论文
                success = await milvus_service.insert_papers(papers)
                if success:
                    total_ingested += len(papers)
                    logger.info(f"✓ Inserted {len(papers)} papers for {date_str}")
                else:
                    logger.error(f"✗ Failed to insert papers for {date_str}")
            else:
                logger.info(f"No papers found for {date_str}")
                
        except Exception as e:
            logger.error(f"✗ Error processing papers for {date_str}: {e}")
    
    logger.info(f"✓ Total papers processed: {total_ingested}")
    
    # 步骤4: 验证数据
    logger.info("Step 4: Verifying data...")
    try:
        # 测试搜索
        results = await milvus_service.hybrid_search("information retrieval", limit=1)
        logger.info(f"✓ Search test returned {len(results)} results")
        
        # 测试元数据查询
        results = await milvus_service.query_by_metadata('id >= 0')
        logger.info(f"✓ Metadata query test returned {len(results)} results")
        
    except Exception as e:
        logger.error(f"✗ Error verifying data: {e}")
    
    logger.info("✓ Milvus data setup completed!")
    return True

async def main():
    """主函数"""
    logger.info("=== Milvus Data Setup Script ===")
    logger.info(f"Milvus URI: {Config.MILVUS_URI}")
    logger.info(f"Collection: {Config.COLLECTION_NAME}")
    logger.info(f"Embedding Model: {Config.EMBEDDING_MODEL}")
    
    try:
        success = await setup_milvus_data()
        if success:
            logger.info("🎉 Setup completed successfully!")
            logger.info("You can now use the Papers API endpoints.")
        else:
            logger.error("❌ Setup failed. Please check the logs above.")
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())