"""
报告数据设置脚本
用于初始化 Milvus 报告集合并入库报告数据
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.report_service import ReportService
from backend.config import Config
from loguru import logger

async def setup_reports_data():
    """设置报告数据"""
    
    logger.info("Starting reports data setup...")
    
    # 初始化报告服务
    report_service = ReportService(
        uri=Config.MILVUS_URI,
        token=Config.MILVUS_TOKEN,
        collection_name="ir_reports",
        embedding_model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_base_url=Config.OPENAI_BASE_URL
    )
    
    # 步骤1: 初始化报告集合
    logger.info("Step 1: Initializing report collection...")
    try:
        success = await report_service.initialize_collection(
            drop_existing=True
        )
        if success:
            logger.info("✓ Report collection initialized successfully")
        else:
            logger.error("✗ Failed to initialize report collection")
            return False
    except Exception as e:
        logger.error(f"✗ Error initializing report collection: {e}")
        return False
    
    # 步骤2: 生成最近4周的周报
    logger.info("Step 2: Generating weekly reports (last 4 weeks)...")
    total_generated = 2
    
    try:
        # 生成最近2周的周报
        reports = await report_service.generate_weekly_reports(num_weeks=6)
        logger.info(f"Generated {len(reports)} weekly reports")
        
        if not reports:
            logger.warning("No reports generated")
            return False
        
        # 步骤3: 准备报告数据用于插入Milvus
        logger.info("Step 3: Preparing report data for insertion...")
        reports_data = []
        
        for report in reports:
            # 将 Pydantic 模型转换为字典
            report_dict = report.model_dump() if hasattr(report, 'model_dump') else report.dict()
            
            # 构建要插入的数据（现在 WeeklyReport 已经包含了所有字段）
            report_record = {
                "report_id": report_dict.get("id", ""),
                "week": report_dict.get("week", ""),
                "date_range": report_dict.get("dateRange", ""),
                "publish_date": report_dict.get("publishDate", ""),
                "weekly_report": report_dict  # 存储完整的 WeeklyReport（包含所有字段）
            }
            
            reports_data.append(report_record)
        
        # 步骤4: 插入报告数据到Milvus
        logger.info("Step 4: Inserting reports into Milvus...")
        inserted_count = await report_service.insert_reports(
            reports=reports_data
        )
        
        if inserted_count > 0:
            logger.info(f"✓ Inserted {inserted_count} reports into Milvus")
            total_generated = inserted_count
        else:
            logger.warning("No new reports inserted (may already exist)")
        
    except Exception as e:
        logger.error(f"✗ Error generating/inserting reports: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # 步骤5: 验证数据
    logger.info("Step 5: Verifying data...")
    try:
        # 测试查询所有报告
        all_reports = await report_service.query_reports()
        logger.info(f"✓ Query test returned {len(all_reports)} reports")
        
        # 测试按ID查询
        if reports_data:
            test_id = reports_data[0]["report_id"]
            test_report = await report_service.get_report_detail(
                report_id=test_id
            )
            if test_report:
                logger.info(f"✓ Get by ID test successful for report {test_id}")
            else:
                logger.warning(f"✗ Get by ID test failed for report {test_id}")
        
    except Exception as e:
        logger.error(f"✗ Error verifying data: {e}")
    
    logger.info(f"✓ Total reports processed: {total_generated}")
    logger.info("✓ Reports data setup completed!")
    return True

async def main():
    """主函数"""
    logger.info("=== Reports Data Setup Script ===")
    logger.info(f"Milvus URI: {Config.MILVUS_URI}")
    logger.info(f"Report Collection: ir_reports")
    
    try:
        success = await setup_reports_data()
        if success:
            logger.info("🎉 Setup completed successfully!")
            logger.info("You can now use the Reports API endpoints.")
        else:
            logger.error("❌ Setup failed. Please check the logs above.")
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())

