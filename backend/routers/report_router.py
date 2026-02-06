"""Report router for weekly reports API"""
from fastapi import APIRouter, HTTPException, Query
from typing import List
from loguru import logger

from backend.models.schemas import WeeklyReport, WeeklyReportsResponse
from backend.services.report_service import ReportService
report_service=ReportService()
router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/weekly", response_model=WeeklyReportsResponse)
async def get_all_weekly_reports():
    """Get all weekly reports"""
    try:
        reports = await report_service.get_all_weekly_reports()
        return WeeklyReportsResponse(reports=reports)
    except Exception as e:
        logger.error(f"Error in get_all_weekly_reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weekly reports: {str(e)}")


@router.get("/weekly/generate", response_model=WeeklyReportsResponse)
async def generate_weekly_reports(num_weeks: int = Query(1, ge=1, le=52, description="Number of weeks to generate (1-52)")):
    """
    Generate weekly reports for specified number of weeks

    Args:
        num_weeks: Number of weeks to generate (1 = current week, 2 = current + last week, etc.)

    Returns:
        List of weekly reports
    """
    try:
        reports = await report_service.generate_weekly_reports(num_weeks=num_weeks)
        return WeeklyReportsResponse(reports=reports)
    except Exception as e:
        logger.error(f"Error in generate_weekly_reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate weekly reports: {str(e)}")


@router.get("/weekly/{report_id}", response_model=WeeklyReport)
async def get_weekly_report(report_id: str):
    """Get a weekly report by ID"""
    try:
        report = await report_service.get_report_detail(report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Weekly report {report_id} not found")
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_weekly_report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weekly report: {str(e)}")

