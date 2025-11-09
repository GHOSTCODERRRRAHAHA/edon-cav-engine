"""Memory management routes for Adaptive Memory Engine."""

from fastapi import APIRouter, HTTPException
from app.routes import cav
from typing import Dict

router = APIRouter(prefix="/memory", tags=["Memory"])

# Use shared memory engine instance from cav.py
memory_engine = cav.memory_engine


@router.get("/summary")
async def get_memory_summary() -> Dict:
    """
    Get 24-hour summary of memory statistics.
    
    Returns:
        Dictionary with:
        - total_records: Number of records in last 24h
        - window_hours: Rolling window size
        - hourly_stats: Per-hour statistics (mean, std, state probabilities)
        - overall_stats: Overall statistics (mean, std, state distribution)
    """
    try:
        summary = memory_engine.get_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memory summary: {str(e)}")


@router.post("/clear")
async def clear_memory() -> Dict:
    """
    Clear all memory (buffer and database).
    
    WARNING: This will delete all stored CAV history.
    Use for testing or resetting the adaptive engine.
    
    Returns:
        Confirmation message
    """
    try:
        memory_engine.clear()
        return {
            "status": "success",
            "message": "Memory cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

