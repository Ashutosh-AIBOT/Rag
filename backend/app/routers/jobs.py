from fastapi import APIRouter, HTTPException
from app.database.database import get_job_status, update_job_status
import json

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.get("/{job_id}")
async def get_status(job_id: str):
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Parse JSON result if job completed successfully
    result_data = None
    if job["result"]:
        try:
            result_data = json.loads(job["result"])
        except Exception:
            result_data = job["result"]
            
    return {
        "job_id": job["id"],
        "type": job["type"],
        "status": job["status"],
        "progress": job["progress"],
        "result": result_data,
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"]
    }

@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] in ["completed", "failed", "cancelled"]:
        return {"message": f"Job is already in terminal state: {job['status']}"}
        
    update_job_status(job_id, "cancelled", job["progress"])
    return {"message": "Job cancellation requested"}
