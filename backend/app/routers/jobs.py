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


from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import redis
from app.config import settings

@router.websocket("/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = get_job_status(job_id)
    if not job:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    # If already in terminal state, return and close
    if job["status"] in ["completed", "failed", "cancelled"]:
        result_data = None
        if job["result"]:
            try:
                result_data = json.loads(job["result"])
            except Exception:
                result_data = job["result"]
        await websocket.send_json({
            "job_id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "result": result_data,
            "error": job["error"]
        })
        await websocket.close()
        return

    # Subscribe to Redis Pub/Sub channel
    pubsub = None
    try:
        r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe(f"job_progress:{job_id}")
        
        # Send initial status
        await websocket.send_json({
            "job_id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "result": None,
            "error": job["error"]
        })

        while True:
            # Check for messages (non-blocking)
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if msg and msg.get("data"):
                data = json.loads(msg["data"])
                
                # Parse result payload if present
                if data.get("result"):
                    try:
                        data["result"] = json.loads(data["result"])
                    except Exception:
                        pass
                
                await websocket.send_json(data)
                if data.get("status") in ["completed", "failed", "cancelled"]:
                    break
            
            # Simple heartbeat/sleep to avoid CPU pinning
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        if pubsub:
            try:
                pubsub.unsubscribe(f"job_progress:{job_id}")
                pubsub.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
