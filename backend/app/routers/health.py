from fastapi import APIRouter, Request

from app.core.lifespan import get_health_status

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    return get_health_status(request.app)
