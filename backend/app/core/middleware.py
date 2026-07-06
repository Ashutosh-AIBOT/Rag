from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.database.database import check_user_budget
from app.core.logging import get_logger

logger = get_logger(__name__)


class TokenGatingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Apply gating only to query, compare, hybrid and evaluation paths
        path = request.url.path
        if any(p in path for p in ["/query", "/compare", "/hybrid", "/evaluate"]):
            # Resolve user ID (e.g. from headers, default to 'default_user' if missing)
            user_id = request.headers.get("X-User-ID", "default_user")
            
            # Check budget
            if not check_user_budget(user_id):
                logger.warning(f"User '{user_id}' blocked. Token budget exceeded.")
                return JSONResponse(
                    status_code=402,
                    content={"detail": "API Token Budget Exceeded. Please upgrade your daily token quota."}
                )

        response = await call_next(request)
        return response
