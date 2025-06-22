from fastapi import APIRouter, HTTPException
import time
import logging
import httpx
from typing import Dict, Any

# Import from the new providers module
from app.providers import get_api_key
from app.config import VERTEX_PROJECT_ID, VERTEX_LOCATION

# 复用 server.py 中的日志格式
logger = logging.getLogger(__name__)

# Fallback get_api_key is no longer needed

router = APIRouter(prefix="/v1/models", tags=["models"])

TIMEOUT = httpx.Timeout(10.0, connect=15.0)

# ======== 简单 TTL 缓存装饰器 ========

def cache_ttl(seconds: int = 300):
    """Decorate coroutine to cache result for given TTL seconds."""
    def decorator(func):
        cache_data: Dict[str, Any] = {}
        cache_expiry: Dict[str, float] = {}
        async def wrapper(*args, refresh: bool = False, **kwargs):
            key = func.__name__ + str(args) + str(kwargs)
            if not refresh and key in cache_data and cache_expiry.get(key, 0) > time.time():
                return cache_data[key]
            result = await func(*args, refresh=refresh, **kwargs)
            cache_data[key] = result
            cache_expiry[key] = time.time() + seconds
            return result
        return wrapper
    return decorator

# ---------- OpenRouter ----------
@cache_ttl(300)
async def _fetch_openrouter_models(refresh: bool = False):
    api_key = (get_api_key("openrouter") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY not configured or is empty.")

    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://openrouter.ai/api/v1/models"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, headers=headers)
        if r.status_code != 200:
            logger.warning(f"OpenRouter model list failed: {r.status_code} {r.text}")
            raise HTTPException(r.status_code, r.text)
        data = r.json()

    # 优先把免费模型放在前面（prompt_token=completion_token=0 视为免费）
    models = []
    for m in data.get("data", []):
        pricing = m.get("pricing", {}) or {}
        is_free = pricing.get("prompt_token", 1) == 0 and pricing.get("completion_token", 1) == 0
        models.append((m.get("id"), is_free))

    # free=True 排到前面，其余保持原顺序
    ordered = [mid for mid, _ in sorted(models, key=lambda x: (not x[1]))]
    return ordered

@router.get("/openrouter")
async def list_openrouter_models(refresh: bool = False):
    models = await _fetch_openrouter_models(refresh=refresh) if refresh else await _fetch_openrouter_models()
    return {"provider": "openrouter", "models": models}

# ---------- Gemini Developer API ----------
@cache_ttl(300)
async def _fetch_gemini_models(refresh: bool = False):
    api_key = (get_api_key("gemini") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not configured or is empty.")

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            logger.warning(f"Gemini model list failed: {r.status_code} {r.text}")
            raise HTTPException(r.status_code, r.text)
        data = r.json()
    return [m.get("name") for m in data.get("models", [])]

@router.get("/gemini")
async def list_gemini_models(refresh: bool = False):
    models = await _fetch_gemini_models(refresh=refresh) if refresh else await _fetch_gemini_models()
    return {"provider": "gemini", "models": models}

# ---------- Vertex AI ----------
@cache_ttl(300)
async def _fetch_vertex_models(project_id: str, location: str, refresh: bool = False):
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError:
        raise HTTPException(500, "google-auth not installed; install google-auth to use Vertex model list")

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/models"
    headers = {"Authorization": f"Bearer {credentials.token}"}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, headers=headers)
        if r.status_code != 200:
            logger.warning(f"Vertex model list failed: {r.status_code} {r.text}")
            raise HTTPException(r.status_code, r.text)
        data = r.json()
    return [m.get("name") for m in data.get("models", [])]

@router.get("/vertex")
async def list_vertex_models(
    project_id: str = VERTEX_PROJECT_ID or "",
    location: str = VERTEX_LOCATION or "us-central1",
    refresh: bool = False,
):
    if not project_id:
        raise HTTPException(400, "VERTEX_PROJECT_ID not configured")
    models = await _fetch_vertex_models(project_id, location, refresh=refresh) if refresh else await _fetch_vertex_models(project_id, location)
    return {"provider": "vertex_ai", "models": models} 