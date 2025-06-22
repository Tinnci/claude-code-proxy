import asyncio
import logging
from app.config import (
    OPENAI_MODELS, GEMINI_MODELS, VERTEX_AI_MODELS,
    LATEST_GEMINI_PRO, LATEST_GEMINI_FLASH, LATEST_GEMINI_FLASH_LITE,
    VERTEX_PROJECT_ID, VERTEX_LOCATION,
    _pick_latest_gemini
)

logger = logging.getLogger(__name__)

async def _refresh_available_models():
    """Fetch model lists from providers at startup and update global constants."""
    try:
        from app.models_router import _fetch_openrouter_models, _fetch_gemini_models, _fetch_vertex_models
    except ImportError as e:
        logger.warning(f"Could not import model fetchers, skipping refresh: {e}")
        return

    # Using global keyword to modify the lists in config.py
    global OPENAI_MODELS, GEMINI_MODELS, VERTEX_AI_MODELS
    global LATEST_GEMINI_PRO, LATEST_GEMINI_FLASH, LATEST_GEMINI_FLASH_LITE

    # Refresh OpenRouter models
    try:
        openrouter_models = await _fetch_openrouter_models(refresh=True)
        if openrouter_models:
            OPENAI_MODELS[:] = list(sorted(set(openrouter_models)))
            logger.info(f"🔄 OpenRouter models refreshed: {len(OPENAI_MODELS)} models")
    except Exception as e:
        logger.warning(f"Could not refresh OpenRouter models: {e}")

    # Refresh Gemini models
    try:
        gemini_models = await _fetch_gemini_models(refresh=True)
        if gemini_models:
            GEMINI_MODELS[:] = list(sorted(set(m.replace("models/", "") for m in gemini_models)))
            logger.info(f"🔄 Gemini models refreshed: {len(GEMINI_MODELS)} models")
            # Update latest variants
            LATEST_GEMINI_PRO = _pick_latest_gemini("-pro") or LATEST_GEMINI_PRO
            LATEST_GEMINI_FLASH = _pick_latest_gemini("-flash") or LATEST_GEMINI_FLASH
            LATEST_GEMINI_FLASH_LITE = _pick_latest_gemini("flash-lite") or LATEST_GEMINI_FLASH_LITE
            logger.info(f"🟡 Gemini latest variants refreshed → pro:{LATEST_GEMINI_PRO}, flash:{LATEST_GEMINI_FLASH}")
    except Exception as e:
        logger.warning(f"Could not refresh Gemini models: {e}")

    # Refresh Vertex AI models
    try:
        if VERTEX_PROJECT_ID:
             vertex_models = await _fetch_vertex_models(VERTEX_PROJECT_ID, VERTEX_LOCATION, refresh=True)
             if vertex_models:
                cleaned_vertex = [m.split('/')[-1] for m in vertex_models]
                VERTEX_AI_MODELS[:] = list(sorted(set(cleaned_vertex)))
                logger.info(f"🔄 Vertex AI models refreshed: {len(VERTEX_AI_MODELS)} models")
    except Exception as e:
        logger.warning(f"Could not refresh Vertex models: {e}")

def register_startup_events(app):
    """Registers startup events for the FastAPI application."""
    @app.on_event("startup")
    async def startup_refresh_models():
        """Kick off model list refresh in the background."""
        logger.info("⏳ Scheduling model list refresh in background...")
        asyncio.create_task(_background_refresh())

async def _background_refresh():
    """The actual background task for refreshing models."""
    try:
        await _refresh_available_models()
        logger.info("✅ Model lists refreshed successfully.")
    except Exception as e:
        logger.error(f"background model list refresh failed: {e}", exc_info=True) 