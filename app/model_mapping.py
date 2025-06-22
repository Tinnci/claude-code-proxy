import logging
from app.config import (
    PREFERRED_PROVIDER,
    BIG_MODEL,
    SMALL_MODEL,
    LATEST_GEMINI_PRO,
    LATEST_GEMINI_FLASH,
    LATEST_GEMINI_FLASH_LITE,
    OPENAI_MODELS,
    GEMINI_MODELS,
    VERTEX_AI_MODELS,
    XAI_MODELS,
)

logger = logging.getLogger(__name__)


def get_mapped_model_name(original_model: str) -> str:
    """
    Maps a model name to its target based on the preferred provider and other settings.
    This function contains the core model routing logic.
    """
    new_model = original_model
    v = original_model

    # Remove provider prefixes for easier matching
    clean_v = v
    if clean_v.startswith("anthropic/"):
        clean_v = clean_v[10:]
    elif clean_v.startswith("openai/"):
        clean_v = clean_v[7:]
    elif clean_v.startswith("gemini/"):
        clean_v = clean_v[7:]
    elif clean_v.startswith("vertex_ai/"):
        clean_v = clean_v[10:]
    elif clean_v.startswith("xai/"):
        clean_v = clean_v[4:]

    mapped = False
    lower_clean = clean_v.lower()

    # === Claude series name mapping ===
    if "haiku" in lower_clean:
        if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_FLASH_LITE:
            new_model, mapped = f"gemini/{LATEST_GEMINI_FLASH_LITE}", True
        elif PREFERRED_PROVIDER == "xai" and SMALL_MODEL in XAI_MODELS:
            new_model, mapped = f"xai/{SMALL_MODEL}", True
        elif PREFERRED_PROVIDER == "vertex" and SMALL_MODEL in VERTEX_AI_MODELS:
            new_model, mapped = f"vertex_ai/{SMALL_MODEL}", True
        elif PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            new_model, mapped = f"gemini/{SMALL_MODEL}", True
        else:
            new_model, mapped = f"openai/{SMALL_MODEL}", True

    elif "sonnet" in lower_clean:
        if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_FLASH:
            new_model, mapped = f"gemini/{LATEST_GEMINI_FLASH}", True
        elif PREFERRED_PROVIDER == "xai" and BIG_MODEL in XAI_MODELS:
            new_model, mapped = f"xai/{BIG_MODEL}", True
        elif PREFERRED_PROVIDER == "vertex" and BIG_MODEL in VERTEX_AI_MODELS:
            new_model, mapped = f"vertex_ai/{BIG_MODEL}", True
        elif PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            new_model, mapped = f"gemini/{BIG_MODEL}", True
        else:
            new_model, mapped = f"openai/{BIG_MODEL}", True

    elif "opus" in lower_clean:
        if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_PRO:
            new_model, mapped = f"gemini/{LATEST_GEMINI_PRO}", True
        elif PREFERRED_PROVIDER == "vertex" and BIG_MODEL in VERTEX_AI_MODELS:
            new_model, mapped = f"vertex_ai/{BIG_MODEL}", True
        else:
            new_model, mapped = f"openai/{BIG_MODEL}", True

    # === Direct use of gemini/pro etc. aliases ===
    elif PREFERRED_PROVIDER == "google" and v.startswith("gemini/"):
        alias = lower_clean
        if alias in {"pro", "flash", "flash-lite", "flash_lite"}:
            chosen = {
                "pro": LATEST_GEMINI_PRO,
                "flash": LATEST_GEMINI_FLASH,
                "flash-lite": LATEST_GEMINI_FLASH_LITE,
                "flash_lite": LATEST_GEMINI_FLASH_LITE,
            }.get(alias)
            if chosen:
                new_model, mapped = f"gemini/{chosen}", True

    # === Add prefixes to non-mapped models if they match known lists ===
    elif not mapped:
        if clean_v in XAI_MODELS and not v.startswith("xai/"):
            new_model, mapped = f"xai/{clean_v}", True
        elif clean_v in VERTEX_AI_MODELS and not v.startswith("vertex_ai/"):
            new_model, mapped = f"vertex_ai/{clean_v}", True
        elif clean_v in GEMINI_MODELS and not v.startswith("gemini/"):
            new_model, mapped = f"gemini/{clean_v}", True
        elif clean_v in OPENAI_MODELS and not v.startswith("openai/"):
            new_model, mapped = f"openai/{clean_v}", True

    # --- Logging ---
    if mapped:
        provider_map = {
            "vertex_ai/": f"🔵 VERTEX MODEL MAPPING: '{original_model}' ➡️ '{new_model}'",
            "xai/": f"🟣 XAI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'",
            "gemini/": f"🟡 GEMINI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'",
            "openai/": f"🟢 OPENAI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'",
        }
        log_msg = next(
            (
                msg
                for prefix, msg in provider_map.items()
                if new_model.startswith(prefix)
            ),
            f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'",
        )
        logger.info(log_msg)
    else:
        if not any(
            v.startswith(p)
            for p in ["openai/", "gemini/", "anthropic/", "vertex_ai/", "xai/"]
        ):
            logger.warning(
                f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is."
            )
        else:
            logger.debug(f"ℹ️ Using already prefixed model: '{new_model}'")

    return new_model
