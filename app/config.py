import os
import logging
import re
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Set up a logger for this module
logger = logging.getLogger(__name__)

# --- API Key Management ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# --- Vertex AI Configuration ---
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

if VERTEX_PROJECT_ID:
    logger.info(f"🔵 Vertex AI Project ID configured: {VERTEX_PROJECT_ID}")
    logger.info(f"🔵 Vertex AI Location configured: {VERTEX_LOCATION}")
    if GOOGLE_APPLICATION_CREDENTIALS:
        logger.info(f"🔵 Vertex AI using service account credentials from: {GOOGLE_APPLICATION_CREDENTIALS}")
    else:
        logger.info("🔵 Vertex AI will use Application Default Credentials (ADC)")
else:
    logger.warning("⚠️ VERTEX_PROJECT_ID not set. Vertex AI models will not work correctly without it.")

if XAI_API_KEY:
    logger.info("🟣 xAI API key configured")
else:
    logger.warning("⚠️ XAI_API_KEY not set. xAI models will not work without it.")


# --- Provider and Model Configuration ---
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
logger.info(f"🔧 Preferred provider set to: {PREFERRED_PROVIDER}")

if PREFERRED_PROVIDER == "google":
    BIG_MODEL_DEFAULT = "gemini-1.5-pro-latest"
    SMALL_MODEL_DEFAULT = "gemini-1.5-flash-latest"
elif PREFERRED_PROVIDER == "vertex":
    BIG_MODEL_DEFAULT = "gemini-1.5-pro-preview-0514"
    SMALL_MODEL_DEFAULT = "gemini-1.5-flash-preview-0514"
elif PREFERRED_PROVIDER == "xai":
    BIG_MODEL_DEFAULT = "grok-3-beta"
    SMALL_MODEL_DEFAULT = "grok-3-mini-beta"
elif PREFERRED_PROVIDER == "openrouter":
    BIG_MODEL_DEFAULT = "anthropic/claude-3-sonnet"
    SMALL_MODEL_DEFAULT = "mistralai/mistral-7b-instruct:free"
else:  # Default to openai
    BIG_MODEL_DEFAULT = "gpt-4o"
    SMALL_MODEL_DEFAULT = "gpt-4o-mini"

BIG_MODEL = os.environ.get("BIG_MODEL", BIG_MODEL_DEFAULT).strip()
SMALL_MODEL = os.environ.get("SMALL_MODEL", SMALL_MODEL_DEFAULT).strip()

logger.info(f"🗺️  Model Mapping: Sonnet (big) -> {BIG_MODEL}, Haiku (small) -> {SMALL_MODEL}")


# --- Static Model Lists (will be updated at runtime) ---
OPENAI_MODELS = [
    "o3-mini", "o1", "o1-mini", "o1-pro", "gpt-4.5-preview", "gpt-4o",
    "gpt-4o-audio-preview", "chatgpt-4o-latest", "gpt-4o-mini",
    "gpt-4o-mini-audio-preview", "gpt-4.1", "gpt-4.1-mini"
]

GEMINI_MODELS = [
    "gemini-2.5-pro-preview-03-25", "gemini-2.0-flash"
]

VERTEX_AI_MODELS = [
    "gemini-2.5-pro-preview-03-25", "gemini-2.0-flash",
    "gemini-1.5-flash-preview-0514", "gemini-1.5-pro-preview-0514",
    "gemini-2.5-flash-preview-04-17"
]
logger.info(f"🔵 Vertex AI models available: {', '.join(VERTEX_AI_MODELS)}")

XAI_MODELS = [
    "grok-3-mini-beta", "grok-2-vision-latest", "grok-3-beta", "grok-2", "grok-1"
]
logger.info(f"🟣 xAI models available: {', '.join(XAI_MODELS)}")


# --- Gemini Model Helper ---
def _pick_latest_gemini(variant_keyword: str) -> Optional[str]:
    """
    Return the newest Gemini model name containing the variant keyword.
    Sorts by version number first, then by name. Returns None if no match.
    """
    candidates = [m for m in GEMINI_MODELS if variant_keyword in m and "tts" not in m.lower()]
    if not candidates:
        return None

    def _version_score(name: str) -> float:
        match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return float(match.group(1)) if match else 0.0

    return max(candidates, key=lambda n: (_version_score(n), n))

LATEST_GEMINI_PRO = _pick_latest_gemini("-pro") or "gemini-1.5-pro-latest"
LATEST_GEMINI_FLASH = _pick_latest_gemini("-flash") or "gemini-1.5-flash-latest"
LATEST_GEMINI_FLASH_LITE = _pick_latest_gemini("flash-lite") or LATEST_GEMINI_FLASH

logger.info(
    f"🟡 Gemini variants → pro:{LATEST_GEMINI_PRO}, flash:{LATEST_GEMINI_FLASH}, flash-lite:{LATEST_GEMINI_FLASH_LITE}"
)
