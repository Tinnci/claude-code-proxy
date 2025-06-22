from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
from itertools import cycle
import asyncio, importlib

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]

        message = record.getMessage()
        for phrase in blocked_phrases:
            if phrase in message:
                return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# -------- Redact sensitive tokens from any log messages --------
class RedactAPIKeyFilter(logging.Filter):
    _patterns = [
        (re.compile(r"key=[A-Za-z0-9_\-]+"), "key=***"),
        (re.compile(r"Bearer [A-Za-z0-9_\-]+"), "Bearer ***"),
        (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-***"),
        (re.compile(r"AIza[A-Za-z0-9_-]+"), "AIza***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg = record.msg
            for pat, repl in self._patterns:
                msg = pat.sub(repl, msg)
            record.msg = msg
        return True

root_logger.addFilter(RedactAPIKeyFilter())

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Enable dropping unsupported parameters for LiteLLM
litellm.drop_params = True
logger.info("✅ Enabled LiteLLM drop_params=True to ignore unsupported API parameters.")

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"    # Vertex AI
    GREEN = "\033[92m"   # OpenAI
    YELLOW = "\033[93m"  # Gemini
    RED = "\033[91m"     # Anthropic (Direct)
    MAGENTA = "\033[95m" # Tools/Messages count
    CYAN = "\033[96m"    # Claude (Original) & xAI
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.DEBUG and isinstance(record.msg, str) and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys/credentials from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# For Vertex AI, LiteLLM typically uses Application Default Credentials (ADC)
# Ensure ADC is set up (e.g., `gcloud auth application-default login`)
# Or set GOOGLE_APPLICATION_CREDENTIALS environment variable
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") # Required by LiteLLM for Vertex
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")  # Required by LiteLLM for Vertex

# Check if Vertex AI credentials are properly configured
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

# API Key for xAI
XAI_API_KEY = os.environ.get("XAI_API_KEY")
if XAI_API_KEY:
    logger.info("🟣 xAI API key configured")
else:
    logger.warning("⚠️ XAI_API_KEY not set. xAI models will not work without it.")

# GROQ_API_KEY removed

# Get preferred provider (default to openai)
# Possible values: 'openai', 'google', 'vertex', 'xai'
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
logger.info(f"🔧 Preferred provider set to: {PREFERRED_PROVIDER}")

# Get model mapping configuration from environment
# Default models now depend on the preferred provider
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

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models (Note: These are often used via Google AI Studio, not Vertex directly)
GEMINI_MODELS = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash"
    # Add other relevant Gemini models if needed
]

# List of Vertex AI models (examples, adjust based on availability)
# LiteLLM uses 'vertex_ai/' prefix for these
VERTEX_AI_MODELS = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash",
    "gemini-1.5-flash-preview-0514",
    "gemini-1.5-pro-preview-0514",
    "gemini-2.5-flash-preview-04-17" # Added for compatibility with newer models
    # Add other specific Vertex AI model IDs
]
logger.info(f"🔵 Vertex AI models available: {', '.join(VERTEX_AI_MODELS)}")

# GROQ_MODELS list removed

# List of xAI models (examples, adjust based on availability/correct IDs)
# LiteLLM uses 'xai/' prefix for these
XAI_MODELS = [
    "grok-3-mini-beta", # As per LiteLLM docs
    "grok-2-vision-latest", # As per LiteLLM docs
    "grok-3-beta", # Actual Grok-3 model
    "grok-2", # Grok-2 model
    "grok-1" # Grok-1 model
]
logger.info(f"🟣 xAI models available: {', '.join(XAI_MODELS)}")

# ---------- Gemini model helper ----------
# 动态挑选最新的 pro / flash / flash-lite 变体，避免硬编码具体版本号
import re
from typing import Optional

def _pick_latest_gemini(variant_keyword: str) -> Optional[str]:
    """Return the newest Gemini model name that contains the given variant keyword (e.g. 'pro', 'flash', 'flash-lite').
    排序策略：先比较版本号（如 2.5 > 2.0 > 1.5），再比较完整字符串。
    如果没有找到匹配的模型则返回 None。"""
    # 确保 GEMINI_MODELS 已刷新（startup_refresh_models 会更新）。
    candidates = [m for m in GEMINI_MODELS if variant_keyword in m and "tts" not in m.lower()]
    if not candidates:
        return None

    def _version_score(name: str) -> float:
        match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return float(match.group(1)) if match else 0.0

    # 取版本号最高的；若相同按字符串排序保证确定性
    return max(candidates, key=lambda n: (_version_score(n), n))

LATEST_GEMINI_PRO: Optional[str] = _pick_latest_gemini("-pro") or "gemini-1.5-pro-latest"
LATEST_GEMINI_FLASH: Optional[str] = _pick_latest_gemini("-flash") or "gemini-1.5-flash-latest"
LATEST_GEMINI_FLASH_LITE: Optional[str] = _pick_latest_gemini("flash-lite") or LATEST_GEMINI_FLASH

logger.info(
    f"🟡 Gemini variants → pro:{LATEST_GEMINI_PRO}, flash:{LATEST_GEMINI_FLASH}, flash-lite:{LATEST_GEMINI_FLASH_LITE}"
)

# Helper function to clean schema for Gemini/Vertex
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini/Vertex."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini/Vertex schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

# Updated ThinkingConfig to accept client format
class ThinkingConfigClient(BaseModel):
    budget_tokens: Optional[int] = None
    type: Optional[str] = None

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfigClient] = None # Use the updated client-facing model
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('vertex_ai/'):
            clean_v = clean_v[10:]
        # elif clean_v.startswith('groq/'): # Removed Groq prefix check
        #     clean_v = clean_v[5:]
        elif clean_v.startswith('xai/'):
            clean_v = clean_v[4:]

        # --- Mapping Logic --- START ---
        mapped = False
        lower_clean = clean_v.lower()

        # === Claude 系列名称映射 ===
        if "haiku" in lower_clean:  # 最小模型
            if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_FLASH_LITE:
                new_model = f"gemini/{LATEST_GEMINI_FLASH_LITE}"
                mapped = True
            elif PREFERRED_PROVIDER == "xai" and SMALL_MODEL in XAI_MODELS:
                new_model = f"xai/{SMALL_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "vertex" and SMALL_MODEL in VERTEX_AI_MODELS:
                new_model = f"vertex_ai/{SMALL_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        elif "sonnet" in lower_clean:  # 中等模型
            if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_FLASH:
                new_model = f"gemini/{LATEST_GEMINI_FLASH}"
                mapped = True
            elif PREFERRED_PROVIDER == "xai" and BIG_MODEL in XAI_MODELS:
                new_model = f"xai/{BIG_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "vertex" and BIG_MODEL in VERTEX_AI_MODELS:
                new_model = f"vertex_ai/{BIG_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        elif "opus" in lower_clean:  # 最大模型
            if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_PRO:
                new_model = f"gemini/{LATEST_GEMINI_PRO}"
                mapped = True
            elif PREFERRED_PROVIDER == "vertex" and BIG_MODEL in VERTEX_AI_MODELS:
                new_model = f"vertex_ai/{BIG_MODEL}"
                mapped = True
            else:
                # 默认使用 OpenAI 或其他大模型，保持向后兼容
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # === 直接使用 gemini/pro 等别名 ===
        elif PREFERRED_PROVIDER == "google" and v.startswith("gemini/"):
            alias = lower_clean  # 已去掉前缀
            if alias in {"pro", "flash", "flash-lite", "flash_lite"}:
                chosen = {
                    "pro": LATEST_GEMINI_PRO,
                    "flash": LATEST_GEMINI_FLASH,
                    "flash-lite": LATEST_GEMINI_FLASH_LITE,
                    "flash_lite": LATEST_GEMINI_FLASH_LITE,
                }.get(alias)
                if chosen:
                    new_model = f"gemini/{chosen}"
                    mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in XAI_MODELS and not v.startswith('xai/'):
                new_model = f"xai/{clean_v}"
                mapped = True
            # elif clean_v in GROQ_MODELS and not v.startswith('groq/'): # Removed Groq prefixing
            #     new_model = f"groq/{clean_v}"
            #     mapped = True
            elif clean_v in VERTEX_AI_MODELS and not v.startswith('vertex_ai/'):
                new_model = f"vertex_ai/{clean_v}"
                mapped = True
            elif clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            # Enhanced logging with provider-specific emojis
            if new_model.startswith("vertex_ai/"):
                logger.info(f"🔵 VERTEX MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            elif new_model.startswith("xai/"):
                logger.info(f"🟣 XAI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            elif new_model.startswith("gemini/"):
                logger.info(f"🟡 GEMINI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            elif new_model.startswith("openai/"):
                logger.info(f"🟢 OPENAI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            else:
                logger.info(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'vertex_ai/', 'xai/')): # Removed 'groq/'
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
                 new_model = v # Ensure we return the original if no rule applied or prefix exists
             else:
                 new_model = v # Use the already prefixed model name
                 logger.debug(f"ℹ️ Using already prefixed model: '{new_model}'")

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfigClient] = None # Use the updated client-facing model
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('vertex_ai/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('xai/'):
            clean_v = clean_v[4:]

        # --- Mapping Logic --- START ---
        mapped = False
        lower_clean = clean_v.lower()

        # === Claude 系列名称映射 ===
        if "haiku" in lower_clean:  # 最小模型
            if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_FLASH_LITE:
                new_model = f"gemini/{LATEST_GEMINI_FLASH_LITE}"
                mapped = True
            elif PREFERRED_PROVIDER == "xai" and SMALL_MODEL in XAI_MODELS:
                new_model = f"xai/{SMALL_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "vertex" and SMALL_MODEL in VERTEX_AI_MODELS:
                new_model = f"vertex_ai/{SMALL_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        elif "sonnet" in lower_clean:  # 中等模型
            if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_FLASH:
                new_model = f"gemini/{LATEST_GEMINI_FLASH}"
                mapped = True
            elif PREFERRED_PROVIDER == "xai" and BIG_MODEL in XAI_MODELS:
                new_model = f"xai/{BIG_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "vertex" and BIG_MODEL in VERTEX_AI_MODELS:
                new_model = f"vertex_ai/{BIG_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        elif "opus" in lower_clean:  # 最大模型
            if PREFERRED_PROVIDER == "google" and LATEST_GEMINI_PRO:
                new_model = f"gemini/{LATEST_GEMINI_PRO}"
                mapped = True
            elif PREFERRED_PROVIDER == "vertex" and BIG_MODEL in VERTEX_AI_MODELS:
                new_model = f"vertex_ai/{BIG_MODEL}"
                mapped = True
            else:
                # 默认使用 OpenAI 或其他大模型，保持向后兼容
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # === 直接使用 gemini/pro 等别名 ===
        elif PREFERRED_PROVIDER == "google" and v.startswith("gemini/"):
            alias = lower_clean  # 已去掉前缀
            if alias in {"pro", "flash", "flash-lite", "flash_lite"}:
                chosen = {
                    "pro": LATEST_GEMINI_PRO,
                    "flash": LATEST_GEMINI_FLASH,
                    "flash-lite": LATEST_GEMINI_FLASH_LITE,
                    "flash_lite": LATEST_GEMINI_FLASH_LITE,
                }.get(alias)
                if chosen:
                    new_model = f"gemini/{chosen}"
                    mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in XAI_MODELS and not v.startswith('xai/'):
                new_model = f"xai/{clean_v}"
                mapped = True
            elif clean_v in VERTEX_AI_MODELS and not v.startswith('vertex_ai/'):
                new_model = f"vertex_ai/{clean_v}"
                mapped = True
            elif clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'vertex_ai/', 'xai/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
                 new_model = v # Ensure we return the original if no rule applied or prefix exists
             else:
                 new_model = v # Use the already prefixed model name

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")

    # Process the request and get the response
    response = await call_next(request)

    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    messages = []

    # Add system message if present
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                text_content = ""
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            result_content = parse_tool_result_content(getattr(block, "content", None)) # Use helper
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            source = getattr(block, 'source', {})
                            if isinstance(source, dict) and source.get('type') == 'base64':
                                image_url = f"data:image/{source.get('media_type', 'jpeg')};base64,{source.get('data', '')}"
                                processed_content.append({"type": "image_url", "image_url": {"url": image_url}})
                            else:
                                logger.warning(f"Image block source format not explicitly handled for LiteLLM: {source}")
                                processed_content.append({"type": "image", "source": source})

                        elif block.type == "tool_use":
                             if msg.role == "assistant":
                                 tool_call_data = {
                                     "id": block.id,
                                     "type": "function",
                                     "function": {
                                         "name": block.name,
                                         "arguments": json.dumps(block.input)
                                     }
                                 }
                                 if not messages: messages.append({"role": msg.role, "content": None, "tool_calls": []})
                                 if messages[-1]["role"] != msg.role or "tool_calls" not in messages[-1]:
                                      messages.append({"role": msg.role, "content": None, "tool_calls": []})
                                 messages[-1]["tool_calls"].append(tool_call_data)
                             else:
                                 logger.warning(f"Unexpected tool_use block in user message: {block}")

                        elif block.type == "tool_result":
                             messages.append({
                                 "role": "tool",
                                 "tool_call_id": block.tool_use_id,
                                 "content": parse_tool_result_content(getattr(block, "content", None))
                             })
                if processed_content:
                     if messages and messages[-1]["role"] == "assistant" and messages[-1].get("content") is None and messages[-1].get("tool_calls"):
                         messages[-1]["content"] = processed_content
                     else:
                         messages.append({"role": msg.role, "content": processed_content})


    # LiteLLM request dict structure
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": anthropic_request.max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters
    if anthropic_request.stop_sequences: litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p: litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k: litellm_request["top_k"] = anthropic_request.top_k

    # Convert tools
    is_vertex_model = anthropic_request.model.startswith("vertex_ai/")
    is_gemini_model = anthropic_request.model.startswith("gemini/")

    if anthropic_request.tools:
        openai_tools = []
        for tool in anthropic_request.tools:
            if hasattr(tool, 'model_dump'): tool_dict = tool.model_dump(exclude_unset=True)
            elif hasattr(tool, 'dict'): tool_dict = tool.dict(exclude_unset=True)
            else:
                try: tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}"); continue
            input_schema = tool_dict.get("input_schema", {})
            if is_vertex_model or is_gemini_model:
                 target_provider = "Vertex AI" if is_vertex_model else "Gemini"
                 logger.debug(f"Cleaning schema for {target_provider} tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)
            openai_tools.append({
                "type": "function",
                "function": {"name": tool_dict["name"], "description": tool_dict.get("description", ""), "parameters": input_schema}
            })
        litellm_request["tools"] = openai_tools

    # Convert tool_choice
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'model_dump'): tool_choice_dict = anthropic_request.tool_choice.model_dump(exclude_unset=True)
        elif hasattr(anthropic_request.tool_choice, 'dict'): tool_choice_dict = anthropic_request.tool_choice.dict(exclude_unset=True)
        else: tool_choice_dict = anthropic_request.tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto": litellm_request["tool_choice"] = "auto"
        elif choice_type == "any": litellm_request["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {"type": "function", "function": {"name": tool_choice_dict["name"]}}
        else: litellm_request["tool_choice"] = "auto"

    # Clean up messages for final request
    litellm_request["messages"] = [m for m in litellm_request["messages"] if m.get("content") or m.get("tool_calls")]

    # Handle thinking field conversion
    if anthropic_request.thinking and anthropic_request.thinking.type == "enabled":
        # Add the format expected by the downstream API
        litellm_request["thinking"] = {"enabled": True}
        logger.debug("💡 Added 'thinking: {\"enabled\": True}' to LiteLLM request.")
    # Else: if thinking is None or type != "enabled", don't add thinking field

    return litellm_request


def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any],
                                 original_request: MessagesRequest,
                                 original_model_override: Optional[str] = None) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    try:
        prompt_tokens, completion_tokens, response_id = 0, 0, f"msg_{uuid.uuid4()}"
        choices, message, finish_reason = [], {}, "stop"

        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            choices, usage_info = litellm_response.choices, litellm_response.usage
            response_id = getattr(litellm_response, 'id', response_id)
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        elif isinstance(litellm_response, dict):
             choices = litellm_response.get("choices", [{}])
             usage_info = litellm_response.get("usage", {})
             response_id = litellm_response.get("id", response_id)
             prompt_tokens = usage_info.get("prompt_tokens", 0)
             completion_tokens = usage_info.get("completion_tokens", 0)
        else: # Try conversion
            try:
                response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                choices = response_dict.get("choices", [{}])
                usage_info = response_dict.get("usage", {})
                response_id = response_dict.get("id", response_id)
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
            except Exception as conv_err:
                 logger.error(f"Could not convert litellm_response: {conv_err}")
                 raise conv_err # Re-raise after logging

        if choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                 message = first_choice.get("message", {})
                 finish_reason = first_choice.get("finish_reason", "stop")
            elif hasattr(first_choice, 'message'):
                 message = first_choice.message
                 finish_reason = getattr(first_choice, 'finish_reason', "stop")
                 if hasattr(message, 'model_dump'): message = message.model_dump()
                 elif hasattr(message, '__dict__'): message = message.__dict__

        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        content = []
        if content_text: content.append({"type": "text", "text": content_text})

        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")
            if not isinstance(tool_calls, list): tool_calls = [tool_calls]
            for tool_call in tool_calls:
                 tool_call_dict = {}
                 if isinstance(tool_call, dict): tool_call_dict = tool_call
                 elif hasattr(tool_call, 'model_dump'): tool_call_dict = tool_call.model_dump()
                 elif hasattr(tool_call, '__dict__'): tool_call_dict = tool_call.__dict__
                 function_call = tool_call_dict.get("function", {})
                 tool_id = tool_call_dict.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                 name = function_call.get("name", "")
                 arguments_str = function_call.get("arguments", "{}")
                 try: arguments = json.loads(arguments_str)
                 except json.JSONDecodeError: arguments = {"raw_arguments": arguments_str}
                 content.append({"type": "tool_use", "id": tool_id, "name": name, "input": arguments})

        stop_reason_map = {"length": "max_tokens", "tool_calls": "tool_use", "stop": "end_turn"}
        stop_reason = stop_reason_map.get(finish_reason, finish_reason or "end_turn")

        if not content: content.append({"type": "text", "text": ""})

        # Use the override if available, otherwise fall back to pydantic model fields
        final_model_name = original_model_override or original_request.original_model or original_request.model

        return MessagesResponse(
            id=response_id, model=final_model_name,
            role="assistant", content=content, stop_reason=stop_reason, stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        )
    except Exception as e:
        import traceback
        logger.error(f"Error converting response: {str(e)}\n{traceback.format_exc()}")
        return MessagesResponse(
            id=f"msg_error_{uuid.uuid4()}", model=original_request.model, role="assistant",
            content=[{"type": "text", "text": f"Error converting backend response: {str(e)}"}],
            stop_reason="error", usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original_request.original_model or original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

        text_buffer, tool_buffers = "", {}
        current_block_index, text_block_index = -1, -1
        tool_block_indices = {}
        final_usage = {"input_tokens": 0, "output_tokens": 0}
        final_stop_reason = "end_turn"

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        async for chunk in response_generator:
            if hasattr(chunk, 'usage') and chunk.usage:
                final_usage["input_tokens"] = getattr(chunk.usage, 'prompt_tokens', final_usage["input_tokens"])
                final_usage["output_tokens"] = getattr(chunk.usage, 'completion_tokens', final_usage["output_tokens"])

            if not chunk.choices: continue
            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                stop_reason_map = {"length": "max_tokens", "tool_calls": "tool_use", "stop": "end_turn"}
                final_stop_reason = stop_reason_map.get(choice.finish_reason, choice.finish_reason or "end_turn")

            if delta.content:
                if text_block_index == -1:
                    current_block_index += 1; text_block_index = current_block_index
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_buffer += delta.content
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': delta.content}})}\n\n"

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    tool_index = tool_call_delta.index
                    if tool_index not in tool_block_indices:
                        current_block_index += 1; anthropic_block_index = current_block_index
                        tool_block_indices[tool_index] = anthropic_block_index
                        tool_id = tool_call_delta.id or f"toolu_{uuid.uuid4().hex[:24]}"
                        tool_name = tool_call_delta.function.name if tool_call_delta.function else ""
                        tool_buffers[anthropic_block_index] = {"id": tool_id, "name": tool_name, "input": ""}
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {}}})}\n\n"

                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        anthropic_block_index = tool_block_indices[tool_index]
                        arg_chunk = tool_call_delta.function.arguments
                        tool_buffers[anthropic_block_index]["input"] += arg_chunk
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': arg_chunk}})}\n\n"

        if text_block_index != -1: yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
        for _, anthropic_idx in tool_block_indices.items(): yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_idx})}\n\n"

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': final_usage['output_tokens']}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    except Exception as e:
        import traceback
        logger.error(f"Error during streaming conversion: {e}\n{traceback.format_exc()}")
        try:
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error'}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        except: pass


@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")

        display_model = original_model
        if "/" in display_model: display_model = display_model.split("/")[-1]

        # Clean model name for logging/prefix check
        provider_prefix = ""
        clean_model_name = request.model
        if clean_model_name.startswith("anthropic/"): provider_prefix = "anthropic/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("openai/"): provider_prefix = "openai/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("gemini/"): provider_prefix = "gemini/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("vertex_ai/"): provider_prefix = "vertex_ai/"; clean_model_name = clean_model_name[len(provider_prefix):]
        # elif clean_model_name.startswith("groq/"): provider_prefix = "groq/"; clean_model_name = clean_model_name[len(provider_prefix):] # Removed Groq check
        elif clean_model_name.startswith("xai/"): provider_prefix = "xai/"; clean_model_name = clean_model_name[len(provider_prefix):]


        logger.debug(f"📊 PROCESSING REQUEST: Original Model='{original_model}', Mapped Model='{request.model}', Stream={request.stream}")

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # Add global timeout and retries for robustness
        litellm_request['timeout'] = 60
        litellm_request['max_retries'] = 2

        # Determine API key/credentials based on the final model's provider prefix
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = get_api_key("openai") or OPENAI_API_KEY
            logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("openrouter/"):
            litellm_request["api_key"] = get_api_key("openrouter") or OPENROUTER_API_KEY
            # 指定 OpenRouter 专属 base URL
            litellm_request["api_base"] = "https://openrouter.ai/api/v1"
            logger.debug(f"Using OpenRouter API key for model: {request.model}")
        elif request.model.startswith("gemini/"): # Google AI Studio
            litellm_request["api_key"] = get_api_key("gemini") or GEMINI_API_KEY
            logger.debug(f"Using Gemini (Google AI Studio) API key for model: {request.model}")
        elif request.model.startswith("vertex_ai/"):
            if VERTEX_PROJECT_ID:
                litellm_request["vertex_project"] = VERTEX_PROJECT_ID
                litellm_request["vertex_location"] = VERTEX_LOCATION
                logger.info(f"🔵 Using Vertex AI credentials (Project: {VERTEX_PROJECT_ID}, Location: {VERTEX_LOCATION}) for model: {request.model}")

                # Check for Application Default Credentials or service account
                if GOOGLE_APPLICATION_CREDENTIALS:
                    logger.info(f"🔵 Vertex AI using service account from: {GOOGLE_APPLICATION_CREDENTIALS}")
                else:
                    logger.info("🔵 Vertex AI using Application Default Credentials (ADC)")
            else:
                logger.warning(f"⚠️ VERTEX_PROJECT_ID not set for Vertex AI model {request.model}. LiteLLM will attempt default auth but may fail.")
        # elif request.model.startswith("groq/"): # Removed Groq block
        #     litellm_request["api_key"] = GROQ_API_KEY
        #     logger.debug(f"Using Groq API key for model: {request.model}")
        elif request.model.startswith("xai/"):
            selected_key = get_api_key("xai") or XAI_API_KEY
            if selected_key:
                litellm_request["api_key"] = selected_key
                logger.info(f"🟣 Using xAI API key for model: {request.model}")

                # Validate model name against known models
                model_name = request.model.replace("xai/", "")
                if model_name in XAI_MODELS:
                    logger.info(f"🟣 Validated xAI model: {model_name}")
                else:
                    logger.warning(f"⚠️ Unknown xAI model: {model_name}. Request may fail.")
            else:
                logger.error(f"❌ XAI_API_KEY not set for xAI model: {request.model}. Request will fail.")
        else: # Default to Anthropic if no other known prefix
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using default Anthropic API key for model: {request.model}")

        # Specific provider adjustments (Example for OpenAI, might need for others)
        if request.model.startswith("openai/") and "messages" in litellm_request:
            # OpenAI specific message processing... (keep existing logic if needed)
            pass # Keep the OpenAI specific message processing logic here if it's still relevant

        logger.debug(f"LiteLLM Request (keys filtered): { {k:v for k,v in litellm_request.items() if k != 'api_key'} }")

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", raw_request.url.path, display_model, request.model, # Use mapped model name for logging
            len(litellm_request['messages']), num_tools, 200 # Assuming success for logging before call
        )

        if request.stream:
            response_generator = await send_with_key_retry(litellm_request, request.model)
            return StreamingResponse(handle_streaming(response_generator, request), media_type="text/event-stream")
        else:
            start_time = time.time()
            logger.info(f"🚀 Sending request to {request.model}...")
            litellm_response = await send_with_key_retry(litellm_request, request.model)
            elapsed_time = time.time() - start_time

            # Enhanced provider-specific response logging
            if request.model.startswith("vertex_ai/"):
                logger.info(f"🔵 VERTEX RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("xai/"):
                logger.info(f"🟣 XAI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("gemini/"):
                logger.info(f"🟡 GEMINI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("openai/"):
                logger.info(f"🟢 OPENAI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            else:
                logger.info(f"✅ RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request, original_model_override=original_model)
            return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_details = {"error": str(e), "type": type(e).__name__, "traceback": error_traceback}
        # Safely add additional error attributes, converting non-serializable ones
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                value = getattr(e, attr)
                # Convert potentially non-serializable types to strings for JSON logging
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    try:
                        value = str(value) # Attempt string conversion
                    except:
                        value = f"<{type(value).__name__} object (unserializable)>" # Fallback
                error_details[attr] = value
        try:
            logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        except Exception as log_e:
            # Fallback logging if json.dumps still fails (shouldn't happen now, but safe)
            logger.error(f"Error processing request (fallback log): {error_details}")
            logger.error(f"Logging failed due to: {log_e}")

        status_code = getattr(e, 'status_code', 500)
        # Ensure detail_message is a string
        detail_message = getattr(e, 'message', None) # Get original message if possible
        if detail_message is None:
            detail_message = str(e) # Fallback to string representation of exception
        if isinstance(detail_message, bytes): detail_message = detail_message.decode('utf-8', errors='ignore')
        raise HTTPException(status_code=status_code, detail=str(detail_message))


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        original_model = request.original_model or request.model
        display_model = original_model
        if "/" in display_model: display_model = display_model.split("/")[-1]

        # Clean model name for logging/prefix check
        provider_prefix = ""
        clean_model_name = request.model
        if clean_model_name.startswith("anthropic/"): provider_prefix = "anthropic/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("openai/"): provider_prefix = "openai/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("gemini/"): provider_prefix = "gemini/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("vertex_ai/"): provider_prefix = "vertex_ai/"; clean_model_name = clean_model_name[len(provider_prefix):]
        # elif clean_model_name.startswith("groq/"): provider_prefix = "groq/"; clean_model_name = clean_model_name[len(provider_prefix):] # Removed Groq check
        elif clean_model_name.startswith("xai/"): provider_prefix = "xai/"; clean_model_name = clean_model_name[len(provider_prefix):]

        temp_msg_request_data = request.model_dump()
        temp_msg_request_data['max_tokens'] = 1 # Dummy value
        temp_msg_request = MessagesRequest(**temp_msg_request_data)
        converted_request = convert_anthropic_to_litellm(temp_msg_request)

        from litellm import token_counter

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", raw_request.url.path, display_model, request.model,
            len(converted_request['messages']), num_tools, 200
        )

        token_count = token_counter(
            model=converted_request["model"],
            messages=converted_request["messages"],
        )
        return TokenCountResponse(input_tokens=token_count)

    except ImportError:
        logger.error("Could not import token_counter from litellm")
        return TokenCountResponse(input_tokens=1000) # Fallback
    except Exception as e:
        import traceback
        logger.error(f"Error counting tokens: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"    # Claude (Original) & xAI
    BLUE = "\033[94m"    # Vertex AI
    GREEN = "\033[92m"   # OpenAI
    YELLOW = "\033[93m"  # Gemini
    RED = "\033[91m"     # Anthropic (Direct)
    MAGENTA = "\033[95m" # Tools/Messages count
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

def log_request_beautifully(method, path, original_model_display, mapped_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing model mapping."""
    original_display = f"{Colors.CYAN}{original_model_display}{Colors.RESET}" # Original uses Claude's color for now

    endpoint = path
    if "?" in endpoint: endpoint = endpoint.split("?")[0]

    target_provider = "unknown"
    target_model_name = mapped_model
    target_color = Colors.GREEN # Default

    if "/" in mapped_model:
        try:
            target_provider, target_model_name = mapped_model.split("/", 1)
            if target_provider == "openai": target_color = Colors.GREEN
            elif target_provider == "openrouter": target_color = Colors.GREEN
            elif target_provider == "gemini": target_color = Colors.YELLOW
            elif target_provider == "vertex_ai": target_color = Colors.BLUE
            # elif target_provider == "groq": target_color = Colors.MAGENTA # Removed Groq color
            elif target_provider == "xai": target_color = Colors.CYAN # Use Cyan for xAI
            elif target_provider == "anthropic": target_color = Colors.RED
        except ValueError:
            logger.warning(f"Could not parse provider from mapped model: {mapped_model}")
            target_provider = "unknown"
            target_model_name = mapped_model

    target_display = f"{target_color}{target_model_name}{Colors.RESET}"
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{original_display} → {target_display} ({target_provider}) {tools_str} {messages_str}"

    print(log_line)
    print(model_line)
    sys.stdout.flush()

# ====== Multi-API Key Support ======
def _parse_key_list(keys_str: str):
    """Parse key list supporting comma或换行分隔，并去除多余的反斜杠。"""
    if not keys_str:
        return []
    # 先替换换行为空格方便统一处理
    cleaned = keys_str.replace("\n", ",")
    parts = [p.strip().lstrip("\\").rstrip("\\") for p in cleaned.split(",")]
    return [p for p in parts if p]

# Pools for each provider
OPENAI_KEY_POOL = _parse_key_list(os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", "")))
GEMINI_KEY_POOL = _parse_key_list(os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")))
XAI_KEY_POOL = _parse_key_list(os.environ.get("XAI_API_KEYS", os.environ.get("XAI_API_KEY", "")))
OPENROUTER_KEY_POOL = _parse_key_list(os.environ.get("OPENROUTER_API_KEYS", os.environ.get("OPENROUTER_API_KEY", "")))

# Register pools for easy lookup
KEY_POOLS = {"openai": OPENAI_KEY_POOL, "openrouter": OPENROUTER_KEY_POOL, "gemini": GEMINI_KEY_POOL, "xai": XAI_KEY_POOL}

# Create cycling iterators for round-robin selection
ITERATORS = {provider: cycle(keys) for provider, keys in KEY_POOLS.items() if keys}

# Log key pool status for transparency
for _prov, _keys in KEY_POOLS.items():
    if _keys:
        logger.info(f"🔑 Loaded {_prov.upper()} key pool: {len(_keys)} keys (rotation enabled)")
    else:
        logger.warning(f"⚠️ No API keys found for provider '{_prov}'. Requests will fail unless keys are added.")

def get_api_key(provider: str):
    """Return the next API key for the given provider (round-robin)."""
    iterator = ITERATORS.get(provider)
    if iterator:
        return next(iterator)
    return None

# 现在挂载模型路由（get_api_key 已就绪）
try:
    from models_router import router as models_router
    import models_router as _mr
    _mr.get_api_key = get_api_key  # 注入真实轮询函数
    app.include_router(models_router)
    logger.info("📚 Model list endpoints mounted under /v1/models")
except Exception as e:
    logger.warning(f"Could not mount models router: {e}")

async def send_with_key_retry(req: Dict[str, Any], model_name: str, max_attempts: int = 5):
    """Send a request with automatic API-key rotation when failures occur."""
    provider = None
    if model_name.startswith("openai/"): provider = "openai"
    elif model_name.startswith("openrouter/"): provider = "openrouter"
    elif model_name.startswith("gemini/"): provider = "gemini"
    elif model_name.startswith("xai/"): provider = "xai"
    elif model_name.startswith("vertex_ai/"): provider = "vertex_ai"  # 新增 Vertex AI 识别

    attempts = 0
    tried_keys = set()
    current_key = None
    while attempts < max_attempts:
        attempts += 1
        # Pick/rotate key if pool exists
        if provider in KEY_POOLS and KEY_POOLS[provider]:
            current_key = get_api_key(provider)
            tried_keys.add(current_key)
            req["api_key"] = current_key
            obfuscated = f"{current_key[:6]}...{current_key[-4:]}" if len(current_key) > 10 else current_key
            logger.info(f"🔄 [{provider}] Attempt {attempts} using key: {obfuscated}")

        # Quick fail for unsupported Vertex
        if provider == "vertex_ai":
            if not importlib.util.find_spec("google.auth") or not VERTEX_PROJECT_ID:
                raise HTTPException(501, "Vertex AI not configured (missing google-auth or VERTEX_PROJECT_ID)")

        try:
            return await litellm.acompletion(**req)
        except Exception as e:
            msg = str(e)
            # 识别 Google key 被封
            if provider == "gemini" and ("PERMISSION_DENIED" in msg or "CONSUMER_SUSPENDED" in msg):
                logger.error(f"🚫 Key suspended, removing from pool: {obfuscated}")
                # 从池中移除
                if current_key in KEY_POOLS.get(provider, []):
                    KEY_POOLS[provider].remove(current_key)
                    ITERATORS[provider] = cycle(KEY_POOLS[provider]) if KEY_POOLS[provider] else None

            logger.warning(f"⚠️ Attempt {attempts} for provider {provider} failed: {e}")

            # Stop retrying if池空或已试完
            if provider not in KEY_POOLS or not KEY_POOLS[provider] or len(tried_keys) >= len(KEY_POOLS[provider]):
                raise
            continue

# ========== 自动预取模型列表并更新 ===========
async def _refresh_available_models():
    """Fetch model lists from providers at startup and update global constants."""
    try:
        from models_router import _fetch_openrouter_models, _fetch_gemini_models, _fetch_vertex_models
    except Exception as e:
        logger.warning(f"Model router not available for refresh: {e}")
        return

    global OPENAI_MODELS, GEMINI_MODELS, VERTEX_AI_MODELS

    try:
        openrouter_models = await _fetch_openrouter_models(refresh=True)
        if openrouter_models:
            OPENAI_MODELS = list(sorted(set(openrouter_models)))
            logger.info(f"🔄 OpenRouter models refreshed: {len(OPENAI_MODELS)} models")
    except Exception as e:
        logger.warning(f"Could not refresh OpenRouter models: {type(e).__name__} - {getattr(e, 'detail', str(e))}")

    try:
        gemini_models = await _fetch_gemini_models(refresh=True)
        if gemini_models:
            # Strip "models/" prefix and update the list
            GEMINI_MODELS[:] = list(sorted(set([m.replace("models/", "") for m in gemini_models])))
            logger.info(f"🔄 Gemini models refreshed: {len(GEMINI_MODELS)} models")
            # 同步更新最新变体
            global LATEST_GEMINI_PRO, LATEST_GEMINI_FLASH, LATEST_GEMINI_FLASH_LITE
            LATEST_GEMINI_PRO = _pick_latest_gemini("-pro") or LATEST_GEMINI_PRO
            LATEST_GEMINI_FLASH = _pick_latest_gemini("-flash") or LATEST_GEMINI_FLASH
            LATEST_GEMINI_FLASH_LITE = _pick_latest_gemini("flash-lite") or LATEST_GEMINI_FLASH_LITE
            logger.info(f"🟡 Gemini latest variants refreshed → pro:{LATEST_GEMINI_PRO}, flash:{LATEST_GEMINI_FLASH}, flash-lite:{LATEST_GEMINI_FLASH_LITE}")
    except Exception as e:
        logger.warning(f"Could not refresh Gemini models: {type(e).__name__} - {getattr(e, 'detail', str(e))}")

    try:
        if VERTEX_PROJECT_ID:
             vertex_models = await _fetch_vertex_models(VERTEX_PROJECT_ID, VERTEX_LOCATION, refresh=True)
             if vertex_models:
                # Vertex models can have complex paths, extract the final part
                cleaned_vertex = [m.split('/')[-1] for m in vertex_models]
                VERTEX_AI_MODELS[:] = list(sorted(set(cleaned_vertex)))
                logger.info(f"🔄 Vertex AI models refreshed: {len(VERTEX_AI_MODELS)} models")
    except Exception as e:
        logger.warning(f"Could not refresh Vertex models: {type(e).__name__} - {getattr(e, 'detail', str(e))}")

# 在 FastAPI 启动时启动后台任务
@app.on_event("startup")
async def startup_refresh_models():
    """Kick off model list refresh in background so第一条请求不被阻塞"""
    async def _background():
        logger.info("⏳ Refreshing model lists in background…")
        try:
            await _refresh_available_models()
            logger.info("✅ Model lists refreshed.")
        except Exception as e:
            logger.warning(f"Model list refresh failed: {e}")

    asyncio.create_task(_background())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
