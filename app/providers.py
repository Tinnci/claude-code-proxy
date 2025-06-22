import os
import logging
from itertools import cycle
from typing import Dict, Any, Set
from fastapi import HTTPException

import litellm
from app.config import (
    OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY,
    VERTEX_PROJECT_ID, VERTEX_LOCATION
)

logger = logging.getLogger(__name__)

# ====== Multi-API Key Support ======
def _parse_key_list(keys_str: str):
    """Parse key list from a string (comma or newline separated)."""
    if not keys_str: 
        return []
    cleaned = keys_str.replace("\n", ",")
    return [p.strip() for p in cleaned.split(",") if p.strip()]

KEY_POOLS = {
    "openai": _parse_key_list(os.environ.get("OPENAI_API_KEYS", OPENAI_API_KEY or "")),
    "gemini": _parse_key_list(os.environ.get("GEMINI_API_KEYS", GEMINI_API_KEY or "")),
    "xai": _parse_key_list(os.environ.get("XAI_API_KEYS", XAI_API_KEY or "")),
    "openrouter": _parse_key_list(os.environ.get("OPENROUTER_API_KEYS", OPENROUTER_API_KEY or "")),
}

ITERATORS = {provider: cycle(keys) for provider, keys in KEY_POOLS.items() if keys}

for provider, keys in KEY_POOLS.items():
    if keys:
        logger.info(f"🔑 Loaded {provider.upper()} key pool: {len(keys)} keys (rotation enabled)")
    else:
        logger.warning(f"⚠️ No API keys found for provider '{provider}'.")

def get_api_key(provider: str):
    """Return the next API key for the given provider using round-robin."""
    return next(ITERATORS[provider], None) if provider in ITERATORS else None

def _prepare_litellm_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares the LiteLLM request by adding API keys and provider-specific credentials."""
    model_name = req.get("model", "")
    
    if model_name.startswith("openai/"):
        req["api_key"] = get_api_key("openai")
    elif model_name.startswith("openrouter/"):
        req["api_key"] = get_api_key("openrouter")
        req["api_base"] = "https://openrouter.ai/api/v1"
    elif model_name.startswith("gemini/"):
        req["api_key"] = get_api_key("gemini")
    elif model_name.startswith("xai/"):
        req["api_key"] = get_api_key("xai")
    elif model_name.startswith("vertex_ai/"):
        req["vertex_project"] = VERTEX_PROJECT_ID
        req["vertex_location"] = VERTEX_LOCATION

    # Add global timeout and retries
    req['timeout'] = 60
    req['max_retries'] = 2 # LiteLLM's internal retry, not the key rotation one
    return req

async def send_with_key_retry(req: Dict[str, Any], max_attempts: int = 5):
    """Send a request with automatic API-key rotation on failure."""
    # 从模型名前缀推断出 provider，例如 "openai/gpt-4o" ➜ "openai"
    provider: str | None = req.get("model", "").split("/")[0] if "/" in req.get("model", "") else None
    
    attempts = 0
    tried_keys: Set[str] = set()
    last_exception = None

    # 当 provider 为 None 时，认为没有可用的 key
    provider_keys = KEY_POOLS.get(provider, []) if provider else []
    effective_max_attempts = min(max_attempts, len(provider_keys) if provider_keys else 1)

    while attempts < effective_max_attempts:
        attempts += 1
        current_key = None
        
        # Prepare request with the next available key
        prepared_req = _prepare_litellm_request(req.copy())
        
        if provider in ITERATORS:
            current_key = prepared_req.get("api_key")
            if not current_key or current_key in tried_keys:
                # If we've tried all keys, stop
                if len(tried_keys) >= len(provider_keys):
                    logger.error(f"🚫 All keys for provider '{provider}' have failed.")
                    break
                # This should not happen with cycle, but as a safeguard:
                continue
            
            tried_keys.add(current_key)
            obfuscated = f"{current_key[:6]}...{current_key[-4:]}" if current_key and len(current_key) > 10 else current_key
            logger.info(f"🔄 [{provider.upper()}] Attempt {attempts}/{effective_max_attempts} using key: {obfuscated}")

        try:
            return await litellm.acompletion(**prepared_req)
        except Exception as e:
            last_exception = e
            logger.warning(f"⚠️ Attempt {attempts} failed for provider '{provider}': {e}")
            
            # Specific error handling for suspended keys
            msg = str(e)
            if provider == "gemini" and ("PERMISSION_DENIED" in msg or "CONSUMER_SUSPENDED" in msg):
                if current_key and current_key in KEY_POOLS["gemini"]:
                    logger.error(f"🚫 Gemini key suspended, removing from pool: {obfuscated}")
                    KEY_POOLS["gemini"].remove(current_key)
                    # 若移除后仍有剩余 key，重新生成轮询器；否则从 ITERATORS 删除该 provider
                    if KEY_POOLS["gemini"]:
                        ITERATORS["gemini"] = cycle(KEY_POOLS["gemini"])
                    else:
                        ITERATORS.pop("gemini", None)
    
    logger.error(f"❌ All {attempts} attempts failed for provider '{provider}'.")
    raise last_exception or HTTPException(status_code=500, detail="All attempts to contact the provider failed.")
