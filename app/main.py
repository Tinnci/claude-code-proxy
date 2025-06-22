# Import modules
from app.schemas import MessagesRequest, TokenCountRequest, TokenCountResponse
from app.converters import (
    convert_anthropic_to_litellm,
    convert_litellm_to_anthropic,
    handle_streaming,
)
from app.providers import send_with_key_retry
from app.startup import register_startup_events
from app.models_router import router as models_router

import json
import time
import logging
import traceback
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from typing import cast, AsyncIterable, Any

# --- Local Imports ---
# Initialize logging and configuration first
from app.logging_config import setup_logging, log_request_beautifully

setup_logging()


# --- App Initialization ---
app = FastAPI(title="Claude Code Proxy")
logger = logging.getLogger(__name__)

# Some versions of litellm may not expose token_counter in stubs
try:
    from litellm import token_counter  # type: ignore[attr-defined]
except ImportError:
    token_counter = None  # type: ignore


# --- Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log basic request details."""
    logger.debug(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    return response


# --- API Endpoints ---
@app.post("/v1/messages", response_model=None)  # Response model is dynamic
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        body = await raw_request.body()
        original_model = json.loads(body.decode("utf-8")).get("model", "unknown")

        litellm_request = convert_anthropic_to_litellm(request)

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            original_model,
            request.model,
            len(litellm_request.get("messages", [])),
            num_tools,
            200,
        )

        if request.stream:
            response_generator = await send_with_key_retry(litellm_request)
            return StreamingResponse(
                handle_streaming(cast(AsyncIterable[Any], response_generator), request),
                media_type="text/event-stream",
            )
        else:
            start_time = time.time()
            logger.info(f"🚀 Sending request to {request.model}...")
            litellm_response = await send_with_key_retry(litellm_request)
            elapsed = time.time() - start_time
            logger.info(f"✅ Response received from {request.model} in {elapsed:.2f}s")

            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            return anthropic_response

    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        status_code = getattr(e, "status_code", 500)
        detail = getattr(e, "message", str(e))
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="ignore")
        raise HTTPException(status_code=status_code, detail=str(detail))


@app.post("/v1/messages/count_tokens", response_model=TokenCountResponse)
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        temp_msg_request = MessagesRequest(**request.model_dump(), max_tokens=1)
        converted_request = convert_anthropic_to_litellm(temp_msg_request)

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            request.original_model or request.model,
            request.model,
            len(converted_request.get("messages", [])),
            num_tools,
            200,
        )

        if token_counter:
            token_count = token_counter(
                model=converted_request["model"], messages=converted_request["messages"]
            )
        else:
            # Fallback rough count
            token_count = sum(
                len(m.get("content", "")) // 4 for m in converted_request["messages"]
            )
        return TokenCountResponse(input_tokens=token_count)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


# --- Router and Event Registration ---
app.include_router(models_router)
logger.info("📚 Model list endpoints mounted under /v1/models")

register_startup_events(app)

# --- Main Execution ---
if __name__ == "__main__":
    # The app is now run via: uvicorn app.main:app --reload --host 0.0.0.0 --port 8082
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
