import json
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator, AsyncIterable, cast, Literal

from app.schemas import (
    MessagesRequest,
    MessagesResponse,
    Usage,
    ContentBlockText,
    ContentBlockToolUse,
)

logger = logging.getLogger(__name__)

# --- Schema and Content Helpers ---


def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini/Vertex."""
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        if schema.get("type") == "string" and "format" in schema:
            if schema["format"] not in {"enum", "date-time"}:
                logger.debug(
                    f"Removing unsupported format '{schema['format']}' for string type."
                )
                schema.pop("format")
        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema


def parse_tool_result_content(content: Any) -> str:
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Process list items, assuming they might be text blocks or other structures
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            return str(content)
    try:
        return str(content)
    except Exception:
        return "Unparseable content"


# --- Core Conversion Functions ---


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM's OpenAI-compatible format."""
    messages = []
    # System Prompt
    if anthropic_request.system:
        system_text = ""
        if isinstance(anthropic_request.system, str):
            system_text = anthropic_request.system
        elif isinstance(anthropic_request.system, list):
            system_text = "\n\n".join(
                b.text
                for b in anthropic_request.system
                if hasattr(b, "type") and b.type == "text"
            )
        if system_text:
            messages.append({"role": "system", "content": system_text.strip()})

    # Process Messages
    for msg in anthropic_request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
            continue

        processed_content, tool_calls, tool_results = [], [], []
        for block in msg.content:
            if block.type == "text":
                processed_content.append({"type": "text", "text": block.text})
            elif block.type == "image":
                source = getattr(block, "source", {})
                if isinstance(source, dict) and source.get("type") == "base64":
                    image_url = f"data:image/{source.get('media_type', 'jpeg')};base64,{source.get('data', '')}"
                    processed_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            elif block.type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": parse_tool_result_content(
                            getattr(block, "content", None)
                        ),
                    }
                )

        # Add to messages list
        if msg.role == "assistant":
            if processed_content or tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": processed_content,
                        "tool_calls": tool_calls or None,
                    }
                )
        else:  # user role
            if processed_content:
                messages.append({"role": "user", "content": processed_content})
        messages.extend(tool_results)

    # Final LiteLLM request structure
    litellm_request = {
        "model": anthropic_request.model,
        "messages": [m for m in messages if m.get("content") or m.get("tool_calls")],
        "max_tokens": anthropic_request.max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    # Optional parameters
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    # Tools
    is_gemini_family = anthropic_request.model.startswith(("vertex_ai/", "gemini/"))
    if anthropic_request.tools:
        litellm_request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": clean_gemini_schema(tool.input_schema)
                    if is_gemini_family
                    else tool.input_schema,
                },
            }
            for tool in anthropic_request.tools
        ]

    # Tool Choice
    if anthropic_request.tool_choice:
        choice_type = anthropic_request.tool_choice.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "required"
        elif choice_type == "tool":
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": anthropic_request.tool_choice["name"]},
            }

    return litellm_request


def convert_litellm_to_anthropic(
    litellm_response: Any, original_request: MessagesRequest
) -> MessagesResponse:
    """Convert a LiteLLM response object to the Anthropic MessagesResponse format."""
    try:
        if not hasattr(litellm_response, "choices"):
            litellm_response = litellm_response.model_dump()

        choice = litellm_response.choices[0]
        message = choice.message
        usage = litellm_response.usage

        content = []
        if message.content:
            content.append(ContentBlockText(type="text", text=message.content))
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": tc.function.arguments}
                content.append(
                    ContentBlockToolUse(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=arguments,
                    )
                )

        if not content:
            content.append(ContentBlockText(type="text", text=""))

        stop_reason_map = {
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "stop": "end_turn",
        }
        final_stop_reason = stop_reason_map.get(
            choice.finish_reason, choice.finish_reason or "end_turn"
        )

        stop_reason_literal = cast(
            Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]],
            final_stop_reason,
        )
        return MessagesResponse(
            id=litellm_response.id,
            model=original_request.original_model or original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason_literal,
            usage=Usage(
                input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens
            ),
        )
    except Exception as e:
        logger.error(
            f"Error converting LiteLLM response to Anthropic: {e}", exc_info=True
        )
        return MessagesResponse(
            id=f"msg_error_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[
                ContentBlockText(
                    type="text", text=f"Error converting backend response: {e}"
                )
            ],
            stop_reason=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )


async def handle_streaming(
    response_generator: AsyncIterable[Any], original_request: MessagesRequest
) -> AsyncGenerator[str, None]:
    """Handle streaming responses from LiteLLM and convert to Anthropic's event stream format."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = original_request.original_model or original_request.model

    # Message Start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'usage': {}}})}\n\n"
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    final_usage, final_stop_reason = {"input_tokens": 0, "output_tokens": 0}, "end_turn"
    tool_buffers: Dict[int, Dict[str, str]] = {}
    tool_block_indices: Dict[int, int] = {}
    text_block_index, current_block_index = -1, -1

    async for chunk in response_generator:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if chunk.usage:
            final_usage["input_tokens"] = getattr(chunk.usage, "prompt_tokens", 0)
            final_usage["output_tokens"] = getattr(chunk.usage, "completion_tokens", 0)

        if choice.finish_reason:
            stop_map = {
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "stop": "end_turn",
            }
            final_stop_reason = stop_map.get(
                choice.finish_reason, choice.finish_reason or "end_turn"
            )

        # Text delta
        if delta.content:
            if text_block_index == -1:
                current_block_index += 1
                text_block_index = current_block_index
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': delta.content}})}\n\n"

        # Tool delta
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if tc_delta.index not in tool_block_indices:
                    current_block_index += 1
                    anthropic_idx = current_block_index
                    tool_block_indices[tc_delta.index] = anthropic_idx
                    tool_buffers[anthropic_idx] = {
                        "id": tc_delta.id or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": tc_delta.function.name,
                        "input": "",
                    }
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_idx, 'content_block': {'type': 'tool_use', 'id': tool_buffers[anthropic_idx]['id'], 'name': tool_buffers[anthropic_idx]['name'], 'input': {}}})}\n\n"

                if tc_delta.function and tc_delta.function.arguments:
                    anthropic_idx = tool_block_indices[tc_delta.index]
                    tool_buffers[anthropic_idx]["input"] += tc_delta.function.arguments
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_idx, 'delta': {'type': 'input_json_delta', 'partial_json': tc_delta.function.arguments}})}\n\n"

    # Stop events
    if text_block_index != -1:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
    for idx in tool_block_indices.values():
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

    # Message Delta and Stop
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_stop_reason}, 'usage': {'output_tokens': final_usage['output_tokens']}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
