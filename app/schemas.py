from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Union, Literal

from app.model_mapping import get_mapped_model_name

# --- Content Block Models ---
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

# --- Message and Tool Models ---
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

# --- Configuration Models ---
class ThinkingConfigClient(BaseModel):
    budget_tokens: Optional[int] = None
    type: Optional[str] = None

# --- Request Models ---
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
    thinking: Optional[ThinkingConfigClient] = None
    original_model: Optional[str] = None

    @field_validator('model')
    def validate_and_map_model(cls, v, info):
        original_model = v
        new_model = get_mapped_model_name(original_model)
        
        # Store the original model name in the object
        if info.data and isinstance(info.data, dict):
            info.data['original_model'] = original_model
            
        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfigClient] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator('model')
    def validate_and_map_model(cls, v, info):
        original_model = v
        new_model = get_mapped_model_name(original_model)

        # Store the original model name in the object
        if info.data and isinstance(info.data, dict):
            info.data['original_model'] = original_model

        return new_model

# --- Response Models ---
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
