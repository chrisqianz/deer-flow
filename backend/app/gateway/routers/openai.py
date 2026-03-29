import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from deerflow.config import get_app_config
from deerflow.config.openai_api_config import get_openai_api_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])
v1_router = APIRouter(prefix="/v1", tags=["openai"])

_model_cache: dict = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        config = get_app_config()
        model_config = config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found")

        model_cls = _import_class(model_config.use)
        model = model_cls(model=model_config.model, **{k: v for k, v in model_config.model_dump().items() if k not in ["name", "display_name", "description", "model", "model_config"] and v is not None})
        _model_cache[model_name] = model
    return _model_cache[model_name]


def _import_class(class_path: str):
    """Import a class from a string like 'module.submodule:ClassName'"""
    if ":" in class_path:
        module_path, class_name = class_path.rsplit(":", 1)
    else:
        parts = class_path.rsplit(".", 1)
        module_path = parts[0]
        class_name = parts[1]

    importlib = __import__(module_path, fromlist=[class_name])
    return getattr(importlib, class_name)


class Message(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float | None = 0.7
    top_p: float | None = None
    n: int | None = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class StreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: str | None = None


class StreamChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


@v1_router.get("/models", response_model=ModelList)
async def list_models() -> ModelList:
    config = get_app_config()
    models = [Model(id=model.name, owned_by="deerflow") for model in config.models]
    return ModelList(data=models)


@v1_router.post("/chat/completions", response_model=None)
async def create_chat_completion(request: ChatCompletionRequest):
    config = get_openai_api_config()
    app_config = get_app_config()

    model_name = request.model or config.default_model or (app_config.models[0].name if app_config.models else None)
    if not model_name:
        raise HTTPException(status_code=400, detail="No model configured")

    model = _get_model(model_name)

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content or ""))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content or ""))

    if not lc_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(model, lc_messages, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await _create_non_streaming_completion(model, lc_messages, model_name)


async def _create_non_streaming_completion(
    model,
    lc_messages,
    model_name: str,
) -> ChatCompletionResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(uuid.uuid1().time)

    try:
        response = model.invoke(lc_messages)
        content = response.content if hasattr(response, "content") else str(response)

        usage = Usage(prompt_tokens=len(str(lc_messages)) // 4, completion_tokens=len(content) // 4, total_tokens=(len(str(lc_messages)) + len(content)) // 4)

        message_obj = ChatMessage(role="assistant", content=content)

        choice = ChatCompletionChoice(
            index=0,
            message=message_obj,
            finish_reason="stop",
        )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[choice],
            usage=usage,
        )

    except Exception as e:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat_completion(
    model,
    lc_messages,
    model_name: str,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(uuid.uuid1().time)

    try:
        first_chunk = True

        async for chunk in model.astream(lc_messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)

            if content:
                delta: dict[str, Any] = {"content": content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                choice = StreamChoice(index=0, delta=delta, finish_reason=None)
                chunk_obj = StreamChatCompletionResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[choice],
                )
                yield f"data: {chunk_obj.model_dump_json(exclude_none=True)}\n\n"

        choice = StreamChoice(index=0, delta={}, finish_reason="stop")
        chunk_obj = StreamChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[choice],
        )
        yield f"data: {chunk_obj.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("Error in streaming chat completion")
        yield f"data: {json.dumps({'error': {'message': 'Streaming error', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"


# ------------------------------------------------------------------
# Responses API (OpenAI 2024-05-01)
# ------------------------------------------------------------------


class ResponseInputItem(BaseModel):
    type: str = "message"
    role: str = "user"
    text: str | None = None
    image: str | list[dict] | None = None


class ResponseFunctionTool(BaseModel):
    type: str = "function"
    name: str
    description: str | None = None
    parameters: dict | None = None


class ResponseRequest(BaseModel):
    model: str | None = None
    input: list[ResponseInputItem] | str | None = None
    tools: list[ResponseFunctionTool] | None = None
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ResponseOutputText(BaseModel):
    type: str = "message"
    id: str
    status: str = "completed"
    role: str = "assistant"
    content: list[dict]


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Response(BaseModel):
    id: str
    object: str = "response"
    created: int
    model: str
    output: list[ResponseOutputText]
    usage: ResponseUsage


class StreamResponseOutput(BaseModel):
    type: str = "response.output_item"
    item: dict


class StreamResponse(BaseModel):
    id: str
    object: str = "response"
    created: int
    model: str
    choices: list[dict] = []


@router.post("/responses", response_model=None)
async def create_response(request: ResponseRequest):
    config = get_openai_api_config()
    app_config = get_app_config()

    model_name = request.model or config.default_model or (app_config.models[0].name if app_config.models else None)
    if not model_name:
        raise HTTPException(status_code=400, detail="No model configured")

    model = _get_model(model_name)

    from langchain_core.messages import HumanMessage, SystemMessage

    lc_messages = []
    if isinstance(request.input, str):
        lc_messages.append(HumanMessage(content=request.input))
    elif isinstance(request.input, list):
        for item in request.input:
            if isinstance(item, ResponseInputItem):
                if item.type == "message":
                    if item.role == "user" and item.text:
                        lc_messages.append(HumanMessage(content=item.text))
                    elif item.role == "system" and item.text:
                        lc_messages.append(SystemMessage(content=item.text))
            elif isinstance(item, dict):
                if item.get("type") == "message":
                    if item.get("role") == "user" and item.get("text"):
                        lc_messages.append(HumanMessage(content=item["text"]))
                    elif item.get("role") == "system" and item.get("text"):
                        lc_messages.append(SystemMessage(content=item["text"]))

    if not lc_messages:
        raise HTTPException(status_code=400, detail="No user input found")

    if request.stream:
        return StreamingResponse(
            _stream_response(model, lc_messages, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await _create_non_streaming_response(model, lc_messages, model_name)


async def _create_non_streaming_response(
    model,
    lc_messages,
    model_name: str,
) -> Response:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    created = int(uuid.uuid1().time)

    try:
        response = model.invoke(lc_messages)
        content = response.content if hasattr(response, "content") else str(response)

        input_tokens = len(str(lc_messages)) // 4
        output_tokens = len(content) // 4

        content_list = [{"type": "output_text", "text": content}]

        output = ResponseOutputText(
            id=response_id,
            content=content_list,
        )

        return Response(
            id=response_id,
            created=created,
            model=model_name,
            output=[output],
            usage=ResponseUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    except Exception as e:
        logger.exception("Error in response creation")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(
    model,
    lc_messages,
    model_name: str,
):
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    created = int(uuid.uuid1().time)

    try:
        async for chunk in model.astream(lc_messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)

            if content:
                output_item = {
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [{"type": "content_block", "text": content}],
                }
                chunk_obj = StreamResponse(
                    id=response_id,
                    created=created,
                    model=model_name,
                    choices=[{"index": 0, "delta": output_item, "finish_reason": None}],
                )
                yield f"data: {chunk_obj.model_dump_json(exclude_none=True)}\n\n"

        final_chunk = StreamResponse(
            id=response_id,
            created=created,
            model=model_name,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("Error in streaming response")
        yield f"data: {json.dumps({'error': {'message': 'Streaming error', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"
