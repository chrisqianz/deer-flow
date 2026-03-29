import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deerflow.client import DeerFlowClient
from deerflow.config import get_app_config
from deerflow.config.openai_api_config import get_openai_api_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])
v1_router = APIRouter(prefix="/v1", tags=["openai"])

_client: DeerFlowClient | None = None


def _get_client() -> DeerFlowClient:
    global _client
    if _client is None:
        _client = DeerFlowClient()
    return _client


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

    client = _get_client()

    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(client, user_message, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await _create_non_streaming_completion(client, user_message, model_name)


async def _create_non_streaming_completion(
    client: DeerFlowClient,
    message: str,
    model_name: str,
) -> ChatCompletionResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(uuid.uuid1().time)

    try:
        full_content = ""
        tool_calls = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for event in client.stream(message, model_name=model_name):
            if event.type == "messages-tuple":
                data = event.data
                if data.get("type") == "ai":
                    if data.get("content"):
                        full_content += data["content"]
                    if data.get("tool_calls"):
                        tool_calls.extend(data["tool_calls"])
                    if data.get("usage_metadata"):
                        usage["prompt_tokens"] += data["usage_metadata"].get("input_tokens", 0)
                        usage["completion_tokens"] += data["usage_metadata"].get("output_tokens", 0)
                        usage["total_tokens"] += data["usage_metadata"].get("total_tokens", 0)

        message_obj = ChatMessage(role="assistant", content=full_content or None)
        if tool_calls:
            message_obj.tool_calls = tool_calls

        choice = ChatCompletionChoice(
            index=0,
            message=message_obj,
            finish_reason="stop" if not tool_calls else "tool_calls",
        )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[choice],
            usage=Usage(**usage),
        )

    except Exception as e:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat_completion(
    client: DeerFlowClient,
    message: str,
    model_name: str,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(uuid.uuid1().time)

    try:
        first_chunk = True
        tool_calls = []

        for event in client.stream(message, model_name=model_name):
            if event.type == "messages-tuple":
                data = event.data
                if data.get("type") == "ai":
                    content = data.get("content", "")

                    if content:
                        delta: dict[str, Any] = {"content": content}
                        if first_chunk:
                            delta["role"] = "assistant"
                            first_chunk = False

                        choice = StreamChoice(index=0, delta=delta, finish_reason=None)
                        chunk = StreamChatCompletionResponse(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[choice],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                    if data.get("tool_calls"):
                        for tc in data["tool_calls"]:
                            tool_calls.append(tc)
                            delta = {
                                "tool_calls": [
                                    {
                                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("name", ""),
                                            "arguments": json.dumps(tc.get("args", {})),
                                        },
                                    }
                                ]
                            }
                            choice = StreamChoice(index=0, delta=delta, finish_reason=None)
                            chunk = StreamChatCompletionResponse(
                                id=completion_id,
                                created=created,
                                model=model_name,
                                choices=[choice],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = StreamChoice(index=0, delta={}, finish_reason=finish_reason)
        chunk = StreamChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[choice],
        )
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in streaming chat completion")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
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

    client = _get_client()

    user_message = None
    if isinstance(request.input, str):
        user_message = request.input
    elif isinstance(request.input, list):
        for item in reversed(request.input):
            if isinstance(item, ResponseInputItem):
                if item.type == "message" and item.role == "user" and item.text:
                    user_message = item.text
                    break
            elif isinstance(item, dict):
                if item.get("type") == "message" and item.get("role") == "user":
                    user_message = item.get("text")
                    break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user input found")

    if request.stream:
        return StreamingResponse(
            _stream_response(client, user_message, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await _create_non_streaming_response(client, user_message, model_name)


async def _create_non_streaming_response(
    client: DeerFlowClient,
    message: str,
    model_name: str,
) -> Response:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    created = int(uuid.uuid1().time)

    try:
        full_content = ""
        tool_calls = []
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for event in client.stream(message, model_name=model_name):
            if event.type == "messages-tuple":
                data = event.data
                if data.get("type") == "ai":
                    if data.get("content"):
                        full_content += data["content"]
                    if data.get("tool_calls"):
                        tool_calls.extend(data["tool_calls"])
                    if data.get("usage_metadata"):
                        usage["input_tokens"] += data["usage_metadata"].get("input_tokens", 0)
                        usage["output_tokens"] += data["usage_metadata"].get("output_tokens", 0)
                        usage["total_tokens"] += data["usage_metadata"].get("total_tokens", 0)

        content_list = []
        if full_content:
            content_list.append({"type": "output_text", "text": full_content})

        for tc in tool_calls:
            content_list.append(
                {
                    "type": "function_call",
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(tc.get("args", {})),
                    "id": tc.get("id", ""),
                }
            )

        output = ResponseOutputText(
            id=response_id,
            content=content_list,
        )

        return Response(
            id=response_id,
            created=created,
            model=model_name,
            output=[output],
            usage=ResponseUsage(**usage),
        )

    except Exception as e:
        logger.exception("Error in response creation")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(
    client: DeerFlowClient,
    message: str,
    model_name: str,
):
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    created = int(uuid.uuid1().time)

    try:
        for event in client.stream(message, model_name=model_name):
            if event.type == "messages-tuple":
                data = event.data
                if data.get("type") == "ai":
                    content = data.get("content", "")
                    if content:
                        output_item = {
                            "id": f"msg_{uuid.uuid4().hex[:8]}",
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [{"type": "content_block", "text": content}],
                        }
                        chunk = StreamResponse(
                            id=response_id,
                            created=created,
                            model=model_name,
                            choices=[{"index": 0, "delta": output_item, "finish_reason": None}],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                    if data.get("tool_calls"):
                        for tc in data["tool_calls"]:
                            output_item = {
                                "id": f"msg_{uuid.uuid4().hex[:8]}",
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "function_call",
                                        "name": tc.get("name", ""),
                                        "arguments": json.dumps(tc.get("args", {})),
                                        "id": tc.get("id", ""),
                                    }
                                ],
                            }
                            chunk = StreamResponse(
                                id=response_id,
                                created=created,
                                model=model_name,
                                choices=[{"index": 0, "delta": output_item, "finish_reason": None}],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        final_chunk = StreamResponse(
            id=response_id,
            created=created,
            model=model_name,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in streaming response")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"
