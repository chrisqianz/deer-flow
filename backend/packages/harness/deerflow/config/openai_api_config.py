import os

from pydantic import BaseModel, Field


class OpenAICompatibleConfig(BaseModel):
    """Configuration for the OpenAI-compatible API server."""

    enabled: bool = Field(
        default=False,
        description="Enable the OpenAI-compatible API server",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind the OpenAI-compatible API server",
    )
    port: int = Field(
        default=8000,
        description="Port to bind the OpenAI-compatible API server",
    )
    default_model: str | None = Field(
        default=None,
        description="Default model to use when not specified in request",
    )


_openai_compatible_config: OpenAICompatibleConfig | None = None


def get_openai_api_config() -> OpenAICompatibleConfig:
    """Get OpenAI-compatible API config, loading from environment if available."""
    global _openai_compatible_config
    if _openai_compatible_config is None:
        _openai_compatible_config = OpenAICompatibleConfig(
            enabled=os.getenv("OPENAI_COMPATIBLE_ENABLED", "false").lower() == "true",
            host=os.getenv("OPENAI_COMPATIBLE_HOST", "127.0.0.1"),
            port=int(os.getenv("OPENAI_COMPATIBLE_PORT", "8000")),
            default_model=os.getenv("OPENAI_COMPATIBLE_DEFAULT_MODEL"),
        )
    return _openai_compatible_config
