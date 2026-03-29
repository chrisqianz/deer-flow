import os

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """Configuration for the API Gateway."""

    host: str = Field(default="0.0.0.0", description="Host to bind the gateway server")
    port: int = Field(default=8001, description="Port to bind the gateway server")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"], description="Allowed CORS origins")
    openai_compatible_enabled: bool = Field(default=False, description="Enable OpenAI-compatible API endpoints")
    openai_compatible_host: str = Field(default="127.0.0.1", description="Host for OpenAI-compatible API")
    openai_compatible_port: int = Field(default=8000, description="Port for OpenAI-compatible API")
    openai_compatible_default_model: str | None = Field(default=None, description="Default model for OpenAI-compatible API")


_gateway_config: GatewayConfig | None = None


def get_gateway_config() -> GatewayConfig:
    """Get gateway config, loading from environment if available."""
    global _gateway_config
    if _gateway_config is None:
        cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        _gateway_config = GatewayConfig(
            host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
            port=int(os.getenv("GATEWAY_PORT", "8001")),
            cors_origins=cors_origins_str.split(","),
            openai_compatible_enabled=os.getenv("OPENAI_COMPATIBLE_ENABLED", "false").lower() == "true",
            openai_compatible_host=os.getenv("OPENAI_COMPATIBLE_HOST", "127.0.0.1"),
            openai_compatible_port=int(os.getenv("OPENAI_COMPATIBLE_PORT", "8000")),
            openai_compatible_default_model=os.getenv("OPENAI_COMPATIBLE_DEFAULT_MODEL"),
        )
    return _gateway_config
