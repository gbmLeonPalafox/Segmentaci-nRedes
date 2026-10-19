"""Read server-only settings on demand so keys can be added without restarting."""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(default="", repr=False, exclude=True)
    model: str = Field(default="gpt-5.6-terra", min_length=1, max_length=150)
    timeout_seconds: int = Field(default=180, ge=10, le=600)
    max_input_chars: int = Field(default=32000, ge=4000, le=100000)


class HighlightOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_count: int = Field(default=5, ge=1, le=15)
    min_seconds: int = Field(default=20, ge=1, le=300)
    max_seconds: int = Field(default=90, ge=5, le=600)
    objective: str = Field(
        default="Select the most useful insights, memorable stories, and actionable takeaways for social media. "
        "Each passage should make sense on its own and communicate a complete idea.",
        min_length=10,
        max_length=2000,
    )

    def check_range(self):
        if self.min_seconds > self.max_seconds:
            raise ValueError("Minimum highlight duration must not exceed maximum duration.")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    highlights: HighlightOptions = Field(default_factory=HighlightOptions)


def load_settings(path=None):
    path = Path(path or os.getenv("APP_CONFIG_FILE", PROJECT_ROOT / "config.toml"))
    try:
        content = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        settings = Settings.model_validate(content)
        settings.highlights.check_range()
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        # Validation/TOML errors can contain source values, including credentials.
        raise ValueError(
            "Check config.toml: invalid TOML, unknown settings, or invalid highlight limits."
        ) from exc
    settings.llm.api_key = os.getenv("OPENAI_API_KEY", settings.llm.api_key).strip()
    if settings.llm.api_key.lower() in {"your_api_key_here", "your-openai-api-key", "sk-..."}:
        settings.llm.api_key = ""
    return settings
