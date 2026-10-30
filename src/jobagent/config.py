"""Application configuration and settings."""

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings loaded from environment or .env file."""

    llm_api_key: str = ""
    model_small: str = ""
    model_strong: str = ""
    profile_path: Path = Path("data/profile.yaml")
    output_dir: Path = Path("data/output")
    database_path: Path = Path("data/tracker.db")


def get_settings(env_file: str | Path | None = None) -> Settings:
    """Load settings from environment variables and an optional .env file.

    Environment variables take precedence over values in the env file.
    If neither provides a value, sensible defaults are used.

    Args:
        env_file: Optional path to an env file. Defaults to '.env'.

    Returns:
        Configured Settings instance.
    """
    target_env = Path(env_file) if env_file is not None else Path(".env")
    file_vals: dict[str, str | None] = {}
    if target_env.is_file():
        file_vals = dotenv_values(dotenv_path=target_env)

    def resolve(env_key: str, default: Any) -> Any:
        if env_key in os.environ:
            return os.environ[env_key]
        if env_key in file_vals and file_vals[env_key] is not None:
            return file_vals[env_key]
        return default

    return Settings(
        llm_api_key=resolve("LLM_API_KEY", ""),
        model_small=resolve("LLM_MODEL_SMALL", ""),
        model_strong=resolve("LLM_MODEL_STRONG", ""),
        profile_path=resolve("PROFILE_PATH", Path("data/profile.yaml")),
        output_dir=resolve("OUTPUT_DIR", Path("data/output")),
        database_path=resolve("DATABASE_PATH", Path("data/tracker.db")),
    )
