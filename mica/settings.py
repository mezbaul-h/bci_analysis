"""
This module provides settings related to the application.
"""

from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DISABLE_PROGRESS_BAR: bool = False

    HF_DEVICE_MAP: str = "auto"
    HF_TOKEN: str = ""
    PACKAGE_ROOT_DIR: Path = Path(__file__).parent

    OUTPUT_DIR: Path = PACKAGE_ROOT_DIR / ".." / "output"

    TORCH_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
