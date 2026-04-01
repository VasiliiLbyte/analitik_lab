"""Конфигурация приложения — загрузка из .env с валидацией через pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Все настройки загружаются из переменных окружения / .env файла."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        frozen=True,
    )

    telegram_bot_token: str = Field(min_length=1)
    telegram_proxy: str | None = None
    gigachat_credentials: str = Field(min_length=1)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def get_settings() -> Settings:
    """Фабрика настроек — вызывается один раз при старте приложения."""
    return Settings()  # type: ignore[call-arg]
