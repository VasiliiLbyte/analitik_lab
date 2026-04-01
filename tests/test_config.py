"""Тесты для src/config.py — валидация настроек из окружения."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings


class TestSettingsValidation:
    """Проверяем, что обязательные переменные требуются, а значения по умолчанию корректны."""

    def test_valid_settings(self, env_vars: dict[str, str]) -> None:
        settings = Settings()  # type: ignore[call-arg]

        assert settings.telegram_bot_token == env_vars["TELEGRAM_BOT_TOKEN"]
        assert settings.telegram_proxy is None
        assert settings.gigachat_credentials == env_vars["GIGACHAT_CREDENTIALS"]
        assert settings.gigachat_scope == "GIGACHAT_API_PERS"
        assert settings.log_level == "DEBUG"

    def test_telegram_proxy_from_env(self, env_vars: dict[str, str]) -> None:
        proxy_url = "socks5://127.0.0.1:10808"
        with patch.dict(
            os.environ,
            {**env_vars, "TELEGRAM_PROXY": proxy_url},
            clear=True,
        ):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.telegram_proxy == proxy_url

    def test_missing_telegram_token_raises(self, env_vars: dict[str, str]) -> None:
        patched = {k: v for k, v in env_vars.items() if k != "TELEGRAM_BOT_TOKEN"}
        with patch.dict(os.environ, patched, clear=True):
            with pytest.raises(ValidationError, match="telegram_bot_token"):
                Settings(_env_file=None)  # type: ignore[call-arg]

    def test_missing_gigachat_credentials_raises(self, env_vars: dict[str, str]) -> None:
        patched = {k: v for k, v in env_vars.items() if k != "GIGACHAT_CREDENTIALS"}
        with patch.dict(os.environ, patched, clear=True):
            with pytest.raises(ValidationError, match="gigachat_credentials"):
                Settings(_env_file=None)  # type: ignore[call-arg]

    def test_default_log_level(self, env_vars: dict[str, str]) -> None:
        patched = {k: v for k, v in env_vars.items() if k != "LOG_LEVEL"}
        with patch.dict(os.environ, patched, clear=True):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.log_level == "INFO"

    def test_default_gigachat_scope(self, env_vars: dict[str, str]) -> None:
        patched = {k: v for k, v in env_vars.items() if k != "GIGACHAT_SCOPE"}
        with patch.dict(os.environ, patched, clear=True):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.gigachat_scope == "GIGACHAT_API_PERS"

    def test_settings_frozen(self, env_vars: dict[str, str]) -> None:
        settings = Settings()  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            settings.log_level = "ERROR"  # type: ignore[misc]

    def test_invalid_log_level_raises(self, env_vars: dict[str, str]) -> None:
        with patch.dict(os.environ, {**env_vars, "LOG_LEVEL": "INVALID"}, clear=True):
            with pytest.raises(ValidationError, match="log_level"):
                Settings()  # type: ignore[call-arg]
