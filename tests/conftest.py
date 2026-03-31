"""Общие фикстуры для всех тестов."""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture()
def env_vars() -> Iterator[dict[str, str]]:
    """Минимальный набор переменных окружения для валидного Settings."""
    values = {
        "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF-test-token",
        "GIGACHAT_CREDENTIALS": "test-gigachat-credentials",
        "GIGACHAT_SCOPE": "GIGACHAT_API_PERS",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, values, clear=False):
        yield values
