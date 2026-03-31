"""Загрузка few-shot примеров КП из PDF-файлов."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
_DEFAULT_EXAMPLES_DIR = _KNOWLEDGE_DIR / "examples" / "kp"
_MAX_CHARS_PER_EXAMPLE = 1400
_MAX_TOTAL_CHARS = 4200


def _extract_pdf_text(pdf_path: Path) -> str:
    """Извлекает текст из PDF через pdfplumber или PyMuPDF."""
    try:
        import pdfplumber  # type: ignore

        pages: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception:
        try:
            import fitz  # type: ignore

            pages: list[str] = []
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    pages.append(page.get_text("text") or "")
            return "\n".join(pages).strip()
        except Exception as exc:
            logger.warning("Не удалось прочитать PDF {}: {}", pdf_path.name, exc)
            return ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _score_relevance(text: str, query_text: str) -> int:
    if not query_text:
        return 0
    hay = _normalize(text.lower())
    query_tokens = {t for t in re.split(r"[^\wа-яА-Я]+", query_text.lower()) if len(t) > 2}
    return sum(hay.count(token) for token in query_tokens)


def _safe_examples_dir(examples_dir: Path) -> Path:
    """Нормализует путь к директории примеров."""
    return examples_dir.resolve()


def load_kp_examples(
    query_text: str = "",
    *,
    examples_dir: Path | None = None,
    top_k: int = 3,
) -> str | None:
    """Загружает 2-3 релевантных PDF-примера и возвращает few-shot строку."""
    if top_k <= 0:
        return None

    target_dir = _safe_examples_dir(examples_dir or _DEFAULT_EXAMPLES_DIR)
    if not target_dir.exists() or not target_dir.is_dir():
        logger.info("Папка примеров КП не найдена: {}", target_dir)
        return None

    candidates: list[tuple[int, Path, str]] = []
    for pdf_path in sorted(target_dir.glob("*.pdf")):
        try:
            text = _extract_pdf_text(pdf_path)
        except Exception as exc:
            logger.warning("Пропускаем PDF {} из-за ошибки чтения: {}", pdf_path.name, exc)
            continue
        if not text:
            continue
        snippet = _normalize(text)[:_MAX_CHARS_PER_EXAMPLE]
        if not snippet:
            continue
        score = _score_relevance(snippet, query_text)
        candidates.append((score, pdf_path, snippet))

    if not candidates:
        return None

    if len(candidates) <= top_k:
        selected = candidates
    else:
        selected = sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True)[:top_k]

    blocks: list[str] = []
    total_chars = 0
    for idx, (_, path, snippet) in enumerate(selected, start=1):
        block = f"### Example {idx} ({path.name})\n{snippet}"
        if total_chars + len(block) > _MAX_TOTAL_CHARS:
            break
        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks) if blocks else None
