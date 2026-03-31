"""Тесты для загрузки few-shot примеров КП из PDF."""

from __future__ import annotations

from pathlib import Path

from src.tools.knowledge_loader import load_kp_examples


def _make_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n")


def test_returns_none_for_missing_directory(tmp_path: Path) -> None:
    result = load_kp_examples(query_text="вода", examples_dir=tmp_path / "missing")
    assert result is None


def test_returns_all_examples_if_less_than_top_k(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("a.pdf", "b.pdf"):
        _make_pdf(tmp_path / name)

    def fake_extract(path: Path) -> str:
        return f"Пример для {path.stem}"

    monkeypatch.setattr("src.tools.knowledge_loader._extract_pdf_text", fake_extract)

    result = load_kp_examples(query_text="пример", examples_dir=tmp_path, top_k=3)

    assert result is not None
    assert "a.pdf" in result
    assert "b.pdf" in result


def test_selects_top_k_relevant_examples(tmp_path: Path, monkeypatch) -> None:
    for name in ("water.pdf", "soil.pdf", "air.pdf", "noise.pdf"):
        _make_pdf(tmp_path / name)

    texts = {
        "water.pdf": "анализ питьевой воды вода вода",
        "soil.pdf": "анализ почвы",
        "air.pdf": "анализ воздуха",
        "noise.pdf": "измерение шума",
    }

    def fake_extract(path: Path) -> str:
        return texts[path.name]

    monkeypatch.setattr("src.tools.knowledge_loader._extract_pdf_text", fake_extract)
    result = load_kp_examples(query_text="вода", examples_dir=tmp_path, top_k=2)

    assert result is not None
    assert result.count("### Example") == 2
    assert "water.pdf" in result


def test_skips_unreadable_or_empty_files(tmp_path: Path, monkeypatch) -> None:
    _make_pdf(tmp_path / "ok.pdf")
    _make_pdf(tmp_path / "bad.pdf")
    _make_pdf(tmp_path / "empty.pdf")

    def fake_extract(path: Path) -> str:
        if path.name == "bad.pdf":
            raise ValueError("broken")
        if path.name == "empty.pdf":
            return ""
        return "рабочий пример по воде"

    monkeypatch.setattr("src.tools.knowledge_loader._extract_pdf_text", fake_extract)
    result = load_kp_examples(query_text="вода", examples_dir=tmp_path)

    assert result is not None
    assert "ok.pdf" in result
    assert "bad.pdf" not in result
    assert "empty.pdf" not in result
