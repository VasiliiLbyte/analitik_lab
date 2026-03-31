"""Тесты для генератора DOCX-файлов коммерческих предложений."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.schemas.state import ProposalData, ProposalItem
from src.tools.pdf_generator import _prepare_context, generate_proposal_docx

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "src" / "knowledge" / "proposal_template.docx"


@pytest.fixture()
def sample_proposal() -> ProposalData:
    return ProposalData(
        proposal_number="АЛ-2026-TEST",
        client_name="Тест Тестович",
        items=[
            ProposalItem(name="Анализ воды", params_count=14, price=8500),
            ProposalItem(name="Выезд", params_count=0, price=3000),
        ],
        total_price=11500,
        address="СПб, Невский 1",
        deadlines="3 рабочих дня",
    )


class TestPrepareContext:
    def test_formats_price_with_spaces(self, sample_proposal: ProposalData) -> None:
        ctx = _prepare_context(sample_proposal)
        assert ctx["total_price"] == "11 500"

    def test_formats_date(self, sample_proposal: ProposalData) -> None:
        ctx = _prepare_context(sample_proposal)
        expected = date.today().strftime("%d.%m.%Y")
        assert ctx["proposal_date"] == expected

    def test_default_client_name(self) -> None:
        data = ProposalData(proposal_number="АЛ-2026-001")
        ctx = _prepare_context(data)
        assert ctx["client_name"] == "Уважаемый клиент"

    def test_items_count(self, sample_proposal: ProposalData) -> None:
        ctx = _prepare_context(sample_proposal)
        assert len(ctx["items"]) == 2


class TestGenerateProposalDocx:
    def test_generates_file(self, sample_proposal: ProposalData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_proposal_docx(
                sample_proposal,
                template_path=TEMPLATE_PATH,
                output_dir=Path(tmpdir),
            )
            assert result.exists()
            assert result.suffix == ".docx"
            assert result.stat().st_size > 0

    def test_filename_contains_proposal_number(self, sample_proposal: ProposalData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_proposal_docx(
                sample_proposal,
                template_path=TEMPLATE_PATH,
                output_dir=Path(tmpdir),
            )
            assert "АЛ_2026_TEST" in result.name

    def test_raises_on_missing_template(self, sample_proposal: ProposalData) -> None:
        with pytest.raises(FileNotFoundError, match="Шаблон не найден"):
            generate_proposal_docx(
                sample_proposal,
                template_path=Path("/nonexistent/template.docx"),
            )

    def test_creates_output_dir_if_missing(self, sample_proposal: ProposalData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir"
            result = generate_proposal_docx(
                sample_proposal,
                template_path=TEMPLATE_PATH,
                output_dir=nested,
            )
            assert result.exists()
