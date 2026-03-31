"""Генерация коммерческого предложения из DOCX-шаблона (docxtpl + Jinja2).

MVP: генерируем DOCX. PDF-конвертация через LibreOffice CLI — опциональна.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from docxtpl import DocxTemplate
from loguru import logger

from src.schemas.state import ProposalData

_DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "knowledge" / "proposal_template.docx"


def _prepare_context(data: ProposalData) -> dict:
    """Преобразует ProposalData в контекст для Jinja2-шаблона."""
    return {
        "proposal_number": data.proposal_number,
        "client_name": data.client_name or "Уважаемый клиент",
        "proposal_date": data.proposal_date.strftime("%d.%m.%Y"),
        "items": [
            {
                "name": item.name,
                "params_count": item.params_count,
                "price": f"{item.price:,.0f}".replace(",", " "),
            }
            for item in data.items
        ],
        "total_price": f"{data.total_price:,.0f}".replace(",", " "),
        "validity_days": data.validity_days,
        "address": data.address or "—",
        "deadlines": data.deadlines or "по согласованию",
    }


def generate_proposal_docx(
    data: ProposalData,
    *,
    template_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Рендерит DOCX из шаблона и возвращает путь к файлу.

    Args:
        data: структурированные данные КП
        template_path: путь к .docx-шаблону (по умолчанию — knowledge/proposal_template.docx)
        output_dir: папка для результата (по умолчанию — системная temp)

    Returns:
        Path к сгенерированному .docx файлу
    """
    template_path = template_path or _DEFAULT_TEMPLATE
    if not template_path.exists():
        raise FileNotFoundError(f"Шаблон не найден: {template_path}")

    context = _prepare_context(data)
    doc = DocxTemplate(str(template_path))
    doc.render(context)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="analitik_lab_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"KP_{data.proposal_number.replace('-', '_')}.docx"
    output_path = output_dir / filename
    doc.save(str(output_path))
    logger.info("КП сгенерировано: {}", output_path)
    return output_path


def convert_docx_to_pdf(docx_path: Path) -> Path | None:
    """Конвертирует DOCX → PDF через LibreOffice CLI. Возвращает None, если недоступен."""
    try:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(docx_path.parent),
                str(docx_path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        pdf_path = docx_path.with_suffix(".pdf")
        if pdf_path.exists():
            logger.info("PDF сконвертирован: {}", pdf_path)
            return pdf_path
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("LibreOffice недоступен — возвращаем DOCX")
    return None
