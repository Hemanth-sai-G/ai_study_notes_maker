"""Create polished, in-memory PDFs for local study-material downloads."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

from app.models.exports import StudyExportRequest


def build_study_pdf(export: StudyExportRequest) -> bytes:
    """Render study material and code-derived sources without persisting user data."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=export.title,
        author="StudyMate AI",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "StudyMateTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=24,
        textColor=colors.HexColor("#244338"),
        alignment=TA_CENTER,
        spaceAfter=7 * mm,
    )
    body_style = ParagraphStyle(
        "StudyMateBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#27372F"),
        spaceAfter=4 * mm,
    )
    source_heading = ParagraphStyle(
        "SourceHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#526B32"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )
    source_style = ParagraphStyle(
        "SourceText",
        parent=body_style,
        fontSize=9,
        leading=13,
        leftIndent=4 * mm,
        textColor=colors.HexColor("#52605A"),
        bulletIndent=0,
    )

    story = [Paragraph(escape(export.title), title_style)]
    for paragraph in export.content.split("\n\n"):
        normalized = paragraph.strip().replace("\n", "<br/>")
        if normalized:
            story.append(Paragraph(escape(normalized).replace("&lt;br/&gt;", "<br/>"), body_style))

    sources = _group_sources(export)
    if sources:
        story.extend([Spacer(1, 2 * mm), Paragraph("Verified sources", source_heading)])
        story.append(KeepTogether([
            Paragraph(f"• {escape(source)}", source_style) for source in sources
        ]))

    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output.getvalue()


def _group_sources(export: StudyExportRequest) -> list[str]:
    grouped: dict[str, set[str]] = {}
    passage_counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for citation in export.citations:
        names[citation.document_id] = citation.document_name
        grouped.setdefault(citation.document_id, set())
        if citation.page_or_slide:
            label = "Slide" if citation.file_type == "PPTX" else "Page"
            grouped[citation.document_id].add(f"{label} {citation.page_or_slide}")
        passage_counts[citation.document_id] = passage_counts.get(citation.document_id, 0) + 1
    return [
        f"{names[document_id]}"
        f"{' - ' + ', '.join(sorted(locations)) if locations else ''}"
        f" ({passage_counts[document_id]} supporting passage{'s' if passage_counts[document_id] != 1 else ''})"
        for document_id, locations in grouped.items()
    ]


def _draw_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#738078"))
    canvas.drawString(18 * mm, 11 * mm, "StudyMate AI - generated locally")
    canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"Page {document.page}")
    canvas.restoreState()
