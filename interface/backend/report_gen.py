"""
PDF Report Generator (ReportLab)
Generates a clinical-grade report PDF with:
  - Institution header (logo + name)
  - Patient demographics
  - Centiloid score with risk category + interpretation
  - MRI slices (axial / coronal / sagittal)
  - Time series table (optional)
  - DEMO watermark if applicable
"""
from __future__ import annotations

import base64
import io
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

# ── Unicode font support ─────────────────────────────────────────────────────
# ReportLab's built-in Helvetica does not render Turkish characters reliably.
# DejaVu Sans is embedded into the PDF so characters such as ğ, ü, ş, ı, İ, ö, ç
# appear correctly on every machine.
_FONT_REGULAR = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"


def _register_unicode_fonts() -> None:
    global _FONT_REGULAR, _FONT_BOLD

    candidates = [
        # Linux / Docker
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        # macOS
        (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
        # Windows
        (
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ),
    ]

    for regular_path, bold_path in candidates:
        regular = Path(regular_path)
        bold = Path(bold_path)
        if not regular.exists():
            continue

        pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular)))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold if bold.exists() else regular)))
        registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold="DejaVuSans-Bold",
            italic="DejaVuSans",
            boldItalic="DejaVuSans-Bold",
        )
        _FONT_REGULAR = "DejaVuSans"
        _FONT_BOLD = "DejaVuSans-Bold"
        return


_register_unicode_fonts()


# Colour palette matching the UI
_NAVY      = colors.HexColor("#0b1f3a")
_NAVY_MID  = colors.HexColor("#132d52")
_ACCENT    = colors.HexColor("#2563eb")
_TEXT      = colors.HexColor("#0f172a")
_TEXT2     = colors.HexColor("#334155")
_TEXT3     = colors.HexColor("#64748b")
_BG_LIGHT  = colors.HexColor("#f4f6fb")
_BORDER    = colors.HexColor("#e2e8f0")

_NEG_COL   = colors.HexColor("#16a34a")
_BORDER_COL= colors.HexColor("#d97706")
_ELEV_COL  = colors.HexColor("#ea580c")
_HIGH_COL  = colors.HexColor("#dc2626")

_RISK_COLOURS = {
    "negative":  _NEG_COL,
    "borderline":_BORDER_COL,
    "elevated":  _ELEV_COL,
    "high":      _HIGH_COL,
}

W, H = A4   # 595.27 x 841.89 pts


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(
    report_id: str,
    patient_id: str,
    patient_name: str | None,
    birth_year: int,
    sex: str,
    scan_date: str,
    analysis_date: str,
    requesting_clinician: str,
    clinician_note: str | None,
    centiloid: float,
    confidence_low: float,
    confidence_high: float,
    risk_category: str,
    risk_label: str,
    interpretation: str,
    model_version: str = "LateFusion-v1.0",
    processing_time_s: float = 0.0,
    institution_name: str = "NeuroAnalyse",
    institution_logo_b64: str | None = None,
    slice_axial: str | None = None,
    slice_coronal: str | None = None,
    slice_sagittal: str | None = None,
    previous_analyses: list[dict] | None = None,
    is_demo: bool = False,
    language: str = "tr",
) -> bytes:

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    L = _LABELS[language]

    story: list = []

    # ── Header ──────────────────────────────────────────────────────────────
    story += _header(institution_name, institution_logo_b64, report_id, analysis_date, L)
    story.append(HRFlowable(width="100%", thickness=1.5, color=_NAVY, spaceAfter=10))

    # ── Patient info ─────────────────────────────────────────────────────────
    story += _patient_block(patient_id, patient_name, birth_year, sex, scan_date,
                             requesting_clinician, clinician_note, L)
    story.append(Spacer(1, 8))

    # ── Centiloid gauge ───────────────────────────────────────────────────────
    story += _gauge_block(centiloid, confidence_low, confidence_high,
                           risk_category, risk_label, L)
    story.append(Spacer(1, 8))

    # ── Interpretation ────────────────────────────────────────────────────────
    story += _interpretation_block(interpretation, risk_category, L)
    story.append(Spacer(1, 10))

    # ── MRI slices ────────────────────────────────────────────────────────────
    if slice_axial or slice_coronal or slice_sagittal:
        story += _slices_block(slice_axial, slice_coronal, slice_sagittal, L)
        story.append(Spacer(1, 10))

    # ── Time series ───────────────────────────────────────────────────────────
    if previous_analyses:
        story += _history_block(previous_analyses, centiloid, scan_date, L)
        story.append(Spacer(1, 10))

    def _on_page(canvas, doc):
        _page_footer(canvas, doc, report_id, institution_name, L)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


# ── Story builders ─────────────────────────────────────────────────────────────

def _header(institution_name, logo_b64, report_id, analysis_date, L):
    rows = []
    logo_cell = _default_logo_mark()
    if logo_b64:
        try:
            raw = base64.b64decode(logo_b64.split(",")[-1])
            img = Image(io.BytesIO(raw), width=2 * cm, height=2 * cm)
            logo_cell = img
        except Exception:
            pass

    date_fmt = _fmt_date(analysis_date)
    header_data = [[
        logo_cell,
        Paragraph(f"<b>{institution_name}</b>", _style("title")),
        Paragraph(
            f"{L['report_id']}: <b>{report_id}</b><br/>{L['date']}: {date_fmt}",
            _style("small_right"),
        ),
    ]]
    t = Table(header_data, colWidths=[2.5 * cm, None, 5 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [t]


def _default_logo_mark():
    mark = Table(
        [[Paragraph("<b>NA</b>", ParagraphStyle(
            "logo_text",
            fontName=_FONT_BOLD,
            fontSize=16,
            textColor=colors.white,
            alignment=TA_CENTER,
            leading=18,
        ))]],
        colWidths=[1.55 * cm],
        rowHeights=[1.55 * cm],
    )
    mark.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.25, _ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return mark


def _patient_block(patient_id, patient_name, birth_year, sex, scan_date,
                   clinician, note, L):
    sex_label = {"M": L["male"], "F": L["female"]}.get(sex, sex)
    rows = [
        [L.get("patient_name", "Hasta Adı"), patient_name or "—"],
        [L["patient_id"],   patient_id],
        [L["birth_year"],   str(birth_year)],
        [L["sex"],          sex_label],
        [L["scan_date"],    _fmt_date(scan_date)],
        [L["clinician"],    clinician],
    ]
    if note:
        rows.append([L["note"], note])

    data = [[Paragraph(f"<b>{r}</b>", _style("label")),
             Paragraph(str(v), _style("value"))] for r, v in rows]

    t = Table(data, colWidths=[4.5 * cm, None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _BG_LIGHT),
        ("TEXTCOLOR",  (0, 0), (0, -1), _TEXT2),
        ("TEXTCOLOR",  (1, 0), (1, -1), _TEXT),
        ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return [Paragraph(L["patient_info"], _style("section_title")), t]


def _gauge_block(centiloid, ci_low, ci_high, risk_cat, risk_label, L):
    colour = _RISK_COLOURS.get(risk_cat, _TEXT)

    score_para = Paragraph(
        f'<font size="28" color="{colour.hexval()}"><b>{centiloid:.1f} CL</b></font>',
        _style("centered"),
    )
    ci_para = Paragraph(
        f'95% CI: [{ci_low:.1f} – {ci_high:.1f} CL]',
        _style("small_center"),
    )
    risk_para = Paragraph(
        f'<font color="{colour.hexval()}"><b>{risk_label}</b></font>',
        _style("risk_label"),
    )

    # Horizontal bar
    thresholds_para = Paragraph(
        '<font size="8" color="#64748b">0 CL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;25 CL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;50 CL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;100 CL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;130 CL</font>',
        _style("small_center"),
    )

    data = [[score_para, Spacer(1, 1), risk_para]]
    t = Table(data, colWidths=[None, 1 * cm, None])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",  (0, 0), (0, -1), "CENTER"),
        ("ALIGN",  (2, 0), (2, -1), "CENTER"),
    ]))

    return [
        Paragraph(L["result_heading"], _style("section_title")),
        t,
        ci_para,
    ]


def _interpretation_block(interpretation, risk_cat, L):
    colour = _RISK_COLOURS.get(risk_cat, _TEXT)
    bg = colors.HexColor("#f0fdf4") if risk_cat == "negative" else colors.HexColor("#fff7ed")
    t = Table(
        [[Paragraph(interpretation, _style("interp"))]],
        colWidths=[None],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEAFTER",  (0, 0), (0, -1), 3, colour),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    return [Paragraph(L["interpretation_heading"], _style("section_title")), t]


def _slices_block(ax, cor, sag, L):
    def _img(b64):
        if not b64:
            return Paragraph("—", _style("small_center"))
        try:
            raw = base64.b64decode(b64.split(",")[-1])
            return Image(io.BytesIO(raw), width=4.2 * cm, height=4.2 * cm)
        except Exception:
            return Paragraph("—", _style("small_center"))

    rows = [
        [L["axial"], L["coronal"], L["sagittal"]],
        [_img(ax),   _img(cor),    _img(sag)],
    ]
    t = Table(rows, colWidths=[None, None, None])
    t.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  _TEXT3),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [Paragraph(L["mri_views"], _style("section_title")), t]


def _history_block(prev_analyses, current_cl, current_scan_date, L):
    header = [L["scan_date"], L["centiloid"], L["risk_category"]]
    rows_data = [header]
    for a in prev_analyses:
        cat = a.get("risk_category", "")
        c = _RISK_COLOURS.get(cat, _TEXT)
        rows_data.append([
            _fmt_date(a.get("scan_date", "")),
            f"{a.get('centiloid', 0):.1f} CL",
            a.get("risk_label", cat),
        ])
    # Current
    cat = _label_from_cl(current_cl, L)
    rows_data.append([
        f"{_fmt_date(current_scan_date)} ({L['current']})",
        f"{current_cl:.1f} CL",
        cat,
    ])

    t = Table(rows_data, colWidths=[4 * cm, 3 * cm, None])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), _NAVY_MID),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.5, _BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_BG_LIGHT, colors.white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("FONTNAME",    (0, -1), (-1, -1), _FONT_BOLD),
    ]))
    return [Paragraph(L["history_heading"], _style("section_title")), t]


def _metadata_block(model_version, processing_time_s, L):
    rows = [
        [L["model_version"], model_version],
        [L["processing_time"], f"{processing_time_s:.1f} s"],
        [L["analysis_system"], "NeuroAnalyse v1.0"],
    ]
    data = [[Paragraph(f"<b>{r}</b>", _style("label")),
             Paragraph(str(v), _style("value"))] for r, v in rows]
    t = Table(data, colWidths=[4.5 * cm, None])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), _TEXT3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t]


# ── Canvas callbacks ──────────────────────────────────────────────────────────

def _watermark(canvas, text):
    canvas.saveState()
    canvas.setFont(_FONT_BOLD, 40)
    canvas.setFillColorRGB(0.9, 0.1, 0.1, alpha=0.12)
    canvas.rotate(45)
    for dy in range(-200, 1200, 200):
        canvas.drawCentredString(W // 2 + dy * 0.5, dy, text)
    canvas.restoreState()


def _page_footer(canvas, doc, report_id, institution_name, L):
    canvas.saveState()
    canvas.setFont(_FONT_REGULAR, 7)
    canvas.setFillColor(_TEXT3)
    canvas.drawString(2 * cm, 1.2 * cm, f"{institution_name}  |  {report_id}")
    canvas.drawRightString(
        W - 2 * cm, 1.2 * cm,
        f"{L['page']} {doc.page}",
    )
    canvas.restoreState()


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fmt_date(iso: str) -> str:
    try:
        d = datetime.strptime(iso[:10], "%Y-%m-%d")
        return d.strftime("%d.%m.%Y")
    except Exception:
        return iso


def _label_from_cl(cl: float, L: dict | None = None) -> str:
    if L:
        if cl < 25:   return L.get("risk_negative", "Negatif")
        if cl < 50:   return L.get("risk_borderline", "Sınırda")
        if cl < 100:  return L.get("risk_elevated", "Yüksek")
        return L.get("risk_high", "Çok Yüksek")
    if cl < 25:   return "Negative"
    if cl < 50:   return "Borderline"
    if cl < 100:  return "Elevated"
    return "High"


def _style(name: str) -> ParagraphStyle:
    styles = getSampleStyleSheet()
    base = styles["Normal"]
    base.fontName = _FONT_REGULAR

    variants = {
        "title": ParagraphStyle(
            "title", parent=base, fontSize=16, textColor=_NAVY,
            leading=20, fontName=_FONT_BOLD,
        ),
        "section_title": ParagraphStyle(
            "section_title", parent=base, fontSize=10, textColor=_NAVY_MID,
            leading=14, fontName=_FONT_BOLD, spaceBefore=10, spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "label", parent=base, fontSize=9, textColor=_TEXT2,
            leading=12, fontName=_FONT_BOLD,
        ),
        "value": ParagraphStyle(
            "value", parent=base, fontSize=9, textColor=_TEXT,
            leading=12, fontName=_FONT_REGULAR,
        ),
        "centered": ParagraphStyle(
            "centered", parent=base, alignment=TA_CENTER,
            fontName=_FONT_REGULAR,
        ),
        "small_center": ParagraphStyle(
            "small_center", parent=base, fontSize=8, textColor=_TEXT3,
            alignment=TA_CENTER, fontName=_FONT_REGULAR,
        ),
        "small_right": ParagraphStyle(
            "small_right", parent=base, fontSize=8, textColor=_TEXT2,
            alignment=TA_RIGHT, fontName=_FONT_REGULAR,
        ),
        "risk_label": ParagraphStyle(
            "risk_label", parent=base, fontSize=12, alignment=TA_CENTER,
            fontName=_FONT_BOLD,
        ),
        "interp": ParagraphStyle(
            "interp", parent=base, fontSize=9, textColor=_TEXT,
            leading=14, fontName=_FONT_REGULAR,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", parent=base, fontSize=7.5, textColor=_TEXT3,
            leading=11, alignment=TA_CENTER, fontName=_FONT_REGULAR,
        ),
    }
    return variants.get(name, base)

# ── Translations ──────────────────────────────────────────────────────────────

_LABELS = {
    "tr": {
        "report_id":             "Rapor No",
        "date":                  "Tarih",
        "patient_info":          "Hasta Bilgileri",
        "patient_name":          "Hasta Adı",
        "patient_id":            "Hasta ID",
        "birth_year":            "Doğum Yılı",
        "sex":                   "Cinsiyet",
        "male":                  "Erkek",
        "female":                "Kadın",
        "scan_date":             "Çekim Tarihi",
        "clinician":             "İstekçi Klinisyen",
        "note":                  "Klinisyen Notu",
        "result_heading":        "Analiz Sonucu",
        "interpretation_heading":"Klinik Yorum",
        "mri_views":             "MRI Görüntüleri",
        "axial":                 "Aksiyel",
        "coronal":               "Koronal",
        "sagittal":              "Sagital",
        "history_heading":       "Önceki Analizler",
        "centiloid":             "Centiloid (CL)",
        "risk_category":         "Risk Kategorisi",
        "current":               "Mevcut",
        "model_version":         "Model Versiyonu",
        "processing_time":       "İşlem Süresi",
        "analysis_system":       "Analiz Sistemi",
        "disclaimer":            (
            "Bu tahmin yapay zeka tarafından üretilmiştir; klinik tanı yerine geçmez. "
            "Sonuçlar yetkili klinisyen tarafından değerlendirilmelidir."
        ),
        "demo_watermark":        "DEMO — KLİNİK KULLANIM İÇİN ONAYLI DEĞİLDİR",
        "risk_negative":         "Negatif",
        "risk_borderline":       "Sınırda",
        "risk_elevated":         "Yüksek",
        "risk_high":             "Çok Yüksek",
        "page":                  "Sayfa",
    },
    "en": {
        "report_id":             "Report ID",
        "date":                  "Date",
        "patient_info":          "Patient Information",
        "patient_name":          "Patient Name",
        "patient_id":            "Patient ID",
        "birth_year":            "Birth Year",
        "sex":                   "Sex",
        "male":                  "Male",
        "female":                "Female",
        "scan_date":             "Scan Date",
        "clinician":             "Requesting Clinician",
        "note":                  "Clinician Note",
        "result_heading":        "Analysis Result",
        "interpretation_heading":"Clinical Interpretation",
        "mri_views":             "MRI Views",
        "axial":                 "Axial",
        "coronal":               "Coronal",
        "sagittal":              "Sagittal",
        "history_heading":       "Previous Analyses",
        "centiloid":             "Centiloid (CL)",
        "risk_category":         "Risk Category",
        "current":               "Current",
        "model_version":         "Model Version",
        "processing_time":       "Processing Time",
        "analysis_system":       "Analysis System",
        "disclaimer":            (
            "This estimate is AI-generated and does not replace clinical diagnosis. "
            "Results must be evaluated by a qualified clinician."
        ),
        "demo_watermark":        "DEMO — NOT APPROVED FOR CLINICAL USE",
        "risk_negative":         "Negative",
        "risk_borderline":       "Borderline",
        "risk_elevated":         "Elevated",
        "risk_high":             "High",
        "page":                  "Page",
    },
}
