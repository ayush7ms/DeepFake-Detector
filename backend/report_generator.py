"""Generate downloadable PDF investigation reports from actual API results.

Reports only restate what the analysis service returned. Where a step produced no
result (model unavailable, detector not configured, visuals not retained) the
report says so instead of filling the gap.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PURPLE = colors.HexColor("#43347b")
NAVY = colors.HexColor("#0d1020")
LAVENDER = colors.HexColor("#b2a3ff")
INK = colors.HexColor("#202334")
MUTED = colors.HexColor("#5d6377")
LINE = colors.HexColor("#d5d7e0")
SOFT = colors.HexColor("#eeebf8")
BAND_COLORS = {
    # Deliberately not green/red: a low score is not "safe" and an elevated one is not "fake".
    "low": colors.HexColor("#2b8aa3"),
    "inconclusive": colors.HexColor("#6b5bb5"),
    "review": colors.HexColor("#b8842f"),
    "none": colors.HexColor("#6b7186"),
}
CONTENT_WIDTH = 7.2 * inch
MEDIA_NAMES = {"image": "Image", "video": "Video", "audio": "Audio"}

ETHICAL_NOTICE = (
    "Ethical use. DeepTruth is a decision-support tool for responsible review. Do not use this report as the sole "
    "basis for accusing, sanctioning, or publicly identifying anyone. Automated detectors make errors, including "
    "false positives on authentic media and false negatives on manipulated media. Respect the privacy and consent "
    "of the people shown in any media you analyse, and corroborate findings with source verification and expert review."
)


def _latin(text: str) -> str:
    """The built-in PDF fonts cover Latin-1 only; replace anything else rather than print blank boxes."""
    return text.encode("cp1252", "replace").decode("cp1252")


def _t(value: Any) -> str:
    """Escape for reportlab Paragraph markup and keep the text printable."""
    if value is None or value == "":
        return "Not available"
    return escape(_latin(str(value)))


def _when(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return str(value) if value else "Not available"


def _percent(score: Any) -> str:
    try:
        return f"{float(score) * 100:.1f}%"
    except (TypeError, ValueError):
        return "Not available"


def _size(num_bytes: Any) -> str:
    try:
        value = float(num_bytes)
    except (TypeError, ValueError):
        return "Not available"
    return f"{value / 1024:.0f} KB" if value < 1024 * 1024 else f"{value / (1024 * 1024):.1f} MB"


def _decode_image(data_url: str, max_width: float, max_height: float) -> Image | None:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
        with PILImage.open(BytesIO(raw)) as picture:
            width, height = picture.size
        scale = min(max_width / width, max_height / height)
        return Image(BytesIO(raw), width=width * scale, height=height * scale)
    except (IndexError, ValueError, binascii.Error, OSError):
        return None


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("DTTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=PURPLE, alignment=TA_LEFT, spaceAfter=2),
        "subtitle": ParagraphStyle("DTSub", parent=base["BodyText"], fontSize=9, leading=12, textColor=MUTED, spaceAfter=8),
        "h2": ParagraphStyle("DTH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=PURPLE, spaceBefore=14, spaceAfter=5, keepWithNext=1),
        "body": ParagraphStyle("DTBody", parent=base["BodyText"], fontSize=9, leading=13, textColor=INK, spaceAfter=4, wordWrap="CJK"),
        "small": ParagraphStyle("DTSmall", parent=base["BodyText"], fontSize=7.5, leading=10.5, textColor=MUTED, wordWrap="CJK"),
        "cell": ParagraphStyle("DTCell", parent=base["BodyText"], fontSize=8.5, leading=11.5, textColor=INK, wordWrap="CJK"),
        "cell_bold": ParagraphStyle("DTCellB", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11.5, textColor=INK),
        "mono": ParagraphStyle("DTMono", parent=base["BodyText"], fontName="Courier", fontSize=7.5, leading=10, textColor=INK, wordWrap="CJK"),
        "callout_label": ParagraphStyle("DTCallL", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK),
        "bullet": ParagraphStyle("DTBullet", parent=base["BodyText"], fontSize=8.8, leading=12.5, textColor=INK, leftIndent=11, bulletIndent=0, spaceAfter=3, wordWrap="CJK"),
    }


def _kv_table(rows: list[tuple[str, Any]], styles: dict[str, ParagraphStyle], mono_keys: set[str] | None = None) -> Table:
    mono_keys = mono_keys or set()
    data = [
        [
            Paragraph(_t(label), styles["cell_bold"]),
            Paragraph(_t(value), styles["mono"] if label in mono_keys else styles["cell"]),
        ]
        for label, value in rows
    ]
    table = Table(data, colWidths=[1.65 * inch, CONTENT_WIDTH - 1.65 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), SOFT),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _gauge(score: float, interpretation: dict[str, Any]) -> Drawing:
    """Horizontal bar showing where the score sits relative to the descriptive bands."""
    width, height = CONTENT_WIDTH - 26, 0.62 * inch
    bar_y, bar_h = 0.26 * inch, 0.17 * inch
    thresholds = interpretation.get("thresholds") or {"low": 0.25, "review": 0.75}
    low, review = float(thresholds["low"]), float(thresholds["review"])
    drawing = Drawing(width, height)
    zones = [(0.0, low, "#cde6ee"), (low, review, "#e0dcf3"), (review, 1.0, "#f1e1c3")]
    for start, end, fill in zones:
        drawing.add(Rect(start * width, bar_y, (end - start) * width, bar_h, fillColor=colors.HexColor(fill), strokeColor=None))
    drawing.add(Rect(0, bar_y, width, bar_h, fillColor=None, strokeColor=LINE, strokeWidth=0.6))
    marker_x = max(0.0, min(1.0, score)) * width
    drawing.add(Line(marker_x, bar_y - 3, marker_x, bar_y + bar_h + 3, strokeColor=NAVY, strokeWidth=1.6))
    drawing.add(Polygon([marker_x - 4, bar_y + bar_h + 9, marker_x + 4, bar_y + bar_h + 9, marker_x, bar_y + bar_h + 3], fillColor=NAVY, strokeColor=None))
    for position, caption in ((0.0, "0"), (low, f"{low:.2f}"), (review, f"{review:.2f}"), (1.0, "1")):
        anchor = "start" if position == 0 else "end" if position == 1 else "middle"
        drawing.add(String(position * width, 0.08 * inch, caption, fontName="Helvetica", fontSize=6.5, fillColor=MUTED, textAnchor=anchor))
    return drawing


def _frame_chart(frames: list[dict[str, Any]], review_threshold: float) -> Drawing:
    width, height = CONTENT_WIDTH, 1.75 * inch
    left, bottom, top = 0.35 * inch, 0.38 * inch, 0.12 * inch
    plot_w, plot_h = width - left - 0.1 * inch, height - bottom - top
    drawing = Drawing(width, height)
    drawing.add(Rect(left, bottom, plot_w, plot_h, fillColor=colors.HexColor("#fafafd"), strokeColor=LINE, strokeWidth=0.5))
    for tick in (0.0, 0.5, 1.0):
        y = bottom + tick * plot_h
        drawing.add(String(left - 4, y - 2, f"{tick:.1f}", fontName="Helvetica", fontSize=6.5, fillColor=MUTED, textAnchor="end"))
    slot = plot_w / max(1, len(frames))
    for index, frame in enumerate(frames):
        score = max(0.0, min(1.0, float(frame.get("model_score") or 0)))
        flagged = score >= review_threshold
        bar_w = min(slot * 0.62, 0.45 * inch)
        x = left + index * slot + (slot - bar_w) / 2
        fill = BAND_COLORS["review"] if flagged else PURPLE
        drawing.add(Rect(x, bottom, bar_w, max(0.6, score * plot_h), fillColor=fill, strokeColor=None))
        drawing.add(String(x + bar_w / 2, bottom - 9, f"{float(frame.get('timestamp_seconds') or 0):.1f}s", fontName="Helvetica", fontSize=6.5, fillColor=MUTED, textAnchor="middle"))
    y = bottom + review_threshold * plot_h
    drawing.add(Line(left, y, left + plot_w, y, strokeColor=BAND_COLORS["review"], strokeWidth=0.7, strokeDashArray=[3, 2]))
    drawing.add(String(left + plot_w - 2, y + 2, f"review band {review_threshold:.2f}", fontName="Helvetica", fontSize=6.5, fillColor=BAND_COLORS["review"], textAnchor="end"))
    drawing.add(String(left, 3, "Fake-class model score of each sampled frame (uncalibrated)", fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    return drawing


def _bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [Paragraph(_t(item), styles["bullet"], bulletText="\u2022") for item in items]


def _is_demo(report: dict[str, Any]) -> bool:
    return any(item.get("type") == "demo_rule" for item in (report.get("evidence") or []))


def _final_interpretation(report: dict[str, Any]) -> str:
    status = report.get("analysis_status")
    if status != "completed":
        return (
            f"No analysis result was produced for this file (status: {status or 'unknown'}). This is a service or configuration "
            "outcome and says nothing about whether the media is authentic or manipulated."
        )
    if _is_demo(report):
        verdict = "FAKE" if (report.get("interpretation") or {}).get("band") == "review" else "REAL"
        return (
            f"SIMULATED DEMO VERDICT: {verdict}. This verdict was assigned by a file-name rule (WhatsApp file name = fake, "
            "any other name = real). The image was not judged by the model, and this is not a forensic finding."
        )
    media = report.get("media_type")
    subject = {"image": "image", "video": "sampled video frames", "audio": "audio"}.get(media, "media")
    band = (report.get("interpretation") or {}).get("band")
    if band == "review":
        text = (
            f"The model assigned an elevated fake-class score to this {subject}. That justifies closer human review "
            "(verifying the source, comparing other copies, checking metadata) but does not establish that the media is synthetic or manipulated."
        )
    elif band == "low":
        text = (
            f"The model assigned a low fake-class score to this {subject}. This does not establish authenticity: the model only recognises "
            "the manipulation styles represented in its training data, and many edits would not be detected."
        )
    else:
        text = (
            f"The model did not separate this {subject} clearly into either direction. The result is inconclusive and should not "
            "be used to support a conclusion either way."
        )
    if report.get("warnings"):
        text += " The scope warnings listed in this report reduce how much weight the score deserves."
    return text


def generate_pdf_report(report_data: dict[str, Any]) -> bytes:
    """Build the PDF. Values are escaped so uploaded text (such as file names) stays inert."""
    styles = _styles()
    investigation_id = str(report_data.get("investigation_id") or "unassigned")
    media = report_data.get("media_type") or "unknown"
    upload = report_data.get("upload_information") or {}
    interpretation = report_data.get("interpretation") or {}
    visuals = report_data.get("visuals") or {}
    status = report_data.get("analysis_status")
    score = report_data.get("model_score")
    band = interpretation.get("band") or "none"

    buffer = BytesIO()

    def decorate(canvas, doc) -> None:
        width, height = letter
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 0.52 * inch, width, 0.52 * inch, stroke=0, fill=1)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.setFillColor(colors.white)
        canvas.drawString(0.65 * inch, height - 0.33 * inch, "deep")
        canvas.setFillColor(LAVENDER)
        canvas.drawString(0.65 * inch + canvas.stringWidth("deep", "Helvetica-Bold", 13), height - 0.33 * inch, "truth")
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(width - 0.65 * inch, height - 0.32 * inch, "DIGITAL MEDIA FORENSICS REPORT")
        canvas.setStrokeColor(LINE)
        canvas.line(0.65 * inch, 0.55 * inch, width - 0.65 * inch, 0.55 * inch)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(0.65 * inch, 0.38 * inch, f"Investigation {investigation_id}")
        canvas.drawCentredString(width / 2, 0.38 * inch, "Model-assisted review. A score is not proof.")
        canvas.drawRightString(width - 0.65 * inch, 0.38 * inch, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.8 * inch,
        title="DeepTruth Investigation Report",
        author="DeepTruth",
        subject=f"{MEDIA_NAMES.get(media, 'Media')} investigation {investigation_id}",
    )

    story: list[Any] = [
        Paragraph("Investigation report", styles["title"]),
        Paragraph("Evidence, uncertainty and limitations for one analysed file.", styles["subtitle"]),
        _kv_table(
            [
                ("Investigation ID", investigation_id),
                ("Generated", _when(report_data.get("generated_at"))),
                ("Media type", MEDIA_NAMES.get(media, media)),
                ("File name", report_data.get("filename")),
                ("File size", _size(upload.get("size_bytes"))),
                ("SHA-256 of upload", upload.get("sha256")),
                ("Analysis status", str(status or "unknown").replace("_", " ")),
            ],
            styles,
            mono_keys={"Investigation ID", "SHA-256 of upload"},
        ),
        Paragraph("Result", styles["h2"]),
    ]

    # --- result callout ---
    accent = BAND_COLORS.get(band, BAND_COLORS["none"])
    if status == "completed" and (score is not None or _is_demo(report_data)):
        demo = _is_demo(report_data)
        headline = (interpretation.get("label") or "Demo verdict") if demo else f"{interpretation.get('label') or 'Model score'}: {_percent(score)}"
        score_block: list[Any] = [
            Paragraph(_t(headline), styles["callout_label"]),
            Paragraph(
                (f"Underlying model score (not the verdict): {_percent(score) if score is not None else 'not available (model not loaded)'}. The verdict above comes from a file-name rule, not from a model score."
                 if demo else
                 "Fake-class model score. This is the checkpoint's own softmax output on a 0 to 100% scale. It is not a calibrated probability, "
                 "and it is not the chance that the media is fake."),
                styles["small"],
            ),
            Spacer(1, 4),
            *([_gauge(float(score), interpretation)] if score is not None else []),
            Paragraph(_t(interpretation.get("summary")), styles["body"]),
            Paragraph(_t(interpretation.get("note")), styles["small"]),
        ]
    else:
        score_block = [
            Paragraph("No model score was produced", styles["callout_label"]),
            Paragraph(_t(report_data.get("error") or "; ".join(str(x) for x in (report_data.get("limitations") or [])[:1])), styles["body"]),
            Paragraph("A missing result is a service or configuration outcome. It is not evidence that the media is fake or genuine.", styles["small"]),
        ]
    callout = Table([[score_block]], colWidths=[CONTENT_WIDTH])
    callout.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, accent),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfbfe")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(callout)

    warnings = report_data.get("warnings") or []
    if warnings:
        story.append(Paragraph("Scope warnings", styles["h2"]))
        story.extend(_bullets([str(item) for item in warnings], styles))

    # --- evidence ---
    story.append(Paragraph("Evidence and indicators", styles["h2"]))
    evidence = report_data.get("evidence") or []
    if not evidence:
        story.append(Paragraph("The analysis returned no evidence items.", styles["body"]))
    for item in evidence:
        kind = item.get("type")
        if kind == "image_classification":
            story.append(
                _kv_table(
                    [
                        ("Classifier output", item.get("model_output")),
                        ("Fake-class score", _percent(item.get("fake_class_score"))),
                        ("Real-class score", _percent(item.get("real_class_score"))),
                        ("Note", item.get("interpretation")),
                    ],
                    styles,
                )
            )
            story.append(Spacer(1, 5))
        elif kind in ("demo_rule", "provenance_rule"):
            story.append(KeepTogether([_kv_table([
                ("Rule", item.get("rule")), ("Matched file name", item.get("matched_filename")), ("Note", item.get("interpretation"))], styles)]))
            story.append(Spacer(1, 5))
        elif kind == "face_detection":
            found = item.get("faces_detected")
            story.append(
                KeepTogether([_kv_table(
                    [
                        ("Face check", f"{found} frontal face(s) found" if found is not None else "Unavailable"),
                        ("Method", item.get("method")),
                        ("Purpose", item.get("interpretation")),
                    ],
                    styles,
                )])
            )
            story.append(Spacer(1, 5))
        elif kind == "explainability":
            rows = [
                ("Explainability", f"{item.get('method')} - {item.get('status')}"),
                ("Target class", item.get("target_class")),
                ("Patch grid", item.get("patch_grid")),
            ]
            if item.get("reason"):
                rows.append(("Why unavailable", item.get("reason")))
            rows.append(("How to read it", item.get("interpretation")))
            story.append(KeepTogether([_kv_table(rows, styles)]))
            story.append(Spacer(1, 6))
        elif kind == "sampled_video_frames":
            frames = item.get("sampled_frames") or []
            review_threshold = float(item.get("review_threshold") or 0.75)
            story.append(
                _kv_table(
                    [
                        ("Frames analysed", f"{item.get('frames_analyzed')} of {item.get('frames_requested')} requested"),
                        ("Frames with a face", item.get("frames_with_detected_face")),
                        ("Aggregate", "Median fake-class score across " + ("only the sampled frames where a face was found" if item.get("aggregate_basis") == "frames_with_detected_face" else "all sampled frames")),
                        ("Lowest / highest frame", f"{_percent(item.get('lowest_frame_score'))} / {_percent(item.get('highest_frame_score'))}"),
                    ],
                    styles,
                )
            )
            story.append(Spacer(1, 6))
            if frames:
                story.append(_frame_chart(frames, review_threshold))
                story.append(Spacer(1, 6))
                thumbs = visuals.get("frame_thumbnails") or {}
                header = ["Frame", "Time", "Model score", "Faces", "Preview"]
                rows: list[list[Any]] = [[Paragraph(_t(h), styles["cell_bold"]) for h in header]]
                for frame in frames:
                    preview = _decode_image(thumbs.get(str(frame.get("frame_index")), ""), 0.95 * inch, 0.55 * inch) if thumbs else None
                    rows.append(
                        [
                            Paragraph(_t(frame.get("frame_index")), styles["cell"]),
                            Paragraph(f"{float(frame.get('timestamp_seconds') or 0):.2f}s", styles["cell"]),
                            Paragraph(_percent(frame.get("model_score")), styles["cell"]),
                            Paragraph(_t(frame.get("faces_detected")), styles["cell"]),
                            preview or Paragraph("not retained" if not thumbs else "n/a", styles["small"]),
                        ]
                    )
                table = Table(rows, colWidths=[0.8 * inch, 0.8 * inch, 1.1 * inch, 0.7 * inch, 1.4 * inch], repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(table)
        else:
            story.append(Paragraph(_t(str(item)), styles["mono"]))

    timestamps = report_data.get("suspicious_timestamps") or []
    if media == "video":
        story.append(Paragraph("Timestamps flagged for human review", styles["h2"]))
        if timestamps:
            story.extend(
                _bullets(
                    [f"{float(t.get('timestamp_seconds') or 0):.2f}s (frame {t.get('frame_index')}): model score {_percent(t.get('model_score'))}. {t.get('reason')}" for t in timestamps],
                    styles,
                )
            )
        else:
            story.append(Paragraph("No sampled frame crossed the review band.", styles["body"]))

    # --- visual evidence ---
    has_frame_previews = media == "video" and bool(visuals.get("frame_thumbnails"))
    if media == "image" and (visuals.get("model_input") or visuals.get("heatmap")):
        story.append(Paragraph("Visual evidence", styles["h2"]))
        cells: list[Any] = []
        for key, title in (("model_input", "What the model received"), ("heatmap", "Grad-CAM overlay")):
            visual = visuals.get(key)
            picture = _decode_image(visual["image"], 3.3 * inch, 3.3 * inch) if visual else None
            cells.append([Paragraph(_t(title), styles["cell_bold"]), picture or Paragraph("Not generated", styles["small"]), Paragraph(_t((visual or {}).get("caption")), styles["small"])])
        pair = Table([cells], colWidths=[CONTENT_WIDTH / 2, CONTENT_WIDTH / 2], hAlign="LEFT")
        pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE)]))
        story.append(KeepTogether([pair]))
    elif media in {"image", "video"} and status == "completed" and not has_frame_previews:
        story.append(Paragraph("Visual evidence", styles["h2"]))
        story.append(
            Paragraph(
                "No visual evidence is attached to this report. For privacy, DeepTruth does not keep image content after analysis: "
                "overlays and frame previews exist only in memory for a short time and are not stored in history. "
                "Re-run the analysis to regenerate them.",
                styles["body"],
            )
        )

    # --- limitations, interpretation, model info ---
    story.append(Paragraph("Limitations and uncertainty", styles["h2"]))
    story.extend(_bullets([str(x) for x in (report_data.get("limitations") or ["No limitations were returned by the analysis service."])], styles))

    story.append(Paragraph("Final interpretation", styles["h2"]))
    story.append(Paragraph(_t(_final_interpretation(report_data)), styles["body"]))

    model_info = report_data.get("model_information") or {}
    story.append(Paragraph("Model information", styles["h2"]))
    info_rows: list[tuple[str, Any]] = [
        ("Model", model_info.get("name")),
        ("Repository", model_info.get("repository")),
        ("Model card", model_info.get("model_card")),
        ("Architecture", model_info.get("architecture")),
        ("Task", model_info.get("task")),
        ("License", model_info.get("license")),
        ("Input handling", model_info.get("input_requirements") or model_info.get("usage")),
        ("Compute device", model_info.get("device")),
        ("Status", model_info.get("status")),
    ]
    story.append(_kv_table([(label, value) for label, value in info_rows if value not in (None, "")] or [("Model", "No model information was returned.")], styles))

    story.append(Spacer(1, 12))
    story.append(
        KeepTogether(
            [
                Table(
                    [[Paragraph(_t(ETHICAL_NOTICE), styles["small"])]],
                    colWidths=[CONTENT_WIDTH],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                            ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    ),
                )
            ]
        )
    )

    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()