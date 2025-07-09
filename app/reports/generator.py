# app/reports/generator.py
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import matplotlib.pyplot as plt
from app.models import ChargingSession, User
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session
from svglib.svglib import svg2rlg

from ..config import APP_NAME, REPORT_TITLE

# --- Custom Doc Template with Section Awareness & Footer ---


class SectionAwareDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height, id="normal"
        )
        template = PageTemplate(
            id="SectionPage", frames=[frame], onPageEnd=self.draw_footer
        )
        self.addPageTemplates([template])
        self._current_section = None

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_section_label"):
            self._current_section = flowable._section_label

    def draw_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.drawString(
            15 * mm, 10 * mm, f"Generated on {datetime.now():%Y-%m-%d %H:%M}"
        )
        canvas.drawRightString(
            140 * mm, 10 * mm, f"{APP_NAME} - {self._current_section}"
        )
        canvas.drawRightString(200 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()


def format_duration(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    elif not parts:  # Only show seconds if nothing else was added
        parts.append(f"{secs}s")

    return " ".join(parts)


def build_logo_title_header(styles, doc_width):
    logo = SVGImage("app/static/images/flash_logo.svg", width=30, height=30)
    title = Paragraph(f"<b>{APP_NAME}</b> – {REPORT_TITLE}", styles["AppName"])

    inner_table = Table([[logo, title]], colWidths=[34, None])
    inner_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    wrapper = Table([[inner_table]], colWidths=[doc_width])
    wrapper.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9f9f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dcdcdc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    return wrapper


def build_report_header(styles, doc_width, username, start_date, end_date, section):
    elements = []

    # Add the logo header first (without the section label)
    logo_header = build_logo_title_header(styles, doc_width)
    elements.append(logo_header)
    elements.append(Spacer(1, 16))

    # Then add the section heading paragraph that carries the _section_label attribute
    heading = Paragraph(section, styles["Heading2"])
    heading._section_label = section
    elements.append(heading)
    elements.append(Spacer(1, 4))

    # Add user and date info below
    elements.append(Paragraph(f"User: <b>{username}</b>", styles["SubHeading"]))
    elements.append(
        Paragraph(
            f"Date Range: {start_date.date()} – {end_date.date()}", styles["Small"]
        )
    )
    elements.append(Spacer(1, 12))

    return elements


def fetch_user_sessions(
    db: Session,
    username: str,
    start_date: datetime,
    end_date: datetime,
    tag: Optional[str] = None,
    charger_ids: Optional[List[str]] = None,
    group_by_tag: bool = False,
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise ValueError(f"User '{username}' not found.")

    user_chargers = user.chargers
    if not user_chargers:
        return {}

    # Filter chargers by charger_id string (if provided)
    if charger_ids:
        user_chargers = [c for c in user_chargers if c.charger_id in charger_ids]

    if not user_chargers:
        return {}

    charger_db_ids = [c.id for c in user_chargers]

    query = db.query(ChargingSession).filter(
        ChargingSession.end_time >= start_date,
        ChargingSession.start_time <= end_date,
        ChargingSession.charger_id.in_(charger_db_ids),
    )

    if tag:
        query = query.filter(ChargingSession.tag.ilike(f"%{tag}%"))

    sessions = query.order_by(ChargingSession.start_time).all()

    # Grouping logic
    grouped = {}
    for session in sessions:
        key = (
            session.tag
            if group_by_tag
            else (session.charger.description or session.charger.charger_id)
        )
        grouped.setdefault(key, []).append(session)

    return grouped


def render_session_section(
    label,
    sessions,
    doc,
    styles,
    username,
    start_date,
    end_date,
    is_tag=False,
    is_first_section=False,
):
    elements = []

    # Only add a PageBreak if this is not the first section
    if not is_first_section:
        elements.append(PageBreak())

    # --- Monthly summary chart & table ---
    monthly_data = aggregate_monthly_data({label: sessions})
    if monthly_data:
        elements.extend(
            build_report_header(
                styles,
                doc.width,
                username,
                start_date,
                end_date,
                f"Monthly Summary for {label}",
            )
        )
        # --- Add Total Summary Before Chart ---

        total_kwh = sum(data["energy"] for _, data in monthly_data)
        total_sessions = sum(data["sessions"] for _, data in monthly_data)
        total_cost = sum(data["cost"] for _, data in monthly_data)
        total_duration = (
            sum(data["duration"] for _, data in monthly_data) / 3600
        )  # convert from seconds to hours

        # --- Create summary table data ---
        summary_data = [
            ["Total Energy (kWh)", "Total Sessions", "Total Cost (€)", "Total Duration (hrs)"],
            [f"{total_kwh:.2f}", str(total_sessions), f"{total_cost:.1f}", f"{total_duration:.1f}"],
        ]

        summary_table = Table(summary_data, colWidths=[doc.width / 5] * 4)
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("TOPPADDING", (1, 0), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )

        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # --- Chart ---
        chart = generate_monthly_summary_chart(monthly_data, doc.width)
        if chart:
            elements.append(chart)
            elements.append(Spacer(1, 8))

        summary_table = create_monthly_summary_table(monthly_data, doc.width)
        elements.append(summary_table)
        # **Add page break here**
        elements.append(PageBreak())

    # --- Energy chart for the current group ---
    elements.extend(
        build_report_header(
            styles,
            doc.width,
            username,
            start_date,
            end_date,
            f"Session details for {label}",
        )
    )
    chart = generate_energy_chart(sessions, label)
    if chart:
        elements.append(chart)
        elements.append(Spacer(1, 8))

    # --- Session table ---
    if is_tag:
        data = [
            ["Start Time", "End Time", "Duration", "Energy kWh", "Avg kW", "Cost", "Charger"]
        ]
    else:
        data = [["Start Time", "End Time", "Duration", "Energy kWh", "Avg kW", "Cost", "Tag"]]

    for s in sessions:
        duration = format_duration(s.duration_seconds)
        avg_power = (
            s.energy_charged_kwh / (s.duration_seconds / 3600)
            if s.duration_seconds > 0
            else 0
        )
        final_col = s.charger.charger_id if is_tag else (s.tag or "")
        data.append(
            [
                s.start_time.strftime("%Y-%m-%d %H:%M"),
                s.end_time.strftime("%Y-%m-%d %H:%M"),
                duration,
                f"{s.energy_charged_kwh:.2f}",
                f"{avg_power:.2f}",
                f"{s.cost:.2f}",
                final_col,
            ]
        )

    col_widths = [
        doc.width * 0.18,
        doc.width * 0.18,
        doc.width * 0.12,
        doc.width * 0.14,
        doc.width * 0.10,
        doc.width * 0.08,
        doc.width * 0.20,
    ]

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                # --- Header alignment ---
                ("ALIGN", (0, 0), (1, 0), "LEFT"),  # Start & End Time
                ("ALIGN", (2, 0), (5, 0), "RIGHT"),  # Duration, Energy, Avg kW
                ("ALIGN", (6, 0), (6, 0), "LEFT"),  # Tag
                # --- Body alignment ---
                ("ALIGN", (0, 1), (1, -1), "LEFT"),
                ("ALIGN", (2, 1), (5, -1), "RIGHT"),
                ("ALIGN", (6, 1), (6, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Courier"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.whitesmoke, colors.HexColor("#ecf0f1")],
                ),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 24))

    return elements


def generate_energy_chart(sessions, charger_label):
    daily_energy = defaultdict(float)
    for s in sessions:
        date_key = s.start_time.date()
        daily_energy[date_key] += s.energy_charged_kwh

    if not daily_energy:
        return None

    dates = sorted(daily_energy)
    energy_values = [daily_energy[date] for date in dates]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(dates, energy_values, color="skyblue")
    ax.set_title(f"Energy per Day – {charger_label}")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=400, height=180)
    img.hAlign = "LEFT"
    return img


class SVGImage(Flowable):
    def __init__(self, svg_path, width=40, height=40):
        import os

        full_path = os.path.abspath(svg_path)
        self.drawing = svg2rlg(full_path)
        if self.drawing is None:
            raise ValueError(f"Could not load SVG from path: {full_path}")

        scale_x = width / self.drawing.width
        scale_y = height / self.drawing.height
        self.scale = min(scale_x, scale_y)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        self.canv.saveState()
        self.canv.scale(self.scale, self.scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


def aggregate_monthly_data(grouped_sessions):
    monthly_data = defaultdict(lambda: {"energy": 0, "duration": 0, "cost": 0, "sessions": 0})

    for sessions in grouped_sessions.values():
        for s in sessions:
            month_key = s.start_time.strftime("%Y-%m")
            monthly_data[month_key]["energy"] += s.energy_charged_kwh
            monthly_data[month_key]["duration"] += s.duration_seconds
            monthly_data[month_key]["cost"] += s.cost
            monthly_data[month_key]["sessions"] += 1

    sorted_months = sorted(monthly_data.keys())
    return [(month, monthly_data[month]) for month in sorted_months]


def generate_monthly_summary_chart(monthly_data, doc_width):
    from io import BytesIO

    import matplotlib.pyplot as plt
    import numpy as np
    from reportlab.platypus import Image

    months = [m for m, _ in monthly_data]
    energies = [data["energy"] for _, data in monthly_data]
    durations = [
        data["duration"] / 3600 for _, data in monthly_data
    ]  # seconds to hours
    sessions = [data["sessions"] for _, data in monthly_data]

    x = np.arange(len(months))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.bar(x - width, energies, width, label="Energy (kWh)", color="skyblue")
    ax.bar(x, durations, width, label="Duration (hrs)", color="lightgreen")
    ax.bar(x + width, sessions, width, label="Sessions", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_title("Monthly Summary")
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=doc_width, height=150)
    img.hAlign = "CENTER"
    return img


def create_monthly_summary_table(monthly_data, doc_width):
    data = [["Month", "Total Energy (kWh)", "Total Duration", "Cost (€)", "Sessions"]]
    for month, vals in monthly_data:
        data.append(
            [
                month,
                f"{vals['energy']:.2f}",
                format_duration(vals["duration"]),
                f"{vals['cost']:.2f}",
                str(vals["sessions"]),
            ]
        )

    col_widths = [
        doc_width * 0.15,
        doc_width * 0.25,
        doc_width * 0.20,
        doc_width * 0.15,
        doc_width * 0.15,
    ]

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                # --- Header alignment ---
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (4, 0), "RIGHT"),
                # --- Body alignment ---
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("ALIGN", (1, 1), (4, -1), "RIGHT"),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Courier"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.whitesmoke, colors.HexColor("#ecf0f1")],
                ),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def generate_pdf_report(
    username, grouped_sessions, start_date, end_date, output_path, group_by_tag=False
):
    doc = SectionAwareDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="AppName", fontSize=18, alignment=0, spaceAfter=12, leading=24
        )
    )
    styles.add(
        ParagraphStyle(
            name="CenteredHeading",
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12,
            leading=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="LeftHeading", fontSize=16, alignment=0, spaceAfter=12, leading=20
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading", fontSize=13, spaceAfter=8, textColor=colors.darkblue
        )
    )
    styles.add(
        ParagraphStyle(name="Small", fontSize=10, spaceAfter=4, textColor=colors.grey)
    )

    elements = []
    first_section = True  # Track whether it's the first section

    if not group_by_tag:
        # Render sessions grouped by charger
        for charger, sessions in grouped_sessions.items():
            elements.extend(
                render_session_section(
                    f"charger '{charger}'",
                    sessions,
                    doc,
                    styles,
                    username,
                    start_date,
                    end_date,
                    is_tag=False,
                    is_first_section=first_section,
                )
            )
            first_section = False
    else:
        # Flatten all sessions and regroup by tag
        all_sessions = [s for sessions in grouped_sessions.values() for s in sessions]
        tag_groups = defaultdict(list)
        for session in all_sessions:
            tag = session.tag or "No Tag"
            tag_groups[tag].append(session)

        # Render each tag group on a new page (except the first_section)
        for tag, sessions in tag_groups.items():
            elements.extend(
                render_session_section(
                    f"tag '{tag}'",
                    sessions,
                    doc,
                    styles,
                    username,
                    start_date,
                    end_date,
                    is_tag=True,
                    is_first_section=first_section,
                )
            )
            first_section = False

    doc.build(elements)
