"""Student Productivity Dashboard.

A multi-page Streamlit dashboard exploring how digital habits, academic effort
and wellness behaviours relate to final grades across 20K student records.
Data pipeline mirrors the cleaning steps performed in analysis.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).with_name("student_productivity.csv")


PAGES = [
    ("overview", "Overview"),
    ("digital", "Digital Habits"),
    ("effort", "Academic Effort"),
    ("wellness", "Wellness and Cognition"),
    ("about", "About"),
]

PAGE_ACCENTS = {
    "overview": "#111111",
    "digital": "#2563EB",
    "effort": "#10B981",
    "wellness": "#8B5CF6",
    "about": "#71717A",
}

SEGMENT_ORDER = ["Critical", "At Risk", "Stable", "Strong", "Elite"]
SEGMENT_COLORS = {
    "Critical": "#EF4444",
    "At Risk": "#F59E0B",
    "Stable": "#A1A1AA",
    "Strong": "#2563EB",
    "Elite": "#10B981",
}

DIGITAL_ORDER = ["Low", "Balanced", "Heavy", "Overloaded"]
DIGITAL_COLORS = {
    "Low": "#10B981",
    "Balanced": "#2563EB",
    "Heavy": "#F59E0B",
    "Overloaded": "#EF4444",
}

SLEEP_ORDER = ["Short", "Balanced", "Long"]
STUDY_ORDER = ["Light", "Core", "Focused", "Intensive"]
STRESS_ORDER = ["Calm", "Managed", "High", "Burnout"]
STRESS_COLORS = {
    "Calm": "#10B981",
    "Managed": "#2563EB",
    "High": "#F59E0B",
    "Burnout": "#EF4444",
}
EXERCISE_ORDER = ["Low", "Med", "High"]
EXERCISE_COLORS = {"Low": "#EF4444", "Med": "#F59E0B", "High": "#10B981"}

LABELS = {
    "age": "Age",
    "study_hours_per_day": "Daily study hours",
    "sleep_hours": "Sleep hours",
    "phone_usage_hours": "Phone usage (h/day)",
    "social_media_hours": "Social media (h/day)",
    "youtube_hours": "YouTube (h/day)",
    "gaming_hours": "Gaming (h/day)",
    "screen_time_hours": "Total entertainment screen time",
    "breaks_per_day": "Breaks per day",
    "coffee_intake_mg": "Coffee intake (mg/day)",
    "exercise_minutes": "Exercise (min/day)",
    "assignments_completed": "Assignments completed",
    "attendance_percentage": "Attendance (%)",
    "stress_level": "Stress level (1-10)",
    "focus_score": "Focus score",
    "productivity_score": "Productivity score",
    "final_grade": "Final grade",
    "distraction_ratio": "Distraction ratio",
    "performance_segment": "Performance segment",
    "digital_load": "Digital load",
    "sleep_quality": "Sleep quality",
    "study_band": "Study band",
    "stress_zone": "Stress zone",
    "exercise_group": "Exercise group",
    "gender": "Gender",
}

# (column, bar colour) — overview interactive distribution; order = selectbox order.
OVERVIEW_DIST_VARS: list[tuple[str, str]] = [
    ("final_grade", "#2563EB"),
    ("focus_score", "#8B5CF6"),
    ("study_hours_per_day", "#10B981"),
    ("sleep_hours", "#F59E0B"),
    ("productivity_score", "#6366F1"),
    ("stress_level", "#EF4444"),
    ("screen_time_hours", "#EC4899"),
    ("phone_usage_hours", "#64748B"),
    ("social_media_hours", "#F97316"),
    ("youtube_hours", "#FB7185"),
    ("gaming_hours", "#A855F7"),
    ("coffee_intake_mg", "#78716C"),
    ("exercise_minutes", "#22C55E"),
    ("assignments_completed", "#0D9488"),
    ("attendance_percentage", "#0284C7"),
    ("breaks_per_day", "#CA8A04"),
    ("distraction_ratio", "#7C3AED"),
    ("age", "#4B5563"),
]

INK = "#0A0A0A"
MUTED = "#71717A"
GRID = "#ECECEC"
CARD = "#FFFFFF"


st.set_page_config(
    page_title="Student Productivity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined&display=swap');

            :root {
                --ink: #0A0A0A;
                --muted: #71717A;
                --soft: #A1A1AA;
                --grid: #ECECEC;
                --paper: #FAFAFA;
                --card: #FFFFFF;
                --accent: #111111;
            }

            html, body, [class*="st-"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            html { scroll-behavior: smooth; }
            .stApp { background: var(--paper); color: var(--ink); }

            [data-testid="stHeader"] { background: transparent; display: none !important; }
            footer, #MainMenu { visibility: hidden; }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 4rem;
                max-width: 1640px;
            }

            /* Streamlit selectbox: no text I-beam on hover (pick-only control). */
            [data-testid="stSelectbox"],
            [data-testid="stSelectbox"] * {
                cursor: pointer !important;
            }
            [data-testid="stSelectbox"] input {
                caret-color: transparent !important;
            }

            /* ───── Sidebar (real st.sidebar) ───── */
            section[data-testid="stSidebar"] {
                background: #FFFFFF;
                border-right: 1px solid var(--grid);
                width: 260px !important;
                min-width: 260px !important;
            }
            section[data-testid="stSidebar"] > div {
                padding-top: 0 !important;
            }
            section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                padding: 0 !important;
                overflow-x: visible !important;
            }
            section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
                padding: 0 0.8rem 1.4rem 0.8rem !important;
            }
            /* Hide the native sidebar collapse button (icon font fallback shows as raw text) */
            [data-testid="stSidebarCollapseButton"],
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stBaseButton-headerNoPadding"],
            section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
                display: none !important;
            }

            /* Brand block */
            .sb-brand {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                padding: 1.3rem 0.4rem 1.1rem;
                border-bottom: 1px solid var(--grid);
                margin: 0 -0.4rem 0.8rem -0.4rem;
            }
            .sb-logo {
                width: 36px; height: 36px;
                flex-shrink: 0;
                border-radius: 10px;
                background: #18181B;
                display: flex; align-items: center; justify-content: center;
            }
            .sb-brand-text {
                display: flex; flex-direction: column;
                line-height: 1.18;
            }
            .sb-brand-name {
                color: var(--ink);
                font-size: 0.92rem;
                font-weight: 700;
                letter-spacing: -0.018em;
            }
            .sb-brand-sub {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 500;
                margin-top: 1px;
            }

            /* Cohort — compact typographic hierarchy (not a single long sentence) */
            .sb-cohort {
                padding: 0.65rem 0.75rem 0.7rem;
                border: 1px solid var(--grid);
                border-radius: 10px;
                background: linear-gradient(180deg, #FAFAFA 0%, #FFFFFF 100%);
                margin: 0.2rem 0 0.85rem 0;
            }
            .sb-cohort-kicker {
                font-size: 0.62rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--soft);
                margin-bottom: 0.35rem;
            }
            .sb-cohort-row {
                display: flex;
                align-items: baseline;
                flex-wrap: wrap;
                gap: 0.15rem 0.35rem;
                line-height: 1.15;
            }
            .sb-cohort-main {
                font-size: 1.15rem;
                font-weight: 750;
                letter-spacing: -0.03em;
                color: var(--ink);
                font-variant-numeric: tabular-nums;
            }
            .sb-cohort-den {
                font-size: 0.72rem;
                font-weight: 500;
                color: var(--muted);
                font-variant-numeric: tabular-nums;
            }
            .sb-cohort-pct {
                margin-top: 0.35rem;
                font-size: 0.68rem;
                color: var(--muted);
                font-variant-numeric: tabular-nums;
            }

            /* Meta labels: read as UI chrome, not page section titles */
            .sb-label {
                margin: 0.55rem 0 0.38rem 0;
                padding: 0 0.1rem;
            }
            .sb-label-inner {
                display: inline-block;
                padding: 0.22rem 0.5rem;
                border-radius: 6px;
                background: #F4F4F5;
                border: 1px solid #E4E4E7;
                color: #52525B;
                font-size: 0.58rem;
                font-weight: 750;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }

            /* Section nav: no outer frame — only the active row gets a rectangle (scroll-spy). */
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(a[href^="#section-"]) {
                margin-bottom: 0.35rem;
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a[href^="#section-"] {
                display: flex !important;
                align-items: center !important;
                box-sizing: border-box !important;
                padding: 10px 12px !important;
                border-radius: 10px !important;
                border: 1px solid transparent !important;
                outline: none !important;
                color: #71717A !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                text-decoration: none !important;
                transition: background 0.22s ease, color 0.22s ease, border-color 0.22s ease,
                    box-shadow 0.22s ease;
                line-height: 1.25 !important;
                margin: 2px 0 !important;
            }
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a[href^="#section-"]:not(.active) {
                box-shadow: none !important;
            }
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a[href^="#section-"]:hover:not(.active) {
                background: rgba(24, 24, 27, 0.04) !important;
                color: #3F3F46 !important;
            }
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a[href^="#section-"]:focus-visible:not(.active) {
                box-shadow: 0 0 0 2px rgba(24, 24, 27, 0.14) !important;
            }
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a[href^="#section-"].active:focus-visible {
                box-shadow:
                    0 0 0 1px rgba(24, 24, 27, 0.06),
                    0 2px 6px rgba(24, 24, 27, 0.07) !important;
            }
            /* Exactly one strong frame when .active */
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a[href^="#section-"].active {
                background: #FFFFFF !important;
                color: var(--ink) !important;
                font-weight: 650 !important;
                border-color: #27272A !important;
                box-shadow:
                    0 0 0 1px rgba(24, 24, 27, 0.06),
                    0 2px 6px rgba(24, 24, 27, 0.07) !important;
            }
            /* Tighten paragraphs that wrap the link list. */
            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                margin: 0 !important;
            }

            /* Filter / glossary popover triggers inside sidebar */
            section[data-testid="stSidebar"] [data-testid="stPopover"] {
                width: 100%;
                margin-bottom: 6px;
            }
            section[data-testid="stSidebar"] [data-testid="stPopover"] button {
                font-weight: 600 !important;
                border: 1px solid var(--grid) !important;
                background: #FFFFFF !important;
                color: var(--ink) !important;
                border-radius: 10px !important;
                padding: 0.55rem 0.85rem !important;
                box-shadow: none !important;
                font-size: 0.86rem !important;
                width: 100% !important;
                justify-content: flex-start !important;
            }
            section[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
                background: #FAFAFA !important;
                border-color: #D4D4D8 !important;
            }

            /* Filters / Glossary popover body: fit viewport, scroll inside (sidebar popovers get clipped). */
            div[data-baseweb="popover"] {
                max-height: min(88vh, 920px) !important;
                max-width: min(560px, calc(100vw - 20px)) !important;
                min-width: min(100%, 300px) !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                box-sizing: border-box !important;
                z-index: 1000002 !important;
                -webkit-overflow-scrolling: touch;
                padding: 0.35rem 0.5rem !important;
            }

            h1 { letter-spacing: -0.045em; }
            h2 { letter-spacing: -0.030em; }
            h3 { letter-spacing: -0.020em; }

            .page-kicker {
                margin-top: 0.4rem;
                color: var(--accent-page, var(--muted));
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }
            .page-title {
                margin: 0.1rem 0 0.25rem 0;
                color: var(--ink);
                font-size: 1.95rem;
                font-weight: 750;
                letter-spacing: -0.040em;
                position: relative;
                padding-left: 0.7rem;
                border-left: 4px solid var(--accent-page, #111111);
            }
            .page-copy {
                margin: 0 0 1.0rem 0;
                color: var(--muted);
                font-size: 0.95rem;
                line-height: 1.55;
                max-width: 920px;
            }

            /* Hide all raw icon text fallbacks inside popover buttons (tune, expand_more). */
            [data-testid="stPopover"] button [data-testid*="stIconMaterial"],
            [data-testid="stPopover"] button [data-testid*="Icon"],
            [data-testid="stPopover"] button .material-symbols-outlined,
            [data-testid="stPopover"] button span[class*="material"] {
                display: none !important;
            }
            [data-testid="stPopover"] button p { margin: 0; }

            /* Glossary popover content */
            .glossary-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 0.55rem;
                margin-top: 0.4rem;
            }
            .gloss-row {
                display: grid;
                grid-template-columns: 110px 1fr;
                gap: 0.6rem;
                align-items: start;
                padding: 0.45rem 0.55rem;
                border: 1px solid var(--grid);
                border-radius: 10px;
                background: var(--card);
            }
            .gloss-tag {
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                font-weight: 700;
                font-size: 0.78rem;
                color: var(--ink);
            }
            .gloss-tag::before {
                content: "";
                display: inline-block;
                width: 8px; height: 8px;
                border-radius: 3px;
                background: var(--tag-color, #18181B);
            }
            .gloss-text { color: var(--muted); font-size: 0.82rem; line-height: 1.45; }
            .gloss-section {
                margin-top: 0.6rem;
                color: var(--ink);
                font-weight: 700;
                font-size: 0.86rem;
            }

            /* KPI cards */
            .kpi {
                padding: 1.25rem 1.35rem;
                background: var(--card);
                border: 1px solid var(--grid);
                border-radius: 14px;
                min-height: 150px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 0.55rem;
            }
            .kpi-label {
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 650;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }
            .kpi-value {
                color: var(--ink);
                font-size: 2.35rem;
                font-weight: 750;
                letter-spacing: -0.04em;
                margin: 0.45rem 0 0.3rem 0;
                line-height: 1.05;
            }
            .kpi-note { color: var(--muted); font-size: 0.9rem; line-height: 1.5; }
            .kpi-up   { color: #10B981; font-weight: 650; }
            .kpi-down { color: #EF4444; font-weight: 650; }

            .insight-row { margin-top: 1.1rem; }
            .insight {
                padding: 1.1rem 1.2rem;
                border: 1px solid var(--grid);
                border-radius: 14px;
                background: var(--card);
                min-height: 110px;
            }
            .insight .badge {
                display: inline-block;
                padding: 0.22rem 0.6rem;
                border-radius: 999px;
                background: #F4F4F5;
                color: #18181B;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .insight .text {
                margin-top: 0.6rem;
                color: var(--ink);
                font-size: 1rem;
                line-height: 1.55;
                font-weight: 500;
            }

            .panel {
                padding: 0.9rem 1rem 0.5rem 1rem;
                background: var(--card);
                border: 1px solid var(--grid);
                border-radius: 14px;
            }

            .section-head {
                margin: 1.6rem 0 0.5rem 0;
                color: var(--ink);
                font-size: 1.18rem;
                font-weight: 700;
                letter-spacing: -0.020em;
            }
            .section-sub {
                margin: 0 0 0.85rem 0;
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
                max-width: 920px;
            }

            .small-note { color: var(--muted); font-size: 0.82rem; }

            .info-tip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 16px; height: 16px;
                margin-left: 6px;
                border-radius: 50%;
                background: #F4F4F5;
                color: #52525B;
                font-size: 10px;
                font-weight: 800;
                cursor: help;
                user-select: none;
                border: 1px solid var(--grid);
            }
            .info-tip:hover { background: #E4E4E7; color: #18181B; }

            div[data-testid="stMetric"] {
                background: var(--card);
                border: 1px solid var(--grid);
                border-radius: 14px;
                padding: 0.9rem;
            }
            div[data-baseweb="popover"] { border-radius: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


SP_LOGO_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 16 L9 9 L13 13 L20 5" stroke="white" stroke-width="2.4"
        stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="4" cy="16" r="1.7" fill="white"/>
  <circle cx="9" cy="9" r="1.7" fill="white"/>
  <circle cx="13" cy="13" r="1.7" fill="white"/>
  <circle cx="20" cy="5" r="1.7" fill="white"/>
  <path d="M4 19 L20 19" stroke="white" stroke-width="2" stroke-linecap="round"/>
</svg>
"""


def set_page_accent(page_key: str) -> None:
    accent = PAGE_ACCENTS.get(page_key, "#111111")
    st.markdown(
        f"<style>:root {{ --accent-page: {accent}; }}</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data pipeline (mirrors analysis.ipynb)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    raw = pd.read_csv(DATA_PATH)
    missing_before = raw.isna().sum()

    df = raw.copy()

    df["coffee_intake_mg"] = pd.to_numeric(
        df["coffee_intake_mg"].astype(str).str.replace(" mg", "", regex=False),
        errors="coerce",
    )
    df["attendance_percentage"] = pd.to_numeric(
        df["attendance_percentage"].astype(str).str.rstrip("%"),
        errors="coerce",
    )

    numeric_cols = [
        "age", "study_hours_per_day", "sleep_hours", "phone_usage_hours",
        "social_media_hours", "youtube_hours", "gaming_hours", "breaks_per_day",
        "coffee_intake_mg", "exercise_minutes", "assignments_completed",
        "attendance_percentage", "stress_level", "focus_score",
        "productivity_score", "final_grade",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
    df["gender"] = df["gender"].fillna("Other")

    for col in ["phone_usage_hours", "coffee_intake_mg", "exercise_minutes", "final_grade"]:
        df[col] = df[col].fillna(df[col].mean())

    df["row_id"] = np.arange(len(df))

    df["screen_time_hours"] = df[
        ["social_media_hours", "youtube_hours", "gaming_hours"]
    ].sum(axis=1)
    df["distraction_ratio"] = df["screen_time_hours"] / (df["study_hours_per_day"] + 0.5)

    df["performance_segment"] = pd.cut(
        df["final_grade"],
        bins=[0, 60, 70, 80, 90, 101],
        labels=SEGMENT_ORDER,
        right=False,
    )
    df["digital_load"] = pd.cut(
        df["screen_time_hours"],
        bins=[-0.1, 4, 7, 10, 30],
        labels=DIGITAL_ORDER,
    )
    df["sleep_quality"] = pd.cut(
        df["sleep_hours"],
        bins=[0, 6, 8, 24],
        labels=SLEEP_ORDER,
        right=False,
    )
    df["study_band"] = pd.cut(
        df["study_hours_per_day"],
        bins=[-0.1, 3, 6, 8, 24],
        labels=STUDY_ORDER,
    )
    df["stress_zone"] = pd.cut(
        df["stress_level"],
        bins=[0, 3, 6, 8, 10],
        labels=STRESS_ORDER,
        include_lowest=True,
    )
    df["exercise_group"] = pd.qcut(
        df["exercise_minutes"], q=3, labels=EXERCISE_ORDER, duplicates="drop",
    )

    for col, order in {
        "performance_segment": SEGMENT_ORDER,
        "digital_load": DIGITAL_ORDER,
        "sleep_quality": SLEEP_ORDER,
        "study_band": STUDY_ORDER,
        "stress_zone": STRESS_ORDER,
        "exercise_group": EXERCISE_ORDER,
    }.items():
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    miss_df = pd.DataFrame(
        {"column": missing_before.index, "missing_rows": missing_before.values}
    )
    miss_df = miss_df[miss_df["missing_rows"] > 0].reset_index(drop=True)

    return df, miss_df, len(raw)


def friendly(col: str) -> str:
    return LABELS.get(col, col.replace("_", " ").title())


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------


def style_fig(fig: go.Figure, height: int = 380, legend_below: bool = True) -> go.Figure:
    legend_dict = (
        dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=12, color=MUTED),
            bgcolor="rgba(0,0,0,0)",
        )
        if legend_below
        else dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="left", x=0,
            font=dict(size=12, color=MUTED),
            bgcolor="rgba(0,0,0,0)",
        )
    )

    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=66, b=70 if legend_below else 28),
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            color=INK, size=12,
        ),
        title=dict(
            x=0, xanchor="left", y=0.96, yanchor="top",
            font=dict(size=17, color=INK),
        ),
        legend=legend_dict,
        hoverlabel=dict(
            bgcolor="#111111", font_color="#FFFFFF", bordercolor="#111111",
            font=dict(size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=GRID, showline=False,
        tickfont=dict(size=12, color=MUTED),
        title_font=dict(size=13, color=MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=GRID, showline=False,
        tickfont=dict(size=12, color=MUTED),
        title_font=dict(size=13, color=MUTED),
    )
    return fig


def heatmap_mean_grade(
    pivot: pd.DataFrame,
    *,
    title: str,
    x_title: str,
    y_title: str,
    hover_template: str,
) -> go.Figure:
    """Mean final grade heatmap with per-cell values and categorical axes."""
    x_cats = [str(c) for c in pivot.columns]
    y_cats = [str(c) for c in pivot.index]
    z = pivot.values.astype(float)
    text = [
        [f"{v:.1f}" if np.isfinite(v) else "" for v in row]
        for row in z
    ]
    return go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_cats,
            y=y_cats,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=13, color="#0f172a"),
            colorscale=[[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"]],
            colorbar=dict(title="Mean final grade"),
            hovertemplate=hover_template,
        )
    ).update_layout(
        title=title,
        xaxis=dict(title=x_title, type="category", categoryorder="array", categoryarray=x_cats),
        yaxis=dict(title=y_title, type="category", categoryorder="array", categoryarray=y_cats),
    )


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# UI primitives
# ---------------------------------------------------------------------------


def kpi(label: str, value: str, note_html: str = "", info: str = "") -> None:
    tip = ""
    if info:
        safe = info.replace('"', "&quot;")
        tip = f'<span class="info-tip" title="{safe}">i</span>'
    st.markdown(
        f"""
        <div class="kpi">
            <div>
                <div class="kpi-label">{label}{tip}</div>
                <div class="kpi-value">{value}</div>
            </div>
            <div class="kpi-note">{note_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(badge: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="insight">
            <span class="badge">{badge}</span>
            <div class="text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="page-kicker">{kicker}</div>
        <div class="page-title">{title}</div>
        <p class="page-copy">{copy}</p>
        """,
        unsafe_allow_html=True,
    )


def section_head(title: str, sub: str = "") -> None:
    sub_html = f"<p class='section-sub'>{sub}</p>" if sub else ""
    st.markdown(
        f"<div class='section-head'>{title}</div>{sub_html}",
        unsafe_allow_html=True,
    )


def delta_html(value: float, baseline: float, suffix: str = "", lower_is_better: bool = False) -> str:
    if pd.isna(value) or pd.isna(baseline):
        return "<span class='small-note'>No baseline</span>"
    diff = value - baseline
    if abs(diff) < 1e-9:
        return "<span class='small-note'>Matches cohort baseline</span>"
    is_positive = diff >= 0
    is_good = (not is_positive) if lower_is_better else is_positive
    cls = "kpi-up" if is_good else "kpi-down"
    sign = "+" if is_positive else ""
    return f"<span class='{cls}'>{sign}{diff:.1f}{suffix}</span> vs cohort"


def top_decile_card(
    df_full: pd.DataFrame,
    view: pd.DataFrame,
    column: str,
    decile: str = "top",
    label: str | None = None,
    unit: str = "",
    fmt: str = "{:.1f}",
    lower_is_better: bool = False,
) -> None:
    cohort_mean = view[column].mean()
    if decile == "top":
        threshold = view[column].quantile(0.9)
        cohort = view[view[column] >= threshold]
        decile_label = f"Top 10% (>= {fmt.format(threshold)}{unit})"
    else:
        threshold = view[column].quantile(0.1)
        cohort = view[view[column] <= threshold]
        decile_label = f"Bottom 10% (<= {fmt.format(threshold)}{unit})"

    decile_mean = cohort[column].mean() if not cohort.empty else float("nan")
    diff = decile_mean - cohort_mean if not pd.isna(decile_mean) else 0
    if cohort_mean and not pd.isna(cohort_mean):
        pct = (diff / cohort_mean) * 100 if cohort_mean != 0 else 0
    else:
        pct = 0

    is_positive = diff >= 0
    is_good = (not is_positive) if lower_is_better else is_positive
    cls = "kpi-up" if is_good else "kpi-down"
    sign = "+" if is_positive else ""

    note = (
        f"Cohort avg <b>{fmt.format(cohort_mean)}{unit}</b><br>"
        f"<span class='{cls}'>{sign}{pct:.1f}%</span> {decile_label}"
    )
    kpi(label or friendly(column), f"{fmt.format(decile_mean)}{unit}", note)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def init_filters(df: pd.DataFrame) -> None:
    if st.session_state.get("filters_initialized"):
        return
    st.session_state.f_gender = sorted(df["gender"].dropna().unique().tolist())
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    st.session_state.f_age = (age_min, age_max)
    st.session_state.f_segments = SEGMENT_ORDER.copy()
    st.session_state.f_stress = (int(df["stress_level"].min()), int(df["stress_level"].max()))
    screen_max = float(np.ceil(df["screen_time_hours"].max()))
    st.session_state.f_screen = (0.0, screen_max)
    study_max = float(np.ceil(df["study_hours_per_day"].max()))
    st.session_state.f_study = (0.0, study_max)
    st.session_state.filters_initialized = True


def reset_filters(df: pd.DataFrame) -> None:
    for key in [
        "f_gender", "f_age", "f_segments", "f_stress",
        "f_screen", "f_study", "filters_initialized",
    ]:
        st.session_state.pop(key, None)
    init_filters(df)


def render_filter_popover(df: pd.DataFrame) -> None:
    init_filters(df)
    label = f"Filters  ({active_filter_count(df)})"
    with st.popover(label, width="stretch"):
        st.markdown("**Filter the cohort**")
        st.caption("Filters apply to every page.")
        st.session_state.f_gender = st.multiselect(
            "Gender",
            options=sorted(df["gender"].dropna().unique().tolist()),
            default=st.session_state.f_gender,
        )
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        st.session_state.f_age = st.slider("Age", age_min, age_max, st.session_state.f_age)
        st.session_state.f_segments = st.multiselect(
            "Performance segment", options=SEGMENT_ORDER, default=st.session_state.f_segments,
        )
        st.session_state.f_stress = st.slider(
            "Stress level", 1, 10, st.session_state.f_stress,
        )
        screen_max = float(np.ceil(df["screen_time_hours"].max()))
        st.session_state.f_screen = st.slider(
            "Screen time (h)", 0.0, screen_max, st.session_state.f_screen, 0.5,
        )
        study_max = float(np.ceil(df["study_hours_per_day"].max()))
        st.session_state.f_study = st.slider(
            "Study hours (h)", 0.0, study_max, st.session_state.f_study, 0.5,
        )
        st.divider()
        if st.button("Reset filters", width="stretch"):
            reset_filters(df)
            st.rerun()


def active_filter_count(df: pd.DataFrame) -> int:
    count = 0
    if sorted(st.session_state.f_gender) != sorted(df["gender"].dropna().unique().tolist()):
        count += 1
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    if tuple(st.session_state.f_age) != (age_min, age_max):
        count += 1
    if sorted(st.session_state.f_segments) != sorted(SEGMENT_ORDER):
        count += 1
    if tuple(st.session_state.f_stress) != (1, 10):
        count += 1
    screen_max = float(np.ceil(df["screen_time_hours"].max()))
    if tuple(st.session_state.f_screen) != (0.0, screen_max):
        count += 1
    study_max = float(np.ceil(df["study_hours_per_day"].max()))
    if tuple(st.session_state.f_study) != (0.0, study_max):
        count += 1
    return count


GLOSSARY_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Performance segments (built from final grade)",
        [
            ("Elite", "#10B981", "Final grade 90 or higher."),
            ("Strong", "#2563EB", "Final grade in [80, 90)."),
            ("Stable", "#A1A1AA", "Final grade in [70, 80)."),
            ("At Risk", "#F59E0B", "Final grade in [60, 70)."),
            ("Critical", "#EF4444", "Final grade below 60."),
        ],
    ),
    (
        "Digital load (total entertainment hours per day)",
        [
            ("Low", "#10B981", "Less than 4 hours of social media + YouTube + gaming."),
            ("Balanced", "#2563EB", "Between 4 and 7 hours."),
            ("Heavy", "#F59E0B", "Between 7 and 10 hours."),
            ("Overloaded", "#EF4444", "More than 10 hours of entertainment screen time."),
        ],
    ),
    (
        "Sleep quality",
        [
            ("Short", "#EF4444", "Less than 6 hours of sleep on average."),
            ("Balanced", "#2563EB", "Between 6 and 8 hours of sleep."),
            ("Long", "#10B981", "More than 8 hours of sleep."),
        ],
    ),
    (
        "Study band (daily study hours)",
        [
            ("Light", "#EF4444", "Up to 3 study hours per day."),
            ("Core", "#F59E0B", "Between 3 and 6 hours."),
            ("Focused", "#2563EB", "Between 6 and 8 hours."),
            ("Intensive", "#10B981", "More than 8 hours of study per day."),
        ],
    ),
    (
        "Stress zone (1-10 self-reported)",
        [
            ("Calm", "#10B981", "Stress level 1-3."),
            ("Managed", "#2563EB", "Stress level 4-6."),
            ("High", "#F59E0B", "Stress level 7-8."),
            ("Burnout", "#EF4444", "Stress level 9-10."),
        ],
    ),
    (
        "Exercise group (qcut tertiles)",
        [
            ("Low", "#EF4444", "Bottom third of daily exercise minutes."),
            ("Med", "#F59E0B", "Middle third of daily exercise minutes."),
            ("High", "#10B981", "Top third of daily exercise minutes."),
        ],
    ),
]


GLOSSARY_METRICS: list[tuple[str, str]] = [
    ("Focus score", "Self-reported attention score during study sessions."),
    ("Productivity score", "Output index combining study output, completion and self-rating."),
    ("Final grade", "End-of-term overall grade out of 100."),
    ("Distraction ratio", "Entertainment screen time divided by (study hours + 0.5). Higher = more distracted."),
    ("Top 10%", "Top decile of the selected cohort by final grade, used as a 'success benchmark'."),
    ("Binned mean", "Black overlay line on scatters: the average of Y inside equal-width X bins."),
]


def render_glossary_popover() -> None:
    label = f"Glossary  ({sum(len(items) for _, items in GLOSSARY_GROUPS) + len(GLOSSARY_METRICS)})"
    with st.popover(label, width="stretch"):
        st.markdown("**What the terms mean**")
        st.caption("Definitions for every derived label used in the charts.")
        for group_title, items in GLOSSARY_GROUPS:
            st.markdown(f"<div class='gloss-section'>{group_title}</div>", unsafe_allow_html=True)
            rows = "".join(
                f"<div class='gloss-row' style='--tag-color: {color};'>"
                f"<div class='gloss-tag'>{tag}</div>"
                f"<div class='gloss-text'>{text}</div>"
                f"</div>"
                for tag, color, text in items
            )
            st.markdown(f"<div class='glossary-grid'>{rows}</div>", unsafe_allow_html=True)
        st.markdown("<div class='gloss-section'>Metrics and helpers</div>", unsafe_allow_html=True)
        rows = "".join(
            f"<div class='gloss-row' style='--tag-color: #18181B;'>"
            f"<div class='gloss-tag'>{name}</div>"
            f"<div class='gloss-text'>{text}</div>"
            f"</div>"
            for name, text in GLOSSARY_METRICS
        )
        st.markdown(f"<div class='glossary-grid'>{rows}</div>", unsafe_allow_html=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["gender"].isin(st.session_state.f_gender)
        & df["age"].between(*st.session_state.f_age)
        & df["performance_segment"].astype(str).isin(st.session_state.f_segments)
        & df["stress_level"].between(*st.session_state.f_stress)
        & df["screen_time_hours"].between(*st.session_state.f_screen)
        & df["study_hours_per_day"].between(*st.session_state.f_study)
    ].copy()


# ---------------------------------------------------------------------------
# Header & navigation
# ---------------------------------------------------------------------------


def render_sidebar(df: pd.DataFrame, view: pd.DataFrame) -> None:
    """Render the real Streamlit sidebar with brand, cohort badge, nav anchors and popovers."""
    share_pct = (len(view) / len(df) * 100) if len(df) else 0

    # Brand block + cohort badge (HTML)
    st.sidebar.markdown(
        f"""
        <div class="sb-brand">
            <span class="sb-logo">{SP_LOGO_SVG}</span>
            <span class="sb-brand-text">
                <span class="sb-brand-name">Student Productivity</span>
                <span class="sb-brand-sub">Dashboard &middot; Kaggle 20K</span>
            </span>
        </div>
        <div class="sb-cohort">
            <div class="sb-cohort-kicker">Active cohort</div>
            <div class="sb-cohort-row">
                <span class="sb-cohort-main">{len(view):,}</span>
                <span class="sb-cohort-den">/ {len(df):,} students</span>
            </div>
            <div class="sb-cohort-pct">{share_pct:.1f}% of full dataset after filters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation anchor links (single long page, scroll-spy active state via JS).
    # Use <div> wrappers (always allowed by Streamlit's HTML sanitizer) and target the
    # bare <a> children via the parent class; <nav> and class="sb-link" on <a> get stripped.
    nav_links = "".join(
        f'<a href="#section-{key}">{label}</a>' for key, label in PAGES
    )
    st.sidebar.markdown(
        '<div class="sb-label"><span class="sb-label-inner">Page list</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div class="sb-nav">{nav_links}</div>',
        unsafe_allow_html=True,
    )

    # Filters & glossary popovers (real Streamlit components)
    st.sidebar.markdown(
        '<div class="sb-label"><span class="sb-label-inner">Tools</span></div>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        render_filter_popover(df)
        render_glossary_popover()


def inject_nav_scroll_spy() -> None:
    """Run scroll-spy after main blocks exist. Fixes sidebar highlight stuck on Overview.

    (1) Script in sidebar ran before section DOM existed; (2) pickActiveKey used an
    erroneous early ``break`` so the active key rarely advanced past the first section.
    """
    keys_json = json.dumps([k for k, _ in PAGES])
    st.html(
        f"""
        <script>
        (function() {{
            var KEYS = {keys_json};
            var NAV_SEL = 'section[data-testid="stSidebar"] a[href^="#section-"]';

            function navLinks() {{
                return Array.prototype.slice.call(document.querySelectorAll(NAV_SEL));
            }}

            function sectionPairs() {{
                var out = [];
                KEYS.forEach(function(k) {{
                    var el = document.getElementById('section-' + k);
                    if (el) out.push({{ key: k, el: el }});
                }});
                return out;
            }}

            function applyActive(key) {{
                var sel = '#section-' + key;
                navLinks().forEach(function(x) {{
                    x.classList.toggle('active', (x.getAttribute('href') || '') === sel);
                }});
            }}

            function collectScrollRoots(fromEl) {{
                var roots = [];
                var seen = new Set();
                function add(el) {{
                    if (!el || seen.has(el)) return;
                    seen.add(el);
                    roots.push(el);
                }}
                var p = fromEl;
                while (p && p !== document.documentElement) {{
                    add(p);
                    p = p.parentElement;
                }}
                [
                    document.querySelector('[data-testid="stAppViewContainer"]'),
                    document.querySelector('[data-testid="stMain"]'),
                    document.querySelector('[data-testid="stMainBlockContainer"]'),
                    document.scrollingElement,
                    document.documentElement,
                    document.body,
                ].forEach(add);
                return roots;
            }}

            /** Last section whose top has crossed the anchor line (no early break). */
            function pickActiveKey(pairs) {{
                var anchor = Math.min(240, Math.max(96, Math.floor(window.innerHeight * 0.2)));
                var key = pairs.length ? pairs[0].key : KEYS[0];
                for (var i = 0; i < pairs.length; i++) {{
                    var top = pairs[i].el.getBoundingClientRect().top;
                    if (top - 8 <= anchor) {{
                        key = pairs[i].key;
                    }}
                }}
                return key;
            }}

            function init() {{
                var pairs = sectionPairs();
                var links = navLinks();
                if (!pairs.length || !links.length) {{
                    return setTimeout(init, 250);
                }}

                if (typeof window.__spSpyCleanup === 'function') {{
                    try {{ window.__spSpyCleanup(); }} catch (e0) {{}}
                }}

                var rafId = null;
                function schedule() {{
                    if (rafId) return;
                    rafId = requestAnimationFrame(function() {{
                        rafId = null;
                        applyActive(pickActiveKey(sectionPairs()));
                    }});
                }}

                var sb = document.querySelector('section[data-testid="stSidebar"]');
                function onNavClick(e) {{
                    var t = e.target;
                    if (!t || !t.closest) return;
                    var a = t.closest('a[href^="#section-"]');
                    if (!a || !sb || !sb.contains(a)) return;
                    var href = a.getAttribute('href') || '';
                    var id = href.replace(/^#/, '');
                    var target = document.getElementById(id);
                    if (!target) return;
                    e.preventDefault();
                    applyActive(id.replace(/^section-/, ''));
                    target.scrollIntoView({{ behavior: 'auto', block: 'start' }});
                    requestAnimationFrame(schedule);
                    setTimeout(schedule, 100);
                }}
                if (sb) {{
                    sb.addEventListener('click', onNavClick);
                }}

                var listeners = [];
                function on(el, ev, fn, opts) {{
                    el.addEventListener(ev, fn, opts || {{ passive: true }});
                    listeners.push([el, ev, fn, opts]);
                }}

                collectScrollRoots(pairs[0].el).forEach(function(r) {{
                    on(r, 'scroll', schedule);
                }});
                on(window, 'scroll', schedule, {{ passive: true, capture: true }});
                on(window, 'resize', schedule);

                var io = null;
                if (typeof IntersectionObserver !== 'undefined') {{
                    io = new IntersectionObserver(schedule, {{
                        root: null,
                        rootMargin: '-10% 0px -52% 0px',
                        threshold: [0, 0.05, 0.25, 0.5, 1],
                    }});
                    pairs.forEach(function(p) {{ io.observe(p.el); }});
                }}

                var pollId = setInterval(schedule, 160);

                window.__spSpyCleanup = function() {{
                    if (rafId) cancelAnimationFrame(rafId);
                    clearInterval(pollId);
                    if (sb) {{
                        try {{ sb.removeEventListener('click', onNavClick); }} catch (e4) {{}}
                    }}
                    listeners.forEach(function(l) {{
                        try {{ l[0].removeEventListener(l[1], l[2], l[3]); }} catch (e2) {{}}
                    }});
                    if (io) {{ try {{ io.disconnect(); }} catch (e3) {{}} }}
                }};

                schedule();
            }}
            setTimeout(init, 0);
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


# ---------------------------------------------------------------------------
# Shared analytics helpers
# ---------------------------------------------------------------------------


CORR_COLS = [
    "study_hours_per_day", "sleep_hours", "phone_usage_hours",
    "social_media_hours", "youtube_hours", "gaming_hours",
    "screen_time_hours", "exercise_minutes", "assignments_completed",
    "attendance_percentage", "stress_level", "focus_score",
    "productivity_score", "coffee_intake_mg", "breaks_per_day",
]


def grade_corr(view: pd.DataFrame, cols: list[str] | None = None) -> pd.Series:
    use = [c for c in (cols or CORR_COLS) if c in view.columns]
    series = (
        view[use + ["final_grade"]]
        .corr(numeric_only=True)["final_grade"]
        .drop("final_grade")
        .dropna()
    )
    return series


def binned_mean(view: pd.DataFrame, x: str, y: str, bins: int = 12) -> pd.DataFrame:
    if view[x].nunique() < 2:
        return pd.DataFrame({x: [], y: []})
    edges = np.linspace(view[x].min(), view[x].max(), bins + 1)
    binned = pd.cut(view[x], bins=edges, include_lowest=True)
    out = (
        view.assign(_bin=binned)
        .groupby("_bin", observed=True)
        .agg(**{y: (y, "mean")})
        .dropna()
    )
    mids = [interval.mid for interval in out.index]
    out = out.reset_index(drop=True)
    out[x] = mids
    return out[[x, y]]


def add_binned_overlay(fig: go.Figure, view: pd.DataFrame, x: str, y: str, name: str = "Average") -> None:
    binned = binned_mean(view, x, y, bins=12)
    if binned.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=binned[x], y=binned[y],
            mode="lines+markers", name=name,
            line=dict(color="#0A0A0A", width=3),
            marker=dict(size=7, color="#0A0A0A"),
        )
    )


# ---------------------------------------------------------------------------
# Page 1 - Overview
# ---------------------------------------------------------------------------


def page_overview(view: pd.DataFrame, df: pd.DataFrame) -> None:
    page_intro(
        "Overview",
        "Snapshot of the cohort",
        "A high-level look at how the selected students distribute across grade, focus, study and sleep, "
        "plus the strongest variables linked to academic outcomes.",
    )

    if view.empty:
        st.warning("No students match the current filters. Adjust the filter popover.")
        return

    top10_grade = view[view["final_grade"] >= view["final_grade"].quantile(0.9)]
    cohort_grade = view["final_grade"].mean()
    top10_grade_mean = top10_grade["final_grade"].mean()
    elite_strong_share = view["performance_segment"].astype(str).isin(["Elite", "Strong"]).mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Students", f"{len(view):,}",
            f"{len(view) / len(df) * 100:.1f}% of clean records")
    with k2:
        kpi("Average final grade", f"{cohort_grade:.1f}",
            f"Top 10% average <b>{top10_grade_mean:.1f}</b> "
            f"<span class='kpi-up'>+{(top10_grade_mean - cohort_grade):.1f}</span> vs cohort")
    with k3:
        focus_top10 = view[view["focus_score"] >= view["focus_score"].quantile(0.9)]["focus_score"].mean()
        kpi("Average focus score", f"{view['focus_score'].mean():.1f}",
            f"Top decile <b>{focus_top10:.1f}</b>")
    with k4:
        prod_top10 = view[view["productivity_score"] >= view["productivity_score"].quantile(0.9)]["productivity_score"].mean()
        kpi("Average productivity", f"{view['productivity_score'].mean():.1f}",
            f"Top decile <b>{prod_top10:.1f}</b>")
    with k5:
        kpi("Strong + Elite share", f"{elite_strong_share:.1f}%",
            f"Risk share <b>{view['performance_segment'].astype(str).isin(['At Risk', 'Critical']).mean() * 100:.1f}%</b>",
            info="Strong = final grade in [80, 90). Elite = 90 or higher. See Glossary for the full ladder.")

    corr = grade_corr(view)
    if not corr.empty:
        pos = corr.idxmax()
        neg = corr.idxmin()
        biggest = corr.abs().idxmax()
        st.markdown("<div class='insight-row'></div>", unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        with i1:
            insight(
                "Positive driver",
                f"<b>{friendly(pos)}</b> is the strongest positive link with final grade "
                f"(r = {corr[pos]:+.2f}).",
            )
        with i2:
            insight(
                "Negative drag",
                f"<b>{friendly(neg)}</b> drags grades down the hardest "
                f"(r = {corr[neg]:+.2f}).",
            )
        with i3:
            insight(
                "Decision signal",
                f"<b>{friendly(biggest)}</b> carries the highest absolute correlation with final grade.",
            )

    section_head(
        "Cohort distribution",
        "Choose a numeric field for the horizontal axis. The histogram, colour, title, "
        "and dashed cohort average update to match.",
    )

    dist_pairs = [
        (c, h) for c, h in OVERVIEW_DIST_VARS
        if c in view.columns and pd.api.types.is_numeric_dtype(view[c])
        and view[c].notna().any()
    ]
    if not dist_pairs:
        st.caption("No numeric columns in this view for a distribution plot.")
    else:
        cols_order = [c for c, _ in dist_pairs]
        color_by_col = dict(dist_pairs)
        default_i = cols_order.index("final_grade") if "final_grade" in cols_order else 0
        x_col = st.selectbox(
            "Variable on horizontal axis",
            options=cols_order,
            index=default_i,
            format_func=friendly,
            key="overview_dist_x",
            # No in-widget typing / caret — classic pick-only dropdown (Streamlit ≥ 1.45).
            filter_mode=None,
        )
        bar_color = color_by_col.get(x_col, "#3B82F6")
        xl = friendly(x_col)
        fig = px.histogram(
            view.dropna(subset=[x_col]),
            x=x_col,
            nbins=min(48, max(24, int(np.clip(len(view) ** 0.45, 24, 48)))),
            title=f"Distribution of {xl.lower()}",
            color_discrete_sequence=[bar_color],
            labels={x_col: xl},
        )
        fig.update_traces(marker_line_width=0, opacity=0.88)
        mean = float(view[x_col].mean())
        fig.add_vline(
            x=mean,
            line_dash="dot",
            line_color="#0A0A0A",
            annotation_text=f"Cohort average {mean:.2f}",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#0A0A0A"),
        )
        fig.update_yaxes(title="Students")
        fig.update_layout(showlegend=False)
        render_chart(style_fig(fig, height=460, legend_below=False))

    section_head(
        "Segment composition and variable impact",
        "Who is in each performance tier and which variables move with final grade the most.",
    )

    g_l, g_r = st.columns([0.95, 1.05])
    with g_l:
        seg_counts = (
            view["performance_segment"].value_counts()
            .reindex(SEGMENT_ORDER).fillna(0).astype(int).reset_index()
        )
        seg_counts.columns = ["segment", "students"]
        seg_counts["share"] = seg_counts["students"] / seg_counts["students"].sum() * 100
        fig = px.pie(
            seg_counts, values="students", names="segment",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            hole=0.55,
            title="Performance segment composition",
        )
        fig.update_traces(
            textinfo="label+percent", textposition="outside",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        fig.update_layout(showlegend=False)
        render_chart(style_fig(fig, height=400, legend_below=False))

    with g_r:
        if not corr.empty:
            impact = corr.abs().sort_values(ascending=True).tail(10)
            impact_df = pd.DataFrame({
                "metric": [friendly(c) for c in impact.index],
                "impact": impact.values,
                "direction": ["Positive" if corr[c] >= 0 else "Negative" for c in impact.index],
            })
            fig = px.bar(
                impact_df,
                x="impact", y="metric",
                color="direction",
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
                orientation="h",
                title="Variables ranked by absolute correlation with final grade",
                text=[f"{v:.2f}" for v in impact_df["impact"]],
            )
            fig.update_traces(marker_line_width=0, textposition="outside",
                              textfont=dict(size=11, color=MUTED))
            fig.update_xaxes(title="Absolute correlation |r|", range=[0, max(0.6, impact_df["impact"].max() * 1.18)])
            fig.update_yaxes(title=None)
            fig.update_layout(legend_title=None)
            render_chart(style_fig(fig, height=400, legend_below=True))

    section_head(
        "Demographics",
        "Performance distribution by age and gender across the selected cohort.",
    )
    demo_left, demo_right = st.columns(2)
    with demo_left:
        age_bin = pd.cut(view["age"], bins=[16, 19, 22, 25, 28, 31],
                         labels=["17-18", "19-21", "22-24", "25-27", "28-30"])
        demo = (
            view.assign(age_bin=age_bin)
            .groupby(["age_bin", "performance_segment"], observed=True)
            .size().reset_index(name="students")
        )
        fig = px.bar(
            demo, x="age_bin", y="students", color="performance_segment",
            color_discrete_map=SEGMENT_COLORS, barmode="stack",
            category_orders={"performance_segment": SEGMENT_ORDER},
            title="Performance segment by age band",
            labels={"age_bin": "Age band", "students": "Students",
                    "performance_segment": "Segment"},
        )
        fig.update_traces(marker_line_width=0)
        render_chart(style_fig(fig, height=380, legend_below=True))
    with demo_right:
        demo_g = (
            view.groupby(["gender", "performance_segment"], observed=True)
            .size().reset_index(name="students")
        )
        fig = px.bar(
            demo_g, x="gender", y="students", color="performance_segment",
            color_discrete_map=SEGMENT_COLORS, barmode="stack",
            category_orders={"performance_segment": SEGMENT_ORDER},
            title="Performance segment by gender",
            labels={"gender": "Gender", "students": "Students",
                    "performance_segment": "Segment"},
        )
        fig.update_traces(marker_line_width=0)
        render_chart(style_fig(fig, height=380, legend_below=True))


# ---------------------------------------------------------------------------
# Page 2 - Digital Habits
# ---------------------------------------------------------------------------


DIGITAL_CHANNELS = [
    ("phone_usage_hours", "Phone usage", "#2563EB"),
    ("social_media_hours", "Social media", "#8B5CF6"),
    ("youtube_hours", "YouTube", "#EF4444"),
    ("gaming_hours", "Gaming", "#F59E0B"),
]


def page_digital(view: pd.DataFrame, df: pd.DataFrame) -> None:
    page_intro(
        "Digital habits",
        "How daily screen exposure tracks with final grade",
        "Phone, social media, YouTube and gaming hours are compared directly against final grade. "
        "Two- and three-way views highlight where digital load becomes a real performance drag.",
    )

    if view.empty:
        st.warning("No students match the current filters.")
        return

    section_head(
        "Top decile cards",
        "Top 10% students (by final grade) typically spend less time on entertainment channels than the overall cohort.",
    )
    top10 = view[view["final_grade"] >= view["final_grade"].quantile(0.9)].copy()
    bot10 = view[view["final_grade"] <= view["final_grade"].quantile(0.1)].copy()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        top_phone = top10["phone_usage_hours"].mean()
        cohort_phone = view["phone_usage_hours"].mean()
        pct = (top_phone - cohort_phone) / cohort_phone * 100 if cohort_phone else 0
        cls = "kpi-up" if pct < 0 else "kpi-down"
        kpi("Phone usage (top 10% grade)", f"{top_phone:.1f}h",
            f"Cohort <b>{cohort_phone:.1f}h</b><br>"
            f"<span class='{cls}'>{pct:+.1f}%</span> vs cohort",
            info="Top 10% = the highest-grade decile of the filtered cohort. We compare their average phone usage with the cohort average.")
    with k2:
        top_social = top10["social_media_hours"].mean()
        cohort_social = view["social_media_hours"].mean()
        pct = (top_social - cohort_social) / cohort_social * 100 if cohort_social else 0
        cls = "kpi-up" if pct < 0 else "kpi-down"
        kpi("Social media (top 10% grade)", f"{top_social:.1f}h",
            f"Cohort <b>{cohort_social:.1f}h</b><br>"
            f"<span class='{cls}'>{pct:+.1f}%</span> vs cohort")
    with k3:
        top_yt = top10["youtube_hours"].mean()
        cohort_yt = view["youtube_hours"].mean()
        pct = (top_yt - cohort_yt) / cohort_yt * 100 if cohort_yt else 0
        cls = "kpi-up" if pct < 0 else "kpi-down"
        kpi("YouTube (top 10% grade)", f"{top_yt:.1f}h",
            f"Cohort <b>{cohort_yt:.1f}h</b><br>"
            f"<span class='{cls}'>{pct:+.1f}%</span> vs cohort")
    with k4:
        top_game = top10["gaming_hours"].mean()
        cohort_game = view["gaming_hours"].mean()
        pct = (top_game - cohort_game) / cohort_game * 100 if cohort_game else 0
        cls = "kpi-up" if pct < 0 else "kpi-down"
        kpi("Gaming (top 10% grade)", f"{top_game:.1f}h",
            f"Cohort <b>{cohort_game:.1f}h</b><br>"
            f"<span class='{cls}'>{pct:+.1f}%</span> vs cohort")

    digital_cols = ["phone_usage_hours", "social_media_hours", "youtube_hours", "gaming_hours"]
    corr = grade_corr(view, digital_cols + ["screen_time_hours"])
    if not corr.empty:
        worst = corr.idxmin()
        screen_gap = top10["screen_time_hours"].mean() - bot10["screen_time_hours"].mean()
        i1, i2 = st.columns(2)
        with i1:
            insight(
                "Heaviest negative driver",
                f"Among digital channels, <b>{friendly(worst)}</b> shows the most negative link "
                f"with final grade (r = {corr[worst]:+.2f}).",
            )
        with i2:
            insight(
                "Top vs bottom decile gap",
                f"Top 10% students use <b>{abs(screen_gap):.1f}h</b> "
                f"{'less' if screen_gap < 0 else 'more'} entertainment screen time per day than the bottom 10%.",
            )

    section_head(
        "Each digital channel vs final grade",
        "Hex bins make the cloud of 20K rows readable, and the black line shows the binned average for each channel.",
    )
    row1 = st.columns(2)
    row2 = st.columns(2)
    cells = [row1[0], row1[1], row2[0], row2[1]]
    for cell, (col, title, color) in zip(cells, DIGITAL_CHANNELS):
        with cell:
            r, g, b = _hex_to_rgb(color)
            fig = px.density_heatmap(
                view, x=col, y="final_grade",
                nbinsx=24, nbinsy=24,
                color_continuous_scale=[
                    [0.0, "#FFFFFF"],
                    [0.05, "#EEF2FF"],
                    [0.4, f"rgba({r},{g},{b},0.55)"],
                    [1.0, color],
                ],
                title=f"{title} vs final grade",
                labels={col: friendly(col), "final_grade": "Final grade"},
            )
            add_binned_overlay(fig, view, col, "final_grade", name="Binned mean")
            fig.update_layout(coloraxis_showscale=False)
            render_chart(style_fig(fig, height=320, legend_below=False))

    section_head(
        "Digital load tiers vs grade",
        "Tiers are built on total entertainment screen time (social + YouTube + gaming).",
    )
    col_l, col_r = st.columns([1, 1])
    with col_l:
        fig = px.box(
            view, x="digital_load", y="final_grade",
            color="digital_load", color_discrete_map=DIGITAL_COLORS,
            category_orders={"digital_load": DIGITAL_ORDER}, points=False,
            title="Final grade across digital-load tiers",
            labels={"digital_load": "Digital load", "final_grade": "Final grade"},
        )
        fig.update_traces(line=dict(width=1.4))
        fig.update_layout(showlegend=False)
        render_chart(style_fig(fig, height=380, legend_below=False))
    with col_r:
        channels_cols = ["social_media_hours", "youtube_hours", "gaming_hours"]
        grouped = (
            view.groupby("performance_segment", observed=True)[channels_cols]
            .mean().reindex(SEGMENT_ORDER).dropna(how="all").reset_index()
        )
        melted = grouped.melt("performance_segment", var_name="channel", value_name="hours")
        melted["channel"] = melted["channel"].map(friendly)
        fig = px.bar(
            melted, x="performance_segment", y="hours", color="channel",
            barmode="stack",
            color_discrete_sequence=["#8B5CF6", "#EF4444", "#F59E0B"],
            category_orders={"performance_segment": SEGMENT_ORDER},
            title="Daily entertainment hours by performance segment",
            labels={"performance_segment": "Segment", "hours": "Avg hours/day", "channel": "Channel"},
        )
        fig.update_traces(marker_line_width=0)
        render_chart(style_fig(fig, height=380, legend_below=True))

    section_head(
        "Three-way views",
        "Two variables on the axes, a third encoded by color or position. Heatmaps reveal interaction effects.",
    )
    three_l, three_r = st.columns(2)
    with three_l:
        sm_bins = pd.cut(view["social_media_hours"],
                         bins=[-0.1, 1, 2, 3, 4, 6, 10],
                         labels=["<1", "1-2", "2-3", "3-4", "4-6", "6+"])
        gm_bins = pd.cut(view["gaming_hours"],
                         bins=[-0.1, 0.5, 1.5, 3, 5, 10],
                         labels=["<0.5", "0.5-1.5", "1.5-3", "3-5", "5+"])
        pivot = (
            view.assign(sm=sm_bins, gm=gm_bins)
            .pivot_table(index="gm", columns="sm", values="final_grade", aggfunc="mean", observed=True)
        )
        fig = heatmap_mean_grade(
            pivot,
            title="Final grade by social media x gaming",
            x_title="Social media (h/day)",
            y_title="Gaming (h/day)",
            hover_template="Social media: %{x}h<br>Gaming: %{y}h<br>Mean grade: %{z:.1f}<extra></extra>",
        )
        render_chart(style_fig(fig, height=420, legend_below=False))

    with three_r:
        sample = view.sample(min(len(view), 3500), random_state=11) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="social_media_hours", y="phone_usage_hours",
            color="performance_segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"performance_segment": SEGMENT_ORDER},
            size="final_grade", size_max=14, opacity=0.6,
            title="Phone vs social media, sized by final grade",
            labels={"social_media_hours": "Social media (h)",
                    "phone_usage_hours": "Phone usage (h)",
                    "performance_segment": "Segment", "final_grade": "Grade"},
            hover_data={"final_grade": ":.1f"},
        )
        fig.update_traces(marker=dict(line=dict(width=0)))
        render_chart(style_fig(fig, height=420, legend_below=True))


# ---------------------------------------------------------------------------
# Page 3 - Academic Effort
# ---------------------------------------------------------------------------


def page_effort(view: pd.DataFrame, df: pd.DataFrame) -> None:
    page_intro(
        "Academic effort",
        "How study, attendance, assignments and physical activity feed grades",
        "Effort metrics are intentionally grouped: study and attendance set the academic baseline, "
        "while assignments completed and exercise reveal discipline and recovery.",
    )

    if view.empty:
        st.warning("No students match the current filters.")
        return

    top10 = view[view["final_grade"] >= view["final_grade"].quantile(0.9)]
    cohort = view

    section_head(
        "Effort scorecards",
        "Top-decile averages and the gap vs the overall cohort.",
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        coh = cohort["attendance_percentage"].mean()
        top = top10["attendance_percentage"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct > 0 else "kpi-down"
        kpi("Attendance (top 10% grade)", f"{top:.1f}%",
            f"Cohort <b>{coh:.1f}%</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort",
            info="Average attendance for students in the top 10% of final grades vs the cohort average.")
    with k2:
        coh = cohort["assignments_completed"].mean()
        top = top10["assignments_completed"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct > 0 else "kpi-down"
        kpi("Assignments (top 10% grade)", f"{top:.1f}",
            f"Cohort <b>{coh:.1f}</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort")
    with k3:
        coh = cohort["study_hours_per_day"].mean()
        top = top10["study_hours_per_day"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct > 0 else "kpi-down"
        kpi("Study hours (top 10% grade)", f"{top:.1f}h",
            f"Cohort <b>{coh:.1f}h</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort")
    with k4:
        coh = cohort["exercise_minutes"].mean()
        top = top10["exercise_minutes"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct > 0 else "kpi-down"
        kpi("Exercise (top 10% grade)", f"{top:.1f} min",
            f"Cohort <b>{coh:.1f} min</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort")

    corr = grade_corr(view, ["study_hours_per_day", "attendance_percentage",
                              "assignments_completed", "exercise_minutes", "breaks_per_day"])
    if not corr.empty:
        positive = corr.idxmax()
        gap = top10[["study_hours_per_day", "attendance_percentage",
                     "assignments_completed", "exercise_minutes"]].mean() - \
            view[view["final_grade"] <= view["final_grade"].quantile(0.1)][
                ["study_hours_per_day", "attendance_percentage",
                 "assignments_completed", "exercise_minutes"]].mean()
        gap_pct = (gap / view[gap.index].mean()) * 100
        biggest_gap = gap_pct.abs().idxmax()
        i1, i2 = st.columns(2)
        with i1:
            insight(
                "Strongest positive driver",
                f"<b>{friendly(positive)}</b> is the strongest effort signal "
                f"(r = {corr[positive]:+.2f}).",
            )
        with i2:
            insight(
                "Biggest top vs bottom gap",
                f"<b>{friendly(biggest_gap)}</b> shows the largest top-vs-bottom decile gap "
                f"({gap_pct[biggest_gap]:+.1f}%).",
            )

    section_head(
        "Effort vs final grade",
        "Two-way relationships between core effort metrics and final grade.",
    )
    row1 = st.columns(2)
    with row1[0]:
        ac = view.groupby("assignments_completed", observed=True).agg(
            mean=("final_grade", "mean"),
            std=("final_grade", "std"),
            n=("final_grade", "count"),
        ).reset_index()
        ac["sem"] = ac["std"] / np.sqrt(ac["n"].clip(lower=1))
        fig = px.bar(
            ac, x="assignments_completed", y="mean",
            error_y="sem",
            color="mean", color_continuous_scale="Greens",
            title="Mean final grade by assignments completed",
            labels={"assignments_completed": "Assignments completed",
                    "mean": "Mean final grade"},
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False)
        render_chart(style_fig(fig, height=380, legend_below=False))

    with row1[1]:
        sample = view.sample(min(len(view), 3500), random_state=21) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="attendance_percentage", y="final_grade",
            color_discrete_sequence=["#10B981"], opacity=0.4,
            title="Attendance vs final grade",
            labels={"attendance_percentage": "Attendance (%)", "final_grade": "Final grade"},
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        add_binned_overlay(fig, view, "attendance_percentage", "final_grade")
        render_chart(style_fig(fig, height=380, legend_below=True))

    row2 = st.columns(2)
    with row2[0]:
        sample = view.sample(min(len(view), 3500), random_state=22) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="exercise_minutes", y="final_grade",
            color_discrete_sequence=["#10B981"], opacity=0.4,
            title="Daily exercise vs final grade",
            labels={"exercise_minutes": "Exercise (min/day)", "final_grade": "Final grade"},
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        add_binned_overlay(fig, view, "exercise_minutes", "final_grade")
        render_chart(style_fig(fig, height=380, legend_below=True))

    with row2[1]:
        line_df = (
            view.groupby(["assignments_completed", "exercise_group"], observed=True)
            .agg(mean=("final_grade", "mean"), n=("final_grade", "count"))
            .reset_index()
        )
        line_df = line_df[line_df["n"] >= 8]
        fig = px.line(
            line_df, x="assignments_completed", y="mean",
            color="exercise_group", markers=True,
            color_discrete_map=EXERCISE_COLORS,
            category_orders={"exercise_group": EXERCISE_ORDER},
            title="Assignments vs grade split by exercise group",
            labels={"assignments_completed": "Assignments completed",
                    "mean": "Mean final grade",
                    "exercise_group": "Exercise tertile"},
        )
        fig.update_traces(line=dict(width=2.8), marker=dict(size=7))
        render_chart(style_fig(fig, height=380, legend_below=True))

    section_head(
        "Three-way interactions",
        "Where two effort dimensions combine, and how stress reshapes the study payoff.",
    )
    row3 = st.columns(2)
    with row3[0]:
        att_band = pd.cut(view["attendance_percentage"],
                          bins=[0, 60, 70, 80, 90, 101],
                          labels=["<60", "60-70", "70-80", "80-90", "90+"])
        asg_band = pd.cut(view["assignments_completed"],
                          bins=[-0.1, 3, 6, 9, 12],
                          labels=["0-3", "4-6", "7-9", "10-12"])
        pivot = (
            view.assign(att=att_band, asg=asg_band)
            .pivot_table(index="asg", columns="att",
                         values="final_grade", aggfunc="mean", observed=True)
        )
        fig = heatmap_mean_grade(
            pivot,
            title="Grade by attendance x assignments band",
            x_title="Attendance band",
            y_title="Assignments band",
            hover_template="Attendance: %{x}<br>Assignments: %{y}<br>Mean grade: %{z:.1f}<extra></extra>",
        )
        render_chart(style_fig(fig, height=400, legend_below=False))

    with row3[1]:
        sample = view.sample(min(len(view), 3500), random_state=23) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="study_hours_per_day", y="final_grade",
            color="stress_zone", size="sleep_hours", size_max=14,
            color_discrete_map=STRESS_COLORS,
            category_orders={"stress_zone": STRESS_ORDER},
            opacity=0.55,
            title="Study hours vs grade, colored by stress, sized by sleep",
            labels={"study_hours_per_day": "Daily study hours",
                    "final_grade": "Final grade",
                    "stress_zone": "Stress zone",
                    "sleep_hours": "Sleep (h)"},
            hover_data={"sleep_hours": ":.1f"},
        )
        fig.update_traces(marker=dict(line=dict(width=0)))
        render_chart(style_fig(fig, height=400, legend_below=True))


# ---------------------------------------------------------------------------
# Page 4 - Wellness and Cognition
# ---------------------------------------------------------------------------


def page_wellness(view: pd.DataFrame, df: pd.DataFrame) -> None:
    page_intro(
        "Wellness and cognition",
        "How stress, sleep, focus and coffee interact with grades",
        "This page focuses on cognitive output. It pairs the obvious variables (stress and sleep) with "
        "the headline cognitive metrics (focus and productivity) and tests whether coffee carries weight.",
    )

    if view.empty:
        st.warning("No students match the current filters.")
        return

    top10 = view[view["final_grade"] >= view["final_grade"].quantile(0.9)]
    cohort = view

    section_head(
        "Cognitive and wellness scorecards",
        "Top 10% students by final grade typically show higher focus and lower stress than the average cohort.",
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        coh = cohort["focus_score"].mean()
        top = top10["focus_score"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct > 0 else "kpi-down"
        kpi("Focus score (top 10% grade)", f"{top:.1f}",
            f"Cohort <b>{coh:.1f}</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort",
            info="Self-reported attention score for the top decile of final grades vs the cohort average.")
    with k2:
        coh = cohort["stress_level"].mean()
        top = top10["stress_level"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct < 0 else "kpi-down"
        kpi("Stress level (top 10% grade)", f"{top:.1f}",
            f"Cohort <b>{coh:.1f}</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort")
    with k3:
        coh = cohort["sleep_hours"].mean()
        top = top10["sleep_hours"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct > 0 else "kpi-down"
        kpi("Sleep hours (top 10% grade)", f"{top:.1f}h",
            f"Cohort <b>{coh:.1f}h</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort")
    with k4:
        coh = cohort["coffee_intake_mg"].mean()
        top = top10["coffee_intake_mg"].mean()
        pct = (top - coh) / coh * 100 if coh else 0
        cls = "kpi-up" if pct < 0 else "kpi-down"
        kpi("Coffee (top 10% grade)", f"{top:.0f} mg",
            f"Cohort <b>{coh:.0f} mg</b><br><span class='{cls}'>{pct:+.1f}%</span> vs cohort")

    coffee_focus_corr = view[["coffee_intake_mg", "focus_score"]].corr().iloc[0, 1]
    stress_focus_corr = view[["stress_level", "focus_score"]].corr().iloc[0, 1]
    i1, i2 = st.columns(2)
    with i1:
        insight(
            "Calm minds, sharper focus",
            f"Stress correlates with focus at r = {stress_focus_corr:+.2f}. Students in the Calm zone "
            "consistently sit at the highest end of the focus distribution.",
        )
    with i2:
        relation = "rises with" if coffee_focus_corr > 0 else "drops with"
        insight(
            "Coffee and focus",
            f"In this cohort, focus {relation} caffeine (r = {coffee_focus_corr:+.2f}). "
            "The scatter below makes the shape of the relationship explicit.",
        )

    section_head(
        "Two-way views",
        "Direct comparisons before any third dimension is layered on.",
    )
    row1 = st.columns(2)
    with row1[0]:
        sample = view.sample(min(len(view), 3500), random_state=31) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="coffee_intake_mg", y="focus_score",
            color_discrete_sequence=["#8B5CF6"], opacity=0.45,
            title="Coffee intake vs focus score",
            labels={"coffee_intake_mg": "Coffee intake (mg/day)",
                    "focus_score": "Focus score"},
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)),
                          selector=dict(mode="markers"))
        add_binned_overlay(fig, view, "coffee_intake_mg", "focus_score", name="Binned mean")
        render_chart(style_fig(fig, height=380, legend_below=True))

    with row1[1]:
        stress_grouped = (
            view.groupby("stress_level", observed=True)
            .agg(focus=("focus_score", "mean"),
                 productivity=("productivity_score", "mean"),
                 grade=("final_grade", "mean"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stress_grouped["stress_level"], y=stress_grouped["focus"],
            mode="lines+markers", name="Focus",
            line=dict(color="#8B5CF6", width=3),
            marker=dict(size=8, color="#8B5CF6"),
        ))
        fig.add_trace(go.Scatter(
            x=stress_grouped["stress_level"], y=stress_grouped["productivity"],
            mode="lines+markers", name="Productivity",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=8, color="#F59E0B"),
        ))
        fig.update_layout(
            title="Focus and productivity by stress level",
            xaxis_title="Stress level (1-10)", yaxis_title="Average score",
        )
        render_chart(style_fig(fig, height=380, legend_below=True))

    row2 = st.columns(2)
    with row2[0]:
        fig = px.violin(
            view, x="sleep_quality", y="final_grade",
            color="sleep_quality",
            category_orders={"sleep_quality": SLEEP_ORDER},
            color_discrete_sequence=["#EF4444", "#2563EB", "#10B981"],
            box=True, points=False,
            title="Final grade by sleep quality",
            labels={"sleep_quality": "Sleep quality",
                    "final_grade": "Final grade"},
        )
        fig.update_layout(showlegend=False)
        render_chart(style_fig(fig, height=380, legend_below=False))

    with row2[1]:
        sample = view.sample(min(len(view), 3500), random_state=33) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="focus_score", y="final_grade",
            color_discrete_sequence=["#2563EB"], opacity=0.4,
            title="Focus score vs final grade",
            labels={"focus_score": "Focus score", "final_grade": "Final grade"},
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)),
                          selector=dict(mode="markers"))
        add_binned_overlay(fig, view, "focus_score", "final_grade", name="Binned mean")
        render_chart(style_fig(fig, height=380, legend_below=True))

    section_head(
        "Three-way interactions",
        "Two cognitive axes plus a third dimension (final grade or hierarchy) to reveal interaction effects.",
    )
    row3 = st.columns(2)
    with row3[0]:
        focus_bins = pd.cut(view["focus_score"],
                            bins=[-0.1, 10, 20, 30, 45, 80],
                            labels=["0-10", "10-20", "20-30", "30-45", "45+"])
        stress_bins = pd.cut(view["stress_level"],
                             bins=[0, 3, 5, 7, 10],
                             labels=["1-3", "4-5", "6-7", "8-10"],
                             include_lowest=True)
        pivot = (
            view.assign(fb=focus_bins, sb=stress_bins)
            .pivot_table(index="fb", columns="sb",
                         values="final_grade", aggfunc="mean", observed=True)
        )
        fig = heatmap_mean_grade(
            pivot,
            title="Final grade by focus x stress",
            x_title="Stress band",
            y_title="Focus band",
            hover_template="Stress: %{x}<br>Focus: %{y}<br>Mean grade: %{z:.1f}<extra></extra>",
        )
        render_chart(style_fig(fig, height=440, legend_below=False))

    with row3[1]:
        sample = view.sample(min(len(view), 3500), random_state=34) if len(view) > 3500 else view.copy()
        fig = px.scatter(
            sample, x="focus_score", y="productivity_score",
            color="final_grade",
            color_continuous_scale=[
                [0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"],
            ],
            opacity=0.7,
            title="Focus vs productivity, colored by final grade",
            labels={"focus_score": "Focus score",
                    "productivity_score": "Productivity score",
                    "final_grade": "Final grade"},
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        render_chart(style_fig(fig, height=420, legend_below=False))

    section_head(
        "Behavior hierarchy",
        "Three nested rings: performance segment → stress zone → sleep quality. "
        "Slice area is student count; colour is mean final grade for that slice.<br><br>"
        "<b>Inner ring — segment</b> (final grade bands): "
        "Critical &lt;60 · At Risk 60–70 · Stable 70–80 · Strong 80–90 · Elite ≥90.<br>"
        "<b>Middle ring — stress zone</b> (self-report 1–10): "
        "Calm 1–3 · Managed 4–6 · High 7–8 · Burnout 9–10.<br>"
        "<b>Outer ring — sleep quality</b> (hours/night): "
        "Short &lt;6 · Balanced 6–8 · Long ≥8.",
    )

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    colors: list[float] = []
    total = len(view)

    lvl1 = (
        view.groupby("performance_segment", observed=True)
        .agg(n=("row_id", "count"), avg=("final_grade", "mean"))
        .reset_index()
    )
    lvl1 = lvl1[lvl1["n"] > 0]
    for _, r in lvl1.iterrows():
        seg = str(r["performance_segment"])
        ids.append(seg)
        labels.append(seg)
        parents.append("")
        values.append(int(r["n"])); colors.append(float(r["avg"]))

    lvl2 = (
        view.groupby(["performance_segment", "stress_zone"], observed=True)
        .agg(n=("row_id", "count"), avg=("final_grade", "mean"))
        .reset_index()
    )
    lvl2 = lvl2[lvl2["n"] > 0]
    for _, r in lvl2.iterrows():
        seg = str(r["performance_segment"]); sz = str(r["stress_zone"])
        nid = f"{seg}/{sz}"
        ids.append(nid)
        labels.append(sz)
        parents.append(seg)
        values.append(int(r["n"])); colors.append(float(r["avg"]))

    lvl3 = (
        view.groupby(
            ["performance_segment", "stress_zone", "sleep_quality"], observed=True,
        )
        .agg(n=("row_id", "count"), avg=("final_grade", "mean"))
        .reset_index()
    )
    lvl3 = lvl3[lvl3["n"] > 0]
    for _, r in lvl3.iterrows():
        seg = str(r["performance_segment"])
        sz = str(r["stress_zone"]); sl = str(r["sleep_quality"])
        nid = f"{seg}/{sz}/{sl}"
        ids.append(nid)
        labels.append(sl)
        parents.append(f"{seg}/{sz}")
        values.append(int(r["n"])); colors.append(float(r["avg"]))

    customdata = np.array([
        [grade, n, (n / total * 100 if total else 0.0)]
        for grade, n in zip(colors, values)
    ])
    grade_min = float(view["final_grade"].min())
    grade_max = float(view["final_grade"].max())

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            customdata=customdata,
            marker=dict(
                colors=colors,
                colorscale=[[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"]],
                cmin=grade_min,
                cmax=grade_max,
                showscale=True,
                colorbar=dict(title=dict(text="Mean grade")),
                line=dict(width=1.2, color="#FFFFFF"),
            ),
            insidetextorientation="auto",
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.1f}",
            textfont=dict(size=12, color="#0f172a"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Students: %{customdata[1]:,} (%{customdata[2]:.1f}%)<br>"
                "Mean grade: %{customdata[0]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Segment -> stress -> sleep hierarchy",
        margin=dict(l=10, r=10, t=72, b=10),
    )
    render_chart(style_fig(fig, height=820, legend_below=False))


# ---------------------------------------------------------------------------
# Page 5 - About
# ---------------------------------------------------------------------------


def page_about(df: pd.DataFrame, miss_df: pd.DataFrame, raw_rows: int) -> None:
    page_intro(
        "About",
        "Methodology, cleaning steps and data quality",
        "How the raw Kaggle file becomes the cohort behind every chart in this dashboard.",
    )

    a, b = st.columns([1.25, 0.75])
    with a:
        st.markdown(
            """
            **Source.** Student Productivity and Behavior Dataset (20K), Kaggle, Apache 2.0. The data is
            synthetically generated, so the dashboard speaks in correlation rather than causation.

            **Cleaning steps (mirroring `analysis.ipynb`).**
            - `coffee_intake_mg` parsed by stripping the trailing `mg` and converting to numeric.
            - `attendance_percentage` parsed by stripping the trailing `%` and converting to numeric.
            - Gender normalized: `M` -> `Male`, `F` -> `Female`, anything else kept as-is.
            - Missing values in `phone_usage_hours`, `coffee_intake_mg`, `exercise_minutes` and
              `final_grade` filled with the column mean (notebook cell 9).

            **Derived features.**
            - `screen_time_hours` = social media + YouTube + gaming.
            - `distraction_ratio` = screen time divided by `study_hours + 0.5`.
            - `performance_segment` (Critical / At Risk / Stable / Strong / Elite) from final grade bands.
            - `digital_load`, `sleep_quality`, `study_band`, `stress_zone` from threshold-cut bins.
            - `exercise_group` from `pd.qcut(exercise_minutes, q=3)`.

            **Visual logic.**
            - One filter popover at the top keeps the cohort consistent across pages.
            - Top-decile cards quantify "what does success look like?" on every themed page.
            - Two-way charts show direct relationships, three-way charts show interactions.
            - Categorical color is reserved for semantic groups; sequential color only on metric pivots.
            """
        )
    with b:
        kpi("Raw rows", f"{raw_rows:,}", "Before cleaning")
        kpi("Clean rows", f"{len(df):,}",
            f"{len(df) / raw_rows * 100:.2f}% retained")
        kpi("Numeric variables", f"{len(df.select_dtypes('number').columns)}",
            "Used in charts and correlations")

    section_head("Missing values in the source file")
    if miss_df.empty:
        st.caption("No missing values detected in the raw file.")
    else:
        st.dataframe(miss_df, width="stretch", hide_index=True)

    section_head("Sample of cleaned and enriched data")
    sample_cols = [
        "gender", "age", "performance_segment", "digital_load",
        "sleep_quality", "study_band", "stress_zone", "exercise_group",
        "study_hours_per_day", "screen_time_hours", "sleep_hours",
        "stress_level", "focus_score", "final_grade",
    ]
    st.dataframe(df[sample_cols].head(100), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SECTION_RENDERERS = {
    "overview": "page_overview",
    "digital": "page_digital",
    "effort": "page_effort",
    "wellness": "page_wellness",
    "about": "page_about",
}


def render_section(key: str, view: pd.DataFrame, df: pd.DataFrame,
                   miss_df: pd.DataFrame, raw_rows: int) -> None:
    """Wrap each page in a scroll-spy anchor div."""
    st.markdown(
        f'<div id="section-{key}" data-section="{key}" style="scroll-margin-top: 16px;">',
        unsafe_allow_html=True,
    )
    set_page_accent(key)
    if key == "overview":
        page_overview(view, df)
    elif key == "digital":
        page_digital(view, df)
    elif key == "effort":
        page_effort(view, df)
    elif key == "wellness":
        page_wellness(view, df)
    elif key == "about":
        page_about(df, miss_df, raw_rows)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_theme()
    df, miss_df, raw_rows = load_data()

    init_filters(df)
    view = apply_filters(df)

    render_sidebar(df, view)

    # Render every section in one long scrollable page
    for i, (key, _) in enumerate(PAGES):
        render_section(key, view, df, miss_df, raw_rows)
        if i < len(PAGES) - 1:
            st.markdown(
                "<hr style='margin:2.2rem 0 1.6rem 0; border:none;"
                " border-top:1px dashed #E4E4E7;'>",
                unsafe_allow_html=True,
            )

    inject_nav_scroll_spy()


if __name__ == "__main__":
    main()
