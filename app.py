"""Student Productivity Dashboard.

A multi-page Streamlit dashboard exploring how digital habits, academic effort
and wellness behaviours relate to final grades across 20K student records.
Data pipeline mirrors the cleaning steps performed in analysis.ipynb.
"""

from __future__ import annotations

from pathlib import Path

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

INK = "#0A0A0A"
MUTED = "#71717A"
GRID = "#ECECEC"
CARD = "#FFFFFF"


st.set_page_config(
    page_title="Student Productivity Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
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

            .stApp { background: var(--paper); color: var(--ink); }

            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebarCollapsedControl"] { display: none; }
            section[data-testid="stSidebar"] { display: none; }
            footer, #MainMenu { visibility: hidden; }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                max-width: 1480px;
            }

            h1 { letter-spacing: -0.045em; }
            h2 { letter-spacing: -0.030em; }
            h3 { letter-spacing: -0.020em; }

            .brand-row {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                padding: 0.55rem 0.2rem;
            }
            .brand-dot {
                width: 9px; height: 9px;
                border-radius: 999px;
                background: var(--ink);
                display: inline-block;
            }
            .brand-title {
                color: var(--ink);
                font-weight: 700;
                font-size: 1.02rem;
                letter-spacing: -0.020em;
            }
            .brand-sep { color: var(--soft); margin: 0 0.15rem; }
            .brand-meta { color: var(--muted); font-size: 0.86rem; }

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

            /* nav pills via Streamlit buttons */
            .nav-wrap div[data-testid="column"] .stButton button {
                background: transparent;
                color: var(--muted);
                border: 1px solid transparent;
                border-radius: 10px;
                padding: 0.45rem 0.7rem;
                font-weight: 550;
                font-size: 0.95rem;
                box-shadow: none;
                transition: all 0.14s ease;
            }
            .nav-wrap div[data-testid="column"] .stButton button:hover {
                background: #F4F4F5;
                color: var(--ink);
            }
            .nav-wrap div[data-testid="column"] .stButton button[kind="primary"] {
                background: #18181B !important;
                color: #FFFFFF !important;
                border-color: #18181B !important;
            }

            /* Hide all raw icon text fallbacks inside popover buttons (tune, expand_more). */
            [data-testid="stPopover"] button [data-testid*="stIconMaterial"],
            [data-testid="stPopover"] button [data-testid*="Icon"],
            [data-testid="stPopover"] button .material-symbols-outlined,
            [data-testid="stPopover"] button span[class*="material"],
            [data-testid="stPopover"] button > div > div > span:not(:only-child):not(.label-text) {
                display: none !important;
            }
            [data-testid="stPopover"] button {
                font-weight: 600;
                border: 1px solid var(--grid) !important;
                background: var(--card) !important;
                color: var(--ink) !important;
                border-radius: 12px !important;
                padding: 0.55rem 1rem !important;
                box-shadow: 0 1px 0 rgba(17, 24, 39, 0.02) !important;
            }
            [data-testid="stPopover"] button:hover {
                border-color: #D4D4D8 !important;
                background: #FAFAFA !important;
            }
            [data-testid="stPopover"] button p {
                font-weight: 600;
                margin: 0;
                color: var(--ink);
            }
            /* Brand bar */
            .brand-card {
                display: flex;
                align-items: center;
                gap: 0.85rem;
                padding: 0.55rem 0.2rem;
            }
            .brand-mark {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 30px; height: 30px;
                border-radius: 9px;
                background: linear-gradient(135deg, #18181B 0%, #3F3F46 100%);
                color: #FFFFFF;
                font-weight: 800;
                font-size: 0.86rem;
                letter-spacing: -0.02em;
            }
            .brand-stack { display: flex; flex-direction: column; gap: 2px; }
            .brand-name {
                color: var(--ink);
                font-size: 1.04rem;
                font-weight: 750;
                letter-spacing: -0.022em;
                line-height: 1.15;
            }
            .brand-sub {
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.2;
            }
            .pill-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.28rem 0.55rem;
                border-radius: 999px;
                background: #F4F4F5;
                color: #18181B;
                font-size: 0.74rem;
                font-weight: 650;
                letter-spacing: 0.01em;
                border: 1px solid var(--grid);
            }
            .pill-dot {
                width: 6px; height: 6px;
                border-radius: 999px;
                background: #10B981;
            }
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
                padding: 1.05rem 1.15rem;
                background: var(--card);
                border: 1px solid var(--grid);
                border-radius: 14px;
                min-height: 130px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .kpi-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 650;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }
            .kpi-value {
                color: var(--ink);
                font-size: 2.1rem;
                font-weight: 750;
                letter-spacing: -0.04em;
                margin: 0.35rem 0 0.2rem 0;
                line-height: 1.05;
            }
            .kpi-note { color: var(--muted); font-size: 0.82rem; line-height: 1.4; }
            .kpi-up   { color: #10B981; font-weight: 650; }
            .kpi-down { color: #EF4444; font-weight: 650; }

            .insight {
                padding: 0.95rem 1.05rem;
                border: 1px solid var(--grid);
                border-radius: 14px;
                background: var(--card);
            }
            .insight .badge {
                display: inline-block;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                background: #F4F4F5;
                color: #18181B;
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .insight .text {
                margin-top: 0.5rem;
                color: var(--ink);
                font-size: 0.96rem;
                line-height: 1.5;
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


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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
    with st.popover(label, use_container_width=True):
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
        if st.button("Reset filters", use_container_width=True):
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
    with st.popover(label, use_container_width=True):
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


def render_header(df: pd.DataFrame, view: pd.DataFrame) -> None:
    page_label = dict(PAGES).get(st.session_state.get("page", "overview"), "Overview")
    share_pct = (len(view) / len(df) * 100) if len(df) else 0
    h_l, h_r = st.columns([6.2, 3.8], gap="medium")
    with h_l:
        st.markdown(
            f"""
            <div class="brand-card">
                <span class="brand-mark">SP</span>
                <div class="brand-stack">
                    <span class="brand-name">Student Productivity Dashboard</span>
                    <span class="brand-sub">Page: <b>{page_label}</b> &middot; Data: Kaggle 20K synthetic cohort</span>
                </div>
                <span class="pill-badge"><span class="pill-dot"></span>
                    {len(view):,} / {len(df):,} students &middot; {share_pct:.1f}% of cohort
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h_r:
        c1, c2 = st.columns(2)
        with c1:
            render_filter_popover(df)
        with c2:
            render_glossary_popover()


def render_nav() -> None:
    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    cols = st.columns(len(PAGES) + 1)
    for i, (key, label) in enumerate(PAGES):
        with cols[i]:
            is_active = st.session_state.get("page", "overview") == key
            clicked = st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            if clicked:
                st.session_state.page = key
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


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
        "Cohort distributions",
        "How four headline variables are spread across the selected students. Dashed lines mark the cohort averages.",
    )

    dist_cols = [
        ("final_grade", "Final grade", "#2563EB"),
        ("focus_score", "Focus score", "#8B5CF6"),
        ("study_hours_per_day", "Daily study hours", "#10B981"),
        ("sleep_hours", "Sleep hours", "#F59E0B"),
    ]
    row1 = st.columns(2)
    row2 = st.columns(2)
    cells = [row1[0], row1[1], row2[0], row2[1]]
    for cell, (col, title, color) in zip(cells, dist_cols):
        with cell:
            fig = px.histogram(
                view, x=col, nbins=36,
                title=f"Distribution of {title.lower()}",
                color_discrete_sequence=[color],
                labels={col: friendly(col)},
            )
            fig.update_traces(marker_line_width=0, opacity=0.88)
            mean = view[col].mean()
            fig.add_vline(
                x=mean, line_dash="dot", line_color="#0A0A0A",
                annotation_text=f"Avg {mean:.1f}",
                annotation_position="top right",
                annotation_font=dict(size=11, color="#0A0A0A"),
            )
            fig.update_yaxes(title="Students")
            fig.update_layout(showlegend=False)
            render_chart(style_fig(fig, height=320, legend_below=False))

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
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns.astype(str)),
                y=list(pivot.index.astype(str)),
                colorscale=[[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"]],
                colorbar=dict(title="Mean final grade"),
                hovertemplate="Social media: %{x}h<br>Gaming: %{y}h<br>Mean grade: %{z:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Final grade by social media x gaming",
            xaxis_title="Social media (h/day)", yaxis_title="Gaming (h/day)",
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
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns.astype(str)),
                y=list(pivot.index.astype(str)),
                colorscale=[[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"]],
                colorbar=dict(title="Mean final grade"),
                hovertemplate="Attendance: %{x}<br>Assignments: %{y}<br>Mean grade: %{z:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Grade by attendance x assignments band",
            xaxis_title="Attendance band", yaxis_title="Assignments band",
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
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns.astype(str)),
                y=list(pivot.index.astype(str)),
                colorscale=[[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"]],
                colorbar=dict(title="Mean final grade"),
                hovertemplate="Stress: %{x}<br>Focus: %{y}<br>Mean grade: %{z:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Final grade by focus x stress",
            xaxis_title="Stress band", yaxis_title="Focus band",
        )
        render_chart(style_fig(fig, height=420, legend_below=False))

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
        "Performance segment broken down by stress zone, then sleep quality. Color encodes mean final grade.",
    )
    sun = (
        view.groupby(
            ["performance_segment", "stress_zone", "sleep_quality"],
            observed=True,
        )
        .agg(students=("row_id", "count"), avg_final=("final_grade", "mean"))
        .reset_index()
    )
    sun = sun[sun["students"] > 0]
    fig = px.sunburst(
        sun,
        path=["performance_segment", "stress_zone", "sleep_quality"],
        values="students",
        color="avg_final",
        color_continuous_scale=[
            [0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#10B981"],
        ],
        labels={"avg_final": "Mean final grade"},
        title="Segment -> stress -> sleep hierarchy",
    )
    fig.update_traces(branchvalues="total", insidetextorientation="radial")
    render_chart(style_fig(fig, height=520, legend_below=False))


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
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    section_head("Sample of cleaned and enriched data")
    sample_cols = [
        "gender", "age", "performance_segment", "digital_load",
        "sleep_quality", "study_band", "stress_zone", "exercise_group",
        "study_hours_per_day", "screen_time_hours", "sleep_hours",
        "stress_level", "focus_score", "final_grade",
    ]
    st.dataframe(df[sample_cols].head(100), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    inject_theme()
    df, miss_df, raw_rows = load_data()

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    init_filters(df)
    view = apply_filters(df)

    set_page_accent(st.session_state.page)

    render_header(df, view)
    render_nav()

    page = st.session_state.page
    if page == "overview":
        page_overview(view, df)
    elif page == "digital":
        page_digital(view, df)
    elif page == "effort":
        page_effort(view, df)
    elif page == "wellness":
        page_wellness(view, df)
    elif page == "about":
        page_about(df, miss_df, raw_rows)


if __name__ == "__main__":
    main()
