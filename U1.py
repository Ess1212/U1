
# =============================================================================
# ENERGY POWER DASHBOARD
# PART 1 — FINAL FOUNDATION (BLUE + WHITE)
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Tuple
from streamlit.components.v1 import html

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Energy Power Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONSTANTS
# =============================================================================

APP_TITLE = "Power Dispatch Dashboard"
COMPANY_NAME = "SchneiTech Group"

SWG_LIST = ["SWG1", "SWG2", "SWG3"]

ACTIVE_MIN, ACTIVE_MAX = -150.0, 150.0
REACTIVE_MIN, REACTIVE_MAX = -150.0, 150.0
SOC_MIN, SOC_MAX = 0.0, 100.0

LOCAL_TZ = ZoneInfo("Asia/Phnom_Penh")
DISPLAY_DT_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# =============================================================================
# REMOVE STREAMLIT DEFAULT UI
# =============================================================================

st.markdown(
    """
<style>
header[data-testid="stHeader"] {display:none;}
div[data-testid="stToolbar"] {display:none;}
#MainMenu {display:none;}
section.main > div {padding-top:0rem;}
</style>
""",
    unsafe_allow_html=True
)

# =============================================================================
# BLUE & WHITE DASHBOARD CSS
# =============================================================================

st.markdown(
    """
<style>

/* ===== BASE ===== */
html, body {
    background-color: #ffffff;
    font-family: "Segoe UI", Inter, sans-serif;
    color: #0f172a;
}

/* ===== HEADER ===== */
.dashboard-header {
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
    color: white;
    padding: 28px 32px;
    border-radius: 18px;
    font-size: 42px;
    font-weight: 900;
    box-shadow: 0 14px 36px rgba(0,0,0,0.22);
}

/* ===== CARD ===== */
.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 26px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.1);
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 16px;
}

/* ===== SWG TITLE ===== */
.swg-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* ===== BUTTON ===== */
button[kind="primary"] {
    background-color: #2563eb;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 700;
    height: 46px;
}

/* ===== TABLE ===== */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    font-size: 14px;
}

</style>
""",
    unsafe_allow_html=True
)

# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f'<div class="dashboard-header">{APP_TITLE}</div>',
    unsafe_allow_html=True
)

# =============================================================================
# SUBTITLE BAR WITH ICONS + LIVE CLOCK (MATCH HEADER)
# =============================================================================

html(
    f"""
<div style="
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
    color: white;
    border-radius: 14px;
    padding: 12px 22px;
    margin-top: 10px;
    margin-bottom: 26px;
    font-size: 15px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: flex-start;   /* 👈 LEFT ALIGN */
    gap: 14px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
">
    <span>🏢 <strong>{COMPANY_NAME}</strong></span>
    <span>|</span>
    <span style="color:#22c55e;">🟢</span>
    <span id="date-text"></span>
    <span>|</span>
    <span>🕒 <span id="time-text"></span></span>
</div>

<script>
function updateClockBar() {{
    const now = new Date();

    const dateOptions = {{
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    }};

    const timeOptions = {{
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    }};

    document.getElementById("date-text").innerHTML =
        now.toLocaleDateString([], dateOptions);

    document.getElementById("time-text").innerHTML =
        now.toLocaleTimeString([], timeOptions);
}}

setInterval(updateClockBar, 1000);
updateClockBar();
</script>
""",
    height=56
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_state():
    for swg in SWG_LIST:
        if f"{swg}_data" not in st.session_state:
            st.session_state[f"{swg}_data"] = pd.DataFrame(
                columns=[
                    f"{swg}_DateTime",
                    "Active Power (MW)",
                    "Reactive Power (Mvar)",
                    "SOC (%)"
                ]
            )
    if "history" not in st.session_state:
        st.session_state.history = []
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []

init_state()


# =============================================================================
# UNDO / REDO CORE (BULLETPROOF)
# =============================================================================

def snapshot_state():
    """
    Save current state BEFORE any user action that modifies data.
    This invalidates the redo stack.
    """
    st.session_state.history.append({
        swg: st.session_state[f"{swg}_data"].copy(deep=True)
        for swg in SWG_LIST
    })
    st.session_state.redo_stack.clear()


def restore_state(snapshot: Dict[str, pd.DataFrame]):
    """Restore a previously saved snapshot"""
    for swg in SWG_LIST:
        st.session_state[f"{swg}_data"] = snapshot[swg].copy(deep=True)


def undo_action() -> bool:
    """
    Perform undo safely.
    Returns True if undo succeeded.
    """
    if len(st.session_state.history) == 0:
        return False

    # Save current state for redo
    st.session_state.redo_stack.append({
        swg: st.session_state[f"{swg}_data"].copy(deep=True)
        for swg in SWG_LIST
    })

    prev_snapshot = st.session_state.history.pop()
    restore_state(prev_snapshot)
    return True


def redo_action() -> bool:
    """
    Perform redo safely.
    Returns True if redo succeeded.
    """
    if len(st.session_state.redo_stack) == 0:
        return False

    # Save current state for undo
    st.session_state.history.append({
        swg: st.session_state[f"{swg}_data"].copy(deep=True)
        for swg in SWG_LIST
    })

    next_snapshot = st.session_state.redo_stack.pop()
    restore_state(next_snapshot)
    return True

# =============================================================================
# UNDO / REDO UI (SAFE BUTTONS)
# =============================================================================

uc, rc = st.columns(2)

with uc:
    if st.button(
        "↩ Undo",
        use_container_width=True,
        disabled=len(st.session_state.history) == 0
    ):
        if undo_action():
            st.success("Undo applied")

with rc:
    if st.button(
        "↪ Redo",
        use_container_width=True,
        disabled=len(st.session_state.redo_stack) == 0
    ):
        if redo_action():
            st.success("Redo applied")


# =============================================================================
# INPUT DATA
# =============================================================================

st.markdown(
    """
<div class="card">
    <div class="section-title">📥 Input Data</div>
    <p style="color:#6b7280;font-size:13px;max-width:900px;">
        Enter real-time operational data for each SWG including
        Active Power, Reactive Power, and State of Charge (SOC).
        All inputs are validated against system limits and every
        successful entry is tracked with full Undo / Redo support.
    </p>
</div>
""",
    unsafe_allow_html=True
)


cols = st.columns(3)

def validate_inputs(p, q, s) -> Tuple[bool, str]:
    if p is None or q is None or s is None:
        return False, "All fields are required"
    if not ACTIVE_MIN <= p <= ACTIVE_MAX:
        return False, "Active Power out of range"
    if not REACTIVE_MIN <= q <= REACTIVE_MAX:
        return False, "Reactive Power out of range"
    if not SOC_MIN <= s <= SOC_MAX:
        return False, "SOC out of range"
    return True, ""

def insert_row(swg, p, q, s):
    snapshot_state()
    st.session_state[f"{swg}_data"] = pd.concat(
        [
            st.session_state[f"{swg}_data"],
            pd.DataFrame([{
                f"{swg}_DateTime": datetime.now(tz=LOCAL_TZ),
                "Active Power (MW)": p,
                "Reactive Power (Mvar)": q,
                "SOC (%)": s,
            }])
        ],
        ignore_index=True
    )

for i, swg in enumerate(SWG_LIST):
    with cols[i]:
        st.markdown(
            f'<div class="swg-title">{swg.replace("SWG","SWG-")}</div>',
            unsafe_allow_html=True
        )

        p = st.number_input("⚡ Active Power (MW)", value=None, step=1.0, key=f"{swg}_p")
        q = st.number_input("🔌 Reactive Power (Mvar)", value=None, step=1.0, key=f"{swg}_q")
        s = st.number_input("🔋 SOC (%)", value=None, step=1.0, key=f"{swg}_s")

        if st.button(f"➕ Add {swg.replace('SWG','SWG-')}", key=f"add_{swg}", use_container_width=True):
            ok, msg = validate_inputs(p, q, s)
            if not ok:
                st.error(msg)
            else:
                insert_row(swg, p, q, s)
                st.success("Data added")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TABLE PREVIEW
# =============================================================================

st.markdown(
    """
<div class="card">
    <div class="section-title">📊 Live Table Preview</div>
    <p style="color:#6b7280;font-size:13px;max-width:900px;">
        Live operational measurements collected from each SWG unit.
        This table updates in real time as data is added or edited and
        represents the current state of the system. All modifications
        are tracked and fully support Undo / Redo operations.
    </p>
</div>
""",
    unsafe_allow_html=True
)


def format_preview(df):
    if df.empty:
        return df
    df = df.copy()
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).dt.tz_localize(None).dt.strftime(DISPLAY_DT_FORMAT)
    return df

tcols = st.columns(3)
for i, swg in enumerate(SWG_LIST):
    with tcols[i]:
        st.dataframe(
            format_preview(st.session_state[f"{swg}_data"]),
            use_container_width=True,
            hide_index=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 1
# =============================================================================

# =============================================================================
# PART 2 — EDIT TABLE DATA (LIVE PREVIEW, SMOOTH, UNDO/REDO SAFE)
# =============================================================================

st.markdown(
    """
<div class="card">
    <div class="section-title">✏️ Edit Table Data</div>
    <p style="color:#6b7280;font-size:13px;max-width:900px;">
        Modify existing SWG records using structured edit modes such as
        cell editing, row and column operations, and data restructuring.
        All changes are applied in real time with full Undo / Redo support.
        Tables can be locked to prevent accidental modifications.
    </p>
</div>
""",
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# STATE INIT
# -----------------------------------------------------------------------------

if "edit_session_active" not in st.session_state:
    st.session_state.edit_session_active = False

if "edit_buffer" not in st.session_state:
    st.session_state.edit_buffer = {}

# -----------------------------------------------------------------------------
# EDIT MODE + LOCK
# -----------------------------------------------------------------------------

edit_mode = st.selectbox(
    "Edit Mode",
    [
        "Off",
        "Edit Cells",
        "Insert Row",
        "Delete Row",
        "Insert Column",
        "Delete Column",
        "Rename Column",
        "Move Column Left",
        "Move Column Right",
        "Merge Columns",
    ],
    index=0
)

st.session_state.table_locked = st.toggle(
    "🔒 Lock Table (lock all SWG tables)",
    value=st.session_state.get("table_locked", False)
)

if st.session_state.table_locked:
    st.info("Tables are locked (read-only).")
elif edit_mode != "Off":
    st.success(f"Edit mode: {edit_mode}")

# -----------------------------------------------------------------------------
# SELECT SWG
# -----------------------------------------------------------------------------

swg_target = st.selectbox(
    "Select SWG",
    SWG_LIST,
    format_func=lambda x: x.replace("SWG", "SWG-")
)

df_key = f"{swg_target}_data"
current_df = st.session_state[df_key]

# -----------------------------------------------------------------------------
# LIVE TABLE PREVIEW (ALWAYS VISIBLE)
# -----------------------------------------------------------------------------

st.markdown(
    """
<div class="card">
    <div class="section-title">📊 Edit Live Table Preview</div>
    <p style="color:#6b7280;font-size:13px;max-width:900px;">
        View and modify SWG data in real time. Any edits made here are immediately
        reflected in the system while remaining fully protected by Undo / Redo controls.
        Use this preview to verify, adjust, and validate operational data before analysis
        or export.
    </p>
</div>
""",
    unsafe_allow_html=True
)


if st.session_state.table_locked:
    st.dataframe(current_df, use_container_width=True)
else:
    st.dataframe(current_df, use_container_width=True)

# -----------------------------------------------------------------------------
# EDIT CELLS MODE (LIVE BUFFER + APPLY / CANCEL)
# -----------------------------------------------------------------------------

if edit_mode == "Edit Cells" and not st.session_state.table_locked:

    # Start edit session ONCE
    if not st.session_state.edit_session_active:
        snapshot_state()  # save undo snapshot
        st.session_state.edit_session_active = True
        st.session_state.edit_buffer[swg_target] = current_df.copy(deep=True)

    st.markdown("### ✍️ Cell Editor")

    edited_df = st.data_editor(
        st.session_state.edit_buffer[swg_target],
        use_container_width=True,
        num_rows="dynamic",
        key=f"{swg_target}_editor"
    )

    # Buttons side by side
    btn_apply, btn_cancel = st.columns(2)

    with btn_apply:
        if st.button("💾 Apply Changes", use_container_width=True):
            st.session_state[df_key] = edited_df
            st.session_state.edit_session_active = False
            st.session_state.edit_buffer.pop(swg_target, None)
            st.success("Changes applied (Undo supported)")

    with btn_cancel:
        if st.button("❌ Cancel Changes", use_container_width=True):
            undo_action()
            st.session_state.edit_session_active = False
            st.session_state.edit_buffer.pop(swg_target, None)
            st.warning("Changes canceled")

# -----------------------------------------------------------------------------
# STRUCTURAL EDIT ACTIONS (ROW / COLUMN)
# -----------------------------------------------------------------------------

df = st.session_state[df_key]

if df.empty:
    st.warning("No data available.")
else:

    # INSERT ROW
    if edit_mode == "Insert Row" and not st.session_state.table_locked:
        if st.button("➕ Insert Empty Row"):
            snapshot_state()
            st.session_state[df_key] = pd.concat(
                [df, pd.DataFrame([{}])],
                ignore_index=True
            )
            st.success("Row inserted")

    # DELETE ROW
    elif edit_mode == "Delete Row" and not st.session_state.table_locked:
        row_idx = st.number_input(
            "Row index to delete",
            min_value=0,
            max_value=len(df) - 1,
            step=1
        )
        if st.button("🗑 Delete Row"):
            snapshot_state()
            st.session_state[df_key] = df.drop(index=row_idx).reset_index(drop=True)
            st.success("Row deleted")

    # INSERT COLUMN
    elif edit_mode == "Insert Column" and not st.session_state.table_locked:
        new_col = st.text_input("New column name")
        if st.button("➕ Insert Column") and new_col:
            snapshot_state()
            df[new_col] = None
            st.session_state[df_key] = df
            st.success("Column inserted")

    # DELETE COLUMN
    elif edit_mode == "Delete Column" and not st.session_state.table_locked:
        col = st.selectbox("Column to delete", df.columns)
        if st.button("🗑 Delete Column"):
            snapshot_state()
            st.session_state[df_key] = df.drop(columns=[col])
            st.success("Column deleted")

    # RENAME COLUMN
    elif edit_mode == "Rename Column" and not st.session_state.table_locked:
        col = st.selectbox("Column to rename", df.columns)
        new_name = st.text_input("New column name")
        if st.button("✏ Rename Column") and new_name:
            snapshot_state()
            st.session_state[df_key] = df.rename(columns={col: new_name})
            st.success("Column renamed")

    # MOVE COLUMN LEFT
    elif edit_mode == "Move Column Left" and not st.session_state.table_locked:
        col = st.selectbox("Column", df.columns)
        if st.button("⬅ Move Left"):
            idx = list(df.columns).index(col)
            if idx > 0:
                snapshot_state()
                cols = list(df.columns)
                cols[idx - 1], cols[idx] = cols[idx], cols[idx - 1]
                st.session_state[df_key] = df[cols]
                st.success("Column moved left")

    # MOVE COLUMN RIGHT
    elif edit_mode == "Move Column Right" and not st.session_state.table_locked:
        col = st.selectbox("Column", df.columns)
        if st.button("➡ Move Right"):
            idx = list(df.columns).index(col)
            if idx < len(df.columns) - 1:
                snapshot_state()
                cols = list(df.columns)
                cols[idx + 1], cols[idx] = cols[idx], cols[idx + 1]
                st.session_state[df_key] = df[cols]
                st.success("Column moved right")

    # MERGE COLUMNS
    elif edit_mode == "Merge Columns" and not st.session_state.table_locked:
        cols_to_merge = st.multiselect(
            "Select exactly 2 columns",
            df.columns,
            max_selections=2
        )
        new_col = st.text_input("Merged column name")

        if st.button("🔗 Merge Columns") and len(cols_to_merge) == 2 and new_col:
            snapshot_state()
            df[new_col] = (
                df[cols_to_merge[0]].astype(str)
                + " | "
                + df[cols_to_merge[1]].astype(str)
            )
            st.session_state[df_key] = df.drop(columns=cols_to_merge)
            st.success("Columns merged")

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# END PART 2
# =============================================================================

# =============================================================================
# PART 3 — ADVANCED STATISTICAL ANALYSIS (FIXED VERSION)
# =============================================================================

import numpy as np
import altair as alt
from io import BytesIO

alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# UI HEADER
# -----------------------------------------------------------------------------

st.markdown(
    """
<div class="card">
    <div class="section-title">📊 Advanced Statistical Analysis</div>
    <p style="color:#6b7280;font-size:13px;">
        Detailed distribution, box plots, outliers, and summary statistics per SWG.
    </p>
</div>
""",
    unsafe_allow_html=True
)

show_stats = st.toggle("Show Advanced Statistical Analysis", value=False)

if not show_stats:
    st.info("Advanced Statistical Analysis is hidden.")
else:

    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================

    def descriptive_stats(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        return {
            "Count": s.count(),
            "Missing": s.isna().sum(),
            "Mean": s.mean(),
            "Median": s.median(),
            "Std Dev": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "Range": s.max() - s.min(),
            "IQR": s.quantile(0.75) - s.quantile(0.25),
            "Variance": s.var(),
        }

    def detect_outliers(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return []
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return s[(s < lower) | (s > upper)]

    def boxplot(df, col, title):
        tmp = df[[col]].dropna()
        return (
            alt.Chart(tmp)
            .mark_boxplot(size=60, extent="min-max")
            .encode(y=alt.Y(f"{col}:Q", title=col))
            .properties(title=title, height=220)
        )

    # =============================================================================
    # ANALYSIS CONTENT
    # =============================================================================

    METRICS = [
        "Active Power (MW)",
        "Reactive Power (Mvar)",
        "SOC (%)"
    ]

    excel_buffer = BytesIO()
    writer = pd.ExcelWriter(excel_buffer, engine="xlsxwriter")

    for swg in SWG_LIST:
        df = st.session_state[f"{swg}_data"]
        label = swg.replace("SWG", "SWG-")

        st.markdown(f"### 🔋 {label}")

        if df.empty:
            st.warning("No data available.")
            continue

        # ---------------------------------------------------------------------
        # DESCRIPTIVE STATISTICS TABLE
        # ---------------------------------------------------------------------

        stats_df = pd.DataFrame({
            metric: descriptive_stats(df[metric]) for metric in METRICS
        })

        st.dataframe(
            stats_df.style.format("{:.3f}"),
            use_container_width=True
        )

        stats_df.to_excel(writer, sheet_name=f"{label}_Stats")

        # ---------------------------------------------------------------------
        # BOX PLOTS
        # ---------------------------------------------------------------------

        st.markdown("#### 📦 Distribution (Box Plots)")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.altair_chart(boxplot(df, METRICS[0], "Active Power"), use_container_width=True)
        with c2:
            st.altair_chart(boxplot(df, METRICS[1], "Reactive Power"), use_container_width=True)
        with c3:
            st.altair_chart(boxplot(df, METRICS[2], "SOC (%)"), use_container_width=True)

        # ---------------------------------------------------------------------
        # OUTLIERS
        # ---------------------------------------------------------------------

        st.markdown("#### 🚨 Outlier Detection (IQR Method)")
        outlier_rows = []

        for metric in METRICS:
            for value in detect_outliers(df[metric]):
                outlier_rows.append({
                    "Metric": metric,
                    "Outlier Value": value
                })

        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
        else:
            st.success("No outliers detected.")

        st.markdown("---")

    writer.close()

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    st.download_button(
        "📄 Download Statistics (Excel)",
        data=excel_buffer.getvalue(),
        file_name="SWG_Advanced_Statistics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =============================================================================
# END STATISTICS SECTION
# =============================================================================

# =============================================================================
# ADVANCED DATA VISUALIZATION (FINAL – ROLLING FIX + COMPACT)
# =============================================================================

import altair as alt

alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------

st.markdown(
    """
<div class="card">
    <div class="section-title">📈 Advanced Data Visualization</div>
    <p style="color:#6b7280;font-size:13px;">
        Compact interactive charts for trend analysis and SWG comparison.
    </p>
</div>
""",
    unsafe_allow_html=True
)

show_viz = st.toggle("Show Advanced Data Visualization", value=False)

if not show_viz:
    st.info("Advanced Data Visualization is hidden.")
else:

    # =============================================================================
    # CONTROLS — ROW 1
    # =============================================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        metric = st.selectbox(
            "Metric",
            ["Active Power (MW)", "Reactive Power (Mvar)", "SOC (%)"]
        )

    with c2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Step Line", "Area"]
        )

    with c3:
        view_mode = st.selectbox(
            "View Mode",
            ["Per SWG", "Compare All SWG"]
        )

    # =============================================================================
    # CONTROLS — ROW 2
    # =============================================================================

    c4, c5, c6 = st.columns(3)

    with c4:
        show_rolling = st.checkbox("Show Rolling Average")

    with c5:
        show_points = st.checkbox("Show Points", value=True)

    with c6:
        window = st.slider(
            "Rolling Window",
            2, 10, 3,
            disabled=not show_rolling
        )

    # =============================================================================
    # CONTROLS — VISUAL OPTIONS
    # =============================================================================

    c7, c8, c9 = st.columns(3)

    with c7:
        show_legend = st.checkbox("Show Legend", value=True)

    with c8:
        show_labels = st.checkbox("Show Value Labels")

    with c9:
        show_axis_labels = st.checkbox("Show Axis Labels", value=True)

    # =============================================================================
    # DATA PREPARATION
    # =============================================================================

    frames = []

    for swg in SWG_LIST:
        df = st.session_state[f"{swg}_data"]
        if not df.empty:
            tmp = df.copy()
            tmp["DateTime"] = pd.to_datetime(tmp.iloc[:, 0], errors="coerce")
            tmp["Value"] = pd.to_numeric(tmp[metric], errors="coerce")
            tmp["SWG"] = swg.replace("SWG", "SWG-")
            frames.append(tmp[["DateTime", "Value", "SWG"]])

    if not frames:
        st.warning("No data available for visualization.")
    else:
        viz_df = pd.concat(frames).dropna()

        # =============================================================================
        # SAFE ROLLING AVERAGE (DUPLICATE DATETIME FIX)
        # =============================================================================

        if show_rolling:
            viz_df = (
                viz_df
                .sort_values(["SWG", "DateTime"])
                .reset_index(drop=True)
            )

            viz_df["Rolling"] = (
                viz_df
                .groupby("SWG", group_keys=False)["Value"]
                .apply(lambda s: s.rolling(window=window, min_periods=1).mean())
            )

        # =============================================================================
        # CHART BUILDERS (ALTAR SAFE)
        # =============================================================================

        interpolate = "step-after" if chart_type == "Step Line" else "linear"

        def build_chart(df, height):
            if chart_type == "Area":
                base = alt.Chart(df).mark_area(
                    interpolate=interpolate,
                    opacity=0.35
                )
            else:
                base = alt.Chart(df).mark_line(
                    interpolate=interpolate,
                    strokeWidth=2,
                    point=show_points
                )

            return base.encode(
                x=alt.X(
                    "DateTime:T",
                    title="Date & Time" if show_axis_labels else None,
                    axis=alt.Axis(
                        format="%H:%M:%S",
                        labelAngle=-30
                    )
                ),
                y=alt.Y(
                    "Value:Q",
                    title=metric if show_axis_labels else None
                ),
                color=alt.Color(
                    "SWG:N",
                    legend=alt.Legend(title="SWG") if show_legend else None
                ),
                tooltip=[
                    "SWG",
                    alt.Tooltip("DateTime:T", title="DateTime"),
                    alt.Tooltip("Value:Q", title=metric, format=".2f")
                ]
            ).properties(height=height)

        def rolling_chart(df):
            return alt.Chart(df).mark_line(
                strokeDash=[6, 4],
                strokeWidth=1.5
            ).encode(
                x="DateTime:T",
                y="Rolling:Q",
                color="SWG:N"
            )

        def value_labels(df):
            return alt.Chart(df).mark_text(
                dy=-8,
                fontSize=10
            ).encode(
                x="DateTime:T",
                y="Value:Q",
                text=alt.Text("Value:Q", format=".1f"),
                color="SWG:N"
            )

        # =============================================================================
        # RENDER — COMPACT (NO SCROLL)
        # =============================================================================

        if view_mode == "Compare All SWG":
            st.markdown("### 🔀 Compare All SWGs")

            chart = build_chart(viz_df, height=300)

            if show_rolling:
                chart += rolling_chart(viz_df)

            if show_labels:
                chart += value_labels(viz_df)

            st.altair_chart(chart, use_container_width=True)

        else:
            st.markdown("### 🔍 Per-SWG Analysis")

            cols = st.columns(len(viz_df["SWG"].unique()))

            for col, swg in zip(cols, viz_df["SWG"].unique()):
                with col:
                    st.markdown(f"**{swg}**")
                    sub = viz_df[viz_df["SWG"] == swg]

                    chart = build_chart(sub, height=220)

                    if show_rolling:
                        chart += rolling_chart(sub)

                    if show_labels:
                        chart += value_labels(sub)

                    st.altair_chart(chart, use_container_width=True)

# =============================================================================
# END ADVANCED DATA VISUALIZATION
# =============================================================================

# =============================================================================
# PART — DOWNLOAD DATA (FINAL • TIMEZONE-SAFE • ERROR-PROOF)
# =============================================================================

import json
import numpy as np
from io import BytesIO

st.markdown(
    """
<div class="card">
    <div class="section-title">⬇️ Download Data</div>
    <p style="color:#6b7280;font-size:13px;">
        Export SWG data in CSV, Excel, or JSON format (timezone-safe).
    </p>
</div>
""",
    unsafe_allow_html=True
)

show_download = st.toggle("Show Download Options", value=True)

if show_download:

    # -------------------------------------------------------------------------
    # 1️⃣ ALIGN ALL SWG DATA BY ROW INDEX
    # -------------------------------------------------------------------------

    max_len = max(len(st.session_state[f"{swg}_data"]) for swg in SWG_LIST)

    aligned_frames = []

    for swg in SWG_LIST:
        df = st.session_state[f"{swg}_data"].copy()

        # Ensure dataframe has correct shape
        df = df.reset_index(drop=True)
        df = df.reindex(range(max_len))

        # 🔥 HARD FIX: FORCE DateTime → STRING (REMOVE TZ COMPLETELY)
        dt_col = df.columns[0]
        df[dt_col] = (
            pd.to_datetime(df[dt_col], errors="coerce")
            .dt.tz_localize(None)     # ⬅ REMOVE TIMEZONE
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Rename columns EXACTLY as required
        df.columns = [
            f"{swg}_DateTime",
            "Active Power (MW)",
            "Reactive Power (Mvar)",
            "SOC (%)",
        ]

        aligned_frames.append(df)

    # Combine into one dataframe
    export_df = pd.concat(aligned_frames, axis=1)

    # -------------------------------------------------------------------------
    # 2️⃣ FINAL SANITIZATION (EXCEL + JSON SAFE)
    # -------------------------------------------------------------------------

    def sanitize_value(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return str(v)

    export_df = export_df.applymap(sanitize_value)

    # -------------------------------------------------------------------------
    # 3️⃣ CSV EXPORT
    # -------------------------------------------------------------------------

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    # -------------------------------------------------------------------------
    # 4️⃣ EXCEL EXPORT (100% SAFE)
    # -------------------------------------------------------------------------

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        export_df.to_excel(
            writer,
            sheet_name="SWG_Data",
            index=False
        )

    excel_bytes = excel_buffer.getvalue()

    # -------------------------------------------------------------------------
    # 5️⃣ JSON EXPORT (ABSOLUTELY SAFE)
    # -------------------------------------------------------------------------

    json_data = {}

    for swg in SWG_LIST:
        df = st.session_state[f"{swg}_data"]

        records = []
        for _, row in df.iterrows():
            clean_row = {}
            for col, val in row.items():
                clean_row[col] = sanitize_value(val)
            records.append(clean_row)

        json_data[swg.replace("SWG", "SWG-")] = records

    json_bytes = json.dumps(json_data, indent=2).encode("utf-8")

    # -------------------------------------------------------------------------
    # 6️⃣ DOWNLOAD BUTTONS
    # -------------------------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="SWG_Data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c2:
        st.download_button(
            "⬇️ Download Excel",
            data=excel_bytes,
            file_name="SWG_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with c3:
        st.download_button(
            "⬇️ Download JSON",
            data=json_bytes,
            file_name="SWG_Data.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("Download section is hidden.")

# =============================================================================
# END DOWNLOAD DATA
# =============================================================================

