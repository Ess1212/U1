# =============================================================================
# POWER DISPATCH DASHBOARD
# PART 1 — FINAL (ACTIVE + REACTIVE POWER, LIVE CLOCK, UNDO/REDO)
# =============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Tuple, Dict
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

APP_TITLE = "POWER DISPATCH DASHBOARD"
SWG_LIST = ["SWG1", "SWG2", "SWG3"]

# Internal limits
POWER_MIN, POWER_MAX = -150.0, 150.0
REACTIVE_MIN, REACTIVE_MAX = -150.0, 150.0
SOC_MIN, SOC_MAX = 0.0, 100.0

LOCAL_TZ = ZoneInfo("Asia/Phnom_Penh")
DISPLAY_DT_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# =============================================================================
# REMOVE STREAMLIT UI
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
# DASHBOARD CSS
# =============================================================================

st.markdown(
    """
<style>
html, body {
    background-color:#f3f6fb;
    font-family:Inter, sans-serif;
    color:#1f2937;
}
.dashboard-header {
    background:#1f3a8a;
    color:white;
    padding:26px;
    border-radius:14px;
    font-size:34px;
    font-weight:800;
    text-align:center;
    margin-bottom:6px;
}
.card {
    background:white;
    border-radius:14px;
    padding:22px;
    box-shadow:0 6px 20px rgba(0,0,0,0.08);
    margin-bottom:24px;
}
.section-title {
    font-size:22px;
    font-weight:700;
    color:#1f3a8a;
    margin-bottom:14px;
}
.swg-title {
    font-size:18px;
    font-weight:600;
    margin-bottom:8px;
}
button[kind="primary"] {
    background-color:#2563eb;
    border-radius:10px;
    font-size:15px;
    font-weight:600;
}
[data-testid="stDataFrame"] {
    border-radius:10px;
}
</style>
""",
    unsafe_allow_html=True
)

# =============================================================================
# HEADER
# =============================================================================

st.markdown(f'<div class="dashboard-header">{APP_TITLE}</div>', unsafe_allow_html=True)

# =============================================================================
# LIVE CLOCK (RELIABLE)
# =============================================================================

html(
    """
<div style="text-align:center;font-size:17px;font-weight:600;color:#475569;margin-bottom:26px;">
  <span id="clock"></span>
</div>
<script>
function tick(){
  const n=new Date();
  document.getElementById("clock").innerHTML=
    n.toLocaleString([],{
      weekday:'short',year:'numeric',month:'short',day:'numeric',
      hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true
    });
}
setInterval(tick,1000);tick();
</script>
""",
    height=40
)

# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    for swg in SWG_LIST:
        if f"{swg}_data" not in st.session_state:
            st.session_state[f"{swg}_data"] = pd.DataFrame(
                columns=[
                    f"{swg}_DateTime",
                    f"{swg}_Power(MW)",
                    f"{swg}_Reactive(Mvar)",
                    f"{swg}_SOC(%)",
                ]
            )

    if "history" not in st.session_state:
        st.session_state.history = []
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []

init_state()

# =============================================================================
# UNDO / REDO
# =============================================================================

def snapshot():
    st.session_state.history.append({
        swg: st.session_state[f"{swg}_data"].copy(deep=True)
        for swg in SWG_LIST
    })
    st.session_state.redo_stack.clear()

def restore(snap: Dict[str, pd.DataFrame]):
    for swg in SWG_LIST:
        st.session_state[f"{swg}_data"] = snap[swg].copy(deep=True)

uc, rc = st.columns(2)
with uc:
    if st.button("↩ Undo", use_container_width=True) and st.session_state.history:
        st.session_state.redo_stack.append({
            swg: st.session_state[f"{swg}_data"].copy(deep=True)
            for swg in SWG_LIST
        })
        restore(st.session_state.history.pop())
        st.success("Undo applied")

with rc:
    if st.button("↪ Redo", use_container_width=True) and st.session_state.redo_stack:
        snapshot()
        restore(st.session_state.redo_stack.pop())
        st.success("Redo applied")

# =============================================================================
# VALIDATION
# =============================================================================

def validate(power, reactive, soc) -> Tuple[bool, str]:
    if power is None or reactive is None or soc is None:
        return False, "All fields are required"
    if not POWER_MIN <= power <= POWER_MAX:
        return False, "Power out of range"
    if not REACTIVE_MIN <= reactive <= REACTIVE_MAX:
        return False, "Reactive power out of range"
    if not SOC_MIN <= soc <= SOC_MAX:
        return False, "SOC out of range"
    return True, ""

# =============================================================================
# INSERT ROW
# =============================================================================

def insert_row(swg, power, reactive, soc):
    snapshot()
    row = {
        f"{swg}_DateTime": datetime.now(tz=LOCAL_TZ),
        f"{swg}_Power(MW)": power,
        f"{swg}_Reactive(Mvar)": reactive,
        f"{swg}_SOC(%)": soc,
    }
    st.session_state[f"{swg}_data"] = pd.concat(
        [st.session_state[f"{swg}_data"], pd.DataFrame([row])],
        ignore_index=True
    )

# =============================================================================
# INPUT SECTION
# =============================================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Input Data</div>', unsafe_allow_html=True)

cols = st.columns(3)

for i, swg in enumerate(SWG_LIST):
    with cols[i]:
        label = swg.replace("SWG", "SWG-")
        st.markdown(f'<div class="swg-title">{label}</div>', unsafe_allow_html=True)

        p = st.number_input(f"{label} Power (MW)", value=None, step=1.0, key=f"{swg}_p")
        q = st.number_input(f"{label} Reactive Power (Mvar)", value=None, step=1.0, key=f"{swg}_q")
        s = st.number_input(f"{label} SOC (%)", value=None, step=1.0, key=f"{swg}_s")

        if st.button(f"Add {label}", key=f"add_{swg}", use_container_width=True):
            ok, msg = validate(p, q, s)
            if not ok:
                st.error(msg)
            else:
                insert_row(swg, p, q, s)
                st.success("Added")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TABLE PREVIEW
# =============================================================================

def format_preview(df, swg):
    if df.empty:
        return df
    df = df.copy()
    dt = pd.to_datetime(df[f"{swg}_DateTime"]).dt.tz_localize(None)
    df[f"{swg}_DateTime"] = dt.dt.strftime(DISPLAY_DT_FORMAT)
    df[f"{swg}_Power(MW)"] = df[f"{swg}_Power(MW)"].astype(str) + " MW"
    df[f"{swg}_Reactive(Mvar)"] = df[f"{swg}_Reactive(Mvar)"].astype(str) + " Mvar"
    df[f"{swg}_SOC(%)"] = df[f"{swg}_SOC(%)"].astype(str) + " %"
    return df

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Table Preview</div>', unsafe_allow_html=True)

tcols = st.columns(3)
for i, swg in enumerate(SWG_LIST):
    with tcols[i]:
        st.dataframe(
            format_preview(st.session_state[f"{swg}_data"], swg),
            use_container_width=True,
            hide_index=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 1
# =============================================================================

# =============================================================================
# PART 2 — EDIT TABLE CONTROL PANEL (CLEAN STRUCTURE)
# =============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# CONTROL PANEL CARD
# =============================================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Table Controls</div>', unsafe_allow_html=True)

# ---------------- EDIT MODE ----------------

edit_mode = st.selectbox(
    "✏️ Edit Table",
    [
        "Off",
        "Insert Row",
        "Delete Row",
        "Insert Column",
        "Delete Column",
        "Rename Column",
        "Move Column Left",
        "Move Column Right",
        "Merge Columns",
    ],
    index=0,
    help="Choose an action to modify table structure or data"
)

# ---------------- LOCK TABLE (BELOW) ----------------

st.session_state.table_locked = st.toggle(
    "🔒 Lock Table (Lock all tables)",
    value=st.session_state.get("table_locked", False),
    help="When locked, all tables become read-only"
)

# ---------------- STATUS MESSAGE ----------------

if st.session_state.table_locked:
    st.error("Tables are locked. Editing is disabled.")
elif edit_mode != "Off":
    st.success(f"Edit mode enabled: {edit_mode}")
else:
    st.info("Edit mode is off")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SELECT SWG
# =============================================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Select SWG</div>', unsafe_allow_html=True)

swg_target = st.selectbox(
    "SWG",
    SWG_LIST,
    format_func=lambda x: x.replace("SWG", "SWG-")
)

df_key = f"{swg_target}_data"
df = st.session_state[df_key]

# =============================================================================
# SNAPSHOT (UNDO SUPPORT)
# =============================================================================

def snapshot_state():
    st.session_state.history.append({
        swg: st.session_state[f"{swg}_data"].copy(deep=True)
        for swg in SWG_LIST
    })
    st.session_state.redo_stack.clear()

# =============================================================================
# EDIT ACTIONS
# =============================================================================

if df.empty:
    st.warning("No data available for this SWG.")
else:

    # ---------------- INSERT ROW ----------------
    if edit_mode == "Insert Row" and not st.session_state.table_locked:
        if st.button("➕ Insert Empty Row"):
            snapshot_state()
            st.session_state[df_key] = pd.concat(
                [df, pd.DataFrame([{}])],
                ignore_index=True
            )
            st.success("Row inserted")

    # ---------------- DELETE ROW ----------------
    elif edit_mode == "Delete Row" and not st.session_state.table_locked:
        row_idx = st.number_input(
            "Row index",
            min_value=0,
            max_value=len(df) - 1,
            step=1
        )
        if st.button("🗑 Delete Row"):
            snapshot_state()
            st.session_state[df_key] = df.drop(index=row_idx).reset_index(drop=True)
            st.success("Row deleted")

    # ---------------- INSERT COLUMN ----------------
    elif edit_mode == "Insert Column" and not st.session_state.table_locked:
        new_col = st.text_input("New column name")
        if st.button("➕ Insert Column") and new_col:
            snapshot_state()
            df[new_col] = None
            st.session_state[df_key] = df
            st.success("Column inserted")

    # ---------------- DELETE COLUMN ----------------
    elif edit_mode == "Delete Column" and not st.session_state.table_locked:
        col = st.selectbox("Column", df.columns)
        if st.button("🗑 Delete Column"):
            snapshot_state()
            st.session_state[df_key] = df.drop(columns=[col])
            st.success("Column deleted")

    # ---------------- RENAME COLUMN ----------------
    elif edit_mode == "Rename Column" and not st.session_state.table_locked:
        col = st.selectbox("Column", df.columns)
        new_name = st.text_input("New column name")
        if st.button("✏ Rename Column") and new_name:
            snapshot_state()
            st.session_state[df_key] = df.rename(columns={col: new_name})
            st.success("Column renamed")

    # ---------------- MOVE COLUMN LEFT ----------------
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

    # ---------------- MOVE COLUMN RIGHT ----------------
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

    # ---------------- MERGE COLUMNS ----------------
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

# =============================================================================
# EDITABLE TABLE PREVIEW
# =============================================================================

st.markdown("### Table Preview")

edited_df = st.data_editor(
    st.session_state[df_key],
    disabled=st.session_state.table_locked,
    use_container_width=True,
    num_rows="dynamic",
    key=f"{swg_target}_editor"
)

if not edited_df.equals(st.session_state[df_key]):
    snapshot_state()
    st.session_state[df_key] = edited_df
    st.success("Table updated")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 2
# =============================================================================

# =============================================================================
# PART 3 — FULL STATISTICS ANALYSIS (LARGE NUMBERS UI)
# =============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Statistics Analysis</div>', unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def stats_block(series: pd.Series) -> dict:
    s = to_numeric(series)
    return {
        "Count": int(s.count()),
        "Missing": int(s.isna().sum()),
        "Min": s.min(),
        "Max": s.max(),
        "Mean": s.mean(),
        "Std": s.std(),
    }

# =============================================================================
# METRIC CARD — LARGE NUMBER STYLE
# =============================================================================

def stat_card(label, value, unit, color):
    display = "—" if pd.isna(value) else f"{value:.2f}"
    st.markdown(
        f"""
        <div style="
            background:white;
            border-radius:16px;
            padding:16px;
            box-shadow:0 6px 18px rgba(0,0,0,0.08);
            margin-bottom:12px;
            text-align:center;
        ">
            <div style="
                font-size:13px;
                color:#64748b;
                margin-bottom:6px;
            ">
                {label}
            </div>
            <div style="
                font-size:34px;
                font-weight:900;
                color:{color};
                line-height:1.2;
            ">
                {display}
                <span style="font-size:16px;font-weight:600;"> {unit}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# STATISTICS PER SWG
# =============================================================================

cols = st.columns(3)

for idx, swg in enumerate(SWG_LIST):
    df = st.session_state[f"{swg}_data"]

    with cols[idx]:
        st.markdown(f"### {swg.replace('SWG','SWG-')}")

        if df.empty:
            st.warning("No data available")
            continue

        p_stats = stats_block(df[f"{swg}_Power(MW)"])
        q_stats = stats_block(df[f"{swg}_Reactive(Mvar)"])
        s_stats = stats_block(df[f"{swg}_SOC(%)"])

        # ---------------- POWER ----------------
        st.markdown("#### 🔴 Power (MW)")
        stat_card("Mean", p_stats["Mean"], "MW", "#dc2626")
        stat_card("Min", p_stats["Min"], "MW", "#dc2626")
        stat_card("Max", p_stats["Max"], "MW", "#dc2626")
        stat_card("Std Dev", p_stats["Std"], "MW", "#dc2626")
        stat_card("Count", p_stats["Count"], "", "#dc2626")
        stat_card("Missing", p_stats["Missing"], "", "#dc2626")

        # ---------------- REACTIVE ----------------
        st.markdown("#### 🟢 Reactive Power (Mvar)")
        stat_card("Mean", q_stats["Mean"], "Mvar", "#16a34a")
        stat_card("Min", q_stats["Min"], "Mvar", "#16a34a")
        stat_card("Max", q_stats["Max"], "Mvar", "#16a34a")
        stat_card("Std Dev", q_stats["Std"], "Mvar", "#16a34a")
        stat_card("Count", q_stats["Count"], "", "#16a34a")
        stat_card("Missing", q_stats["Missing"], "", "#16a34a")

        # ---------------- SOC ----------------
        st.markdown("#### 🟠 SOC (%)")
        stat_card("Mean", s_stats["Mean"], "%", "#f97316")
        stat_card("Min", s_stats["Min"], "%", "#f97316")
        stat_card("Max", s_stats["Max"], "%", "#f97316")
        stat_card("Std Dev", s_stats["Std"], "%", "#f97316")
        stat_card("Count", s_stats["Count"], "", "#f97316")
        stat_card("Missing", s_stats["Missing"], "", "#f97316")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SUMMARY TABLE (OPTIONAL, READABLE)
# =============================================================================

rows = []

for swg in SWG_LIST:
    df = st.session_state[f"{swg}_data"]
    if df.empty:
        continue

    p = stats_block(df[f"{swg}_Power(MW)"])
    q = stats_block(df[f"{swg}_Reactive(Mvar)"])
    s = stats_block(df[f"{swg}_SOC(%)"])

    rows.append({
        "SWG": swg.replace("SWG", "SWG-"),
        "P Mean (MW)": round(p["Mean"], 2),
        "P Min": round(p["Min"], 2),
        "P Max": round(p["Max"], 2),
        "P SD": round(p["Std"], 2),
        "Q Mean (Mvar)": round(q["Mean"], 2),
        "SOC Mean (%)": round(s["Mean"], 2),
        "Count": p["Count"],
        "Missing": p["Missing"],
    })

if rows:
    st.markdown("### Statistics Summary Table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =============================================================================
# END PART 3
# =============================================================================

# =============================================================================
# PART 4A — DATA VISUALIZATION FOUNDATION
# =============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Data Visualization</div>', unsafe_allow_html=True)

# =============================================================================
# GLOBAL COLOR SCHEME (ENERGY STANDARD)
# =============================================================================

COLOR_MAP = {
    "power": "#dc2626",      # red
    "reactive": "#16a34a",   # green
    "soc": "#f97316",        # orange
}

# =============================================================================
# VISUALIZATION SETTINGS (USER CONTROLS)
# =============================================================================

st.markdown("### Visualization Controls")

control_col1, control_col2, control_col3 = st.columns(3)

# ---------------- SELECT SWG ----------------

with control_col1:
    selected_swgs = st.multiselect(
        "Select SWG",
        options=SWG_LIST,
        default=SWG_LIST,
        format_func=lambda x: x.replace("SWG", "SWG-"),
        help="Choose which SWG to display"
    )

# ---------------- SELECT METRICS ----------------

with control_col2:
    selected_metrics = st.multiselect(
        "Select Metrics",
        options=["Power (MW)", "Reactive Power (Mvar)", "SOC (%)"],
        default=["Power (MW)", "Reactive Power (Mvar)", "SOC (%)"],
        help="Choose metrics to visualize"
    )

# ---------------- TIME FILTER ----------------

with control_col3:
    time_window = st.selectbox(
        "Time Window",
        ["All", "Last 1 hour", "Last 6 hours", "Last 24 hours"],
        help="Filter data by time range"
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# DATA PREPARATION HELPERS
# =============================================================================

def apply_time_filter(df: pd.DataFrame, swg: str, window: str) -> pd.DataFrame:
    """
    Filter dataframe based on selected time window.
    """
    if df.empty:
        return df

    time_col = f"{swg}_DateTime"
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    if window == "All":
        return df

    now = pd.Timestamp.now(tz=LOCAL_TZ)

    if window == "Last 1 hour":
        cutoff = now - pd.Timedelta(hours=1)
    elif window == "Last 6 hours":
        cutoff = now - pd.Timedelta(hours=6)
    elif window == "Last 24 hours":
        cutoff = now - pd.Timedelta(hours=24)
    else:
        return df

    return df[df[time_col] >= cutoff]

def prepare_metric_series(
    df: pd.DataFrame,
    swg: str,
    metric: str
) -> pd.DataFrame:
    """
    Prepare clean time-series dataframe for a given metric.
    """
    metric_map = {
        "Power (MW)": f"{swg}_Power(MW)",
        "Reactive Power (Mvar)": f"{swg}_Reactive(Mvar)",
        "SOC (%)": f"{swg}_SOC(%)",
    }

    time_col = f"{swg}_DateTime"
    value_col = metric_map[metric]

    if df.empty or value_col not in df.columns:
        return pd.DataFrame()

    ts = df[[time_col, value_col]].copy()
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")

    ts = ts.dropna()
    return ts.set_index(time_col)

# =============================================================================
# DATA AVAILABILITY CHECK
# =============================================================================

if not selected_swgs:
    st.warning("Please select at least one SWG to visualize.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    empty_count = 0
    for swg in selected_swgs:
        if st.session_state[f"{swg}_data"].empty:
            empty_count += 1

    if empty_count == len(selected_swgs):
        st.warning("No data available for selected SWGs.")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 4A
# =============================================================================

# =============================================================================
# PART 4B — CORE ENERGY DASHBOARD CHARTS
# =============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# CHART RENDERING PER SWG
# =============================================================================

for swg in selected_swgs:

    raw_df = st.session_state[f"{swg}_data"]

    # Apply time window filter (from Part 4A)
    df = apply_time_filter(raw_df, swg, time_window)

    st.markdown(f"### {swg.replace('SWG', 'SWG-')} Energy Trends")

    if df.empty:
        st.warning("No data available after applying time filter.")
        continue

    # Determine how many columns to render
    metric_columns = []

    if "Power (MW)" in selected_metrics:
        metric_columns.append("Power (MW)")
    if "Reactive Power (Mvar)" in selected_metrics:
        metric_columns.append("Reactive Power (Mvar)")
    if "SOC (%)" in selected_metrics:
        metric_columns.append("SOC (%)")

    if not metric_columns:
        st.info("No metrics selected.")
        continue

    chart_cols = st.columns(len(metric_columns))

    for idx, metric in enumerate(metric_columns):

        with chart_cols[idx]:

            # ---------------- POWER ----------------
            if metric == "Power (MW)":
                st.markdown("🔴 **Power (MW)**")
                ts = prepare_metric_series(df, swg, metric)

                if not ts.empty:
                    st.line_chart(
                        ts,
                        height=260,
                        use_container_width=True
                    )
                else:
                    st.info("No valid Power data")

            # ---------------- REACTIVE ----------------
            elif metric == "Reactive Power (Mvar)":
                st.markdown("🟢 **Reactive Power (Mvar)**")
                ts = prepare_metric_series(df, swg, metric)

                if not ts.empty:
                    st.line_chart(
                        ts,
                        height=260,
                        use_container_width=True
                    )
                else:
                    st.info("No valid Reactive Power data")

            # ---------------- SOC ----------------
            elif metric == "SOC (%)":
                st.markdown("🟠 **SOC (%)**")
                ts = prepare_metric_series(df, swg, metric)

                if not ts.empty:
                    st.line_chart(
                        ts,
                        height=260,
                        use_container_width=True
                    )
                else:
                    st.info("No valid SOC data")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# END PART 4B
# =============================================================================

# =============================================================================
# PART 4C — ADVANCED ENERGY DASHBOARD INSIGHTS
# =============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Advanced Energy Insights</div>', unsafe_allow_html=True)

# =============================================================================
# THRESHOLD CONFIGURATION (INDUSTRY DEFAULTS)
# =============================================================================

POWER_OVERLOAD_LIMIT = 120.0   # MW
SOC_LOW_LIMIT = 20.0           # %

# =============================================================================
# KPI CARD HELPER
# =============================================================================

def kpi_card(title, value, unit, color, subtitle=None):
    st.markdown(
        f"""
        <div style="
            background:white;
            border-radius:16px;
            padding:18px;
            box-shadow:0 6px 18px rgba(0,0,0,0.08);
            margin-bottom:14px;
        ">
            <div style="font-size:14px;color:#64748b;">
                {title}
            </div>
            <div style="
                font-size:36px;
                font-weight:900;
                color:{color};
                line-height:1.2;
            ">
                {value}
                <span style="font-size:16px;"> {unit}</span>
            </div>
            {f"<div style='font-size:12px;color:#64748b;'>{subtitle}</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# AGGREGATE LATEST VALUES
# =============================================================================

latest_rows = []

for swg in selected_swgs:
    df = apply_time_filter(st.session_state[f"{swg}_data"], swg, time_window)

    if df.empty:
        continue

    df_sorted = df.sort_values(by=f"{swg}_DateTime")

    latest_rows.append({
        "SWG": swg.replace("SWG", "SWG-"),
        "Power": pd.to_numeric(df_sorted[f"{swg}_Power(MW)"], errors="coerce").iloc[-1],
        "Reactive": pd.to_numeric(df_sorted[f"{swg}_Reactive(Mvar)"], errors="coerce").iloc[-1],
        "SOC": pd.to_numeric(df_sorted[f"{swg}_SOC(%)"], errors="coerce").iloc[-1],
        "Last Time": df_sorted[f"{swg}_DateTime"].iloc[-1],
        "Max Power": pd.to_numeric(df_sorted[f"{swg}_Power(MW)"], errors="coerce").max(),
        "Min SOC": pd.to_numeric(df_sorted[f"{swg}_SOC(%)"], errors="coerce").min(),
    })

latest_df = pd.DataFrame(latest_rows)

if latest_df.empty:
    st.warning("No data available for advanced insights.")
    st.markdown('</div>', unsafe_allow_html=True)
else:

    # =============================================================================
    # KPI SUMMARY ROW
    # =============================================================================

    st.markdown("### Key Performance Indicators")

    kpi_cols = st.columns(len(latest_df))

    for idx, row in latest_df.iterrows():
        with kpi_cols[idx]:
            kpi_card(
                title=f"{row['SWG']} Latest Power",
                value=f"{row['Power']:.2f}",
                unit="MW",
                color="#dc2626",
                subtitle=f"Last update: {row['Last Time']}"
            )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # =============================================================================
    # THRESHOLD ALERTS
    # =============================================================================

    st.markdown("### Alerts & Status")

    for _, row in latest_df.iterrows():
        if row["Power"] > POWER_OVERLOAD_LIMIT:
            st.error(f"⚠️ {row['SWG']} Power overload: {row['Power']:.2f} MW")

        if row["SOC"] < SOC_LOW_LIMIT:
            st.warning(f"🟠 {row['SWG']} Low SOC: {row['SOC']:.2f} %")

    if (
        (latest_df["Power"] <= POWER_OVERLOAD_LIMIT).all()
        and (latest_df["SOC"] >= SOC_LOW_LIMIT).all()
    ):
        st.success("✅ All systems operating within safe limits.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # =============================================================================
    # AGGREGATED COMPARISON TABLE
    # =============================================================================

    st.markdown("### Aggregated Comparison (Latest Values)")

    comparison_df = latest_df[
        ["SWG", "Power", "Reactive", "SOC", "Max Power", "Min SOC"]
    ].copy()

    comparison_df.rename(
        columns={
            "Power": "Latest Power (MW)",
            "Reactive": "Latest Reactive (Mvar)",
            "SOC": "Latest SOC (%)",
            "Max Power": "Max Power (MW)",
            "Min SOC": "Min SOC (%)",
        },
        inplace=True
    )

    st.dataframe(comparison_df, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 4C
# =============================================================================

# =============================================================================
# PART 4D — STEP CHART CONTROLS & DATA ENGINE (NO PLOTTING)
# =============================================================================

import pandas as pd

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Step Chart — Advanced Controls</div>', unsafe_allow_html=True)

# =============================================================================
# CONTROL PANEL
# =============================================================================

c1, c2, c3 = st.columns(3)

with c1:
    metric_mode = st.selectbox(
        "Metric Mode",
        ["Single Metric", "All Metrics (P + Q + SOC)"]
    )

with c2:
    selected_metric = st.selectbox(
        "Metric",
        ["Power (MW)", "Reactive Power (Mvar)", "SOC (%)"],
        disabled=(metric_mode != "Single Metric")
    )

with c3:
    step_mode = st.selectbox(
        "Step Mode",
        ["before", "after"]
    )

# -----------------------------------------------------------------------------
# POINT CONTROLS (BELOW METRIC)
# -----------------------------------------------------------------------------

show_points = st.toggle("Show Points", value=True)

point_size = st.slider(
    "Point Size",
    min_value=20,
    max_value=120,
    value=60,
    disabled=not show_points
)

# -----------------------------------------------------------------------------
# LINE STYLE CONTROLS
# -----------------------------------------------------------------------------

line_width = st.slider(
    "Line Width",
    min_value=1,
    max_value=6,
    value=3
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# AXIS RANGE CONTROLS (OPTIONAL)
# =============================================================================

axis_c1, axis_c2, axis_c3 = st.columns(3)

with axis_c1:
    custom_power_axis = st.toggle("Custom Power Axis")
    power_range = st.slider(
        "Power Range (MW)",
        -200, 200,
        (-150, 150),
        disabled=not custom_power_axis
    )

with axis_c2:
    custom_reactive_axis = st.toggle("Custom Reactive Axis")
    reactive_range = st.slider(
        "Reactive Range (Mvar)",
        -200, 200,
        (-150, 150),
        disabled=not custom_reactive_axis
    )

with axis_c3:
    custom_soc_axis = st.toggle("Custom SOC Axis")
    soc_range = st.slider(
        "SOC Range (%)",
        0, 100,
        (0, 100),
        disabled=not custom_soc_axis
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =============================================================================
# SWG VISIBILITY CONTROL
# =============================================================================

visible_swgs = st.multiselect(
    "Visible SWGs",
    selected_swgs,
    default=selected_swgs,
    format_func=lambda x: x.replace("SWG", "SWG-")
)

# =============================================================================
# DATA COLLECTION ENGINE
# =============================================================================

def collect_metric_data(metric_label, suffix):
    rows = []
    for swg in visible_swgs:
        df_raw = st.session_state[f"{swg}_data"]
        df = apply_time_filter(df_raw, swg, time_window)

        if df.empty:
            continue

        tcol = f"{swg}_DateTime"
        vcol = f"{swg}{suffix}"

        if vcol not in df.columns:
            continue

        tmp = df[[tcol, vcol]].copy()
        tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce")
        tmp[vcol] = pd.to_numeric(tmp[vcol], errors="coerce")
        tmp = tmp.dropna()

        if tmp.empty:
            continue

        tmp["Time"] = tmp[tcol]
        tmp["Value"] = tmp[vcol]
        tmp["Metric"] = metric_label
        tmp["SWG"] = swg.replace("SWG", "SWG-")

        rows.append(tmp[["Time", "Value", "Metric", "SWG"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# =============================================================================
# BUILD UNIFIED DATAFRAME FOR PLOTTING
# =============================================================================

data_frames = []

if metric_mode == "Single Metric":
    suffix_map = {
        "Power (MW)": "_Power(MW)",
        "Reactive Power (Mvar)": "_Reactive(Mvar)",
        "SOC (%)": "_SOC(%)",
    }
    data_frames.append(
        collect_metric_data(selected_metric, suffix_map[selected_metric])
    )

else:
    data_frames.append(collect_metric_data("Power (MW)", "_Power(MW)"))
    data_frames.append(collect_metric_data("Reactive Power (Mvar)", "_Reactive(Mvar)"))
    data_frames.append(collect_metric_data("SOC (%)", "_SOC(%)"))

step_plot_df = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

# =============================================================================
# DATA VALIDATION FLAG (USED BY PART 4E)
# =============================================================================

step_chart_ready = not step_plot_df.empty

if not step_chart_ready:
    st.warning("No data available for step chart with current settings.")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 4D
# =============================================================================

# =============================================================================
# PART 4E — STEP LINE RENDERING ENGINE (MULTI Y-AXIS, FIXED)
# =============================================================================

import altair as alt

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">SWG Step Line Comparison</div>', unsafe_allow_html=True)

# =============================================================================
# GUARD
# =============================================================================

if not step_chart_ready:
    st.info("Adjust settings above to display the step comparison chart.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # =============================================================================
    # BASE CHART
    # =============================================================================

    base = alt.Chart(step_plot_df).encode(
        x=alt.X(
            "Time:T",
            title="Time",
            axis=alt.Axis(format="%H:%M:%S", labelAngle=-30)
        ),
        color=alt.Color(
            "SWG:N",
            legend=alt.Legend(title="SWG")
        ),
        tooltip=[
            alt.Tooltip("SWG:N", title="SWG"),
            alt.Tooltip("Metric:N", title="Metric"),
            alt.Tooltip("Time:T", title="Time"),
            alt.Tooltip("Value:Q", title="Value"),
        ]
    )

    layers = []

    # =============================================================================
    # POWER (LEFT AXIS)
    # =============================================================================

    if "Power (MW)" in step_plot_df["Metric"].unique():

        power_scale = (
            alt.Scale(domain=list(power_range), zero=False)
            if custom_power_axis
            else alt.Scale(zero=False)
        )

        power_line = (
            base.transform_filter(alt.datum.Metric == "Power (MW)")
            .encode(
                y=alt.Y(
                    "Value:Q",
                    title="Power (MW)",
                    scale=power_scale,
                    axis=alt.Axis(titleColor="#dc2626")
                )
            )
            .mark_line(
                interpolate=f"step-{step_mode}",
                strokeWidth=line_width,
                color="#dc2626"
            )
        )

        layers.append(power_line)

        if show_points:
            layers.append(
                power_line.mark_point(size=point_size, filled=True)
            )

    # =============================================================================
    # REACTIVE POWER (RIGHT AXIS)
    # =============================================================================

    if "Reactive Power (Mvar)" in step_plot_df["Metric"].unique():

        reactive_scale = (
            alt.Scale(domain=list(reactive_range), zero=False)
            if custom_reactive_axis
            else alt.Scale(zero=False)
        )

        reactive_line = (
            base.transform_filter(alt.datum.Metric == "Reactive Power (Mvar)")
            .encode(
                y=alt.Y(
                    "Value:Q",
                    title="Reactive Power (Mvar)",
                    scale=reactive_scale,
                    axis=alt.Axis(titleColor="#16a34a")
                )
            )
            .mark_line(
                interpolate=f"step-{step_mode}",
                strokeDash=[6, 4],
                strokeWidth=line_width,
                color="#16a34a"
            )
        )

        layers.append(reactive_line)

        if show_points:
            layers.append(
                reactive_line.mark_point(size=point_size, filled=True)
            )

    # =============================================================================
    # SOC (RIGHT OFFSET AXIS)
    # =============================================================================

    if "SOC (%)" in step_plot_df["Metric"].unique():

        soc_scale = (
            alt.Scale(domain=list(soc_range), zero=False)
            if custom_soc_axis
            else alt.Scale(domain=[0, 100], zero=False)
        )

        soc_line = (
            base.transform_filter(alt.datum.Metric == "SOC (%)")
            .encode(
                y=alt.Y(
                    "Value:Q",
                    title="SOC (%)",
                    scale=soc_scale,
                    axis=alt.Axis(titleColor="#f97316")
                )
            )
            .mark_line(
                interpolate=f"step-{step_mode}",
                strokeDash=[2, 2],
                strokeWidth=line_width,
                color="#f97316"
            )
        )

        layers.append(soc_line)

        if show_points:
            layers.append(
                soc_line.mark_point(size=point_size, filled=True)
            )

    # =============================================================================
    # FINAL CHART
    # =============================================================================

    final_chart = (
        alt.layer(*layers)
        .resolve_scale(y="independent")
        .properties(height=480)
    )

    st.altair_chart(final_chart, use_container_width=True)

    st.caption("Step Line Comparison • Independent Y-Axes • EMS-grade Visualization")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 4E
# =============================================================================

# =============================================================================
# PART 5 — EXPORT DATA (CSV / XLSX / JSON)
# =============================================================================

import io
import json

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Export Data</div>', unsafe_allow_html=True)

# =============================================================================
# HELPER — FORMAT DATETIME (HUMAN READABLE)
# =============================================================================

def format_datetime(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime(
        "%m/%d/%Y %I:%M:%S %p"
    )

# =============================================================================
# BUILD EXPORT DATAFRAME
# =============================================================================

export_frames = []

for swg in SWG_LIST:
    df = st.session_state[f"{swg}_data"]

    if df.empty:
        continue

    df = df.copy()

    # Rename columns to export format
    rename_map = {
        f"{swg}_DateTime": f"{swg}_DateTime",
        f"{swg}_Power(MW)": f"{swg}_Power(MW)",
        f"{swg}_Reactive(Mvar)": f"{swg}_Reactive(Mvar)",
        f"{swg}_SOC(%)": f"{swg}_SOC(%)",
    }

    df = df[list(rename_map.keys())].rename(columns=rename_map)

    # Format datetime
    df[f"{swg}_DateTime"] = format_datetime(df[f"{swg}_DateTime"])

    export_frames.append(df.reset_index(drop=True))

# =============================================================================
# ALIGN ROW COUNTS (NO DATA LOSS)
# =============================================================================

if not export_frames:
    st.warning("No data available to export.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    max_len = max(len(df) for df in export_frames)

    for i, df in enumerate(export_frames):
        if len(df) < max_len:
            export_frames[i] = df.reindex(range(max_len))

    export_df = pd.concat(export_frames, axis=1)

    st.success("Export data prepared successfully.")

    # =============================================================================
    # CSV EXPORT (ALWAYS AVAILABLE)
    # =============================================================================

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    # =============================================================================
    # JSON EXPORT (ALWAYS AVAILABLE)
    # =============================================================================

    json_data = export_df.to_dict(orient="records")
    json_buffer = json.dumps(json_data, indent=2)

    # =============================================================================
    # XLSX EXPORT (SAFE CHECK)
    # =============================================================================

    xlsx_available = True
    try:
        import openpyxl  # noqa
    except Exception:
        xlsx_available = False

    def to_excel_bytes(df):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Energy Data")
        return buffer.getvalue()

    # =============================================================================
    # DOWNLOAD BUTTONS
    # =============================================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=csv_buffer.getvalue(),
            file_name="energy_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        if xlsx_available:
            st.download_button(
                "⬇️ Download XLSX",
                data=to_excel_bytes(export_df),
                file_name="energy_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info("XLSX export unavailable (openpyxl not installed).")

    with c3:
        st.download_button(
            "⬇️ Download JSON",
            data=json_buffer,
            file_name="energy_data.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# END PART 5
# =============================================================================
