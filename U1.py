# =============================================================================
# SECTION 1 — Imports & App Configuration
# =============================================================================
# PURPOSE:
# - Centralize ALL imports (no imports allowed after this section)
# - Configure Streamlit, Pandas, Altair globally
# - Define application-wide metadata & formatting standards
# - Prepare environment for SQLite persistence & exports
#
# RULES:
# - NO UI rendering
# - NO database access
# - NO session_state usage
# - NO business logic
# =============================================================================


# -----------------------------------------------------------------------------
# 1.1 — Standard Library Imports
# -----------------------------------------------------------------------------
import os
import json
import sqlite3
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO


# -----------------------------------------------------------------------------
# 1.2 — Third-Party Libraries
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from streamlit.components.v1 import html


# -----------------------------------------------------------------------------
# 1.3 — Streamlit Application Configuration
# -----------------------------------------------------------------------------
# NOTE:
# This MUST be called once and ONLY once in the entire application.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Power Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -----------------------------------------------------------------------------
# 1.4 — Global Pandas Configuration
# -----------------------------------------------------------------------------
# Ensures safe dataframe mutation behavior and predictable performance.
# -----------------------------------------------------------------------------
pd.options.mode.copy_on_write = True
pd.options.display.float_format = "{:,.3f}".format


# -----------------------------------------------------------------------------
# 1.5 — Global Altair Configuration
# -----------------------------------------------------------------------------
# Prevents row-limit issues when visualizing large datasets.
# -----------------------------------------------------------------------------
alt.data_transformers.disable_max_rows()


# -----------------------------------------------------------------------------
# 1.6 — Timezone & Datetime Standards
# -----------------------------------------------------------------------------
# All internal datetimes are timezone-aware.
# All exports are timezone-stripped strings.
# -----------------------------------------------------------------------------
LOCAL_TIMEZONE: ZoneInfo = ZoneInfo("Asia/Phnom_Penh")

DISPLAY_DATETIME_FORMAT: str = "%m/%d/%Y %I:%M:%S %p"
EXPORT_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DB_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# -----------------------------------------------------------------------------
# 1.7 — Application Identity Metadata
# -----------------------------------------------------------------------------
# Used by UI headers, exports, reports, and audit trails.
# -----------------------------------------------------------------------------
APP_TITLE: str = "Power Dispatch Dashboard"
APP_SUBTITLE: str = "Real-Time Energy Operations & Analysis"
COMPANY_NAME: str = "SchneiTech Group"
APP_VERSION: str = "1.0.0"
APP_ENVIRONMENT: str = "PRODUCTION"   # DEV | TEST | PRODUCTION


# -----------------------------------------------------------------------------
# 1.8 — Database File Naming Standards
# -----------------------------------------------------------------------------
# Daily SQLite files are auto-generated using this convention:
# energy_YYYY-MM-DD.db
# -----------------------------------------------------------------------------
DB_FILE_PREFIX: str = "energy_"
DB_FILE_EXTENSION: str = ".db"
DB_DATE_FORMAT: str = "%Y-%m-%d"


# -----------------------------------------------------------------------------
# 1.9 — Supported SWG Units (Logical Identifiers)
# -----------------------------------------------------------------------------
# These identifiers are used consistently across:
# - session_state
# - database schema
# - exports
# - visualizations
# -----------------------------------------------------------------------------
SWG_IDS: Tuple[str, ...] = ("SWG1", "SWG2", "SWG3")


# -----------------------------------------------------------------------------
# 1.10 — Export File Naming Defaults
# -----------------------------------------------------------------------------
EXPORT_CSV_NAME: str = "SWG_Data.csv"
EXPORT_EXCEL_NAME: str = "SWG_Data.xlsx"
EXPORT_JSON_NAME: str = "SWG_Data.json"


# -----------------------------------------------------------------------------
# 1.11 — Safety Flags & Feature Toggles
# -----------------------------------------------------------------------------
# These allow future expansion without refactoring.
# -----------------------------------------------------------------------------
ENABLE_UNDO_REDO: bool = True
ENABLE_LIVE_EDITING: bool = True
ENABLE_SQLITE_PERSISTENCE: bool = True


# =============================================================================
# END SECTION 1 — Imports & App Configuration
# =============================================================================

# =============================================================================
# SECTION 2 — Constants & Limits
# =============================================================================
# PURPOSE:
# - Centralize ALL limits, schemas, column names, labels, and rules
# - Prevent magic numbers and hard-coded strings
# - Guarantee consistency across UI, DB, exports, and analysis
#
# RULES:
# - NO logic
# - NO functions
# - NO database access
# - NO Streamlit UI
# =============================================================================


# -----------------------------------------------------------------------------
# 2.1 — General Numeric Precision Rules
# -----------------------------------------------------------------------------
FLOAT_PRECISION_UI: int = 3
FLOAT_PRECISION_DB: int = 6
FLOAT_PRECISION_EXPORT: int = 3

INTEGER_STEP_DEFAULT: int = 1
FLOAT_STEP_DEFAULT: float = 0.1


# -----------------------------------------------------------------------------
# 2.2 — SWG Physical Operating Limits
# -----------------------------------------------------------------------------
# These limits are enforced at:
# - UI input
# - Edit validation
# - Save-to-database
# - Import/export sanity checks
# -----------------------------------------------------------------------------

# Active Power (MW)
ACTIVE_POWER_MIN: float = -150.0
ACTIVE_POWER_MAX: float = 150.0
ACTIVE_POWER_UNIT: str = "MW"

# Reactive Power (Mvar)
REACTIVE_POWER_MIN: float = -150.0
REACTIVE_POWER_MAX: float = 150.0
REACTIVE_POWER_UNIT: str = "Mvar"

# State of Charge (%)
SOC_MIN: float = 0.0
SOC_MAX: float = 100.0
SOC_UNIT: str = "%"


# -----------------------------------------------------------------------------
# 2.3 — Validation Error Messages (User-Facing)
# -----------------------------------------------------------------------------
ERROR_REQUIRED_FIELD: str = "All fields are required."
ERROR_ACTIVE_RANGE: str = "Active Power is out of allowable range."
ERROR_REACTIVE_RANGE: str = "Reactive Power is out of allowable range."
ERROR_SOC_RANGE: str = "State of Charge (SOC) is out of allowable range."
ERROR_DATETIME_INVALID: str = "Invalid DateTime value."
ERROR_TABLE_LOCKED: str = "Table is locked and cannot be modified."
ERROR_EMPTY_TABLE: str = "No data available."
ERROR_INDEX_OUT_OF_RANGE: str = "Selected index is out of range."


# -----------------------------------------------------------------------------
# 2.4 — Canonical Column Names (IN-MEMORY PER SWG)
# -----------------------------------------------------------------------------
# These column names are used ONLY in session_state tables
# -----------------------------------------------------------------------------
COLUMN_DATETIME: str = "DateTime"
COLUMN_ACTIVE: str = "Active Power (MW)"
COLUMN_REACTIVE: str = "Reactive Power (Mvar)"
COLUMN_SOC: str = "SOC (%)"

SWG_BASE_COLUMNS: Tuple[str, ...] = (
    COLUMN_DATETIME,
    COLUMN_ACTIVE,
    COLUMN_REACTIVE,
    COLUMN_SOC,
)


# -----------------------------------------------------------------------------
# 2.5 — Wide Table Column Mapping (DATABASE + EXPORT)
# -----------------------------------------------------------------------------
# These are EXACT column names used in SQLite and Excel/CSV exports
# -----------------------------------------------------------------------------
WIDE_TABLE_COLUMNS: Tuple[str, ...] = (
    "SWG1_DateTime", "SWG1_Active", "SWG1_Reactive", "SWG1_SOC",
    "SWG2_DateTime", "SWG2_Active", "SWG2_Reactive", "SWG2_SOC",
    "SWG3_DateTime", "SWG3_Active", "SWG3_Reactive", "SWG3_SOC",
)


# -----------------------------------------------------------------------------
# 2.6 — SWG → Wide Column Mapping Dictionary
# -----------------------------------------------------------------------------
SWG_TO_WIDE_COLUMNS: Dict[str, Tuple[str, str, str, str]] = {
    "SWG1": ("SWG1_DateTime", "SWG1_Active", "SWG1_Reactive", "SWG1_SOC"),
    "SWG2": ("SWG2_DateTime", "SWG2_Active", "SWG2_Reactive", "SWG2_SOC"),
    "SWG3": ("SWG3_DateTime", "SWG3_Active", "SWG3_Reactive", "SWG3_SOC"),
}


# -----------------------------------------------------------------------------
# 2.7 — SQLite Data Types
# -----------------------------------------------------------------------------
SQLITE_TYPE_TEXT: str = "TEXT"
SQLITE_TYPE_REAL: str = "REAL"
SQLITE_TYPE_INTEGER: str = "INTEGER"


# -----------------------------------------------------------------------------
# 2.8 — SQLite Wide Table Schema Definition
# -----------------------------------------------------------------------------
SQLITE_WIDE_TABLE_NAME: str = "energy_wide"

SQLITE_WIDE_TABLE_SCHEMA: Dict[str, str] = {
    "id": SQLITE_TYPE_INTEGER,
    "SWG1_DateTime": SQLITE_TYPE_TEXT,
    "SWG1_Active": SQLITE_TYPE_REAL,
    "SWG1_Reactive": SQLITE_TYPE_REAL,
    "SWG1_SOC": SQLITE_TYPE_REAL,
    "SWG2_DateTime": SQLITE_TYPE_TEXT,
    "SWG2_Active": SQLITE_TYPE_REAL,
    "SWG2_Reactive": SQLITE_TYPE_REAL,
    "SWG2_SOC": SQLITE_TYPE_REAL,
    "SWG3_DateTime": SQLITE_TYPE_TEXT,
    "SWG3_Active": SQLITE_TYPE_REAL,
    "SWG3_Reactive": SQLITE_TYPE_REAL,
    "SWG3_SOC": SQLITE_TYPE_REAL,
}


# -----------------------------------------------------------------------------
# 2.9 — Default Empty DataFrame Templates (PER SWG)
# -----------------------------------------------------------------------------
DEFAULT_SWG_DF_TEMPLATE: Dict[str, List[str]] = {
    "SWG1": list(SWG_BASE_COLUMNS),
    "SWG2": list(SWG_BASE_COLUMNS),
    "SWG3": list(SWG_BASE_COLUMNS),
}


# -----------------------------------------------------------------------------
# 2.10 — Export Formatting Rules
# -----------------------------------------------------------------------------
EXPORT_NA_REPLACEMENT: str = ""
EXPORT_FLOAT_FORMAT: str = "{:.3f}"
EXPORT_DATE_FORMAT: str = EXPORT_DATETIME_FORMAT


# -----------------------------------------------------------------------------
# 2.11 — UI Labels (Single Source of Truth)
# -----------------------------------------------------------------------------
LABEL_ACTIVE: str = "⚡ Active Power (MW)"
LABEL_REACTIVE: str = "🔌 Reactive Power (Mvar)"
LABEL_SOC: str = "🔋 State of Charge (%)"

LABEL_ADD_ROW: str = "➕ Add Entry"
LABEL_UNDO: str = "↩ Undo"
LABEL_REDO: str = "↪ Redo"
LABEL_SAVE_DB: str = "💾 Save to Database"
LABEL_CLEAR_ALL: str = "🧹 Clear All Data"


# -----------------------------------------------------------------------------
# 2.12 — Undo / Redo Constraints
# -----------------------------------------------------------------------------
MAX_HISTORY_LENGTH: int = 100
UNDO_STACK_NAME: str = "history"
REDO_STACK_NAME: str = "redo_stack"


# -----------------------------------------------------------------------------
# 2.13 — Table Editing Rules
# -----------------------------------------------------------------------------
ALLOW_CELL_EDIT: bool = True
ALLOW_ROW_INSERT: bool = True
ALLOW_ROW_DELETE: bool = True
ALLOW_COLUMN_EDIT: bool = True

MAX_DYNAMIC_COLUMNS: int = 50
MAX_DYNAMIC_ROWS: int = 10_000


# -----------------------------------------------------------------------------
# 2.14 — Analysis Configuration
# -----------------------------------------------------------------------------
ANALYSIS_METRICS: Tuple[str, ...] = (
    COLUMN_ACTIVE,
    COLUMN_REACTIVE,
    COLUMN_SOC,
)

ROLLING_WINDOW_MIN: int = 2
ROLLING_WINDOW_MAX: int = 30
ROLLING_WINDOW_DEFAULT: int = 5


# -----------------------------------------------------------------------------
# 2.15 — Visualization Defaults
# -----------------------------------------------------------------------------
DEFAULT_CHART_HEIGHT: int = 280
DEFAULT_CHART_OPACITY: float = 0.35
DEFAULT_LINE_WIDTH: int = 2

COLOR_SCHEME: Tuple[str, ...] = (
    "#2563eb",  # Blue
    "#16a34a",  # Green
    "#dc2626",  # Red
)


# -----------------------------------------------------------------------------
# 2.16 — Safety Guards
# -----------------------------------------------------------------------------
ALLOW_EMPTY_SAVE: bool = False
CONFIRM_CLEAR_REQUIRED: bool = True
AUTO_SAVE_ENABLED: bool = True


# -----------------------------------------------------------------------------
# 2.17 — Reserved Keys (session_state protection)
# -----------------------------------------------------------------------------
RESERVED_SESSION_KEYS: Tuple[str, ...] = (
    "history",
    "redo_stack",
    "edit_buffer",
    "table_locked",
)


# -----------------------------------------------------------------------------
# 2.18 — Future Expansion Placeholders (LOCKED)
# -----------------------------------------------------------------------------
# DO NOT REMOVE — these ensure forward compatibility
ENABLE_SMOOTHING: bool = False
ENABLE_FORECASTING: bool = False
ENABLE_USER_AUTH: bool = False
ENABLE_ROLE_BASED_ACCESS: bool = False


# =============================================================================
# END SECTION 2 — Constants & Limits
# =============================================================================

# =============================================================================
# SECTION 3 — Database Connection
# =============================================================================
# PURPOSE:
# - Handle SQLite connection lifecycle
# - Enforce daily database file rotation
# - Provide safe, reusable DB connections
# - Isolate ALL low-level database access
#
# RULES:
# - NO table creation
# - NO insert / update / select logic
# - NO business rules
# - NO Streamlit UI
# =============================================================================


# -----------------------------------------------------------------------------
# 3.1 — Database Root Directory
# -----------------------------------------------------------------------------
# All SQLite files are stored in the application root directory.
# This avoids permission issues in Streamlit Cloud / local servers.
# -----------------------------------------------------------------------------
DB_ROOT_DIR: str = os.getcwd()


# -----------------------------------------------------------------------------
# 3.2 — Build Daily Database Filename
# -----------------------------------------------------------------------------
def get_today_db_filename(target_date: Optional[date] = None) -> str:
    """
    Build daily SQLite database filename.

    Format:
        energy_YYYY-MM-DD.db

    Args:
        target_date: Optional date override (used for testing)

    Returns:
        str: Database filename
    """
    d = target_date or date.today()
    return f"{DB_FILE_PREFIX}{d.strftime(DB_DATE_FORMAT)}{DB_FILE_EXTENSION}"


# -----------------------------------------------------------------------------
# 3.3 — Resolve Full Database Path
# -----------------------------------------------------------------------------
def get_today_db_path(target_date: Optional[date] = None) -> str:
    """
    Resolve absolute path for today's database file.

    Args:
        target_date: Optional date override

    Returns:
        str: Absolute file path
    """
    filename = get_today_db_filename(target_date)
    return os.path.join(DB_ROOT_DIR, filename)


# -----------------------------------------------------------------------------
# 3.4 — Ensure Database File Exists
# -----------------------------------------------------------------------------
def ensure_db_file_exists(db_path: str) -> None:
    """
    Ensure SQLite database file exists.
    If not, create an empty file safely.

    Args:
        db_path: Absolute database path
    """
    if not os.path.exists(db_path):
        # Touch the file safely
        open(db_path, "a", encoding="utf-8").close()


# -----------------------------------------------------------------------------
# 3.5 — SQLite Connection Factory
# -----------------------------------------------------------------------------
def create_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """
    Create a SQLite connection with safe defaults.

    - check_same_thread=False → Streamlit compatibility
    - isolation_level=None → Autocommit mode
    - row_factory → dict-like rows (future-ready)

    Args:
        db_path: Absolute database path

    Returns:
        sqlite3.Connection
    """
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        isolation_level=None,
        timeout=30
    )

    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------------------------------------------------------
# 3.6 — Cached Daily Connection (Streamlit-Safe)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_daily_connection(target_date: Optional[date] = None) -> sqlite3.Connection:
    """
    Get a cached SQLite connection for today's database.

    Streamlit guarantees:
    - One connection per process
    - Safe reuse across reruns
    - Auto cleanup on app restart

    Args:
        target_date: Optional date override

    Returns:
        sqlite3.Connection
    """
    db_path = get_today_db_path(target_date)
    ensure_db_file_exists(db_path)
    return create_sqlite_connection(db_path)


# -----------------------------------------------------------------------------
# 3.7 — Connection Health Check
# -----------------------------------------------------------------------------
def validate_connection(conn: sqlite3.Connection) -> bool:
    """
    Validate SQLite connection health.

    Args:
        conn: sqlite3.Connection

    Returns:
        bool: True if connection is valid
    """
    try:
        conn.execute("SELECT 1;")
        return True
    except sqlite3.Error:
        return False


# -----------------------------------------------------------------------------
# 3.8 — Safe Connection Access Wrapper
# -----------------------------------------------------------------------------
def get_validated_connection(target_date: Optional[date] = None) -> sqlite3.Connection:
    """
    Get a validated SQLite connection.
    Recreates connection if invalid.

    Args:
        target_date: Optional date override

    Returns:
        sqlite3.Connection
    """
    conn = get_daily_connection(target_date)

    if not validate_connection(conn):
        # Clear cached resource and recreate
        st.cache_resource.clear()
        conn = get_daily_connection(target_date)

    return conn


# -----------------------------------------------------------------------------
# 3.9 — Database File Rotation Guard
# -----------------------------------------------------------------------------
def has_day_changed(last_date: Optional[date]) -> bool:
    """
    Check if calendar day has changed.

    Used to trigger DB rotation logic in higher layers.

    Args:
        last_date: Previously recorded date

    Returns:
        bool
    """
    if last_date is None:
        return True
    return date.today() != last_date


# -----------------------------------------------------------------------------
# 3.10 — Database Metadata Snapshot
# -----------------------------------------------------------------------------
def get_db_metadata(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Retrieve basic database metadata.

    Returns:
        Dict with file path, size, and table list
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        db_path = conn.execute("PRAGMA database_list;").fetchone()["file"]
        size_bytes = os.path.getsize(db_path) if db_path else 0

        return {
            "path": db_path,
            "size_bytes": size_bytes,
            "tables": tables,
        }
    except Exception:
        return {
            "path": None,
            "size_bytes": 0,
            "tables": [],
        }


# -----------------------------------------------------------------------------
# 3.11 — Explicit Connection Close (Rare Use)
# -----------------------------------------------------------------------------
def close_connection(conn: sqlite3.Connection) -> None:
    """
    Explicitly close SQLite connection.
    Normally NOT required due to Streamlit lifecycle.

    Args:
        conn: sqlite3.Connection
    """
    try:
        conn.close()
    except Exception:
        pass


# =============================================================================
# END SECTION 3 — Database Connection
# =============================================================================

# =============================================================================
# SECTION 4 — Database Schema (Wide Table)
# =============================================================================
# PURPOSE:
# - Define the canonical SQLite WIDE TABLE schema
# - Ensure Excel-aligned column layout
# - Enforce consistent column naming and ordering
# - Support missing values and misaligned timestamps
#
# RULES:
# - NO data insertion
# - NO data selection
# - NO session_state access
# - NO Streamlit UI
# =============================================================================


# -----------------------------------------------------------------------------
# 4.1 — Canonical Wide Table Name
# -----------------------------------------------------------------------------
# This table stores ALL SWG data in ONE ROW per aligned index.
# -----------------------------------------------------------------------------
WIDE_TABLE_NAME: str = SQLITE_WIDE_TABLE_NAME


# -----------------------------------------------------------------------------
# 4.2 — Wide Table Column Order (CRITICAL)
# -----------------------------------------------------------------------------
# Column order MUST remain stable forever.
# This order matches Excel export EXACTLY.
# -----------------------------------------------------------------------------
WIDE_TABLE_COLUMN_ORDER: Tuple[str, ...] = (
    # ---- Primary Key ----
    "id",

    # ---- SWG1 ----
    "SWG1_DateTime",
    "SWG1_Active",
    "SWG1_Reactive",
    "SWG1_SOC",

    # ---- SWG2 ----
    "SWG2_DateTime",
    "SWG2_Active",
    "SWG2_Reactive",
    "SWG2_SOC",

    # ---- SWG3 ----
    "SWG3_DateTime",
    "SWG3_Active",
    "SWG3_Reactive",
    "SWG3_SOC",
)


# -----------------------------------------------------------------------------
# 4.3 — SQLite Column Type Mapping
# -----------------------------------------------------------------------------
# SQLite is typeless, but we enforce logical types explicitly.
# -----------------------------------------------------------------------------
WIDE_TABLE_COLUMN_TYPES: Dict[str, str] = {
    # Primary key
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",

    # SWG1
    "SWG1_DateTime": "TEXT",
    "SWG1_Active": "REAL",
    "SWG1_Reactive": "REAL",
    "SWG1_SOC": "REAL",

    # SWG2
    "SWG2_DateTime": "TEXT",
    "SWG2_Active": "REAL",
    "SWG2_Reactive": "REAL",
    "SWG2_SOC": "REAL",

    # SWG3
    "SWG3_DateTime": "TEXT",
    "SWG3_Active": "REAL",
    "SWG3_Reactive": "REAL",
    "SWG3_SOC": "REAL",
}


# -----------------------------------------------------------------------------
# 4.4 — Nullability Rules
# -----------------------------------------------------------------------------
# ALL measurement fields are nullable by design.
# This supports misaligned timestamps across SWGs.
# -----------------------------------------------------------------------------
WIDE_TABLE_NULLABLE_COLUMNS: Tuple[str, ...] = (
    "SWG1_DateTime", "SWG1_Active", "SWG1_Reactive", "SWG1_SOC",
    "SWG2_DateTime", "SWG2_Active", "SWG2_Reactive", "SWG2_SOC",
    "SWG3_DateTime", "SWG3_Active", "SWG3_Reactive", "SWG3_SOC",
)


# -----------------------------------------------------------------------------
# 4.5 — Wide Table CREATE SQL (Authoritative)
# -----------------------------------------------------------------------------
def build_create_wide_table_sql() -> str:
    """
    Build CREATE TABLE SQL for the wide energy table.

    Returns:
        str: CREATE TABLE statement
    """
    column_defs: List[str] = []

    for col in WIDE_TABLE_COLUMN_ORDER:
        col_type = WIDE_TABLE_COLUMN_TYPES[col]
        column_defs.append(f"{col} {col_type}")

    columns_sql = ",\n    ".join(column_defs)

    sql = f"""
    CREATE TABLE IF NOT EXISTS {WIDE_TABLE_NAME} (
        {columns_sql}
    );
    """
    return sql.strip()


# -----------------------------------------------------------------------------
# 4.6 — Execute Schema Creation
# -----------------------------------------------------------------------------
def ensure_wide_table_exists(conn: sqlite3.Connection) -> None:
    """
    Ensure the wide table exists in the database.

    Args:
        conn: sqlite3.Connection
    """
    sql = build_create_wide_table_sql()
    conn.execute(sql)


# -----------------------------------------------------------------------------
# 4.7 — Schema Introspection Utilities
# -----------------------------------------------------------------------------
def get_existing_table_columns(
    conn: sqlite3.Connection,
    table_name: str
) -> List[str]:
    """
    Get column names from an existing SQLite table.

    Args:
        conn: sqlite3.Connection
        table_name: str

    Returns:
        List[str]
    """
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    return [row["name"] for row in cursor.fetchall()]


# -----------------------------------------------------------------------------
# 4.8 — Schema Validation (Strict)
# -----------------------------------------------------------------------------
def validate_wide_table_schema(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    """
    Validate that the wide table schema matches expectations exactly.

    Returns:
        (is_valid, issues)
    """
    issues: List[str] = []

    try:
        existing_columns = get_existing_table_columns(conn, WIDE_TABLE_NAME)
    except sqlite3.Error:
        return False, ["Wide table does not exist"]

    # Check column order
    if existing_columns != list(WIDE_TABLE_COLUMN_ORDER):
        issues.append("Column order mismatch")

    # Check missing columns
    for col in WIDE_TABLE_COLUMN_ORDER:
        if col not in existing_columns:
            issues.append(f"Missing column: {col}")

    # Check unexpected columns
    for col in existing_columns:
        if col not in WIDE_TABLE_COLUMN_ORDER:
            issues.append(f"Unexpected column: {col}")

    return len(issues) == 0, issues


# -----------------------------------------------------------------------------
# 4.9 — Schema Repair Strategy (Non-Destructive)
# -----------------------------------------------------------------------------
def needs_schema_rebuild(conn: sqlite3.Connection) -> bool:
    """
    Determine if schema rebuild is required.

    Returns:
        bool
    """
    is_valid, _ = validate_wide_table_schema(conn)
    return not is_valid


# -----------------------------------------------------------------------------
# 4.10 — Canonical Empty Row Template
# -----------------------------------------------------------------------------
def build_empty_wide_row() -> Dict[str, Optional[Any]]:
    """
    Build an empty wide-table row with NULL values.

    Returns:
        Dict[str, Optional[Any]]
    """
    row: Dict[str, Optional[Any]] = {}

    for col in WIDE_TABLE_COLUMN_ORDER:
        if col == "id":
            continue
        row[col] = None

    return row


# -----------------------------------------------------------------------------
# 4.11 — Column Groupings (Semantic)
# -----------------------------------------------------------------------------
SWG1_COLUMNS: Tuple[str, ...] = (
    "SWG1_DateTime",
    "SWG1_Active",
    "SWG1_Reactive",
    "SWG1_SOC",
)

SWG2_COLUMNS: Tuple[str, ...] = (
    "SWG2_DateTime",
    "SWG2_Active",
    "SWG2_Reactive",
    "SWG2_SOC",
)

SWG3_COLUMNS: Tuple[str, ...] = (
    "SWG3_DateTime",
    "SWG3_Active",
    "SWG3_Reactive",
    "SWG3_SOC",
)


# -----------------------------------------------------------------------------
# 4.12 — Wide Table Integrity Rules (Documentation)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 4.13 — Defensive Assertions (Development Only)
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert len(WIDE_TABLE_COLUMN_ORDER) == len(WIDE_TABLE_COLUMN_TYPES)
    assert "id" in WIDE_TABLE_COLUMN_ORDER
    assert WIDE_TABLE_COLUMN_ORDER[0] == "id"


# -----------------------------------------------------------------------------
# 4.14 — Forward Compatibility Hooks
# -----------------------------------------------------------------------------
# These are intentionally unused placeholders.
# DO NOT REMOVE.
# -----------------------------------------------------------------------------
ALLOW_SCHEMA_EVOLUTION: bool = True
SCHEMA_VERSION: int = 1


# =============================================================================
# END SECTION 4 — Database Schema
# =============================================================================

# =============================================================================
# SECTION 5 — Repository: Save Logic
# =============================================================================
# PURPOSE:
# - Persist in-memory SWG tables into SQLite
# - Combine separate SWG tables into ONE wide table
# - Align by row index (Excel-style)
# - Allow missing values & misaligned timestamps
#
# RULES:
# - NO UI
# - NO st.button
# - NO loading logic
# - NO session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 5.1 — Safe Value Sanitization (DB-Level)
# -----------------------------------------------------------------------------
def sanitize_db_value(value: Any) -> Optional[Any]:
    """
    Sanitize value for SQLite insertion.

    Rules:
    - NaN → None
    - Timestamp → string (timezone removed)
    - Numeric → float
    - Other → str

    Args:
        value: Any Python object

    Returns:
        SQLite-safe value
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return (
            pd.to_datetime(value, errors="coerce")
            .tz_localize(None)
            .strftime(DB_DATETIME_FORMAT)
        )

    if isinstance(value, (np.integer, np.floating)):
        return float(value)

    return str(value)


# -----------------------------------------------------------------------------
# 5.2 — Normalize SWG DataFrame
# -----------------------------------------------------------------------------
def normalize_swg_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize SWG dataframe for wide-table merge.

    - Reset index
    - Ensure correct column order
    - Force DateTime column name
    - No mutation of original df

    Args:
        df: SWG dataframe from session_state

    Returns:
        Normalized dataframe
    """
    if df.empty:
        return df.copy()

    tmp = df.copy(deep=True).reset_index(drop=True)

    # Force canonical column names
    tmp.columns = [
        COLUMN_DATETIME,
        COLUMN_ACTIVE,
        COLUMN_REACTIVE,
        COLUMN_SOC,
    ]

    return tmp


# -----------------------------------------------------------------------------
# 5.3 — Build Wide DataFrame from Session State
# -----------------------------------------------------------------------------
def build_wide_dataframe(
    swg_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine SWG1 / SWG2 / SWG3 dataframes into one wide dataframe.

    Alignment:
    - Row index based
    - Max row count across all SWGs
    - Missing values allowed

    Args:
        swg_data: Dict of SWG dataframes

    Returns:
        Wide dataframe ready for DB insert
    """
    # Determine max length
    max_len = max(len(df) for df in swg_data.values())

    wide_frames: List[pd.DataFrame] = []

    for swg_id in SWG_IDS:
        df = normalize_swg_dataframe(swg_data.get(swg_id, pd.DataFrame()))

        # Reindex to max length
        df = df.reindex(range(max_len))

        # Rename columns to wide format
        wide_cols = SWG_TO_WIDE_COLUMNS[swg_id]
        df.columns = list(wide_cols)

        wide_frames.append(df)

    wide_df = pd.concat(wide_frames, axis=1)

    # Ensure column order EXACTLY matches schema (excluding id)
    wide_df = wide_df[list(WIDE_TABLE_COLUMN_ORDER[1:])]

    return wide_df


# -----------------------------------------------------------------------------
# 5.4 — Convert Wide DataFrame to DB Rows
# -----------------------------------------------------------------------------
def convert_wide_df_to_rows(wide_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert wide dataframe into list of DB-ready rows.

    Args:
        wide_df: Wide dataframe

    Returns:
        List of row dictionaries
    """
    rows: List[Dict[str, Any]] = []

    for _, record in wide_df.iterrows():
        row: Dict[str, Any] = {}

        for col in wide_df.columns:
            row[col] = sanitize_db_value(record[col])

        rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# 5.5 — Build INSERT SQL (Parameterized)
# -----------------------------------------------------------------------------
def build_insert_sql() -> str:
    """
    Build parameterized INSERT SQL for wide table.

    Returns:
        SQL string
    """
    columns = WIDE_TABLE_COLUMN_ORDER[1:]  # exclude id
    col_sql = ", ".join(columns)
    val_sql = ", ".join([f":{c}" for c in columns])

    return f"""
        INSERT INTO {WIDE_TABLE_NAME} ({col_sql})
        VALUES ({val_sql});
    """.strip()


# -----------------------------------------------------------------------------
# 5.6 — Execute Bulk Insert
# -----------------------------------------------------------------------------
def insert_rows(
    conn: sqlite3.Connection,
    rows: List[Dict[str, Any]]
) -> int:
    """
    Insert multiple rows into SQLite safely.

    Args:
        conn: SQLite connection
        rows: List of row dictionaries

    Returns:
        Number of rows inserted
    """
    if not rows:
        return 0

    sql = build_insert_sql()
    cursor = conn.cursor()
    cursor.executemany(sql, rows)

    return cursor.rowcount


# -----------------------------------------------------------------------------
# 5.7 — Public Save Entry Point
# -----------------------------------------------------------------------------
def save_session_state_to_database(
    session_state: Dict[str, Any],
    target_date: Optional[date] = None
) -> int:
    """
    Save current session_state SWG data into daily SQLite database.

    FLOW:
    1. Get DB connection
    2. Ensure schema exists
    3. Build wide dataframe
    4. Convert to rows
    5. Insert into SQLite

    Args:
        session_state: st.session_state
        target_date: Optional date override

    Returns:
        Number of rows saved
    """
    # Extract SWG data only
    swg_data: Dict[str, pd.DataFrame] = {
        swg: session_state.get(f"{swg}_data", pd.DataFrame())
        for swg in SWG_IDS
    }

    # Do not allow empty save unless explicitly allowed
    if not ALLOW_EMPTY_SAVE:
        if all(df.empty for df in swg_data.values()):
            return 0

    # Build wide dataframe
    wide_df = build_wide_dataframe(swg_data)

    # Convert to DB rows
    rows = convert_wide_df_to_rows(wide_df)

    # Get database connection
    conn = get_validated_connection(target_date)

    # Ensure schema
    ensure_wide_table_exists(conn)

    # Insert
    inserted = insert_rows(conn, rows)

    return inserted


# =============================================================================
# END SECTION 5 — Repository: Save Logic
# =============================================================================

# =============================================================================
# SECTION 6 — Repository: Load Logic
# =============================================================================
# PURPOSE:
# - Load persisted data from SQLite
# - Read WIDE table structure
# - Convert back to per-SWG tables
# - Prepare clean DataFrames for session_state
#
# RULES:
# - NO UI
# - NO Streamlit widgets
# - NO session_state mutation
# - NO saving logic
# =============================================================================


# -----------------------------------------------------------------------------
# 6.1 — Load Wide Table from SQLite
# -----------------------------------------------------------------------------
def load_wide_table_from_db(
    conn: sqlite3.Connection
) -> pd.DataFrame:
    """
    Load the entire wide table from SQLite.

    Args:
        conn: SQLite connection

    Returns:
        Wide table DataFrame (may be empty)
    """
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM {WIDE_TABLE_NAME} ORDER BY id ASC;",
            conn
        )
    except Exception:
        return pd.DataFrame(columns=WIDE_TABLE_COLUMN_ORDER)

    # Drop primary key (id) — not needed in session_state
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


# -----------------------------------------------------------------------------
# 6.2 — Parse Datetime Columns Safely
# -----------------------------------------------------------------------------
def parse_datetime_column(series: pd.Series) -> pd.Series:
    """
    Parse SQLite TEXT datetime column into timezone-aware pandas datetime.

    Rules:
    - Empty / invalid → NaT
    - Always attach LOCAL_TIMEZONE

    Args:
        series: Pandas Series

    Returns:
        Parsed datetime Series
    """
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.tz_localize(LOCAL_TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")


# -----------------------------------------------------------------------------
# 6.3 — Split Wide DataFrame into SWG Tables
# -----------------------------------------------------------------------------
def split_wide_dataframe(
    wide_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Split wide dataframe into SWG1 / SWG2 / SWG3 DataFrames.

    Args:
        wide_df: Wide DataFrame (no id column)

    Returns:
        Dict[str, DataFrame] keyed by SWG ID
    """
    result: Dict[str, pd.DataFrame] = {}

    for swg in SWG_IDS:
        dt_col, p_col, q_col, soc_col = SWG_TO_WIDE_COLUMNS[swg]

        # Extract SWG columns
        sub = wide_df[[dt_col, p_col, q_col, soc_col]].copy()

        # Rename to in-memory canonical names
        sub.columns = [
            COLUMN_DATETIME,
            COLUMN_ACTIVE,
            COLUMN_REACTIVE,
            COLUMN_SOC,
        ]

        # Parse datetime safely
        sub[COLUMN_DATETIME] = parse_datetime_column(sub[COLUMN_DATETIME])

        # Drop fully empty rows
        sub = sub.dropna(how="all").reset_index(drop=True)

        result[swg] = sub

    return result


# -----------------------------------------------------------------------------
# 6.4 — Public Load Entry Point
# -----------------------------------------------------------------------------
def load_session_state_from_database(
    target_date: Optional[date] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load SWG data from daily SQLite database.

    FLOW:
    1. Get DB connection
    2. Ensure wide table exists
    3. Load wide table
    4. Split into SWG tables

    Args:
        target_date: Optional date override

    Returns:
        Dict[str, DataFrame] ready for session_state assignment
    """
    conn = get_validated_connection(target_date)

    # Ensure schema exists (safe even if empty)
    ensure_wide_table_exists(conn)

    wide_df = load_wide_table_from_db(conn)

    if wide_df.empty:
        # Return empty templates
        return {
            swg: pd.DataFrame(columns=SWG_BASE_COLUMNS)
            for swg in SWG_IDS
        }

    return split_wide_dataframe(wide_df)


# -----------------------------------------------------------------------------
# 6.5 — Load Safety Check
# -----------------------------------------------------------------------------
def has_persisted_data(
    target_date: Optional[date] = None
) -> bool:
    """
    Check if today's database contains persisted data.

    Args:
        target_date: Optional date override

    Returns:
        True if data exists, False otherwise
    """
    conn = get_validated_connection(target_date)

    try:
        cursor = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM {WIDE_TABLE_NAME};"
        )
        return cursor.fetchone()["cnt"] > 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 6.6 — Defensive Assertions (DEV ONLY)
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert len(SWG_TO_WIDE_COLUMNS) == len(SWG_IDS)
    for swg in SWG_IDS:
        assert swg in SWG_TO_WIDE_COLUMNS


# =============================================================================
# END SECTION 6 — Repository: Load Logic
# =============================================================================

# =============================================================================
# SECTION 7 — Repository: Clear Logic
# =============================================================================
# PURPOSE:
# - Clear persisted data safely
# - Support full reset of daily database
# - Support in-memory session clearing (delegated to UI)
#
# RULES:
# - NO UI
# - NO Streamlit widgets
# - NO session_state mutation
# - NO save or load logic
# =============================================================================


# -----------------------------------------------------------------------------
# 7.1 — Clear Wide Table (SQLite Only)
# -----------------------------------------------------------------------------
def clear_wide_table(
    conn: sqlite3.Connection
) -> int:
    """
    Clear all rows from the wide table in SQLite.

    This operation:
    - Deletes ALL persisted rows for the current day
    - Preserves table schema
    - Does NOT drop the table

    Args:
        conn: SQLite connection

    Returns:
        Number of rows deleted
    """
    try:
        cursor = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM {WIDE_TABLE_NAME};"
        )
        count = cursor.fetchone()["cnt"]

        conn.execute(f"DELETE FROM {WIDE_TABLE_NAME};")

        return int(count)
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# 7.2 — Drop Wide Table (Schema Reset)
# -----------------------------------------------------------------------------
def drop_wide_table(
    conn: sqlite3.Connection
) -> bool:
    """
    Drop the wide table completely.

    ⚠️ EXTREME OPERATION ⚠️
    - Removes schema
    - Removes all data
    - Requires schema recreation after

    Args:
        conn: SQLite connection

    Returns:
        True if successful, False otherwise
    """
    try:
        conn.execute(f"DROP TABLE IF EXISTS {WIDE_TABLE_NAME};")
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 7.3 — Clear Daily Database File (Physical File)
# -----------------------------------------------------------------------------
def delete_daily_database_file(
    target_date: Optional[date] = None
) -> bool:
    """
    Delete the entire daily SQLite database file.

    ⚠️ EXTREME OPERATION ⚠️
    - Deletes the physical .db file
    - All data is permanently lost
    - New DB will be recreated automatically on next access

    Args:
        target_date: Optional date override

    Returns:
        True if deleted, False otherwise
    """
    try:
        db_path = get_today_db_path(target_date)

        if os.path.exists(db_path):
            os.remove(db_path)
            return True

        return False
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 7.4 — Public Clear Entry Point (Safe Mode)
# -----------------------------------------------------------------------------
def clear_persisted_data(
    *,
    clear_db_rows: bool = True,
    drop_schema: bool = False,
    delete_db_file: bool = False,
    target_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Clear persisted data using controlled options.

    DEFAULT BEHAVIOR (SAFE):
    - Clears rows only
    - Preserves schema
    - Preserves database file

    Args:
        clear_db_rows: Delete rows from wide table
        drop_schema: Drop wide table
        delete_db_file: Delete database file entirely
        target_date: Optional date override

    Returns:
        Result summary dict
    """
    result: Dict[str, Any] = {
        "rows_deleted": 0,
        "schema_dropped": False,
        "file_deleted": False,
    }

    # Handle file deletion first (most destructive)
    if delete_db_file:
        result["file_deleted"] = delete_daily_database_file(target_date)
        return result

    # Otherwise operate within SQLite
    conn = get_validated_connection(target_date)

    if drop_schema:
        result["schema_dropped"] = drop_wide_table(conn)
        return result

    if clear_db_rows:
        result["rows_deleted"] = clear_wide_table(conn)

    return result


# -----------------------------------------------------------------------------
# 7.5 — Session Reset Template (Returned, NOT Applied)
# -----------------------------------------------------------------------------
def build_empty_session_payload() -> Dict[str, pd.DataFrame]:
    """
    Build empty SWG dataframes for session reset.

    This function DOES NOT touch session_state.
    UI layer decides when and how to apply it.

    Returns:
        Dict[str, empty DataFrame]
    """
    return {
        swg: pd.DataFrame(columns=SWG_BASE_COLUMNS)
        for swg in SWG_IDS
    }


# -----------------------------------------------------------------------------
# 7.6 — Defensive Guards (DEV Only)
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    # Prevent accidental destructive defaults
    assert CONFIRM_CLEAR_REQUIRED is True
    assert ALLOW_EMPTY_SAVE in (True, False)


# =============================================================================
# END SECTION 7 — Repository: Clear Logic
# =============================================================================

# =============================================================================
# SECTION 8 — Session State Initialization
# =============================================================================
# PURPOSE:
# - Initialize st.session_state in a controlled manner
# - Restore persisted data on page refresh
# - Prepare Undo / Redo infrastructure
# - Ensure reload safety (no duplicate loads)
#
# RULES:
# - NO UI rendering
# - NO Streamlit widgets
# - NO database writes
# =============================================================================


# -----------------------------------------------------------------------------
# 8.1 — Required Session Keys (Authoritative List)
# -----------------------------------------------------------------------------
SESSION_KEYS_CORE: Tuple[str, ...] = (
    "initialized",
    "last_loaded_date",
    "history",
    "redo_stack",
    "table_locked",
)

SESSION_KEYS_SWG: Tuple[str, ...] = tuple(
    f"{swg}_data" for swg in SWG_IDS
)

SESSION_KEYS_EDITING: Tuple[str, ...] = (
    "edit_session_active",
    "edit_buffer",
)


# -----------------------------------------------------------------------------
# 8.2 — Initialize Core Session Keys
# -----------------------------------------------------------------------------
def init_core_session_state() -> None:
    """
    Initialize core session_state keys.

    These keys must exist regardless of data presence.
    """
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "last_loaded_date" not in st.session_state:
        st.session_state.last_loaded_date = None

    if "history" not in st.session_state:
        st.session_state.history = []

    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []

    if "table_locked" not in st.session_state:
        st.session_state.table_locked = False


# -----------------------------------------------------------------------------
# 8.3 — Initialize SWG DataFrames
# -----------------------------------------------------------------------------
def init_swg_tables(empty_payload: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Initialize SWG data tables in session_state.

    Args:
        empty_payload: Optional payload from DB load or clear logic
    """
    for swg in SWG_IDS:
        key = f"{swg}_data"

        if key not in st.session_state:
            if empty_payload and swg in empty_payload:
                st.session_state[key] = empty_payload[swg]
            else:
                st.session_state[key] = pd.DataFrame(columns=SWG_BASE_COLUMNS)


# -----------------------------------------------------------------------------
# 8.4 — Initialize Editing State
# -----------------------------------------------------------------------------
def init_editing_state() -> None:
    """
    Initialize edit-related session keys.
    """
    if "edit_session_active" not in st.session_state:
        st.session_state.edit_session_active = False

    if "edit_buffer" not in st.session_state:
        st.session_state.edit_buffer = {}


# -----------------------------------------------------------------------------
# 8.5 — Load Persisted Data (Once Per Day)
# -----------------------------------------------------------------------------
def load_persisted_data_if_needed() -> None:
    """
    Load data from daily SQLite DB into session_state.

    SAFETY RULES:
    - Load ONLY once per calendar day
    - Never overwrite existing session data
    - Never load twice on reruns
    """
    today = date.today()

    if st.session_state.initialized:
        return

    if st.session_state.last_loaded_date == today:
        st.session_state.initialized = True
        return

    if has_persisted_data(today):
        loaded_payload = load_session_state_from_database(today)
        init_swg_tables(loaded_payload)
    else:
        init_swg_tables()

    st.session_state.last_loaded_date = today
    st.session_state.initialized = True


# -----------------------------------------------------------------------------
# 8.6 — Reset Undo / Redo Stacks
# -----------------------------------------------------------------------------
def reset_history() -> None:
    """
    Reset Undo / Redo stacks.
    """
    st.session_state.history.clear()
    st.session_state.redo_stack.clear()


# -----------------------------------------------------------------------------
# 8.7 — Master Initialization Entry Point
# -----------------------------------------------------------------------------
def initialize_session_state() -> None:
    """
    Master initialization function.

    This MUST be called ONCE at the top of the app
    before any UI or logic execution.
    """
    init_core_session_state()
    init_editing_state()
    load_persisted_data_if_needed()
    reset_history()


# -----------------------------------------------------------------------------
# 8.8 — Execute Initialization (SAFE)
# -----------------------------------------------------------------------------
initialize_session_state()


# -----------------------------------------------------------------------------
# 8.9 — Defensive Assertions (DEV Only)
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    for swg in SWG_IDS:
        assert f"{swg}_data" in st.session_state
    assert isinstance(st.session_state.history, list)
    assert isinstance(st.session_state.redo_stack, list)


# =============================================================================
# END SECTION 8 — Session State Initialization
# =============================================================================

# =============================================================================
# SECTION 9 — Validation Functions
# =============================================================================
# PURPOSE:
# - Centralize ALL validation logic
# - Enforce numeric, datetime, and schema rules
# - Ensure data integrity before edit, save, analysis
#
# RULES:
# - NO UI
# - NO Streamlit widgets
# - NO database access
# - NO session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 9.1 — Validation Result Model
# -----------------------------------------------------------------------------
def validation_result(
    valid: bool,
    errors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build a standardized validation result object.

    Args:
        valid: Validation status
        errors: Optional list of error messages

    Returns:
        Dict[str, Any]
    """
    return {
        "valid": bool(valid),
        "errors": errors or [],
    }


# -----------------------------------------------------------------------------
# 9.2 — Primitive Validators
# -----------------------------------------------------------------------------
def is_required(value: Any) -> bool:
    """Check if value is not None / NaN."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return True


def is_number(value: Any) -> bool:
    """Check if value is numeric."""
    try:
        float(value)
        return True
    except Exception:
        return False


def in_range(value: float, min_v: float, max_v: float) -> bool:
    """Check numeric range."""
    return min_v <= value <= max_v


# -----------------------------------------------------------------------------
# 9.3 — Datetime Validator
# -----------------------------------------------------------------------------
def validate_datetime(value: Any) -> validation_result:
    """
    Validate datetime value.

    Rules:
    - Must be convertible to pandas datetime
    - Timezone-aware after parsing

    Args:
        value: Any

    Returns:
        Validation result
    """
    try:
        dt = pd.to_datetime(value, errors="raise")

        if dt.tzinfo is None:
            dt = dt.tz_localize(LOCAL_TIMEZONE)

        return validation_result(True)
    except Exception:
        return validation_result(False, [ERROR_DATETIME_INVALID])


# -----------------------------------------------------------------------------
# 9.4 — Metric Validators (Single Value)
# -----------------------------------------------------------------------------
def validate_active_power(value: Any) -> validation_result:
    if not is_required(value):
        return validation_result(False, [ERROR_REQUIRED_FIELD])

    if not is_number(value):
        return validation_result(False, [ERROR_ACTIVE_RANGE])

    v = float(value)
    if not in_range(v, ACTIVE_POWER_MIN, ACTIVE_POWER_MAX):
        return validation_result(False, [ERROR_ACTIVE_RANGE])

    return validation_result(True)


def validate_reactive_power(value: Any) -> validation_result:
    if not is_required(value):
        return validation_result(False, [ERROR_REQUIRED_FIELD])

    if not is_number(value):
        return validation_result(False, [ERROR_REACTIVE_RANGE])

    v = float(value)
    if not in_range(v, REACTIVE_POWER_MIN, REACTIVE_POWER_MAX):
        return validation_result(False, [ERROR_REACTIVE_RANGE])

    return validation_result(True)


def validate_soc(value: Any) -> validation_result:
    if not is_required(value):
        return validation_result(False, [ERROR_REQUIRED_FIELD])

    if not is_number(value):
        return validation_result(False, [ERROR_SOC_RANGE])

    v = float(value)
    if not in_range(v, SOC_MIN, SOC_MAX):
        return validation_result(False, [ERROR_SOC_RANGE])

    return validation_result(True)


# -----------------------------------------------------------------------------
# 9.5 — Validate Complete SWG Row
# -----------------------------------------------------------------------------
def validate_swg_row(row: Dict[str, Any]) -> validation_result:
    """
    Validate one SWG row.

    Expected keys:
    - DateTime
    - Active Power (MW)
    - Reactive Power (Mvar)
    - SOC (%)

    Args:
        row: Dict representing one row

    Returns:
        Validation result
    """
    errors: List[str] = []

    dt_res = validate_datetime(row.get(COLUMN_DATETIME))
    if not dt_res["valid"]:
        errors.extend(dt_res["errors"])

    ap_res = validate_active_power(row.get(COLUMN_ACTIVE))
    if not ap_res["valid"]:
        errors.extend(ap_res["errors"])

    rp_res = validate_reactive_power(row.get(COLUMN_REACTIVE))
    if not rp_res["valid"]:
        errors.extend(rp_res["errors"])

    soc_res = validate_soc(row.get(COLUMN_SOC))
    if not soc_res["valid"]:
        errors.extend(soc_res["errors"])

    return validation_result(len(errors) == 0, errors)


# -----------------------------------------------------------------------------
# 9.6 — Validate Entire SWG DataFrame
# -----------------------------------------------------------------------------
def validate_swg_dataframe(df: pd.DataFrame) -> validation_result:
    """
    Validate an entire SWG dataframe.

    Rules:
    - Required columns must exist
    - Each row must be valid

    Args:
        df: SWG DataFrame

    Returns:
        Validation result
    """
    errors: List[str] = []

    # Column validation
    for col in SWG_BASE_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return validation_result(False, errors)

    # Row validation
    for idx, row in df.iterrows():
        res = validate_swg_row(row.to_dict())
        if not res["valid"]:
            for err in res["errors"]:
                errors.append(f"Row {idx}: {err}")

    return validation_result(len(errors) == 0, errors)


# -----------------------------------------------------------------------------
# 9.7 — Validate Wide DataFrame Before Save
# -----------------------------------------------------------------------------
def validate_wide_dataframe(wide_df: pd.DataFrame) -> validation_result:
    """
    Validate wide dataframe before DB save.

    Rules:
    - Must contain all required columns
    - Allows missing values
    - Column order enforced

    Args:
        wide_df: Wide DataFrame

    Returns:
        Validation result
    """
    errors: List[str] = []

    expected_cols = list(WIDE_TABLE_COLUMN_ORDER[1:])  # exclude id

    if list(wide_df.columns) != expected_cols:
        errors.append("Wide table column order mismatch")

    return validation_result(len(errors) == 0, errors)


# -----------------------------------------------------------------------------
# 9.8 — Aggregate Validation (All SWGs)
# -----------------------------------------------------------------------------
def validate_all_swg_data(
    swg_data: Dict[str, pd.DataFrame]
) -> validation_result:
    """
    Validate all SWG tables together.

    Args:
        swg_data: Dict of SWG DataFrames

    Returns:
        Validation result
    """
    errors: List[str] = []

    for swg, df in swg_data.items():
        res = validate_swg_dataframe(df)
        if not res["valid"]:
            for err in res["errors"]:
                errors.append(f"{swg}: {err}")

    return validation_result(len(errors) == 0, errors)


# -----------------------------------------------------------------------------
# 9.9 — Defensive Assertions (DEV Only)
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    # Sanity checks
    assert ACTIVE_POWER_MIN < ACTIVE_POWER_MAX
    assert REACTIVE_POWER_MIN < REACTIVE_POWER_MAX
    assert SOC_MIN < SOC_MAX
    assert len(SWG_BASE_COLUMNS) == 4


# =============================================================================
# END SECTION 9 — Validation Functions
# =============================================================================

# =============================================================================
# SECTION 10 — Undo / Redo Core
# =============================================================================
# PURPOSE:
# - Centralize Undo / Redo logic
# - Ensure atomic state restoration
# - Protect session_state from corruption
#
# RULES:
# - NO UI
# - NO database access
# - ONLY session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 10.1 — Snapshot Model
# -----------------------------------------------------------------------------
def build_snapshot() -> Dict[str, pd.DataFrame]:
    """
    Build a deep snapshot of all SWG tables.

    Returns:
        Dict[str, DataFrame] snapshot
    """
    snapshot: Dict[str, pd.DataFrame] = {}

    for swg in SWG_IDS:
        key = f"{swg}_data"
        snapshot[swg] = st.session_state[key].copy(deep=True)

    return snapshot


# -----------------------------------------------------------------------------
# 10.2 — Push Snapshot to History Stack
# -----------------------------------------------------------------------------
def push_undo_snapshot() -> None:
    """
    Push current state to undo history.

    Rules:
    - Clears redo stack
    - Caps history length
    """
    snapshot = build_snapshot()

    st.session_state.history.append(snapshot)
    st.session_state.redo_stack.clear()

    # Enforce max history length
    if len(st.session_state.history) > MAX_HISTORY_LENGTH:
        st.session_state.history.pop(0)


# -----------------------------------------------------------------------------
# 10.3 — Restore Snapshot
# -----------------------------------------------------------------------------
def restore_snapshot(snapshot: Dict[str, pd.DataFrame]) -> None:
    """
    Restore snapshot atomically.

    Args:
        snapshot: Snapshot dictionary
    """
    for swg, df in snapshot.items():
        st.session_state[f"{swg}_data"] = df.copy(deep=True)


# -----------------------------------------------------------------------------
# 10.4 — Undo Operation
# -----------------------------------------------------------------------------
def undo_action() -> bool:
    """
    Perform Undo operation.

    Returns:
        True if successful, False otherwise
    """
    if not st.session_state.history:
        return False

    # Save current state to redo
    current = build_snapshot()
    st.session_state.redo_stack.append(current)

    # Restore previous state
    snapshot = st.session_state.history.pop()
    restore_snapshot(snapshot)

    return True


# -----------------------------------------------------------------------------
# 10.5 — Redo Operation
# -----------------------------------------------------------------------------
def redo_action() -> bool:
    """
    Perform Redo operation.

    Returns:
        True if successful, False otherwise
    """
    if not st.session_state.redo_stack:
        return False

    # Save current state to undo
    current = build_snapshot()
    st.session_state.history.append(current)

    # Restore redo state
    snapshot = st.session_state.redo_stack.pop()
    restore_snapshot(snapshot)

    return True


# -----------------------------------------------------------------------------
# 10.6 — Guarded Mutation Helper
# -----------------------------------------------------------------------------
def apply_mutation(mutation_fn) -> bool:
    """
    Apply a mutation safely with undo support.

    Usage:
        apply_mutation(lambda: do_something())

    Args:
        mutation_fn: Callable with no arguments

    Returns:
        True if mutation applied, False otherwise
    """
    try:
        push_undo_snapshot()
        mutation_fn()
        return True
    except Exception:
        # Rollback immediately
        undo_action()
        return False


# -----------------------------------------------------------------------------
# 10.7 — History State Inspection
# -----------------------------------------------------------------------------
def can_undo() -> bool:
    """Check if Undo is available."""
    return len(st.session_state.history) > 0


def can_redo() -> bool:
    """Check if Redo is available."""
    return len(st.session_state.redo_stack) > 0


# -----------------------------------------------------------------------------
# 10.8 — Clear History (After Load / Clear)
# -----------------------------------------------------------------------------
def clear_history() -> None:
    """
    Clear Undo / Redo stacks.

    Used after:
    - DB load
    - Full clear
    """
    st.session_state.history.clear()
    st.session_state.redo_stack.clear()


# -----------------------------------------------------------------------------
# 10.9 — Defensive Assertions (DEV Only)
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert isinstance(st.session_state.history, list)
    assert isinstance(st.session_state.redo_stack, list)


# =============================================================================
# END SECTION 10 — Undo / Redo Core
# =============================================================================

# =============================================================================
# SECTION 11 — Page Layout & CSS
# =============================================================================
# PURPOSE:
# - Define global UI shell
# - Apply enterprise-grade styling
# - Standardize layout, spacing, colors, typography
#
# RULES:
# - UI ONLY
# - NO business logic
# - NO database access
# - NO session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 11.1 — Remove Streamlit Default Chrome
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    div[data-testid="stToolbar"] { display: none; }
    #MainMenu { display: none; }
    section.main > div { padding-top: 0rem; }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.2 — CSS Variables (Design Tokens)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {

        /* ===========================
           COLOR SYSTEM
           =========================== */
        --color-bg-main: #f8fafc;
        --color-bg-card: #ffffff;
        --color-bg-soft: #f1f5f9;

        --color-primary: #1e3a8a;
        --color-primary-light: #2563eb;
        --color-primary-dark: #1e40af;

        --color-accent-green: #16a34a;
        --color-accent-red: #dc2626;
        --color-accent-yellow: #facc15;

        --color-text-main: #0f172a;
        --color-text-muted: #64748b;
        --color-text-light: #e5e7eb;

        --color-border: #e5e7eb;

        /* ===========================
           TYPOGRAPHY
           =========================== */
        --font-main: "Segoe UI", Inter, system-ui, sans-serif;

        --font-size-xs: 11px;
        --font-size-sm: 13px;
        --font-size-md: 15px;
        --font-size-lg: 18px;
        --font-size-xl: 24px;
        --font-size-xxl: 36px;

        /* ===========================
           SPACING
           =========================== */
        --space-xs: 4px;
        --space-sm: 8px;
        --space-md: 16px;
        --space-lg: 24px;
        --space-xl: 32px;

        /* ===========================
           RADIUS
           =========================== */
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 18px;

        /* ===========================
           SHADOWS
           =========================== */
        --shadow-sm: 0 2px 6px rgba(0,0,0,0.06);
        --shadow-md: 0 8px 22px rgba(0,0,0,0.12);
        --shadow-lg: 0 18px 36px rgba(0,0,0,0.18);

        /* ===========================
           TRANSITIONS
           =========================== */
        --transition-fast: 0.15s ease;
        --transition-normal: 0.3s ease;
        --transition-slow: 0.6s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.3 — Base Page Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body {
        background-color: var(--color-bg-main);
        font-family: var(--font-main);
        color: var(--color-text-main);
    }

    * {
        box-sizing: border-box;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.4 — Global Containers
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .page-container {
        max-width: 1600px;
        margin: auto;
        padding: var(--space-lg);
    }

    .section {
        margin-bottom: var(--space-xl);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.5 — Header Components
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .dashboard-header {
        background: linear-gradient(135deg,
            var(--color-primary),
            var(--color-primary-light)
        );
        color: white;
        padding: var(--space-xl);
        border-radius: var(--radius-lg);
        font-size: var(--font-size-xxl);
        font-weight: 900;
        box-shadow: var(--shadow-lg);
        margin-bottom: var(--space-lg);
    }

    .dashboard-subtitle {
        font-size: var(--font-size-md);
        opacity: 0.9;
        margin-top: var(--space-sm);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.6 — Card System
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: var(--color-bg-card);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        box-shadow: var(--shadow-md);
        margin-bottom: var(--space-lg);
        transition: transform var(--transition-fast),
                    box-shadow var(--transition-fast);
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .card.soft {
        background-color: var(--color-bg-soft);
    }

    .card-title {
        font-size: var(--font-size-lg);
        font-weight: 800;
        color: var(--color-primary);
        margin-bottom: var(--space-md);
    }

    .card-subtitle {
        font-size: var(--font-size-sm);
        color: var(--color-text-muted);
        margin-bottom: var(--space-md);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.7 — Grid Layout Utilities
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .grid {
        display: grid;
        gap: var(--space-lg);
    }

    .grid-2 {
        grid-template-columns: repeat(2, 1fr);
    }

    .grid-3 {
        grid-template-columns: repeat(3, 1fr);
    }

    .grid-4 {
        grid-template-columns: repeat(4, 1fr);
    }

    @media (max-width: 1200px) {
        .grid-3, .grid-4 {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 700px) {
        .grid-2, .grid-3, .grid-4 {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.8 — Buttons Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    button[kind="primary"] {
        background-color: var(--color-primary-light);
        border-radius: var(--radius-md);
        font-weight: 700;
        font-size: var(--font-size-md);
        height: 44px;
        transition: all var(--transition-fast);
    }

    button[kind="primary"]:hover {
        background-color: var(--color-primary-dark);
        transform: translateY(-1px);
    }

    button[kind="secondary"] {
        border-radius: var(--radius-md);
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.9 — Table Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    thead tr th {
        background-color: var(--color-bg-soft);
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.10 — Status Colors
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .status-success { color: var(--color-accent-green); }
    .status-warning { color: var(--color-accent-yellow); }
    .status-error   { color: var(--color-accent-red); }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.11 — Animations
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.12 — Scrollbar Customization
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--color-primary-light);
        border-radius: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--color-bg-soft);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 11.13 — Utility Classes
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .text-center { text-align: center; }
    .text-right  { text-align: right; }
    .text-muted  { color: var(--color-text-muted); }

    .mt-sm { margin-top: var(--space-sm); }
    .mt-md { margin-top: var(--space-md); }
    .mt-lg { margin-top: var(--space-lg); }

    .mb-sm { margin-bottom: var(--space-sm); }
    .mb-md { margin-bottom: var(--space-md); }
    .mb-lg { margin-bottom: var(--space-lg); }
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# END SECTION 11 — Page Layout & CSS
# =============================================================================

# =============================================================================
# GLOBAL HEADER — TITLE + COMPANY + LIVE CLOCK
# =============================================================================

st.markdown(
    f"""
    <div class="dashboard-header">
        Power Dispatch Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

html(
    f"""
    <div style="
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 14px;
        padding: 12px 22px;
        margin-top: 10px;
        margin-bottom: 24px;
        font-size: 15px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    ">
        <span>🏢 <strong>SchneiTech Group</strong></span>
        <span>|</span>
        <span style="color:#22c55e;">🟢</span>
        <span id="date-text"></span>
        <span>|</span>
        <span>🕒 <span id="time-text"></span></span>
    </div>

    <script>
    function updateClockBar() {{
        const now = new Date();

        document.getElementById("date-text").innerHTML =
            now.toLocaleDateString(undefined, {{
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            }});

        document.getElementById("time-text").innerHTML =
            now.toLocaleTimeString(undefined, {{
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: true
            }});
    }}

    setInterval(updateClockBar, 1000);
    updateClockBar();
    </script>
    """,
    height=56
)
# =============================================================================
# UNDO / REDO CONTROLS (GLOBAL)
# =============================================================================

uc, rc = st.columns(2)

with uc:
    if st.button(
        "↩ Undo",
        use_container_width=True,
        disabled=not can_undo()
    ):
        if undo_action():
            st.success("Undo applied")

with rc:
    if st.button(
        "↪ Redo",
        use_container_width=True,
        disabled=not can_redo()
    ):
        if redo_action():
            st.success("Redo applied")

# =============================================================================
# SECTION 12 — Input Panels (SWG1 / SWG2 / SWG3)
# =============================================================================
# PURPOSE:
# - Clean, empty inputs (no auto-filled values)
# - Safe submission using Streamlit forms
# - Centralized validation
# - Undo / Redo protected mutations
#
# RULES:
# - UI allowed
# - NO database access
# - NO direct session_state mutation for widgets
# =============================================================================


# -----------------------------------------------------------------------------
# 12.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">📥 Real-Time Input</div>
        <div class="card-subtitle">
            Enter operational values for each SWG unit.
            Inputs start empty and reset automatically after submission.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 12.2 — Helper: Insert Row (Undo / Redo Safe)
# -----------------------------------------------------------------------------
def insert_swg_row(
    swg: str,
    active: float,
    reactive: float,
    soc: float,
) -> None:
    """
    Insert one validated row into an SWG table
    using Undo / Redo protection.
    """
    def mutation():
        df = st.session_state[f"{swg}_data"]

        new_row = {
            COLUMN_DATETIME: datetime.now(tz=LOCAL_TIMEZONE),
            COLUMN_ACTIVE: float(active),
            COLUMN_REACTIVE: float(reactive),
            COLUMN_SOC: float(soc),
        }

        st.session_state[f"{swg}_data"] = pd.concat(
            [df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    apply_mutation(mutation)


# -----------------------------------------------------------------------------
# 12.3 — Render SWG Input Panels (FORM-BASED)
# -----------------------------------------------------------------------------
cols = st.columns(len(SWG_IDS))

for idx, swg in enumerate(SWG_IDS):

    with cols[idx]:
        swg_label = swg.replace("SWG", "SWG-")

        st.markdown(
            f"""
            <div class="card fade-in">
                <div class="card-title">⚙ {swg_label}</div>
                <div class="card-subtitle">
                    Manual data entry
                </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------------------
        # FORM (KEY PART — PREVENTS STATE ERRORS)
        # -------------------------------------------------------------
        with st.form(
            key=f"{swg}_input_form",
            clear_on_submit=True
        ):
            # Active Power (EMPTY BY DEFAULT)
            active = st.number_input(
                LABEL_ACTIVE,
                value=None,
                step=FLOAT_STEP_DEFAULT,
                format=f"%.{FLOAT_PRECISION_UI}f",
                disabled=st.session_state.table_locked,
            )

            # Reactive Power (EMPTY BY DEFAULT)
            reactive = st.number_input(
                LABEL_REACTIVE,
                value=None,
                step=FLOAT_STEP_DEFAULT,
                format=f"%.{FLOAT_PRECISION_UI}f",
                disabled=st.session_state.table_locked,
            )

            # SOC (EMPTY BY DEFAULT)
            soc = st.number_input(
                LABEL_SOC,
                value=None,
                step=FLOAT_STEP_DEFAULT,
                format=f"%.{FLOAT_PRECISION_UI}f",
                disabled=st.session_state.table_locked,
            )

            submitted = st.form_submit_button(
                f"➕ Add Entry ({swg_label})",
                use_container_width=True,
                disabled=st.session_state.table_locked,
            )

            # ---------------------------------------------------------
            # SUBMIT HANDLER
            # ---------------------------------------------------------
            if submitted:
                row = {
                    COLUMN_DATETIME: datetime.now(tz=LOCAL_TIMEZONE),
                    COLUMN_ACTIVE: active,
                    COLUMN_REACTIVE: reactive,
                    COLUMN_SOC: soc,
                }

                result = validate_swg_row(row)

                if not result["valid"]:
                    for err in result["errors"]:
                        st.error(err)
                else:
                    insert_swg_row(swg, active, reactive, soc)
                    st.success("Data added successfully")

        # -------------------------------------------------------------
        # LOCK INDICATOR
        # -------------------------------------------------------------
        if st.session_state.table_locked:
            st.warning("🔒 Table is locked")

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# END SECTION 12 — Input Panels
# =============================================================================


# =============================================================================
# SECTION 13 — Live Table Preview
# =============================================================================
# PURPOSE:
# - Provide real-time visibility of SWG data
# - Display per-SWG tables
# - Display combined WIDE table preview
# - Read-only, reactive, safe
#
# RULES:
# - NO data mutation
# - NO database access
# - NO session_state writes
# =============================================================================


# -----------------------------------------------------------------------------
# 13.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">📊 Live Data Preview</div>
        <div class="card-subtitle">
            Real-time view of operational data.
            All tables update instantly as data changes.
            Preview is read-only and safe.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 13.2 — Display Options
# -----------------------------------------------------------------------------
preview_mode = st.radio(
    "Preview Mode",
    options=[
        "Per SWG (Individual Tables)",
        "Combined Wide Table (Excel View)",
        "Both",
    ],
    horizontal=True,
    index=2,
)

show_row_count = st.toggle("Show Row Count", value=True)
show_empty_notice = st.toggle("Show Empty Table Notices", value=True)


# -----------------------------------------------------------------------------
# 13.3 — Helper: Format DateTime for Display
# -----------------------------------------------------------------------------
def format_datetime_for_preview(series: pd.Series) -> pd.Series:
    """
    Format datetime column for UI preview.

    - Converts to pandas datetime
    - Removes timezone
    - Applies DISPLAY_DATETIME_FORMAT
    """
    if series.empty:
        return series

    try:
        return (
            pd.to_datetime(series, errors="coerce")
            .dt.tz_localize(None)
            .dt.strftime(DISPLAY_DATETIME_FORMAT)
        )
    except Exception:
        return series.astype(str)


# -----------------------------------------------------------------------------
# 13.4 — Helper: Prepare SWG Preview DataFrame
# -----------------------------------------------------------------------------
def prepare_swg_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare SWG dataframe for read-only preview.

    - Copy only
    - Format datetime
    - Preserve column order
    """
    if df.empty:
        return df.copy()

    tmp = df.copy(deep=True)

    tmp[COLUMN_DATETIME] = format_datetime_for_preview(
        tmp[COLUMN_DATETIME]
    )

    return tmp


# -----------------------------------------------------------------------------
# 13.5 — Helper: Build Combined Wide Preview (NO DB)
# -----------------------------------------------------------------------------
def build_live_wide_preview(
    swg_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build combined WIDE preview directly from session_state.

    - Align by index
    - Allow missing values
    - Match Excel layout exactly
    """
    max_len = max(len(df) for df in swg_data.values())

    frames: List[pd.DataFrame] = []

    for swg in SWG_IDS:
        df = swg_data[swg].copy()
        df = df.reset_index(drop=True)
        df = df.reindex(range(max_len))

        # Format datetime
        if not df.empty:
            df[COLUMN_DATETIME] = format_datetime_for_preview(
                df[COLUMN_DATETIME]
            )

        # Rename columns
        wide_cols = SWG_TO_WIDE_COLUMNS[swg]
        df.columns = list(wide_cols)

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=WIDE_TABLE_COLUMN_ORDER[1:])

    wide_df = pd.concat(frames, axis=1)
    return wide_df


# -----------------------------------------------------------------------------
# 13.6 — Per-SWG Preview
# -----------------------------------------------------------------------------
def render_per_swg_preview() -> None:
    """
    Render individual SWG tables side-by-side.
    """
    st.markdown("### 🔍 Individual SWG Tables")

    cols = st.columns(len(SWG_IDS))

    for idx, swg in enumerate(SWG_IDS):
        with cols[idx]:
            df = st.session_state[f"{swg}_data"]

            st.markdown(
                f"**{swg.replace('SWG', 'SWG-')}**"
            )

            if df.empty:
                if show_empty_notice:
                    st.info("No data available.")
                continue

            preview_df = prepare_swg_preview(df)

            if show_row_count:
                st.caption(f"Rows: {len(preview_df)}")

            st.dataframe(
                preview_df,
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------------------------------------------------------
# 13.7 — Combined Wide Preview
# -----------------------------------------------------------------------------
def render_wide_preview() -> None:
    """
    Render combined WIDE table preview.
    """
    st.markdown("### 🧾 Combined Wide Table (Excel Style)")

    swg_data = {
        swg: st.session_state[f"{swg}_data"]
        for swg in SWG_IDS
    }

    if all(df.empty for df in swg_data.values()):
        if show_empty_notice:
            st.info("No data available in any SWG.")
        return

    wide_df = build_live_wide_preview(swg_data)

    if show_row_count:
        st.caption(f"Rows: {len(wide_df)}")

    st.dataframe(
        wide_df,
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# 13.8 — Conditional Rendering
# -----------------------------------------------------------------------------
if preview_mode == "Per SWG (Individual Tables)":
    render_per_swg_preview()

elif preview_mode == "Combined Wide Table (Excel View)":
    render_wide_preview()

else:
    render_per_swg_preview()
    st.markdown("---")
    render_wide_preview()


# -----------------------------------------------------------------------------
# 13.9 — Data Integrity Summary
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card soft">
        <div class="card-title">ℹ️ Preview Notes</div>
        <ul style="color: var(--color-text-muted); font-size: 13px;">
            <li>This preview is read-only and cannot modify data.</li>
            <li>All values reflect the current session state.</li>
            <li>Missing values are expected and allowed.</li>
            <li>Undo / Redo operations update this view instantly.</li>
            <li>Database persistence is handled separately.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 13.10 — Defensive DEV Assertions
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    for swg in SWG_IDS:
        assert f"{swg}_data" in st.session_state


# =============================================================================
# END SECTION 13 — Live Table Preview
# =============================================================================

# =============================================================================
# SECTION 14 — Edit Table Logic
# =============================================================================
# PURPOSE:
# - Provide controlled editing of SWG tables
# - Support cell, row, and column operations
# - Ensure Undo / Redo safety
#
# RULES:
# - NO database access
# - All mutations must go through Undo system
# - Respect table lock
# =============================================================================


# -----------------------------------------------------------------------------
# 14.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">✏️ Edit Table Data</div>
        <div class="card-subtitle">
            Modify SWG tables using controlled edit modes.
            All changes are Undo / Redo protected.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 14.2 — Edit Mode Selection
# -----------------------------------------------------------------------------
EDIT_MODES: Tuple[str, ...] = (
    "Off",
    "Edit Cells",
    "Insert Row",
    "Delete Row",
    "Insert Column",
    "Delete Column",
    "Rename Column",
    "Move Column Left",
    "Move Column Right",
)

edit_mode = st.selectbox(
    "Edit Mode",
    EDIT_MODES,
    index=0,
    disabled=st.session_state.table_locked,
)

swg_target = st.selectbox(
    "Select SWG",
    SWG_IDS,
    format_func=lambda x: x.replace("SWG", "SWG-"),
    disabled=st.session_state.table_locked,
)

df_key = f"{swg_target}_data"
current_df = st.session_state[df_key]


# -----------------------------------------------------------------------------
# 14.3 — Table Lock Indicator
# -----------------------------------------------------------------------------
if st.session_state.table_locked:
    st.warning("🔒 Table is locked. Editing is disabled.")
elif edit_mode != "Off":
    st.success(f"Edit mode active: {edit_mode}")


# -----------------------------------------------------------------------------
# 14.4 — Always-Visible Preview
# -----------------------------------------------------------------------------
st.markdown("### 📊 Current Table")
st.dataframe(
    current_df,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------------------------------------------------------
# 14.5 — EDIT CELLS MODE (Buffered)
# -----------------------------------------------------------------------------
if edit_mode == "Edit Cells" and not st.session_state.table_locked:

    if not st.session_state.edit_session_active:
        push_undo_snapshot()
        st.session_state.edit_session_active = True
        st.session_state.edit_buffer[swg_target] = current_df.copy(deep=True)

    st.markdown("### ✍️ Cell Editor")

    edited_df = st.data_editor(
        st.session_state.edit_buffer[swg_target],
        use_container_width=True,
        num_rows="dynamic",
        key=f"{swg_target}_cell_editor",
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("💾 Apply Changes", use_container_width=True):
            res = validate_swg_dataframe(edited_df)
            if not res["valid"]:
                for err in res["errors"]:
                    st.error(err)
            else:
                st.session_state[df_key] = edited_df
                st.session_state.edit_session_active = False
                st.session_state.edit_buffer.pop(swg_target, None)
                st.success("Changes applied (Undo supported)")

    with c2:
        if st.button("❌ Cancel", use_container_width=True):
            undo_action()
            st.session_state.edit_session_active = False
            st.session_state.edit_buffer.pop(swg_target, None)
            st.warning("Edit canceled")


# -----------------------------------------------------------------------------
# 14.6 — INSERT ROW
# -----------------------------------------------------------------------------
elif edit_mode == "Insert Row" and not st.session_state.table_locked:

    if st.button("➕ Insert Empty Row", use_container_width=True):

        def mutation():
            df = st.session_state[df_key]
            empty_row = {col: None for col in df.columns}
            st.session_state[df_key] = pd.concat(
                [df, pd.DataFrame([empty_row])],
                ignore_index=True,
            )

        apply_mutation(mutation)
        st.success("Row inserted")


# -----------------------------------------------------------------------------
# 14.7 — DELETE ROW
# -----------------------------------------------------------------------------
elif edit_mode == "Delete Row" and not st.session_state.table_locked:

    if current_df.empty:
        st.warning("No rows to delete.")
    else:
        row_idx = st.number_input(
            "Row index to delete",
            min_value=0,
            max_value=len(current_df) - 1,
            step=1,
        )

        if st.button("🗑 Delete Row", use_container_width=True):

            def mutation():
                df = st.session_state[df_key]
                st.session_state[df_key] = (
                    df.drop(index=row_idx).reset_index(drop=True)
                )

            apply_mutation(mutation)
            st.success("Row deleted")


# -----------------------------------------------------------------------------
# 14.8 — INSERT COLUMN
# -----------------------------------------------------------------------------
elif edit_mode == "Insert Column" and not st.session_state.table_locked:

    new_col = st.text_input("New column name")

    if st.button("➕ Insert Column", use_container_width=True) and new_col:

        def mutation():
            df = st.session_state[df_key]
            df[new_col] = None
            st.session_state[df_key] = df

        apply_mutation(mutation)
        st.success("Column inserted")


# -----------------------------------------------------------------------------
# 14.9 — DELETE COLUMN
# -----------------------------------------------------------------------------
elif edit_mode == "Delete Column" and not st.session_state.table_locked:

    col = st.selectbox("Column to delete", current_df.columns)

    if st.button("🗑 Delete Column", use_container_width=True):

        def mutation():
            df = st.session_state[df_key]
            st.session_state[df_key] = df.drop(columns=[col])

        apply_mutation(mutation)
        st.success("Column deleted")


# -----------------------------------------------------------------------------
# 14.10 — RENAME COLUMN
# -----------------------------------------------------------------------------
elif edit_mode == "Rename Column" and not st.session_state.table_locked:

    col = st.selectbox("Column to rename", current_df.columns)
    new_name = st.text_input("New column name")

    if st.button("✏ Rename Column", use_container_width=True) and new_name:

        def mutation():
            df = st.session_state[df_key]
            st.session_state[df_key] = df.rename(columns={col: new_name})

        apply_mutation(mutation)
        st.success("Column renamed")


# -----------------------------------------------------------------------------
# 14.11 — MOVE COLUMN LEFT
# -----------------------------------------------------------------------------
elif edit_mode == "Move Column Left" and not st.session_state.table_locked:

    col = st.selectbox("Column", current_df.columns)

    if st.button("⬅ Move Left", use_container_width=True):

        def mutation():
            df = st.session_state[df_key]
            cols = list(df.columns)
            idx = cols.index(col)
            if idx > 0:
                cols[idx - 1], cols[idx] = cols[idx], cols[idx - 1]
                st.session_state[df_key] = df[cols]

        apply_mutation(mutation)
        st.success("Column moved left")


# -----------------------------------------------------------------------------
# 14.12 — MOVE COLUMN RIGHT
# -----------------------------------------------------------------------------
elif edit_mode == "Move Column Right" and not st.session_state.table_locked:

    col = st.selectbox("Column", current_df.columns)

    if st.button("➡ Move Right", use_container_width=True):

        def mutation():
            df = st.session_state[df_key]
            cols = list(df.columns)
            idx = cols.index(col)
            if idx < len(cols) - 1:
                cols[idx + 1], cols[idx] = cols[idx], cols[idx + 1]
                st.session_state[df_key] = df[cols]

        apply_mutation(mutation)
        st.success("Column moved right")


# -----------------------------------------------------------------------------
# 14.13 — Reset Edit Session If Mode Off
# -----------------------------------------------------------------------------
if edit_mode == "Off" and st.session_state.edit_session_active:
    st.session_state.edit_session_active = False
    st.session_state.edit_buffer.clear()


# -----------------------------------------------------------------------------
# 14.14 — DEV SAFETY ASSERTIONS
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert df_key in st.session_state


# =============================================================================
# END SECTION 14 — Edit Table Logic
# =============================================================================

# =============================================================================
# SECTION 15 — Save / Clean / Lock Controls
# =============================================================================
# PURPOSE:
# - Provide safe control actions (Save / Clear / Lock)
# - Bridge UI → Repository layer
# - Ensure Undo / Redo consistency
#
# RULES:
# - UI allowed
# - NO direct table mutation
# - DB operations via repository functions only
# =============================================================================


# -----------------------------------------------------------------------------
# 15.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">🛠 Control Panel</div>
        <div class="card-subtitle">
            Save, lock, or reset system data safely.
            All destructive actions require confirmation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 15.2 — Lock Table Toggle
# -----------------------------------------------------------------------------
lock_col, status_col = st.columns([1, 2])

with lock_col:
    st.session_state.table_locked = st.toggle(
        "🔒 Lock Tables",
        value=st.session_state.table_locked,
    )

with status_col:
    if st.session_state.table_locked:
        st.info("Tables are locked. Editing and input are disabled.")
    else:
        st.success("Tables are unlocked. Editing is enabled.")


# -----------------------------------------------------------------------------
# 15.3 — SAVE TO DATABASE
# -----------------------------------------------------------------------------
st.markdown("### 💾 Save Data")

save_col, save_info_col = st.columns([1, 3])

with save_col:
    save_clicked = st.button(
        LABEL_SAVE_DB,
        use_container_width=True,
        disabled=st.session_state.table_locked,
    )

with save_info_col:
    st.caption(
        "Saves current session data into today's SQLite database file. "
        "Data remains available after refresh or restart."
    )

if save_clicked:
    with st.spinner("Saving data to database..."):
        rows_saved = save_session_state_to_database(
            st.session_state,
            target_date=date.today(),
        )

    if rows_saved > 0:
        clear_history()
        st.success(f"✅ {rows_saved} row(s) saved successfully.")
    else:
        st.warning("No data was saved (tables may be empty).")


# -----------------------------------------------------------------------------
# 15.4 — CLEAR CONTROLS
# -----------------------------------------------------------------------------
st.markdown("### 🧹 Clear Data")

clear_mode = st.radio(
    "Clear Mode",
    options=[
        "Clear Session Only",
        "Clear Database Rows (Today)",
        "Delete Entire Daily Database",
    ],
    index=0,
)

confirm_clear = False

if CONFIRM_CLEAR_REQUIRED:
    confirm_clear = st.checkbox(
        "⚠️ I understand this action cannot be undone",
        value=False,
    )
else:
    confirm_clear = True


clear_btn = st.button(
    LABEL_CLEAR_ALL,
    use_container_width=True,
    disabled=not confirm_clear,
)

if clear_btn:
    with st.spinner("Clearing data..."):

        # ---------------------------------------------------------
        # CLEAR SESSION ONLY
        # ---------------------------------------------------------
        if clear_mode == "Clear Session Only":
            payload = build_empty_session_payload()
            for swg, df in payload.items():
                st.session_state[f"{swg}_data"] = df

            clear_history()
            st.success("Session data cleared (database untouched).")

        # ---------------------------------------------------------
        # CLEAR DB ROWS
        # ---------------------------------------------------------
        elif clear_mode == "Clear Database Rows (Today)":
            result = clear_persisted_data(
                clear_db_rows=True,
                drop_schema=False,
                delete_db_file=False,
                target_date=date.today(),
            )

            clear_history()
            st.success(
                f"Database cleared — {result['rows_deleted']} row(s) removed."
            )

        # ---------------------------------------------------------
        # DELETE ENTIRE DB FILE
        # ---------------------------------------------------------
        elif clear_mode == "Delete Entire Daily Database":
            result = clear_persisted_data(
                delete_db_file=True,
                target_date=date.today(),
            )

            clear_history()
            if result["file_deleted"]:
                st.success("Daily database file deleted successfully.")
            else:
                st.warning("No database file found to delete.")


# -----------------------------------------------------------------------------
# 15.5 — STATUS FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card soft">
        <div class="card-title">ℹ️ Control Notes</div>
        <ul style="font-size: 13px; color: var(--color-text-muted);">
            <li>Save writes data to a daily SQLite file.</li>
            <li>Clear Session does not affect the database.</li>
            <li>Clear Database affects only today’s data.</li>
            <li>Deleting the database file is irreversible.</li>
            <li>Undo / Redo history resets after save or clear.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 15.6 — DEV SAFETY ASSERTIONS
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert isinstance(st.session_state.table_locked, bool)


# =============================================================================
# END SECTION 15 — Save / Clean / Lock Controls
# =============================================================================

# =============================================================================
# SECTION 16 — Analysis
# =============================================================================
# PURPOSE:
# - Provide analytical insight into SWG data
# - Compute statistics safely
# - Support operational decision-making
#
# RULES:
# - READ-ONLY
# - NO database access
# - NO session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 16.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">📈 Data Analysis</div>
        <div class="card-subtitle">
            Statistical overview and operational insight.
            All results are calculated live from session data.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 16.2 — Analysis Controls
# -----------------------------------------------------------------------------
show_analysis = st.toggle(
    "Show Analysis",
    value=True,
)

analysis_scope = st.radio(
    "Analysis Scope",
    options=[
        "Per SWG",
        "Compare All SWGs",
    ],
    horizontal=True,
)


if not show_analysis:
    st.info("Analysis is hidden.")
    st.stop()


# -----------------------------------------------------------------------------
# 16.3 — Helper: Safe Numeric Series
# -----------------------------------------------------------------------------
def to_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert series to numeric safely.
    """
    return pd.to_numeric(series, errors="coerce")


# -----------------------------------------------------------------------------
# 16.4 — Helper: Descriptive Statistics
# -----------------------------------------------------------------------------
def compute_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Compute descriptive statistics for a numeric series.
    """
    s = to_numeric_series(series)

    return {
        "Count": int(s.count()),
        "Missing": int(s.isna().sum()),
        "Mean": float(s.mean()) if not s.empty else None,
        "Median": float(s.median()) if not s.empty else None,
        "Std Dev": float(s.std()) if not s.empty else None,
        "Min": float(s.min()) if not s.empty else None,
        "Max": float(s.max()) if not s.empty else None,
    }


# -----------------------------------------------------------------------------
# 16.5 — Helper: Time Coverage Metrics
# -----------------------------------------------------------------------------
def compute_time_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute time coverage metrics for a SWG table.
    """
    if df.empty:
        return {
            "Rows": 0,
            "Start Time": None,
            "End Time": None,
            "Duration (min)": None,
        }

    dt = pd.to_datetime(df[COLUMN_DATETIME], errors="coerce").dropna()

    if dt.empty:
        return {
            "Rows": len(df),
            "Start Time": None,
            "End Time": None,
            "Duration (min)": None,
        }

    duration = (dt.max() - dt.min()).total_seconds() / 60.0

    return {
        "Rows": len(df),
        "Start Time": dt.min(),
        "End Time": dt.max(),
        "Duration (min)": round(duration, 2),
    }


# -----------------------------------------------------------------------------
# 16.6 — Per-SWG Analysis
# -----------------------------------------------------------------------------
def render_per_swg_analysis() -> None:
    """
    Render analysis tables per SWG.
    """
    for swg in SWG_IDS:
        df = st.session_state[f"{swg}_data"]

        st.markdown(f"### 🔍 {swg.replace('SWG', 'SWG-')}")

        if df.empty:
            st.warning("No data available.")
            continue

        # ---------------------------------------------------------
        # Time Coverage
        # ---------------------------------------------------------
        time_stats = compute_time_coverage(df)
        st.markdown("**Time Coverage**")
        st.dataframe(
            pd.DataFrame([time_stats]),
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------
        # Metric Statistics
        # ---------------------------------------------------------
        stats = {
            COLUMN_ACTIVE: compute_descriptive_stats(df[COLUMN_ACTIVE]),
            COLUMN_REACTIVE: compute_descriptive_stats(df[COLUMN_REACTIVE]),
            COLUMN_SOC: compute_descriptive_stats(df[COLUMN_SOC]),
        }

        stats_df = pd.DataFrame(stats)
        st.markdown("**Descriptive Statistics**")
        st.dataframe(
            stats_df.style.format("{:.3f}"),
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# 16.7 — Cross-SWG Comparison (SAFE FORMAT)
# -----------------------------------------------------------------------------
def render_cross_swg_analysis() -> None:
    """
    Render comparison across SWGs (SAFE numeric formatting).
    """
    rows = []

    for swg in SWG_IDS:
        df = st.session_state[f"{swg}_data"]

        row = {
            "SWG": swg.replace("SWG", "SWG-"),
            "Rows": len(df),
            "Active Mean": None,
            "Reactive Mean": None,
            "SOC Mean": None,
        }

        if not df.empty:
            row["Active Mean"] = pd.to_numeric(
                df[COLUMN_ACTIVE], errors="coerce"
            ).mean()

            row["Reactive Mean"] = pd.to_numeric(
                df[COLUMN_REACTIVE], errors="coerce"
            ).mean()

            row["SOC Mean"] = pd.to_numeric(
                df[COLUMN_SOC], errors="coerce"
            ).mean()

        rows.append(row)

    compare_df = pd.DataFrame(rows)

    # ✅ Format ONLY numeric columns
    numeric_cols = compare_df.select_dtypes(include="number").columns

    styled_df = compare_df.style.format(
        {col: "{:.3f}" for col in numeric_cols}
    )

    st.markdown("### 🔀 Cross-SWG Comparison")
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )



# -----------------------------------------------------------------------------
# 16.8 — Render Based on Scope
# -----------------------------------------------------------------------------
if analysis_scope == "Per SWG":
    render_per_swg_analysis()
else:
    render_cross_swg_analysis()


# -----------------------------------------------------------------------------
# 16.9 — Data Quality Summary
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card soft">
        <div class="card-title">🧪 Data Quality Notes</div>
        <ul style="font-size: 13px; color: var(--color-text-muted);">
            <li>Missing values are allowed and expected.</li>
            <li>Statistics ignore invalid or missing numeric entries.</li>
            <li>Datetime parsing is timezone-safe.</li>
            <li>No data is modified during analysis.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 16.10 — DEV SAFETY ASSERTIONS
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    for swg in SWG_IDS:
        assert isinstance(st.session_state[f"{swg}_data"], pd.DataFrame)


# =============================================================================
# END SECTION 16 — Analysis
# =============================================================================

# =============================================================================
# SECTION 17 — Visualization
# =============================================================================
# PURPOSE:
# - Visualize SWG data safely
# - Provide trend insight
# - Support operational comparison
#
# RULES:
# - READ-ONLY
# - NO database access
# - NO session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 17.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">📊 Data Visualization</div>
        <div class="card-subtitle">
            Interactive charts for trend analysis and SWG comparison.
            All visuals are generated live from session data.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 17.2 — Visualization Controls
# -----------------------------------------------------------------------------
show_visuals = st.toggle("Show Visualization", value=True)

if not show_visuals:
    st.info("Visualization is hidden.")
    st.stop()


metric = st.selectbox(
    "Metric",
    options=[
        COLUMN_ACTIVE,
        COLUMN_REACTIVE,
        COLUMN_SOC,
    ],
)

view_mode = st.radio(
    "View Mode",
    options=[
        "Per SWG",
        "Compare All SWGs",
    ],
    horizontal=True,
)

chart_type = st.selectbox(
    "Chart Type",
    options=[
        "Line",
        "Step Line",
        "Area",
    ],
)

show_points = st.checkbox("Show Points", value=True)
show_rolling = st.checkbox("Show Rolling Average", value=False)

rolling_window = st.slider(
    "Rolling Window",
    min_value=ROLLING_WINDOW_MIN,
    max_value=ROLLING_WINDOW_MAX,
    value=ROLLING_WINDOW_DEFAULT,
    disabled=not show_rolling,
)


# -----------------------------------------------------------------------------
# 17.3 — Helper: Prepare Visualization Data
# -----------------------------------------------------------------------------
def prepare_viz_dataframe(
    swg: str,
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Prepare a SWG dataframe for visualization.
    """
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()

    tmp["DateTime"] = pd.to_datetime(
        tmp[COLUMN_DATETIME],
        errors="coerce"
    )

    tmp["Value"] = pd.to_numeric(
        tmp[metric],
        errors="coerce"
    )

    tmp["SWG"] = swg.replace("SWG", "SWG-")

    return tmp[["DateTime", "Value", "SWG"]].dropna()


# -----------------------------------------------------------------------------
# 17.4 — Build Combined Visualization Data
# -----------------------------------------------------------------------------
frames: List[pd.DataFrame] = []

for swg in SWG_IDS:
    df = st.session_state[f"{swg}_data"]
    viz_df = prepare_viz_dataframe(swg, df, metric)
    if not viz_df.empty:
        frames.append(viz_df)

if not frames:
    st.warning("No data available for visualization.")
    st.stop()

viz_data = pd.concat(frames).sort_values("DateTime")


# -----------------------------------------------------------------------------
# 17.5 — Rolling Average (Optional)
# -----------------------------------------------------------------------------
if show_rolling:
    viz_data = viz_data.sort_values(["SWG", "DateTime"])
    viz_data["Rolling"] = (
        viz_data
        .groupby("SWG")["Value"]
        .rolling(
            window=rolling_window,
            min_periods=1
        )
        .mean()
        .reset_index(level=0, drop=True)
    )


# -----------------------------------------------------------------------------
# 17.6 — Chart Builders
# -----------------------------------------------------------------------------
interpolate = "step-after" if chart_type == "Step Line" else "linear"

def base_chart(df: pd.DataFrame, height: int):
    if chart_type == "Area":
        chart = alt.Chart(df).mark_area(
            interpolate=interpolate,
            opacity=DEFAULT_CHART_OPACITY,
        )
    else:
        chart = alt.Chart(df).mark_line(
            interpolate=interpolate,
            strokeWidth=DEFAULT_LINE_WIDTH,
            point=show_points,
        )

    return chart.encode(
        x=alt.X(
            "DateTime:T",
            title="Date & Time",
            axis=alt.Axis(
                format="%H:%M:%S",
                labelAngle=-30
            )
        ),
        y=alt.Y(
            "Value:Q",
            title=metric,
        ),
        color=alt.Color(
            "SWG:N",
            legend=alt.Legend(title="SWG"),
        ),
        tooltip=[
            "SWG",
            alt.Tooltip("DateTime:T", title="DateTime"),
            alt.Tooltip("Value:Q", title=metric, format=".2f"),
        ],
    ).properties(height=height)


def rolling_chart(df: pd.DataFrame):
    return alt.Chart(df).mark_line(
        strokeDash=[6, 4],
        strokeWidth=1.5,
    ).encode(
        x="DateTime:T",
        y="Rolling:Q",
        color="SWG:N",
    )


# -----------------------------------------------------------------------------
# 17.7 — Render Charts
# -----------------------------------------------------------------------------
if view_mode == "Compare All SWGs":
    st.markdown("### 🔀 Compare All SWGs")

    chart = base_chart(viz_data, DEFAULT_CHART_HEIGHT)

    if show_rolling:
        chart += rolling_chart(viz_data)

    st.altair_chart(chart, use_container_width=True)

else:
    st.markdown("### 🔍 Per-SWG Trends")

    swgs = viz_data["SWG"].unique()
    cols = st.columns(len(swgs))

    for col, swg_label in zip(cols, swgs):
        with col:
            st.markdown(f"**{swg_label}**")
            sub = viz_data[viz_data["SWG"] == swg_label]

            chart = base_chart(sub, 220)

            if show_rolling:
                chart += rolling_chart(sub)

            st.altair_chart(chart, use_container_width=True)


# -----------------------------------------------------------------------------
# 17.8 — Visualization Notes
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card soft">
        <div class="card-title">ℹ️ Visualization Notes</div>
        <ul style="font-size: 13px; color: var(--color-text-muted);">
            <li>Charts update instantly with Undo / Redo.</li>
            <li>Rolling averages smooth short-term fluctuations.</li>
            <li>Missing values are skipped automatically.</li>
            <li>No data is modified during visualization.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 17.9 — DEV SAFETY ASSERTIONS
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert not viz_data.empty


# =============================================================================
# END SECTION 17 — Visualization
# =============================================================================

# =============================================================================
# SECTION 18 — Download (CSV / Excel / JSON)
# =============================================================================
# PURPOSE:
# - Export live session data
# - Provide Excel-aligned wide table
# - Support CSV / Excel / JSON
#
# RULES:
# - READ-ONLY
# - NO database access
# - NO session_state mutation
# =============================================================================


# -----------------------------------------------------------------------------
# 18.1 — Section Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">⬇️ Download Data</div>
        <div class="card-subtitle">
            Export current session data in CSV, Excel, or JSON format.
            Downloads reflect the live state of the system.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 18.2 — Helper: Build Wide DataFrame (Read-Only)
# -----------------------------------------------------------------------------
def build_wide_export_dataframe() -> pd.DataFrame:
    """
    Build Excel-style wide dataframe from session_state.

    Returns:
        Wide DataFrame ready for export
    """
    swg_data = {
        swg: st.session_state[f"{swg}_data"]
        for swg in SWG_IDS
    }

    # Determine max row count
    max_len = max(len(df) for df in swg_data.values())

    frames: List[pd.DataFrame] = []

    for swg in SWG_IDS:
        df = swg_data[swg].copy()
        df = df.reset_index(drop=True)
        df = df.reindex(range(max_len))

        # Format datetime safely
        if COLUMN_DATETIME in df.columns:
            df[COLUMN_DATETIME] = (
                pd.to_datetime(df[COLUMN_DATETIME], errors="coerce")
                .dt.tz_localize(None)
                .dt.strftime(EXPORT_DATETIME_FORMAT)
            )

        # Rename columns to wide format
        df.columns = list(SWG_TO_WIDE_COLUMNS[swg])

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=WIDE_TABLE_COLUMN_ORDER[1:])

    wide_df = pd.concat(frames, axis=1)
    return wide_df


# -----------------------------------------------------------------------------
# 18.3 — Helper: Sanitize Values for Export
# -----------------------------------------------------------------------------
def sanitize_export_value(value: Any) -> Any:
    """
    Sanitize values for export compatibility.
    """
    if value is None or pd.isna(value):
        return EXPORT_NA_REPLACEMENT

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime(EXPORT_DATETIME_FORMAT)

    if isinstance(value, (np.integer, np.floating)):
        return round(float(value), FLOAT_PRECISION_EXPORT)

    return value


# -----------------------------------------------------------------------------
# 18.4 — Build Export Data
# -----------------------------------------------------------------------------
wide_export_df = build_wide_export_dataframe()

if wide_export_df.empty:
    st.info("No data available for download.")
    st.stop()

wide_export_df = wide_export_df.applymap(sanitize_export_value)


# -----------------------------------------------------------------------------
# 18.5 — CSV Export
# -----------------------------------------------------------------------------
csv_bytes = wide_export_df.to_csv(
    index=False
).encode("utf-8")


# -----------------------------------------------------------------------------
# 18.6 — Excel Export (Safe)
# -----------------------------------------------------------------------------
excel_buffer = BytesIO()

with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    wide_export_df.to_excel(
        writer,
        sheet_name="SWG_Data",
        index=False
    )

excel_bytes = excel_buffer.getvalue()


# -----------------------------------------------------------------------------
# 18.7 — JSON Export (Structured by SWG)
# -----------------------------------------------------------------------------
json_payload: Dict[str, List[Dict[str, Any]]] = {}

for swg in SWG_IDS:
    df = st.session_state[f"{swg}_data"]
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            record[col] = sanitize_export_value(val)
        records.append(record)

    json_payload[swg.replace("SWG", "SWG-")] = records

json_bytes = json.dumps(
    json_payload,
    indent=2
).encode("utf-8")


# -----------------------------------------------------------------------------
# 18.8 — Download Buttons
# -----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name=EXPORT_CSV_NAME,
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=EXPORT_EXCEL_NAME,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with c3:
    st.download_button(
        "⬇️ Download JSON",
        data=json_bytes,
        file_name=EXPORT_JSON_NAME,
        mime="application/json",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# 18.9 — Export Notes
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="card soft">
        <div class="card-title">ℹ️ Export Notes</div>
        <ul style="font-size: 13px; color: var(--color-text-muted);">
            <li>Wide table layout matches Excel exactly.</li>
            <li>Missing values are preserved.</li>
            <li>All datetimes are timezone-stripped for compatibility.</li>
            <li>Exports reflect the live session state.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 18.10 — DEV SAFETY ASSERTIONS
# -----------------------------------------------------------------------------
if APP_ENVIRONMENT != "PRODUCTION":
    assert not wide_export_df.empty


# =============================================================================
# END SECTION 18 — Download (CSV / Excel / JSON)
# =============================================================================
