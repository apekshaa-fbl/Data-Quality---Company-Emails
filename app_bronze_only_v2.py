
# app_bronze_only_v2.py
import os
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

NOTION_LINK = "https://www.notion.so/firmable/Company-Emails-RCA-Coverage-gaps-2dfd5c6ffd8780fdb32cfff46548fcfb"

st.set_page_config(page_title="Company Email Data Quality (Bronze)", layout="wide")

st.title("Company Email Data Quality – Bronze Layer")
st.caption("Release checks • Insights • SLA breaches • Action items • Run logs")
st.markdown(f"🔗 **Notion RCA:** [{NOTION_LINK}]({NOTION_LINK})")
st.write("")

def _get_env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v

def get_pg_conn():
    required = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    missing = [k for k in required if not _get_env(k)]
    if missing:
        st.error(f"Missing env vars: {', '.join(missing)}")
        st.caption(f"Loaded .env from: {ENV_PATH}")
        st.stop()

    return psycopg2.connect(
        host=_get_env("PGHOST"),
        port=int(_get_env("PGPORT", "5432")),
        dbname=_get_env("PGDATABASE"),
        user=_get_env("PGUSER"),
        password=_get_env("PGPASSWORD"),
        sslmode=_get_env("PGSSLMODE", "require"),
    )

@st.cache_data(ttl=300, show_spinner=False)
def run_sql_df(sql: str, params: Optional[list] = None) -> pd.DataFrame:
    with get_pg_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def exec_sql(sql: str, params: Optional[dict] = None) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()

def ensure_log_tables():
    exec_sql("CREATE SCHEMA IF NOT EXISTS dq;")
    exec_sql("""
    CREATE TABLE IF NOT EXISTS dq.email_dq_runs (
      run_id          uuid PRIMARY KEY,
      run_ts          timestamptz NOT NULL DEFAULT now(),
      release_label   text NULL,
      created_by      text NULL,
      notes           text NULL
    );
    """)
    exec_sql("""
    CREATE TABLE IF NOT EXISTS dq.email_dq_metrics (
      run_id        uuid NOT NULL REFERENCES dq.email_dq_runs(run_id),
      metric_key    text NOT NULL,
      metric_label  text NOT NULL,
      value_num     numeric NULL,
      value_pct     numeric NULL,
      severity      text NULL,
      details_json  jsonb NULL,
      PRIMARY KEY (run_id, metric_key)
    );
    """)
    exec_sql("""
    CREATE TABLE IF NOT EXISTS dq.email_dq_comments (
      run_id        uuid NOT NULL REFERENCES dq.email_dq_runs(run_id),
      metric_key    text NOT NULL,
      comment_ts    timestamptz NOT NULL DEFAULT now(),
      author        text NULL,
      comment_text  text NULL,
      sample_link   text NULL
    );
    """)

def insert_run(release_label: str, created_by: str, notes: str) -> str:
    run_id = str(uuid.uuid4())
    exec_sql("""
      INSERT INTO dq.email_dq_runs (run_id, release_label, created_by, notes)
      VALUES (%s, %s, %s, %s)
    """, {"run_id": run_id, "release_label": release_label, "created_by": created_by, "notes": notes})
    return run_id

def upsert_metric(run_id: str, metric_key: str, metric_label: str,
                  value_num: Optional[float], value_pct: Optional[float],
                  severity: str, details: Dict[str, Any]):
    exec_sql("""
      INSERT INTO dq.email_dq_metrics (run_id, metric_key, metric_label, value_num, value_pct, severity, details_json)
      VALUES (%(run_id)s, %(metric_key)s, %(metric_label)s, %(value_num)s, %(value_pct)s, %(severity)s, %(details)s::jsonb)
      ON CONFLICT (run_id, metric_key)
      DO UPDATE SET
        metric_label = EXCLUDED.metric_label,
        value_num = EXCLUDED.value_num,
        value_pct = EXCLUDED.value_pct,
        severity = EXCLUDED.severity,
        details_json = EXCLUDED.details_json
    """, {
        "run_id": run_id,
        "metric_key": metric_key,
        "metric_label": metric_label,
        "value_num": value_num,
        "value_pct": value_pct,
        "severity": severity,
        "details": json.dumps(details or {})
    })

def add_comment(run_id: str, metric_key: str, author: str, comment_text: str, sample_link: str):
    exec_sql("""
      INSERT INTO dq.email_dq_comments (run_id, metric_key, author, comment_text, sample_link)
      VALUES (%s, %s, %s, %s, %s)
    """, {"run_id": run_id, "metric_key": metric_key, "author": author, "comment_text": comment_text, "sample_link": sample_link})

BRZ_EMAILS  = "zeus_bronze.brz_comp_emails"
BRZ_DOMAINS = "zeus_bronze.brz_comp_domains"

def severity_from_threshold(pct: Optional[float], threshold: float, direction: str = "above") -> str:
    if pct is None:
        return "unknown"
    if direction == "above":
        return "breach" if pct >= threshold else "ok"
    return "breach" if pct <= threshold else "ok"

def action_items_from_breaches(breaches: List[dict]) -> List[str]:
    items = []
    for b in breaches:
        if b["metric_key"] == "domain_status_not_200_pct":
            items.append("[SLA] High non-200 domains — improve domain enrichment / retry logic and validate vendor domains.")
        elif b["metric_key"] == "invalid_format_pct":
            items.append("[DQ] High invalid formats — adjust cleaning rules, block placeholders/masked values, and fix upstream.")
        elif b["metric_key"] == "duplicate_email_pct":
            items.append("[DQ] High duplicates — enforce dedupe per (company,country,email_clean) and identify duplicate-producing source.")
        elif b["metric_key"] == "domain_mismatch_pct":
            items.append("[DQ] High mismatch — improve parsing + mapping, and add controlled exceptions (free-mail, group domains).")
        elif b["metric_key"] == "blocked_email_pct":
            items.append("[DQ] High blocked emails — expand blocklist patterns and drop no-reply emails early.")
        elif b["metric_key"] == "gt5_email_country_pct_max":
            items.append("[DQ] High >5 emails rate — enforce top-5 ranking policy and prune noisy emails per country.")
        elif b["metric_key"] == "coverage_without_email_pct":
            items.append("[SLA] Low email coverage — investigate ingestion gaps and ensure emails map correctly to company IDs.")
        else:
            items.append(f"[DQ] Review breach: {b['metric_label']}")
    return items

COMPOSITE_COMPONENTS = [
    ("coverage_with_email_pct", "Coverage: % companies with ≥1 email", "higher"),
    ("invalid_format_pct", "Email format: % invalid", "lower"),
    ("duplicate_email_pct", "Duplicate emails: %", "lower"),
    ("blocked_email_pct", "Blocked emails: %", "lower"),
    ("domain_mismatch_pct", "Domain mismatch: %", "lower"),
    ("domain_status_not_200_pct", "Domains: % not-200", "lower"),
    ("gt5_email_country_pct_max", "Max country % of companies with >5 emails", "lower"),
]

def compute_composite_score(metrics: Dict[str, float]) -> Optional[float]:
    vals = []
    for key, _, direction in COMPOSITE_COMPONENTS:
        if key not in metrics or metrics[key] is None:
            continue
        v = float(metrics[key])
        vals.append(v if direction == "higher" else (100.0 - v))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)

with st.sidebar:
    st.subheader("Run Metadata")
    release_label = st.text_input("Release label (e.g., 2026-01-16 release)")
    created_by = st.text_input("Your name / handle", value=os.getenv("USER", ""))
    run_notes = st.text_area("Run notes (optional)")

    st.divider()
    st.subheader("Dynamic SLA Rules (toggles)")
    coverage_without_threshold = st.slider("Coverage breach threshold (% without email)", 0, 100, 30)
    blocked_email_threshold = st.slider("Blocked email % breach threshold", 0, 100, 1)
    duplicate_email_threshold = st.slider("Duplicate email % breach threshold", 0, 100, 5)
    invalid_format_threshold = st.slider("Invalid format % breach threshold", 0, 100, 2)
    domain_mismatch_threshold = st.slider("Domain mismatch % breach threshold", 0, 100, 30)
    not200_threshold = st.slider("Not-200 domain % breach threshold", 0, 100, 30)
    gt5_country_threshold = st.slider("Country >5 emails % breach threshold", 0, 100, 2)

    st.divider()
    st.subheader("Persistence")
    persist_to_db = st.toggle("Save this run + comments to Postgres (dq schema)", value=False)

if persist_to_db:
    try:
        ensure_log_tables()
    except Exception as e:
        st.error(f"Failed to initialise dq log tables: {e}")
        persist_to_db = False

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "breaches" not in st.session_state:
    st.session_state.breaches = []
if "metrics_for_dashboard" not in st.session_state:
    st.session_state.metrics_for_dashboard = []

def section_comment_ui(metric_key: str):
    st.markdown("**Comments & Sample Link**")
    c1, c2 = st.columns([2, 1])
    with c1:
        comment_text = st.text_area(f"Comment ({metric_key})", key=f"comment_{metric_key}")
    with c2:
        sample_link = st.text_input(f"Link (Excel/GSheet/Jira/etc.)", key=f"link_{metric_key}")
    if st.button("Save comment", key=f"save_{metric_key}"):
        if not persist_to_db:
            st.info("Enable 'Save this run + comments to Postgres' to persist comments.")
        elif not st.session_state.run_id:
            st.warning("Run not started yet. Click 'Run all checks' first.")
        else:
            add_comment(st.session_state.run_id, metric_key, created_by, comment_text, sample_link)
            st.success("Saved.")

def add_metric_for_dashboard(metric_key: str, metric_label: str, pct: Optional[float], severity: str, details: Dict[str, Any]):
    st.session_state.metrics_for_dashboard.append({
        "metric_key": metric_key,
        "metric_label": metric_label,
        "pct": pct,
        "severity": severity,
        "details": details or {}
    })

def record_breach(metric_key: str, metric_label: str, pct: Optional[float], severity: str, details: Dict[str, Any]):
    if severity == "breach":
        st.session_state.breaches.append({
            "metric_key": metric_key,
            "metric_label": metric_label,
            "pct": pct,
            "details": details or {}
        })

def show_examples(title: str, sql: str):
    st.subheader(title)
    with st.expander(f"SQL Query (Examples: {title})", expanded=False):
        st.code(sql, language="sql")
    df = run_sql_df(sql)
    st.dataframe(df, use_container_width=True)

checks_tab, dashboard_tab = st.tabs(["✅ Checks (Bronze)", "📊 Dashboard & Trends"])

with checks_tab:
    run_all = st.button("Run all checks", type="primary")
    st.caption("This version adds **Top-20 example records** for key metrics (invalid, duplicates, mismatch, blocked, >5 emails).")

    if run_all:
        st.session_state.breaches = []
        st.session_state.metrics_for_dashboard = []
        if persist_to_db:
            st.session_state.run_id = insert_run(release_label, created_by, run_notes)
        else:
            st.session_state.run_id = None
        st.success("Run started. Scroll down for results.")

    st.divider()

    # 1) Coverage
    st.header("1) Coverage")
    COVERAGE_SQL = f"""
    with email_per_company as (
      select d.id, count(e.email) as email_count
      from {BRZ_DOMAINS} d
      left join {BRZ_EMAILS} e on d.id = e.id
      where d.id is not null
      group by d.id
    )
    select
      count(id) as total_companies,
      sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
      sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
      round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
      round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
    from email_per_company;
    """
    cov = run_sql_df(COVERAGE_SQL)
    st.dataframe(cov, use_container_width=True)

    pct_with = float(cov.iloc[0]["pct_with_email"])
    pct_without = float(cov.iloc[0]["pct_without_email"])
    sev = severity_from_threshold(pct_without, float(coverage_without_threshold), direction="above")
    add_metric_for_dashboard("coverage_with_email_pct", "Coverage: % companies with ≥1 email", pct_with, "ok", {"pct_without_email": pct_without})
    add_metric_for_dashboard("coverage_without_email_pct", "Coverage: % companies without email", pct_without, sev, {"threshold": coverage_without_threshold})
    record_breach("coverage_without_email_pct", "Coverage: % companies without email", pct_without, sev, {"threshold": coverage_without_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "coverage_with_email_pct", "Coverage: % companies with ≥1 email", None, pct_with, "ok", {"pct_without_email": pct_without})
        upsert_metric(st.session_state.run_id, "coverage_without_email_pct", "Coverage: % companies without email", None, pct_without, sev, {"threshold": coverage_without_threshold})
    section_comment_ui("coverage")

    st.divider()

    # 2) Domain Status
    st.header("2) Domain status code")
    STATUS_CODE_SQL = f"""
    WITH base AS (SELECT status_code FROM {BRZ_DOMAINS}),
    total AS (SELECT COUNT(*)::numeric AS total_domains FROM base),
    counts AS (
      SELECT 'Status Code 200' AS status, COUNT(*)::numeric AS n FROM base WHERE status_code = 200
      UNION ALL
      SELECT 'Status Code not 200' AS status, COUNT(*)::numeric AS n FROM base WHERE status_code IS NOT NULL AND status_code <> 200
      UNION ALL
      SELECT 'Missing Status Code' AS status, COUNT(*)::numeric AS n FROM base WHERE status_code IS NULL
    )
    SELECT status AS "Status", n::bigint AS "Count of Domains",
      ROUND(n / NULLIF(total.total_domains, 0) * 100, 2) AS "Pct"
    FROM counts CROSS JOIN total
    ORDER BY CASE status WHEN 'Status Code 200' THEN 1 WHEN 'Status Code not 200' THEN 2 ELSE 3 END;
    """
    sc = run_sql_df(STATUS_CODE_SQL)
    st.dataframe(sc, use_container_width=True)
    st.plotly_chart(px.bar(sc, x="Status", y="Pct", hover_data=["Count of Domains"]), use_container_width=True)
    not200_pct = float(sc.loc[sc["Status"] == "Status Code not 200", "Pct"].iloc[0]) if (sc["Status"] == "Status Code not 200").any() else None
    sev = severity_from_threshold(not200_pct, float(not200_threshold), direction="above")
    add_metric_for_dashboard("domain_status_not_200_pct", "Domains: % not-200", not200_pct, sev, {"threshold": not200_threshold})
    record_breach("domain_status_not_200_pct", "Domains: % not-200", not200_pct, sev, {"threshold": not200_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "domain_status_not_200_pct", "Domains: % not-200", None, not200_pct, sev, {"threshold": not200_threshold})
    section_comment_ui("status_code")

    st.divider()

    # 3) Format + examples
    st.header("3) Proper email format coverage")
    FORMAT_SQL = f"""
    with base as (
      select id, country, email,
        regexp_replace(lower(trim(coalesce(email, ''))),'[^a-z0-9@._%+\\-]','','g') as email_clean
      from {BRZ_EMAILS}
    ),
    classified as (
      select *,
        case
          when email_clean is null or trim(email_clean) = '' then 'empty'
          when email_clean not like '%@%' then 'invalid'
          when email_clean !~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$' then 'invalid'
          else 'valid'
        end as format_status
      from base
    )
    select format_status, count(*) as email_rows,
      round(100.0 * count(*) / nullif(sum(count(*)) over (),0), 2) as pct
    from classified
    group by format_status
    order by email_rows desc;
    """
    fmt = run_sql_df(FORMAT_SQL)
    st.dataframe(fmt, use_container_width=True)
    st.plotly_chart(px.bar(fmt, x="format_status", y="pct", hover_data=["email_rows"]), use_container_width=True)
    invalid_pct = float(fmt.loc[fmt["format_status"] == "invalid", "pct"].iloc[0]) if (fmt["format_status"] == "invalid").any() else None
    sev = severity_from_threshold(invalid_pct, float(invalid_format_threshold), direction="above")
    add_metric_for_dashboard("invalid_format_pct", "Email format: % invalid", invalid_pct, sev, {"threshold": invalid_format_threshold})
    record_breach("invalid_format_pct", "Email format: % invalid", invalid_pct, sev, {"threshold": invalid_format_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "invalid_format_pct", "Email format: % invalid", None, invalid_pct, sev, {"threshold": invalid_format_threshold})
    section_comment_ui("format")

    INVALID_EXAMPLES_SQL = f"""
    with base as (
      select id, country, email,
        regexp_replace(lower(trim(coalesce(email, ''))),'[^a-z0-9@._%+\\-]','','g') as email_clean
      from {BRZ_EMAILS}
      where email is not null and trim(email) <> ''
    )
    select id, country, email, email_clean
    from base
    where email_clean not like '%@%'
       or email_clean !~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    limit 20;
    """
    show_examples("Top 20 invalid email examples", INVALID_EXAMPLES_SQL)

    st.divider()

    # 4) Duplicates + examples
    st.header("4) Duplicate emails")
    DUP_SUMMARY_SQL = f"""
    WITH base AS (
      SELECT id AS firmable_id, country,
        regexp_replace(lower(trim(coalesce(email,''))),'[^a-z0-9@._%+\\-]','','g') AS email_clean
      FROM {BRZ_EMAILS}
      WHERE email IS NOT NULL AND trim(email) <> ''
    ),
    valid AS (
      SELECT * FROM base
      WHERE email_clean LIKE '%@%'
        AND email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    ),
    email_counts AS (
      SELECT firmable_id, country, email_clean, COUNT(*) AS occurrences
      FROM valid
      GROUP BY 1,2,3
    ),
    summary AS (
      SELECT COUNT(*) AS total_email,
             COUNT(*) FILTER (WHERE occurrences = 1) AS unique_email,
             COUNT(*) FILTER (WHERE occurrences > 1) AS duplicate_email
      FROM email_counts
    )
    SELECT total_email, unique_email, duplicate_email,
           ROUND(duplicate_email * 100.0 / NULLIF(total_email, 0), 2) AS duplicate_percentage
    FROM summary;
    """
    dup = run_sql_df(DUP_SUMMARY_SQL)
    st.dataframe(dup, use_container_width=True)
    dup_pct = float(dup.iloc[0]["duplicate_percentage"])
    sev = severity_from_threshold(dup_pct, float(duplicate_email_threshold), direction="above")
    add_metric_for_dashboard("duplicate_email_pct", "Duplicate emails: %", dup_pct, sev, {"threshold": duplicate_email_threshold})
    record_breach("duplicate_email_pct", "Duplicate emails: %", dup_pct, sev, {"threshold": duplicate_email_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "duplicate_email_pct", "Duplicate emails: %", None, dup_pct, sev, {"threshold": duplicate_email_threshold})
    section_comment_ui("duplicates")

    DUP_TOP20_SQL = f"""
    WITH base AS (
      SELECT regexp_replace(lower(trim(coalesce(email,''))),'[^a-z0-9@._%+\\-]','','g') AS email_clean
      FROM {BRZ_EMAILS}
      WHERE email IS NOT NULL AND trim(email) <> ''
    ),
    valid AS (
      SELECT * FROM base
      WHERE email_clean LIKE '%@%'
        AND email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    )
    SELECT email_clean, COUNT(*) AS occurrences
    FROM valid
    GROUP BY email_clean
    HAVING COUNT(*) > 1
    ORDER BY occurrences DESC
    LIMIT 20;
    """
    show_examples("Top 20 duplicate emails (by occurrences)", DUP_TOP20_SQL)

    st.divider()

    # 5) Domain mismatch + examples
    st.header("5) Domain match rate")
    DOMAIN_MATCH_SQL = f"""
    WITH base AS (
        SELECT d.id, d.country, d.status_code,
               lower(trim(coalesce(d.fqdn, ''))) AS fqdn,
               lower(trim(coalesce(e.email, ''))) AS email
        FROM {BRZ_DOMAINS} d
        LEFT JOIN {BRZ_EMAILS} e ON d.id = e.id
    ),
    parsed AS (
        SELECT *,
               split_part(fqdn, '.', 1) AS company_root,
               split_part(split_part(email, '@', 2), '.', 1) AS email_root
        FROM base
        WHERE fqdn <> '' AND email <> '' AND email LIKE '%@%' AND fqdn LIKE '%.%'
    )
    SELECT
      ROUND(100.0 * SUM(CASE WHEN company_root <> '' AND email_root <> '' AND company_root = email_root THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0), 2) AS match_pct
    FROM parsed;
    """
    dm = run_sql_df(DOMAIN_MATCH_SQL)
    st.dataframe(dm, use_container_width=True)
    match_pct = float(dm.iloc[0]["match_pct"])
    mismatch_pct = round(100.0 - match_pct, 2)
    sev = severity_from_threshold(mismatch_pct, float(domain_mismatch_threshold), direction="above")
    add_metric_for_dashboard("domain_mismatch_pct", "Domain mismatch: %", mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
    record_breach("domain_mismatch_pct", "Domain mismatch: %", mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "domain_mismatch_pct", "Domain mismatch: %", None, mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
    section_comment_ui("domain_match")

    DOMAIN_MISMATCH_EXAMPLES_SQL = f"""
    WITH base AS (
        SELECT d.id, d.country, d.status_code,
               lower(trim(coalesce(d.fqdn, ''))) AS fqdn,
               lower(trim(coalesce(e.email, ''))) AS email
        FROM {BRZ_DOMAINS} d
        LEFT JOIN {BRZ_EMAILS} e ON d.id = e.id
    ),
    parsed AS (
        SELECT *,
               split_part(fqdn, '.', 1) AS company_root,
               split_part(split_part(email, '@', 2), '.', 1) AS email_root
        FROM base
        WHERE fqdn <> '' AND email <> '' AND email LIKE '%@%' AND fqdn LIKE '%.%'
    )
    SELECT id, country, status_code, fqdn, email, company_root, email_root
    FROM parsed
    WHERE company_root <> '' AND email_root <> '' AND company_root <> email_root
    LIMIT 20;
    """
    show_examples("Top 20 domain mismatch examples", DOMAIN_MISMATCH_EXAMPLES_SQL)

    st.divider()

    # 6) Blocked + examples
    st.header("6) Blocked “no-reply” emails")
    BLOCKED_SQL = f"""
    with base as (
      select id, country, email,
        regexp_replace(lower(trim(coalesce(email, ''))),'[^a-z0-9@._%+\\-]','','g') as email_clean
      from {BRZ_EMAILS}
      where email is not null and trim(email) <> ''
    ),
    valid as (
      select *
      from base
      where email_clean like '%@%'
        and email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    ),
    classified as (
      select *,
             case when email_clean ~ '^(no-reply|noreply|do-not-reply|donotreply)@' then 1 else 0 end as is_blocked
      from valid
    )
    select round(100.0 * sum(is_blocked) / nullif(count(*),0), 2) as pct_blocked
    from classified;
    """
    bl = run_sql_df(BLOCKED_SQL)
    st.dataframe(bl, use_container_width=True)
    blocked_pct = float(bl.iloc[0]["pct_blocked"])
    sev = severity_from_threshold(blocked_pct, float(blocked_email_threshold), direction="above")
    add_metric_for_dashboard("blocked_email_pct", "Blocked emails: %", blocked_pct, sev, {"threshold": blocked_email_threshold})
    record_breach("blocked_email_pct", "Blocked emails: %", blocked_pct, sev, {"threshold": blocked_email_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "blocked_email_pct", "Blocked emails: %", None, blocked_pct, sev, {"threshold": blocked_email_threshold})
    section_comment_ui("blocked")

    BLOCKED_EXAMPLES_SQL = f"""
    with base as (
      select id, country, email,
        regexp_replace(lower(trim(coalesce(email, ''))),'[^a-z0-9@._%+\\-]','','g') as email_clean
      from {BRZ_EMAILS}
      where email is not null and trim(email) <> ''
    ),
    valid as (
      select *
      from base
      where email_clean like '%@%'
        and email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    )
    select id, country, email, email_clean
    from valid
    where email_clean ~ '^(no-reply|noreply|do-not-reply|donotreply)@'
    limit 20;
    """
    show_examples("Top 20 blocked email examples", BLOCKED_EXAMPLES_SQL)

    st.divider()

    # 7) Deliverability
    st.header("7) Deliverability status")
    DELIVERABILITY_SQL = f"""
    WITH base AS (SELECT verification_status FROM {BRZ_EMAILS}),
    totals AS (SELECT COUNT(*) AS total_rows FROM base)
    SELECT COALESCE(NULLIF(trim(verification_status), ''), 'NULL') AS verification_status,
           COUNT(*) AS row_count,
           ROUND(COUNT(*)::numeric / NULLIF(totals.total_rows,0) * 100, 2) AS pct
    FROM base CROSS JOIN totals
    GROUP BY COALESCE(NULLIF(trim(verification_status), ''), 'NULL'), totals.total_rows
    ORDER BY row_count DESC;
    """
    dv = run_sql_df(DELIVERABILITY_SQL)
    st.dataframe(dv, use_container_width=True)
    st.plotly_chart(px.bar(dv, x="verification_status", y="pct", hover_data=["row_count"]), use_container_width=True)
    section_comment_ui("deliverability")

    st.divider()

    # 8) >5 emails + examples
    st.header("8) Companies with >5 emails (by country)")
    GT5_SQL = f"""
    with per_company_country as (
      select e.id, coalesce(nullif(trim(e.country), ''), 'UNKNOWN') as country,
             count(*) filter (where e.email is not null and trim(e.email) <> '') as email_count
      from {BRZ_EMAILS} e
      group by e.id, coalesce(nullif(trim(e.country), ''), 'UNKNOWN')
    ),
    country_rollup as (
      select country,
             count(*) as company_country_pairs,
             sum(case when email_count > 5 then 1 else 0 end) as company_country_pairs_gt5
      from per_company_country
      group by country
    )
    select country,
           round(100.0 * company_country_pairs_gt5 / nullif(company_country_pairs,0), 2) as pct_companies_gt5_emails
    from country_rollup
    order by pct_companies_gt5_emails desc;
    """
    gt5 = run_sql_df(GT5_SQL)
    st.dataframe(gt5, use_container_width=True)
    st.plotly_chart(px.bar(gt5.head(20), x="country", y="pct_companies_gt5_emails"), use_container_width=True)
    max_pct = float(gt5["pct_companies_gt5_emails"].max()) if len(gt5) else None
    sev = severity_from_threshold(max_pct, float(gt5_country_threshold), direction="above")
    add_metric_for_dashboard("gt5_email_country_pct_max", "Max country % of companies with >5 emails", max_pct, sev, {"threshold": gt5_country_threshold})
    record_breach("gt5_email_country_pct_max", "Max country % of companies with >5 emails", max_pct, sev, {"threshold": gt5_country_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "gt5_email_country_pct_max", "Max country % of companies with >5 emails", None, max_pct, sev, {"threshold": gt5_country_threshold})

    if not gt5.empty:
        top_country = str(gt5.iloc[0]["country"])
        GT5_EXAMPLES_SQL = f"""
        with per_company_country as (
          select e.id, coalesce(nullif(trim(e.country), ''), 'UNKNOWN') as country,
                 count(*) filter (where e.email is not null and trim(e.email) <> '') as email_count
          from {BRZ_EMAILS} e
          group by e.id, coalesce(nullif(trim(e.country), ''), 'UNKNOWN')
        )
        select id, country, email_count
        from per_company_country
        where country = '{top_country}' and email_count > 5
        order by email_count desc
        limit 20;
        """
        show_examples(f"Top 20 companies with >5 emails (country={top_country})", GT5_EXAMPLES_SQL)

    section_comment_ui("gt5_emails")

with dashboard_tab:
    st.header("Dashboard & Trends")

    if not st.session_state.metrics_for_dashboard:
        st.info("Run checks in the **✅ Checks (Bronze)** tab to populate this dashboard.")
    else:
        dash_df = pd.DataFrame(st.session_state.metrics_for_dashboard)
        st.subheader("Latest run metrics (this session)")
        st.dataframe(dash_df[["metric_label", "pct", "severity"]], use_container_width=True)

        st.subheader("SLA breaches (this session)")
        if st.session_state.breaches:
            st.dataframe(pd.DataFrame(st.session_state.breaches), use_container_width=True)
            st.subheader("Auto-generated action items")
            items = action_items_from_breaches(st.session_state.breaches)
            for i in items:
                st.write(f"- {i}")
        else:
            st.success("No SLA breaches under the current rules.")

        st.divider()
        st.subheader("Release Trend (one bar per release)")

        if not persist_to_db:
            st.info("Enable persistence to store each run. Then trends will show one bar per release.")
        else:
            current_metrics_map = {row["metric_key"]: row["pct"] for row in st.session_state.metrics_for_dashboard if row.get("pct") is not None}
            composite = compute_composite_score(current_metrics_map)

            if st.session_state.run_id and composite is not None:
                upsert_metric(st.session_state.run_id, "composite_score", "Composite DQ score (higher is better)", None, composite, "info", {"components": COMPOSITE_COMPONENTS})

            trend_sql = """
            select
              r.run_ts,
              coalesce(r.release_label, to_char(r.run_ts, 'YYYY-MM-DD')) as release_label,
              m.value_pct::numeric as composite_score
            from dq.email_dq_runs r
            join dq.email_dq_metrics m
              on m.run_id = r.run_id
             and m.metric_key = 'composite_score'
            order by r.run_ts asc;
            """
            trend = run_sql_df(trend_sql)
            if trend.empty:
                st.info("No persisted runs yet. Run checks with persistence enabled at least once.")
            else:
                st.plotly_chart(px.bar(trend, x="release_label", y="composite_score", hover_data=["run_ts"], title="Composite DQ score by release"), use_container_width=True)

    st.divider()
    st.caption("Health check")
    st.dataframe(run_sql_df("select now() as server_time;"), use_container_width=True)
