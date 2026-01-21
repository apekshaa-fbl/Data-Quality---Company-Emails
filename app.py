
# app.py
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

# ============================================================
# Load .env from SAME folder as app.py (as you requested)
# ============================================================
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

NOTION_LINK = "https://www.notion.so/firmable/Company-Emails-RCA-Coverage-gaps-2dfd5c6ffd8780fdb32cfff46548fcfb"

st.set_page_config(page_title="Company Email Data Quality (Bronze)", layout="wide")

st.markdown(
    """
    <div style="padding: 0.6rem 0.9rem; border-radius: 12px; border: 1px solid #e6e6e6;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="margin:0;">Company Email Data Quality – Bronze Layer</h2>
          <p style="margin:0.15rem 0 0 0; color:#6b6b6b;">Release checks • Insights • SLA breaches • Action items • Run logs</p>
        </div>
        <div>
          <a href="""" + NOTION_LINK + """" target="_blank">Open Notion RCA</a>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ============================================================
# Connection helpers (your pattern)
# ============================================================
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

# ============================================================
# Persistence (run logs, metrics, comments) - optional toggle
# ============================================================
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

# ============================================================
# Tables (from your spec)
# ============================================================
SRC_PERSONA = "src_companies.persona"
SRC_EMAILS  = "src_companies.emails_comp"
BRZ_EMAILS  = "zeus_bronze.brz_comp_emails"
BRZ_DOMAINS = "zeus_bronze.brz_comp_domains"

# ============================================================
# SLA helpers
# ============================================================
def severity_from_threshold(pct: Optional[float], threshold: float, direction: str = "above") -> str:
    if pct is None:
        return "unknown"
    if direction == "above":
        return "breach" if pct >= threshold else "ok"
    return "breach" if pct <= threshold else "ok"

def action_items_from_breaches(breaches: List[dict]) -> List[str]:
    items = []
    for b in breaches:
        if b["metric_key"] == "vendor_coverage_max_without_pct":
            items.append(f"[SLA] Reach out to vendor(s) breaching threshold: {b['details'].get('breaching_vendors','(see table)')}")
        elif b["metric_key"] == "domain_status_not_200_pct":
            items.append("[DQ] Investigate non-200 domains: validate vendor domains + improve enrichment retries.")
        elif b["metric_key"] == "invalid_format_pct":
            items.append("[DQ] Improve cleaning rules; block masked/placeholder patterns; fix upstream transformations.")
        elif b["metric_key"] == "duplicate_email_pct":
            items.append("[DQ] Enforce dedupe per (company,country,email_clean); audit vendor duplicates.")
        elif b["metric_key"] == "domain_mismatch_pct":
            items.append("[DQ] Improve domain parsing + mapping; add free-mail exceptions for SMBs; fix orphan/duplicate companies.")
        elif b["metric_key"] == "blocked_email_pct":
            items.append("[DQ] Filter no-reply style emails immediately; expand blocklist patterns.")
        elif b["metric_key"] == "gt5_email_country_pct_max":
            items.append("[DQ] Enforce top-5 ranking + prune noise for countries with high >5 email rates.")
        else:
            items.append(f"[DQ] Review breach: {b['metric_label']}")
    return items

# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.subheader("Run Metadata")
    release_label = st.text_input("Release label (e.g., 2026-01-16 release)")
    created_by = st.text_input("Your name / handle", value=os.getenv("USER", ""))
    run_notes = st.text_area("Run notes (optional)")

    st.divider()
    st.subheader("Dynamic SLA Rules (toggles)")
    vendor_pct_without_email_threshold = st.slider("Vendor pct_without_email breach threshold", 0, 100, 70)
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

# Session state
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

# ============================================================
# Button to run everything
# ============================================================
run_col1, run_col2 = st.columns([1, 2])
with run_col1:
    run_all = st.button("Run all checks", type="primary")
with run_col2:
    st.caption("Tip: Keep your .env in the same folder as app.py. This app loads it automatically.")

if run_all:
    st.session_state.breaches = []
    st.session_state.metrics_for_dashboard = []
    if persist_to_db:
        st.session_state.run_id = insert_run(release_label, created_by, run_notes)
    else:
        st.session_state.run_id = None
    st.success("Checks started. Scroll down for results.")

st.divider()

# ============================================================
# 1) Coverage (Source vendor + Source vs Bronze)
# ============================================================
st.header("1) Email coverage (vendor) + Source vs Bronze comparison")

SOURCE_VENDOR_COVERAGE_SQL = f"""
with company_email_counts as (
  select
    p.source,
    p.id,
    count(e.email) as email_count
  from {SRC_PERSONA} p
  left join {SRC_EMAILS} e
    on p.slug = e.slug
  where status = 'PUBLISHED'
  group by p.source, p.id
)
select
  source,
  count(*) as total_companies,
  sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
  sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
  round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
  round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
from company_email_counts
group by source
order by total_companies desc;
"""

SOURCE_TOTAL_COVERAGE_SQL = f"""
with company_email_counts as (
  select
    p.source,
    p.id,
    count(e.email) as email_count
  from {SRC_PERSONA} p
  left join {SRC_EMAILS} e
    on p.slug = e.slug
  where status = 'PUBLISHED'
  group by p.source, p.id
)
select
  count(distinct id) as total_companies,
  sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
  sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
  round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
  round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
from company_email_counts;
"""

BRONZE_TOTAL_COVERAGE_SQL = f"""
with company_email_counts as (
  select
    d.id,
    count(e.email) as email_count
  from {BRZ_DOMAINS} d
  left join {BRZ_EMAILS} e
    on d.id = e.id
  where d.id is not null
  group by d.id
)
select
  count(distinct id) as total_companies,
  sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
  sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
  round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
  round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
from company_email_counts;
"""

colA, colB = st.columns([1.25, 0.75])

with colA:
    with st.expander("SQL Query (Source: vendor coverage)", expanded=False):
        st.code(SOURCE_VENDOR_COVERAGE_SQL, language="sql")

    df_vendor = run_sql_df(SOURCE_VENDOR_COVERAGE_SQL)
    st.dataframe(df_vendor, use_container_width=True)

    if not df_vendor.empty:
        vendor_df = df_vendor.copy()
        vendor_df["is_breach"] = vendor_df["pct_without_email"].astype(float) >= float(vendor_pct_without_email_threshold)

        fig = px.bar(
            vendor_df.sort_values("pct_without_email", ascending=False),
            x="source",
            y="pct_without_email",
            hover_data=["total_companies", "companies_without_email", "pct_with_email", "is_breach"],
            title=f"Vendor pct_without_email (SLA breach if ≥ {vendor_pct_without_email_threshold}%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        breaching = vendor_df[vendor_df["is_breach"]]["source"].tolist()
        breach_pct_max = float(vendor_df["pct_without_email"].max())
        sev = severity_from_threshold(breach_pct_max, float(vendor_pct_without_email_threshold), direction="above")

        st.markdown("**Insight**")
        if breaching:
            st.warning(f"SLA breach: vendors with pct_without_email ≥ {vendor_pct_without_email_threshold}%: {', '.join(breaching)}")
        else:
            st.success(f"No vendor breaches at the current threshold ({vendor_pct_without_email_threshold}%).")

        add_metric_for_dashboard("vendor_coverage_max_without_pct", "Vendor coverage: max pct_without_email", breach_pct_max, sev, {"breaching_vendors": breaching})
        record_breach("vendor_coverage_max_without_pct", "Vendor coverage: max pct_without_email", breach_pct_max, sev, {"breaching_vendors": breaching})

        if persist_to_db and st.session_state.run_id:
            upsert_metric(
                st.session_state.run_id,
                "vendor_coverage_max_without_pct",
                "Vendor coverage: max pct_without_email",
                None,
                breach_pct_max,
                sev,
                {"threshold": vendor_pct_without_email_threshold, "breaching_vendors": breaching},
            )

section_comment_ui("vendor_coverage")

with colB:
    with st.expander("SQL Query (Source overall)", expanded=False):
        st.code(SOURCE_TOTAL_COVERAGE_SQL, language="sql")
    with st.expander("SQL Query (Bronze overall)", expanded=False):
        st.code(BRONZE_TOTAL_COVERAGE_SQL, language="sql")

    df_src_total = run_sql_df(SOURCE_TOTAL_COVERAGE_SQL)
    df_brz_total = run_sql_df(BRONZE_TOTAL_COVERAGE_SQL)

    st.subheader("Source (overall)")
    st.dataframe(df_src_total, use_container_width=True)
    st.subheader("Bronze (overall)")
    st.dataframe(df_brz_total, use_container_width=True)

    if (not df_src_total.empty) and (not df_brz_total.empty):
        s = df_src_total.iloc[0].to_dict()
        b = df_brz_total.iloc[0].to_dict()

        delta_pct_with = float(b.get("pct_with_email", 0)) - float(s.get("pct_with_email", 0))

        st.markdown("**Insight**")
        st.info(
            "If source vs bronze doesn’t match, common causes include: "
            "different company universe definitions, join-key loss (slug→id), "
            "transformation filtering (dropping invalid/empty emails), "
            "or domain/email presence mismatches between tables."
        )

st.divider()

# ============================================================
# 2) Status Code (Bronze domains)
# ============================================================
st.header("2) Status Code (Domains reachability)")

STATUS_CODE_SQL = f"""
WITH base AS (
  SELECT status_code
  FROM {BRZ_DOMAINS}
),
total AS (
  SELECT COUNT(*)::numeric AS total_domains FROM base
),
counts AS (
  SELECT 'Status Code 200' AS status, COUNT(*)::numeric AS n
  FROM base WHERE status_code = 200
  UNION ALL
  SELECT 'Status Code not 200' AS status, COUNT(*)::numeric AS n
  FROM base WHERE status_code IS NOT NULL AND status_code <> 200
  UNION ALL
  SELECT 'Missing Status Code' AS status, COUNT(*)::numeric AS n
  FROM base WHERE status_code IS NULL
)
SELECT
  status AS "Status",
  n::bigint AS "Count of Domains",
  ROUND(n / NULLIF(total.total_domains, 0) * 100, 2) AS "Pct"
FROM counts CROSS JOIN total
ORDER BY
  CASE status
    WHEN 'Status Code 200' THEN 1
    WHEN 'Status Code not 200' THEN 2
    ELSE 3
  END;
"""

with st.expander("SQL Query (Status Code)", expanded=False):
    st.code(STATUS_CODE_SQL, language="sql")

sc_df = run_sql_df(STATUS_CODE_SQL)
st.dataframe(sc_df, use_container_width=True)
st.plotly_chart(px.bar(sc_df, x="Status", y="Pct", hover_data=["Count of Domains"]), use_container_width=True)

not200_pct = float(sc_df.loc[sc_df["Status"] == "Status Code not 200", "Pct"].iloc[0]) if (sc_df["Status"] == "Status Code not 200").any() else None
sev = severity_from_threshold(not200_pct, float(not200_threshold), direction="above")
add_metric_for_dashboard("domain_status_not_200_pct", "Domains: % not-200", not200_pct, sev, {"threshold": not200_threshold})
record_breach("domain_status_not_200_pct", "Domains: % not-200", not200_pct, sev, {"threshold": not200_threshold})
if persist_to_db and st.session_state.run_id:
    upsert_metric(st.session_state.run_id, "domain_status_not_200_pct", "Domains: % not-200", None, not200_pct, sev, {"threshold": not200_threshold})

st.markdown("**Insight**")
st.info("High 'not 200' can indicate broken websites, blocks, vendor domain issues, or outdated/incorrect domains. Missing status indicates enrichment gaps.")
section_comment_ui("status_code")

st.divider()

# ============================================================
# 3) Proper email format coverage (Bronze)
# ============================================================
st.header("3) Proper email format coverage")

FORMAT_SQL = f"""
with base as (
  select
    regexp_replace(
      lower(trim(coalesce(email, ''))),
      '[^a-z0-9@._%+\\-]',
      '',
      'g'
    ) as email_clean
  from {BRZ_EMAILS}
),
classified as (
  select
    email_clean,
    case
      when email_clean is null or trim(email_clean) = '' then 'empty'
      when email_clean not like '%@%' then 'invalid'
      when email_clean !~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$' then 'invalid'
      else 'valid'
    end as "Format Status"
  from base
)
select
  "Format Status",
  count(*) as "Count of Emails",
  round(100.0 * count(*) / nullif(sum(count(*)) over (),0), 2) as "% of Emails"
from classified
group by "Format Status"
order by "Count of Emails" desc;
"""

FORMAT_INVALID_EXAMPLES_SQL = f"""
with base as (
  select
    regexp_replace(
      lower(trim(coalesce(email, ''))),
      '[^a-z0-9@._%+\\-]',
      '',
      'g'
    ) as email_clean
  from {BRZ_EMAILS}
),
classified as (
  select
    email_clean,
    case
      when email_clean is null or trim(email_clean) = '' then 'empty'
      when email_clean not like '%@%' then 'invalid'
      when email_clean !~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$' then 'invalid'
      else 'valid'
    end as "Format Status"
  from base
)
select email_clean as "Emails"
from classified
where "Format Status" = 'invalid'
limit 20;
"""

with st.expander("SQL Query (Format coverage)", expanded=False):
    st.code(FORMAT_SQL, language="sql")

fmt_df = run_sql_df(FORMAT_SQL)
st.dataframe(fmt_df, use_container_width=True)
st.plotly_chart(px.bar(fmt_df, x="Format Status", y="% of Emails", hover_data=["Count of Emails"]), use_container_width=True)

invalid_pct = float(fmt_df.loc[fmt_df["Format Status"] == "invalid", "% of Emails"].iloc[0]) if (fmt_df["Format Status"] == "invalid").any() else None
sev = severity_from_threshold(invalid_pct, float(invalid_format_threshold), direction="above")
add_metric_for_dashboard("invalid_format_pct", "Email format: % invalid", invalid_pct, sev, {"threshold": invalid_format_threshold})
record_breach("invalid_format_pct", "Email format: % invalid", invalid_pct, sev, {"threshold": invalid_format_threshold})
if persist_to_db and st.session_state.run_id:
    upsert_metric(st.session_state.run_id, "invalid_format_pct", "Email format: % invalid", None, invalid_pct, sev, {"threshold": invalid_format_threshold})

st.markdown("**Insight**")
st.info("Invalid formats are commonly caused by special characters, masked emails, placeholders, or parsing artefacts.")

with st.expander("Examples (top 20 invalid)", expanded=False):
    st.code(FORMAT_INVALID_EXAMPLES_SQL, language="sql")
    st.dataframe(run_sql_df(FORMAT_INVALID_EXAMPLES_SQL), use_container_width=True)

section_comment_ui("format")

st.divider()

# ============================================================
# 4) Duplicate emails
# ============================================================
st.header("4) Duplicate emails")

DUP_SUMMARY_SQL = f"""
WITH base AS (
  SELECT
    id AS firmable_id,
    country,
    regexp_replace(
      lower(trim(coalesce(email,''))),
      '[^a-z0-9@._%+\\-]',
      '',
      'g'
    ) AS email_clean
  FROM {BRZ_EMAILS}
  WHERE email IS NOT NULL
    AND trim(email) <> ''
),
valid AS (
  SELECT *
  FROM base
  WHERE email_clean LIKE '%@%'
    AND email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
),
email_counts AS (
  SELECT
    firmable_id,
    country,
    email_clean,
    COUNT(*) AS occurrences
  FROM valid
  GROUP BY 1,2,3
),
summary AS (
  SELECT
    COUNT(*) AS total_email,
    COUNT(*) FILTER (WHERE occurrences = 1) AS unique_email,
    COUNT(*) FILTER (WHERE occurrences > 1) AS duplicate_email
  FROM email_counts
)
SELECT
  total_email,
  unique_email,
  duplicate_email,
  ROUND(duplicate_email * 100.0 / NULLIF(total_email, 0), 2) AS duplicate_percentage
FROM summary;
"""

DUP_TOP_SQL = f"""
WITH base AS (
  SELECT
    regexp_replace(
      lower(trim(coalesce(email,''))),
      '[^a-z0-9@._%+\\-]',
      '',
      'g'
    ) AS email_clean
  FROM {BRZ_EMAILS}
  WHERE email IS NOT NULL
    AND trim(email) <> ''
),
valid AS (
  SELECT *
  FROM base
  WHERE email_clean LIKE '%@%'
    AND email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
)
SELECT
  email_clean,
  COUNT(*) AS occurrences
FROM valid
GROUP BY email_clean
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;
"""

with st.expander("SQL Query (Duplicate summary)", expanded=False):
    st.code(DUP_SUMMARY_SQL, language="sql")

dup_sum = run_sql_df(DUP_SUMMARY_SQL)
st.dataframe(dup_sum, use_container_width=True)

dup_pct = float(dup_sum.iloc[0]["duplicate_percentage"])
sev = severity_from_threshold(dup_pct, float(duplicate_email_threshold), direction="above")
add_metric_for_dashboard("duplicate_email_pct", "Duplicate emails: %", dup_pct, sev, {"threshold": duplicate_email_threshold})
record_breach("duplicate_email_pct", "Duplicate emails: %", dup_pct, sev, {"threshold": duplicate_email_threshold})
if persist_to_db and st.session_state.run_id:
    upsert_metric(st.session_state.run_id, "duplicate_email_pct", "Duplicate emails: %", None, dup_pct, sev, {"threshold": duplicate_email_threshold})

st.markdown("**Insight**")
st.info("Duplicate emails inflate coverage and reduce signal quality. Review vendor duplicates and enforce dedupe logic per (company,country,email).")

with st.expander("Top 20 duplicate emails", expanded=False):
    st.code(DUP_TOP_SQL, language="sql")
    st.dataframe(run_sql_df(DUP_TOP_SQL), use_container_width=True)

section_comment_ui("duplicates")

st.divider()

# ============================================================
# 5) Domain match rate (Bronze)
# ============================================================
st.header("5) Domain match rate (company domain vs email domain)")

DOMAIN_MATCH_SUMMARY_SQL = f"""
WITH base AS (
    SELECT
        d.id,
        d.country,
        d.status_code,
        lower(trim(coalesce(d.fqdn, ''))) AS fqdn,
        lower(trim(coalesce(e.email, ''))) AS email
    FROM {BRZ_DOMAINS} d
    LEFT JOIN {BRZ_EMAILS} e ON d.id = e.id
),
parsed AS (
    SELECT
        *,
        split_part(fqdn, '.', 1) AS company_root,
        split_part(split_part(email, '@', 2), '.', 1) AS email_root
    FROM base
    WHERE fqdn <> ''
      AND email <> ''
      AND email LIKE '%@%'
      AND fqdn LIKE '%.%'
)
SELECT
  COUNT(*) AS "Total Companies",
  SUM(CASE WHEN company_root <> '' AND email_root <> '' AND company_root = email_root THEN 1 ELSE 0 END) AS "Count of Email and Domain Match",
  ROUND(
    100.0 * SUM(CASE WHEN company_root <> '' AND email_root <> '' AND company_root = email_root THEN 1 ELSE 0 END)
      / NULLIF(COUNT(*), 0),
    2
  ) AS "% Match"
FROM parsed;
"""

DOMAIN_MATCH_MISMATCH_EXAMPLES_SQL = f"""
WITH base AS (
    SELECT
        d.id,
        d.country,
        d.status_code,
        lower(trim(coalesce(d.fqdn, ''))) AS fqdn,
        lower(trim(coalesce(e.email, ''))) AS email
    FROM {BRZ_DOMAINS} d
    LEFT JOIN {BRZ_EMAILS} e ON d.id = e.id
),
parsed AS (
    SELECT
        *,
        split_part(fqdn, '.', 1) AS company_root,
        split_part(split_part(email, '@', 2), '.', 1) AS email_root
    FROM base
    WHERE fqdn <> ''
      AND email <> ''
      AND email LIKE '%@%'
      AND fqdn LIKE '%.%'
)
SELECT
  id, country, status_code, fqdn, email, company_root, email_root
FROM parsed
WHERE company_root <> ''
  AND email_root <> ''
  AND company_root <> email_root
LIMIT 20;
"""

with st.expander("SQL Query (Domain match summary)", expanded=False):
    st.code(DOMAIN_MATCH_SUMMARY_SQL, language="sql")

dm = run_sql_df(DOMAIN_MATCH_SUMMARY_SQL)
st.dataframe(dm, use_container_width=True)

match_pct = float(dm.iloc[0]["% Match"])
mismatch_pct = round(100.0 - match_pct, 2)
sev = severity_from_threshold(mismatch_pct, float(domain_mismatch_threshold), direction="above")
add_metric_for_dashboard("domain_mismatch_pct", "Domain mismatch: % (100 - match%)", mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
record_breach("domain_mismatch_pct", "Domain mismatch: % (100 - match%)", mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
if persist_to_db and st.session_state.run_id:
    upsert_metric(st.session_state.run_id, "domain_mismatch_pct", "Domain mismatch: % (100 - match%)", None, mismatch_pct, sev, {"threshold": domain_mismatch_threshold})

st.markdown("**Insight**")
st.info(
    "Mismatch can be expected for SMBs using free-mail, multi-entity employment, orphan/duplicate companies, "
    "or where company naming doesn’t map cleanly to domains. Improve parsing + mapping where possible."
)

with st.expander("Mismatch examples (top 20)", expanded=False):
    st.code(DOMAIN_MATCH_MISMATCH_EXAMPLES_SQL, language="sql")
    st.dataframe(run_sql_df(DOMAIN_MATCH_MISMATCH_EXAMPLES_SQL), use_container_width=True)

section_comment_ui("domain_match")

st.divider()

# ============================================================
# 6) Blocked emails
# ============================================================
st.header("6) Blocked “no-reply” type emails")

BLOCKED_EMAILS_SQL = f"""
with base as (
  select
    regexp_replace(
      lower(trim(coalesce(email, ''))),
      '[^a-z0-9@._%+\\-]',
      '',
      'g'
    ) as email_clean
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
  select
    *,
    case
      when email_clean ~ '^(no-reply|noreply|do-not-reply|donotreply)@' then 1
      else 0
    end as is_blocked
  from valid
)
select
  count(*) as total_valid_emails,
  sum(is_blocked) as blocked_emails,
  round(100.0 * sum(is_blocked) / nullif(count(*),0), 2) as pct_blocked
from classified;
"""

with st.expander("SQL Query (Blocked emails)", expanded=False):
    st.code(BLOCKED_EMAILS_SQL, language="sql")

bl = run_sql_df(BLOCKED_EMAILS_SQL)
st.dataframe(bl, use_container_width=True)

blocked_pct = float(bl.iloc[0]["pct_blocked"])
sev = severity_from_threshold(blocked_pct, float(blocked_email_threshold), direction="above")
add_metric_for_dashboard("blocked_email_pct", "Blocked emails: %", blocked_pct, sev, {"threshold": blocked_email_threshold})
record_breach("blocked_email_pct", "Blocked emails: %", blocked_pct, sev, {"threshold": blocked_email_threshold})
if persist_to_db and st.session_state.run_id:
    upsert_metric(st.session_state.run_id, "blocked_email_pct", "Blocked emails: %", None, blocked_pct, sev, {"threshold": blocked_email_threshold})

st.markdown("**Insight**")
st.warning("Blocked (no-reply) emails should be removed immediately to protect deliverability and avoid unusable contacts.")
section_comment_ui("blocked")

st.divider()

# ============================================================
# 7) Deliverability
# ============================================================
st.header("7) Deliverability status")

DELIVERABILITY_SQL = f"""
WITH base AS (
  SELECT verification_status
  FROM {BRZ_EMAILS}
),
totals AS (
  SELECT COUNT(*) AS total_rows FROM base
)
SELECT
  COALESCE(NULLIF(trim(verification_status), ''), 'NULL') AS verification_status,
  COUNT(*) AS row_count,
  ROUND(COUNT(*)::numeric / NULLIF(totals.total_rows,0) * 100, 2) AS pct
FROM base
CROSS JOIN totals
GROUP BY COALESCE(NULLIF(trim(verification_status), ''), 'NULL'), totals.total_rows
ORDER BY row_count DESC;
"""

with st.expander("SQL Query (Deliverability)", expanded=False):
    st.code(DELIVERABILITY_SQL, language="sql")

deliv = run_sql_df(DELIVERABILITY_SQL)
st.dataframe(deliv, use_container_width=True)
st.plotly_chart(px.bar(deliv, x="verification_status", y="pct", hover_data=["row_count"]), use_container_width=True)

st.markdown("**Insight**")
st.info("A high proportion of NULL/unknown statuses indicates enrichment gaps. Failed/undeliverable statuses should be excluded or re-verified.")
section_comment_ui("deliverability")

st.divider()

# ============================================================
# 8) Companies with >5 emails
# ============================================================
st.header("8) Companies with > 5 emails (by country)")

GT5_SQL = f"""
with per_company_country as (
  select
    e.id,
    coalesce(nullif(trim(e.country), ''), 'UNKNOWN') as country,
    count(*) filter (where e.email is not null and trim(e.email) <> '') as email_count
  from {BRZ_EMAILS} e
  group by e.id, coalesce(nullif(trim(e.country), ''), 'UNKNOWN')
),
country_rollup as (
  select
    country,
    count(*) as company_country_pairs,
    sum(case when email_count > 5 then 1 else 0 end) as company_country_pairs_gt5
  from per_company_country
  group by country
)
select
  country,
  company_country_pairs as total_company_country,
  company_country_pairs_gt5 as companies_gt5_emails,
  round(100.0 * company_country_pairs_gt5 / nullif(company_country_pairs,0), 2) as pct_companies_gt5_emails
from country_rollup
order by total_company_country desc;
"""

with st.expander("SQL Query (>5 emails)", expanded=False):
    st.code(GT5_SQL, language="sql")

gt5 = run_sql_df(GT5_SQL)
st.dataframe(gt5, use_container_width=True)
st.plotly_chart(px.bar(gt5.sort_values("pct_companies_gt5_emails", ascending=False), x="country", y="pct_companies_gt5_emails"), use_container_width=True)

max_pct = float(gt5["pct_companies_gt5_emails"].max()) if len(gt5) else None
sev = severity_from_threshold(max_pct, float(gt5_country_threshold), direction="above")
add_metric_for_dashboard("gt5_email_country_pct_max", "Max country % of companies with >5 emails", max_pct, sev, {"threshold": gt5_country_threshold})
record_breach("gt5_email_country_pct_max", "Max country % of companies with >5 emails", max_pct, sev, {"threshold": gt5_country_threshold})
if persist_to_db and st.session_state.run_id:
    upsert_metric(st.session_state.run_id, "gt5_email_country_pct_max", "Max country % of companies with >5 emails", None, max_pct, sev, {"threshold": gt5_country_threshold})

st.markdown("**Insight**")
st.info("High >5 email rates indicate noisy sources and require enforcement of top-5 ranking and pruning logic.")
section_comment_ui("gt5_emails")

st.divider()

# ============================================================
# 9) Priority rules (no SQL)
# ============================================================
st.header("9) Priority coverage (Top 5 emails per company per country) — Insights only")
st.markdown("""
**Priority selection rules (recommended):**
1. Valid email format  
2. Remove duplicates (company + country + cleaned email)  
3. Prefer domains with status_code = 200 (confidence)  
4. Company domain ↔ email domain match (with known exceptions)  
5. Exclude blocked/system emails (no-reply/do-not-reply)  
6. Prefer role-based inboxes: sales@, info@, contact@, support@, enquiries@  
""")
section_comment_ui("priority_rules")

st.divider()

# ============================================================
# Dashboard summary
# ============================================================
st.header("Dashboard & Insights Summary")

if st.session_state.metrics_for_dashboard:
    dash_df = pd.DataFrame(st.session_state.metrics_for_dashboard)
    st.dataframe(dash_df[["metric_label", "pct", "severity"]], use_container_width=True)
    st.plotly_chart(px.bar(dash_df, x="metric_label", y="pct", hover_data=["severity"]), use_container_width=True)

    st.subheader("SLA Breaches")
    if st.session_state.breaches:
        bdf = pd.DataFrame(st.session_state.breaches)
        st.dataframe(bdf, use_container_width=True)

        st.subheader("Auto-generated Action Items")
        items = action_items_from_breaches(st.session_state.breaches)
        for i in items:
            st.write(f"- {i}")

        if persist_to_db and st.session_state.run_id:
            upsert_metric(st.session_state.run_id, "action_items", "Auto-generated action items", None, None, "info", {"items": items})
    else:
        st.success("No SLA breaches under the current rules.")
else:
    st.info("Click 'Run all checks' to populate the dashboard.")

st.divider()

# ============================================================
# Run history (if persistence enabled)
# ============================================================
st.header("Run History (logs)")
if not persist_to_db:
    st.info("Enable persistence in the sidebar to store and view run history.")
else:
    runs = run_sql_df("select run_id, run_ts, release_label, created_by, notes from dq.email_dq_runs order by run_ts desc limit 50;")
    st.dataframe(runs, use_container_width=True)

    selected_run = st.selectbox("Select a run_id to view metrics", options=runs["run_id"].tolist() if len(runs) else [])
    if selected_run:
        metrics = run_sql_df("select metric_key, metric_label, value_num, value_pct, severity, details_json from dq.email_dq_metrics where run_id = %s;", params=[selected_run])
        st.dataframe(metrics, use_container_width=True)

        comments = run_sql_df("select metric_key, comment_ts, author, comment_text, sample_link from dq.email_dq_comments where run_id = %s order by comment_ts desc;", params=[selected_run])
        st.dataframe(comments, use_container_width=True)

st.divider()
st.caption("Health check")
st.dataframe(run_sql_df("select now() as server_time;"), use_container_width=True)
