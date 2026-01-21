# app_bronze_only_v2.py
import os
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# -----------------------------
# CONFIG / BOOTSTRAP
# -----------------------------
NOTION_LINK = "https://www.notion.so/firmable/Company-Emails-RCA-Coverage-gaps-2dfd5c6ffd8780fdb32cfff46548fcfb"

st.set_page_config(page_title="Company Email Data Quality", layout="wide")
st.title("Company Email Data Quality – Source vs Bronze")
st.caption("Release checks • Insights • SLA breaches • Action items • Run logs")
st.markdown(f"🔗 **Notion RCA:** [{NOTION_LINK}]({NOTION_LINK})")
st.write("")


def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit Cloud: st.secrets
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            v = st.secrets[key]
            return str(v).strip() if v is not None else default
    except Exception:
        pass

    # Env vars
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else v


def _load_local_env_if_present():
    """
    Local-only helper: if .env exists next to this file, load it.
    Streamlit Cloud won't use it, but it doesn't hurt locally.
    """
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


_load_local_env_if_present()


def get_pg_conn():
    required = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    missing = [k for k in required if not _get_setting(k)]
    if missing:
        st.error(f"Missing env vars / secrets: {', '.join(missing)}")
        st.info(
            "On Streamlit Cloud, set these in **Manage app → Secrets**.\n\n"
            "Example:\n"
            "PGHOST=\"...\"\nPGPORT=\"5432\"\nPGDATABASE=\"...\"\nPGUSER=\"...\"\nPGPASSWORD=\"...\"\nPGSSLMODE=\"require\""
        )
        st.stop()

    return psycopg2.connect(
        host=_get_setting("PGHOST"),
        port=int(_get_setting("PGPORT", "5432")),
        dbname=_get_setting("PGDATABASE"),
        user=_get_setting("PGUSER"),
        password=_get_setting("PGPASSWORD"),
        sslmode=_get_setting("PGSSLMODE", "require"),
    )


@st.cache_data(ttl=300, show_spinner=False)
def run_sql_df(sql: str, params: Optional[Union[Tuple, Dict]] = None) -> pd.DataFrame:
    with get_pg_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def exec_sql(sql: str, params: Optional[Union[Tuple, Dict]] = None) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


# -----------------------------
# LOG TABLES (runs + metrics + comments)
# -----------------------------
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
      VALUES (%(run_id)s, %(release_label)s, %(created_by)s, %(notes)s)
    """, {
        "run_id": run_id,
        "release_label": release_label,
        "created_by": created_by,
        "notes": notes
    })
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
      VALUES (%(run_id)s, %(metric_key)s, %(author)s, %(comment_text)s, %(sample_link)s)
    """, {
        "run_id": run_id,
        "metric_key": metric_key,
        "author": author,
        "comment_text": comment_text,
        "sample_link": sample_link
    })


# -----------------------------
# SLA HELPERS
# -----------------------------
def severity_from_threshold(pct: Optional[float], threshold: float, direction: str = "above") -> str:
    if pct is None:
        return "unknown"
    if direction == "above":
        return "breach" if pct >= threshold else "ok"
    return "breach" if pct <= threshold else "ok"


def action_items_from_breaches(breaches: List[dict]) -> List[str]:
    items = []
    for b in breaches:
        mk = b["metric_key"]
        if mk == "coverage_without_email_pct":
            items.append("[SLA] Low email coverage — investigate ingestion gaps and ID mapping.")
        elif mk == "domain_status_not_200_pct":
            items.append("[SLA] High non-200 domains — improve enrichment/retry and validate vendor domains.")
        elif mk == "invalid_format_pct":
            items.append("[DQ] High invalid formats — improve cleaning, block masked/placeholder values, fix upstream.")
        elif mk == "duplicate_email_pct":
            items.append("[DQ] High duplicates — enforce dedupe and identify duplicate-producing source.")
        elif mk == "domain_mismatch_pct":
            items.append("[DQ] High mismatch — improve parsing/mapping + controlled exceptions (free-mail, group domains).")
        elif mk == "blocked_email_pct":
            items.append("[DQ] High blocked emails — expand blocklist and drop no-reply early.")
        elif mk == "gt5_email_country_pct_max":
            items.append("[DQ] High >5 emails rate — enforce top-5 policy and prune noisy emails per country.")
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


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.subheader("Run Metadata")
    release_label = st.text_input("Release label (e.g., 2026-01-16 release)")
    created_by = st.text_input("Your name / handle", value=os.getenv("USER", ""))  # local convenience
    run_notes = st.text_area("Run notes (optional)")

    st.divider()
    st.subheader("Tables")
    st.caption("Bronze tables (fixed)")
    BRZ_COMPANY = "zeus_bronze.brz_company_hub"
    BRZ_EMAILS = "zeus_bronze.brz_comp_emails"
    BRZ_DOMAINS = "zeus_bronze.brz_comp_domains"
    st.code(f"{BRZ_COMPANY}\n{BRZ_EMAILS}\n{BRZ_DOMAINS}")

    st.caption("Source tables (editable)")
    SRC_COMPANY = st.text_input("Source company hub table", value="src_companies.persona")
    SRC_EMAILS = st.text_input("Source emails table", value="src_companies.emails")
    SRC_DOMAINS = st.text_input("Source domains table", value="src_companies.domains")

    st.divider()
    st.subheader("SLA Rules")
    vendor_reachout_threshold = st.slider("Vendor reach-out threshold (% without email)", 0, 100, 70)

    coverage_without_threshold = st.slider("Coverage breach (% without email)", 0, 100, 30)
    blocked_email_threshold = st.slider("Blocked email % breach", 0, 100, 1)
    duplicate_email_threshold = st.slider("Duplicate email % breach", 0, 100, 5)
    invalid_format_threshold = st.slider("Invalid format % breach", 0, 100, 2)
    domain_mismatch_threshold = st.slider("Domain mismatch % breach", 0, 100, 30)
    not200_threshold = st.slider("Not-200 domain % breach", 0, 100, 30)
    gt5_country_threshold = st.slider("Country >5 emails % breach", 0, 100, 2)

    st.divider()
    st.subheader("Persistence")
    persist_to_db = st.toggle("Save this run + comments to Postgres (dq schema)", value=False)

if persist_to_db:
    try:
        ensure_log_tables()
    except Exception as e:
        st.error(f"Failed to initialise dq log tables: {e}")
        persist_to_db = False


# -----------------------------
# SESSION STATE
# -----------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "breaches" not in st.session_state:
    st.session_state.breaches = []
if "metrics_for_dashboard" not in st.session_state:
    st.session_state.metrics_for_dashboard = []


def section_query(sql: str, title: str = "SQL Query"):
    # “Dropdown that shows the query”: expander works best in Streamlit
    with st.expander(title, expanded=False):
        st.code(sql, language="sql")


def section_comment_ui(metric_key: str):
    st.markdown("**Comments & Sample Link**")
    c1, c2 = st.columns([2, 1])
    with c1:
        comment_text = st.text_area("Comment", key=f"comment_{metric_key}")
    with c2:
        sample_link = st.text_input("Link (Excel/GSheet/Jira/etc.)", key=f"link_{metric_key}")

    if st.button("Save comment", key=f"save_{metric_key}"):
        if not persist_to_db:
            st.info("Enable **Save this run + comments** in the sidebar to persist comments.")
        elif not st.session_state.run_id:
            st.warning("Run not started yet. Click **Run all checks** first.")
        else:
            add_comment(st.session_state.run_id, metric_key, created_by, comment_text, sample_link)
            st.success("Saved.")


def add_metric(metric_key: str, metric_label: str, pct: Optional[float], severity: str, details: Dict[str, Any]):
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
    section_query(sql, title=f"SQL Query (Examples: {title})")
    df = run_sql_df(sql)
    st.dataframe(df, use_container_width=True)


# -----------------------------
# TABS
# -----------------------------
checks_tab, dashboard_tab = st.tabs(["✅ Checks", "📊 Dashboard & Trends"])

with checks_tab:
    run_all = st.button("Run all checks", type="primary")

    if run_all:
        st.session_state.breaches = []
        st.session_state.metrics_for_dashboard = []
        if persist_to_db:
            st.session_state.run_id = insert_run(release_label, created_by, run_notes)
        else:
            st.session_state.run_id = None
        st.success("Run started. Scroll down for results.")

    st.divider()

    # ======================================================
    # 1) COVERAGE: BRONZE + SOURCE + VENDOR CHART + COMPARISON
    # ======================================================
    st.header("1) Coverage (Bronze + Source)")

    COVERAGE_BRONZE_SQL = f"""
    with email_per_company as (
        select h.id, count(e.email) as email_count
        from {BRZ_COMPANY} h
        left join {BRZ_EMAILS} e
          on h.id = e.id
         and e.email is not null
         and trim(e.email) <> ''
        group by h.id
    )
    select
        count(*) as total_companies,
        sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
        sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
        round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
        round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
    from email_per_company;
    """

    COVERAGE_BY_SOURCE_BRONZE_SQL = f"""
    with company_email_counts as (
        select
            p.source,
            p.id,
            count(e.email) filter (where e.email is not null and trim(e.email) <> '') as email_count
        from {BRZ_COMPANY} p
        left join {BRZ_EMAILS} e
          on p.id = e.id
        group by p.source, p.id
    )
    select
        source,
        count(distinct id) as total_companies,
        sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
        sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
        round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
        round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
    from company_email_counts
    group by source
    order by source;
    """

    # SOURCE version (table names are editable in sidebar)
    COVERAGE_SOURCE_SQL = f"""
    with email_per_company as (
        select h.id, count(e.email) as email_count
        from {SRC_COMPANY} h
        left join {SRC_EMAILS} e
          on h.id = e.id
         and e.email is not null
         and trim(e.email) <> ''
        group by h.id
    )
    select
        count(*) as total_companies,
        sum(case when email_count >= 1 then 1 else 0 end) as companies_with_email,
        sum(case when email_count = 0 then 1 else 0 end) as companies_without_email,
        round(100.0 * sum(case when email_count >= 1 then 1 else 0 end) / nullif(count(*),0), 2) as pct_with_email,
        round(100.0 * sum(case when email_count = 0 then 1 else 0 end) / nullif(count(*),0), 2) as pct_without_email
    from email_per_company;
    """

    section_query(COVERAGE_BRONZE_SQL, "SQL Query (Bronze overall coverage)")
    cov_brz = run_sql_df(COVERAGE_BRONZE_SQL)
    st.subheader("Bronze overall coverage")
    st.dataframe(cov_brz, use_container_width=True)

    section_query(COVERAGE_BY_SOURCE_BRONZE_SQL, "SQL Query (Bronze coverage by vendor/source)")
    cov_src_brz = run_sql_df(COVERAGE_BY_SOURCE_BRONZE_SQL)

    st.subheader("Bronze coverage by vendor/source (chart only)")
    # reach-out classification
    cov_src_brz["reach_out_flag"] = cov_src_brz["pct_without_email"] >= float(vendor_reachout_threshold)
    cov_src_brz["reach_out_label"] = cov_src_brz["reach_out_flag"].map(lambda x: "Reach out" if x else "OK")

    fig = px.bar(
        cov_src_brz.sort_values("pct_without_email", ascending=False),
        x="source",
        y="pct_without_email",
        color="reach_out_label",
        hover_data=["total_companies", "companies_without_email", "pct_with_email"],
        title=f"Vendors by % companies without email (Reach out if ≥ {vendor_reachout_threshold}%)"
    )
    st.plotly_chart(fig, use_container_width=True)

    bad_vendors = cov_src_brz.loc[cov_src_brz["reach_out_flag"], "source"].tolist()
    if bad_vendors:
        st.warning(
            f"**Action:** Vendors with **pct_without_email ≥ {vendor_reachout_threshold}%** should be reached out for data coverage.\n\n"
            f"**Vendors:** {', '.join(map(str, bad_vendors[:25]))}{' ...' if len(bad_vendors) > 25 else ''}"
        )
    else:
        st.success(f"No vendors above the {vendor_reachout_threshold}% reach-out threshold.")

    # Bronze: check if overall matches rollup
    # rollup totals
    roll_total = float(cov_src_brz["total_companies"].sum()) if not cov_src_brz.empty else None
    roll_with = float(cov_src_brz["companies_with_email"].sum()) if not cov_src_brz.empty else None
    roll_without = float(cov_src_brz["companies_without_email"].sum()) if not cov_src_brz.empty else None

    overall_total = float(cov_brz.iloc[0]["total_companies"])
    overall_with = float(cov_brz.iloc[0]["companies_with_email"])
    overall_without = float(cov_brz.iloc[0]["companies_without_email"])

    # mismatch insight
    if roll_total is not None and (roll_total != overall_total or roll_with != overall_with or roll_without != overall_without):
        st.error("⚠️ **Mismatch detected** between overall coverage and vendor rollup totals.")
        st.write(
            f"- Overall totals: total={int(overall_total)}, with_email={int(overall_with)}, without_email={int(overall_without)}\n"
            f"- Vendor rollup: total={int(roll_total)}, with_email={int(roll_with)}, without_email={int(roll_without)}"
        )
        st.info(
            "**Most likely reasons:**\n"
            "- Vendor rollup uses `count(distinct id)` per source, but overall uses `count(*)` over a different base set.\n"
            "- Companies may have `source` NULL/blank or inconsistent → rollup grouping changes totals.\n"
            "- Join/filter differences: email trimming + null filters applied differently.\n"
            "- Duplicate `id` rows in hub/source mapping → rollup and overall diverge.\n"
            "\n**Recommendation:** Ensure both queries use the same base (same hub table, same ID distinctness rules, same email filters)."
        )
    else:
        st.success("Overall coverage and vendor rollup totals match (Bronze).")

    # Record SLA metric
    pct_with = float(cov_brz.iloc[0]["pct_with_email"])
    pct_without = float(cov_brz.iloc[0]["pct_without_email"])
    sev = severity_from_threshold(pct_without, float(coverage_without_threshold), direction="above")

    add_metric("coverage_with_email_pct", "Coverage: % companies with ≥1 email", pct_with, "ok", {"pct_without_email": pct_without})
    add_metric("coverage_without_email_pct", "Coverage: % companies without email", pct_without, sev, {"threshold": coverage_without_threshold})
    record_breach("coverage_without_email_pct", "Coverage: % companies without email", pct_without, sev, {"threshold": coverage_without_threshold})

    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "coverage_with_email_pct", "Coverage: % companies with ≥1 email", None, pct_with, "ok", {"pct_without_email": pct_without})
        upsert_metric(st.session_state.run_id, "coverage_without_email_pct", "Coverage: % companies without email", None, pct_without, sev, {"threshold": coverage_without_threshold})

    # Show Source overall coverage (so you can compare source vs bronze quickly)
    st.subheader("Source overall coverage (for comparison)")
    section_query(COVERAGE_SOURCE_SQL, "SQL Query (Source overall coverage)")
    try:
        cov_src = run_sql_df(COVERAGE_SOURCE_SQL)
        st.dataframe(cov_src, use_container_width=True)
        st.info(
            "If Source vs Bronze coverage differs, common causes are:\n"
            "- Source ingestion gaps vs bronze enrichment\n"
            "- Different dedupe rules\n"
            "- ID mapping differences (source IDs not present in bronze hub)\n"
            "- Timing: bronze updated after source snapshot"
        )
    except Exception as e:
        st.warning(f"Could not run Source coverage query. Check Source table names in sidebar.\n\nError: {e}")

    section_comment_ui("coverage")

    st.divider()

    # ======================================================
    # 2) STATUS CODE (Bronze)
    # ======================================================
    st.header("2) Domain status code (Bronze)")
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
    section_query(STATUS_CODE_SQL, "SQL Query (Status Code)")
    sc = run_sql_df(STATUS_CODE_SQL)
    st.dataframe(sc, use_container_width=True)
    st.plotly_chart(px.bar(sc, x="Status", y="Pct", hover_data=["Count of Domains"]), use_container_width=True)

    not200_pct = float(sc.loc[sc["Status"] == "Status Code not 200", "Pct"].iloc[0]) if (sc["Status"] == "Status Code not 200").any() else None
    sev = severity_from_threshold(not200_pct, float(not200_threshold), direction="above")

    st.info(
        "**Insight:** Non-200 and missing status codes reduce confidence in domain matching and deliverability.\n"
        "Prioritise retries, validation and vendor follow-ups for high non-200 rates."
    )

    add_metric("domain_status_not_200_pct", "Domains: % not-200", not200_pct, sev, {"threshold": not200_threshold})
    record_breach("domain_status_not_200_pct", "Domains: % not-200", not200_pct, sev, {"threshold": not200_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "domain_status_not_200_pct", "Domains: % not-200", None, not200_pct, sev, {"threshold": not200_threshold})

    section_comment_ui("status_code")

    st.divider()

    # ======================================================
    # 3) EMAIL FORMAT + EXAMPLES
    # ======================================================
    st.header("3) Proper email format coverage (Bronze)")
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
    section_query(FORMAT_SQL, "SQL Query (Format Summary)")
    fmt = run_sql_df(FORMAT_SQL)
    st.dataframe(fmt, use_container_width=True)
    st.plotly_chart(px.bar(fmt, x="format_status", y="pct", hover_data=["email_rows"]), use_container_width=True)

    invalid_pct = float(fmt.loc[fmt["format_status"] == "invalid", "pct"].iloc[0]) if (fmt["format_status"] == "invalid").any() else None
    sev = severity_from_threshold(invalid_pct, float(invalid_format_threshold), direction="above")

    st.info(
        "**Insight:** Some companies may have emails but still fail format checks.\n"
        "This is commonly caused by special characters, masked/obfuscated emails, or scraping artefacts."
    )

    add_metric("invalid_format_pct", "Email format: % invalid", invalid_pct, sev, {"threshold": invalid_format_threshold})
    record_breach("invalid_format_pct", "Email format: % invalid", invalid_pct, sev, {"threshold": invalid_format_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "invalid_format_pct", "Email format: % invalid", None, invalid_pct, sev, {"threshold": invalid_format_threshold})

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
    section_comment_ui("format")

    st.divider()

    # ======================================================
    # 4) DUPLICATES + TOP20 + BUCKETS + UNIQUE DUP COUNT
    # ======================================================
    st.header("4) Duplicate emails (Bronze)")

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
    section_query(DUP_SUMMARY_SQL, "SQL Query (Duplicate Summary)")
    dup = run_sql_df(DUP_SUMMARY_SQL)
    st.dataframe(dup, use_container_width=True)

    dup_pct = float(dup.iloc[0]["duplicate_percentage"])
    sev = severity_from_threshold(dup_pct, float(duplicate_email_threshold), direction="above")

    # Unique duplicate email count (distinct email_clean where occurrences>1)
    UNIQUE_DUP_EMAIL_COUNT_SQL = f"""
    with base as (
      select regexp_replace(lower(trim(coalesce(email,''))),'[^a-z0-9@._%+\\-]','','g') as email_clean
      from {BRZ_EMAILS}
      where email is not null and trim(email) <> ''
    ),
    valid as (
      select *
      from base
      where email_clean like '%@%'
        and email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    ),
    occ as (
      select email_clean, count(*) as occurrences
      from valid
      group by 1
      having count(*) > 1
    )
    select count(*)::bigint as unique_duplicate_emails
    from occ;
    """
    section_query(UNIQUE_DUP_EMAIL_COUNT_SQL, "SQL Query (Unique duplicate emails count)")
    uniq_dup = run_sql_df(UNIQUE_DUP_EMAIL_COUNT_SQL)
    unique_dup_count = int(uniq_dup.iloc[0]["unique_duplicate_emails"]) if not uniq_dup.empty else None

    st.info(
        f"**Insight:** There are **{unique_dup_count:,} unique duplicate emails** in Bronze (emails appearing more than once)."
        if unique_dup_count is not None else
        "**Insight:** Duplicate emails are present and should be investigated by source and ingestion rules."
    )

    add_metric("duplicate_email_pct", "Duplicate emails: %", dup_pct, sev, {"threshold": duplicate_email_threshold, "unique_duplicate_emails": unique_dup_count})
    record_breach("duplicate_email_pct", "Duplicate emails: %", dup_pct, sev, {"threshold": duplicate_email_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "duplicate_email_pct", "Duplicate emails: %", None, dup_pct, sev,
                      {"threshold": duplicate_email_threshold, "unique_duplicate_emails": unique_dup_count})

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

    DUP_BUCKET_SQL = f"""
    with base as (
      select id as firmable_id,
             regexp_replace(lower(trim(coalesce(email,''))),'[^a-z0-9@._%+\\-]','','g') as email_clean
      from {BRZ_EMAILS}
      where email is not null and trim(email) <> ''
    ),
    valid as (
      select *
      from base
      where email_clean like '%@%'
        and email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    ),
    email_occurrences as (
      select email_clean, count(*) as occurrences
      from valid
      group by 1
      having count(*) > 1
    ),
    company_email_dupes as (
      select v.firmable_id, eo.email_clean, eo.occurrences
      from valid v
      join email_occurrences eo on v.email_clean = eo.email_clean
    ),
    bucketed as (
      select firmable_id, email_clean, occurrences,
        case
          when occurrences < 50 then '<50'
          when occurrences between 50 and 100 then '50-100'
          when occurrences between 101 and 250 then '101-250'
          when occurrences between 251 and 300 then '251-300'
          else '>300'
        end as occ_bucket
      from company_email_dupes
    ),
    total_companies as (
      select count(distinct firmable_id) as total_companies
      from valid
    ),
    bucket_stats as (
      select occ_bucket,
             count(distinct email_clean) as duplicate_email_count,
             count(distinct firmable_id) as companies_in_bucket
      from bucketed
      group by 1
    )
    select
      bs.occ_bucket,
      bs.duplicate_email_count,
      bs.companies_in_bucket,
      round(bs.companies_in_bucket * 100.0 / nullif(tc.total_companies, 0), 2) as pct_of_companies
    from bucket_stats bs
    cross join total_companies tc
    order by
      case bs.occ_bucket
        when '<50' then 1
        when '50-100' then 2
        when '101-250' then 3
        when '251-300' then 4
        when '>300' then 5
        else 99
      end;
    """
    section_query(DUP_BUCKET_SQL, "SQL Query (Duplicate buckets)")
    dup_bucket = run_sql_df(DUP_BUCKET_SQL)
    st.dataframe(dup_bucket, use_container_width=True)

    section_comment_ui("duplicates")

    st.divider()

    # ======================================================
    # 5) DOMAIN MATCH RATE (use your robust company/email logic by country)
    # ======================================================
    st.header("5) Domain match rate between company domain and email domain (Bronze)")

    DOMAIN_MATCH_COUNTRY_SQL = f"""
    WITH company AS (
        SELECT
            h.id,
            h.name,
            lower(
                trim(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(coalesce(h.fqdn, ''), '^https?://', '', 'i'),
                            '/.*$','', 'g'
                        ),
                        '^www\\.', '', 'i'
                    )
                )
            ) AS company_domain,
            split_part(
                lower(
                    trim(
                        regexp_replace(
                            regexp_replace(
                                regexp_replace(coalesce(h.fqdn, ''), '^https?://', '', 'i'),
                                '/.*$','', 'g'
                            ),
                            '^www\\.', '', 'i'
                        )
                    )
                ),
                '.', 1
            ) AS company_root,
            lower(regexp_replace(coalesce(h.name, ''), '[^a-z0-9]+', '', 'g')) AS company_name_clean
        FROM {BRZ_COMPANY} h
        WHERE coalesce(trim(h.fqdn), '') <> ''
          AND h.fqdn LIKE '%.%'
    ),
    emails AS (
        SELECT
            e.id,
            e.country,
            lower(trim(e.email)) AS email,
            lower(trim(split_part(e.email, '@', 1))) AS local_part,
            lower(trim(split_part(e.email, '@', 2))) AS email_domain,
            lower(regexp_replace(split_part(e.email, '@', 1), '[^a-z0-9]+', '', 'g')) AS local_part_clean
        FROM {BRZ_EMAILS} e
        WHERE e.email IS NOT NULL
          AND trim(e.email) <> ''
          AND e.email LIKE '%@%'
    ),
    email_eval AS (
        SELECT
            c.id,
            e.country,
            (
              e.email_domain = c.company_domain
              OR e.email_domain LIKE '%.' || c.company_domain
              OR c.company_domain LIKE '%.' || e.email_domain
              OR (c.company_root <> '' AND e.email_domain LIKE '%' || c.company_root || '%')
              OR (c.company_root <> '' AND e.local_part LIKE '%' || c.company_root || '%')
              OR (
                c.company_name_clean <> ''
                AND length(c.company_name_clean) >= 4
                AND e.local_part_clean LIKE '%' || c.company_name_clean || '%'
              )
            ) AS is_valid_email
        FROM company c
        JOIN emails e ON e.id = c.id
    ),
    company_status AS (
        SELECT
            id,
            country,
            bool_or(is_valid_email) AS has_any_valid_email
        FROM email_eval
        GROUP BY id, country
    )
    SELECT
      country,
      count(*) AS total_companies,
      sum(CASE WHEN has_any_valid_email THEN 1 ELSE 0 END) AS companies_valid,
      round(100.0 * sum(CASE WHEN has_any_valid_email THEN 1 ELSE 0 END) / nullif(count(*),0), 2) AS pct_companies_valid,
      sum(CASE WHEN NOT has_any_valid_email THEN 1 ELSE 0 END) AS companies_invalid,
      round(100.0 * sum(CASE WHEN NOT has_any_valid_email THEN 1 ELSE 0 END) / nullif(count(*),0), 2) AS pct_companies_invalid
    FROM company_status
    GROUP BY country
    ORDER BY total_companies DESC;
    """

    section_query(DOMAIN_MATCH_COUNTRY_SQL, "SQL Query (Domain match rate by country)")
    dm_country = run_sql_df(DOMAIN_MATCH_COUNTRY_SQL)
    st.dataframe(dm_country, use_container_width=True)

    # Build mismatch metric from overall weighted average (approx)
    total = dm_country["total_companies"].sum() if not dm_country.empty else 0
    weighted_valid = (
        (dm_country["pct_companies_valid"] * dm_country["total_companies"]).sum() / total
        if total else None
    )
    mismatch_pct = round(100.0 - float(weighted_valid), 2) if weighted_valid is not None else None
    sev = severity_from_threshold(mismatch_pct, float(domain_mismatch_threshold), direction="above")

    st.info(
        "**Insights to improve domain match:**\n"
        "- For smaller businesses, allow controlled free-mail exceptions (gmail/outlook) where appropriate.\n"
        "- For large groups, staff may email from parent/group domains.\n"
        "- Person→company mapping issues can push emails under the wrong company.\n"
        "- Orphan/duplicate companies reduce domain alignment.\n"
        "- Some company names genuinely differ from domain branding."
    )

    add_metric("domain_mismatch_pct", "Domain mismatch: %", mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
    record_breach("domain_mismatch_pct", "Domain mismatch: %", mismatch_pct, sev, {"threshold": domain_mismatch_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "domain_mismatch_pct", "Domain mismatch: %", None, mismatch_pct, sev, {"threshold": domain_mismatch_threshold})

    section_comment_ui("domain_match")

    st.divider()

    # ======================================================
    # 6) BLOCKED NO-REPLY
    # ======================================================
    st.header("6) Blocked “no-reply” emails (Bronze)")
    BLOCKED_SQL = f"""
    with base as (
      select
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
    select
      count(*) as total_valid_emails,
      sum(is_blocked) as blocked_emails,
      round(100.0 * sum(is_blocked) / nullif(count(*),0), 2) as pct_blocked
    from classified;
    """
    section_query(BLOCKED_SQL, "SQL Query (Blocked Summary)")
    bl = run_sql_df(BLOCKED_SQL)
    st.dataframe(bl, use_container_width=True)

    blocked_pct = float(bl.iloc[0]["pct_blocked"])
    sev = severity_from_threshold(blocked_pct, float(blocked_email_threshold), direction="above")
    st.warning("**Insight:** Blocked emails (no-reply etc.) should be removed immediately as they harm outreach and deliverability.")

    add_metric("blocked_email_pct", "Blocked emails: %", blocked_pct, sev, {"threshold": blocked_email_threshold})
    record_breach("blocked_email_pct", "Blocked emails: %", blocked_pct, sev, {"threshold": blocked_email_threshold})
    if persist_to_db and st.session_state.run_id:
        upsert_metric(st.session_state.run_id, "blocked_email_pct", "Blocked emails: %", None, blocked_pct, sev, {"threshold": blocked_email_threshold})

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
    section_comment_ui("blocked")

    st.divider()

    # ======================================================
    # 7) DELIVERABILITY STATUS
    # ======================================================
    st.header("7) Deliverability status (Bronze)")
    DELIVERABILITY_SQL = f"""
    WITH base AS (SELECT verification_status FROM {BRZ_EMAILS}),
    totals AS (SELECT COUNT(*) AS total_rows FROM base)
    SELECT
      COALESCE(NULLIF(trim(verification_status), ''), 'NULL') AS verification_status,
      COUNT(*) AS row_count,
      ROUND(COUNT(*)::numeric / NULLIF(totals.total_rows,0) * 100, 2) AS pct
    FROM base CROSS JOIN totals
    GROUP BY COALESCE(NULLIF(trim(verification_status), ''), 'NULL'), totals.total_rows
    ORDER BY row_count DESC;
    """
    section_query(DELIVERABILITY_SQL, "SQL Query (Deliverability)")
    dv = run_sql_df(DELIVERABILITY_SQL)
    st.dataframe(dv, use_container_width=True)
    st.plotly_chart(px.bar(dv, x="verification_status", y="pct", hover_data=["row_count"]), use_container_width=True)

    st.info(
        "**Insight:** A high share of 'unknown'/'NULL' deliverability means we cannot reliably prioritise outreach.\n"
        "Improve validation coverage or backfill verification status."
    )
    section_comment_ui("deliverability")

    st.divider()

    # ======================================================
    # 8) COMPANIES WITH >5 EMAILS (show <=5 and >5)
    # ======================================================
    st.header("8) Companies with email count buckets (≤5 vs >5) by country (Bronze)")

    GT5_ROLLUP_SQL = f"""
    WITH per_company_country AS (
      SELECT
        e.id,
        coalesce(nullif(trim(e.country), ''), 'UNKNOWN') AS country,
        count(*) FILTER (WHERE e.email IS NOT NULL AND trim(e.email) <> '') AS email_count
      FROM {BRZ_EMAILS} e
      GROUP BY e.id, coalesce(nullif(trim(e.country), ''), 'UNKNOWN')
    ),
    country_rollup AS (
      SELECT
        country,
        count(*) AS total_company_country,
        sum(CASE WHEN email_count <= 5 THEN 1 ELSE 0 END) AS companies_le5_emails,
        round(100.0 * sum(CASE WHEN email_count <= 5 THEN 1 ELSE 0 END) / nullif(count(*), 0), 2) AS pct_companies_le5_emails,
        sum(CASE WHEN email_count > 5 THEN 1 ELSE 0 END) AS companies_gt5_emails,
        round(100.0 * sum(CASE WHEN email_count > 5 THEN 1 ELSE 0 END) / nullif(count(*), 0), 2) AS pct_companies_gt5_emails
      FROM per_company_country
      GROUP BY country
    )
    SELECT *
    FROM country_rollup
    ORDER BY total_company_country DESC;
    """
    section_query(GT5_ROLLUP_SQL, "SQL Query (≤5 vs >5 by country)")
    gt5_roll = run_sql_df(GT5_ROLLUP_SQL)
    st.dataframe(gt5_roll, use_container_width=True)

    if not gt5_roll.empty:
        top = gt5_roll.sort_values("pct_companies_gt5_emails", ascending=False).iloc[0]
        st.warning(
            f"**Insight:** Country **{top['country']}** has the highest share of companies with **>5 emails** "
            f"({top['pct_companies_gt5_emails']}%). This likely indicates noisy sources / excessive scraping.\n"
            f"Enforce top-5 ranking policy and prune low-quality emails."
        )

        max_pct = float(gt5_roll["pct_companies_gt5_emails"].max())
        sev = severity_from_threshold(max_pct, float(gt5_country_threshold), direction="above")
        add_metric("gt5_email_country_pct_max", "Max country % of companies with >5 emails", max_pct, sev, {"threshold": gt5_country_threshold})
        record_breach("gt5_email_country_pct_max", "Max country % of companies with >5 emails", max_pct, sev, {"threshold": gt5_country_threshold})
        if persist_to_db and st.session_state.run_id:
            upsert_metric(st.session_state.run_id, "gt5_email_country_pct_max", "Max country % of companies with >5 emails", None, max_pct, sev, {"threshold": gt5_country_threshold})

    section_comment_ui("gt5_emails")

    st.divider()

    # ======================================================
    # 9) PRIORITY COVERAGE (Top 5 emails per company per country)
    # NOTE: This is a template. Adjust exact "source tables" and join keys as needed.
    # ======================================================
    st.header("9) Priority coverage: rank emails & select best 5 (template)")

    PRIORITY_TOP5_SQL = f"""
    WITH base AS (
      SELECT
        e.id,
        coalesce(nullif(trim(e.country), ''), 'UNKNOWN') AS country,
        lower(trim(e.email)) AS email,
        regexp_replace(lower(trim(coalesce(e.email,''))),'[^a-z0-9@._%+\\-]','','g') AS email_clean,
        lower(split_part(e.email, '@', 1)) AS local_part,
        lower(split_part(e.email, '@', 2)) AS email_domain,
        coalesce(nullif(trim(e.verification_status), ''), 'NULL') AS verification_status
      FROM {BRZ_EMAILS} e
      WHERE e.email IS NOT NULL AND trim(e.email) <> ''
    ),
    format_ok AS (
      SELECT *,
        CASE WHEN email_clean LIKE '%@%'
          AND email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
        THEN 1 ELSE 0 END AS is_valid_format,
        CASE WHEN email_clean ~ '^(no-reply|noreply|do-not-reply|donotreply)@' THEN 1 ELSE 0 END AS is_blocked
      FROM base
    ),
    dup_flag AS (
      SELECT *,
        CASE WHEN count(*) OVER (PARTITION BY id, country, email_clean) > 1 THEN 1 ELSE 0 END AS is_duplicate_within_company
      FROM format_ok
    ),
    company_domains AS (
      SELECT
        d.id,
        lower(
          trim(
            regexp_replace(
              regexp_replace(regexp_replace(coalesce(d.fqdn, ''), '^https?://', '', 'i'), '/.*$', '', 'g'),
              '^www\\.', '', 'i'
            )
          )
        ) AS company_domain
      FROM {BRZ_DOMAINS} d
      WHERE coalesce(trim(d.fqdn), '') <> '' AND d.fqdn LIKE '%.%'
    ),
    joined AS (
      SELECT
        f.*,
        cd.company_domain,
        CASE
          WHEN cd.company_domain IS NOT NULL AND (
            f.email_domain = cd.company_domain OR f.email_domain LIKE '%.' || cd.company_domain OR cd.company_domain LIKE '%.' || f.email_domain
          ) THEN 1 ELSE 0 END AS domain_match
      FROM dup_flag f
      LEFT JOIN company_domains cd ON cd.id = f.id
    ),
    scored AS (
      SELECT *,
        (
          -- priority scoring (higher is better)
          (is_valid_format * 50)
          + ((1 - is_duplicate_within_company) * 15)
          + ((1 - is_blocked) * 15)
          + (domain_match * 10)
          + (CASE
              WHEN local_part IN ('sales','info','contact','enquiries','support','hello','admin') THEN 10
              WHEN local_part LIKE 'sales%' OR local_part LIKE 'info%' OR local_part LIKE 'contact%' THEN 7
              ELSE 0
            END)
          + (CASE
              WHEN verification_status IN ('valid','deliverable','verified') THEN 5
              ELSE 0
            END)
        ) AS priority_score
      FROM joined
    ),
    ranked AS (
      SELECT *,
        row_number() OVER (PARTITION BY id, country ORDER BY priority_score DESC, email_clean ASC) AS email_rank,
        count(*) OVER (PARTITION BY id, country) AS total_emails_for_company_country
      FROM scored
    )
    SELECT
      id,
      country,
      total_emails_for_company_country,
      email_rank,
      priority_score,
      email,
      is_valid_format,
      is_duplicate_within_company,
      is_blocked,
      domain_match,
      verification_status
    FROM ranked
    WHERE total_emails_for_company_country >= 100
      AND email_rank <= 5
    ORDER BY total_emails_for_company_country DESC, id, country, email_rank
    LIMIT 50;
    """

    section_query(PRIORITY_TOP5_SQL, "SQL Query (Priority Top-5 selection)")
    st.info(
        "This section ranks emails per (company, country) and selects the **best 5**.\n"
        "It’s a practical way to enforce your **Top-5 policy** for the app."
    )
    try:
        pr = run_sql_df(PRIORITY_TOP5_SQL)
        st.dataframe(pr, use_container_width=True)
    except Exception as e:
        st.warning(f"Priority query failed. Likely missing columns (fqdn/verification_status) or table mismatch.\n\nError: {e}")

    section_comment_ui("priority")

with dashboard_tab:
    st.header("Dashboard & Trends")

    if not st.session_state.metrics_for_dashboard:
        st.info("Run checks in the **✅ Checks** tab to populate this dashboard.")
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
                upsert_metric(
                    st.session_state.run_id,
                    "composite_score",
                    "Composite DQ score (higher is better)",
                    None,
                    composite,
                    "info",
                    {"components": COMPOSITE_COMPONENTS}
                )

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
                st.plotly_chart(
                    px.bar(trend, x="release_label", y="composite_score", hover_data=["run_ts"],
                           title="Composite DQ score by release"),
                    use_container_width=True
                )

    st.divider()
    st.caption("Health check")
    st.dataframe(run_sql_df("select now() as server_time;"), use_container_width=True)
