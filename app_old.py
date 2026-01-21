# app.py
import os
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

# -----------------------------
# Load .env from SAME folder as app.py
# -----------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(page_title="Data Quality for Company Emails", layout="wide")
st.title("📧 Data Quality for Company Emails")

st.header("Understanding the source data")

# -----------------------------
# Connection helpers
# -----------------------------
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

@st.cache_data(ttl=300)
def run_sql_df(sql: str) -> pd.DataFrame:
    with get_pg_conn() as conn:
        return pd.read_sql_query(sql, conn)

def show_table(title: str, df: pd.DataFrame):
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

def insight_box(lines: list[str], kind: str = "info"):
    msg = "\n".join([f"- {l}" for l in lines])
    if kind == "success":
        st.success(msg)
    elif kind == "warning":
        st.warning(msg)
    elif kind == "error":
        st.error(msg)
    else:
        st.info(msg)

def fmt_pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "—"

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

# -----------------------------
# Tables
# -----------------------------
SRC_PERSONA = "src_companies.persona"
SRC_EMAILS  = "src_companies.emails_comp"

BRZ_EMAILS  = "zeus_bronze.brz_comp_emails"
BRZ_DOMAINS = "zeus_bronze.brz_comp_domains"

THRESHOLD_PCT_WITHOUT = 70.0

# ============================================================
# 1) Email coverage by vendor (SOURCE) + comparison + BRONZE
# ============================================================
with st.expander("1) Email coverage (vendor) + Source vs Bronze comparison", expanded=True):
    st.markdown("**Goal:** Check email coverage in source data by vendor and compare to bronze coverage.")

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

    colA, colB = st.columns([1.2, 0.8])

    with colA:
        df_vendor = run_sql_df(SOURCE_VENDOR_COVERAGE_SQL)
        show_table("Source: Email coverage by vendor", df_vendor)

        # chart: vendor pct_without_email, red if <70 else green (as you requested)
        if not df_vendor.empty:
            chart = df_vendor.copy()
            chart["color_bucket"] = chart["pct_without_email"].apply(
                lambda x: "green" if float(x) >= THRESHOLD_PCT_WITHOUT else "red"
            )
            fig = px.bar(
                chart,
                x="source",
                y="pct_without_email",
                color="color_bucket",
                title=f"Vendors to reach out (pct_without_email ≥ {THRESHOLD_PCT_WITHOUT})",
                labels={"source": "Vendor", "pct_without_email": "% companies without email"},
                category_orders={"source": chart.sort_values("pct_without_email", ascending=False)["source"].tolist()},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            needs_reachout = chart[chart["pct_without_email"] >= THRESHOLD_PCT_WITHOUT]["source"].tolist()
            if needs_reachout:
                insight_box(
                    [
                        f"Reach out vendors where pct_without_email ≥ {THRESHOLD_PCT_WITHOUT}: {', '.join(needs_reachout)}.",
                        "These vendors are supplying records with low email coverage (or mapping is not landing emails correctly).",
                    ],
                    kind="warning",
                )
            else:
                insight_box(
                    [f"No vendors cross the {THRESHOLD_PCT_WITHOUT}% without-email threshold."],
                    kind="success",
                )

    with colB:
        df_src_total = run_sql_df(SOURCE_TOTAL_COVERAGE_SQL)
        df_brz_total = run_sql_df(BRONZE_TOTAL_COVERAGE_SQL)

        show_table("Source (overall)", df_src_total)
        show_table("Bronze (overall)", df_brz_total)

        if (not df_src_total.empty) and (not df_brz_total.empty):
            s = df_src_total.iloc[0].to_dict()
            b = df_brz_total.iloc[0].to_dict()

            delta_with = safe_int(b.get("companies_with_email")) - safe_int(s.get("companies_with_email"))
            delta_total = safe_int(b.get("total_companies")) - safe_int(s.get("total_companies"))
            delta_pct_with = (float(b.get("pct_with_email")) if b.get("pct_with_email") is not None else 0.0) - (
                float(s.get("pct_with_email")) if s.get("pct_with_email") is not None else 0.0
            )

            kind = "success" if abs(delta_pct_with) < 0.5 else "warning"

            insight_box(
                [
                    f"Source companies_with_email: {safe_int(s.get('companies_with_email')):,} ({fmt_pct(s.get('pct_with_email'))})",
                    f"Bronze companies_with_email: {safe_int(b.get('companies_with_email')):,} ({fmt_pct(b.get('pct_with_email'))})",
                    f"Delta companies_with_email: {delta_with:+,}",
                    f"Delta total_companies: {delta_total:+,}",
                    f"Delta pct_with_email: {delta_pct_with:+.2f} pp",
                ],
                kind=kind,
            )

            # Why mismatches happen (insights only, no extra query)
            insight_box(
                [
                    "If source vs bronze doesn’t match, the most likely reasons are:",
                    "Company universe changed (source uses persona PUBLISHED; bronze uses domains id base).",
                    "Join-key loss: emails in bronze may not map to the same company ids as source slugs.",
                    "Deduping/normalisation or filtering during transformation (dropping empty/invalid emails).",
                    "Orphan/partial records: domain exists but no email row (or email rows exist without domain).",
                    "Status/enablement rules differ between layers.",
                ],
                kind="info",
            )

# ============================================================
# 2) Domain status code (BRONZE)
# ============================================================
with st.expander("2) Status Code (Domains reachability)", expanded=True):
    STATUS_CODE_SQL = f"""
    SELECT
      COUNT(*) AS total_domains,
      COUNT(*) FILTER (WHERE status_code = 200) AS status_200,
      ROUND(
        COUNT(*) FILTER (WHERE status_code = 200) * 100.0 / NULLIF(COUNT(*), 0),
        2
      ) AS pct_200,
      COUNT(*) FILTER (WHERE status_code <> 200 OR status_code IS NULL) AS non_200,
      ROUND(
        COUNT(*) FILTER (WHERE status_code <> 200 OR status_code IS NULL) * 100.0 / NULLIF(COUNT(*), 0),
        2
      ) AS pct_non_200
    FROM {BRZ_DOMAINS};
    """
    df = run_sql_df(STATUS_CODE_SQL)
    show_table("Domain HTTP status summary", df)

    if not df.empty:
        r = df.iloc[0].to_dict()
        insight_box(
            [
                f"HTTP 200 domains: {safe_int(r.get('status_200')):,} ({fmt_pct(r.get('pct_200'))})",
                f"Non-200/NULL: {safe_int(r.get('non_200')):,} ({fmt_pct(r.get('pct_non_200'))})",
                "Low 200% can reduce email-domain matching confidence and downstream enrichment quality.",
            ],
            kind="info",
        )

# ============================================================
# 3) Proper email format coverage (BRONZE) + insight for companies
# ============================================================
with st.expander("3) Proper email format coverage", expanded=True):
    FORMAT_SQL = f"""
    with base as (
      select
        id,
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
        id,
        email_clean,
        case
          when email_clean is null or trim(email_clean) = '' then 'empty'
          when email_clean not like '%@%' then 'invalid'
          when email_clean !~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{2,}$' then 'invalid'
          else 'valid'
        end as format_status
      from base
    )
    select
      format_status,
      count(*) as email_rows,
      round(100.0 * count(*) / nullif(sum(count(*)) over (),0), 2) as pct
    from classified
    group by format_status
    order by email_rows desc;
    """

    # company-level “among companies that have at least one non-empty email, % have at least one valid”
    FORMAT_COMPANY_INSIGHT_SQL = f"""
    with base as (
      select
        id,
        regexp_replace(
          lower(trim(coalesce(email, ''))),
          '[^a-z0-9@._%+\\-]',
          '',
          'g'
        ) as email_clean
      from {BRZ_EMAILS}
      where email is not null and trim(email) <> ''
    ),
    flags as (
      select
        id,
        max(case when email_clean like '%@%'
              and email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
            then 1 else 0 end) as has_valid_email
      from base
      group by id
    )
    select
      count(*) as companies_with_any_email,
      sum(has_valid_email) as companies_with_valid_email,
      round(100.0 * sum(has_valid_email) / nullif(count(*),0), 2) as pct_companies_with_valid_email
    from flags;
    """

    df = run_sql_df(FORMAT_SQL)
    show_table("Email format status distribution (rows)", df)

    dfc = run_sql_df(FORMAT_COMPANY_INSIGHT_SQL)
    show_table("Company-level: % of companies with any email that have ≥1 valid-format email", dfc)

    if not dfc.empty:
        r = dfc.iloc[0].to_dict()
        insight_box(
            [
                f"Among companies with at least one email, {fmt_pct(r.get('pct_companies_with_valid_email'))} have ≥1 valid-format email.",
                "Invalid formats often come from special characters added to emails, masked emails, or non-email tokens stored in the email field.",
            ],
            kind="info",
        )

# ============================================================
# 4) Duplicate emails (BRONZE)
# ============================================================
with st.expander("4) Duplicate emails", expanded=True):
    DUP_TOP20_SQL = f"""
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

    DUP_UNIQUE_COUNT_SQL = f"""
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
    ),
    dups as (
      SELECT email_clean
      FROM valid
      GROUP BY email_clean
      HAVING COUNT(*) > 1
    )
    SELECT COUNT(*) as unique_duplicate_emails
    FROM dups;
    """

    DUP_KPI_SQL = f"""
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

    kpi = run_sql_df(DUP_KPI_SQL)
    show_table("Duplicate KPI summary (company + country + cleaned valid email)", kpi)

    uniq = run_sql_df(DUP_UNIQUE_COUNT_SQL)
    show_table("Unique duplicate email count", uniq)

    top20 = run_sql_df(DUP_TOP20_SQL)
    show_table("Top 20 duplicate emails to review", top20)

    if not uniq.empty:
        r = uniq.iloc[0].to_dict()
        insight_box(
            [
                f"There are {safe_int(r.get('unique_duplicate_emails')):,} unique email addresses that appear more than once.",
                "Duplicates can inflate coverage and skew priority selection unless you dedupe at (company, country, email_clean).",
            ],
            kind="warning",
        )

# ============================================================
# 5) Domain match rate (company website domain vs email domain)
# ============================================================
with st.expander("5) Domain match rate between company domain and email domain", expanded=True):
    DOMAIN_MATCH_SQL = f"""
    with base as (
      select
        d.id,
        lower(regexp_replace(trim(coalesce(d.website, '')), '^www\\.', '')) as company_domain,
        regexp_replace(
          lower(trim(coalesce(e.email, ''))),
          '[^a-z0-9@._%+\\-]',
          '',
          'g'
        ) as email_clean
      from {BRZ_DOMAINS} d
      left join {BRZ_EMAILS} e
        on d.id = e.id
    ),
    valid as (
      select
        *,
        split_part(email_clean, '@', 2) as email_domain
      from base
      where company_domain is not null
        and trim(company_domain) <> ''
        and email_clean <> ''
        and email_clean like '%@%'
        and email_clean ~ '^[a-z0-9._%+\\-]+@[a-z0-9.\\-]+\\.[a-z]{{2,}}$'
    ),
    scored as (
      select
        *,
        case
          when email_domain = company_domain
            or email_domain like '%.' || company_domain
            or company_domain like '%.' || email_domain
          then 1 else 0
        end as is_domain_match
      from valid
    )
    select
      count(*) as total_valid_emails,
      sum(is_domain_match) as domain_match_emails,
      round(100.0 * sum(is_domain_match) / nullif(count(*),0), 2) as pct_domain_match
    from scored;
    """

    df = run_sql_df(DOMAIN_MATCH_SQL)
    show_table("Domain match KPI", df)

    if not df.empty:
        r = df.iloc[0].to_dict()
        insight_box(
            [
                f"Domain match rate: {fmt_pct(r.get('pct_domain_match'))} of valid emails match the company website domain (or subdomain).",
                "Ways to increase domain match:",
                "For smaller companies, consider allowing free email domains as acceptable (gmail/yahoo/outlook) when no website exists.",
                "For larger groups/holdings, staff may use parent/subsidiary domains—maintain hierarchy-aware domain matching.",
                "Improve person↔company mapping (and company dedupe) to reduce mismatched assignments.",
                "Handle orphan companies (no reliable website) and duplicates where name↔domain doesn’t align.",
                "Some industries/brands legitimately use non-matching domains (brand vs legal entity).",
            ],
            kind="info",
        )

# ============================================================
# 6) Blocked “no-reply” type emails
# ============================================================
with st.expander("6) Blocked “no-reply” type emails", expanded=True):
    BLOCKED_SQL = f"""
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
    df = run_sql_df(BLOCKED_SQL)
    show_table("Blocked (system) email KPI", df)

    if not df.empty:
        r = df.iloc[0].to_dict()
        insight_box(
            [
                f"Blocked/system emails: {safe_int(r.get('blocked_emails')):,} ({fmt_pct(r.get('pct_blocked'))} of valid emails).",
                "These should be removed immediately from outbound/contact lists because they are non-contactable and harm deliverability.",
            ],
            kind="warning",
        )

# ============================================================
# 7) Deliverability status
# ============================================================
with st.expander("7) Deliverability status", expanded=True):
    DELIVERABILITY_SQL = f"""
    WITH base AS (
      SELECT NULLIF(trim(verification_status), '') AS verification_status
      FROM {BRZ_EMAILS}
    ),
    totals AS (
      SELECT COUNT(*) AS total_rows FROM base
    )
    SELECT
      COALESCE(verification_status, 'NULL') AS verification_status,
      COUNT(*) AS row_count,
      ROUND(COUNT(*)::numeric / NULLIF(totals.total_rows,0) * 100, 2) AS pct
    FROM base
    CROSS JOIN totals
    GROUP BY COALESCE(verification_status, 'NULL'), totals.total_rows
    ORDER BY row_count DESC;
    """
    df = run_sql_df(DELIVERABILITY_SQL)
    show_table("verification_status distribution", df)

    insight_box(
        [
            "Interpretation guidance:",
            "High NULL share usually means deliverability was not checked for many rows (or status didn’t land in bronze).",
            "A healthy pipeline typically has lower NULL over time and more values in verified/valid/invalid buckets (depending on your status taxonomy).",
            "Use this table to monitor vendor/process improvements and to decide where to run verification jobs.",
        ],
        kind="info",
    )

# ============================================================
# 8) Companies with > 5 emails (by country)
# ============================================================
with st.expander("8) Companies with > 5 emails (by country)", expanded=True):
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
    df = run_sql_df(GT5_SQL)
    show_table("Country rollup: company-country pairs with >5 emails", df)

    if not df.empty:
        top = df.sort_values(["companies_gt5_emails", "pct_companies_gt5_emails"], ascending=False).head(1)
        t = top.iloc[0].to_dict()
        insight_box(
            [
                f"Highest concentration of >5 emails is in **{t.get('country')}**: "
                f"{safe_int(t.get('companies_gt5_emails')):,} company-country pairs "
                f"({fmt_pct(t.get('pct_companies_gt5_emails'))}).",
                "This segment needs stricter prioritisation rules (role-based + domain-match) to avoid noise.",
            ],
            kind="warning",
        )

# ============================================================
# 9) Priority coverage (Top 5 emails per company per country) — insights only
# ============================================================
with st.expander("9) Priority coverage (Top 5 emails per company per country) — Insights", expanded=True):
    insight_box(
        [
            "Priority selection rules (recommended):",
            "Keep only valid email format.",
            "Remove duplicates (company + country + cleaned email).",
            "Prefer domains with status_code = 200 when using domain-based confidence.",
            "Require company domain ↔ email domain match (or known acceptable exceptions).",
            "Exclude blocked/system emails (no-reply/do-not-reply).",
            "Prefer role-based inboxes: sales@, info@, contact@, support@, enquiries@ etc.",
        ],
        kind="info",
    )

# -----------------------------
# Footer: Health check
# -----------------------------
st.divider()
st.caption("Health check")
try:
    st.dataframe(run_sql_df("select now() as server_time;"), use_container_width=True)
except Exception as e:
    st.exception(e)
