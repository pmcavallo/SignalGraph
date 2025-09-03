# app/streamlit_app.py
import pathlib
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds

st.set_page_config(page_title="SignalGraph 5G", layout="wide")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver" / "cell_kpi_minute"

st.title("SignalGraph 5G — Analyst View")
st.caption("Filter a small synthetic slice and preview anomalies before adding ML and graph analytics.")

# Guard: require Silver to exist
if not SILVER.exists():
    st.warning("Silver data not found. Run the Spark Bronze→Silver step first.")
    st.stop()

# Read Parquet as a HIVE-partitioned dataset so `date` and `region` materialize as columns
dataset = ds.dataset(str(SILVER), format="parquet", partitioning="hive")

# Load a compact set of columns to keep memory low
wanted = ["ts", "cell_id", "anomaly_flag", "latency_ms", "prb_util_pct", "thrpt_mbps", "date", "region"]
present = [c for c in wanted if c in dataset.schema.names]
table = dataset.to_table(columns=present)
df = table.to_pandas()

# Normalize dtypes for safe filtering/ordering
if "ts" in df.columns:
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_localize(None)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
if "region" not in df.columns:
    df["region"] = "unknown"

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    regions = sorted(df["region"].astype(str).dropna().unique().tolist())
    region = st.selectbox("Region", regions if regions else ["unknown"])
    df = df[df["region"] == region]
    if "date" in df.columns:
        dates = sorted(df["date"].dropna().unique().tolist())
        if dates:
            sel_date = st.selectbox("Date", dates, index=len(dates) - 1)
            df = df[df["date"] == sel_date]

st.metric("Rows loaded", f"{len(df):,}")

# helper: show risk as percent for readability
def _style_risk(df):
    if "risk_score" not in df.columns:
        return df
    s = df.copy()
    s["risk_score"] = (s["risk_score"] * 100).round(1)  # e.g., 0.237 -> 23.7
    return s

# Sample table
cols_to_show = [c for c in ["ts", "cell_id", "anomaly_flag", "latency_ms", "prb_util_pct", "thrpt_mbps", "region", "date"] if c in df.columns]
if cols_to_show:
    st.subheader("Sample (latest 100)")
    df_sorted = df.sort_values("ts") if "ts" in df.columns else df
    st.dataframe(df_sorted[cols_to_show].tail(100), use_container_width=True)

# Ensure region column exists to avoid KeyError later
if "region" not in df.columns:
    df["region"] = "unknown"

# Aggregates
agg_cols = [c for c in ["anomaly_flag", "latency_ms", "prb_util_pct"] if c in df.columns]
if agg_cols and "cell_id" in df.columns:
    st.subheader("Top cells by anomaly rate")
    sort_key = "anomaly_flag" if "anomaly_flag" in agg_cols else agg_cols[0]
    top = (
        df.groupby("cell_id")[agg_cols]
          .mean(numeric_only=True)
          .sort_values(sort_key, ascending=False, na_position="last")
          .reset_index()
          .head(10)
    )
    st.dataframe(top, use_container_width=True)
else:
    st.info("No aggregations available yet. Build Silver first and ensure expected columns exist.")

st.markdown("Next steps: add XGBoost risk score, Prophet latency forecast, and a neighbor graph drill-in.")

# --- Model-based ranking (latest hour) ---
from pathlib import Path
ART = ROOT / "artifacts"
scores_path = ART / "scores_latest.csv"

if scores_path.exists():
    st.subheader("Top cells by model risk (latest hour)")

    # Read + normalize column names and types
    scores = pd.read_csv(scores_path)
    scores.columns = [c.strip().lower() for c in scores.columns]
    if "ts_hour" in scores.columns:
        scores["ts_hour"] = pd.to_datetime(scores["ts_hour"], errors="coerce")

    # --- Alerts from model risk (uses saved operating threshold if available) ---
import json

ART = ROOT / "artifacts"
scores_path = ART / "scores_latest.csv"
metrics_path = ART / "metrics.json"

# after reading scores
cent_path = ROOT / "artifacts" / "centrality.parquet"
if cent_path.exists():
    cent = pd.read_parquet(cent_path)
    scores = scores.merge(cent, on="cell_id", how="left")

display_cols = [c for c in [
    "cell_id","ts_hour","risk_score",
    "latency_ms_p95","prb_util_pct_avg","sinr_db_avg","thrpt_mbps_avg",
    "neighbor_degree","pagerank","betweenness"
] if c in scores.columns]

st.dataframe(
    scores.sort_values("risk_score", ascending=False)[display_cols].head(12),
    use_container_width=True, height=320
)
st.caption("Includes graph-derived features (degree, PageRank, betweenness) to reflect neighbor topology.")

if scores_path.exists():
    st.subheader("Alerts (model ≥ operating threshold)")
    scores = pd.read_csv(scores_path)
    scores.columns = [c.strip().lower() for c in scores.columns]
    if "ts_hour" in scores.columns:
        scores["ts_hour"] = pd.to_datetime(scores["ts_hour"], errors="coerce")

    # Region filter (if both selection and column exist)
    region_sel = locals().get("region", None)
    if region_sel and "region" in scores.columns:
        scores = scores[scores["region"] == region_sel]

    # Load threshold from metrics.json (fallback to 0.5 if missing)
    thr = 0.5
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
            if "report" in metrics and "1" in metrics["report"]:
                thr = metrics["report"]["1"].get("f1-threshold", thr)
    except Exception as e:
        st.warning(f"Could not load threshold: {e}")

    # Show alerts where risk_score >= thr
    alerts = scores[scores["risk_score"] >= thr].copy()
    if not alerts.empty:
        st.dataframe(alerts, use_container_width=True)
    else:
        st.info("No alerts above threshold for this region/hour.")

    # helper stub until implemented
    def reason_from_row(row):
        # TODO: replace with real feature attribution or SHAP
        return "high latency" if row.get("latency_ms_p95", 0) > 60 else "capacity pressure"

    # inside alerts block
    if not alerts.empty:
        alerts["reason"] = alerts.apply(reason_from_row, axis=1)
        cols = [c for c in ["cell_id","region","ts_hour","risk_score",
                            "latency_ms_p95","prb_util_pct_avg","sinr_db_avg",
                            "thrpt_mbps_avg","reason"] if c in alerts.columns]
        st.dataframe(alerts.sort_values("risk_score", ascending=False)[cols],
                    use_container_width=True)

    # Respect the UI region filter *only if* both the selection and column exist
    region_sel = locals().get("region", None)
    if region_sel and "region" in scores.columns:
        scores = scores[scores["region"] == region_sel]

    # If risk_score missing (shouldn’t be), degrade gracefully
    if "risk_score" not in scores.columns:
        st.info("No 'risk_score' column found in scores. Re-run:  python ml/train_baseline.py")
    else:
        topm = scores.sort_values("risk_score", ascending=False).head(10)
        st.dataframe(_style_risk(topm), use_container_width=True)

else:
    st.info("Train the model and generate scores (run:  python ml/train_baseline.py) to see risk rankings here.")

# --- Model metrics panel ---
metrics_path = ROOT / "artifacts" / "metrics.json"
if metrics_path.exists():
    import json
    with open(metrics_path) as f:
        m = json.load(f)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{m.get('auc_roc', 0):.3f}")
    col2.metric("AUC-PR", f"{m.get('auc_pr', 0):.3f}")
    thr = m.get("operating_point", {}).get("threshold", 0.5)
    col3.metric("Op Threshold", f"{thr:.2f}")
    rep = m.get("report_at_threshold", {}).get("1", {})  # class "1" = positive
    col4.metric("F1@thr", f"{m.get('operating_point', {}).get('f1', 0):.3f}")
    st.caption("Metrics and operating point come from artifacts/metrics.json")

# --- PRB forecast panel ---
fc_path = ROOT / "artifacts" / "forecast_prb_latest.csv"
if fc_path.exists():
    st.subheader("PRB Utilization Forecast")
    import pandas as pd
    fc = pd.read_csv(fc_path, parse_dates=["ds"])
    # let user pick a cell seen in the current region
    cell_opts = sorted(fc["cell_id"].unique().tolist())
    sel_cell = st.selectbox("Cell for forecast", cell_opts)
    plot_df = fc[fc["cell_id"]==sel_cell].copy()
    # simple chart: observed (y) + forecast intervals
    import altair as alt
    base = alt.Chart(plot_df).encode(x="ds:T")
    obs  = base.mark_line().encode(y="y:Q").transform_filter("datum.model == 'observed'")
    pred = base.mark_line().encode(y="yhat:Q").transform_filter("datum.model != 'observed'")
    band = base.mark_area(opacity=0.2).encode(y="yhat_lower:Q", y2="yhat_upper:Q").transform_filter("datum.model != 'observed'")
    st.altair_chart(band + pred + obs, use_container_width=True)
else:
    st.info("Run the PRB forecaster to visualize capacity risk:  python ml/forecast_prb.py --region dfw --h 12 --hist 24 --min-points 24")

import os
import pandas as pd
import streamlit as st

pg_dsn = os.getenv("SG_PG_DSN")  # e.g., postgresql://user:pass@localhost:5432/signalgraph
st.divider()
st.subheader("Warehouse view (Postgres)")

if pg_dsn:
    try:
        import psycopg
        with psycopg.connect(pg_dsn) as conn:
            df_sql = pd.read_sql_query("select * from sg5g.v_last_hour_risk", conn)
            # after:
            df_sql = pd.read_sql_query("select * from sg5g.v_last_hour_risk", conn)

            # ADD THIS:
            hide_empty = st.toggle("Hide empty columns", value=True)
            if hide_empty:
                # keep only columns that have at least one non-null value
                df_sql = df_sql.loc[:, df_sql.notna().any()]

        st.caption("Postgres: sg5g.v_last_hour_risk")
        st.dataframe(df_sql, use_container_width=True, height=320)
    except Exception as e:
        st.warning(f"Postgres connection failed: {e}")
else:
    st.info("Set SG_PG_DSN to preview the Postgres mart (e.g., postgresql://user:pass@host:5432/db)")

