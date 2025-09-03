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
    st.markdown("### Methods & Stack")
    st.markdown(
        "- **Anomaly scoring**: XGBoost on hourly cell KPIs\n"
        "- **Explainability**: SHAP local reasons + global summary\n"
        "- **Forecasting**: Prophet (PRB), 12h breach probability\n"
        "- **Graph features**: neighbor degree, PageRank, betweenness\n"
        "- **Warehouse**: Postgres/DUCKDB mirror + view\n"
        "- **Triage**: priority = risk × forecast × graph"
    )
    st.markdown(
        "- 4G/5G KPIs (PRB, RSRP/SINR, latency, drops)\n"
        "- Scoring + anomaly detection at scale\n"
        "- Data lake/warehouse + NoSQL/graph DB exposure\n"
        "- AI/ML stack (NumPy, scikit-learn, XGBoost, Prophet)\n"
        "- Ops focus: capacity, reliability, scalability"
    )
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
import json

ART = ROOT / "artifacts"
scores_path = ART / "scores_latest.csv"
metrics_path = ART / "metrics.json"

# Helpers (defined before use)
from pathlib import Path as _Path
def _merge_shap_reasons(df):
    try:
        shp = _Path(ROOT / "artifacts" / "shap_top_reasons.parquet")
        if shp.exists():
            import pandas as _pd
            sr = _pd.read_parquet(shp)
            return df.merge(sr, on=["cell_id", "ts_hour"], how="left")
    except Exception as _e:
        st.warning(f"Could not load SHAP reasons: {_e}")
    return df

def reason_from_row(row):
    # fallback reason when SHAP not available
    return "high latency" if row.get("latency_ms_p95", 0) > 60 else "capacity pressure"

if scores_path.exists():
    st.subheader("Top cells by model risk (latest hour)")

    # Read + normalize column names and types
    scores = pd.read_csv(scores_path)
    scores.columns = [c.strip().lower() for c in scores.columns]
    if "ts_hour" in scores.columns:
        scores["ts_hour"] = pd.to_datetime(scores["ts_hour"], errors="coerce")

    # Merge optional graph centrality
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

    # --- Alerts using saved operating threshold ---
    st.subheader("Alerts (model ≥ operating threshold)")

    # Respect the UI region filter if present
    scores_region = scores
    region_sel = locals().get("region", None)
    if region_sel and "region" in scores_region.columns:
        scores_region = scores_region[scores_region["region"] == region_sel]

    # Threshold from metrics.json (operating_point.threshold)
    thr = 0.5
    try:
        with open(metrics_path) as f:
            m = json.load(f)
        thr = float(m.get("operating_point", {}).get("threshold", thr))
    except Exception as e:
        st.warning(f"Could not load threshold: {e}")
    # --- Optional UI override to see more/fewer alerts ---
    thr_default = float(thr)
    thr = st.slider(
        "Alert threshold (override)",
        min_value=0.00, max_value=1.00, value=float(thr), step=0.01,
        help="Default comes from artifacts/metrics.json (operating_point.threshold). "
            "Lower to increase recall (more alerts), higher to reduce false positives."
    )
    if thr != thr_default:
        st.caption(f"Using custom threshold {thr:.2f} (model op thr = {thr_default:.2f}).")
    # Filter and show alerts
    alerts = scores_region[scores_region["risk_score"] >= thr].copy()

    # --- Alert priority score (risk × forecast × graph) ---
    thr_prb = 85.0  # PRB capacity threshold (%)

    # p(breach) over next 12h from Prophet
    f = pd.read_csv(ROOT/"artifacts"/"forecast_prb_latest.csv", parse_dates=["ds"])
    f = f[f["ds"] >= f["ds"].max() - pd.Timedelta(hours=12)]
    pbreach = (f.assign(above=(f["yhat_upper"] >= thr_prb))
                .groupby("cell_id")["above"].mean()
                .rename("p_breach"))
    alerts = alerts.merge(pbreach, on="cell_id", how="left").fillna({"p_breach": 0})

    # normalize centrality (if present)
    for c in ["neighbor_degree","pagerank","betweenness"]:
        if c in alerts.columns:
            m, M = alerts[c].min(), alerts[c].max()
            alerts[c+"_z"] = 0 if M == m else (alerts[c]-m)/(M-m)
        else:
            alerts[c+"_z"] = 0
    
    st.download_button(
        "Download actionable alerts (CSV)",
        alerts.sort_values("priority", ascending=False)[cols].to_csv(index=False).encode(),
        "actionable_alerts.csv",
        "text/csv"
    )

    # weights (tune during demo if needed)
    w_risk, w_fore, w_graph = 0.55, 0.30, 0.15
    alerts["priority"] = (
        w_risk*alerts["risk_score"]
        + w_fore*alerts["p_breach"]
        + w_graph*(0.5*alerts["pagerank_z"] + 0.3*alerts["betweenness_z"] + 0.2*alerts["neighbor_degree_z"])
    )

    # include in table & sort by priority
    cols = [c for c in ["cell_id","region","ts_hour","priority","risk_score",
                        "latency_ms_p95","prb_util_pct_avg","sinr_db_avg","thrpt_mbps_avg",
                        "reason"] if c in alerts.columns]

    # --- Attach SHAP reasons (if artifact exists) ---
    from pathlib import Path as _Path
    try:
        _shap_file = _Path(ROOT / "artifacts" / "shap_top_reasons.parquet")
        if _shap_file.exists():
            _sr = pd.read_parquet(_shap_file)
            # normalize types for a reliable join
            if "ts_hour" in _sr.columns:
                _sr["ts_hour"] = pd.to_datetime(_sr["ts_hour"], errors="coerce")
            if "ts_hour" in alerts.columns:
                alerts["ts_hour"] = pd.to_datetime(alerts["ts_hour"], errors="coerce")
            # only keep the columns we need
            keep = [c for c in ["cell_id", "ts_hour", "top_reasons"] if c in _sr.columns]
            alerts = alerts.merge(_sr[keep], on=["cell_id", "ts_hour"], how="left")
    except Exception as _e:
        st.warning(f"Could not load SHAP reasons: {_e}")

    # --- Prefer SHAP 'top_reasons'; otherwise fall back to simple rule ---
    alerts["reason"] = alerts.get("top_reasons", pd.Series(index=alerts.index, dtype=object)).fillna("")
    _empty = alerts["reason"].eq("")
    if _empty.any():
        alerts.loc[_empty, "reason"] = alerts.loc[_empty].apply(reason_from_row, axis=1)

    if not alerts.empty:
    # choose existing columns only (keeps file integrity)
        cols = [c for c in [
            "cell_id", "region", "ts_hour", "risk_score",
            "latency_ms_p95", "prb_util_pct_avg", "sinr_db_avg",
            "thrpt_mbps_avg", "reason"
        ] if c in alerts.columns]

        max_rows = st.number_input(
            "Max rows to display",
            min_value=5, max_value=200, value=25, step=5,
            help="Limit how many alerts are shown below."
        )

        st.dataframe(
            alerts.sort_values("risk_score", ascending=False)[cols].head(int(max_rows)),
            use_container_width=True
        )

        # Optional, tiny hint to confirm SHAP is being used
        if "top_reasons" in alerts.columns and alerts["top_reasons"].notna().any():
            st.caption("Reasons shown use SHAP top features.")

        # --- Global drivers (SHAP summary) ---
        from pathlib import Path as _Path

        shap_img = _Path(ROOT / "artifacts" / "shap_summary.png")

        with st.expander("Global feature importance — SHAP summary", expanded=False):
            if shap_img.exists():
                st.image(str(shap_img), width=700)
                st.caption(
                    "Features ranked by mean(|SHAP|) on the validation window. "
                    "Higher bars = larger average impact on the model’s risk score."
                )
                # Extra explanation tying local (table) to global (summary)
                st.markdown(
                    "_How to read this_: The **Alerts** table above shows the top SHAP features "
                    "for the currently flagged cell/hour (local explanation). "
                    "This chart shows **global** feature importance across the validation window. "
                    "If an alert’s reasons appear near the top here, local and global drivers are aligned; "
                    "if not, that alert is driven by context-specific conditions for that cell/time."
                )
                # (Optional) echo the reasons for the first visible alert so reviewers connect them at a glance
                if 'reason' in locals().get('alerts', pd.DataFrame()).columns and not alerts.empty:
                    st.caption(f"Current alert reasons: **{alerts.iloc[0]['reason']}**")

            else:
                st.info(
                    "No SHAP summary image found. To generate it: "
                    "pip install shap matplotlib --quiet  &&  python ml\\train_baseline.py"
                )
    else:
        st.info("No alerts above threshold for this region/hour.")

# --- Cell drill-in (neighbors & centrality) ---
st.markdown("### Cell drill-in — neighbors & centrality")

from pathlib import Path as _Path
import numpy as _np

_cent_path = _Path(ROOT / "artifacts" / "centrality.parquet")
_edges_path = _Path(ROOT / "data" / "external" / "cell_neighbors.csv")

# Load centrality once
_cent = None
if _cent_path.exists():
    _cent = pd.read_parquet(_cent_path)
else:
    st.info("No centrality features found (artifacts/centrality.parquet). "
            "Run graph/centrality_from_neighbors.py to generate it.")

# Get candidate cells (prefer those shown in scores for the selected region)
_cell_options = []
if 'scores_region' in locals() and isinstance(scores_region, pd.DataFrame) and not scores_region.empty:
    _cell_options = sorted(scores_region['cell_id'].dropna().unique().tolist())
elif _cent is not None and 'cell_id' in _cent.columns:
    _cell_options = sorted(_cent['cell_id'].dropna().unique().tolist())

if not _cell_options:
    st.stop()

# Preselect the first alert if one exists; else first cell
_default = None
if 'alerts' in locals() and isinstance(alerts, pd.DataFrame) and not alerts.empty:
    _default = alerts.iloc[0]['cell_id']
try:
    _index = _cell_options.index(_default) if _default in _cell_options else 0
except Exception:
    _index = 0

colA, colB = st.columns([1, 3])
with colA:
    cell_sel = st.selectbox("Cell", options=_cell_options, index=_index)

# Show centrality metrics for the selected cell
with colA:
    if _cent is not None:
        row = _cent[_cent['cell_id'] == cell_sel]
        if not row.empty:
            deg = float(row['neighbor_degree'].iloc[0]) if 'neighbor_degree' in row else _np.nan
            pr  = float(row['pagerank'].iloc[0])        if 'pagerank'        in row else _np.nan
            btw = float(row['betweenness'].iloc[0])     if 'betweenness'     in row else _np.nan
            m1, m2, m3 = st.columns(3)
            m1.metric("Neighbor degree", f"{deg:.0f}" if _np.isfinite(deg) else "–")
            m2.metric("PageRank", f"{pr:.3f}" if _np.isfinite(pr) else "–")
            m3.metric("Betweenness", f"{btw:.3f}" if _np.isfinite(btw) else "–")

# List immediate neighbors (from build_neighbors output) ordered by weight
with colB:
    if _edges_path.exists():
        ed = pd.read_csv(_edges_path)
        need = {"src_cell_id","dst_cell_id"}
        if need.issubset(ed.columns):
            # unify column names
            if "weight" not in ed.columns:  # older file may not have 'weight'
                ed["weight"] = 1.0
            sel = ed[(ed["src_cell_id"] == cell_sel) | (ed["dst_cell_id"] == cell_sel)].copy()
            if not sel.empty:
                sel["neighbor"] = _np.where(sel["src_cell_id"] == cell_sel, sel["dst_cell_id"], sel["src_cell_id"])
                neighbors = sel[["neighbor","weight"]].sort_values("weight", ascending=False)
                st.caption("Immediate neighbors (sorted by edge weight)")
                st.dataframe(neighbors.head(25), use_container_width=True, height=280)
            else:
                st.info("This cell has no neighbors in data/external/cell_neighbors.csv.")
        else:
            st.warning(f"Neighbors file missing required columns {need}.")
    else:
        st.info("No neighbors CSV found. Generate with:  python graph\\build_neighbors.py --region dfw --k 2")

st.caption("Centrality features are consumed by the model and shown in the risk table; "
           "use this drill-in to understand a cell’s graph context when triaging alerts.")


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
    col3.metric("Op Threshold", f"{float(m.get('operating_point', {}).get('threshold', thr)):.2f}")
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

st.markdown("### Network SLO summary")

import duckdb
try:
    con = duckdb.connect(str(ROOT/"artifacts"/"signalgraph.duckdb"), read_only=True)
    df = con.execute("""
        select cell_id, ts_hour, prb_util_pct_avg, latency_ms_p95, drop_rate_pct_avg
        from gold_cell_hourly
    """).fetch_df()
    df["ts_hour"] = pd.to_datetime(df["ts_hour"])
    df["date"] = df["ts_hour"].dt.date

    slo = df.groupby("date").apply(lambda g: pd.Series({
        "capacity_slo"  : (g["prb_util_pct_avg"] < 85).mean(),    # PRB < 85%
        "latency_slo"   : (g["latency_ms_p95"] < 60).mean(),      # p95 < 60 ms
        "reliability_slo": (g["drop_rate_pct_avg"] < 0.5).mean()  # drop < 0.5%
    })).reset_index()

    last = slo.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Capacity SLO (PRB<85%)", f"{last['capacity_slo']*100:.1f}%")
    c2.metric("Latency SLO (p95<60ms)", f"{last['latency_slo']*100:.1f}%")
    c3.metric("Reliability SLO (drop<0.5%)", f"{last['reliability_slo']*100:.1f}%")
except Exception as e:
    st.info(f"SLO panel: {e}")


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

# --- Data quality / freshness ---
# try:
#     con = duckdb.connect(str(ROOT/"artifacts"/"signalgraph.duckdb"), read_only=True)
#     dq = con.execute("select max(ts_hour) as max_ts, "
#                      "sum(prb_util_pct_avg is null)::double / count(*) as null_prb, "
#                      "sum(latency_ms_p95 is null)::double / count(*) as null_lat "
#                      "from gold_cell_hourly").fetch_df().iloc[0]
#     max_ts = pd.to_datetime(dq["max_ts"])
#     hrs = (pd.Timestamp.utcnow() - max_ts.tz_localize('UTC')).total_seconds()/3600
#     msg = f"Data freshness: {hrs:.1f}h, null(PRB)={dq['null_prb']:.2%}, null(latency)={dq['null_lat']:.2%}"
#     if hrs <= 2 and dq['null_prb'] < 0.05 and dq['null_lat'] < 0.05:
#         st.success(msg)
#     else:
#         st.warning(msg)
# except Exception:
#     pass


