# SignalGraph 5G — Verizon-Aligned Network AI Demo

*A compact, interview-ready system that ingests 4G/5G RF & IP KPIs, builds lakehouse tables on Hadoop-style partitions, and exposes a small analyst UI. This repo is intentionally written to **show (not just use)** the skills in Verizon’s job description: scoring models & anomaly detection, large-scale data engineering, data warehouse/lakehouse, ML/forecasting stacks (NumPy, scikit-learn, XGBoost, Prophet, Jupyter, Dataproc), 4G/5G & IP networking know-how, and performance/capacity/reliability thinking.*

---

## What’s working now (verified)

- **Bronze → Silver pipeline (Spark):** Synthetic KPIs → Parquet (Bronze) → **Spark** job to Silver, partitioned by `date` and `region`, with a rule-based `anomaly_flag`.
- **Partition-aware UI (Streamlit):** Reads Silver as a **hive-partitioned dataset**; Region/Date filters; shows a sample table and **Top cells by anomaly rate**.
- **Platform guardrails:** Python **3.10** venv; wheels for pandas/pyarrow; **PySpark 3.5.1**; JDK 11/17. Timestamps standardized to **timezone-naive microseconds** to keep Arrow ↔ Spark parquet schemas compatible.
- **Correct boolean logic in Spark:** OR operations are applied to booleans first, then cast to `int` for flags (prevents datatype errors).

---

## Why this matters (tie-back to Verizon requirements)

1) **Scoring models & anomaly detection**
   - Today: deterministic anomaly rule for fast triage and label seeding.
   - Next: supervised **scoring model** predicting “**anomaly next hour?**” for each cell; artifacts + metrics for operational acceptance.

2) **Experience with large datasets**
   - Hive partitions (`date`, `region`) + columnar Parquet → **predicate/column pruning** and vectorized I/O.
   - Spark jobs are contract-driven → drop-in scale on **Dataproc/EMR**.

3) **Data warehouse & lake technology**
   - **Hadoop/Spark** lakehouse discipline (Bronze→Silver→Gold).
   - **Teradata-style marts** mirrored via DuckDB/Postgres (views like “top-10 risky cells per region”).
   - **NoSQL/Graph** integration points for cell metadata and neighbor graphs (Neo4j).

4) **AI/ML toolchain**
   - **NumPy/pandas, scikit-learn, XGBoost**, and **Prophet** planned; **Jupyter notebooks** for EDA/model cards; **Dataproc** runbooks for scale.

5) **Network expertise (4G/5G & IP)**
   - RF: **RSRP/RSRQ/SINR**. Capacity: **PRB utilization**. Throughput. IP QoS: **latency, jitter, packet loss, drop rate**.
   - The system encodes realistic relationships (e.g., high PRB + low SINR → throughput suppression & latency spikes).

6) **Performance, capacity, reliability, scalability**
   - Explicit data contracts; timestamp precision guardrails; partitioning strategy; job metrics planned; easy cloud migration.

---

## KPI primer (what these features mean and why)

- **RSRP (dBm)** — Downlink reference signal power (coverage). Low RSRP → poor coverage → more retransmissions.
- **RSRQ (dB)** — Signal quality considering interference; complements SINR.
- **SINR (dB)** — Signal vs interference+noise; low SINR hurts modulation/coding, lowering throughput and raising latency.
- **PRB utilization (%)** — RAN resource consumption; sustained high PRB indicates congestion.
- **Throughput (Mb/s)** — User plane performance proxy; drops when PRB is saturated or SINR is poor.
- **Latency (ms), Jitter (ms), Packet loss (%), Drop rate (%)** — IP QoS metrics reflecting user experience.

**Current anomaly rule (seed for learning):**  
`anomaly_flag = (latency_ms > 60) OR (prb_util_pct > 85)`

This simple rule (a) makes the demo visually interesting, (b) resembles congestion/service degradation, and (c) provides ground truth to bootstrap supervised scoring.

---

## Project layout

app/
streamlit_app.py # Partition-aware UI reading hive-partitioned parquet
data/
raw/ # Bronze (cell_kpi_minute.parquet)
silver/ # Silver hive partitions (date=/region=)
sim/
simulate_kpis.py # KPI simulator with controlled degradations
spark/
bronze_to_silver.py # Bronze→Silver Spark job (+ anomaly flag)

(next) silver_to_gold.py # Hourly features + next-hour label

ml/

(next) train_baseline.py # XGBoost / scikit-learn training + metrics

sql/

(next) build_duckdb.py # DuckDB mart + Postgres/Teradata mirror DDL

neo4j/

(next) schema.cypher + centrality.cypher

dq/

(next) data quality checks (nulls, ranges, schema drift)

README.md
requirements.txt


---

## Quick start (Windows PowerShell)

> Assumes Python **3.10** and a JDK (11/17) are installed.

```powershell
# 1) Create venv & install deps
py -3.10 -m venv .venv; Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; .\.venv\Scripts\Activate.ps1
@('pandas==2.2.2','pyarrow==16.1.0','pyspark==3.5.1','streamlit==1.37.1','duckdb==1.0.0') | Set-Content -Encoding ASCII requirements.txt
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# 2) Generate KPIs (Bronze)
python data\simulate_kpis.py

# 3) Build Silver with Spark
$env:JAVA_HOME=(Split-Path -Parent (Split-Path -Parent (Get-Command java | Select-Object -ExpandProperty Source))); $env:PATH="$env:JAVA_HOME\bin;$env:PATH"
python spark\bronze_to_silver.py

# 4) Launch UI
streamlit run app\streamlit_app.py


## Lakehouse → Warehouse Mirroring (DuckDB → Postgres/Teradata)

**Why:** Verizon runs enterprise warehouses (Teradata/Postgres). We keep ML-friendly Parquet in the lake (**DuckDB views/tables**) and publish **clean, governed tables** to the warehouse for BI, finance, and ops. This demonstrates warehouse/lake expertise and clean contracts across systems.

### 1) Authoritative Lake Objects (already built)

- **Gold Parquet**: `data/gold/cell_hourly/*`  
- **DuckDB mart**: `data/mart/sg.duckdb` built by `python sql/build_duckdb.py`  
  - `gold_cell_hourly` *(view on Parquet)*  
  - `v_latest_by_region`, `v_top10_risk_by_region`, `v_region_summary`  
  - `scores_latest` *(latest model risk)*

> Contract: same column names/types across DuckDB and the warehouse → safe for downstream tools.

---

### 2) Postgres/Teradata DDL (warehouse schema)

Create a file `sql/postgres_gold_cell_hourly.sql` with:

```sql
-- gold_cell_hourly (Teradata/Postgres-friendly types)
CREATE TABLE IF NOT EXISTS gold_cell_hourly (
  cell_id           VARCHAR(20),
  region            VARCHAR(20),
  ts_hour           TIMESTAMP,
  n_samples         INTEGER,
  latency_ms_avg    FLOAT,
  latency_ms_p95    FLOAT,
  prb_util_pct_avg  FLOAT,
  thrpt_mbps_avg    FLOAT,
  jitter_ms_avg     FLOAT,
  pkt_loss_pct_avg  FLOAT,
  drop_rate_pct_avg FLOAT,
  sinr_db_avg       FLOAT,
  rsrp_dbm_avg      FLOAT,
  anomaly_rate      FLOAT,
  anomaly_any       SMALLINT,
  hour              SMALLINT,
  y_next_anomaly    SMALLINT
);

-- Suggested indexes/partitioning:
-- Postgres:
CREATE INDEX IF NOT EXISTS ix_gold_region_hour ON gold_cell_hourly(region, ts_hour);
-- Optional if very large: BRIN(ts_hour) for time-ranged scans
-- Teradata (conceptual):
-- PRIMARY INDEX (region, ts_hour);
-- PARTITION BY RANGE_N(ts_hour BETWEEN DATE '2025-01-01' AND DATE '2026-01-01' EACH INTERVAL '1' DAY);

## ✅ Forecasting: p95 Latency (Prophet/Holt-Winters) — Verified

**Why (maps to Verizon requirements):** Demonstrates applied analytical modeling on network KPIs (Prophet / statistical forecasting), production guardrails, and operational value for RF/IP performance monitoring and early-warning signals.

**What we did**
- Built a **time-series forecast** for hourly **latency p95** at the **cell** level using a safe training routine:
  - Resamples to hourly and **interpolates small gaps**.
  - Allows `--min-points` (default 24) to avoid short-series failures.
  - **Falls back to Holt-Winters** if Prophet isnt available.
- Used the latest **Gold** table (`data/gold/cell_hourly/`) for the region (12 cells, ~37 hourly points per cell).
- Produced an analyst-friendly **PNG plot** and a machine-readable **CSV artifact** with predictions and confidence bounds.

**How to run (Windows PowerShell)**
```powershell
# 1) Train + forecast (12h horizon, 24h history window fits our ~37h of data)
python ml\forecast_latency.py --region dfw --h 12 --hist 24

# 2) Sanity check the artifact
Get-Content .\artifacts\forecast_latest.csv | Select-Object -First 10

# 3) Create a plot for a given cell (last 48h context + future 12h)
python viz\plot_forecast.py --region dfw --cell CELL-001 --hours 48

### KPI Focus: P## PRB Utilization Forecasting (CELL-012, dfw)

We extended forecasting to **PRB utilization (%)**, a core RAN capacity KPI. Sustained PRB ≥80% indicates congestion that hurts throughput/latency; forecasting PRB shows applied **4G/5G + RF/SP engineering** judgment.

### Train & Visualize

```bash
# Train PRB forecast with safer settings and enough history
python ml\forecast_prb.py --region dfw --h 12 --hist 36 --min-points 36

# Visualize for CELL-012 (last 48h window)
python viz\plot_prb.py --region dfw --cell CELL-012 --hours 48
```
- **Why PRB?**  
  PRB utilization is a core 4G/5G KPI, directly reflecting how efficiently spectrum resources are used. Sustained PRB >80% is a warning sign for capacity issues, dropped calls, and degraded QoS.

- **Case: CELL-012 (region=dfw)**  
  Our baseline analysis identified CELL-012 as the most problematic cell, with anomalous spikes above 80% PRB utilization.  
  Using Prophet, we forecasted PRB load and uncertainty intervals. The results highlight widening risk bands post-cutoff, consistent with **radio congestion and scheduling pressure** in busy urban sites.

- **Network Expertise Demonstration**  
  - RF Engineering: PRB ties directly to spectrum scheduling, interference, and handover efficiency.  
  - IP Networking: High PRB utilization increases latency/jitter for IP traffic flows.  
  - System Reliability: Identifying congested PRB patterns supports proactive actions such as small-cell deployment or carrier aggregation.

This shows applied **RF/SP engineering principles** and demonstrates the ability to link AI forecasting with **real operator KPIs**.

**Artifacts**
- `artifacts/forecast_prb_latest.csv` — all cells/series  
- `artifacts/forecast_prb_dfw_CELL-012.png` — figure

---

### Problem We Found (and Why)

Our first quick run showed the forecast **plunging toward 0%** after the cutoff — **not physically plausible** for PRB and inconsistent with history.

**Root causes**
- **Short history (~24h)** → overfit to a local downward slope.  
- **Unbounded trend** → additive growth allowed negative/near-zero projections (then clipped 0–100 on plot, exaggerating the “cliff”).  
- **Spiky series / edge effects** with limited seasonality context.

This is the kind of modeling artifact **RF/ops engineers must catch before acting**.

---

### Fix Applied (Guardrails)

- Increased history and required coverage: `--hist 36`, `--min-points 36`.  
- Refit and replot; the forecast stabilized near recent PRB (≈55–60%) with realistic uncertainty.  
- Documented a sanity rule: if forecast **min** ≪ (recent **median − 2×std**) **and** no consistent downtrend, flag and retrain with more context.

**Takeaway:** Align model configuration with **network engineering intuition** so forecasts are trustworthy for proactive capacity actions (load-balancing, neighbor tuning, carrier add).

⚠️ **Project Status**: Work in progress. 
Current features: Spark Bronze→Silver, anomaly flags, partition-aware UI, analyst view.
Next features: Gold hourly features, baseline ML model (XGBoost), Prophet forecasts, Neo4j neighbor graphs. 
