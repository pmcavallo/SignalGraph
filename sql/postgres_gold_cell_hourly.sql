-- gold_cell_hourly (teradata-style types)
CREATE TABLE gold_cell_hourly (
  cell_id        VARCHAR(20),
  region         VARCHAR(20),
  ts_hour        TIMESTAMP,
  n_samples      INTEGER,
  latency_ms_avg FLOAT,
  latency_ms_p95 FLOAT,
  prb_util_pct_avg FLOAT,
  thrpt_mbps_avg FLOAT,
  jitter_ms_avg  FLOAT,
  pkt_loss_pct_avg FLOAT,
  drop_rate_pct_avg FLOAT,
  sinr_db_avg    FLOAT,
  rsrp_dbm_avg   FLOAT,
  anomaly_rate   FLOAT,
  anomaly_any    BYTEINT,
  hour           BYTEINT,
  y_next_anomaly BYTEINT
);

-- schema
create schema if not exists sg5g;

-- core table (if not already created)
create table if not exists sg5g.fact_cell_hour(
  cell_id text,
  region text,
  ts_hour timestamptz,
  latency_ms_avg double precision,
  latency_ms_p95 double precision,
  prb_util_pct_avg double precision,
  thrpt_mbps_avg double precision,
  sinr_db_avg double precision,
  rsrp_dbm_avg double precision,
  anomaly_rate double precision,
  anomaly_any smallint,
  y_next_anomaly smallint,
  neighbor_degree double precision,
  pagerank double precision,
  betweenness double precision
);

-- scores mart (if not already present)
create table if not exists sg5g.mart_cell_hour_scores(
  cell_id text,
  region text,
  ts_hour timestamptz,
  risk_score double precision,
  latency_ms_p95 double precision,
  prb_util_pct_avg double precision,
  sinr_db_avg double precision,
  thrpt_mbps_avg double precision
);

-- indexes used by the app/queries
create index if not exists idx_sg5g_fch_hour_region on sg5g.fact_cell_hour (ts_hour, region);
create index if not exists idx_sg5g_fch_cell on sg5g.fact_cell_hour (cell_id);
create index if not exists idx_sg5g_scores_hour on sg5g.mart_cell_hour_scores (ts_hour);
create index if not exists idx_sg5g_scores_cell on sg5g.mart_cell_hour_scores (cell_id);

-- “last hour, high-risk” view that your Streamlit page can query directly
create or replace view sg5g.v_last_hour_risk as
with h as (
  select max(ts_hour) as ts_hour from sg5g.mart_cell_hour_scores
)
select s.*
from sg5g.mart_cell_hour_scores s
join h on s.ts_hour = h.ts_hour
order by s.risk_score desc
limit 50;
