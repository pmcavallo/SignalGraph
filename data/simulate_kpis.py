import os, pathlib, numpy as np, pandas as pd
rng = np.random.default_rng(7)
N_CELLS = 12
MINUTES = 24 * 6  # 6 hours at 15-minute cadence for a fast demo
cell_ids = [f"CELL-{i:03d}" for i in range(1, N_CELLS+1)]
regions = rng.choice(["dfw","nyc","sea"], size=N_CELLS, replace=True)
cells = pd.DataFrame({"cell_id": cell_ids, "region": regions})
start = pd.Timestamp.utcnow().floor("15min") - pd.Timedelta(minutes=15*(MINUTES-1))
ts = pd.date_range(start=start, periods=MINUTES, freq="15min")
rows = []
for _, r in cells.iterrows():
    base_rsrp = rng.normal(-95, 5)
    base_sinr = rng.normal(12, 3)
    prb = np.clip(rng.normal(55, 15, size=MINUTES), 0, 100)
    thrpt = np.clip(rng.normal(150, 40, size=MINUTES) - prb*0.5, 1, None)
    latency = np.clip(rng.normal(25, 8, size=MINUTES) + prb*0.2, 1, None)
    jitter = np.clip(rng.normal(4, 2, size=MINUTES), 0, None)
    loss = np.clip(rng.normal(0.2, 0.15, size=MINUTES), 0, None)
    # Inject a controlled degradation window to make the demo interesting
    spike_idx = rng.integers(low=MINUTES//3, high=MINUTES-10)
    latency[spike_idx:spike_idx+4] += rng.normal(40, 10, size=4)
    prb[spike_idx:spike_idx+4] += rng.normal(25, 8, size=4)
    for i, t in enumerate(ts):
        rows.append({
            "ts": t, "cell_id": r.cell_id, "region": r.region,
            "rsrp_dbm": rng.normal(base_rsrp, 3),
            "rsrq_db": rng.normal(-10, 2),
            "sinr_db": np.clip(rng.normal(base_sinr, 2), -5, 30),
            "prb_util_pct": float(prb[i]),
            "thrpt_mbps": float(thrpt[i]),
            "drop_rate_pct": float(np.clip(rng.normal(0.6, 0.4), 0, 5)),
            "latency_ms": float(latency[i]),
            "jitter_ms": float(jitter[i]),
            "pkt_loss_pct": float(loss[i]),
        })
df = pd.DataFrame(rows).sort_values(["ts","cell_id"]).reset_index(drop=True)
root = pathlib.Path(__file__).resolve().parents[1]
outdir = root / "data" / "raw"
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / "cell_kpi_minute.parquet"
df.to_parquet(outpath)
print(f"Wrote {len(df):,} rows to {outpath}")
