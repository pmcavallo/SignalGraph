from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def main(region: str | None, cell: str | None, hours: int):
    fpath = ART / "forecast_latest.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing {fpath}. Run the forecasting step first.")

    df = pd.read_csv(fpath)
    # Basic hygiene
    df.columns = [c.strip().lower() for c in df.columns]
    if "ds" not in df.columns:
        raise ValueError(f"{fpath} missing 'ds' column; got {df.columns}")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if region and "region" in df.columns:
        df = df[df["region"] == region]

    # Choose a cell if not specified
    if cell is None:
        cell = df["cell_id"].dropna().astype(str).unique().tolist()[0]

    d = df[df["cell_id"].astype(str) == str(cell)].sort_values("ds")
    if d.empty:
        raise ValueError(f"No rows for cell_id={cell} (region={region}).")

    # Split observed vs forecast
    obs = d[d["model"] == "observed"].copy()
    fc  = d[d["model"] != "observed"].copy()

    if obs.empty or fc.empty:
        raise ValueError("Need both observed and forecast rows for a good plot. "
                         f"Observed={len(obs)}, Forecast={len(fc)}")

    last_obs = obs["ds"].max()
    start    = last_obs - pd.Timedelta(hours=hours)

    # Limit window for readability
    obs = obs[obs["ds"] >= start]
    fc  = fc[fc["ds"] >= start]

    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(obs["ds"], obs["y"], label="Observed p95 latency (ms)", linewidth=2)

    # Forecast mean
    if "yhat" in fc:
        ax.plot(fc["ds"], fc["yhat"], label="Forecast (yhat)", linewidth=2)

    # Confidence band if present
    if {"yhat_lower", "yhat_upper"}.issubset(fc.columns):
        ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                        alpha=0.2, label="Forecast interval")

    # Vertical cutover line
    ax.axvline(last_obs, color="gray", linestyle="--", linewidth=1)
    ax.text(last_obs, ax.get_ylim()[1]*0.98, "cutover", rotation=90,
            va="top", ha="right", color="gray", fontsize=8)

    ax.set_title(f"Latency p95 forecast — cell {cell} (region={region or 'n/a'})")
    ax.set_xlabel("Time (hourly)")
    ax.set_ylabel("Latency p95 (ms)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=False)

    out = ART / f"forecast_{(region or 'all')}_{cell}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # Console summary
    horizon = fc["ds"].nunique()
    model_name = fc["model"].dropna().astype(str).unique().tolist()
    print(f"Saved plot: {out}")
    print(f"Last observed: {last_obs} | Forecast horizon: {horizon} hours | Model(s): {model_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="dfw")
    ap.add_argument("--cell",   default=None, help="e.g., CELL-001 (defaults to first available)")
    ap.add_argument("--hours",  type=int, default=48, help="plot window back from cutover")
    args = ap.parse_args()
    try:
        main(args.region, args.cell, args.hours)
    except ModuleNotFoundError as e:
        if "matplotlib" in str(e):
            print("matplotlib not found. Install with:\n  python -m pip install matplotlib")
        raise
