# viz/plot_prb.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"

def main(region: str | None, cell: str | None, hours: int):
    fpath = ART / "forecast_prb_latest.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing {fpath}. Run the PRB forecast first.")

    df = pd.read_csv(fpath)
    df.columns = [c.strip().lower() for c in df.columns]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if region and "region" in df.columns:
        df = df[df["region"] == region]

    if cell is None:
        cell = df["cell_id"].dropna().astype(str).unique().tolist()[0]

    d = df[df["cell_id"].astype(str) == str(cell)].sort_values("ds")
    if d.empty:
        raise ValueError(f"No rows for cell_id={cell} (region={region}).")

    obs = d[d["model"] == "observed"].copy()
    fc  = d[d["model"] != "observed"].copy()
    last_obs = obs["ds"].max()
    start    = last_obs - pd.Timedelta(hours=hours)
    obs = obs[obs["ds"] >= start]
    fc  = fc[fc["ds"] >= start]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(obs["ds"], obs["y"], label="Observed PRB util (%)", linewidth=2)
    if "yhat" in fc:
        ax.plot(fc["ds"], fc["yhat"], label="Forecast (yhat)", linewidth=2)
    if {"yhat_lower","yhat_upper"}.issubset(fc.columns):
        ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.2, label="Forecast interval")

    ax.axvline(last_obs, color="gray", linestyle="--", linewidth=1)
    ax.text(last_obs, 98, "cutover", rotation=90, va="top", ha="right", color="gray", fontsize=8)

    ax.set_title(f"PRB utilization forecast — cell {cell} (region={region or 'n/a'})")
    ax.set_xlabel("Time (hourly)")
    ax.set_ylabel("PRB utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=False)

    out = ART / f"forecast_prb_{(region or 'all')}_{cell}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    horizon = fc["ds"].nunique()
    model_name = fc["model"].dropna().astype(str).unique().tolist()
    print(f"Saved plot: {out}")
    print(f"Last observed: {last_obs} | Forecast horizon: {horizon} hours | Model(s): {model_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="dfw")
    ap.add_argument("--cell",   default=None)
    ap.add_argument("--hours",  type=int, default=48)
    args = ap.parse_args()
    try:
        main(args.region, args.cell, args.hours)
    except ModuleNotFoundError as e:
        if "matplotlib" in str(e):
            print("matplotlib not found. Install with:\n  python -m pip install matplotlib")
        raise
