# ml/forecast_latency.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "cell_hourly"
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def load_gold(region: str) -> pd.DataFrame:
    files = list(GOLD.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No gold parquet found under {GOLD}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "region" in df.columns:
        df = df[df["region"] == region]
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    return df.sort_values(["cell_id","ts_hour"])

def try_import_prophet():
    try:
        from prophet import Prophet  # type: ignore
        return Prophet
    except Exception:
        return None

def fit_predict_prophet(hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    Prophet = try_import_prophet()
    if Prophet is None:
        raise ImportError("prophet not available")
    m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False,
                interval_width=0.8)
    m.add_seasonality(name="daily", period=24, fourier_order=6)
    m.fit(hist.rename(columns={"ds":"ds","y":"y"}))
    future = m.make_future_dataframe(periods=horizon, freq="H")
    fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
    fc["model"] = "prophet"
    return fc

def fit_predict_hw(hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
    y = hist.set_index("ds")["y"].asfreq("H").interpolate(limit_direction="both")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HW(y, trend="add", seasonal="add", seasonal_periods=24)
        fit = model.fit(optimized=True)
    pred = fit.forecast(horizon)
    resid_std = float(np.sqrt(np.nanmean((fit.fittedvalues - y) ** 2)))
    lo = pred - 1.28 * resid_std
    hi = pred + 1.28 * resid_std
    fc = pd.DataFrame({"ds": pred.index, "yhat": pred.values,
                       "yhat_lower": lo.values, "yhat_upper": hi.values})
    fc["model"] = "holt_winters"
    return fc

def run(region: str, horizon: int, keep_hist_hours: int, min_points: int):
    df = load_gold(region)
    need = {"cell_id","ts_hour","latency_ms_p95"}
    if not need.issubset(df.columns):
        raise ValueError(f"Gold missing columns: need {need}, got {set(df.columns)}")

    out = []
    too_short = []

    for cell, g in df.groupby("cell_id"):
        ts = (g[["ts_hour","latency_ms_p95"]]
              .dropna()
              .rename(columns={"ts_hour":"ds","latency_ms_p95":"y"})
              .sort_values("ds"))

        # enforce strict hourly grid + fill small gaps
        ts = ts.set_index("ds").resample("H").mean().interpolate(limit_direction="both").reset_index()

        if len(ts) < min_points:
            too_short.append((cell, len(ts)))
            continue

        # forecast
        try:
            fc = fit_predict_prophet(ts, horizon)
        except Exception:
            fc = fit_predict_hw(ts, horizon)

        # add recent history for plotting context
        hist_tail = ts.tail(keep_hist_hours).copy()
        hist_tail["yhat"] = np.nan
        hist_tail["yhat_lower"] = np.nan
        hist_tail["yhat_upper"] = np.nan
        hist_tail["model"] = "observed"
        hist_tail["cell_id"] = cell
        hist_tail["region"] = region

        fc["cell_id"] = cell
        fc["region"] = region
        fc["y"] = np.nan

        out.append(pd.concat([hist_tail, fc], ignore_index=True))

    if not out:
        raise RuntimeError(f"No series were long enough to forecast (min_points={min_points}). "
                           f"Examples: {too_short[:5]}")

    res = pd.concat(out, ignore_index=True)
    res.to_csv(ART / "forecast_latest.csv", index=False)
    print(f"✅ Wrote {len(res):,} rows to {ART/'forecast_latest.csv'} "
          f"(cells={res['cell_id'].nunique()}, region={region})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="dfw")
    p.add_argument("--h", type=int, default=24, help="forecast horizon (hours)")
    p.add_argument("--hist", type=int, default=48, help="historic hours to keep for plot context")
    p.add_argument("--min-points", type=int, default=24, help="minimum history points per cell")
    a = p.parse_args()
    run(a.region, a.h, a.hist, a.min_points)
