# ml/forecast_prb.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import pandas as pd
import numpy as np
# Optional: interactive charts. Stay quiet if Plotly isn't available.
try:
    import plotly.express as 
    _PLOTLY = True
except Exception:
    _PLOTLY = False  # no print — skip charts silently

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "cell_hourly"
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

TARGET_COL = "prb_util_pct_avg"   # PRB utilization in percent (0-100)
OUT_CSV    = ART / "forecast_prb_latest.csv"

# ---------- helpers ----------
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
    """hist columns: ds, y (0..100). Uses bounded logistic growth."""
    from prophet import Prophet  # type: ignore

    # Add logistic bounds: 0..100
    h = hist.copy()
    h["cap"] = 100.0
    h["floor"] = 0.0

    m = Prophet(
        growth="logistic",
        seasonality_mode="multiplicative",  # percentages behave better
        interval_width=0.8,
        changepoint_prior_scale=0.1,        # slightly smoother trend
        seasonality_prior_scale=5.0,
    )
    m.add_seasonality(name="daily", period=24, fourier_order=6)

    m.fit(h[["ds", "y", "cap", "floor"]])

    future = m.make_future_dataframe(periods=horizon, freq="h")
    future["cap"] = 100.0
    future["floor"] = 0.0

    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc["model"] = "prophet_logistic"
    return fc


def fit_predict_hw(hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Holt-Winters fallback (needs statsmodels)."""
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

def fit_predict_naive(hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Last-value naive fallback if both Prophet and statsmodels are unavailable."""
    y = hist.set_index("ds")["y"].asfreq("H").interpolate(limit_direction="both")
    last = float(y.iloc[-1])
    idx  = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="H")
    fc   = pd.DataFrame({"ds": idx, "yhat": last, "yhat_lower": last, "yhat_upper": last})
    fc["model"] = "naive_last"
    return fc

# ---------- main ----------
def run(region: str, horizon: int, keep_hist_hours: int, min_points: int):
    df = load_gold(region)
    need = {"cell_id","ts_hour",TARGET_COL}
    if not need.issubset(df.columns):
        raise ValueError(f"Gold missing columns: need {need}, got {set(df.columns)}")

    # Ensure numeric + valid range
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").clip(0, 100)

    out_frames = []
    too_short = []

    for cell, g in df.groupby("cell_id"):
        ts = (g[["ts_hour", TARGET_COL]]
              .dropna()
              .rename(columns={"ts_hour":"ds", TARGET_COL:"y"})
              .sort_values("ds"))

        # strict hourly grid + fill small gaps
        ts = ts.set_index("ds").resample("h").mean().interpolate(limit_direction="both").reset_index()

        # Winsorize extreme spikes (robust to outliers)
        low, high = np.nanpercentile(ts["y"], [1, 99])
        ts["y"] = ts["y"].clip(low, high)

        # Optional light smoothing to reduce single-hour noise
        # ts["y"] = ts["y"].rolling(3, min_periods=1, center=True).mean()

        if len(ts) < min_points:
            too_short.append((cell, len(ts)))
            continue

        # forecast (Prophet -> Holt-Winters -> Naive)
        try:
            fc = fit_predict_prophet(ts, horizon)
        except Exception:
            try:
                fc = fit_predict_hw(ts, horizon)
            except Exception:
                fc = fit_predict_naive(ts, horizon)

        # Constrain to [0,100]
        for c in ["yhat","yhat_lower","yhat_upper"]:
            if c in fc:
                fc[c] = pd.to_numeric(fc[c], errors="coerce").clip(0, 100)

        # Recent history for plotting context
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

        out_frames.append(pd.concat([hist_tail, fc], ignore_index=True))

    if not out_frames:
        raise RuntimeError(f"No series were long enough to forecast (min_points={min_points}). "
                           f"Examples: {too_short[:5]}")

    res = pd.concat(out_frames, ignore_index=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {len(res):,} rows to {OUT_CSV} (cells={res['cell_id'].nunique()}, region={region})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Forecast PRB utilization (%) per cell")
    p.add_argument("--region", default="dfw")
    p.add_argument("--h", type=int, default=12, help="forecast horizon (hours)")
    p.add_argument("--hist", type=int, default=24, help="historic hours to keep for plot context")
    p.add_argument("--min-points", type=int, default=24, help="minimum history points per cell")
    a = p.parse_args()
    run(a.region, a.h, a.hist, a.min_points)
