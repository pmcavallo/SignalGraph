# ml/inspect_gold.py
from pathlib import Path
import argparse, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "cell_hourly"

def main(region: str):
    files = list(GOLD.rglob("*.parquet"))
    if not files:
        print(f"❌ No Gold parquet found under {GOLD}")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "ts_hour" not in df.columns or "cell_id" not in df.columns:
        print("❌ Expected columns not found. Got:", list(df.columns))
        return
    df["ts_hour"] = pd.to_datetime(df["ts_hour"], errors="coerce")
    if "region" in df.columns and region:
        df = df[df["region"] == region]

    print(f"✅ rows={len(df)}  cells={df['cell_id'].nunique()}  region={region}")
    g = (
        df.groupby("cell_id")["ts_hour"]
          .agg(n="count", start="min", end="max")
          .sort_values("n", ascending=False)
    )
    print("\nPer-cell counts and span:\n")
    print(g.to_string())

    # quick gap check on the densest cell
    if not g.empty:
        top = g.index[0]
        s = (df[df["cell_id"] == top]
             .set_index("ts_hour")
             .sort_index()
             .resample("H").size())
        missing = int((s == 0).sum())
        print(f"\nSample gap check: {top}  hours={len(s)}  missing_hours={missing}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="dfw")
    a = ap.parse_args()
    main(a.region)
