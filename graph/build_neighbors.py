# graph/build_neighbors.py
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "cell_hourly"
ART  = ROOT / "artifacts"
OUT  = ROOT / "data" / "external" / "cell_neighbors.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_cells(prefer_region: str | None):
    """Return DataFrame with columns ['region','cell_id'] (region optional)."""
    # 1) Gold parquet (best)
    try:
        import pyarrow.dataset as ds
        dset = ds.dataset(str(GOLD), format="parquet", partitioning="hive")
        cols = [c for c in ["region","cell_id"] if c in dset.schema.names]
        if cols:
            df = dset.to_table(columns=list(set(cols+["cell_id","region"]))).to_pandas()
            df = df.dropna(subset=["cell_id"]).drop_duplicates(subset=["region","cell_id"], keep="first")
            return df
    except Exception:
        pass

    # 2) PRB forecast (commonly has region + cell_id)
    f_prb = ART / "forecast_prb_latest.csv"
    if f_prb.exists():
        df = pd.read_csv(f_prb)
        # normalize possible variants
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "cell" or lc == "cellid": rename[c] = "cell_id"
            if lc == "market" or lc == "cluster": rename[c] = "region"
        if rename: df = df.rename(columns=rename)
        keep = [c for c in ["region","cell_id"] if c in df.columns]
        if "cell_id" in keep:
            df = df[keep].drop_duplicates()
            if "region" not in df.columns:
                df["region"] = prefer_region or "global"
            return df[["region","cell_id"]]

    # 3) scores (may lack region)
    f_sc = ART / "scores_latest.csv"
    if f_sc.exists():
        df = pd.read_csv(f_sc)
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "cell" or lc == "cellid": rename[c] = "cell_id"
            if lc == "market" or lc == "cluster": rename[c] = "region"
        if rename: df = df.rename(columns=rename)
        if "cell_id" in df.columns:
            if "region" not in df.columns:
                df["region"] = prefer_region or "global"
            return df[["region","cell_id"]].drop_duplicates()

    raise FileNotFoundError("Could not derive cells from Gold, forecast_prb_latest.csv, or scores_latest.csv.")

def make_ring_edges(cells: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    edges = []
    for region, grp in cells.groupby("region"):
        ids = sorted(grp["cell_id"].astype(str).unique().tolist())
        n = len(ids)
        for i, cid in enumerate(ids):
            for d in range(1, k+1):
                j = i + d
                if j < n:
                    edges.append((cid, ids[j]))
    return pd.DataFrame(edges, columns=["src_cell_id","dst_cell_id"]).drop_duplicates()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--region", default=None, help="Assign a region when source lacks one (e.g., dfw)")
    p.add_argument("--k", type=int, default=2, help="neighbors per side for ring graph")
    a = p.parse_args()

    cells = load_cells(a.region)
    E = make_ring_edges(cells, k=a.k)
    E.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(E)} edges for {cells['cell_id'].nunique()} cells → {OUT}")

