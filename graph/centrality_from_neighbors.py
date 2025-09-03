# graph/centrality_from_neighbors.py
from pathlib import Path
import pandas as pd, networkx as nx
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"; ART.mkdir(exist_ok=True, parents=True)
EDGES = ROOT / "data" / "external" / "cell_neighbors.csv"

if not EDGES.exists():
    print("No neighbor list found. Building a default one...")
    subprocess.check_call([sys.executable, str(ROOT/"graph"/"build_neighbors.py"), "--region", "dfw"])

E = pd.read_csv(EDGES)
G = nx.Graph(); G.add_edges_from(E[["src_cell_id","dst_cell_id"]].itertuples(index=False, name=None))
deg = pd.Series(dict(nx.degree(G)), name="neighbor_degree")
pr  = pd.Series(nx.pagerank(G), name="pagerank")
btw = pd.Series(nx.betweenness_centrality(G), name="betweenness")
out = (pd.concat([deg, pr, btw], axis=1).rename_axis("cell_id").reset_index())
out.to_parquet(ART / "centrality.parquet", index=False)
print(f"✅ Wrote centrality features for {len(out)} cells → {ART/'centrality.parquet'}")

