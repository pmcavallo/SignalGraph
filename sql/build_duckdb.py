import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
scores_path = ART / "scores_latest.csv"

# load scores from CSV
scores = pd.read_csv(scores_path)

# connect to DuckDB file (will create if not exists)
con = duckdb.connect(str(ART / "signalgraph.duckdb"))

# register dataframe into DuckDB
con.register("scores_view", scores)

# create or replace the table from the registered view
con.execute("""
    CREATE OR REPLACE TABLE scores_latest AS 
    SELECT * FROM scores_view
""")

# optional: check row count
row_count = con.execute("SELECT COUNT(*) FROM scores_latest").fetchone()[0]
print(f"✅ Saved scores_latest to DuckDB with {row_count} rows")

# after saving scores_latest
gold_path = ROOT / "data" / "gold" / "cell_hourly"
if gold_path.exists():
    gold = pd.concat([pd.read_parquet(p) for p in gold_path.rglob("*.parquet")], ignore_index=True)
    con.register("gold_cell_hourly_view", gold)
    con.execute("CREATE OR REPLACE TABLE gold_cell_hourly AS SELECT * FROM gold_cell_hourly_view")
    print(f"✅ Saved gold_cell_hourly to DuckDB with {len(gold):,} rows")
