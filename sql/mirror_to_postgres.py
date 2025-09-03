# sql/mirror_to_postgres.py
from pathlib import Path
import os, io
import duckdb
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"
DB   = ART / "signalgraph.duckdb"

PG_DSN = os.getenv("SG_PG_DSN")
if not PG_DSN:
    raise SystemExit("❌ SG_PG_DSN not set (export your Neon DSN)")

# ---------- helpers ----------
def pg_type_from_dtype(dtype_str: str) -> str:
    s = str(dtype_str)
    if "int" in s:        return "BIGINT"
    if "float" in s:      return "DOUBLE PRECISION"
    if "bool" in s:       return "BOOLEAN"
    if "datetime" in s:   return "TIMESTAMPTZ"
    return "TEXT"

def coerce_df_to_pg_types(df: pd.DataFrame, pg_types: dict) -> pd.DataFrame:
    """Cast df columns to match PG column types to make COPY safe."""
    for c, t in pg_types.items():
        if c not in df.columns:
            continue
        try:
            if t in ("smallint", "integer", "bigint"):
                df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
            elif t == "boolean":
                s = pd.to_numeric(df[c], errors="coerce")
                df[c] = (s.fillna(0).round().astype("Int64") != 0).astype("boolean")
            elif t.startswith("timestamp"):
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        except Exception:
            df[c] = df[c].astype(str)
    return df

def mirror_table(conn, table_name: str, df: pd.DataFrame):
    """Create/extend PG table, coerce df types, truncate + COPY."""
    with conn.cursor() as cur:
        # Ensure table exists
        col_defs = [f"{c} {pg_type_from_dtype(t)}" for c, t in zip(df.columns, df.dtypes)]
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)});")

        # Get existing PG columns/types
        cur.execute("""
            SELECT column_name, data_type 
              FROM information_schema.columns 
             WHERE table_name = %s
        """, (table_name,))
        info = cur.fetchall()
        pg_cols  = {r[0] for r in info}
        pg_types = {r[0]: r[1] for r in info}

        # Add any missing columns
        for c in df.columns:
            if c not in pg_cols:
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {c} {pg_type_from_dtype(df[c].dtype)};")
                pg_types[c] = pg_type_from_dtype(df[c].dtype)

        # Coerce DF to PG types and COPY
        df = coerce_df_to_pg_types(df, pg_types)
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)

        cur.execute(f"TRUNCATE TABLE {table_name};")
        cur.copy_expert(f"COPY {table_name} ({', '.join(df.columns)}) FROM STDIN WITH CSV", buf)

    print(f"✅ Mirrored {len(df):,} rows -> {table_name}")

# ---------- main ----------
ddcon = duckdb.connect(str(DB))
df_gold = ddcon.execute("SELECT * FROM gold_cell_hourly").df()
ddcon.close()

with psycopg2.connect(PG_DSN) as conn:
    # Mirror only gold_cell_hourly
    mirror_table(conn, "gold_cell_hourly", df_gold)

    with conn.cursor() as cur:
        # Schema & index
        cur.execute("CREATE SCHEMA IF NOT EXISTS sg5g;")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scores_hour ON gold_cell_hourly(ts_hour);")

        # View: drop and recreate (avoids rename-in-view errors)
        cur.execute("DROP VIEW IF EXISTS sg5g.v_last_hour_risk;")
        cur.execute("""
            CREATE VIEW sg5g.v_last_hour_risk AS
            WITH h AS (SELECT max(ts_hour) ts_hour FROM gold_cell_hourly)
            SELECT s.*
              FROM gold_cell_hourly s
              JOIN h USING(ts_hour)
             ORDER BY ts_hour DESC
             LIMIT 50;
        """)
    conn.commit()

print("🎯 Postgres schema + index + view ready (fresh).")




