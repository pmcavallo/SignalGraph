import os, pathlib
from pyspark.sql import SparkSession, functions as F, Window
spark = SparkSession.builder.appName("sg5g_bronze_to_silver").getOrCreate()
root = pathlib.Path(__file__).resolve().parents[1]
in_path = str(root / "data" / "raw" / "cell_kpi_minute.parquet")
out_root = str(root / "data" / "silver" / "cell_kpi_minute")
df = spark.read.parquet(in_path)
# Enforce schema and add partitions
df2 = (df
    .withColumn("date", F.to_date("ts"))
    .withColumn("hour", F.hour("ts"))
    .withColumn("region", F.coalesce(F.col("region"), F.lit("unknown"))))
# Basic quality screens and a simple rule-based anomaly flag for the demo
df2 = (df2
    .filter(F.col("rsrp_dbm").isNotNull() & F.col("sinr_db").isNotNull())
    .withColumn("latency_poor", (F.col("latency_ms") > F.lit(60)))
    .withColumn("prb_high", (F.col("prb_util_pct") > F.lit(85)))
    .withColumn("anomaly_flag", (F.col("latency_poor") | F.col("prb_high")).cast("int")))
(df2
 .repartition("date","region")
 .write
 .mode("overwrite")
 .partitionBy("date","region")
 .parquet(out_root))
print(f"Wrote Silver partitions to {out_root}")

