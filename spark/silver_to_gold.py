# spark/silver_to_gold.py
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder.appName("sg5g_silver_to_gold").getOrCreate()
ROOT = Path(__file__).resolve().parents[1]
SILVER = str(ROOT / "data" / "silver" / "cell_kpi_minute")
GOLD   = str(ROOT / "data" / "gold"  / "cell_hourly")

df = spark.read.parquet(SILVER)

# Hour bucket for grouping & ordering
df = df.withColumn("ts_hour", F.date_trunc("hour", F.col("ts")))

# Hourly aggregates per cell/region/hour
agg = (
    df.groupBy("cell_id", "region", "ts_hour")
      .agg(
        F.count("*").alias("n_samples"),
        F.avg("latency_ms").alias("latency_ms_avg"),
        F.expr("percentile_approx(latency_ms, 0.95, 100)").alias("latency_ms_p95"),
        F.avg("prb_util_pct").alias("prb_util_pct_avg"),
        F.avg("thrpt_mbps").alias("thrpt_mbps_avg"),
        F.avg("jitter_ms").alias("jitter_ms_avg"),
        F.avg("pkt_loss_pct").alias("pkt_loss_pct_avg"),
        F.avg("drop_rate_pct").alias("drop_rate_pct_avg"),
        F.avg("sinr_db").alias("sinr_db_avg"),
        F.avg("rsrp_dbm").alias("rsrp_dbm_avg"),
        F.avg(F.col("anomaly_flag").cast("double")).alias("anomaly_rate"),
        F.max(F.col("anomaly_flag")).alias("anomaly_any")  # <- any anomaly this hour
      )
      .withColumn("date", F.to_date("ts_hour"))
      .withColumn("hour", F.hour("ts_hour"))
)

# Label = any anomaly next hour? (per cell)
w = Window.partitionBy("cell_id").orderBy(F.col("ts_hour").asc())
agg = agg.withColumn("y_next_anomaly", F.lead(F.col("anomaly_any"), 1).over(w).cast("int"))

# Write partitioned Gold
(agg.repartition("date","region")
    .write.mode("overwrite")
    .partitionBy("date","region")
    .parquet(GOLD))

print(f"Wrote Gold partitions to {GOLD}")
spark.stop()
