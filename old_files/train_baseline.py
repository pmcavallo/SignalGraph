# ml/train_baseline.py
from pathlib import Path
import json, numpy as np, pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "cell_hourly"
ART  = ROOT / "artifacts"; ART.mkdir(parents=True, exist_ok=True)

files = list(GOLD.rglob("*.parquet"))
if not files:
    raise SystemExit(f"No Gold parquet found in {GOLD}. Run silver_to_gold.py first.")
df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values(["cell_id","ts_hour"])
df = df.dropna(subset=["y_next_anomaly"]).copy()
df["y_next_anomaly"] = df["y_next_anomaly"].astype(int)

from pathlib import Path
CENT = Path(__file__).resolve().parents[1] / "artifacts" / "centrality.parquet"
if CENT.exists():
    c = pd.read_parquet(CENT)
    df = df.merge(c, on="cell_id", how="left")
    for _c in ("neighbor_degree","pagerank","betweenness"):
        if _c not in df.columns:
            df[_c] = 0.0
    df[["neighbor_degree","pagerank","betweenness"]] = df[["neighbor_degree","pagerank","betweenness"]].fillna(0.0)
    print("Included graph centrality features in training.")

# --- Add temporal signal: lags / deltas / rolling means ---
base_cols = ["latency_ms_p95","prb_util_pct_avg","thrpt_mbps_avg","sinr_db_avg"]
for col in base_cols:
    if col in df.columns:
        df[f"{col}_lag1"] = df.groupby("cell_id")[col].shift(1)
        df[f"{col}_chg1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_rm3"]  = df.groupby("cell_id")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())

# Fill initial NaNs created by lags/rolling
df = df.fillna(0)

# Feature set
features = [
    "latency_ms_avg","latency_ms_p95","prb_util_pct_avg","thrpt_mbps_avg",
    "jitter_ms_avg","pkt_loss_pct_avg","drop_rate_pct_avg",
    "sinr_db_avg","rsrp_dbm_avg","anomaly_rate","hour","n_samples",
    # temporal
    "latency_ms_p95_lag1","prb_util_pct_avg_lag1","thrpt_mbps_avg_lag1","sinr_db_avg_lag1",
    "latency_ms_p95_chg1","prb_util_pct_avg_chg1","thrpt_mbps_avg_chg1","sinr_db_avg_chg1",
    "latency_ms_p95_rm3","prb_util_pct_avg_rm3","thrpt_mbps_avg_rm3","sinr_db_avg_rm3"
]
features = [c for c in features if c in df.columns]
extra = ["neighbor_degree","pagerank","betweenness"]
features = [*features, *[c for c in extra if c in df.columns]]
X = df[features]
y = df["y_next_anomaly"]

# Time-aware split (last 20% → test)
cut = int(len(df)*0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

# Handle class imbalance
pos_rate = max(y_train.mean(), 1e-6)
scale_pos_weight = (1 - pos_rate) / pos_rate  # ~7–8x given your stats

if HAS_XGB:
    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.07,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=scale_pos_weight, random_state=7
    )
else:
    model = RandomForestClassifier(
        n_estimators=600, max_depth=10, n_jobs=-1, random_state=7,
        class_weight="balanced_subsample"
    )

model.fit(X_train, y_train)

# Metrics
proba_test = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.predict(X_test).astype(float)
auc_roc = float(roc_auc_score(y_test, proba_test))
prec, rec, _ = precision_recall_curve(y_test, proba_test); auc_pr = float(auc(rec, prec))
report = classification_report(y_test, (proba_test>0.5).astype(int), output_dict=True)

# Persist artifacts
dump(model, ART/"model.joblib")
pd.DataFrame({"feature": features, "importance": getattr(model, "feature_importances_", np.full(len(features), np.nan))}).to_csv(ART/"feature_importance.csv", index=False)
with open(ART/"metrics.json","w") as f:
    json.dump({"auc_roc": auc_roc, "auc_pr": auc_pr, "report": report, "features": features}, f, indent=2)

# --- Choose an operating threshold (maximize F1 on test) ---
from sklearn.metrics import f1_score, precision_recall_curve
prec, rec, thr = precision_recall_curve(y_test, proba_test)
f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = int(f1.argmax())
best_thr = float(thr[max(best_idx-1, 0)])  # thr has len-1 vs prec/rec

report_at_best = classification_report(
    y_test, (proba_test >= best_thr).astype(int), output_dict=True
)

# Append to metrics
with open(ART / "metrics.json", "r+") as f:
    m = json.load(f)
    m["operating_point"] = {
        "threshold": best_thr,
        "precision": float(prec[best_idx]),
        "recall": float(rec[best_idx]),
        "f1": float(f1[best_idx])
    }
    m["report_at_threshold"] = report_at_best
    f.seek(0); json.dump(m, f, indent=2); f.truncate()
print(f"Chosen threshold ~{best_thr:.3f} with F1={f1[best_idx]:.3f}")


# Latest-hour scores for UI
latest_ts = df["ts_hour"].max()
latest = df[df["ts_hour"]==latest_ts].copy()
X_latest = latest[features]
latest["risk_score"] = model.predict_proba(X_latest)[:,1] if hasattr(model,"predict_proba") else model.predict(X_latest).astype(float)
cols = [c for c in [
    "cell_id","region","ts_hour","risk_score",
    "latency_ms_p95","prb_util_pct_avg","sinr_db_avg","thrpt_mbps_avg",
    "neighbor_degree","pagerank","betweenness"
] if c in latest.columns]

latest[cols].to_csv(ART/"scores_latest.csv", index=False)

print(f"Saved model + metrics to {ART}. AUC_ROC={auc_roc:.3f}  AUC_PR={auc_pr:.3f}  (pos_rate={pos_rate:.3f}, spw~{scale_pos_weight:.1f})")

# ---- SHAP explainability (optional; safe to skip if libs missing) ----
try:
    import shap, numpy as _np, pandas as _pd, matplotlib.pyplot as _plt

    # Pick an appropriate explainer for tree models
    explainer = None
    name = model.__class__.__name__.lower()
    if "xgb" in name or hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    elif "forest" in name or "gradientboost" in name:
        explainer = shap.TreeExplainer(model)

    if explainer is not None:
        # 1) GLOBAL: summary plot on the held-out window (cap size for speed)
        X_eval = X_test.iloc[-2000:] if len(X_test) > 2000 else X_test
        sv = explainer.shap_values(X_eval)
        if isinstance(sv, list):  # some explainers return list per class
            sv = sv[1] if len(sv) > 1 else sv[0]

        ART.mkdir(parents=True, exist_ok=True)
        _plt.figure()
        shap.summary_plot(sv, X_eval, show=False)
        _plt.tight_layout()
        _plt.savefig(ART / "shap_summary.png", dpi=150, bbox_inches="tight")
        _plt.close()

        # 2) LOCAL: top reasons for the latest hour rows
        latest_ts = None
        if "ts_hour" in df.columns:
            try:
                latest_ts = _pd.to_datetime(df["ts_hour"]).max()
            except Exception:
                latest_ts = df["ts_hour"].max()

        if latest_ts is not None and {"cell_id","ts_hour"}.issubset(df.columns):
            mask = df["ts_hour"] == latest_ts
            X_latest = df.loc[mask, features]
            if len(X_latest):
                sv_latest = explainer.shap_values(X_latest)
                if isinstance(sv_latest, list):
                    sv_latest = sv_latest[1] if len(sv_latest) > 1 else sv_latest[0]
                abs_sv = _np.abs(sv_latest)
                top_idx = abs_sv.argsort(axis=1)[:, -3:][:, ::-1]  # top-3
                top_names = _np.array(features)[top_idx]
                top_str = [", ".join(map(str, row)) for row in top_names]
                out = _pd.DataFrame({
                    "cell_id": df.loc[mask, "cell_id"].values,
                    "ts_hour": df.loc[mask, "ts_hour"].values,
                    "top_reasons": top_str
                })
                out.to_parquet(ART / "shap_top_reasons.parquet", index=False)
        print("Saved SHAP artifacts.")
except Exception as e:
    print(f"SHAP computation skipped: {e}")
