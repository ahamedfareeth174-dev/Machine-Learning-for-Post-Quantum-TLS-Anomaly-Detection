"""Generate report figures from dataset + saved model artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Dataset"
ART = ROOT / "artifacts"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

data = pd.read_csv(DATA / "ml_features_and_labels.csv")
manifest = pd.read_csv(DATA / "scenario_manifest.csv")
manifest["join_id"] = manifest["ID"].str.replace(r"\.pcap$", "", regex=True)

meta = {"ID", "label", "split", "taxonomy"}
feature_cols = [c for c in data.columns if c not in meta]
X = data[feature_cols].astype(float)
y = data["label"].astype(int)
masks = {s: data["split"].eq(s) for s in ("train", "val", "test")}
X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

# 01 class imbalance
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.countplot(data=data, x="label", ax=ax, palette=["#1f7a4c", "#b42318"])
ax.set_xticklabels(["Normal (0)", "Anomaly (1)"])
ax.set_title("Class imbalance in CIC-PQC_OAV")
fig.tight_layout()
fig.savefig(FIG / "01_class_imbalance.png", dpi=160)
plt.close(fig)

# 02 boxplots
numeric_features = [
    "e2c_total_bytes", "e4_entropy_h", "e5_entropy_c", "e6b_flow_duration_ms"
]
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, feature in zip(axes.ravel(), numeric_features):
    sns.boxplot(
        data=data, x="label", y=feature, showfliers=False, ax=ax,
        palette=["#1f7a4c", "#b42318"],
    )
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_title(feature)
fig.suptitle("Numeric feature distributions by class", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "02_numeric_boxplots.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Load models + thresholds from artifacts
import json

summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
thresholds = summary["thresholds"]
models = {}
for name in summary["models"]:
    safe = name.lower().replace(" ", "_")
    models[name] = joblib.load(ART / f"model_{safe}.joblib")

# 03 confusion matrices
for name, model in models.items():
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= thresholds[name]).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Normal", "Anomaly"], cmap="Blues", ax=ax
    )
    ax.set_title(name)
    fig.tight_layout()
    safe = name.lower().replace(" ", "_")
    fig.savefig(FIG / f"03_confusion_{safe}.png", dpi=160)
    plt.close(fig)

# Best model error analysis
best = summary["best_model"]
best_model = models[best]
prob = best_model.predict_proba(X_test)[:, 1]
pred = (prob >= thresholds[best]).astype(int)

errors = data.loc[masks["test"], ["ID", "label"]].copy()
errors["prediction"] = pred
errors["anomaly_probability"] = prob
errors["error_type"] = np.select(
    [
        (errors.label == 0) & (errors.prediction == 1),
        (errors.label == 1) & (errors.prediction == 0),
    ],
    ["False positive", "False negative"],
    default="Correct",
)
errors = errors.merge(
    manifest.drop(columns=["ID"]),
    left_on="ID",
    right_on="join_id",
    how="left",
    validate="one_to_one",
)

anom = errors[errors["label"] == 1]
subtype_recall = (
    anom.groupby("subtype")
    .apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "detected": int((d["prediction"] == 1).sum()),
                "recall": float((d["prediction"] == 1).mean()),
            }
        ),
        include_groups=False,
    )
    .sort_values("recall")
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    data=subtype_recall.reset_index(), x="recall", y="subtype", ax=ax, color="#0f7f78"
)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Anomaly recall")
ax.set_title(f"Test anomaly recall by subtype ({best})")
fig.tight_layout()
fig.savefig(FIG / "04_subtype_recall.png", dpi=160)
plt.close(fig)

# SHAP global
tree_name = best if best in ("Random Forest", "XGBoost") else "XGBoost"
tree = models[tree_name]
sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
imputer = tree.named_steps["imputer"]
feature_names = list(imputer.get_feature_names_out(feature_cols))
sample_for_shap = pd.DataFrame(imputer.transform(sample), columns=feature_names, index=sample.index)
explainer = shap.TreeExplainer(tree.named_steps["classifier"])
explanation = explainer(sample_for_shap)
if explanation.values.ndim == 3:
    explanation = explanation[:, :, 1]

plt.figure()
shap.summary_plot(explanation.values, sample_for_shap, show=False)
plt.tight_layout()
plt.savefig(FIG / "05_shap_summary.png", dpi=160, bbox_inches="tight")
plt.close()

mean_abs = np.abs(explanation.values).mean(axis=0)
importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 5))
importance.iloc[::-1].plot(kind="barh", ax=ax, color="#0f7f78")
ax.set_xlabel("mean |SHAP|")
ax.set_title(f"Global feature importance ({tree_name})")
fig.tight_layout()
fig.savefig(FIG / "06_shap_mean_abs.png", dpi=160)
plt.close(fig)

# Waterfalls
id_to_index = pd.Series(
    data.loc[masks["test"]].index.values,
    index=data.loc[masks["test"], "ID"].values,
)

def pick(tag_filter, sort_col, ascending):
    subset = errors[tag_filter]
    if subset.empty:
        return None
    return subset.sort_values(sort_col, ascending=ascending).iloc[0]

picks = [
    ("correct_anomaly", pick((errors.label == 1) & (errors.error_type == "Correct"), "anomaly_probability", False)),
    ("false_negative", pick(errors.error_type == "False negative", "anomaly_probability", True)),
    ("correct_normal", pick((errors.label == 0) & (errors.error_type == "Correct"), "anomaly_probability", True)),
]

for tag, row in picks:
    if row is None:
        continue
    idx = id_to_index[row["ID"]]
    X_row = X_test.loc[[idx]]
    Xt = pd.DataFrame(imputer.transform(X_row), columns=feature_names, index=X_row.index)
    local_exp = explainer(Xt)
    if local_exp.values.ndim == 3:
        local_exp = local_exp[:, :, 1]
    plt.figure()
    shap.plots.waterfall(local_exp[0], max_display=12, show=False)
    plt.title(f"{tag}: {str(row['ID'])[:48]}")
    plt.tight_layout()
    plt.savefig(FIG / f"07_shap_waterfall_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"saved waterfall {tag}")

# Copy metrics table
pd.DataFrame(summary["metrics"]["test"]).to_csv(FIG / "model_comparison.csv", index=False)

print("Wrote figures to", FIG)
for p in sorted(FIG.glob("*")):
    print(" -", p.name)
