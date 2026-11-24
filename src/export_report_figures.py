"""Export report figures using saved artifacts (no full retrain)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import ConfusionMatrixDisplay

from .data import load_manifest, load_ml_frame, split_xy
from .paths import ARTIFACTS_DIR, ROOT

FIGURES_DIR = ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    summary = json.loads((ARTIFACTS_DIR / "summary.json").read_text(encoding="utf-8"))
    data = load_ml_frame()
    manifest = load_manifest()
    feature_cols, X, y, masks = split_xy(data)
    X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

    # 01 class imbalance
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.countplot(data=data, x="label", ax=ax, palette=["#1f7a4c", "#b42318"])
    ax.set_xticklabels(["Normal (0)", "Anomaly (1)"])
    ax.set_title("Class imbalance in CIC-PQC_OAV")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_imbalance.png", dpi=160)
    plt.close(fig)

    # 02 numeric boxplots
    numeric_features = [
        "e2c_total_bytes",
        "e4_entropy_h",
        "e5_entropy_c",
        "e6b_flow_duration_ms",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, feature in zip(axes.ravel(), numeric_features):
        sns.boxplot(
            data=data,
            x="label",
            y=feature,
            showfliers=False,
            ax=ax,
            palette=["#1f7a4c", "#b42318"],
        )
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_title(feature)
    fig.suptitle("Numeric feature distributions by class", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_numeric_boxplots.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Load models + confusion matrices
    models = {}
    for name in summary["models"]:
        safe = name.lower().replace(" ", "_")
        models[name] = joblib.load(ARTIFACTS_DIR / f"model_{safe}.joblib")

    thresholds = summary["thresholds"]
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
        fig.savefig(FIGURES_DIR / f"03_confusion_{safe}.png", dpi=160)
        plt.close(fig)

    # Metrics CSV
    pd.DataFrame(summary["metrics"]["test"]).to_csv(
        FIGURES_DIR / "model_comparison.csv", index=False
    )

    # Subtype recall from test predictions
    preds = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    anom = preds[preds["label"] == 1].copy()
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
        data=subtype_recall.reset_index(),
        x="recall",
        y="subtype",
        ax=ax,
        color="#0f7f78",
    )
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Test anomaly recall by subtype ({summary['best_model']})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_subtype_recall.png", dpi=160)
    plt.close(fig)

    # SHAP on best tree model
    best = summary["best_model"]
    tree_name = best if best in ("Random Forest", "XGBoost") else "XGBoost"
    pipeline = models[tree_name]
    sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    imputer = pipeline.named_steps["imputer"]
    transformed = imputer.transform(sample)
    feature_names = list(imputer.get_feature_names_out(feature_cols))
    sample_df = pd.DataFrame(transformed, columns=feature_names, index=sample.index)
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    explanation = explainer(sample_df)
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]

    plt.figure()
    shap.summary_plot(explanation.values, sample_df, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(explanation.values).mean(axis=0)
    importance = (
        pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.iloc[::-1].plot(kind="barh", ax=ax, color="#0f7f78")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"Global feature importance ({tree_name})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_shap_mean_abs.png", dpi=160)
    plt.close(fig)

    # Local waterfalls
    preds["error_type"] = np.select(
        [
            (preds.label == 0) & (preds.prediction == 1),
            (preds.label == 1) & (preds.prediction == 0),
        ],
        ["False positive", "False negative"],
        default="Correct",
    )
    id_to_index = pd.Series(
        data.loc[masks["test"]].index.values, index=data.loc[masks["test"], "ID"].values
    )

    picks = []
    correct_anom = preds[(preds.label == 1) & (preds.error_type == "Correct")]
    missed = preds[preds.error_type == "False negative"]
    correct_norm = preds[(preds.label == 0) & (preds.error_type == "Correct")]
    if len(correct_anom):
        picks.append(
            (
                "correct_anomaly",
                correct_anom.sort_values("anomaly_probability", ascending=False).iloc[0],
            )
        )
    if len(missed):
        picks.append(("false_negative", missed.sort_values("anomaly_probability").iloc[0]))
    if len(correct_norm):
        picks.append(
            ("correct_normal", correct_norm.sort_values("anomaly_probability").iloc[0])
        )

    for tag, row in picks:
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
        plt.savefig(
            FIGURES_DIR / f"07_shap_waterfall_{tag}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close()

    # Write a short captions file for the report
    captions = FIGURES_DIR / "FIGURE_CAPTIONS.md"
    captions.write_text(
        "\n".join(
            [
                "# Report figure captions",
                "",
                "1. `01_class_imbalance.png` — Normal vs Anomaly counts (~87.5% / 12.5%).",
                "2. `02_numeric_boxplots.png` — Handshake size, entropy, and duration by class.",
                "3. `03_confusion_*.png` — Test-set confusion matrices for each model.",
                "4. `04_subtype_recall.png` — Anomaly recall by scenario subtype (RF).",
                "5. `05_shap_summary.png` — Global SHAP beeswarm on a test sample.",
                "6. `06_shap_mean_abs.png` — Top features by mean |SHAP|.",
                "7. `07_shap_waterfall_*.png` — Local explanations for correct anomaly, false negative, and correct normal.",
                "",
                f"Best model (validation F1): **{summary['best_model']}**.",
                "Main false-negative subtypes: `hybrid_pki`, `high_loss`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Wrote figures to", FIGURES_DIR)
    for p in sorted(FIGURES_DIR.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
