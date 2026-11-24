"""Train classifiers and export artifacts used by the demo API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import feature_columns, load_manifest, load_ml_frame, split_xy
from .models import build_models, choose_threshold, metric_row
from .paths import ARTIFACTS_DIR, RANDOM_STATE


FEATURE_HELP = {
    "e2c_total_bytes": "Total handshake bytes (client↔server).",
    "e4_entropy_h": "Shannon entropy of Client/Server Hello headers.",
    "e5_entropy_c": "Shannon entropy of encrypted ciphertext (first 4KB).",
    "e6_time_char": "Timing characteristic of the handshake.",
    "e6b_flow_duration_ms": "Handshake duration in milliseconds.",
    "e2_client_size": "Client key-share size (proxy for KEM group).",
    "e2_client_record_len": "Client Hello record length.",
    "e2_server_record_len": "Server Hello record length (cert + KEM CT).",
    "e3_cert_parsed": "Whether a certificate was successfully parsed.",
}


def _safe_float(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def export_demo_sessions(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    feature_cols: list[str],
    n_per_class: int = 40,
) -> list[dict]:
    test = data[data["split"].eq("test")].copy()
    joined = test.merge(
        manifest.drop(columns=["ID"]),
        left_on="ID",
        right_on="join_id",
        how="left",
        validate="one_to_one",
    )
    samples = []
    for label, tag in ((0, "normal"), (1, "anomaly")):
        pool = joined[joined["label"].eq(label)]
        take = min(n_per_class, len(pool))
        picked = pool.sample(n=take, random_state=RANDOM_STATE)
        for _, row in picked.iterrows():
            features = {c: _safe_float(row[c]) for c in feature_cols}
            samples.append(
                {
                    "id": row["ID"],
                    "label": int(row["label"]),
                    "label_name": "Normal" if row["label"] == 0 else "Anomaly",
                    "scenario_id": row.get("scenario_id"),
                    "taxonomy_class": row.get("taxonomy_class"),
                    "subtype": row.get("subtype"),
                    "operational_class": row.get("operational_class"),
                    "tag": tag,
                    "features": features,
                }
            )
    return samples


def main() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_ml_frame()
    manifest = load_manifest()
    feature_cols, X, y, masks = split_xy(data)

    X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
    X_val, y_val = X.loc[masks["val"]], y.loc[masks["val"]]
    X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

    models = build_models(y_train)
    thresholds: dict[str, float] = {}
    metrics = {"validation": [], "test": []}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        thresholds[name] = choose_threshold(y_val, val_prob)
        metrics["validation"].append(
            metric_row(name, y_val, val_prob, thresholds[name])
        )
        test_prob = model.predict_proba(X_test)[:, 1]
        metrics["test"].append(
            metric_row(name, y_test, test_prob, thresholds[name])
        )
        print(
            f"  val F1={metrics['validation'][-1]['f1']:.4f} "
            f"test F1={metrics['test'][-1]['f1']:.4f} "
            f"threshold={thresholds[name]:.4f}"
        )

    best_name = max(metrics["validation"], key=lambda r: r["f1"])["model"]
    print(f"Best model by validation F1: {best_name}")

    # Persist pipelines
    for name, model in models.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, ARTIFACTS_DIR / f"model_{safe}.joblib")

    feature_meta = []
    for col in feature_cols:
        series = data[col]
        feature_meta.append(
            {
                "name": col,
                "dtype": "bool" if series.dropna().isin([0, 1, True, False]).all()
                and series.nunique(dropna=True) <= 2
                else "numeric",
                "help": FEATURE_HELP.get(col, "Engineered TLS / PQC handshake feature."),
                "min": _safe_float(series.min(skipna=True)),
                "max": _safe_float(series.max(skipna=True)),
                "median": _safe_float(series.median(skipna=True)),
                "missing_rate": float(series.isna().mean()),
            }
        )

    demo_sessions = export_demo_sessions(data, manifest, feature_cols)

    # Default form values: median normal train session profile
    normal_train = data[masks["train"] & data["label"].eq(0)]
    defaults = {
        c: _safe_float(normal_train[c].median(skipna=True)) for c in feature_cols
    }

    # Class balance summary
    summary = {
        "n_sessions": int(len(data)),
        "n_normal": int((data["label"] == 0).sum()),
        "n_anomaly": int((data["label"] == 1).sum()),
        "n_features": len(feature_cols),
        "splits": {
            s: {
                "n": int(masks[s].sum()),
                "anomaly": int(((masks[s]) & (data["label"] == 1)).sum()),
            }
            for s in ("train", "val", "test")
        },
        "best_model": best_name,
        "models": list(models.keys()),
        "thresholds": thresholds,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "feature_meta": feature_meta,
        "defaults": defaults,
        "citation": (
            "Michael O. Mills and Ali A. Ghorbani, "
            '"Beyond Rules: Behavioral Anomaly Detection for Post-Quantum '
            'Cryptography Operational Assurance in TLS," '
            "Journal of Information Security and Applications (under Review), 2025."
        ),
        "dataset_url": "https://www.unb.ca/cic/datasets/pqc-oav-dataset-2025.html",
    }

    (ARTIFACTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "demo_sessions.json").write_text(
        json.dumps(demo_sessions, indent=2), encoding="utf-8"
    )

    # Save a compact test prediction table for the notebook / report
    best_model = models[best_name]
    test_ids = data.loc[masks["test"], ["ID", "label"]].copy()
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= thresholds[best_name]).astype(int)
    test_ids["prediction"] = test_pred
    test_ids["anomaly_probability"] = test_prob
    test_ids = test_ids.merge(
        manifest.drop(columns=["ID"]),
        left_on="ID",
        right_on="join_id",
        how="left",
        validate="one_to_one",
    )
    test_ids.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    print(f"Artifacts written to {ARTIFACTS_DIR}")
    return ARTIFACTS_DIR


if __name__ == "__main__":
    main()
