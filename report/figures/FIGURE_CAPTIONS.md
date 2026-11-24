# Report figure captions

1. `01_class_imbalance.png` — Normal vs Anomaly counts (~87.5% / 12.5%).
2. `02_numeric_boxplots.png` — Handshake size, entropy, and duration by class.
3. `03_confusion_*.png` — Test-set confusion matrices for each model.
4. `04_subtype_recall.png` — Anomaly recall by scenario subtype (RF).
5. `05_shap_summary.png` — Global SHAP beeswarm on a test sample.
6. `06_shap_mean_abs.png` — Top features by mean |SHAP|.
7. `07_shap_waterfall_*.png` — Local explanations for correct anomaly, false negative, and correct normal.
8. `08_demo_hero.png` — PQC-TLS Sentinel landing / hero view.
9. `09_demo_analyzer.png` — Live session classification + feature contributions.
10. `10_demo_metrics.png` — Held-out test metrics table in the demo UI.

Best model (validation F1): **Random Forest**.
Main false-negative subtypes: `hybrid_pki`, `high_loss`.
