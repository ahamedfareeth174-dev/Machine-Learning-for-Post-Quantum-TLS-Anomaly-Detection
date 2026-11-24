# Explainable ML for Post-Quantum TLS Anomaly Detection

Classify hybrid Post-Quantum TLS 1.3 sessions as **Normal** vs **Anomaly**, compare Logistic Regression / Random Forest / XGBoost, explain predictions with SHAP, and explore results in a live demo UI.

**Dataset:** [CIC-PQC_OAV v1](https://www.unb.ca/cic/datasets/pqc-oav-dataset-2025.html) (40,010 sessions).

## Project layout

| Path | Purpose |
|------|---------|
| `Dataset/` | `ml_features_and_labels.csv`, `scenario_manifest.csv`, `raw_sessions.jsonl`, research notebook |
| `Dataset/PQC_TLS.ipynb` | End-to-end EDA → train → evaluate → SHAP notebook |
| `src/` | Shared training / export code used by the demo |
| `artifacts/` | Saved models, metrics, demo sessions (generated) |
| `demo/` | FastAPI backend + frontend analyzer |
| `report/figures/` | Report-ready plots + demo screenshots |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Place the three CIC dataset files under `Dataset/` if they are not already there.

## Train models (required once for the demo)

```bash
python -m src.train_and_export
```

This writes pipelines, thresholds, test metrics, and sample sessions into `artifacts/`.

## Run the demo frontend

```bash
uvicorn demo.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

- Pick a held-out test session (or edit features)
- Classify with LR / RF / XGBoost
- Inspect anomaly probability vs validation-tuned threshold
- View top feature contributions (SHAP for tree models)

## Notebook

Open `Dataset/PQC_TLS.ipynb` for the full coursework pipeline (EDA, metrics, error analysis, SHAP plots). Paths resolve to the local `Dataset/` folder.

## Report figures

```bash
python -m src.export_report_figures
```

Outputs land in `report/figures/` (class imbalance, boxplots, confusion matrices, subtype recall, SHAP plots, captions). Demo UI screenshots are `08_demo_hero.png`, `09_demo_analyzer.png`, and `10_demo_metrics.png`.

## Citation

> Michael O. Mills and Ali A. Ghorbani, "Beyond Rules: Behavioral Anomaly Detection for Post-Quantum Cryptography Operational Assurance in TLS," Journal of Information Security and Applications (under Review), 2025.
