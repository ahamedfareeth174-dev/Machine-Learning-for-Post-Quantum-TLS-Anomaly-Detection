from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
ML_FEATURES_CSV = DATASET_DIR / "ml_features_and_labels.csv"
SCENARIO_MANIFEST_CSV = DATASET_DIR / "scenario_manifest.csv"

META_COLS = {"ID", "label", "split", "taxonomy"}
RANDOM_STATE = 42
