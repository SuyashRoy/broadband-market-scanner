from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MODELS_DIR = DATA_DIR / "models"
REFERENCE_DIR = DATA_DIR / "reference"

FEATURE_MATRIX_PATH = INTERMEDIATE_DIR / "feature_matrix_full.parquet"
PREDICTIONS_PATH = MODELS_DIR / "cb_predictions.parquet"
CENTROID_PATH = REFERENCE_DIR / "2020_Gaz_tracts_national.txt"

LATEST_PERIOD = "2025-06"

# FIPS state codes for the 5 target states
TARGET_STATE_FIPS = {"06", "13", "17", "36", "48"}

# Provider column name → display name mapping
PROVIDER_DISPLAY_NAMES = {
    "att": "AT&T",
    "charter": "Charter/Spectrum",
    "comcast": "Comcast/Xfinity",
    "verizon": "Verizon",
    "frontier": "Frontier",
    "cox": "Cox",
    "google_fiber": "Google Fiber",
    "altice": "Altice/Optimum",
    "windstream": "Windstream",
    "lumen": "Lumen/CenturyLink",
}

MAX_AREA_RESULTS = 5000
