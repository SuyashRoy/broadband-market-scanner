#!/usr/bin/env bash
#
# Data Refresh Pipeline
# Orchestrates: run notebooks → rebuild features → retrain model → restart API
#
# Usage: ./refresh_data.sh [--notebooks-only | --api-only]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NOTEBOOK_DIR="$PROJECT_ROOT/pipeline/notebooks"

echo "========================================"
echo "Broadband Market Scanner — Data Refresh"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Step 1: Ensure centroid reference data exists
echo "[1/4] Checking reference data..."
if [ ! -f "$PROJECT_ROOT/data/reference/2020_Gaz_tracts_national.txt" ]; then
    echo "  Downloading Census tract centroids..."
    bash "$SCRIPT_DIR/download_centroids.sh"
else
    echo "  Centroid file present."
fi

# Step 2: Run pipeline notebooks (if not --api-only)
if [ "${1:-}" != "--api-only" ]; then
    echo ""
    echo "[2/4] Running pipeline notebooks..."
    if command -v jupyter &> /dev/null || command -v papermill &> /dev/null; then
        for nb in "$NOTEBOOK_DIR"/*.ipynb; do
            name=$(basename "$nb")
            echo "  Running $name..."
            if command -v papermill &> /dev/null; then
                papermill "$nb" "$nb" --no-progress-bar 2>&1 | tail -3
            else
                jupyter nbconvert --to notebook --execute "$nb" --inplace 2>&1 | tail -3
            fi
        done
    else
        echo "  WARNING: jupyter/papermill not found. Run notebooks manually."
        echo "  Skipping notebook execution."
    fi
else
    echo ""
    echo "[2/4] Skipping notebooks (--api-only mode)"
fi

# Step 3: Verify outputs
echo ""
echo "[3/4] Verifying data outputs..."
for f in \
    "data/intermediate/feature_matrix_full.parquet" \
    "data/intermediate/training_data.parquet" \
    "data/models/fiber_forecast_model.joblib" \
    "data/models/cb_predictions.parquet"; do
    path="$PROJECT_ROOT/$f"
    if [ -f "$path" ]; then
        size=$(du -h "$path" | cut -f1)
        echo "  OK: $f ($size)"
    else
        echo "  MISSING: $f"
    fi
done

# Step 4: Restart API (if running)
echo ""
echo "[4/4] Data refresh complete."
echo "  Restart the API server to pick up new data:"
echo "  cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000"
echo ""
echo "Done!"
