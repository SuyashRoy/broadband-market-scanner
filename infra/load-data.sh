#!/usr/bin/env bash
# Broadband Market Scanner — Load parquet data into PostGIS
# Usage: docker compose exec api bash /infra/load-data.sh
#   or:  bash infra/load-data.sh  (from project root, requires psql + Python)

set -euo pipefail

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-fiber_forecast}"
DB_USER="${POSTGRES_USER:-broadband}"
DB_PASS="${POSTGRES_PASSWORD:-broadband_secret}"

DATA_DIR="${DATA_DIR:-./data}"

FEATURE_MATRIX="${DATA_DIR}/intermediate/feature_matrix_full.parquet"
PREDICTIONS="${DATA_DIR}/models/cb_predictions.parquet"
CENTROIDS="${DATA_DIR}/reference/2020_Gaz_tracts_national.txt"

export PGPASSWORD="$DB_PASS"

echo "=== Broadband Market Scanner — Data Loader ==="
echo "Database: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Check required files
for f in "$FEATURE_MATRIX" "$CENTROIDS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

echo ""
echo "Loading data with Python helper..."

python3 - <<'PYEOF'
import os
import pandas as pd
import json

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "fiber_forecast")
DB_USER = os.environ.get("POSTGRES_USER", "broadband")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "broadband_secret")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

connstr = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Load feature matrix (latest period only)
print("Loading feature matrix...")
fm = pd.read_parquet(f"{DATA_DIR}/intermediate/feature_matrix_full.parquet")
fm = fm[fm["filing_period"] == "2025-06"].copy()
print(f"  Feature matrix: {len(fm)} rows")

# Load tract centroids
print("Loading tract centroids...")
centroids = pd.read_csv(f"{DATA_DIR}/reference/2020_Gaz_tracts_national.txt", sep="\t", dtype=str)
centroids.columns = centroids.columns.str.strip()
centroids = centroids[["GEOID", "INTPTLAT", "INTPTLONG"]].copy()
centroids.rename(columns={"GEOID": "tract_fips", "INTPTLAT": "lat", "INTPTLONG": "lng"}, inplace=True)
centroids["lat"] = pd.to_numeric(centroids["lat"], errors="coerce")
centroids["lng"] = pd.to_numeric(centroids["lng"], errors="coerce")

# Merge centroids with feature matrix
fm["tract_fips"] = fm["block_geoid"].str[:11]
fm = fm.merge(centroids, on="tract_fips", how="inner")
print(f"  After centroid merge: {len(fm)} rows")

# Load predictions if available
pred_path = f"{DATA_DIR}/models/cb_predictions.parquet"
if os.path.exists(pred_path):
    preds = pd.read_parquet(pred_path)
    print(f"  Predictions: {len(preds)} rows")
else:
    preds = pd.DataFrame()
    print("  No predictions file found, skipping")

# Write to database using sqlalchemy
try:
    from sqlalchemy import create_engine, text
    engine = create_engine(connstr)

    # Truncate existing data
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE analytics.predictions"))
        conn.execute(text("TRUNCATE TABLE geo.census_blocks CASCADE"))

    # Insert census blocks (with point geometry from centroids)
    print("Inserting census blocks...")
    # Prepare insert data
    insert_cols = [c for c in fm.columns if c not in ("tract_fips", "filing_period")]
    fm_insert = fm[insert_cols].copy()
    fm_insert.to_sql("census_blocks", engine, schema="geo", if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"  Inserted {len(fm_insert)} census blocks")

    # Insert predictions
    if not preds.empty:
        print("Inserting predictions...")
        preds.to_sql("predictions", engine, schema="analytics", if_exists="append", index=False, method="multi", chunksize=5000)
        print(f"  Inserted {len(preds)} predictions")

    # Refresh materialized view
    print("Refreshing materialized views...")
    with engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW analytics.tract_summary"))

    print("")
    print("=== Data load complete ===")

    # Print summary
    with engine.connect() as conn:
        blocks = conn.execute(text("SELECT COUNT(*) FROM geo.census_blocks")).scalar()
        fiber = conn.execute(text("SELECT COUNT(*) FROM geo.census_blocks WHERE has_fiber")).scalar()
        pred_count = conn.execute(text("SELECT COUNT(*) FROM analytics.predictions")).scalar()
        tracts = conn.execute(text("SELECT COUNT(*) FROM analytics.tract_summary")).scalar()

    print(f"  Census blocks: {blocks}")
    print(f"  Fiber blocks:  {fiber}")
    print(f"  Predictions:   {pred_count}")
    print(f"  Tract summaries: {tracts}")

except ImportError:
    print("ERROR: sqlalchemy not installed. Install with: pip install sqlalchemy psycopg2-binary")
    exit(1)
PYEOF

echo ""
echo "Done."
