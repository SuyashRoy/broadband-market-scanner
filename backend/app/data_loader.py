import logging
from pathlib import Path

import pandas as pd

from app.config import (
    CENTROID_PATH,
    FEATURE_MATRIX_PATH,
    LATEST_PERIOD,
    PREDICTIONS_PATH,
    TARGET_STATE_FIPS,
)

logger = logging.getLogger(__name__)


def load_tract_centroids(path: Path) -> pd.DataFrame:
    """Load Census Tract Gazetteer file and return tract_fips → (lat, lng).

    The 2020 Gazetteer only goes down to tract level; no block-level file
    exists.  Each block inherits its parent tract's centroid, which is
    accurate enough for bounding-box filtering at dashboard zoom levels.
    """
    logger.info("Loading tract centroids from %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    # Strip whitespace from column names (Gazetteer files have trailing spaces)
    df.columns = df.columns.str.strip()
    df = df[["GEOID", "INTPTLAT", "INTPTLONG"]].copy()
    # Filter to target states
    df = df[df["GEOID"].str[:2].isin(TARGET_STATE_FIPS)].copy()
    df.rename(
        columns={"GEOID": "tract_fips", "INTPTLAT": "lat", "INTPTLONG": "lng"},
        inplace=True,
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df.dropna(subset=["lat", "lng"], inplace=True)
    logger.info("Loaded %d tract centroids for target states", len(df))
    return df


def load_feature_matrix(path: Path, period: str) -> pd.DataFrame:
    """Load feature matrix parquet, filtered to the latest filing period."""
    logger.info("Loading feature matrix from %s (period=%s)", path, period)
    df = pd.read_parquet(path)
    df = df[df["filing_period"] == period].copy()
    logger.info("Feature matrix: %d rows for period %s", len(df), period)
    return df


def load_predictions(path: Path) -> pd.DataFrame:
    """Load model predictions parquet."""
    logger.info("Loading predictions from %s", path)
    if not path.exists():
        logger.warning("Predictions file not found at %s, returning empty DataFrame", path)
        return pd.DataFrame(
            columns=["cb_fips", "fiber_probability", "fiber_forecast_label", "top_contributing_features"]
        )
    df = pd.read_parquet(path)
    logger.info("Predictions: %d rows", len(df))
    return df


def build_dataset() -> pd.DataFrame:
    """Build the full dataset: feature matrix + predictions + tract centroids."""
    # Load feature matrix for latest period
    features = load_feature_matrix(FEATURE_MATRIX_PATH, LATEST_PERIOD)

    # Load predictions and left-join
    predictions = load_predictions(PREDICTIONS_PATH)
    if not predictions.empty:
        features = features.merge(
            predictions,
            left_on="block_geoid",
            right_on="cb_fips",
            how="left",
        )
    else:
        features["cb_fips"] = features["block_geoid"]
        features["fiber_probability"] = None
        features["fiber_forecast_label"] = None
        features["top_contributing_features"] = None

    # Derive tract FIPS from block GEOID (first 11 chars of 15-char block GEOID)
    features["tract_fips"] = features["block_geoid"].str[:11]

    # Load tract centroids and merge
    if CENTROID_PATH.exists():
        centroids = load_tract_centroids(CENTROID_PATH)
        before = len(features)
        features = features.merge(centroids, on="tract_fips", how="inner")
        logger.info("After centroid merge: %d rows (dropped %d with no centroid)", len(features), before - len(features))
    else:
        logger.warning(
            "Centroid file not found at %s. Spatial filtering will not work. "
            "Run backend/scripts/download_centroids.sh to download it.",
            CENTROID_PATH,
        )
        features["lat"] = None
        features["lng"] = None

    return features
