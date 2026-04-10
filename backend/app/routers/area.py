import math

import numpy as np
from fastapi import APIRouter, Query, Request

from app.config import LATEST_PERIOD, MAX_AREA_RESULTS
from app.services.spatial import filter_by_bbox

router = APIRouter()


@router.get("/area/summary")
def get_area_summary(
    request: Request,
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
):
    df = request.app.state.dataset
    subset = filter_by_bbox(df, west, south, east, north)

    total_blocks = len(subset)
    if total_blocks == 0:
        return {
            "total_census_blocks": 0,
            "fiber_census_blocks": 0,
            "fiber_coverage_pct": 0.0,
            "total_households": 0,
            "fiber_household_pct": 0.0,
            "weighted_avg_mhi": 0.0,
            "total_population": 0,
            "avg_housing_density": 0.0,
            "forecast": {
                "high_likelihood": 0,
                "medium_likelihood": 0,
                "low_likelihood": 0,
            },
            "data_as_of": LATEST_PERIOD,
        }

    fiber_blocks = int(subset["has_fiber"].sum())
    fiber_coverage_pct = round(fiber_blocks / total_blocks * 100, 2)

    total_hh = int(subset["occupied_housing_units"].sum())
    fiber_hh = int(subset.loc[subset["has_fiber"] == 1, "occupied_housing_units"].sum())
    fiber_hh_pct = round(fiber_hh / total_hh * 100, 2) if total_hh else 0.0

    mhi_col = "mhi_2024"
    valid_mhi = subset.dropna(subset=[mhi_col])
    if len(valid_mhi) > 0 and valid_mhi["occupied_housing_units"].sum() > 0:
        weighted_mhi = round(
            float(
                (valid_mhi[mhi_col] * valid_mhi["occupied_housing_units"]).sum()
                / valid_mhi["occupied_housing_units"].sum()
            ),
            0,
        )
    else:
        weighted_mhi = 0.0

    total_pop = int(subset["total_population"].sum())

    hd_mean = subset["housing_density"].mean()
    avg_hd = round(float(hd_mean), 1) if np.isfinite(hd_mean) else 0.0

    forecast_col = "fiber_forecast_label"
    if forecast_col in subset.columns:
        label_counts = subset[forecast_col].value_counts()
        high = int(label_counts.get("High", 0))
        medium = int(label_counts.get("Medium", 0))
        low = int(label_counts.get("Low", 0))
    else:
        high = medium = low = 0

    return {
        "total_census_blocks": total_blocks,
        "fiber_census_blocks": fiber_blocks,
        "fiber_coverage_pct": fiber_coverage_pct,
        "total_households": total_hh,
        "fiber_household_pct": fiber_hh_pct,
        "weighted_avg_mhi": weighted_mhi,
        "total_population": total_pop,
        "avg_housing_density": avg_hd,
        "forecast": {
            "high_likelihood": high,
            "medium_likelihood": medium,
            "low_likelihood": low,
        },
        "data_as_of": LATEST_PERIOD,
    }


@router.get("/area")
def get_area_data(
    request: Request,
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
):
    df = request.app.state.dataset
    subset = filter_by_bbox(df, west, south, east, north)

    if len(subset) > MAX_AREA_RESULTS:
        subset = subset.head(MAX_AREA_RESULTS)

    if len(subset) == 0:
        return []

    # Vectorized: build result using DataFrame operations instead of iterrows
    result = subset[
        [
            "block_geoid",
            "has_fiber",
            "mhi_2024",
            "housing_density",
            "pop_density",
            "occupied_housing_units",
            "total_population",
            "fiber_provider_count",
            "total_provider_count",
            "fiber_probability",
            "fiber_forecast_label",
            "top_contributing_features",
        ]
    ].copy()

    result.rename(
        columns={
            "block_geoid": "cb_fips",
            "mhi_2024": "mhi",
            "occupied_housing_units": "households",
        },
        inplace=True,
    )

    result["has_fiber"] = result["has_fiber"].astype(bool)
    result["housing_density"] = result["housing_density"].round(1)
    result["pop_density"] = result["pop_density"].round(1)
    result["households"] = result["households"].fillna(0).astype(int)
    result["total_population"] = result["total_population"].fillna(0).astype(int)
    result["fiber_provider_count"] = result["fiber_provider_count"].fillna(0).astype(int)
    result["total_provider_count"] = result["total_provider_count"].fillna(0).astype(int)
    result["geometry"] = None

    # Replace NaN/None in top_contributing_features with empty list
    result["top_contributing_features"] = result["top_contributing_features"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    # Convert to records and sanitize NaN → None for JSON
    records = result.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rec[k] = None
    return records
