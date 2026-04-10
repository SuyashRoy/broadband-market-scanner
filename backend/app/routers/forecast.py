import numpy as np
from fastapi import APIRouter, Query, Request

from app.config import LATEST_PERIOD
from app.services.spatial import filter_by_bbox

router = APIRouter()


@router.get("/forecast")
def get_forecast(
    request: Request,
    west: float = Query(...),
    south: float = Query(...),
    east: float = Query(...),
    north: float = Query(...),
):
    df = request.app.state.dataset
    subset = filter_by_bbox(df, west, south, east, north)

    # Filter to unserved blocks
    unserved = subset[subset["has_fiber"] == 0]
    total_unserved = len(unserved)

    if total_unserved == 0:
        return {
            "total_unserved": 0,
            "high_likelihood": 0,
            "medium_likelihood": 0,
            "low_likelihood": 0,
            "predictions": [],
            "data_as_of": LATEST_PERIOD,
        }

    # Count by forecast label
    label_col = "fiber_forecast_label"
    if label_col in unserved.columns:
        label_counts = unserved[label_col].value_counts()
        high = int(label_counts.get("High", 0))
        medium = int(label_counts.get("Medium", 0))
        low = int(label_counts.get("Low", 0))
    else:
        high = medium = low = 0

    # Build per-block predictions (vectorized, limit to 500)
    pred_subset = unserved.head(500)
    result = pred_subset[
        ["block_geoid", "fiber_probability", "fiber_forecast_label", "top_contributing_features"]
    ].copy()
    result.rename(columns={"block_geoid": "cb_fips"}, inplace=True)

    # Round probability
    result["fiber_probability"] = result["fiber_probability"].apply(
        lambda v: round(float(v), 4) if v is not None and np.isfinite(v) else None
    )
    result["fiber_forecast_label"] = result["fiber_forecast_label"].where(
        result["fiber_forecast_label"].notna(), None
    )
    result["top_contributing_features"] = result["top_contributing_features"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    return {
        "total_unserved": total_unserved,
        "high_likelihood": high,
        "medium_likelihood": medium,
        "low_likelihood": low,
        "predictions": result.to_dict(orient="records"),
        "data_as_of": LATEST_PERIOD,
    }
