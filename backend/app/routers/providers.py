from fastapi import APIRouter, Query, Request

from app.config import PROVIDER_DISPLAY_NAMES
from app.services.spatial import filter_by_bbox

router = APIRouter()


@router.get("/providers")
def get_providers(
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
        return []

    results = []
    for key, display_name in PROVIDER_DISPLAY_NAMES.items():
        present_col = f"{key}_present"
        fiber_col = f"{key}_fiber"

        if present_col not in subset.columns:
            continue

        cbs_served = int(subset[present_col].sum())
        cbs_fiber = int(subset[fiber_col].sum()) if fiber_col in subset.columns else 0

        if cbs_served == 0:
            continue

        coverage_pct = round(cbs_served / total_blocks * 100, 1)
        fiber_coverage_pct = round(cbs_fiber / total_blocks * 100, 1)

        results.append(
            {
                "name": display_name,
                "cbs_served": cbs_served,
                "cbs_fiber": cbs_fiber,
                "coverage_pct": coverage_pct,
                "fiber_coverage_pct": fiber_coverage_pct,
            }
        )

    # Sort by cbs_served descending
    results.sort(key=lambda x: x["cbs_served"], reverse=True)
    return results
