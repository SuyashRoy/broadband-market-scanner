import pandas as pd


def filter_by_bbox(
    df: pd.DataFrame, west: float, south: float, east: float, north: float
) -> pd.DataFrame:
    """Filter DataFrame rows where (lat, lng) centroid falls within bbox."""
    return df[
        (df["lng"] >= west)
        & (df["lng"] <= east)
        & (df["lat"] >= south)
        & (df["lat"] <= north)
    ]
