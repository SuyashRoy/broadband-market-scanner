import { BoundingBox } from "./api";

/* Compute bounding box from map bounds */
export function computeBboxFromBounds(
  bounds: { _sw: { lng: number; lat: number }; _ne: { lng: number; lat: number } }
): BoundingBox {
  return {
    west: bounds._sw.lng,
    south: bounds._sw.lat,
    east: bounds._ne.lng,
    north: bounds._ne.lat,
  };
}

/* Compute zoom level from bounding box */
export function computeZoomFromBbox(bbox: BoundingBox): number {
  const width = bbox.east - bbox.west;
  const height = bbox.north - bbox.south;
  const maxDim = Math.max(width, height);

  /* Approximate zoom level based on bounding box size */
  if (maxDim > 10) return 8;
  if (maxDim > 5) return 9;
  if (maxDim > 2) return 10;
  if (maxDim > 1) return 11;
  if (maxDim > 0.5) return 12;
  if (maxDim > 0.25) return 13;
  return 14;
}

/* Compute bounding box center */
export function computeBboxCenter(bbox: BoundingBox) {
  return {
    lat: (bbox.south + bbox.north) / 2,
    lng: (bbox.west + bbox.east) / 2,
  };
}
