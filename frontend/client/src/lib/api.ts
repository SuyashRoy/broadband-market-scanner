import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_FRONTEND_FORGE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/* Add API key if available */
if (import.meta.env.VITE_FRONTEND_FORGE_API_KEY) {
  apiClient.defaults.headers.common["Authorization"] = `Bearer ${import.meta.env.VITE_FRONTEND_FORGE_API_KEY}`;
}

export interface AreaData {
  cb_fips: string;
  has_fiber: boolean;
  mhi: number;
  housing_density: number;
  pop_density: number;
  households: number;
  total_population: number;
  fiber_provider_count: number;
  total_provider_count: number;
  fiber_probability: number;
  fiber_forecast_label: "High" | "Medium" | "Low";
  top_contributing_features: string[];
  geometry: GeoJSON.Polygon;
}

export interface AreaSummary {
  total_census_blocks: number;
  fiber_census_blocks: number;
  fiber_coverage_pct: number;
  total_households: number;
  fiber_household_pct: number;
  weighted_avg_mhi: number;
  total_population: number;
  avg_housing_density: number;
  forecast: {
    high_likelihood: number;
    medium_likelihood: number;
    low_likelihood: number;
  };
}

export interface Provider {
  name: string;
  cbs_served: number;
  cbs_fiber: number;
  coverage_pct: number;
  fiber_coverage_pct: number;
}

export interface BoundingBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

/* Fetch census block data within bounding box */
export async function fetchAreaData(bbox: BoundingBox) {
  const response = await apiClient.get("/api/v1/area", {
    params: {
      west: bbox.west,
      south: bbox.south,
      east: bbox.east,
      north: bbox.north,
    },
  });
  return response.data;
}

/* Fetch area summary statistics */
export async function fetchAreaSummary(bbox: BoundingBox): Promise<AreaSummary> {
  const response = await apiClient.get("/api/v1/area/summary", {
    params: {
      west: bbox.west,
      south: bbox.south,
      east: bbox.east,
      north: bbox.north,
    },
  });
  return response.data;
}

/* Fetch provider list */
export async function fetchProviders(bbox: BoundingBox): Promise<Provider[]> {
  const response = await apiClient.get("/api/v1/providers", {
    params: {
      west: bbox.west,
      south: bbox.south,
      east: bbox.east,
      north: bbox.north,
    },
  });
  return response.data;
}

/* Fetch forecast data */
export async function fetchForecast(bbox: BoundingBox) {
  const response = await apiClient.get("/api/v1/forecast", {
    params: {
      west: bbox.west,
      south: bbox.south,
      east: bbox.east,
      north: bbox.north,
    },
  });
  return response.data;
}

/* Geocode address using Mapbox Geocoding API */
export async function geocodeAddress(query: string) {
  const mapboxToken = import.meta.env.VITE_MAPBOX_TOKEN;
  if (!mapboxToken) {
    throw new Error("Mapbox token not configured");
  }

  const response = await axios.get(
    `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json`,
    {
      params: {
        access_token: mapboxToken,
        limit: 10,
      },
    }
  );

  return response.data.features;
}

export default apiClient;
