import { useQuery } from "@tanstack/react-query";
import { geocodeAddress } from "@/lib/api";

export interface GeocodingResult {
  id: string;
  type: string;
  place_name: string;
  center: [number, number]; /* [lng, lat] */
  bbox?: [number, number, number, number]; /* [minX, minY, maxX, maxY] */
  geometry: {
    type: string;
    coordinates: [number, number];
  };
}

export function useGeocoding(query: string | null) {
  return useQuery<GeocodingResult[]>({
    queryKey: ["geocoding", query],
    queryFn: () => {
      if (!query || query.length < 2) throw new Error("Query too short");
      return geocodeAddress(query);
    },
    enabled: !!query && query.length >= 2,
  });
}
