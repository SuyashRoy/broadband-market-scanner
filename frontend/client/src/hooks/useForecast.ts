import { useQuery } from "@tanstack/react-query";
import { fetchForecast, BoundingBox } from "@/lib/api";

export function useForecast(bbox: BoundingBox | null) {
  return useQuery({
    queryKey: ["forecast", bbox],
    queryFn: () => {
      if (!bbox) throw new Error("Bounding box is required");
      return fetchForecast(bbox);
    },
    enabled: !!bbox,
  });
}
