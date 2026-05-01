import { useQuery } from "@tanstack/react-query";
import { fetchAreaSummary, BoundingBox, AreaSummary } from "@/lib/api";

export function useAreaSummary(bbox: BoundingBox | null) {
  return useQuery<AreaSummary>({
    queryKey: ["areaSummary", bbox],
    queryFn: () => {
      if (!bbox) throw new Error("Bounding box is required");
      return fetchAreaSummary(bbox);
    },
    enabled: !!bbox,
  });
}
