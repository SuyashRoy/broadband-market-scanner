import { useQuery } from "@tanstack/react-query";
import { fetchProviders, BoundingBox, Provider } from "@/lib/api";

export function useProviders(bbox: BoundingBox | null) {
  return useQuery<Provider[]>({
    queryKey: ["providers", bbox],
    queryFn: () => {
      if (!bbox) throw new Error("Bounding box is required");
      return fetchProviders(bbox);
    },
    enabled: !!bbox,
  });
}
