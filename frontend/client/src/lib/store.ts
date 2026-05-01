import { create } from "zustand";
import { BoundingBox } from "./api";

export type MapLayer = "fiberCoverage" | "fiberForecast" | "mhi" | "providerDensity";

interface DashboardStore {
  /* Map state */
  activeLayer: MapLayer;
  setActiveLayer: (layer: MapLayer) => void;

  /* Selected census block */
  selectedBlockFips: string | null;
  setSelectedBlockFips: (fips: string | null) => void;

  /* Bounding box for current view */
  currentBbox: BoundingBox | null;
  setCurrentBbox: (bbox: BoundingBox) => void;

  /* Location info */
  locationName: string;
  setLocationName: (name: string) => void;
}

export const useDashboardStore = create<DashboardStore>((set) => ({
  activeLayer: "fiberCoverage",
  setActiveLayer: (layer) => set({ activeLayer: layer }),

  selectedBlockFips: null,
  setSelectedBlockFips: (fips) => set({ selectedBlockFips: fips }),

  currentBbox: null,
  setCurrentBbox: (bbox) => set({ currentBbox: bbox }),

  locationName: "",
  setLocationName: (name) => set({ locationName: name }),
}));
