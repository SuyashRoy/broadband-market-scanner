import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { MapContainer } from "@/components/Map/MapContainer";
import { LayerToggle } from "@/components/Map/LayerToggle";
import { MapLegend } from "@/components/Map/MapLegend";
import { DataPanel } from "@/components/Dashboard/DataPanel";
import { SearchBar } from "@/components/Search/SearchBar";
import { useAreaSummary } from "@/hooks/useAreaSummary";
import { useProviders } from "@/hooks/useProviders";
import { useDashboardStore } from "@/lib/store";
import { BoundingBox } from "@/lib/api";
import { GeocodingResult } from "@/hooks/useGeocoding";

export default function DashboardPage() {
  const [, setLocation] = useLocation();
  const [bbox, setBbox] = useState<BoundingBox | null>(null);
  const [locationName, setLocationName] = useState("");

  /* Parse URL params */
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const name = params.get("name") || "Unknown Location";
    const west = parseFloat(params.get("west") || "0");
    const south = parseFloat(params.get("south") || "0");
    const east = parseFloat(params.get("east") || "0");
    const north = parseFloat(params.get("north") || "0");

    setLocationName(name);
    if (west && south && east && north) {
      setBbox({ west, south, east, north });
    }
  }, []);

  /* Fetch data */
  const { data: summary, isLoading: isLoadingSummary } = useAreaSummary(bbox);
  const { data: providers, isLoading: isLoadingProviders } = useProviders(bbox);

  const handleSearch = (result: GeocodingResult) => {
    const [lng, lat] = result.center;
    const bbox = result.bbox;

    let params = `lat=${lat}&lng=${lng}&name=${encodeURIComponent(result.place_name)}`;

    if (bbox) {
      const [minX, minY, maxX, maxY] = bbox;
      params += `&west=${minX}&south=${minY}&east=${maxX}&north=${maxY}`;
    } else {
      const delta = 0.1;
      params += `&west=${lng - delta}&south=${lat - delta}&east=${lng + delta}&north=${lat + delta}`;
    }

    setLocation(`/dashboard?${params}`);
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Top bar */}
      <div className="flex flex-col md:flex-row items-start md:items-center gap-2 md:gap-4 px-4 md:px-6 py-3 md:py-4 border-b border-border bg-white dark:bg-card">
        <h1 className="text-base md:text-lg font-bold text-foreground">Broadband Market Scanner</h1>
        <div className="flex-1 w-full md:w-auto md:max-w-md">
          <SearchBar onSelect={handleSearch} compact />
        </div>
        <p className="text-xs text-muted-foreground hidden md:block">Data as of: Q1 2026</p>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row gap-4 overflow-hidden p-4">
        {/* Map section (full width on mobile, 60-65% on desktop) */}
        <div className="flex-1 relative bg-gray-100 rounded-lg overflow-hidden min-h-96 lg:min-h-0">
          {bbox && <MapContainer bbox={bbox} locationName={locationName} onBboxChange={setBbox} />}
          <LayerToggle />
          <MapLegend />
        </div>

        {/* Data panel (full width on mobile, 35-40% on desktop) */}
        <div className="w-full lg:w-96 bg-white dark:bg-card border border-border rounded-lg overflow-hidden flex flex-col max-h-96 lg:max-h-none">
          <div className="flex-1 overflow-y-auto p-4">
            {bbox && (
              <DataPanel
                locationName={locationName}
                summary={summary}
                providers={providers}
                isLoadingSummary={isLoadingSummary}
                isLoadingProviders={isLoadingProviders}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
