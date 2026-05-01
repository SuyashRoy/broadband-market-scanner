import { useLocation } from "wouter";
import { SearchBar } from "@/components/Search/SearchBar";
import { QuickAccessChips } from "@/components/Search/QuickAccessChips";
import { GeocodingResult } from "@/hooks/useGeocoding";

export default function SearchPage() {
  const [, setLocation] = useLocation();

  const handleSearch = (result: GeocodingResult) => {
    const [lng, lat] = result.center;
    const bbox = result.bbox;

    /* Compute bounding box if available, otherwise use a default zoom level */
    let params = `lat=${lat}&lng=${lng}&name=${encodeURIComponent(result.place_name)}`;

    if (bbox) {
      const [minX, minY, maxX, maxY] = bbox;
      params += `&west=${minX}&south=${minY}&east=${maxX}&north=${maxY}`;
    } else {
      /* Default zoom level 12 */
      const delta = 0.1;
      params += `&west=${lng - delta}&south=${lat - delta}&east=${lng + delta}&north=${lat + delta}`;
    }

    setLocation(`/dashboard?${params}`);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background px-4 py-8">
      {/* Main content */}
      <div className="w-full max-w-2xl space-y-8">
        {/* Header */}
        <div className="text-center space-y-3">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground">Broadband Market Scanner</h1>
          <p className="text-base md:text-lg text-muted-foreground">
            Hyperlocal fiber coverage analytics and deployment forecasts.
          </p>
        </div>

        {/* Search bar */}
        <div className="w-full">
          <SearchBar onSelect={handleSearch} />
        </div>

        {/* Quick access chips */}
        <div className="w-full">
          <p className="text-xs md:text-sm text-muted-foreground text-center mb-4">Or try an example:</p>
          <QuickAccessChips onSelect={handleSearch} />
        </div>
      </div>
    </div>
  );
}
