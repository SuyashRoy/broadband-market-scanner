import { MapLayer, useDashboardStore } from "@/lib/store";
import { Layers } from "lucide-react";

const LAYERS: { id: MapLayer; label: string }[] = [
  { id: "fiberCoverage", label: "Fiber Coverage" },
  { id: "fiberForecast", label: "Fiber Forecast" },
  { id: "mhi", label: "Median Household Income" },
  { id: "providerDensity", label: "Provider Density" },
];

export function LayerToggle() {
  const { activeLayer, setActiveLayer } = useDashboardStore();

  return (
    <div className="absolute top-4 right-4 bg-white dark:bg-card border border-border rounded-lg shadow-lg p-2 z-10">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
        <Layers className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">Layers</span>
      </div>

      <div className="space-y-1 p-2">
        {LAYERS.map((layer) => (
          <button
            key={layer.id}
            onClick={() => setActiveLayer(layer.id)}
            className={`w-full px-3 py-2 text-sm text-left rounded transition-smooth ${
              activeLayer === layer.id
                ? "bg-accent text-accent-foreground font-medium"
                : "text-foreground hover:bg-muted"
            }`}
          >
            {layer.label}
          </button>
        ))}
      </div>
    </div>
  );
}
