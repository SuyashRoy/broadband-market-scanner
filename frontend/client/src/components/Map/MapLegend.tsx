import { MapLayer, useDashboardStore } from "@/lib/store";
import { layerLegends } from "@/lib/colorScales";

export function MapLegend() {
  const { activeLayer } = useDashboardStore();

  const legend = layerLegends[activeLayer];

  if (!legend) return null;

  return (
    <div className="absolute bottom-4 left-4 bg-white dark:bg-card border border-border rounded-lg shadow-lg p-4 z-10">
      <div className="space-y-2">
        {legend.map((item) => (
          <div key={item.label} className="flex items-center gap-3">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-sm text-foreground">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
