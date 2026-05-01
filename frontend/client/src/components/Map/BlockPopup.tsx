import { AreaData } from "@/lib/api";
import { Badge } from "@/components/common/Badge";
import { formatNumber, formatCurrency, formatDensity, formatPercentage } from "@/lib/formatters";
import { X } from "lucide-react";

interface BlockPopupProps {
  block: AreaData;
  onClose: () => void;
}

export function BlockPopup({ block, onClose }: BlockPopupProps) {
  return (
    <div className="absolute inset-0 flex items-center justify-center z-50 bg-black/50 p-4">
      <div className="bg-white dark:bg-card border border-border rounded-lg shadow-lg max-w-md w-full p-6 space-y-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-foreground">Census Block Details</h3>
            <p className="text-sm text-muted-foreground">FIPS: {block.cb_fips}</p>
          </div>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-smooth"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Fiber Status */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-foreground">Fiber Status</p>
          {block.has_fiber ? (
            <Badge variant="success">Yes — {block.fiber_provider_count} providers offer fiber</Badge>
          ) : (
            <Badge variant="error">No fiber available</Badge>
          )}
        </div>

        {/* Demographics */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-muted-foreground">MHI</p>
            <p className="text-sm font-semibold text-foreground">{formatCurrency(block.mhi)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Population</p>
            <p className="text-sm font-semibold text-foreground">{formatNumber(block.total_population)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Households</p>
            <p className="text-sm font-semibold text-foreground">{formatNumber(block.households)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Housing Density</p>
            <p className="text-sm font-semibold text-foreground">{formatDensity(block.housing_density)}</p>
          </div>
        </div>

        {/* Providers */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-foreground">Providers</p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-muted p-2 rounded">
              <p className="text-xs text-muted-foreground">Total Providers</p>
              <p className="font-semibold text-foreground">{block.total_provider_count}</p>
            </div>
            <div className="bg-muted p-2 rounded">
              <p className="text-xs text-muted-foreground">Fiber Providers</p>
              <p className="font-semibold text-accent">{block.fiber_provider_count}</p>
            </div>
          </div>
        </div>

        {/* Forecast (if no fiber) */}
        {!block.has_fiber && (
          <div className="space-y-2 border-t border-border pt-4">
            <p className="text-sm font-medium text-foreground">Fiber Forecast</p>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Probability</span>
              <span className="text-sm font-semibold text-foreground">{formatPercentage(block.fiber_probability)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Likelihood</span>
              <Badge variant={block.fiber_forecast_label === "High" ? "success" : block.fiber_forecast_label === "Medium" ? "warning" : "error"}>
                {block.fiber_forecast_label}
              </Badge>
            </div>
            {block.top_contributing_features.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-2">Top Contributing Features</p>
                <ul className="text-xs space-y-1">
                  {block.top_contributing_features.slice(0, 3).map((feature, i) => (
                    <li key={i} className="text-foreground">• {feature}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
