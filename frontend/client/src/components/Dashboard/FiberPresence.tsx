import { AreaSummary } from "@/lib/api";
import { Badge } from "@/components/common/Badge";
import { SkeletonLoader } from "@/components/common/SkeletonLoader";

interface FiberPresenceProps {
  summary: AreaSummary | undefined;
  isLoading?: boolean;
}

export function FiberPresence({ summary, isLoading = false }: FiberPresenceProps) {
  if (isLoading || !summary) {
    return (
      <div className="card-subtle p-6 space-y-4">
        <h3 className="text-lg font-semibold text-foreground">Is Fiber Available Here?</h3>
        <SkeletonLoader count={2} className="h-12" />
      </div>
    );
  }

  const hasFiber = summary.fiber_coverage_pct > 0;

  return (
    <div className="card-subtle p-6 space-y-4">
      <h3 className="text-lg font-semibold text-foreground">Is Fiber Available Here?</h3>

      {hasFiber ? (
        <div className="space-y-3">
          <Badge variant="success">
            Yes — {summary.fiber_census_blocks} providers offer fiber
          </Badge>
          <p className="text-sm text-foreground">
            Fiber is available in <span className="font-semibold">{(summary.fiber_coverage_pct * 100).toFixed(1)}%</span> of this area.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          <Badge variant="error">No — fiber is not yet available</Badge>
          <p className="text-sm text-foreground">
            Forecast: <span className="font-semibold">Check the forecast chart above</span> for likelihood of fiber deployment in the next 12 months.
          </p>
        </div>
      )}
    </div>
  );
}
