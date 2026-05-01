import { AreaSummary } from "@/lib/api";
import { MetricCard } from "@/components/common/MetricCard";
import { ProgressBar } from "@/components/common/ProgressBar";
import { formatPercentage, formatNumber } from "@/lib/formatters";
import { SkeletonLoader } from "@/components/common/SkeletonLoader";

interface FiberCoverageProps {
  summary: AreaSummary | undefined;
  isLoading?: boolean;
}

export function FiberCoverage({ summary, isLoading = false }: FiberCoverageProps) {
  if (isLoading || !summary) {
    return (
      <div className="card-subtle p-6 space-y-6">
        <h3 className="text-lg font-semibold text-foreground">Fiber Coverage</h3>
        <SkeletonLoader count={4} className="h-16" />
      </div>
    );
  }

  const blocksCoverage = summary.fiber_coverage_pct;
  const householdsCoverage = summary.fiber_household_pct;

  return (
    <div className="card-subtle p-6 space-y-6">
      <h3 className="text-lg font-semibold text-foreground">Fiber Coverage</h3>

      <div className="grid grid-cols-2 gap-6">
        {/* By Census Blocks */}
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground font-medium">By Census Blocks</p>
          <p className="metric-value text-3xl fiber-positive">{formatPercentage(blocksCoverage, 1)}</p>
          <p className="text-xs text-muted-foreground">
            {formatNumber(summary.fiber_census_blocks)} of {formatNumber(summary.total_census_blocks)} blocks
          </p>
          <ProgressBar value={blocksCoverage} color="bg-accent" />
        </div>

        {/* By Households */}
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground font-medium">By Households</p>
          <p className="metric-value text-3xl fiber-positive">{formatPercentage(householdsCoverage, 1)}</p>
          <p className="text-xs text-muted-foreground">
            {formatNumber(Math.round(summary.total_households * householdsCoverage))} of {formatNumber(summary.total_households)} households
          </p>
          <ProgressBar value={householdsCoverage} color="bg-accent" />
        </div>
      </div>
    </div>
  );
}
