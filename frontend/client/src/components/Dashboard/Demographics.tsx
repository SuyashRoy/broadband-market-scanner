import { AreaSummary } from "@/lib/api";
import { MetricCard } from "@/components/common/MetricCard";
import { formatCurrency, formatNumber, formatDensity } from "@/lib/formatters";
import { SkeletonLoader } from "@/components/common/SkeletonLoader";

interface DemographicsProps {
  summary: AreaSummary | undefined;
  isLoading?: boolean;
}

export function Demographics({ summary, isLoading = false }: DemographicsProps) {
  if (isLoading || !summary) {
    return (
      <div className="card-subtle p-6 space-y-6">
        <h3 className="text-lg font-semibold text-foreground">Area Demographics</h3>
        <SkeletonLoader count={4} className="h-20" />
      </div>
    );
  }

  return (
    <div className="card-subtle p-6 space-y-6">
      <h3 className="text-lg font-semibold text-foreground">Area Demographics</h3>

      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          label="Median Household Income"
          value={formatCurrency(summary.weighted_avg_mhi)}
        />
        <MetricCard
          label="Total Population"
          value={formatNumber(summary.total_population)}
        />
        <MetricCard
          label="Total Households"
          value={formatNumber(summary.total_households)}
        />
        <MetricCard
          label="Avg. Housing Density"
          value={formatDensity(summary.avg_housing_density)}
        />
      </div>
    </div>
  );
}
