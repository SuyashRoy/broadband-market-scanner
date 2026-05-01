import { Provider } from "@/lib/api";
import { ProgressBar } from "@/components/common/ProgressBar";
import { formatPercentage } from "@/lib/formatters";
import { SkeletonLoader } from "@/components/common/SkeletonLoader";
import { useState } from "react";

interface ProviderTableProps {
  providers: Provider[] | undefined;
  isLoading?: boolean;
}

type SortField = "coverage" | "fiber";

export function ProviderTable({ providers, isLoading = false }: ProviderTableProps) {
  const [sortBy, setSortBy] = useState<SortField>("coverage");

  if (isLoading || !providers) {
    return (
      <div className="card-subtle p-6 space-y-6">
        <h3 className="text-lg font-semibold text-foreground">Broadband Providers</h3>
        <SkeletonLoader count={4} className="h-12" />
      </div>
    );
  }

  const sorted = [...providers].sort((a, b) => {
    if (sortBy === "coverage") {
      return b.coverage_pct - a.coverage_pct;
    } else {
      return b.fiber_coverage_pct - a.fiber_coverage_pct;
    }
  });

  return (
    <div className="card-subtle p-6 space-y-4">
      <h3 className="text-lg font-semibold text-foreground">Broadband Providers</h3>

      {providers.length === 0 ? (
        <p className="text-sm text-muted-foreground">No providers available for this area</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3 font-semibold text-foreground">Provider</th>
                <th
                  className="text-left py-2 px-3 font-semibold text-foreground cursor-pointer hover:text-accent transition-smooth"
                  onClick={() => setSortBy("coverage")}
                >
                  Coverage
                </th>
                <th
                  className="text-left py-2 px-3 font-semibold text-foreground cursor-pointer hover:text-accent transition-smooth"
                  onClick={() => setSortBy("fiber")}
                >
                  Fiber
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((provider) => {
                const hasFiberFocus = provider.fiber_coverage_pct > 0.5;
                return (
                  <tr
                    key={provider.name}
                    className={`border-b border-border hover:bg-muted transition-smooth ${
                      hasFiberFocus ? "bg-green-50 dark:bg-green-950/20" : ""
                    }`}
                  >
                    <td className="py-3 px-3 text-foreground font-medium">{provider.name}</td>
                    <td className="py-3 px-3">
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <ProgressBar value={provider.coverage_pct} color="bg-blue-500" />
                          <span className="text-xs font-semibold text-foreground ml-2">
                            {formatPercentage(provider.coverage_pct, 0)}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <ProgressBar value={provider.fiber_coverage_pct} color="bg-accent" />
                          <span className="text-xs font-semibold fiber-positive ml-2">
                            {formatPercentage(provider.fiber_coverage_pct, 0)}
                          </span>
                        </div>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
