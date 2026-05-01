import { AreaSummary } from "@/lib/api";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";
import { formatNumber, formatPercentage } from "@/lib/formatters";
import { SkeletonLoader } from "@/components/common/SkeletonLoader";

interface ForecastSummaryProps {
  summary: AreaSummary | undefined;
  isLoading?: boolean;
}

export function ForecastSummary({ summary, isLoading = false }: ForecastSummaryProps) {
  if (isLoading || !summary) {
    return (
      <div className="card-subtle p-6 space-y-6">
        <h3 className="text-lg font-semibold text-foreground">Fiber Deployment Forecast</h3>
        <SkeletonLoader count={2} className="h-32" />
      </div>
    );
  }

  const { forecast } = summary;
  const total = forecast.high_likelihood + forecast.medium_likelihood + forecast.low_likelihood;

  const data = [
    { name: "High Likelihood", value: forecast.high_likelihood, color: "#22c55e" },
    { name: "Medium Likelihood", value: forecast.medium_likelihood, color: "#fbbf24" },
    { name: "Low Likelihood", value: forecast.low_likelihood, color: "#ef4444" },
  ];

  return (
    <div className="card-subtle p-6 space-y-6">
      <h3 className="text-lg font-semibold text-foreground">Fiber Deployment Forecast</h3>

      {total > 0 ? (
        <>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {data.map((entry) => (
                    <Cell key={`cell-${entry.name}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: "0.5rem",
                  }}
                  formatter={(value: number) => formatNumber(value)}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Legend and stats */}
          <div className="space-y-2">
            {data.map((item) => (
              <div key={item.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <span className="text-foreground">{item.name}</span>
                </div>
                <span className="font-semibold text-foreground">
                  {formatNumber(item.value)} ({formatPercentage(item.value / total, 0)})
                </span>
              </div>
            ))}
          </div>

          {/* Callout */}
          <div className="bg-muted p-4 rounded-lg border border-border">
            <p className="text-sm text-foreground">
              <span className="font-semibold">{formatNumber(forecast.high_likelihood)} blocks</span> expected to gain fiber in the next 12 months
            </p>
          </div>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">No unserved blocks to forecast</p>
      )}
    </div>
  );
}
