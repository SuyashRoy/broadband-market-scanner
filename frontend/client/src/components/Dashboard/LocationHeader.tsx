import { formatNumber } from "@/lib/formatters";

interface LocationHeaderProps {
  locationName: string;
  totalBlocks: number;
  totalHouseholds: number;
  totalPopulation: number;
  isLoading?: boolean;
}

export function LocationHeader({
  locationName,
  totalBlocks,
  totalHouseholds,
  totalPopulation,
  isLoading = false,
}: LocationHeaderProps) {
  if (isLoading) {
    return (
      <div className="card-subtle p-6 space-y-3">
        <div className="h-8 bg-muted rounded animate-pulse w-48" />
        <div className="h-4 bg-muted rounded animate-pulse w-64" />
      </div>
    );
  }

  return (
    <div className="card-subtle p-6 space-y-2">
      <h2 className="text-2xl font-bold text-foreground">{locationName}</h2>
      <p className="text-sm text-muted-foreground">
        Covering {formatNumber(totalBlocks)} census blocks • {formatNumber(totalHouseholds)} households • {formatNumber(totalPopulation)} population
      </p>
    </div>
  );
}
