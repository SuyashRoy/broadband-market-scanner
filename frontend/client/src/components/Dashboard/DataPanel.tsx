import { AreaSummary, Provider } from "@/lib/api";
import { LocationHeader } from "./LocationHeader";
import { FiberCoverage } from "./FiberCoverage";
import { ForecastSummary } from "./ForecastSummary";
import { Demographics } from "./Demographics";
import { ProviderTable } from "./ProviderTable";
import { FiberPresence } from "./FiberPresence";

interface DataPanelProps {
  locationName: string;
  summary: AreaSummary | undefined;
  providers: Provider[] | undefined;
  isLoadingSummary?: boolean;
  isLoadingProviders?: boolean;
}

export function DataPanel({
  locationName,
  summary,
  providers,
  isLoadingSummary = false,
  isLoadingProviders = false,
}: DataPanelProps) {
  return (
    <div className="space-y-4 overflow-y-auto">
      <LocationHeader
        locationName={locationName}
        totalBlocks={summary?.total_census_blocks || 0}
        totalHouseholds={summary?.total_households || 0}
        totalPopulation={summary?.total_population || 0}
        isLoading={isLoadingSummary}
      />

      <FiberCoverage summary={summary} isLoading={isLoadingSummary} />

      <ForecastSummary summary={summary} isLoading={isLoadingSummary} />

      <Demographics summary={summary} isLoading={isLoadingSummary} />

      <ProviderTable providers={providers} isLoading={isLoadingProviders} />

      <FiberPresence summary={summary} isLoading={isLoadingSummary} />
    </div>
  );
}
