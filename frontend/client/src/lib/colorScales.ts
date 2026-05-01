/* Color scales for different map layers */

/* Fiber Coverage: Green for has fiber, Gray for no fiber */
export const fiberCoverageColors = {
  hasFiber: "#14b8a6", /* Teal */
  noFiber: "#cbd5e1", /* Slate gray */
};

/* Fiber Forecast: Red -> Yellow -> Green gradient for probability, Blue for already has fiber */
export function getForecastColor(probability: number, hasFiber: boolean): string {
  if (hasFiber) {
    return "#3b82f6"; /* Blue for blocks with fiber */
  }

  /* Red -> Yellow -> Green gradient based on probability (0-1) */
  if (probability < 0.33) {
    return "#ef4444"; /* Red - Low likelihood */
  } else if (probability < 0.67) {
    return "#fbbf24"; /* Amber - Medium likelihood */
  } else {
    return "#22c55e"; /* Green - High likelihood */
  }
}

/* Median Household Income: Blue (low) -> Yellow -> Red (high) */
export function getMhiColor(mhi: number, minMhi: number, maxMhi: number): string {
  const normalized = (mhi - minMhi) / (maxMhi - minMhi);

  if (normalized < 0.33) {
    return "#3b82f6"; /* Blue - Low MHI */
  } else if (normalized < 0.67) {
    return "#fbbf24"; /* Yellow - Medium MHI */
  } else {
    return "#ef4444"; /* Red - High MHI */
  }
}

/* Provider Density: Light to Dark gradient */
export function getProviderDensityColor(count: number, maxCount: number): string {
  const normalized = count / maxCount;

  if (normalized < 0.25) {
    return "#e0e7ff"; /* Very light */
  } else if (normalized < 0.5) {
    return "#a5b4fc"; /* Light */
  } else if (normalized < 0.75) {
    return "#6366f1"; /* Medium */
  } else {
    return "#312e81"; /* Dark */
  }
}

/* Legend definitions for each layer */
export const layerLegends = {
  fiberCoverage: [
    { label: "Has Fiber", color: fiberCoverageColors.hasFiber },
    { label: "No Fiber", color: fiberCoverageColors.noFiber },
  ],
  fiberForecast: [
    { label: "Has Fiber", color: "#3b82f6" },
    { label: "High Likelihood", color: "#22c55e" },
    { label: "Medium Likelihood", color: "#fbbf24" },
    { label: "Low Likelihood", color: "#ef4444" },
  ],
  mhi: [
    { label: "Low MHI", color: "#3b82f6" },
    { label: "Medium MHI", color: "#fbbf24" },
    { label: "High MHI", color: "#ef4444" },
  ],
  providerDensity: [
    { label: "1 Provider", color: "#e0e7ff" },
    { label: "2 Providers", color: "#a5b4fc" },
    { label: "3 Providers", color: "#6366f1" },
    { label: "4+ Providers", color: "#312e81" },
  ],
};
