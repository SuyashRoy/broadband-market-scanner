/* Format number with commas */
export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

/* Format currency */
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

/* Format percentage */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/* Format housing density */
export function formatDensity(value: number): string {
  return `${Math.round(value)} units/km²`;
}

/* Format FIPS code */
export function formatFips(fips: string): string {
  return `FIPS: ${fips}`;
}
