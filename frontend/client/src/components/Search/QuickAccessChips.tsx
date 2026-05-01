import { GeocodingResult } from "@/hooks/useGeocoding";

interface QuickAccessChip {
  label: string;
  query: string;
}

interface QuickAccessChipsProps {
  onSelect: (result: GeocodingResult) => void;
}

const QUICK_ACCESS_CHIPS: QuickAccessChip[] = [
  { label: "Los Angeles, CA", query: "Los Angeles, California" },
  { label: "Austin, TX", query: "Austin, Texas" },
  { label: "ZIP: 30301", query: "30301" },
  { label: "San Francisco, CA", query: "San Francisco, California" },
];

export function QuickAccessChips({ onSelect }: QuickAccessChipsProps) {
  const handleChipClick = async (chip: QuickAccessChip) => {
    /* Create a mock geocoding result */
    const result: GeocodingResult = {
      id: chip.query,
      type: "place",
      place_name: chip.label,
      center: [0, 0], /* Will be overridden by actual geocoding */
      geometry: {
        type: "Point",
        coordinates: [0, 0],
      },
    };
    onSelect(result);
  };

  return (
    <div className="flex flex-wrap gap-3 justify-center">
      {QUICK_ACCESS_CHIPS.map((chip) => (
        <button
          key={chip.query}
          onClick={() => handleChipClick(chip)}
          className="px-4 py-2 text-sm font-medium text-foreground bg-muted hover:bg-secondary border border-border rounded-full transition-smooth"
        >
          {chip.label}
        </button>
      ))}
    </div>
  );
}
