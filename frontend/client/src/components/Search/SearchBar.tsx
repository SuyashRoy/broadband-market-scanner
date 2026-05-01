import { useState, useRef, useEffect } from "react";
import { useGeocoding, GeocodingResult } from "@/hooks/useGeocoding";
import { Search, Loader2 } from "lucide-react";

interface SearchBarProps {
  onSelect: (result: GeocodingResult) => void;
  placeholder?: string;
  compact?: boolean;
}

export function SearchBar({
  onSelect,
  placeholder = "Search by address, city, zip code, or county...",
  compact = false,
}: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { data: suggestions, isLoading } = useGeocoding(query);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (inputRef.current && !inputRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (result: GeocodingResult) => {
    setQuery(result.place_name);
    setShowSuggestions(false);
    onSelect(result);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && suggestions && suggestions.length > 0) {
      handleSelect(suggestions[0]);
    }
  };

  return (
    <div className="relative w-full">
      <div className={`relative flex items-center gap-3 ${compact ? "px-4 py-2" : "px-6 py-4"} bg-white dark:bg-card border border-border rounded-lg shadow-sm`}>
        <Search className="w-5 h-5 text-muted-foreground flex-shrink-0" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setShowSuggestions(true);
          }}
          onFocus={() => setShowSuggestions(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className={`flex-1 bg-transparent outline-none text-foreground placeholder-muted-foreground ${compact ? "text-sm" : "text-base"}`}
        />
        {isLoading && <Loader2 className="w-5 h-5 text-muted-foreground animate-spin flex-shrink-0" />}
      </div>

      {showSuggestions && suggestions && suggestions.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-card border border-border rounded-lg shadow-lg z-50">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion.id}
              onClick={() => handleSelect(suggestion)}
              className="w-full px-6 py-3 text-left hover:bg-muted dark:hover:bg-secondary transition-smooth border-b border-border last:border-b-0"
            >
              <p className="text-sm font-medium text-foreground">{suggestion.place_name}</p>
              <p className="text-xs text-muted-foreground">{suggestion.type}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
