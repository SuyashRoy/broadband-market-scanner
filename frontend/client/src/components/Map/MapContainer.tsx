import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { BoundingBox } from "@/lib/api";
import { useDashboardStore } from "@/lib/store";

interface MapContainerProps {
  bbox: BoundingBox | null;
  locationName: string;
  onBboxChange?: (bbox: BoundingBox) => void;
}

export function MapContainer({ bbox, locationName, onBboxChange }: MapContainerProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapReady, setMapReady] = useState(false);

  const mapboxToken = import.meta.env.VITE_MAPBOX_TOKEN;

  useEffect(() => {
    if (!mapboxToken) {
      console.error("Mapbox token not configured");
      return;
    }

    mapboxgl.accessToken = mapboxToken;

    if (map.current) return;

    if (!mapContainer.current) return;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/light-v11",
      center: [-95.7129, 37.0902], /* Default center (USA) */
      zoom: 4,
    });

    map.current.on("load", () => {
      setMapReady(true);
      /* Vector tile layers will be added once a PostGIS/tile backend is available */
    });

    map.current.on("moveend", () => {
      if (map.current && onBboxChange) {
        const bounds = map.current.getBounds();
        if (bounds) {
          onBboxChange({
            west: bounds.getWest(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            north: bounds.getNorth(),
          });
        }
      }
    });

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [mapboxToken, onBboxChange]);

  /* Fly to location when bbox changes */
  useEffect(() => {
    if (bbox && map.current && mapReady) {
      const center = {
        lng: (bbox.west + bbox.east) / 2,
        lat: (bbox.south + bbox.north) / 2,
      };

      const width = bbox.east - bbox.west;
      const height = bbox.north - bbox.south;
      const maxDim = Math.max(width, height);

      let zoom = 12;
      if (maxDim > 10) zoom = 8;
      else if (maxDim > 5) zoom = 9;
      else if (maxDim > 2) zoom = 10;
      else if (maxDim > 1) zoom = 11;
      else if (maxDim > 0.5) zoom = 12;

      map.current.flyTo({
        center: [center.lng, center.lat],
        zoom: zoom,
        duration: 1000,
      });
    }
  }, [bbox, mapReady]);

  return (
    <div
      ref={mapContainer}
      className="w-full h-full bg-gray-100 rounded-lg overflow-hidden"
      style={{ minHeight: "500px" }}
    />
  );
}
