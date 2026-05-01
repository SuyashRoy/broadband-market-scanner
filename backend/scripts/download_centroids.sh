#!/usr/bin/env bash
#
# Download Census 2020 Tract Centroid Gazetteer file.
# No block-level Gazetteer exists, so we use tract centroids for bbox filtering.
# Each block inherits its parent tract's centroid — accurate enough for dashboard use.
#
# Source: https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST_DIR="$PROJECT_ROOT/data/reference"

URL="https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_tracts_national.zip"
ZIP_FILE="$DEST_DIR/2020_Gaz_tracts_national.zip"
TXT_FILE="$DEST_DIR/2020_Gaz_tracts_national.txt"

mkdir -p "$DEST_DIR"

if [ -f "$TXT_FILE" ]; then
    echo "Centroid file already exists at $TXT_FILE"
    echo "To re-download, delete it first."
    exit 0
fi

echo "Downloading Census 2020 Tract Gazetteer (centroids)..."
curl -L --fail -o "$ZIP_FILE" "$URL"

echo "Unzipping..."
unzip -o "$ZIP_FILE" -d "$DEST_DIR"

# Clean up zip
rm -f "$ZIP_FILE"

echo "Done! Centroid file available at: $TXT_FILE"
wc -l "$TXT_FILE"
