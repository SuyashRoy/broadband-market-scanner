#!/usr/bin/env bash
# Broadband Market Scanner — Pre-warm Redis cache
# Sends requests for top metro area bounding boxes to populate cache
# Usage: bash infra/seed-cache.sh [API_URL]

set -euo pipefail

API_URL="${1:-http://localhost:8000}"

echo "=== Pre-warming Redis cache ==="
echo "API: ${API_URL}"
echo ""

# Top metro bounding boxes (west, south, east, north)
declare -A METROS
METROS["Austin, TX"]="-97.95,30.10,-97.55,30.52"
METROS["Houston, TX"]="-95.80,29.52,-95.05,30.12"
METROS["Dallas, TX"]="-97.05,32.60,-96.50,33.05"
METROS["San Antonio, TX"]="-98.70,29.30,-98.30,29.60"
METROS["New York, NY"]="-74.10,40.60,-73.75,40.90"
METROS["Los Angeles, CA"]="-118.70,33.70,-117.80,34.20"
METROS["Chicago, IL"]="-87.90,41.65,-87.50,42.05"
METROS["Atlanta, GA"]="-84.60,33.60,-84.20,33.90"
METROS["San Francisco, CA"]="-122.55,37.65,-122.30,37.85"

for metro in "${!METROS[@]}"; do
    IFS=',' read -r west south east north <<< "${METROS[$metro]}"
    echo -n "  ${metro}... "

    for endpoint in "area/summary" "providers" "forecast"; do
        curl -sf "${API_URL}/api/v1/${endpoint}?west=${west}&south=${south}&east=${east}&north=${north}" > /dev/null 2>&1 || true
    done

    echo "done"
done

echo ""
echo "Cache pre-warming complete."
