-- Broadband Market Scanner — Database Initialization
-- Runs automatically on first PostGIS container startup

-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS geo;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Census block features table (loaded from feature_matrix_full.parquet)
CREATE TABLE IF NOT EXISTS geo.census_blocks (
    block_geoid VARCHAR(15) PRIMARY KEY,
    filing_period VARCHAR(10),
    state_fips VARCHAR(2),
    has_fiber BOOLEAN,
    mhi_2024 NUMERIC,
    housing_density NUMERIC,
    pop_density NUMERIC,
    occupied_housing_units INTEGER,
    total_population INTEGER,
    fiber_provider_count INTEGER,
    total_provider_count INTEGER,
    hhi_broadband NUMERIC,
    hhi_fiber NUMERIC,
    cbg_fiber_penetration NUMERIC,
    neighbor_fiber_pct NUMERIC,
    -- Provider presence flags
    att_present BOOLEAN DEFAULT FALSE,
    att_fiber BOOLEAN DEFAULT FALSE,
    charter_present BOOLEAN DEFAULT FALSE,
    charter_fiber BOOLEAN DEFAULT FALSE,
    comcast_present BOOLEAN DEFAULT FALSE,
    comcast_fiber BOOLEAN DEFAULT FALSE,
    verizon_present BOOLEAN DEFAULT FALSE,
    verizon_fiber BOOLEAN DEFAULT FALSE,
    frontier_present BOOLEAN DEFAULT FALSE,
    frontier_fiber BOOLEAN DEFAULT FALSE,
    cox_present BOOLEAN DEFAULT FALSE,
    cox_fiber BOOLEAN DEFAULT FALSE,
    google_fiber_present BOOLEAN DEFAULT FALSE,
    google_fiber_fiber BOOLEAN DEFAULT FALSE,
    altice_present BOOLEAN DEFAULT FALSE,
    altice_fiber BOOLEAN DEFAULT FALSE,
    windstream_present BOOLEAN DEFAULT FALSE,
    windstream_fiber BOOLEAN DEFAULT FALSE,
    lumen_present BOOLEAN DEFAULT FALSE,
    lumen_fiber BOOLEAN DEFAULT FALSE,
    -- Spatial
    geom GEOMETRY(Point, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model predictions table (loaded from cb_predictions.parquet)
CREATE TABLE IF NOT EXISTS analytics.predictions (
    cb_fips VARCHAR(15) PRIMARY KEY REFERENCES geo.census_blocks(block_geoid),
    fiber_probability NUMERIC,
    fiber_forecast_label VARCHAR(10),
    top_contributing_features JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Spatial index on census blocks
CREATE INDEX IF NOT EXISTS idx_census_blocks_geom ON geo.census_blocks USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_census_blocks_state ON geo.census_blocks (state_fips);
CREATE INDEX IF NOT EXISTS idx_census_blocks_fiber ON geo.census_blocks (has_fiber);

-- Materialized view for fast area summaries (refresh after data load)
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.tract_summary AS
SELECT
    LEFT(block_geoid, 11) AS tract_fips,
    COUNT(*) AS total_blocks,
    COUNT(*) FILTER (WHERE has_fiber) AS fiber_blocks,
    ROUND(COUNT(*) FILTER (WHERE has_fiber)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) AS fiber_coverage_pct,
    SUM(occupied_housing_units) AS total_households,
    SUM(total_population) AS total_population,
    AVG(mhi_2024) AS avg_mhi,
    AVG(housing_density) AS avg_housing_density
FROM geo.census_blocks
GROUP BY LEFT(block_geoid, 11);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tract_summary_fips ON analytics.tract_summary (tract_fips);
