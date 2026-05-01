# Broadband Market Scanner

Hyperlocal fiber deployment forecasting platform. Combines FCC Broadband Data Collection (BDC) provider-level fiber availability at the census block level with ACS demographic data to produce interactive maps showing current fiber coverage, demographics, provider analytics, and ML-driven forecasts of future fiber builds.

## Architecture

```
broadband-market-scanner/
├── pipeline/notebooks/         # Jupyter data pipeline (Phases 1-5)
│   ├── 00_build_2025_fcc_master.ipynb
│   ├── 03_provider_features.ipynb
│   ├── 04_feature_matrix_and_labels.ipynb
│   └── 05_model_training.ipynb
├── backend/                    # FastAPI backend (Phases 6-8)
│   ├── app/
│   │   ├── main.py             # App entry, CORS, lifespan, monitoring
│   │   ├── config.py           # Settings, provider names, paths
│   │   ├── data_loader.py      # Load parquets + Census centroids
│   │   ├── routers/            # API endpoints
│   │   │   ├── area.py         # /area + /area/summary
│   │   │   ├── providers.py    # /providers
│   │   │   └── forecast.py     # /forecast
│   │   └── services/
│   │       └── spatial.py      # Bbox filtering
│   ├── scripts/
│   │   ├── download_centroids.sh
│   │   └── refresh_data.sh
│   └── requirements.txt
├── frontend/                   # React + Mapbox GL dashboard (Phase 7)
│   ├── client/src/             # React application
│   ├── server/                 # Express production server
│   └── shared/                 # Shared constants
├── data/                       # (gitignored)
│   ├── intermediate/           # Parquet outputs from pipeline
│   ├── models/                 # Trained model + predictions
│   └── reference/              # Census Gazetteer centroids
├── docker-compose.yml          # Serving stack (PostGIS, Redis, API, frontend)
├── docker-compose.pipeline.yml # Data pipeline (Jupyter)
├── infra/                      # Infrastructure scripts
│   ├── init-db.sql             # PostGIS schema + table DDL
│   ├── load-data.sh            # Load parquets into PostGIS
│   └── seed-cache.sh           # Pre-warm Redis cache
├── Makefile                    # Convenience commands
└── .env.example                # Environment variable template
```

## Coverage

5 U.S. states: **California (06), Georgia (13), Illinois (17), New York (36), Texas (48)**

1,033,822 census blocks with 50 features per block, covering 4 FCC filing periods (2022-06 through 2025-06).

## Data Pipeline (Phases 1-5)

The pipeline runs as Jupyter notebooks in sequence:

| Phase | Notebook | Output | Description |
|-------|----------|--------|-------------|
| 1-2 | `00_build_2025_fcc_master.ipynb` | FCC master dataset | Load FCC BDC data, build demographics |
| 3 | `03_provider_features.ipynb` | `cb_provider_features.parquet` (3.5M rows, 37 cols) | Provider presence, fiber flags, HHI, competition metrics, spatial neighbor features |
| 4 | `04_feature_matrix_and_labels.ipynb` | `feature_matrix_full.parquet` (3.5M rows, 54 cols), `training_data.parquet` (92K rows) | Merge demographics + provider features, create `gained_fiber` training labels via 12-month comparison windows |
| 5 | `05_model_training.ipynb` | `fiber_forecast_model.joblib`, `cb_predictions.parquet` | XGBoost with temporal CV, SHAP analysis, batch scoring of unserved blocks |

**Key features engineered (50 total):**
- Per-provider presence and fiber flags (AT&T, Charter, Comcast, Verizon, Frontier, Cox, Google Fiber, Altice, Windstream, Lumen)
- Competition metrics: HHI (broadband + fiber), major provider counts, fiber competition flag
- Spatial features: CBG fiber penetration, neighbor fiber percentage, neighbor count
- Demographics: housing density, population density, MHI, occupancy rate, household growth rate

## Backend API (Phases 6-8)

FastAPI backend serving data from in-memory pandas DataFrames with Census tract centroid-based spatial filtering.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check + row count + data freshness |
| `GET` | `/api/v1/area/summary?west=&south=&east=&north=` | Aggregated coverage stats for bounding box |
| `GET` | `/api/v1/area?west=&south=&east=&north=` | Per-block data (max 5,000 blocks) |
| `GET` | `/api/v1/providers?west=&south=&east=&north=` | Provider breakdown by blocks served |
| `GET` | `/api/v1/forecast?west=&south=&east=&north=` | Forecast distribution for unserved blocks |

### Features (Phase 8)
- Response time monitoring middleware with `X-Response-Time-Ms` header
- Slow query detection (>2s logged as WARNING)
- `data_as_of` field in summary, forecast, and health responses
- Vectorized DataFrame operations (no `iterrows()` in hot paths)
- NaN-safe JSON serialization
- Data refresh pipeline script

**Performance:** API responses typically <50ms for metro-level bounding box queries.

## Frontend (Phase 7)

React 19 + TypeScript + Vite 7 dashboard with:
- **Mapbox GL** interactive map with fly-to navigation
- **Search bar** with Mapbox Geocoding API
- **Data panel**: coverage summary, provider table, forecast chart, demographics
- **shadcn/ui** component library with Tailwind CSS 4
- **React Query** for viewport-driven data fetching
- **Zustand** for state management

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 20+ with pnpm (enable via `corepack enable`)
- Mapbox API token ([get one here](https://account.mapbox.com/))

### Setup

```bash
# Clone and enter project
cd broadband-market-scanner

# One-command setup
make setup

# Or manually:
cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/download_centroids.sh
cd ../frontend && pnpm install
```

### Configure Environment

```bash
cp .env.example frontend/.env.local
# Edit frontend/.env.local and add your Mapbox token:
# VITE_MAPBOX_TOKEN=pk.your_token_here
```

### Run

```bash
# Terminal 1: Backend API
make backend
# or: cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend dev server
make frontend
# or: cd frontend && pnpm dev
```

- Backend: http://localhost:8000 (API docs at http://localhost:8000/docs)
- Frontend: http://localhost:3000

### Test

```bash
make test
# Tests health endpoint and area summary for Austin, TX
```

## Docker Deployment (Phase 9)

### Prerequisites
- Docker and Docker Compose
- Mapbox API token

### Quick Start (Docker)

```bash
# Copy environment template and configure
cp .env.example .env
# Edit .env — set VITE_MAPBOX_TOKEN and database credentials

# Start the full serving stack
docker compose up -d

# Check services
docker compose ps

# View API logs
docker compose logs api
```

Services:
- **Frontend**: http://localhost (nginx + React SPA)
- **Backend API**: http://localhost:8000 (FastAPI, also proxied via nginx at `/api/`)
- **API Docs**: http://localhost:8000/docs
- **PostGIS**: localhost:5432
- **Redis**: localhost:6379

### Data Pipeline (Docker)

```bash
# Run Jupyter for notebook execution
docker compose -f docker-compose.pipeline.yml up

# Open http://localhost:8888 and run notebooks 01–05 in order
# Outputs land in ./data/intermediate/ and ./data/models/

# After notebooks complete, load data into PostGIS
bash infra/load-data.sh

# Optionally pre-warm Redis cache
bash infra/seed-cache.sh
```

### Architecture (Docker)

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `db` | `postgis/postgis:16-3.4` | 5432 | PostgreSQL + PostGIS spatial database |
| `redis` | `redis:7-alpine` | 6379 | Query result caching |
| `api` | Custom (Python 3.11-slim) | 8000 | FastAPI backend |
| `frontend` | Custom (Node build + nginx) | 80 | React SPA with API proxy |
| `jupyter` | Custom (Python 3.11 + GDAL) | 8888 | Data pipeline notebooks |

## Model Notes

The current training data (92,014 samples across 3 comparison windows) shows **0 positive labels** — no census blocks transitioned from unserved to fiber-served across filing periods. This is because the FCC BDC dataset for these 5 states shows universal fiber coverage by the latest period (2025-06). The model and pipeline infrastructure are fully functional and will produce meaningful predictions when:

1. Additional states with lower fiber penetration are added
2. New FCC filing periods reveal fiber expansion in currently unserved areas
3. The data source is expanded to include non-fiber broadband for comparative analysis

## Roadmap

- [x] Phase 1-2: Data preparation (FCC + demographics)
- [x] Phase 3: Provider feature engineering
- [x] Phase 4: Feature matrix + training labels
- [x] Phase 5: XGBoost model training + SHAP analysis
- [x] Phase 6: Backend API (FastAPI + spatial filtering)
- [x] Phase 7: Frontend dashboard (React + Mapbox GL)
- [x] Phase 8: Optimization, monitoring, data freshness
- [x] Phase 9: Dockerization (PostGIS, Redis, Docker Compose)
- [ ] Phase 10: CI/CD pipeline (GitHub Actions)
