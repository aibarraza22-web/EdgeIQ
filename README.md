# EdgeIQ MVP (Real Data)

EdgeIQ is an MVP backend for AI-assisted sports betting analytics. It uses **real-time bookmaker odds** from The Odds API and computes value picks with confidence scoring.

## What this MVP includes
- FastAPI backend with production-ready structure.
- Real odds ingestion for NFL + NBA (expandable to MLB/NHL).
- Daily picks endpoint with:
  - moneyline, spread, and totals markets
  - best sportsbook line selection
  - implied probability vs model probability
  - edge percentage and conservative confidence score
- Responsible gambling and no-guarantee positioning.

## Quick start
1. Create a virtualenv and install:
   ```bash
   pip install -e .[dev]
   ```
2. Copy environment file:
   ```bash
   cp .env.example .env
   ```
3. Add your `ODDS_API_KEY` in `.env`.
4. Run API:
   ```bash
   uvicorn edgeiq.main:app --reload
   ```
5. Open docs: `http://127.0.0.1:8000/docs`

## Core endpoint
- `GET /api/v1/picks/daily`

Returns the top value picks from live odds. If API key is missing, endpoint returns a 503 with a clear configuration message.

## How to add more data/APIs
You asked how to add APIs and publish this for real users. Practical path:

1. **Add a stats API** (SportsRadar, ESPN, etc.)
   - Create a new client module in `src/edgeiq/` (e.g. `stats_client.py`).
   - Ingest team/player form, injuries, weather, and pace/efficiency features.
   - Merge those features into `estimate_model_probability`.
2. **Add player props**
   - Expand `ODDS_MARKETS` to include player markets supported by your odds provider.
   - Add a `props.py` model scorer using player usage, matchup defense, injuries, and minutes projections.
3. **Store history + evaluate**
   - Add PostgreSQL with tables for events, predictions, closes, outcomes.
   - Train/calibrate models weekly and monitor by sport + market + confidence bucket.
4. **Publish and monetize**
   - Deploy API to Render/Railway/AWS.
   - Add frontend + auth + Stripe subscriptions.
   - Gate premium endpoints by JWT plan claims.

## Responsible gambling
EdgeIQ provides analytics only, not betting execution. Never guarantee wins. Always include help resources like 1-800-GAMBLER and NCPG.
