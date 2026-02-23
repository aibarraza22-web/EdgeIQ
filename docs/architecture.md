# EdgeIQ MVP Architecture

## Components
- **FastAPI service**: exposes `/api/v1/picks/daily` and `/health`.
- **Odds provider integration**: The Odds API (live bookmaker prices).
- **Prediction layer**: market-consensus-aware expected value heuristic.

## Request flow
1. Client calls `/api/v1/picks/daily`.
2. Service fetches odds for configured sports/markets.
3. For each event:
   - choose best odds per outcome across books,
   - compute implied probability,
   - estimate model probability using cross-book consensus,
   - compute edge and confidence.
4. Return top picks sorted by confidence and edge.

## Next upgrades
- Replace heuristic with trained model (XGBoost/LightGBM).
- Add feature store (injuries, weather, player metrics, situational data).
- Add outcome ingestion and backtesting metrics.
- Add subscriptions, alerts, and admin monitoring.
