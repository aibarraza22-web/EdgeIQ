"""EdgeIQ FastAPI application.

Endpoints
---------
GET /health
    Liveness probe.

GET /api/v1/picks/daily
    Live picks powered by Monte Carlo simulation + The Odds API.

GET /api/v1/backtest
    Backtest Monte Carlo predictions against recent completed games
    (no ODDS_API_KEY needed — uses ESPN public data only).

GET /api/v1/simulate
    Run a one-off Monte Carlo simulation for a specific matchup.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from edgeiq.backtester import run_backtest
from edgeiq.config import settings
from edgeiq.models import (
    BacktestReportResponse,
    DailyPicksResponse,
    GamePredictionResponse,
    MonteCarloSummary,
    Pick,
)
from edgeiq.odds import fetch_events_with_odds, generate_picks
from edgeiq.simulator import TeamStats, run_monte_carlo

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description=(
        "AI-assisted sports betting analytics. "
        "Monte Carlo simulations + live bookmaker odds. "
        "Analytics only — not financial advice."
    ),
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Daily picks ───────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/picks/daily",
    response_model=DailyPicksResponse,
    tags=["picks"],
    summary="Today's value picks (Monte Carlo + live odds)",
)
async def daily_picks() -> DailyPicksResponse:
    """Return the top value picks from live bookmaker odds.

    Each pick is backed by a 10 000-simulation Monte Carlo model that
    compares team offensive/defensive season averages to produce realistic
    win, spread, and total probability distributions.

    Requires ``ODDS_API_KEY`` in the environment.  Returns 503 when the key
    is missing or the Odds API quota is exhausted.
    """
    try:
        events = await fetch_events_with_odds()
        raw_picks = await generate_picks(events)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    picks: list[Pick] = []
    for p in raw_picks:
        mc_data = p.pop("monte_carlo", None)
        mc_summary = MonteCarloSummary(**mc_data) if mc_data else None
        picks.append(Pick(**p, monte_carlo=mc_summary))

    return DailyPicksResponse(
        generated_at=datetime.now(timezone.utc),
        picks=picks,
    )


# ── Backtesting ───────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/backtest",
    response_model=BacktestReportResponse,
    tags=["analytics"],
    summary="Backtest MC predictions against real completed games",
)
async def backtest_predictions(
    sport: Literal["basketball_nba", "americanfootball_nfl"] = Query(
        default="basketball_nba",
        description="Sport to evaluate",
    ),
    days_back: int = Query(
        default=30,
        ge=1,
        le=180,
        description="Calendar days of history to scan",
    ),
    max_games: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of games to simulate",
    ),
) -> BacktestReportResponse:
    """Run Monte Carlo backtesting on recently completed games.

    For each completed game the model is run **without** the final score
    (using only team season-average stats), then predictions are compared
    against actual results.  No ODDS_API_KEY is required.

    Metrics returned:

    * **accuracy_pct** — % of games where the correct winner was predicted
    * **brier_score** — calibration quality (0 = perfect, 1 = worst)
    * **avg_spread_error** — mean absolute error on the predicted point spread
    * **avg_total_error** — mean absolute error on the predicted game total
    * **high_confidence_accuracy** — accuracy on games where the model was
      most confident (|home_win_prob − 0.5| > 15 %)
    """
    try:
        report = await run_backtest(
            sport=sport, days_back=days_back, max_games=max_games
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    game_preds = [
        GamePredictionResponse(**dataclasses.asdict(p))
        for p in report.predictions
    ]

    return BacktestReportResponse(
        sport=report.sport,
        games_evaluated=report.games_evaluated,
        correct_winner_predictions=report.correct_winner_predictions,
        accuracy_pct=report.accuracy_pct,
        brier_score=report.brier_score,
        avg_spread_error=report.avg_spread_error,
        avg_total_error=report.avg_total_error,
        high_confidence_games=report.high_confidence_games,
        high_confidence_accuracy=report.high_confidence_accuracy,
        predictions=game_preds,
    )


# ── One-off simulation ────────────────────────────────────────────────────────

@app.get(
    "/api/v1/simulate",
    response_model=MonteCarloSummary,
    tags=["analytics"],
    summary="Run a Monte Carlo simulation for any matchup",
)
async def simulate_matchup(
    sport: Literal["basketball_nba", "americanfootball_nfl"] = Query(
        ..., description="Sport key"
    ),
    home_team: str = Query(..., description="Home team display name"),
    away_team: str = Query(..., description="Away team display name"),
    n_simulations: int = Query(
        default=10000, ge=1000, le=100000, description="Number of MC iterations"
    ),
) -> MonteCarloSummary:
    """Run a Monte Carlo simulation for any home/away matchup.

    Team statistics are loaded automatically from ESPN standings.  Results
    include win probabilities, spread distribution, and projected total.
    No ODDS_API_KEY is required.
    """
    from edgeiq.team_stats import get_team_stats

    try:
        home_stats = await get_team_stats(home_team, sport)
        away_stats = await get_team_stats(away_team, sport)
        mc = run_monte_carlo(home_stats, away_stats, sport, n_simulations=n_simulations)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return MonteCarloSummary(
        n_simulations=mc.n_simulations,
        home_win_probability=mc.home_win_probability,
        away_win_probability=mc.away_win_probability,
        simulated_spread=mc.simulated_spread,
        simulated_total=mc.simulated_total,
        spread_std=mc.spread_std,
        spread_ci_low=mc.spread_ci_low,
        spread_ci_high=mc.spread_ci_high,
    )
