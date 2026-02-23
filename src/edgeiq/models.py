from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OddsQuote(BaseModel):
    sportsbook: str
    market: str
    selection: str
    price_american: int
    implied_probability: float


class MonteCarloSummary(BaseModel):
    """Condensed Monte Carlo output attached to each pick."""

    n_simulations: int
    home_win_probability: float
    away_win_probability: float
    simulated_spread: float       # home − away; positive = home favoured
    simulated_total: float
    spread_std: float
    spread_ci_low: float          # 2.5th-percentile spread
    spread_ci_high: float         # 97.5th-percentile spread


class Pick(BaseModel):
    sport: str
    event_id: str
    commence_time: datetime
    matchup: str
    market: str
    selection: str
    best_odds_american: int
    best_sportsbook: str
    model_probability: float = Field(ge=0, le=1)
    implied_probability: float = Field(ge=0, le=1)
    edge_percentage: float
    confidence_score: int = Field(ge=0, le=100)
    rationale: str
    monte_carlo: Optional[MonteCarloSummary] = None


class DailyPicksResponse(BaseModel):
    generated_at: datetime
    picks: list[Pick]
    disclaimer: str = (
        "EdgeIQ provides analytics only — not financial advice. "
        "Never bet more than you can afford to lose. "
        "Problem gambling help: 1-800-GAMBLER | ncpgambling.org"
    )


# ── Backtesting models ────────────────────────────────────────────────────────

class GamePredictionResponse(BaseModel):
    game_id: str
    date: str
    sport: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_won: bool
    actual_spread: float
    actual_total: int
    mc_home_win_prob: float
    mc_away_win_prob: float
    mc_spread: float
    mc_total: float
    mc_spread_std: float
    predicted_home_win: bool
    prediction_correct: bool


class BacktestReportResponse(BaseModel):
    sport: str
    games_evaluated: int
    correct_winner_predictions: int
    accuracy_pct: float
    brier_score: float
    avg_spread_error: float
    avg_total_error: float
    high_confidence_games: int
    high_confidence_accuracy: float
    predictions: list[GamePredictionResponse]
    methodology_note: str = (
        "Predictions use current-season team stats (offensive/defensive averages). "
        "Each game is simulated 10 000 times via Monte Carlo without the final score. "
        "Accuracy is measured against actual results."
    )
