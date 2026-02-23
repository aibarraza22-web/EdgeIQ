"""Historical game backtesting.

Fetches completed game scores from ESPN's public scoreboard API and runs
Monte Carlo simulations for each game WITHOUT using the actual score —
exactly as the model would have operated in real time before the game.

Accuracy is then measured against the actual outcomes.

Usage
-----
From the FastAPI endpoint::

    report = await run_backtest(sport="basketball_nba", days_back=30)

From the CLI::

    python -m edgeiq.backtester
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from edgeiq.simulator import run_monte_carlo
from edgeiq.team_stats import get_team_stats, _normalise_sport

logger = logging.getLogger(__name__)

# ── ESPN scoreboard endpoints (public, no auth) ───────────────────────────────
_SCOREBOARD_URLS: dict[str, str] = {
    "basketball_nba": (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    ),
    "americanfootball_nfl": (
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    ),
}

_ESPN_HEADERS = {"User-Agent": "EdgeIQ/0.2"}
_N_SIMS = 10_000


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GamePrediction:
    """Single-game prediction vs actual result."""

    game_id: str
    date: str          # YYYYMMDD
    sport: str
    home_team: str
    away_team: str

    # Actual result (unknown at prediction time)
    home_score: int
    away_score: int
    home_won: bool
    actual_spread: float   # home − away
    actual_total: int

    # Monte Carlo prediction (made without the final score)
    mc_home_win_prob: float
    mc_away_win_prob: float
    mc_spread: float       # simulated spread (home − away; + = home favoured)
    mc_total: float        # simulated combined score
    mc_spread_std: float   # spread uncertainty

    predicted_home_win: bool
    prediction_correct: bool


@dataclass
class BacktestReport:
    """Aggregated backtesting metrics."""

    sport: str
    games_evaluated: int
    correct_winner_predictions: int
    accuracy_pct: float

    # Calibration metrics
    brier_score: float         # MSE of win prob vs binary outcome (lower = better)
    avg_spread_error: float    # MAE: |mc_spread − actual_spread|
    avg_total_error: float     # MAE: |mc_total − actual_total|

    # High-confidence subset (|home_prob − 0.5| > 0.15)
    high_confidence_games: int
    high_confidence_accuracy: float

    # Per-game detail
    predictions: list[GamePrediction]


# ── Public API ────────────────────────────────────────────────────────────────

async def run_backtest(
    sport: str = "basketball_nba",
    days_back: int = 30,
    max_games: int = 100,
) -> BacktestReport:
    """Run Monte Carlo backtesting over recently completed games.

    The simulation is run **without** the actual game scores — it uses only
    the current season team statistics (offensive / defensive averages) that
    would be available before the game.  This mirrors live-prediction
    conditions; results show how accurate the MC model is when used blindly.

    Parameters
    ----------
    sport:
        ``"basketball_nba"`` or ``"americanfootball_nfl"``.
    days_back:
        How many calendar days of history to scan (capped by ESPN data
        availability).
    max_games:
        Stop after collecting this many completed games (controls runtime).

    Returns
    -------
    BacktestReport
        Accuracy, calibration, and per-game prediction details.
    """
    sport_key = _normalise_sport(sport)
    url = _SCOREBOARD_URLS.get(sport_key)
    if not url:
        raise ValueError(f"Unsupported sport for backtesting: {sport!r}")

    completed = await _fetch_completed_games(url, days_back, max_games)
    logger.info("Backtesting %d completed %s games", len(completed), sport_key)

    if not completed:
        return _empty_report(sport_key)

    predictions: list[GamePrediction] = []
    for game in completed:
        pred = await _predict_game(game, sport_key)
        predictions.append(pred)

    return _aggregate(sport_key, predictions)


# ── Game prediction ───────────────────────────────────────────────────────────

async def _predict_game(game: dict[str, Any], sport_key: str) -> GamePrediction:
    """Run MC for one completed game and return the prediction vs actual."""
    home_name = game["home_team"]
    away_name = game["away_team"]

    home_stats = await get_team_stats(home_name, sport_key)
    away_stats = await get_team_stats(away_name, sport_key)

    mc = run_monte_carlo(home_stats, away_stats, sport_key, n_simulations=_N_SIMS)

    home_score = game["home_score"]
    away_score = game["away_score"]
    home_won = home_score > away_score
    predicted_home_win = mc.home_win_probability >= 0.5

    return GamePrediction(
        game_id=game["id"],
        date=game["date"],
        sport=sport_key,
        home_team=home_name,
        away_team=away_name,
        home_score=home_score,
        away_score=away_score,
        home_won=home_won,
        actual_spread=float(home_score - away_score),
        actual_total=home_score + away_score,
        mc_home_win_prob=mc.home_win_probability,
        mc_away_win_prob=mc.away_win_probability,
        mc_spread=mc.simulated_spread,
        mc_total=mc.simulated_total,
        mc_spread_std=mc.spread_std,
        predicted_home_win=predicted_home_win,
        prediction_correct=(predicted_home_win == home_won),
    )


# ── ESPN scoreboard fetcher ───────────────────────────────────────────────────

async def _fetch_completed_games(
    url: str, days_back: int, max_games: int
) -> list[dict[str, Any]]:
    """Return completed game records from the ESPN scoreboard."""
    import httpx

    games: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=20, headers=_ESPN_HEADERS) as client:
        for delta in range(1, days_back + 1):
            if len(games) >= max_games:
                break
            target = today - timedelta(days=delta)
            date_str = target.strftime("%Y%m%d")
            try:
                resp = await client.get(url, params={"dates": date_str, "limit": 20})
                resp.raise_for_status()
                for event in resp.json().get("events", []):
                    game = _parse_event(event, date_str)
                    if game:
                        games.append(game)
                    if len(games) >= max_games:
                        break
            except Exception as exc:
                logger.debug("ESPN scoreboard %s failed: %s", date_str, exc)

    return games


def _parse_event(event: dict[str, Any], date_str: str) -> Optional[dict[str, Any]]:
    """Extract a completed game record from an ESPN event object."""
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]

    # Skip games that are not yet completed
    if not comp.get("status", {}).get("type", {}).get("completed", False):
        return None

    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        return None

    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    home_score = _safe_int(home.get("score"))
    away_score = _safe_int(away.get("score"))

    # Reject unscored / zero-zero games
    if home_score == 0 and away_score == 0:
        return None

    home_name = home.get("team", {}).get("displayName", "")
    away_name = away.get("team", {}).get("displayName", "")
    if not home_name or not away_name:
        return None

    return {
        "id": event.get("id", ""),
        "date": date_str,
        "home_team": home_name,
        "away_team": away_name,
        "home_score": home_score,
        "away_score": away_score,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ── Report aggregation ────────────────────────────────────────────────────────

def _aggregate(sport: str, preds: list[GamePrediction]) -> BacktestReport:
    n = len(preds)
    if n == 0:
        return _empty_report(sport)

    correct = sum(1 for p in preds if p.prediction_correct)

    # Brier score: mean((P_home_win − outcome_home_win)²)
    brier = sum(
        (p.mc_home_win_prob - (1.0 if p.home_won else 0.0)) ** 2
        for p in preds
    ) / n

    avg_spread_err = sum(abs(p.mc_spread - p.actual_spread) for p in preds) / n
    avg_total_err = sum(abs(p.mc_total - p.actual_total) for p in preds) / n

    # High-confidence: model clearly favours one side
    hc = [p for p in preds if abs(p.mc_home_win_prob - 0.5) > 0.15]
    hc_acc = sum(1 for p in hc if p.prediction_correct) / len(hc) if hc else 0.0

    return BacktestReport(
        sport=sport,
        games_evaluated=n,
        correct_winner_predictions=correct,
        accuracy_pct=round(correct / n * 100, 1),
        brier_score=round(brier, 4),
        avg_spread_error=round(avg_spread_err, 2),
        avg_total_error=round(avg_total_err, 2),
        high_confidence_games=len(hc),
        high_confidence_accuracy=round(hc_acc * 100, 1),
        predictions=preds,
    )


def _empty_report(sport: str) -> BacktestReport:
    return BacktestReport(
        sport=sport,
        games_evaluated=0,
        correct_winner_predictions=0,
        accuracy_pct=0.0,
        brier_score=0.0,
        avg_spread_error=0.0,
        avg_total_error=0.0,
        high_confidence_games=0,
        high_confidence_accuracy=0.0,
        predictions=[],
    )


# ── CLI entry-point ───────────────────────────────────────────────────────────

async def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EdgeIQ Monte Carlo backtester")
    parser.add_argument(
        "--sport",
        default="basketball_nba",
        choices=["basketball_nba", "americanfootball_nfl"],
        help="Sport to backtest (default: basketball_nba)",
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Calendar days to look back (default: 30)"
    )
    parser.add_argument(
        "--max-games", type=int, default=100, help="Max games to evaluate (default: 100)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = await run_backtest(sport=args.sport, days_back=args.days, max_games=args.max_games)

    print(f"\n{'='*60}")
    print(f"  EdgeIQ Monte Carlo Backtest — {report.sport}")
    print(f"{'='*60}")
    print(f"  Games evaluated       : {report.games_evaluated}")
    print(f"  Correct winner picks  : {report.correct_winner_predictions}")
    print(f"  Overall accuracy      : {report.accuracy_pct}%")
    print(f"  Brier score           : {report.brier_score}  (lower = better)")
    print(f"  Avg spread error (MAE): {report.avg_spread_error} pts")
    print(f"  Avg total error  (MAE): {report.avg_total_error} pts")
    print(f"  High-confidence games : {report.high_confidence_games}")
    print(f"  High-confidence acc   : {report.high_confidence_accuracy}%")
    print(f"{'='*60}\n")

    if report.predictions:
        print("  Sample predictions (first 10):")
        print(f"  {'Date':<10} {'Matchup':<40} {'MC%':>5}  {'Actual':>7}  {'OK':>3}")
        print(f"  {'-'*10} {'-'*40} {'-'*5}  {'-'*7}  {'-'*3}")
        for p in report.predictions[:10]:
            matchup = f"{p.away_team} @ {p.home_team}"[:40]
            mc_pct = f"{p.mc_home_win_prob*100:.0f}%"
            result = f"{p.away_score}-{p.home_score}"
            ok = "✓" if p.prediction_correct else "✗"
            print(f"  {p.date:<10} {matchup:<40} {mc_pct:>5}  {result:>7}  {ok:>3}")


if __name__ == "__main__":
    asyncio.run(_cli_main())
