"""Monte Carlo game simulation engine.

Simulates N games using team offensive/defensive ratings and returns win
probabilities, spread distributions, and expected totals.

Requires numpy when available for performance; falls back to the standard
library (random + math) automatically.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore[import]
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    logger.debug("numpy not installed; Monte Carlo will use stdlib random (slower)")

# ── Sport-specific constants ────────────────────────────────────────────────
# Per-team score standard deviations (calibrated from historical data)
_NBA_SCORE_STD: float = 11.5   # NBA team score σ ≈ 11-12 pts
_NFL_SCORE_STD: float = 10.0   # NFL team score σ ≈ 9-11 pts

# Home advantage in points (well-supported empirically)
_NBA_HOME_ADV: float = 3.0     # NBA home court: ~3 pts
_NFL_HOME_ADV: float = 2.5     # NFL home field: ~2.5 pts

_DEFAULT_N_SIMS: int = 10_000


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TeamStats:
    """Team season-average statistics used as MC simulation inputs."""

    name: str
    points_per_game: float           # Offensive average (pts scored)
    points_allowed_per_game: float   # Defensive average (pts allowed)
    win_pct: float = 0.500
    games_played: int = 0

    # Advanced NBA stats (optional; used when available)
    offensive_rating: Optional[float] = None   # pts per 100 possessions
    defensive_rating: Optional[float] = None   # pts allowed per 100 possessions
    pace: Optional[float] = None               # possessions per 48 min


@dataclass
class MonteCarloResult:
    """Simulation output for a single matchup."""

    home_team: str
    away_team: str

    home_win_probability: float
    away_win_probability: float

    # Spread: home_score − away_score.  Positive = home favoured.
    simulated_spread: float
    simulated_total: float

    home_score_mean: float
    away_score_mean: float
    spread_std: float               # Uncertainty on the spread

    n_simulations: int

    # 95 % confidence interval on the spread
    spread_ci_low: float
    spread_ci_high: float

    # 10th / 90th percentiles (useful for line shopping)
    spread_p10: float
    spread_p90: float


# ── Public API ────────────────────────────────────────────────────────────────

def run_monte_carlo(
    home_stats: TeamStats,
    away_stats: TeamStats,
    sport: str,
    n_simulations: int = _DEFAULT_N_SIMS,
    home_advantage_override: Optional[float] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Run a Monte Carlo simulation for a game.

    Parameters
    ----------
    home_stats, away_stats:
        Season-average stats for each team (offence and defence).
    sport:
        Sport key, e.g. ``"basketball_nba"`` or ``"americanfootball_nfl"``.
    n_simulations:
        Number of independent game simulations.  ≥1 000 recommended;
        10 000 for stable probabilities (default).
    home_advantage_override:
        Override the default home-court/field point advantage.
    seed:
        RNG seed for reproducibility.  Leave ``None`` for live predictions.

    Returns
    -------
    MonteCarloResult
    """
    sport_lower = sport.lower()
    if "nba" in sport_lower or "basketball" in sport_lower:
        hca = home_advantage_override if home_advantage_override is not None else _NBA_HOME_ADV
        score_std = _NBA_SCORE_STD
        lo, hi = 70.0, 170.0
    elif "nfl" in sport_lower or "football" in sport_lower:
        hca = home_advantage_override if home_advantage_override is not None else _NFL_HOME_ADV
        score_std = _NFL_SCORE_STD
        lo, hi = 0.0, 65.0
    else:
        hca = home_advantage_override if home_advantage_override is not None else 2.0
        score_std = 10.0
        lo, hi = 0.0, 300.0

    # Expected scores: blend each team's offence with the opponent's defence
    if (
        home_stats.offensive_rating and home_stats.defensive_rating
        and away_stats.offensive_rating and away_stats.defensive_rating
        and home_stats.pace
    ):
        pace = home_stats.pace
        home_mu = (home_stats.offensive_rating + away_stats.defensive_rating) / 2 * (pace / 100) + hca
        away_mu = (away_stats.offensive_rating + home_stats.defensive_rating) / 2 * (pace / 100)
    else:
        home_mu = (home_stats.points_per_game + away_stats.points_allowed_per_game) / 2 + hca
        away_mu = (away_stats.points_per_game + home_stats.points_allowed_per_game) / 2

    if _HAS_NUMPY:
        return _simulate_numpy(
            home_stats.name, away_stats.name,
            home_mu, away_mu, score_std,
            lo, hi, n_simulations, seed,
        )
    return _simulate_stdlib(
        home_stats.name, away_stats.name,
        home_mu, away_mu, score_std,
        lo, hi, n_simulations, seed,
    )


# ── numpy implementation ──────────────────────────────────────────────────────

def _simulate_numpy(
    home_name: str, away_name: str,
    home_mu: float, away_mu: float,
    score_std: float, lo: float, hi: float,
    n: int, seed: Optional[int],
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    home_scores = np.clip(rng.normal(home_mu, score_std, n), lo, hi)
    away_scores = np.clip(rng.normal(away_mu, score_std, n), lo, hi)
    return _build_result_numpy(home_name, away_name, home_scores, away_scores, n)


def _build_result_numpy(
    home_name: str, away_name: str,
    home_scores, away_scores, n: int,
) -> MonteCarloResult:
    spreads = home_scores - away_scores
    home_wins = int(np.sum(spreads > 0))
    return MonteCarloResult(
        home_team=home_name,
        away_team=away_name,
        home_win_probability=round(home_wins / n, 4),
        away_win_probability=round((n - home_wins) / n, 4),
        simulated_spread=round(float(np.mean(spreads)), 2),
        simulated_total=round(float(np.mean(home_scores + away_scores)), 2),
        home_score_mean=round(float(np.mean(home_scores)), 2),
        away_score_mean=round(float(np.mean(away_scores)), 2),
        spread_std=round(float(np.std(spreads)), 2),
        n_simulations=n,
        spread_ci_low=round(float(np.percentile(spreads, 2.5)), 2),
        spread_ci_high=round(float(np.percentile(spreads, 97.5)), 2),
        spread_p10=round(float(np.percentile(spreads, 10)), 2),
        spread_p90=round(float(np.percentile(spreads, 90)), 2),
    )


# ── stdlib implementation (fallback) ─────────────────────────────────────────

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _simulate_stdlib(
    home_name: str, away_name: str,
    home_mu: float, away_mu: float,
    score_std: float, lo: float, hi: float,
    n: int, seed: Optional[int],
) -> MonteCarloResult:
    rng = random.Random(seed)
    home_scores = [_clamp(rng.gauss(home_mu, score_std), lo, hi) for _ in range(n)]
    away_scores = [_clamp(rng.gauss(away_mu, score_std), lo, hi) for _ in range(n)]
    return _build_result_stdlib(home_name, away_name, home_scores, away_scores, n)


def _percentile_stdlib(data: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a sorted list."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100) * (len(s) - 1)
    lo_i = int(idx)
    hi_i = min(lo_i + 1, len(s) - 1)
    frac = idx - lo_i
    return s[lo_i] * (1 - frac) + s[hi_i] * frac


def _build_result_stdlib(
    home_name: str, away_name: str,
    home_scores: list[float], away_scores: list[float],
    n: int,
) -> MonteCarloResult:
    spreads = [h - a for h, a in zip(home_scores, away_scores)]
    home_wins = sum(1 for s in spreads if s > 0)
    mean_spread = sum(spreads) / n
    mean_total = sum(h + a for h, a in zip(home_scores, away_scores)) / n
    mean_home = sum(home_scores) / n
    mean_away = sum(away_scores) / n
    variance = sum((s - mean_spread) ** 2 for s in spreads) / n
    std_spread = math.sqrt(variance)

    return MonteCarloResult(
        home_team=home_name,
        away_team=away_name,
        home_win_probability=round(home_wins / n, 4),
        away_win_probability=round((n - home_wins) / n, 4),
        simulated_spread=round(mean_spread, 2),
        simulated_total=round(mean_total, 2),
        home_score_mean=round(mean_home, 2),
        away_score_mean=round(mean_away, 2),
        spread_std=round(std_spread, 2),
        n_simulations=n,
        spread_ci_low=round(_percentile_stdlib(spreads, 2.5), 2),
        spread_ci_high=round(_percentile_stdlib(spreads, 97.5), 2),
        spread_p10=round(_percentile_stdlib(spreads, 10), 2),
        spread_p90=round(_percentile_stdlib(spreads, 90), 2),
    )


# ── Spread / total probability helpers ───────────────────────────────────────

def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Cumulative distribution function for N(mu, sigma²)."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def spread_cover_probability(
    mc: MonteCarloResult,
    home_team_name: str,
    selection: str,
    line: float,
) -> float:
    """Return the model probability that *selection* covers *line*.

    Odds-API spread convention:
      * Home −6.5  →  home covers if actual spread > 6.5
      * Away +6.5  →  away covers if actual spread < 6.5

    Parameters
    ----------
    mc:
        Monte Carlo result (provides spread distribution).
    home_team_name:
        Display name of the home team (to detect home/away side).
    selection:
        Team name from the odds outcome.
    line:
        Point spread (negative for home favourite, positive for away underdog).
    """
    mu = mc.simulated_spread
    sigma = max(mc.spread_std, 1.0)  # avoid division-by-zero

    if selection == home_team_name:
        # Home covers: spread > –line  (e.g., line=−6.5 → spread > 6.5)
        threshold = -line
        prob = 1.0 - _normal_cdf(threshold, mu, sigma)
    else:
        # Away covers: spread < –line  (e.g., line=+6.5 → spread < 6.5)
        threshold = -line
        prob = _normal_cdf(threshold, mu, sigma)

    return max(0.05, min(0.95, prob))


def total_cover_probability(
    mc: MonteCarloResult,
    selection: str,
    line: float,
) -> float:
    """Return the model probability for an over/under bet.

    Parameters
    ----------
    mc:
        Monte Carlo result (provides total distribution).
    selection:
        ``"Over"`` or ``"Under"`` (case-insensitive).
    line:
        The posted total (e.g., 220.5).
    """
    mu = mc.simulated_total
    # Total std ≈ sqrt(2) * score_std because both teams contribute variance
    sigma = max(mc.spread_std * 0.9, 1.0)

    if selection.lower() == "over":
        prob = 1.0 - _normal_cdf(line, mu, sigma)
    else:
        prob = _normal_cdf(line, mu, sigma)

    return max(0.05, min(0.95, prob))
