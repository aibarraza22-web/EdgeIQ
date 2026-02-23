"""Tests for the Monte Carlo simulation engine.

All tests use the stdlib fallback so they run without numpy installed.
"""
import math

import pytest

from edgeiq.simulator import (
    MonteCarloResult,
    TeamStats,
    _normal_cdf,
    run_monte_carlo,
    spread_cover_probability,
    total_cover_probability,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def strong_home() -> TeamStats:
    """Dominant home team: 120 ppg offence, 100 papg defence."""
    return TeamStats(name="Home Warriors", points_per_game=120, points_allowed_per_game=100)


@pytest.fixture
def weak_away() -> TeamStats:
    """Weak away team: 100 ppg offence, 120 papg defence."""
    return TeamStats(name="Away Celtics", points_per_game=100, points_allowed_per_game=120)


@pytest.fixture
def balanced_home() -> TeamStats:
    return TeamStats(name="Home Balanced", points_per_game=113, points_allowed_per_game=113)


@pytest.fixture
def balanced_away() -> TeamStats:
    return TeamStats(name="Away Balanced", points_per_game=113, points_allowed_per_game=113)


# ── run_monte_carlo ───────────────────────────────────────────────────────────

def test_mc_returns_correct_type(strong_home, weak_away):
    result = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=500, seed=42)
    assert isinstance(result, MonteCarloResult)


def test_mc_probabilities_sum_to_one(strong_home, weak_away):
    result = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=500, seed=42)
    total = result.home_win_probability + result.away_win_probability
    assert abs(total - 1.0) < 0.01


def test_mc_strong_home_favoured(strong_home, weak_away):
    result = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=2000, seed=0)
    assert result.home_win_probability > 0.70, (
        f"Expected strong home team to win >70% but got {result.home_win_probability}"
    )


def test_mc_balanced_teams_near_fifty_fifty(balanced_home, balanced_away):
    result = run_monte_carlo(
        balanced_home, balanced_away, "basketball_nba", n_simulations=5000, seed=1
    )
    # With balanced teams + home-court advantage, home should be ~60 %
    assert 0.50 < result.home_win_probability < 0.70, (
        f"Balanced teams: home win prob should be ~55-65%, got {result.home_win_probability}"
    )


def test_mc_nfl_sport_key():
    nfl_home = TeamStats(name="Chiefs", points_per_game=28, points_allowed_per_game=20)
    nfl_away = TeamStats(name="Bills", points_per_game=25, points_allowed_per_game=23)
    result = run_monte_carlo(nfl_home, nfl_away, "americanfootball_nfl", n_simulations=500, seed=5)
    assert isinstance(result, MonteCarloResult)
    assert result.home_win_probability + result.away_win_probability == pytest.approx(1.0, abs=0.02)
    assert 0 < result.home_score_mean <= 65
    assert 0 < result.away_score_mean <= 65


def test_mc_scores_in_realistic_nba_range(balanced_home, balanced_away):
    result = run_monte_carlo(
        balanced_home, balanced_away, "basketball_nba", n_simulations=1000, seed=7
    )
    assert 90 < result.home_score_mean < 140
    assert 90 < result.away_score_mean < 140


def test_mc_spread_ci_ordering(strong_home, weak_away):
    result = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=2000, seed=3)
    assert result.spread_ci_low < result.simulated_spread < result.spread_ci_high


def test_mc_spread_p10_p90_ordering(strong_home, weak_away):
    result = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=2000, seed=4)
    assert result.spread_p10 < result.spread_p90


def test_mc_reproducible_with_seed(strong_home, weak_away):
    r1 = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=1000, seed=99)
    r2 = run_monte_carlo(strong_home, weak_away, "basketball_nba", n_simulations=1000, seed=99)
    assert r1.home_win_probability == r2.home_win_probability
    assert r1.simulated_spread == r2.simulated_spread


def test_mc_advanced_stats_used_when_provided():
    """When offensive/defensive ratings + pace are set, the advanced path is taken."""
    home = TeamStats(
        name="Advanced Home",
        points_per_game=115,
        points_allowed_per_game=108,
        offensive_rating=115.0,
        defensive_rating=108.0,
        pace=100.0,
    )
    away = TeamStats(
        name="Advanced Away",
        points_per_game=110,
        points_allowed_per_game=112,
        offensive_rating=110.0,
        defensive_rating=112.0,
        pace=100.0,
    )
    result = run_monte_carlo(home, away, "basketball_nba", n_simulations=500, seed=12)
    assert isinstance(result, MonteCarloResult)
    assert result.home_win_probability > 0.5


# ── normal_cdf ────────────────────────────────────────────────────────────────

def test_normal_cdf_mean_is_half():
    assert _normal_cdf(0.0, 0.0, 1.0) == pytest.approx(0.5, abs=1e-6)


def test_normal_cdf_one_sigma():
    # P(X < μ + σ) ≈ 0.8413
    assert _normal_cdf(1.0, 0.0, 1.0) == pytest.approx(0.8413, abs=0.001)


# ── spread_cover_probability ──────────────────────────────────────────────────

def test_spread_cover_home_favoured():
    mc = run_monte_carlo(
        TeamStats("Home", 120, 100),
        TeamStats("Away", 100, 120),
        "basketball_nba",
        n_simulations=2000,
        seed=20,
    )
    # Home -10 means home covers if spread > 10; strong home team should cover often
    prob = spread_cover_probability(mc, "Home", "Home", line=-10.0)
    assert 0.3 < prob < 0.95  # not certain but plausible


def test_spread_cover_away_side():
    mc = run_monte_carlo(
        TeamStats("Home", 110, 112),
        TeamStats("Away", 115, 108),
        "basketball_nba",
        n_simulations=2000,
        seed=21,
    )
    # Away +3 — away covers if spread < 3
    prob_away = spread_cover_probability(mc, "Home", "Away", line=3.0)
    prob_home = spread_cover_probability(mc, "Home", "Home", line=-3.0)
    # These should be roughly complementary (not exact due to clamping)
    assert abs(prob_away + prob_home - 1.0) < 0.1


# ── total_cover_probability ───────────────────────────────────────────────────

def test_total_over_under_complement():
    mc = run_monte_carlo(
        TeamStats("Home", 113, 113),
        TeamStats("Away", 113, 113),
        "basketball_nba",
        n_simulations=2000,
        seed=30,
    )
    over = total_cover_probability(mc, "Over", mc.simulated_total)
    under = total_cover_probability(mc, "Under", mc.simulated_total)
    # At the mean line, over + under ≈ 1
    assert abs(over + under - 1.0) < 0.15


def test_total_far_over_low_prob():
    mc = run_monte_carlo(
        TeamStats("Home", 110, 112),
        TeamStats("Away", 110, 112),
        "basketball_nba",
        n_simulations=2000,
        seed=31,
    )
    prob = total_cover_probability(mc, "Over", 300.0)  # absurdly high line
    assert prob < 0.1


def test_total_far_under_low_prob():
    mc = run_monte_carlo(
        TeamStats("Home", 110, 112),
        TeamStats("Away", 110, 112),
        "basketball_nba",
        n_simulations=2000,
        seed=32,
    )
    prob = total_cover_probability(mc, "Under", 50.0)  # absurdly low line
    assert prob < 0.1
