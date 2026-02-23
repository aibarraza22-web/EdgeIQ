"""Tests for the backtesting aggregation logic (no network required)."""
import pytest

from edgeiq.backtester import GamePrediction, _aggregate, _empty_report, _parse_event


# ── _parse_event ──────────────────────────────────────────────────────────────

def _make_event(home_score, away_score, completed=True):
    return {
        "id": "evt1",
        "competitions": [
            {
                "status": {"type": {"completed": completed}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": str(home_score),
                        "team": {"displayName": "Home Team"},
                    },
                    {
                        "homeAway": "away",
                        "score": str(away_score),
                        "team": {"displayName": "Away Team"},
                    },
                ],
            }
        ],
    }


def test_parse_event_completed():
    game = _parse_event(_make_event(110, 105), "20250101")
    assert game is not None
    assert game["home_score"] == 110
    assert game["away_score"] == 105
    assert game["home_team"] == "Home Team"
    assert game["away_team"] == "Away Team"


def test_parse_event_not_completed():
    game = _parse_event(_make_event(0, 0, completed=False), "20250101")
    assert game is None


def test_parse_event_zero_zero_completed():
    # 0-0 final — reject (data issue)
    game = _parse_event(_make_event(0, 0, completed=True), "20250101")
    assert game is None


# ── _aggregate ────────────────────────────────────────────────────────────────

def _make_prediction(home_won: bool, mc_home_prob: float, home_score=110, away_score=105):
    predicted_home_win = mc_home_prob >= 0.5
    actual_spread = float(home_score - away_score)
    return GamePrediction(
        game_id="g1",
        date="20250101",
        sport="basketball_nba",
        home_team="Home",
        away_team="Away",
        home_score=home_score,
        away_score=away_score,
        home_won=home_won,
        actual_spread=actual_spread,
        actual_total=home_score + away_score,
        mc_home_win_prob=mc_home_prob,
        mc_away_win_prob=1.0 - mc_home_prob,
        mc_spread=5.0,
        mc_total=215.0,
        mc_spread_std=15.0,
        predicted_home_win=predicted_home_win,
        prediction_correct=(predicted_home_win == home_won),
    )


def test_aggregate_perfect_accuracy():
    preds = [
        _make_prediction(True, 0.70),
        _make_prediction(True, 0.65),
        _make_prediction(False, 0.35),
    ]
    report = _aggregate("basketball_nba", preds)
    assert report.accuracy_pct == 100.0
    assert report.games_evaluated == 3
    assert report.correct_winner_predictions == 3


def test_aggregate_zero_accuracy():
    preds = [
        _make_prediction(False, 0.80),  # predicted home, home lost
        _make_prediction(True, 0.30),   # predicted away, away lost
    ]
    report = _aggregate("basketball_nba", preds)
    assert report.accuracy_pct == 0.0


def test_aggregate_50_percent():
    preds = [
        _make_prediction(True, 0.70),   # correct
        _make_prediction(True, 0.40),   # incorrect (predicted away)
    ]
    report = _aggregate("basketball_nba", preds)
    assert report.accuracy_pct == 50.0


def test_aggregate_brier_score_perfect():
    # Perfect predictions: predicted 1.0 for home when home won
    preds = [
        _make_prediction(True, 1.0),
        _make_prediction(False, 0.0),
    ]
    report = _aggregate("basketball_nba", preds)
    assert report.brier_score == pytest.approx(0.0, abs=1e-6)


def test_aggregate_brier_score_worst():
    # Worst: predicted 1.0 for home but away won
    preds = [
        _make_prediction(False, 1.0),
        _make_prediction(True, 0.0),
    ]
    report = _aggregate("basketball_nba", preds)
    assert report.brier_score == pytest.approx(1.0, abs=1e-6)


def test_aggregate_spread_error():
    # mc_spread = 5.0, actual = 110-105 = 5 → error = 0
    preds = [_make_prediction(True, 0.70, home_score=110, away_score=105)]
    report = _aggregate("basketball_nba", preds)
    assert report.avg_spread_error == pytest.approx(0.0, abs=0.01)


def test_aggregate_high_confidence_subset():
    preds = [
        _make_prediction(True, 0.80),   # high conf, correct
        _make_prediction(False, 0.75),  # high conf, incorrect
        _make_prediction(True, 0.55),   # low conf (|0.55-0.5| < 0.15)
    ]
    report = _aggregate("basketball_nba", preds)
    assert report.high_confidence_games == 2
    assert report.high_confidence_accuracy == pytest.approx(50.0)


def test_aggregate_empty():
    report = _aggregate("basketball_nba", [])
    assert report.games_evaluated == 0
    assert report.accuracy_pct == 0.0


def test_empty_report():
    report = _empty_report("americanfootball_nfl")
    assert report.sport == "americanfootball_nfl"
    assert report.games_evaluated == 0
