"""Tests for odds utility functions."""
from edgeiq.odds import american_to_implied_prob, confidence_from_edge


def test_american_to_implied_prob_positive():
    assert round(american_to_implied_prob(150), 4) == 0.4


def test_american_to_implied_prob_negative():
    assert round(american_to_implied_prob(-150), 4) == 0.6


def test_american_to_implied_prob_even():
    assert round(american_to_implied_prob(100), 4) == 0.5


def test_confidence_bounds():
    assert confidence_from_edge(-1) == 50
    assert confidence_from_edge(20) == 85


def test_confidence_mid():
    score = confidence_from_edge(2.0)
    assert 50 <= score <= 85


def test_confidence_monotone():
    scores = [confidence_from_edge(e) for e in [2, 4, 6, 8, 10]]
    assert scores == sorted(scores)
