"""Tests for team stats matching helpers (no network required)."""
import pytest

from edgeiq.team_stats import (
    _exact_match,
    _normalise_sport,
    _partial_match,
    _word_overlap_match,
)
from edgeiq.simulator import TeamStats


def _ts(name: str) -> TeamStats:
    return TeamStats(name=name, points_per_game=110, points_allowed_per_game=110)


CACHE = {
    "Boston Celtics": _ts("Boston Celtics"),
    "Los Angeles Lakers": _ts("Los Angeles Lakers"),
    "Golden State Warriors": _ts("Golden State Warriors"),
    "Kansas City Chiefs": _ts("Kansas City Chiefs"),
}


# ── _normalise_sport ──────────────────────────────────────────────────────────

def test_normalise_nba():
    assert _normalise_sport("basketball_nba") == "basketball_nba"
    assert _normalise_sport("nba") == "basketball_nba"


def test_normalise_nfl():
    assert _normalise_sport("americanfootball_nfl") == "americanfootball_nfl"
    assert _normalise_sport("nfl") == "americanfootball_nfl"


def test_normalise_unknown():
    assert _normalise_sport("hockey_nhl") == "hockey_nhl"


# ── _exact_match ──────────────────────────────────────────────────────────────

def test_exact_match_found():
    result = _exact_match("Boston Celtics", CACHE)
    assert result is not None
    assert result.name == "Boston Celtics"


def test_exact_match_not_found():
    result = _exact_match("Miami Heat", CACHE)
    assert result is None


# ── _partial_match ────────────────────────────────────────────────────────────

def test_partial_match_substring():
    result = _partial_match("Lakers", CACHE)
    assert result is not None
    assert result.name == "Los Angeles Lakers"


def test_partial_match_full_in_key():
    result = _partial_match("Celtics", CACHE)
    assert result is not None
    assert result.name == "Boston Celtics"


# ── _word_overlap_match ───────────────────────────────────────────────────────

def test_word_overlap_warriors():
    result = _word_overlap_match("Warriors", CACHE)
    assert result is not None
    assert result.name == "Golden State Warriors"


def test_word_overlap_chiefs():
    result = _word_overlap_match("Kansas City Chiefs", CACHE)
    assert result is not None
    assert result.name == "Kansas City Chiefs"


def test_word_overlap_no_match():
    result = _word_overlap_match("Zzzzunknown", CACHE)
    assert result is None
