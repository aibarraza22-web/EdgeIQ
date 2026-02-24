"""Team statistics provider using ESPN's public API.

No API key required.  Season standings are fetched once per process (or
on-demand) and cached in-memory, minimising repeated network calls during
a single server session.

Fallback league-average values are used automatically when the ESPN API is
unreachable (e.g., offline / rate-limited).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from edgeiq.simulator import TeamStats

logger = logging.getLogger(__name__)

# ── ESPN public endpoints (no auth required) ──────────────────────────────────
_ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_ESPN_NFL_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
_ESPN_HEADERS = {"User-Agent": "EdgeIQ/0.2 (+https://github.com/edgeiq)"}

# NBA season slug (e.g., 2026 = the 2025-26 season)
_NBA_SEASON = 2026
# NFL season slug (e.g., 2025 = the 2025-26 season)
_NFL_SEASON = 2025

# ── League-average fallbacks ─────────────────────────────────────────────────
_NBA_LEAGUE_AVG = TeamStats(
    name="NBA League Average",
    points_per_game=113.5,
    points_allowed_per_game=113.5,
    win_pct=0.500,
)
_NFL_LEAGUE_AVG = TeamStats(
    name="NFL League Average",
    points_per_game=23.0,
    points_allowed_per_game=23.0,
    win_pct=0.500,
)

# ── In-process cache ──────────────────────────────────────────────────────────
_STATS_CACHE: dict[str, dict[str, TeamStats]] = {}
_CACHE_LOADED: dict[str, bool] = {}
_CACHE_LOCK = asyncio.Lock()


# ── Public interface ──────────────────────────────────────────────────────────

async def get_team_stats(team_name: str, sport: str) -> TeamStats:
    """Return ``TeamStats`` for *team_name* in *sport*.

    The cache is loaded lazily from ESPN standings on the first call per sport.
    Falls back to league-average values when no matching team is found.
    """
    sport_key = _normalise_sport(sport)
    await _ensure_cache_loaded(sport_key)

    cache = _STATS_CACHE.get(sport_key, {})
    team = (
        _exact_match(team_name, cache)
        or _partial_match(team_name, cache)
        or _word_overlap_match(team_name, cache)
    )
    if team is None:
        logger.debug("No stats for '%s' (%s); using league average", team_name, sport_key)
        avg = _NFL_LEAGUE_AVG if "nfl" in sport_key else _NBA_LEAGUE_AVG
        return TeamStats(
            name=team_name,
            points_per_game=avg.points_per_game,
            points_allowed_per_game=avg.points_allowed_per_game,
        )
    return team


async def invalidate_cache(sport: Optional[str] = None) -> None:
    """Force a reload of the stats cache on the next request.

    Pass *sport* to invalidate only one sport; omit to invalidate all.
    """
    async with _CACHE_LOCK:
        if sport:
            key = _normalise_sport(sport)
            _CACHE_LOADED.pop(key, None)
            _STATS_CACHE.pop(key, None)
        else:
            _CACHE_LOADED.clear()
            _STATS_CACHE.clear()


# ── Cache loading ─────────────────────────────────────────────────────────────

async def _ensure_cache_loaded(sport_key: str) -> None:
    async with _CACHE_LOCK:
        if not _CACHE_LOADED.get(sport_key):
            await _load_stats(sport_key)
            _CACHE_LOADED[sport_key] = True


async def _load_stats(sport_key: str) -> None:
    try:
        import httpx  # defer import so module is importable without httpx
        async with httpx.AsyncClient(timeout=15, headers=_ESPN_HEADERS) as client:
            if sport_key == "basketball_nba":
                await _load_nba(client)
            elif sport_key == "americanfootball_nfl":
                await _load_nfl(client)
        n = len(_STATS_CACHE.get(sport_key, {}))
        logger.info("ESPN stats loaded: %d %s teams", n, sport_key)
    except Exception as exc:
        logger.warning("ESPN stats unavailable for %s (%s); using fallbacks", sport_key, exc)
        _STATS_CACHE.setdefault(sport_key, {})


# ── NBA standings loader ──────────────────────────────────────────────────────

async def _load_nba(client) -> None:
    """Fetch NBA standings and populate team stats cache."""
    resp = await client.get(
        f"{_ESPN_NBA_BASE}/standings",
        params={"season": _NBA_SEASON, "seasontype": 2},
    )
    resp.raise_for_status()
    data = resp.json()

    cache: dict[str, TeamStats] = {}

    # ESPN NBA standings structure:
    # data.children[] (conferences) → standings.entries[] → team + stats[]
    for conf in data.get("children", []):
        entries = conf.get("standings", {}).get("entries", [])
        for entry in entries:
            ts = _parse_nba_entry(entry)
            if ts:
                cache[ts.name] = ts
                short = entry.get("team", {}).get("shortDisplayName", "")
                abbrev = entry.get("team", {}).get("abbreviation", "")
                if short and short != ts.name:
                    cache[short] = ts
                if abbrev and abbrev != ts.name:
                    cache[abbrev] = ts

    _STATS_CACHE["basketball_nba"] = cache


def _parse_nba_entry(entry: dict) -> Optional[TeamStats]:
    team = entry.get("team", {})
    name = team.get("displayName", "")
    if not name:
        return None

    stats = {s["name"]: float(s.get("value", 0)) for s in entry.get("stats", [])}

    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    gp = wins + losses

    # ESPN may report season totals or per-game averages depending on season type
    ppg = stats.get("avgPointsFor") or stats.get("ppg", 0)
    papg = stats.get("avgPointsAgainst") or stats.get("oppg", 0)

    # If per-game fields are zero, try deriving from totals
    if ppg == 0 and gp > 0:
        ppg = stats.get("pointsFor", 0) / gp
    if papg == 0 and gp > 0:
        papg = stats.get("pointsAgainst", 0) / gp

    # Sanity check: NBA teams average 100-130 pts/game
    if ppg < 70 or papg < 70:
        ppg = _NBA_LEAGUE_AVG.points_per_game
        papg = _NBA_LEAGUE_AVG.points_allowed_per_game

    return TeamStats(
        name=name,
        points_per_game=round(ppg, 2),
        points_allowed_per_game=round(papg, 2),
        win_pct=round(wins / gp, 3) if gp > 0 else 0.500,
        games_played=int(gp),
    )


# ── NFL standings loader ──────────────────────────────────────────────────────

async def _load_nfl(client) -> None:
    """Fetch NFL standings and populate team stats cache."""
    resp = await client.get(
        f"{_ESPN_NFL_BASE}/standings",
        params={"season": _NFL_SEASON, "seasontype": 2},
    )
    resp.raise_for_status()
    data = resp.json()

    cache: dict[str, TeamStats] = {}

    # NFL: data.children[] (conferences) → children[] (divisions) → standings.entries[]
    for conf in data.get("children", []):
        for division in conf.get("children", []):
            entries = division.get("standings", {}).get("entries", [])
            for entry in entries:
                ts = _parse_nfl_entry(entry)
                if ts:
                    cache[ts.name] = ts
                    short = entry.get("team", {}).get("shortDisplayName", "")
                    abbrev = entry.get("team", {}).get("abbreviation", "")
                    if short and short != ts.name:
                        cache[short] = ts
                    if abbrev and abbrev != ts.name:
                        cache[abbrev] = ts

    _STATS_CACHE["americanfootball_nfl"] = cache


def _parse_nfl_entry(entry: dict) -> Optional[TeamStats]:
    team = entry.get("team", {})
    name = team.get("displayName", "")
    if not name:
        return None

    stats = {s["name"]: float(s.get("value", 0)) for s in entry.get("stats", [])}

    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    gp = wins + losses

    ppg = stats.get("avgPointsFor") or stats.get("pointsFor", 0) / gp if gp else 0
    papg = stats.get("avgPointsAgainst") or stats.get("pointsAgainst", 0) / gp if gp else 0

    if ppg == 0 and gp > 0:
        ppg = stats.get("pointsFor", 0) / gp
    if papg == 0 and gp > 0:
        papg = stats.get("pointsAgainst", 0) / gp

    if ppg < 5 or papg < 5:
        ppg = _NFL_LEAGUE_AVG.points_per_game
        papg = _NFL_LEAGUE_AVG.points_allowed_per_game

    return TeamStats(
        name=name,
        points_per_game=round(ppg, 2),
        points_allowed_per_game=round(papg, 2),
        win_pct=round(wins / gp, 3) if gp > 0 else 0.500,
        games_played=int(gp),
    )


# ── Matching helpers ──────────────────────────────────────────────────────────

def _normalise_sport(sport: str) -> str:
    s = sport.lower()
    if "nba" in s or "basketball" in s:
        return "basketball_nba"
    if "nfl" in s or "football" in s:
        return "americanfootball_nfl"
    return sport


def _exact_match(name: str, cache: dict[str, TeamStats]) -> Optional[TeamStats]:
    return cache.get(name)


def _partial_match(name: str, cache: dict[str, TeamStats]) -> Optional[TeamStats]:
    name_l = name.lower()
    for key, val in cache.items():
        key_l = key.lower()
        if name_l in key_l or key_l in name_l:
            return val
    return None


def _word_overlap_match(name: str, cache: dict[str, TeamStats]) -> Optional[TeamStats]:
    """Return the entry whose key shares the most words with *name*."""
    words = set(name.lower().split())
    best: Optional[TeamStats] = None
    best_score = 0
    for key, val in cache.items():
        score = len(words & set(key.lower().split()))
        if score > best_score:
            best_score = score
            best = val
    return best if best_score >= 1 else None
