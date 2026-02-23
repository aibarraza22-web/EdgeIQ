"""Odds fetching, Monte Carlo-powered probability estimation, and pick generation.

Pipeline
--------
1. Fetch live odds from The Odds API for each configured sport.
2. For each event, load team season-average stats from ESPN (cached).
3. Run a 10 000-iteration Monte Carlo simulation to get win/spread/total
   probability distributions.
4. Compare MC probabilities to bookmaker-implied probabilities to find edge.
5. Return the top picks sorted by confidence and edge.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from edgeiq.config import settings
from edgeiq.simulator import (
    MonteCarloResult,
    run_monte_carlo,
    spread_cover_probability,
    total_cover_probability,
)
from edgeiq.team_stats import get_team_stats

logger = logging.getLogger(__name__)

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

_N_SIMS = 10_000


# ── Utility functions ─────────────────────────────────────────────────────────

def american_to_implied_prob(american_odds: int) -> float:
    """Convert American-format odds to implied (overround-inclusive) probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return (-american_odds) / ((-american_odds) + 100)


def confidence_from_edge(edge_pct: float) -> int:
    """Map an edge percentage to a conservative 50–85 confidence score.

    2 % edge → 55;  10 %+ edge → capped at 85.
    """
    raw = 55 + (edge_pct - 2.0) * 3.75
    return max(50, min(85, round(raw)))


# ── Odds API fetch ────────────────────────────────────────────────────────────

async def fetch_events_with_odds() -> list[dict[str, Any]]:
    """Fetch live events + odds from The Odds API for all configured sports."""
    if not settings.odds_api_key:
        raise RuntimeError(
            "ODDS_API_KEY not set.  Add it to your .env file.  "
            "Get a free key at https://the-odds-api.com/"
        )

    sports = [s.strip() for s in settings.odds_sports.split(",") if s.strip()]
    events: list[dict[str, Any]] = []

    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        for sport in sports:
            resp = await client.get(
                f"{ODDS_BASE_URL}/sports/{sport}/odds",
                params={
                    "apiKey": settings.odds_api_key,
                    "regions": settings.odds_regions,
                    "markets": settings.odds_markets,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
            )
            resp.raise_for_status()
            for event in resp.json():
                # Embed the sport key so downstream code knows which sport it is
                event.setdefault("sport_key", sport)
            events.extend(resp.json())

    return events


# ── Market helpers ────────────────────────────────────────────────────────────

def build_market_best_prices(
    event: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the best (highest) price per (market, selection) across bookmakers.

    Returns a mapping of ``(market_key, selection_name)`` →
    ``{"price": int, "sportsbook": str, "point": float | None}``.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}

    for bookmaker in event.get("bookmakers", []):
        bname = bookmaker.get("title", "Unknown")
        for market in bookmaker.get("markets", []):
            mkey = market.get("key")
            for outcome in market.get("outcomes", []):
                selection = outcome.get("name")
                price = int(outcome.get("price"))
                point: Optional[float] = outcome.get("point")
                key = (mkey, selection)

                if key not in best or price > best[key]["price"]:
                    best[key] = {"price": price, "sportsbook": bname, "point": point}

    return best


def _consensus_model_probability(
    event: dict[str, Any], market: str, selection: str, best_price: int
) -> float:
    """Cross-book consensus heuristic (fallback when MC stats are unavailable)."""
    implied = american_to_implied_prob(best_price)
    market_prices: list[int] = []
    for bookmaker in event.get("bookmakers", []):
        for m in bookmaker.get("markets", []):
            if m.get("key") != market:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") == selection:
                    market_prices.append(int(o.get("price")))

    if not market_prices:
        return implied

    avg_implied = sum(american_to_implied_prob(p) for p in market_prices) / len(market_prices)
    disagreement = abs(implied - avg_implied)
    adjustment = (avg_implied - implied) * 0.5 - disagreement * 0.25
    return max(0.05, min(0.95, implied + adjustment))


# ── Monte Carlo probability estimation ───────────────────────────────────────

def _mc_model_probability(
    mc: MonteCarloResult,
    market: str,
    selection: str,
    home_team: str,
    point: Optional[float],
) -> Optional[float]:
    """Return MC-based model probability for a given market outcome.

    Returns ``None`` when the market type is not directly supported by MC.
    """
    if market == "h2h":
        return (
            mc.home_win_probability
            if selection == home_team
            else mc.away_win_probability
        )

    if market == "spreads" and point is not None:
        return spread_cover_probability(mc, home_team, selection, point)

    if market == "totals" and point is not None:
        return total_cover_probability(mc, selection, point)

    return None


# ── Pick generation ───────────────────────────────────────────────────────────

async def generate_picks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate value picks from a list of live events.

    For each event:
    1. Load team stats and run Monte Carlo (10 000 sims).
    2. For each market outcome, compute MC-based model probability.
    3. Keep outcomes where edge ≥ ``settings.minimum_edge_pct``.
    4. Return the top 10 picks sorted by confidence × edge.
    """
    picks: list[dict[str, Any]] = []

    for event in events:
        sport = event.get("sport_key", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # ── Load team stats + run Monte Carlo ──────────────────────────────
        mc: Optional[MonteCarloResult] = None
        try:
            home_stats = await get_team_stats(home_team, sport)
            away_stats = await get_team_stats(away_team, sport)
            mc = run_monte_carlo(home_stats, away_stats, sport, n_simulations=_N_SIMS)
        except Exception as exc:
            logger.warning(
                "MC simulation failed for %s @ %s: %s — falling back to consensus",
                away_team, home_team, exc,
            )

        best_prices = build_market_best_prices(event)

        for (market, selection), info in best_prices.items():
            price: int = info["price"]
            sportsbook: str = info["sportsbook"]
            point: Optional[float] = info["point"]
            implied = american_to_implied_prob(price)

            # Prefer MC probability; fall back to cross-book consensus
            if mc is not None:
                model_prob = _mc_model_probability(mc, market, selection, home_team, point)
                if model_prob is None:
                    model_prob = _consensus_model_probability(event, market, selection, price)
            else:
                model_prob = _consensus_model_probability(event, market, selection, price)

            edge_pct = (model_prob - implied) * 100
            if edge_pct < settings.minimum_edge_pct:
                continue

            # Build rationale
            rationale = _build_rationale(mc, market, selection, home_team, model_prob, implied)

            mc_summary = None
            if mc is not None:
                mc_summary = {
                    "n_simulations": mc.n_simulations,
                    "home_win_probability": mc.home_win_probability,
                    "away_win_probability": mc.away_win_probability,
                    "simulated_spread": mc.simulated_spread,
                    "simulated_total": mc.simulated_total,
                    "spread_std": mc.spread_std,
                    "spread_ci_low": mc.spread_ci_low,
                    "spread_ci_high": mc.spread_ci_high,
                }

            picks.append(
                {
                    "sport": event.get("sport_title", sport),
                    "event_id": event.get("id"),
                    "commence_time": datetime.fromisoformat(
                        event.get("commence_time", "").replace("Z", "+00:00")
                    ),
                    "matchup": f"{away_team} @ {home_team}",
                    "market": market,
                    "selection": selection,
                    "best_odds_american": price,
                    "best_sportsbook": sportsbook,
                    "model_probability": round(model_prob, 4),
                    "implied_probability": round(implied, 4),
                    "edge_percentage": round(edge_pct, 2),
                    "confidence_score": confidence_from_edge(edge_pct),
                    "rationale": rationale,
                    "monte_carlo": mc_summary,
                }
            )

    picks.sort(key=lambda p: (p["confidence_score"], p["edge_percentage"]), reverse=True)
    return picks[:10]


def _build_rationale(
    mc: Optional[MonteCarloResult],
    market: str,
    selection: str,
    home_team: str,
    model_prob: float,
    implied_prob: float,
) -> str:
    """Produce a human-readable explanation for a pick."""
    edge = (model_prob - implied_prob) * 100

    if mc is None:
        return (
            f"Cross-book consensus identified a {edge:.1f}% edge over the "
            f"bookmaker-implied probability of {implied_prob*100:.1f}%."
        )

    if market == "h2h":
        side = "home" if selection == home_team else "away"
        return (
            f"Monte Carlo ({mc.n_simulations:,} sims): {selection} wins "
            f"{model_prob*100:.1f}% of simulated games vs bookmaker-implied "
            f"{implied_prob*100:.1f}% — {edge:.1f}% edge.  "
            f"Projected score: {mc.home_score_mean:.0f}–{mc.away_score_mean:.0f} "
            f"({'home' if side == 'home' else 'away'} team {home_team if side == 'home' else 'away'})."
        )

    if market == "spreads":
        return (
            f"Monte Carlo spread distribution: μ={mc.simulated_spread:+.1f}, "
            f"σ={mc.spread_std:.1f} — {selection} covers at "
            f"{model_prob*100:.1f}% vs implied {implied_prob*100:.1f}% ({edge:.1f}% edge)."
        )

    if market == "totals":
        return (
            f"Monte Carlo projected total: {mc.simulated_total:.1f} pts "
            f"(σ={mc.spread_std:.1f}) — {selection} at {model_prob*100:.1f}% "
            f"vs implied {implied_prob*100:.1f}% ({edge:.1f}% edge)."
        )

    return (
        f"Model probability {model_prob*100:.1f}% vs implied "
        f"{implied_prob*100:.1f}% ({edge:.1f}% edge)."
    )


# ── Legacy synchronous wrapper (kept for backwards-compat tests) ──────────────

def estimate_model_probability(
    event: dict[str, Any], market: str, selection: str, best_price: int
) -> float:
    """Synchronous consensus fallback (used in legacy tests)."""
    return _consensus_model_probability(event, market, selection, best_price)
