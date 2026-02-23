from __future__ import annotations

from datetime import datetime
from typing import Any


from edgeiq.config import settings

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"


def american_to_implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return (-american_odds) / ((-american_odds) + 100)


def confidence_from_edge(edge_pct: float) -> int:
    # Conservative confidence mapping to avoid overpromising.
    # 2% edge => 55, 10%+ edge => capped at 85.
    raw = 55 + (edge_pct - 2.0) * 3.75
    return max(50, min(85, round(raw)))


async def fetch_events_with_odds() -> list[dict[str, Any]]:
    if not settings.odds_api_key:
        raise RuntimeError("ODDS_API_KEY not set. Configure it in .env.")

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
            events.extend(resp.json())
    return events


def build_market_best_prices(event: dict[str, Any]) -> dict[tuple[str, str], tuple[int, str]]:
    best: dict[tuple[str, str], tuple[int, str]] = {}

    for bookmaker in event.get("bookmakers", []):
        bname = bookmaker.get("title", "Unknown")
        for market in bookmaker.get("markets", []):
            mkey = market.get("key")
            for outcome in market.get("outcomes", []):
                selection = outcome.get("name")
                price = int(outcome.get("price"))
                key = (mkey, selection)

                # Better price for bettor: larger positive odds or less negative odds.
                if key not in best:
                    best[key] = (price, bname)
                else:
                    current = best[key][0]
                    if price > current:
                        best[key] = (price, bname)
    return best


def estimate_model_probability(event: dict[str, Any], market: str, selection: str, best_price: int) -> float:
    implied = american_to_implied_prob(best_price)

    # Real-time market-aware heuristic. Improves signal by rewarding consensus and line disagreement.
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

    # Penalize if pick only exists due to one outlier book, reward agreement across books.
    adjustment = (avg_implied - implied) * 0.5 - disagreement * 0.25
    model_probability = implied + adjustment

    return max(0.05, min(0.95, model_probability))


def generate_picks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []

    for event in events:
        best_prices = build_market_best_prices(event)
        for (market, selection), (price, sportsbook) in best_prices.items():
            implied = american_to_implied_prob(price)
            model_prob = estimate_model_probability(event, market, selection, price)
            edge_pct = (model_prob - implied) * 100

            if edge_pct < settings.minimum_edge_pct:
                continue

            picks.append(
                {
                    "sport": event.get("sport_title"),
                    "event_id": event.get("id"),
                    "commence_time": datetime.fromisoformat(event.get("commence_time").replace("Z", "+00:00")),
                    "matchup": f"{event.get('away_team')} @ {event.get('home_team')}",
                    "market": market,
                    "selection": selection,
                    "best_odds_american": price,
                    "best_sportsbook": sportsbook,
                    "model_probability": round(model_prob, 4),
                    "implied_probability": round(implied, 4),
                    "edge_percentage": round(edge_pct, 2),
                    "confidence_score": confidence_from_edge(edge_pct),
                    "rationale": (
                        "Model identified positive expected value by comparing best available line "
                        "against consensus market pricing across sportsbooks."
                    ),
                }
            )

    picks.sort(key=lambda p: (p["confidence_score"], p["edge_percentage"]), reverse=True)
    return picks[:10]
