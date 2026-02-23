from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from edgeiq.config import settings
from edgeiq.models import DailyPicksResponse
from edgeiq.odds import fetch_events_with_odds, generate_picks

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/picks/daily", response_model=DailyPicksResponse)
async def daily_picks() -> DailyPicksResponse:
    try:
        events = await fetch_events_with_odds()
        picks = generate_picks(events)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DailyPicksResponse(generated_at=datetime.now(timezone.utc), picks=picks)
