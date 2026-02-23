from datetime import datetime
from pydantic import BaseModel, Field


class OddsQuote(BaseModel):
    sportsbook: str
    market: str
    selection: str
    price_american: int
    implied_probability: float


class Pick(BaseModel):
    sport: str
    event_id: str
    commence_time: datetime
    matchup: str
    market: str
    selection: str
    best_odds_american: int
    best_sportsbook: str
    model_probability: float = Field(ge=0, le=1)
    implied_probability: float = Field(ge=0, le=1)
    edge_percentage: float
    confidence_score: int = Field(ge=0, le=100)
    rationale: str


class DailyPicksResponse(BaseModel):
    generated_at: datetime
    picks: list[Pick]
