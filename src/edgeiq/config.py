from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "EdgeIQ API"
    odds_api_key: str | None = None
    odds_regions: str = "us"
    odds_markets: str = "h2h,spreads,totals"
    odds_sports: str = "americanfootball_nfl,basketball_nba"
    minimum_edge_pct: float = 2.0


settings = Settings()
