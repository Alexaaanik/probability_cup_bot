"""Клиент The Odds API — коэффициенты h2h на матчи ЧМ."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import (
    MIN_BOOKMAKERS_FOR_ODDS,
    ODDS_API_BASE_URL,
    ODDS_FORMAT,
    ODDS_MARKETS,
    ODDS_REGIONS,
    ODDS_SPORT_KEY,
)
from utils import request_with_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutcomeOdds:
    """Средний decimal-коэффициент по всем букмекерам для одного исхода."""

    name: str
    average_odds: float
    bookmaker_count: int


@dataclass(frozen=True)
class OddsMatch:
    """Матч с усреднёнными коэффициентами h2h (дом / ничья / гости)."""

    id: str
    home_team: str
    away_team: str
    commence_time: datetime
    outcomes: tuple[OutcomeOdds, ...]


class OddsApiClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def fetch_world_cup_odds(self) -> list[OddsMatch]:
        """GET /sports/soccer_fifa_world_cup/odds — список матчей с h2h."""
        url = f"{ODDS_API_BASE_URL}/sports/{ODDS_SPORT_KEY}/odds"
        params = {
            "regions": ODDS_REGIONS,
            "markets": ODDS_MARKETS,
            "oddsFormat": ODDS_FORMAT,
            "apiKey": self._api_key,
        }
        response = request_with_retry("GET", url, params=params)
        raw_matches: list[dict[str, Any]] = response.json()
        logger.info("The Odds API: получено %d матчей", len(raw_matches))

        parsed: list[OddsMatch] = []
        for item in raw_matches:
            match = self._parse_match(item)
            if match is not None:
                parsed.append(match)
        return parsed

    def _parse_match(self, item: dict[str, Any]) -> OddsMatch | None:
        bookmakers = item.get("bookmakers") or []
        if len(bookmakers) < MIN_BOOKMAKERS_FOR_ODDS:
            logger.debug(
                "Пропуск матча %s vs %s: недостаточно букмекеров (%d)",
                item.get("home_team"),
                item.get("away_team"),
                len(bookmakers),
            )
            return None

        outcome_odds: dict[str, list[float]] = {}
        for bookmaker in bookmakers:
            for market in bookmaker.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes") or []:
                    name = str(outcome["name"])
                    price = float(outcome["price"])
                    outcome_odds.setdefault(name, []).append(price)

        if not outcome_odds:
            return None

        outcomes = tuple(
            OutcomeOdds(
                name=name,
                average_odds=sum(prices) / len(prices),
                bookmaker_count=len(prices),
            )
            for name, prices in outcome_odds.items()
        )

        commence_raw = item["commence_time"].replace("Z", "+00:00")
        return OddsMatch(
            id=item["id"],
            home_team=item["home_team"],
            away_team=item["away_team"],
            commence_time=datetime.fromisoformat(commence_raw),
            outcomes=outcomes,
        )
