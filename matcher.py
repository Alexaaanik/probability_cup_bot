"""Сопоставление матчей SportsPredict и The Odds API."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from config import MATCH_DATE_TOLERANCE_DAYS, TEAM_ALIASES
from odds_client import OddsMatch
from sportspredict_client import SpMatch

logger = logging.getLogger(__name__)

_VS_PATTERN = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


def normalize_team_name(name: str) -> str:
    """Приводит название команды к канонической форме для сравнения."""
    cleaned = " ".join(name.strip().lower().split())
    return TEAM_ALIASES.get(cleaned, cleaned)


def parse_match_name(name: str) -> tuple[str, str] | None:
    """
    Разбирает «USA vs AUS» или «Team A vs Team B» на (home, away).

    SportsPredict использует home первым в name.
    """
    parts = _VS_PATTERN.split(name.strip(), maxsplit=1)
    if len(parts) != 2:
        logger.warning("Не удалось разобрать название матча: %r", name)
        return None
    return parts[0].strip(), parts[1].strip()


def _dates_compatible(sp_time: datetime, odds_time: datetime) -> bool:
    delta = abs(sp_time.date() - odds_time.date())
    return delta <= timedelta(days=MATCH_DATE_TOLERANCE_DAYS)


def find_odds_match(
    sp_match: SpMatch,
    odds_matches: list[OddsMatch],
) -> OddsMatch | None:
    """
    Ищет соответствующий матч в The Odds API.

    Критерии: совпадение home/away после нормализации + дата ±1 день.
    """
    parsed = parse_match_name(sp_match.name)
    if parsed is None:
        return None

    sp_home_raw, sp_away_raw = parsed
    sp_home = normalize_team_name(sp_home_raw)
    sp_away = normalize_team_name(sp_away_raw)

    candidates: list[OddsMatch] = []
    for odds in odds_matches:
        odds_home = normalize_team_name(odds.home_team)
        odds_away = normalize_team_name(odds.away_team)
        if odds_home == sp_home and odds_away == sp_away:
            if _dates_compatible(sp_match.opening_time, odds.commence_time):
                candidates.append(odds)

    if not candidates:
        logger.warning(
            "Нет линии в The Odds API для «%s» (норм.: %s vs %s, дата %s)",
            sp_match.name,
            sp_home,
            sp_away,
            sp_match.opening_time.date(),
        )
        return None

    if len(candidates) > 1:
        logger.warning(
            "Несколько кандидатов для «%s», берём ближайший по времени",
            sp_match.name,
        )
        candidates.sort(
            key=lambda o: abs(
                (o.commence_time - sp_match.opening_time).total_seconds()
            )
        )

    matched = candidates[0]
    logger.info(
        "Сопоставлен «%s» ↔ %s vs %s (%s)",
        sp_match.name,
        matched.home_team,
        matched.away_team,
        matched.commence_time.isoformat(),
    )
    return matched
