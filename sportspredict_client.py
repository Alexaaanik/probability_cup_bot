"""SportsPredict API client — Probability Cup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from config import PROBABILITY_CUP_TITLE_KEYWORD, SPORTSPREDICT_BASE_URL
from utils import request_with_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpEvent:
    id: str
    title: str
    status: str


@dataclass(frozen=True)
class SpLobby:
    id: str
    name: str
    joined: bool


@dataclass(frozen=True)
class SpMatch:
    id: str
    name: str
    opening_time: datetime
    open_market_count: int


@dataclass(frozen=True)
class SpMarket:
    id: str
    question: str
    status: str
    lobby_id: str
    match_id: str
    match_name: str
    opening_time: datetime


@dataclass(frozen=True)
class SpPredictionPayload:
    market_id: str
    lobby_id: str
    probability: int


class SportsPredictClient:
    def __init__(self, api_key: str) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        url = f"{SPORTSPREDICT_BASE_URL}{path}"
        response = request_with_retry("GET", url, headers=self._headers, params=params)
        return response.json()

    def _post(self, path: str, json_body: dict[str, Any]) -> Any:
        url = f"{SPORTSPREDICT_BASE_URL}{path}"
        response = request_with_retry(
            "POST", url, headers=self._headers, json_body=json_body
        )
        return response.json() if response.content else {}

    def find_probability_cup_event(self) -> SpEvent:
        """Find the Probability Cup event by title (API type field is a UUID)."""
        events: list[dict[str, Any]] = self._get("/events")
        for event in events:
            title = str(event.get("title", "")).lower()
            if PROBABILITY_CUP_TITLE_KEYWORD in title:
                return SpEvent(
                    id=event["id"],
                    title=event["title"],
                    status=event.get("status", ""),
                )
        raise LookupError(
            f"Probability Cup event not found among {len(events)} events"
        )

    def get_lobbies(self, event_id: str) -> list[SpLobby]:
        raw = self._get("/lobbies", params={"event_id": event_id})
        return [
            SpLobby(
                id=item["id"],
                name=item["name"],
                joined=bool(item.get("joined")),
            )
            for item in raw
        ]

    def join_lobby(self, lobby_id: str) -> None:
        """Join a lobby. 409 if already a member is fine."""
        url = f"{SPORTSPREDICT_BASE_URL}/lobbies/{lobby_id}/join"
        try:
            request_with_retry("POST", url, headers=self._headers, json_body={})
            logger.info("Joined lobby %s", lobby_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                logger.info("Already in lobby %s (409), skipping", lobby_id)
                return
            raise

    def get_matches(self, event_id: str) -> list[SpMatch]:
        raw = self._get("/matches", params={"event_id": event_id})
        matches: list[SpMatch] = []
        for item in raw:
            opening_raw = item["opening_time"].replace("Z", "+00:00")
            matches.append(
                SpMatch(
                    id=item["id"],
                    name=item["name"],
                    opening_time=datetime.fromisoformat(opening_raw),
                    open_market_count=int(item.get("open_market_count", 0)),
                )
            )
        return matches

    def get_markets(self, lobby_id: str, match_id: str) -> list[SpMarket]:
        raw = self._get(
            "/markets",
            params={"lobby_id": lobby_id, "match_id": match_id},
        )
        markets: list[SpMarket] = []
        for item in raw:
            match = item.get("match") or {}
            opening_raw = match.get("opening_time", "").replace("Z", "+00:00")
            markets.append(
                SpMarket(
                    id=item["id"],
                    question=item["question"],
                    status=item.get("status", ""),
                    lobby_id=item.get("lobby_id", lobby_id),
                    match_id=match.get("id", match_id),
                    match_name=match.get("name", ""),
                    opening_time=datetime.fromisoformat(opening_raw),
                )
            )
        return markets

    def submit_predictions_batch(
        self, predictions: list[SpPredictionPayload]
    ) -> Any:
        body = {
            "predictions": [
                {
                    "market_id": p.market_id,
                    "lobby_id": p.lobby_id,
                    "probability": p.probability,
                }
                for p in predictions
            ]
        }
        url = f"{SPORTSPREDICT_BASE_URL}/predictions/batch"
        response = request_with_retry(
            "POST", url, headers=self._headers, json_body=body
        )
        return response.json() if response.content else {}

    def get_results(self, lobby_id: str) -> list[dict[str, Any]]:
        return self._get("/results", params={"lobby_id": lobby_id})
