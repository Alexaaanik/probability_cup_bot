"""Sync GET /results with local predictions.jsonl by market_id."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config import LOGS_DIR, PREDICTION_LOG_FILENAME, RESULTS_SYNC_FILENAME
from sportspredict_client import SportsPredictClient
from utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncedResult:
    market_id: str
    match_name: str
    question: str
    predicted_probability: int
    brier_score: float
    logged_at: str
    result_fetched_at: str
    match_date: str | None = None


def load_predictions_by_market(log_path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per market_id from predictions.jsonl."""
    if not log_path.exists():
        return {}

    latest: dict[str, dict[str, Any]] = {}
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            market_id = record.get("market_id")
            if market_id:
                latest[market_id] = record
    return latest


def load_synced_results(sync_path: Path) -> dict[str, SyncedResult]:
    if not sync_path.exists():
        return {}

    results: dict[str, SyncedResult] = {}
    with sync_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sr = SyncedResult(**data)
            results[sr.market_id] = sr
    return results


def append_synced_results(new_rows: list[SyncedResult], sync_path: Path) -> int:
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_path.open("a", encoding="utf-8") as fh:
        for row in new_rows:
            fh.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
    return len(new_rows)


def _extract_brier(entry: dict[str, Any]) -> float | None:
    for key in ("brier_score", "brier", "score"):
        if key in entry and entry[key] is not None:
            return float(entry[key])
    return None


def _extract_market_id(entry: dict[str, Any]) -> str | None:
    for key in ("market_id", "id"):
        if entry.get(key):
            return str(entry[key])
    return None


def _extract_match_date(entry: dict[str, Any]) -> str | None:
    for key in ("match_date", "date", "closing_time", "opening_time"):
        value = entry.get(key)
        if value:
            return str(value)[:10]
    match = entry.get("match") or {}
    for key in ("closing_time", "opening_time", "name"):
        if match.get(key):
            return str(match[key])[:10] if "time" in key else None
    return None


def sync_results(
    sp_client: SportsPredictClient,
    lobby_id: str,
    *,
    predictions_path: Path | None = None,
    sync_path: Path | None = None,
) -> list[SyncedResult]:
    """Fetch /results, match to predictions.jsonl, append new rows."""
    predictions_path = predictions_path or (LOGS_DIR / PREDICTION_LOG_FILENAME)
    sync_path = sync_path or (LOGS_DIR / RESULTS_SYNC_FILENAME)

    predictions = load_predictions_by_market(predictions_path)
    already_synced = load_synced_results(sync_path)
    api_results: list[dict[str, Any]] = sp_client.get_results(lobby_id)

    fetched_at = datetime.utcnow().isoformat() + "Z"
    new_rows: list[SyncedResult] = []

    for entry in api_results:
        market_id = _extract_market_id(entry)
        brier = _extract_brier(entry)
        if not market_id or brier is None:
            continue
        if market_id in already_synced:
            continue

        pred = predictions.get(market_id)
        if pred is None or pred.get("api_probability") is None:
            logger.debug("No local prediction for market_id=%s", market_id)
            continue

        new_rows.append(
            SyncedResult(
                market_id=market_id,
                match_name=str(pred.get("match_name", entry.get("match_name", ""))),
                question=str(pred.get("question", "")),
                predicted_probability=int(pred["api_probability"]),
                brier_score=brier,
                logged_at=str(pred.get("logged_at", "")),
                result_fetched_at=fetched_at,
                match_date=_extract_match_date(entry),
            )
        )

    if new_rows:
        append_synced_results(new_rows, sync_path)
        logger.info("Synced %d new results", len(new_rows))
    else:
        logger.info("No new results to sync")

    return new_rows


def load_all_synced_results(sync_path: Path | None = None) -> list[SyncedResult]:
    sync_path = sync_path or (LOGS_DIR / RESULTS_SYNC_FILENAME)
    return list(load_synced_results(sync_path).values())


def filter_results_for_date(
    results: list[SyncedResult],
    target_date: date,
) -> list[SyncedResult]:
    date_str = target_date.isoformat()
    day_results: list[SyncedResult] = []
    for row in results:
        if row.match_date and row.match_date.startswith(date_str):
            day_results.append(row)
        elif row.logged_at.startswith(date_str):
            day_results.append(row)
        elif row.result_fetched_at.startswith(date_str):
            day_results.append(row)
    return day_results


def average_brier(results: list[SyncedResult]) -> float | None:
    if not results:
        return None
    return sum(r.brier_score for r in results) / len(results)


def main() -> None:
    load_dotenv()
    setup_logging()

    sp_key = os.getenv("SPORTSPREDICT_API_KEY", "").strip()
    if not sp_key:
        raise ValueError("SPORTSPREDICT_API_KEY not set")

    client = SportsPredictClient(sp_key)
    event = client.find_probability_cup_event()
    lobbies = client.get_lobbies(event.id)
    if not lobbies:
        raise RuntimeError("No lobbies found")

    sync_results(client, lobbies[0].id)


if __name__ == "__main__":
    main()
