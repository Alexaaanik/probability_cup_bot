"""Idempotent create/update submission to SportsPredict API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from config import PREDICTIONS_BATCH_SIZE, RATE_LIMIT_DELAY_SEC
from sportspredict_client import SpPredictionPayload, SportsPredictClient

logger = logging.getLogger(__name__)

_CONFLICT_ERROR_MARKERS = (
    "already exists",
    "already predicted",
)
_MARKET_CLOSED_MARKERS = (
    "market closed",
    "market has closed",
    "closing_time",
    "already closed",
    "cannot update",
    "locked",
)


@dataclass(frozen=True)
class MarketSubmissionOutcome:
    market_id: str
    action: str
    status: str
    update_count: int = 0
    error: str = ""


def is_conflict_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _CONFLICT_ERROR_MARKERS)


def is_market_closed_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _MARKET_CLOSED_MARKERS)


def classify_patch_failure(exc: httpx.HTTPStatusError) -> str:
    message = ""
    try:
        payload = exc.response.json()
        message = str(payload.get("message", ""))
    except Exception:
        message = exc.response.text[:200]

    combined = f"{exc.response.status_code} {message}".lower()
    if is_market_closed_error(combined):
        return "update_failed_market_closed"
    return "update_failed"


def load_update_counts(log_path: Path) -> dict[str, int]:
    """How many times each market was successfully updated in prior runs."""
    if not log_path.exists():
        return {}

    counts: dict[str, int] = {}
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            market_id = record.get("market_id")
            if not market_id:
                continue
            if record.get("submission_action") == "update" and record.get(
                "submission_status"
            ) == "updated":
                counts[market_id] = counts.get(market_id, 0) + 1
    return counts


def split_predictions_by_existing(
    predictions: list[SpPredictionPayload],
    existing_by_market: dict[str, str],
) -> tuple[list[SpPredictionPayload], list[tuple[str, SpPredictionPayload]]]:
    creates: list[SpPredictionPayload] = []
    updates: list[tuple[str, SpPredictionPayload]] = []
    for payload in predictions:
        prediction_id = existing_by_market.get(payload.market_id)
        if prediction_id:
            updates.append((prediction_id, payload))
        else:
            creates.append(payload)
    return creates, updates


def _record_batch_item(
    item: dict[str, Any],
    *,
    update_counts: dict[str, int],
    outcomes: dict[str, MarketSubmissionOutcome],
) -> None:
    market_id = str(item.get("market_id", ""))
    if not market_id:
        return

    if item.get("success"):
        outcomes[market_id] = MarketSubmissionOutcome(
            market_id=market_id,
            action="create",
            status="created",
            update_count=0,
        )
        return

    error = str(item.get("error", ""))
    outcomes[market_id] = MarketSubmissionOutcome(
        market_id=market_id,
        action="create",
        status="create_failed",
        update_count=0,
        error=error,
    )


def submit_predictions_idempotent(
    predictions: list[SpPredictionPayload],
    *,
    sp_client: SportsPredictClient,
    lobby_id: str,
    dry_run: bool,
    update_counts: dict[str, int],
) -> dict[str, MarketSubmissionOutcome]:
    """Create new predictions via batch POST, update existing via PATCH."""
    outcomes: dict[str, MarketSubmissionOutcome] = {}

    if not predictions:
        return outcomes

    if dry_run:
        for payload in predictions:
            outcomes[payload.market_id] = MarketSubmissionOutcome(
                market_id=payload.market_id,
                action="dry_run",
                status="dry_run",
                update_count=update_counts.get(payload.market_id, 0),
            )
        logger.info("[dry-run] Would submit %d predictions (create/update split skipped)", len(predictions))
        for payload in predictions:
            logger.info(
                "  market_id=%s lobby_id=%s probability=%d",
                payload.market_id,
                payload.lobby_id,
                payload.probability,
            )
        return outcomes

    existing_by_market = sp_client.get_predictions_by_market(lobby_id)
    creates, updates = split_predictions_by_existing(predictions, existing_by_market)
    logger.info(
        "Submission plan: %d create(s), %d update(s), %d existing in lobby",
        len(creates),
        len(updates),
        len(existing_by_market),
    )

    for batch_start in range(0, len(creates), PREDICTIONS_BATCH_SIZE):
        batch = creates[batch_start : batch_start + PREDICTIONS_BATCH_SIZE]
        try:
            result = sp_client.submit_predictions_batch(batch)
            logger.info(
                "Create batch %d–%d: total=%s succeeded=%s failed=%s",
                batch_start + 1,
                batch_start + len(batch),
                result.get("total"),
                result.get("succeeded"),
                result.get("failed"),
            )
        except Exception as exc:
            logger.error("Create batch %d–%d failed: %s", batch_start + 1, batch_start + len(batch), exc)
            for payload in batch:
                outcomes[payload.market_id] = MarketSubmissionOutcome(
                    market_id=payload.market_id,
                    action="create",
                    status="create_failed",
                    error=str(exc),
                )
            continue

        for item in result.get("results", []):
            _record_batch_item(item, update_counts=update_counts, outcomes=outcomes)

        conflict_items = [
            item
            for item in result.get("results", [])
            if not item.get("success") and is_conflict_error(str(item.get("error", "")))
        ]
        if conflict_items:
            refreshed = sp_client.get_predictions_by_market(lobby_id)
            for item in conflict_items:
                market_id = str(item["market_id"])
                payload = next(p for p in batch if p.market_id == market_id)
                prediction_id = refreshed.get(market_id)
                if prediction_id is None:
                    logger.error(
                        "Conflict on market %s but prediction id not found via GET /predictions",
                        market_id,
                    )
                    continue
                updates.append((prediction_id, payload))
                outcomes.pop(market_id, None)

    for index, (prediction_id, payload) in enumerate(updates):
        if index > 0:
            time.sleep(RATE_LIMIT_DELAY_SEC)
        try:
            sp_client.update_prediction(prediction_id, payload.probability)
        except httpx.HTTPStatusError as exc:
            status = classify_patch_failure(exc)
            error_text = ""
            try:
                error_text = str(exc.response.json().get("message", exc.response.text[:200]))
            except Exception:
                error_text = exc.response.text[:200]
            logger.warning(
                "PATCH failed for market %s (prediction %s): %s — %s",
                payload.market_id,
                prediction_id,
                status,
                error_text,
            )
            outcomes[payload.market_id] = MarketSubmissionOutcome(
                market_id=payload.market_id,
                action="update",
                status=status,
                update_count=update_counts.get(payload.market_id, 0),
                error=error_text,
            )
            continue
        except Exception as exc:
            logger.error(
                "PATCH failed for market %s (prediction %s): %s",
                payload.market_id,
                prediction_id,
                exc,
            )
            outcomes[payload.market_id] = MarketSubmissionOutcome(
                market_id=payload.market_id,
                action="update",
                status="update_failed",
                update_count=update_counts.get(payload.market_id, 0),
                error=str(exc),
            )
            continue

        new_count = update_counts.get(payload.market_id, 0) + 1
        outcomes[payload.market_id] = MarketSubmissionOutcome(
            market_id=payload.market_id,
            action="update",
            status="updated",
            update_count=new_count,
        )
        logger.info(
            "Updated market %s → %d%% (prediction %s, update #%d)",
            payload.market_id,
            payload.probability,
            prediction_id,
            new_count,
        )

    return outcomes
