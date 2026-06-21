"""Unit tests for idempotent prediction submission helpers."""

from __future__ import annotations

import json
from pathlib import Path

from prediction_submit import (
    is_conflict_error,
    is_market_closed_error,
    load_update_counts,
    split_predictions_by_existing,
)
from sportspredict_client import SpPredictionPayload


def test_is_conflict_error() -> None:
    assert is_conflict_error("Prediction for this market already exists in this lobby")
    assert not is_conflict_error("Market not found")


def test_is_market_closed_error() -> None:
    assert is_market_closed_error("Market has closed")
    assert not is_market_closed_error("probability must not be greater than 99")


def test_split_predictions_by_existing() -> None:
    payloads = [
        SpPredictionPayload(market_id="m1", lobby_id="l1", probability=55),
        SpPredictionPayload(market_id="m2", lobby_id="l1", probability=60),
    ]
    creates, updates = split_predictions_by_existing(payloads, {"m1": "pred-1"})
    assert len(creates) == 1
    assert creates[0].market_id == "m2"
    assert updates == [("pred-1", payloads[0])]


def test_load_update_counts(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    rows = [
        {"market_id": "m1", "submission_action": "create", "submission_status": "created"},
        {"market_id": "m1", "submission_action": "update", "submission_status": "updated"},
        {"market_id": "m1", "submission_action": "update", "submission_status": "updated"},
        {"market_id": "m2", "submission_action": "update", "submission_status": "update_failed"},
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    counts = load_update_counts(log_path)
    assert counts == {"m1": 2}
