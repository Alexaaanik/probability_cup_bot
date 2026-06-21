"""Persist decisions waiting for The Odds API match lines."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import LOGS_DIR, PENDING_DECISIONS_FILENAME

logger = logging.getLogger(__name__)


@dataclass
class PendingRecord:
    """Market decision saved until an Odds API line becomes available."""

    market_id: str
    match_id: str
    lobby_id: str
    match_name: str
    question: str
    market_status: str
    match_opening_time: str
    base_probability: float
    adjustment: float
    adjustment_reason: str
    adjusted_probability: float
    shrunk_probability: float
    api_probability: int
    llm_called: bool
    llm_reason: str
    created_at: str
    updated_at: str


def pending_path() -> Path:
    return LOGS_DIR / PENDING_DECISIONS_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pending() -> dict[str, PendingRecord]:
    """Load pending records keyed by market_id (latest line wins)."""
    path = pending_path()
    if not path.exists():
        return {}

    records: dict[str, PendingRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            record = PendingRecord(**data)
            records[record.market_id] = record
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Skipping invalid pending line: %s", exc)
    return records


def save_pending(records: dict[str, PendingRecord]) -> None:
    """Rewrite the pending file from the in-memory map."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = pending_path()
    with path.open("w", encoding="utf-8") as fh:
        for record in sorted(records.values(), key=lambda item: item.updated_at):
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def upsert_pending(record: PendingRecord, records: dict[str, PendingRecord]) -> None:
    existing = records.get(record.market_id)
    if existing is not None:
        record.created_at = existing.created_at
    records[record.market_id] = record


def remove_pending(market_id: str, records: dict[str, PendingRecord]) -> None:
    records.pop(market_id, None)


def record_from_decision(
    *,
    market_id: str,
    match_id: str,
    lobby_id: str,
    match_name: str,
    question: str,
    market_status: str,
    match_opening_time: str,
    base_probability: float,
    adjustment: float,
    adjustment_reason: str,
    adjusted_probability: float,
    shrunk_probability: float,
    api_probability: int,
    llm_called: bool,
    llm_reason: str,
) -> PendingRecord:
    now = _now_iso()
    return PendingRecord(
        market_id=market_id,
        match_id=match_id,
        lobby_id=lobby_id,
        match_name=match_name,
        question=question,
        market_status=market_status,
        match_opening_time=match_opening_time,
        base_probability=base_probability,
        adjustment=adjustment,
        adjustment_reason=adjustment_reason,
        adjusted_probability=adjusted_probability,
        shrunk_probability=shrunk_probability,
        api_probability=api_probability,
        llm_called=llm_called,
        llm_reason=llm_reason,
        created_at=now,
        updated_at=now,
    )
