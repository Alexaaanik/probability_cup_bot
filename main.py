"""Entry point: load matches, compute predictions, submit to API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    BOT_VERSION,
    LOGS_DIR,
    MAX_LLM_CALLS_PER_RUN,
    OPENROUTER_BASE_URL,
    PENDING_DECISIONS_FILENAME,
    PREDICTION_LOG_FILENAME,
    RATE_LIMIT_DELAY_SEC,
)
from llm_client import LlmCallBudget
from matcher import find_odds_match
from model import ModelDecision, build_decision_for_market, step1_fair_probabilities_from_odds
from notifier import send_telegram_message
from odds_client import OddsApiClient
from prediction_submit import load_update_counts, submit_predictions_idempotent
from pending_store import (
    PendingRecord,
    load_pending,
    record_from_decision,
    remove_pending,
    save_pending,
    upsert_pending,
)
from sportspredict_client import SpMatch, SpPredictionPayload, SportsPredictClient
from sync_results import sync_results
from utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    matches_processed: int = 0
    predictions_sent: int = 0
    win_predictions_sent: int = 0
    prop_predictions_sent: int = 0
    markets_pending_no_odds: int = 0
    pending_resolved: int = 0
    markets_skipped: int = 0
    llm_calls: int = 0
    llm_adjustments_applied: int = 0
    predictions_created: int = 0
    predictions_updated: int = 0
    predictions_update_failed: int = 0
    examples: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_settings() -> tuple[str, str, str | None]:
    load_dotenv()
    sp_key = os.getenv("SPORTSPREDICT_API_KEY", "").strip()
    odds_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None

    if not sp_key:
        raise ValueError("SPORTSPREDICT_API_KEY not set in .env")
    if not odds_key:
        raise ValueError("THE_ODDS_API_KEY not set in .env")

    return sp_key, odds_key, openrouter_key


def append_prediction_log(
    decisions: list[ModelDecision],
    log_path: Path,
    *,
    submission_outcomes: dict | None = None,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    submission_outcomes = submission_outcomes or {}
    with log_path.open("a", encoding="utf-8") as fh:
        for decision in decisions:
            record = asdict(decision)
            record["logged_at"] = run_ts
            outcome = submission_outcomes.get(decision.market_id)
            if outcome is not None:
                record["submission_action"] = outcome.action
                record["submission_status"] = outcome.status
                record["submission_update_count"] = outcome.update_count
                if outcome.error:
                    record["submission_error"] = outcome.error
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_pct(probability: float | None) -> str:
    if probability is None:
        return "n/a"
    return f"{round(probability * 100)}%"


def _format_adj(adjustment: float) -> str:
    points = round(adjustment * 100)
    if points > 0:
        return f"+{points}%"
    return f"{points}%"


def build_run_summary_telegram(stats: RunStats, *, max_llm_calls: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"🎯 *Probability Cup — run {now}*",
        "",
        f"Matches processed: {stats.matches_processed}",
        f"Markets predicted: {stats.predictions_sent} "
        f"(win {stats.win_predictions_sent}, prop {stats.prop_predictions_sent})",
        f"Pending (no Odds API line): {stats.markets_pending_no_odds}",
        f"Pending resolved this run: {stats.pending_resolved}",
        f"Markets skipped: {stats.markets_skipped}",
        (
            f"LLM adjustments applied: {stats.llm_adjustments_applied} "
            f"of {stats.llm_calls} calls"
        ),
        f"LLM calls used this run: {stats.llm_calls}/{max_llm_calls}",
    ]

    if stats.predictions_created:
        lines.append(f"Predictions created: {stats.predictions_created}")
    if stats.predictions_updated:
        lines.append(f"Predictions updated: {stats.predictions_updated}")
    if stats.predictions_update_failed:
        lines.append(f"Prediction updates failed: {stats.predictions_update_failed}")

    lines.extend([
        "",
        "Sample predictions:",
    ])

    if stats.examples:
        lines.extend(stats.examples)
    else:
        lines.append("— none submitted")

    lines.append("")
    if stats.errors:
        lines.append(f"Errors: {len(stats.errors)}")
        for err in stats.errors[:3]:
            lines.append(f"• {err}")
    else:
        lines.append("Errors: 0")

    return "\n".join(lines)


def _track_llm_stats(stats: RunStats, decision: ModelDecision) -> None:
    if decision.llm_called and decision.adjustment != 0.0:
        stats.llm_adjustments_applied += 1


def _add_example(stats: RunStats, decision: ModelDecision) -> None:
    if len(stats.examples) >= 5:
        return
    kind = decision.market_kind
    stats.examples.append(
        f"[{kind}] {decision.match_name}: {decision.api_probability}% "
        f"(base {_format_pct(decision.base_probability)}, "
        f"adj {_format_adj(decision.adjustment)})"
    )


def _apply_submission_stats(stats: RunStats, outcomes: dict) -> None:
    for outcome in outcomes.values():
        if outcome.status == "created":
            stats.predictions_created += 1
        elif outcome.status == "updated":
            stats.predictions_updated += 1
        elif outcome.status in {"update_failed_market_closed", "update_failed", "create_failed"}:
            if outcome.status.startswith("update_failed"):
                stats.predictions_update_failed += 1
            stats.errors.append(
                f"{outcome.market_id}: {outcome.status}"
                + (f" — {outcome.error}" if outcome.error else "")
            )


def _submit_predictions(
    predictions: list[SpPredictionPayload],
    *,
    sp_client: SportsPredictClient,
    lobby_id: str,
    dry_run: bool,
    stats: RunStats,
    log_path: Path,
) -> dict:
    update_counts = load_update_counts(log_path)
    outcomes = submit_predictions_idempotent(
        predictions,
        sp_client=sp_client,
        lobby_id=lobby_id,
        dry_run=dry_run,
        update_counts=update_counts,
    )
    _apply_submission_stats(stats, outcomes)
    return outcomes


def resolve_pending_decisions(
    *,
    pending: dict[str, PendingRecord],
    matches_by_id: dict[str, SpMatch],
    odds_matches: list,
    lobby_id: str,
    skip_llm: bool,
    llm_budget: LlmCallBudget,
    openrouter_client: OpenAI | None,
    stats: RunStats,
) -> tuple[list[ModelDecision], list[SpPredictionPayload]]:
    """Retry saved win-market decisions when an Odds API line appears."""
    resolved_decisions: list[ModelDecision] = []
    predictions: list[SpPredictionPayload] = []

    for market_id, record in list(pending.items()):
        sp_match = matches_by_id.get(record.match_id)
        if sp_match is None:
            logger.info(
                "Pending market %s: match %s not in current event — keeping",
                market_id,
                record.match_id,
            )
            continue

        if sp_match.open_market_count == 0:
            remove_pending(market_id, pending)
            logger.info("Pending market %s removed — match has no open markets", market_id)
            continue

        odds_match = find_odds_match(sp_match, odds_matches)
        if odds_match is None:
            continue

        fair_probs = step1_fair_probabilities_from_odds(odds_match)
        decision = build_decision_for_market(
            market_id=record.market_id,
            match_name=record.match_name,
            question=record.question,
            market_status=record.market_status,
            fair_probs=fair_probs,
            skip_llm=skip_llm,
            llm_budget=llm_budget,
            openrouter_client=openrouter_client,
        )
        resolved_decisions.append(decision)

        if decision.skipped or decision.pending:
            logger.info(
                "Pending market %s still not ready: %s",
                market_id,
                decision.skip_reason or "pending",
            )
            continue

        remove_pending(market_id, pending)
        stats.pending_resolved += 1
        _track_llm_stats(stats, decision)
        predictions.append(
            SpPredictionPayload(
                market_id=decision.market_id,
                lobby_id=lobby_id,
                probability=decision.api_probability,
            )
        )
        if decision.market_kind == "win":
            stats.win_predictions_sent += 1
        else:
            stats.prop_predictions_sent += 1
        _add_example(stats, decision)
        logger.info(
            "RESOLVED pending [%s] %s → %d%%",
            record.match_name,
            record.question[:50],
            decision.api_probability,
        )

    return resolved_decisions, predictions


def run(*, dry_run: bool, skip_llm: bool, max_matches: int | None = None) -> int:
    setup_logging()
    logger.info("Probability Cup Bot v%s", BOT_VERSION)

    stats = RunStats()
    sp_key, odds_key, openrouter_key = load_settings()
    sp_client = SportsPredictClient(sp_key)
    odds_client = OddsApiClient(odds_key)
    llm_budget = LlmCallBudget(max_calls=MAX_LLM_CALLS_PER_RUN)
    openrouter_client: OpenAI | None = None
    logger.info(
        "LLM call budget for this run: %d/%d available (in-memory, resets each start)",
        llm_budget.remaining,
        llm_budget.max_calls,
    )

    if skip_llm:
        logger.info("--skip-llm: LLM adjustment disabled")
    elif openrouter_key:
        openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_key,
        )
    else:
        logger.warning("OPENROUTER_API_KEY not set — LLM disabled")

    event = sp_client.find_probability_cup_event()
    logger.info("Event: %s (%s)", event.title, event.id)

    lobbies = sp_client.get_lobbies(event.id)
    if not lobbies:
        logger.error("No lobby for event_id=%s", event.id)
        stats.errors.append("no lobby found for event")
        return 1

    lobby = lobbies[0]
    logger.info("Lobby: %s (joined=%s)", lobby.name, lobby.joined)

    if not lobby.joined and not dry_run:
        sp_client.join_lobby(lobby.id)
    elif not lobby.joined:
        logger.info("[dry-run] Skipping lobby join %s", lobby.id)

    matches = sp_client.get_matches(event.id)
    if max_matches is not None:
        matches = matches[:max_matches]
        logger.info("--max-matches=%d", max_matches)
    logger.info("Matches to process: %d", len(matches))

    matches_by_id = {match.id: match for match in matches}
    odds_matches = odds_client.fetch_world_cup_odds()
    pending = load_pending()
    logger.info("Pending decisions loaded: %d", len(pending))

    all_decisions: list[ModelDecision] = []
    predictions: list[SpPredictionPayload] = []

    resolved_decisions, resolved_predictions = resolve_pending_decisions(
        pending=pending,
        matches_by_id=matches_by_id,
        odds_matches=odds_matches,
        lobby_id=lobby.id,
        skip_llm=skip_llm,
        llm_budget=llm_budget,
        openrouter_client=openrouter_client,
        stats=stats,
    )
    all_decisions.extend(resolved_decisions)
    predictions.extend(resolved_predictions)

    for index, sp_match in enumerate(matches):
        if sp_match.open_market_count == 0:
            logger.info("Skipping %s: open_market_count=0", sp_match.name)
            continue

        stats.matches_processed += 1

        if index > 0:
            time.sleep(RATE_LIMIT_DELAY_SEC)

        try:
            markets = sp_client.get_markets(lobby.id, sp_match.id)
        except Exception as exc:
            logger.warning("Failed to load markets for %s: %s", sp_match.name, exc)
            stats.errors.append(f"markets {sp_match.name}: load failed")
            continue

        odds_match = find_odds_match(sp_match, odds_matches)
        fair_probs = None
        if odds_match is not None:
            fair_probs = step1_fair_probabilities_from_odds(odds_match)
            logger.info(
                "Fair probs %s: home=%.3f draw=%.3f away=%.3f",
                sp_match.name,
                fair_probs.home_win,
                fair_probs.draw,
                fair_probs.away_win,
            )

        for market in markets:
            decision = build_decision_for_market(
                market_id=market.id,
                match_name=market.match_name,
                question=market.question,
                market_status=market.status,
                fair_probs=fair_probs,
                skip_llm=skip_llm,
                llm_budget=llm_budget,
                openrouter_client=openrouter_client,
            )
            all_decisions.append(decision)

            if decision.skipped:
                stats.markets_skipped += 1
                logger.info(
                    "SKIP [%s] %s — %s",
                    sp_match.name,
                    market.question[:60],
                    decision.skip_reason,
                )
                if market.status != "open":
                    remove_pending(market.id, pending)
                continue

            _track_llm_stats(stats, decision)

            if decision.pending:
                stats.markets_pending_no_odds += 1
                upsert_pending(
                    record_from_decision(
                        market_id=market.id,
                        match_id=sp_match.id,
                        lobby_id=lobby.id,
                        match_name=market.match_name,
                        question=market.question,
                        market_status=market.status,
                        match_opening_time=sp_match.opening_time.isoformat(),
                        base_probability=decision.base_probability or 0.5,
                        adjustment=decision.adjustment,
                        adjustment_reason=decision.adjustment_reason,
                        adjusted_probability=decision.adjusted_probability or 0.5,
                        shrunk_probability=decision.shrunk_probability or 0.5,
                        api_probability=decision.api_probability or 50,
                        llm_called=decision.llm_called,
                        llm_reason=decision.llm_reason,
                    ),
                    pending,
                )
                logger.info(
                    "PENDING [%s] %s → %d%% (waiting for Odds API line)",
                    sp_match.name,
                    market.question[:50],
                    decision.api_probability,
                )
                continue

            remove_pending(market.id, pending)
            logger.info(
                "OK   [%s] %s → %d%% (%s base=%.3f adj=%+.3f llm=%s)",
                sp_match.name,
                market.question[:50],
                decision.api_probability,
                decision.market_kind,
                decision.base_probability or 0.0,
                decision.adjustment,
                decision.llm_called,
            )
            predictions.append(
                SpPredictionPayload(
                    market_id=market.id,
                    lobby_id=lobby.id,
                    probability=decision.api_probability,
                )
            )
            if decision.market_kind == "win":
                stats.win_predictions_sent += 1
            else:
                stats.prop_predictions_sent += 1
            _add_example(stats, decision)

    save_pending(pending)
    logger.info("Pending decisions saved: %d → %s", len(pending), PENDING_DECISIONS_FILENAME)

    stats.llm_calls = llm_budget.used
    stats.predictions_sent = len(predictions)

    log_path = LOGS_DIR / PREDICTION_LOG_FILENAME

    submission_outcomes: dict = {}
    if predictions:
        submission_outcomes = _submit_predictions(
            predictions,
            sp_client=sp_client,
            lobby_id=lobby.id,
            dry_run=dry_run,
            stats=stats,
            log_path=log_path,
        )
    else:
        logger.warning("No predictions to submit")

    append_prediction_log(
        all_decisions,
        log_path,
        submission_outcomes=submission_outcomes,
    )
    logger.info("Wrote %d decisions to %s", len(all_decisions), log_path)

    try:
        sync_results(sp_client, lobby.id)
    except Exception as exc:
        logger.warning("Failed to sync /results: %s", exc)
        stats.errors.append("results sync failed")

    if not dry_run:
        logger.info(
            "Done: %d computed, %d created, %d updated, %d update failures",
            len(predictions),
            stats.predictions_created,
            stats.predictions_updated,
            stats.predictions_update_failed,
        )
        send_telegram_message(build_run_summary_telegram(stats, max_llm_calls=MAX_LLM_CALLS_PER_RUN))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probability Cup — automated predictions via SportsPredict API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log without POST /predictions/batch",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip OpenRouter calls (adjustment=0)",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Limit number of matches (debug)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        exit_code = run(
            dry_run=args.dry_run,
            skip_llm=args.skip_llm,
            max_matches=args.max_matches,
        )
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        send_telegram_message(
            "⚠️ *Probability Cup — run*\n\n"
            "Something went wrong. Check server logs for details."
        )
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
