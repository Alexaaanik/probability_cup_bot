"""Точка входа: загрузка матчей, расчёт прогнозов, отправка в API."""

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
    PREDICTION_LOG_FILENAME,
    PREDICTIONS_BATCH_SIZE,
    RATE_LIMIT_DELAY_SEC,
)
from llm_client import LlmCallBudget
from matcher import find_odds_match
from model import ModelDecision, build_decision_for_market, step1_fair_probabilities_from_odds
from notifier import send_telegram_message
from odds_client import OddsApiClient
from sportspredict_client import SpPredictionPayload, SportsPredictClient
from sync_results import sync_results
from utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    matches_processed: int = 0
    predictions_sent: int = 0
    markets_skipped_no_h2h: int = 0
    llm_calls: int = 0
    llm_adjustments_applied: int = 0
    examples: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_settings() -> tuple[str, str, str | None]:
    load_dotenv()
    sp_key = os.getenv("SPORTSPREDICT_API_KEY", "").strip()
    odds_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None

    if not sp_key:
        raise ValueError("SPORTSPREDICT_API_KEY не задан в .env")
    if not odds_key:
        raise ValueError("THE_ODDS_API_KEY не задан в .env")

    return sp_key, odds_key, openrouter_key


def append_prediction_log(decisions: list[ModelDecision], log_path: Path) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        for decision in decisions:
            record = asdict(decision)
            record["logged_at"] = run_ts
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_pct(probability: float | None) -> str:
    if probability is None:
        return "н/д"
    return f"{round(probability * 100)}%"


def _format_adj(adjustment: float) -> str:
    points = round(adjustment * 100)
    if points > 0:
        return f"+{points}%"
    return f"{points}%"


def build_run_summary_telegram(stats: RunStats, *, max_llm_calls: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"🎯 *Probability Cup — прогон {now}*",
        "",
        f"Обработано матчей: {stats.matches_processed}",
        f"Рынков с прогнозом: {stats.predictions_sent} (win-рынки)",
        f"Рынков пропущено: {stats.markets_skipped_no_h2h} (нет h2h-коэффициентов)",
        (
            f"LLM-корректировок применено: {stats.llm_adjustments_applied} "
            f"из {stats.llm_calls} вызовов"
        ),
        f"Лимит LLM-вызовов: {stats.llm_calls}/{max_llm_calls}",
        "",
        "Примеры прогнозов:",
    ]

    if stats.examples:
        lines.extend(stats.examples)
    else:
        lines.append("— нет отправленных прогнозов")

    lines.append("")
    if stats.errors:
        lines.append(f"Ошибок: {len(stats.errors)}")
        for err in stats.errors[:3]:
            lines.append(f"• {err}")
    else:
        lines.append("Ошибок: 0")

    return "\n".join(lines)


def run(*, dry_run: bool, skip_llm: bool, max_matches: int | None = None) -> int:
    setup_logging()
    logger.info("Probability Cup Bot v%s", BOT_VERSION)

    stats = RunStats()
    sp_key, odds_key, openrouter_key = load_settings()
    sp_client = SportsPredictClient(sp_key)
    odds_client = OddsApiClient(odds_key)
    llm_budget = LlmCallBudget(max_calls=MAX_LLM_CALLS_PER_RUN)
    openrouter_client: OpenAI | None = None

    if skip_llm:
        logger.info("Флаг --skip-llm: LLM-корректировка отключена")
    elif openrouter_key:
        openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_key,
        )
    else:
        logger.warning("OPENROUTER_API_KEY не задан — LLM отключён")

    event = sp_client.find_probability_cup_event()
    logger.info("Событие: %s (%s)", event.title, event.id)

    lobbies = sp_client.get_lobbies(event.id)
    if not lobbies:
        logger.error("Нет лобби для event_id=%s", event.id)
        stats.errors.append("не найдено лобби для события")
        return 1

    lobby = lobbies[0]
    logger.info("Лобби: %s (joined=%s)", lobby.name, lobby.joined)

    if not lobby.joined and not dry_run:
        sp_client.join_lobby(lobby.id)
    elif not lobby.joined:
        logger.info("[dry-run] Пропуск join лобби %s", lobby.id)

    matches = sp_client.get_matches(event.id)
    if max_matches is not None:
        matches = matches[:max_matches]
        logger.info("Ограничение --max-matches=%d", max_matches)
    logger.info("Матчей к обработке: %d", len(matches))

    odds_matches = odds_client.fetch_world_cup_odds()
    all_decisions: list[ModelDecision] = []
    predictions: list[SpPredictionPayload] = []

    for index, sp_match in enumerate(matches):
        if sp_match.open_market_count == 0:
            logger.info("Пропуск %s: open_market_count=0", sp_match.name)
            continue

        stats.matches_processed += 1

        if index > 0:
            time.sleep(RATE_LIMIT_DELAY_SEC)

        try:
            markets = sp_client.get_markets(lobby.id, sp_match.id)
        except Exception as exc:
            logger.warning("Ошибка загрузки рынков для %s: %s", sp_match.name, exc)
            stats.errors.append(f"рынки {sp_match.name}: сбой загрузки")
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
                if "h2h" in decision.skip_reason:
                    stats.markets_skipped_no_h2h += 1
                logger.info(
                    "SKIP [%s] %s — %s",
                    sp_match.name,
                    market.question[:60],
                    decision.skip_reason,
                )
                continue

            if decision.llm_called and decision.adjustment != 0.0:
                stats.llm_adjustments_applied += 1

            logger.info(
                "OK   [%s] %s → %d%% (base=%.3f adj=%+.3f llm=%s)",
                sp_match.name,
                market.question[:50],
                decision.api_probability,
                decision.base_probability,
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

            if len(stats.examples) < 5:
                stats.examples.append(
                    f"{decision.match_name}: {decision.api_probability}% "
                    f"(база {_format_pct(decision.base_probability)}, "
                    f"корр. {_format_adj(decision.adjustment)})"
                )

    stats.llm_calls = llm_budget.used
    stats.predictions_sent = len(predictions)

    log_path = LOGS_DIR / PREDICTION_LOG_FILENAME
    append_prediction_log(all_decisions, log_path)
    logger.info("Записано %d решений в %s", len(all_decisions), log_path)

    try:
        sync_results(sp_client, lobby.id)
    except Exception as exc:
        logger.warning("Не удалось синхронизировать /results: %s", exc)
        stats.errors.append("синхронизация результатов не удалась")

    if not predictions:
        logger.warning("Нет прогнозов для отправки")
        if not dry_run:
            send_telegram_message(build_run_summary_telegram(stats, max_llm_calls=MAX_LLM_CALLS_PER_RUN))
        return 0

    if dry_run:
        logger.info("[dry-run] Было бы отправлено %d прогнозов:", len(predictions))
        for payload in predictions:
            logger.info(
                "  market_id=%s lobby_id=%s probability=%d",
                payload.market_id,
                payload.lobby_id,
                payload.probability,
            )
        return 0

    for batch_start in range(0, len(predictions), PREDICTIONS_BATCH_SIZE):
        batch = predictions[batch_start : batch_start + PREDICTIONS_BATCH_SIZE]
        try:
            result = sp_client.submit_predictions_batch(batch)
            logger.info(
                "Отправлен батч %d–%d (%d шт.): %s",
                batch_start + 1,
                batch_start + len(batch),
                len(batch),
                result,
            )
        except Exception as exc:
            logger.error("Ошибка отправки батча: %s", exc)
            stats.errors.append("отправка прогнозов прервана")

    logger.info("Готово: отправлено %d прогнозов", len(predictions))
    send_telegram_message(build_run_summary_telegram(stats, max_llm_calls=MAX_LLM_CALLS_PER_RUN))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probability Cup — автоматические прогнозы через SportsPredict API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Посчитать и залогировать, но не отправлять POST /predictions/batch",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Пропустить вызов OpenRouter (adjustment=0)",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Ограничить число матчей (для отладки)",
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
        logging.exception("Критическая ошибка: %s", exc)
        send_telegram_message(
            "⚠️ *Probability Cup — прогон*\n\n"
            "Что-то пошло не так при выполнении скрипта. Детали в логах на сервере."
        )
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
