"""Ежедневная сводка Brier в Telegram (отдельный cron)."""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timezone

from dotenv import load_dotenv

from config import LOGS_DIR, PREDICTION_LOG_FILENAME
from notifier import send_telegram_message
from sportspredict_client import SportsPredictClient
from sync_results import (
    average_brier,
    filter_results_for_date,
    load_all_synced_results,
    load_predictions_by_market,
    sync_results,
)
from utils import setup_logging

logger = logging.getLogger(__name__)


def _count_predictions_sent(predictions_path) -> int:
    records = load_predictions_by_market(predictions_path)
    return sum(
        1
        for r in records.values()
        if not r.get("skipped") and r.get("api_probability") is not None
    )


def build_daily_report_text(report_date: date) -> str:
    sync_path_results = load_all_synced_results()
    day_results = filter_results_for_date(sync_path_results, report_date)
    all_results = sync_path_results

    day_avg = average_brier(day_results)
    tournament_avg = average_brier(all_results)
    total_predictions = _count_predictions_sent(LOGS_DIR / PREDICTION_LOG_FILENAME)

    lines = [
        f"📊 *Probability Cup — итоги дня {report_date.isoformat()}*",
        "",
        f"Матчей сыграно и засчитано: {len(day_results)}",
    ]

    if day_avg is not None:
        lines.append(f"Средний Brier нашего бота: {day_avg:.3f}")
    else:
        lines.append("Средний Brier нашего бота: н/д (нет сыгранных матчей)")

    lines.extend(
        [
            "_(для сравнения: 0.25 = угадывать 50/50 на всё, 0 = идеально)_",
            "",
        ]
    )

    if day_results:
        best = min(day_results, key=lambda r: r.brier_score)
        worst = max(day_results, key=lambda r: r.brier_score)
        lines.append(f"Лучший прогноз дня: {best.match_name} — Brier {best.brier_score:.3f}")
        lines.append(f"Худший прогноз дня: {worst.match_name} — Brier {worst.brier_score:.3f}")
    else:
        lines.append("Лучший/худший прогноз дня: н/д")

    lines.extend(
        [
            "",
            f"Прогнозов всего отправлено с начала турнира: {total_predictions}",
        ]
    )

    if tournament_avg is not None:
        lines.append(f"Средний Brier за весь турнир: {tournament_avg:.3f}")
    else:
        lines.append("Средний Brier за весь турнир: н/д")

    return "\n".join(lines)


def run(report_date: date | None = None) -> int:
    load_dotenv()
    setup_logging()

    report_date = report_date or datetime.now(timezone.utc).date()

    sp_key = os.getenv("SPORTSPREDICT_API_KEY", "").strip()
    if not sp_key:
        logger.error("SPORTSPREDICT_API_KEY не задан")
        return 1

    try:
        client = SportsPredictClient(sp_key)
        event = client.find_probability_cup_event()
        lobbies = client.get_lobbies(event.id)
        if not lobbies:
            logger.error("Нет лобби")
            return 1

        sync_results(client, lobbies[0].id)
        text = build_daily_report_text(report_date)
        send_telegram_message(text)
        logger.info("Ежедневный отчёт сформирован за %s", report_date)
        return 0
    except Exception as exc:
        logger.exception("Ошибка daily_report: %s", exc)
        send_telegram_message(
            f"⚠️ *Probability Cup — ежедневный отчёт*\n\n"
            f"Что-то пошло не так при формировании отчёта за {report_date}. "
            f"Детали в логах на сервере."
        )
        return 1


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
