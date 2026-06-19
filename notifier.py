"""Telegram notifications."""

from __future__ import annotations

import logging
import os

from utils import request_with_retry

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


def _telegram_configured() -> tuple[str, str] | None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        return token, chat_id
    return None


def send_telegram_message(text: str) -> bool:
    """POST sendMessage. Skips silently if token/chat_id are missing."""
    config = _telegram_configured()
    if config is None:
        logger.info("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — skipping")
        return False

    token, chat_id = config
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"

    try:
        response = request_with_retry(
            "POST",
            url,
            json_body={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            },
        )
        if response.status_code == 200:
            return True
        logger.warning("Telegram API returned %s: %s", response.status_code, response.text[:200])
    except Exception as exc:
        logger.warning("Failed to send Telegram message: %s", exc)

    return False
