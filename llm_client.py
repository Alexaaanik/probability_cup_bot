"""LLM-корректировка через OpenRouter (OpenAI-совместимый API)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

from config import (
    LLM_MAX_ADJUSTMENT,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT_SEC,
    LLM_TEMPERATURE,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

logger = logging.getLogger(__name__)

LLM_SYSTEM_PROMPT = """Ты помогаешь калибровать прогнозы для футбольного турнира.
Дают матч, вопрос по рынку и базовую вероятность из букмекерских коэффициентов.

Проверь, есть ли у тебя достоверное знание о конкретном недавнем событии, которое рынок
мог не успеть учесть: травма/дисквалификация ключевого игрока, неожиданный состав,
смена тренера, форс-мажор на эту игру.

Не корректируй из общих рассуждений о силе команд, форме или мотивации — это уже в коэффициентах.

Если такого факта нет — adjustment: 0. Не выдумывай.

Ответ строго JSON, без текста вокруг:
{"adjustment": <число от -0.05 до 0.05>, "reason": "<короткая фраза>"}"""

_JSON_BLOCK_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class LlmCallBudget:
    """Счётчик LLM-вызовов за один прогон."""

    max_calls: int
    used: int = 0
    limit_logged: bool = field(default=False)

    @property
    def limit_reached(self) -> bool:
        return self.used >= self.max_calls

    def can_call(self) -> bool:
        return not self.limit_reached

    def record_call(self) -> None:
        self.used += 1


def _build_user_message(
    match_name: str,
    market_question: str,
    base_probability: float,
) -> str:
    return (
        f"Матч: {match_name}\n"
        f"Рынок: {market_question}\n"
        f"Базовая вероятность (из коэффициентов): {base_probability:.3f}"
    )


def _parse_llm_json(raw_text: str) -> tuple[float, str]:
    text = raw_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_PATTERN.search(text)
        if match is None:
            raise
        data = json.loads(match.group())

    adjustment = float(data.get("adjustment", 0.0))
    reason = str(data.get("reason", ""))
    return adjustment, reason


def get_adjustment(
    match_name: str,
    base_probability: float,
    market_question: str,
    *,
    client: OpenAI | None = None,
) -> dict[str, str | float]:
    """Запрос к OpenRouter. При ошибке — adjustment=0, без исключения."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {
            "adjustment": 0.0,
            "reason": "OPENROUTER_API_KEY не задан",
            "raw_response": "",
        }

    if client is None:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    user_message = _build_user_message(match_name, market_question, base_probability)
    raw_response = ""

    try:
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_REQUEST_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw_response = (completion.choices[0].message.content or "").strip()
        raw_adj, reason = _parse_llm_json(raw_response)
    except json.JSONDecodeError:
        logger.warning("LLM parse error, raw=%r", raw_response[:200])
        return {
            "adjustment": 0.0,
            "reason": "llm_parse_error",
            "raw_response": raw_response,
        }
    except Exception as exc:
        logger.warning("LLM request error: %s", exc)
        return {
            "adjustment": 0.0,
            "reason": f"llm_request_error: {exc}",
            "raw_response": raw_response,
        }

    clamped = max(-LLM_MAX_ADJUSTMENT, min(LLM_MAX_ADJUSTMENT, raw_adj))
    if clamped != raw_adj:
        reason = f"{reason} [скорректировано до ±{LLM_MAX_ADJUSTMENT}]"

    return {
        "adjustment": clamped,
        "reason": reason,
        "raw_response": raw_response,
    }
