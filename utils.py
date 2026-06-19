"""Shared utilities: HTTP with retry, logging."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from config import (
    HTTP_MAX_RETRIES,
    HTTP_RETRY_BASE_DELAY_SEC,
    HTTP_RETRY_STATUS_CODES,
    HTTP_TIMEOUT_SEC,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    """HTTP request with exponential backoff on network errors and 429/5xx."""
    logger = logging.getLogger("http")
    last_error: Exception | None = None

    for attempt in range(HTTP_MAX_RETRIES):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SEC) as client:
                response = client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                )

            if response.status_code in HTTP_RETRY_STATUS_CODES:
                delay = HTTP_RETRY_BASE_DELAY_SEC * (2**attempt)
                logger.warning(
                    "HTTP %s for %s %s, retry in %.1fs (attempt %d/%d)",
                    response.status_code,
                    method,
                    url,
                    delay,
                    attempt + 1,
                    HTTP_MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            delay = HTTP_RETRY_BASE_DELAY_SEC * (2**attempt)
            logger.warning(
                "Network error %s: %s, retry in %.1fs (attempt %d/%d)",
                method,
                exc,
                delay,
                attempt + 1,
                HTTP_MAX_RETRIES,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed {method} {url} after {HTTP_MAX_RETRIES} attempts")
