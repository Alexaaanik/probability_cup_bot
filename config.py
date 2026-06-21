"""Configuration and constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"

SPORTSPREDICT_BASE_URL = "https://api.sportspredict.com/api/v1"

# API type field is a UUID — find the event by title instead
PROBABILITY_CUP_TITLE_KEYWORD = "probability"

MARKET_EVENT_TYPE_PROBABILITY = "probability"
MARKET_STATUS_OPEN = "open"

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "soccer_fifa_world_cup"
ODDS_REGIONS = "eu"
ODDS_MARKETS = "h2h"
ODDS_FORMAT = "decimal"

MATCH_DATE_TOLERANCE_DAYS = 1

# Team aliases for match mapping
TEAM_ALIASES: dict[str, str] = {
    "usa": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "aus": "australia",
    "australia": "australia",
    "sco": "scotland",
    "scotland": "scotland",
    "mar": "morocco",
    "morocco": "morocco",
    "bra": "brazil",
    "brazil": "brazil",
    "haiti": "haiti",
    "tur": "turkey",
    "turkey": "turkey",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "par": "paraguay",
    "paraguay": "paraguay",
    "ned": "netherlands",
    "netherlands": "netherlands",
    "swe": "sweden",
    "sweden": "sweden",
    "ger": "germany",
    "germany": "germany",
    "civ": "ivory coast",
    "ivory coast": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "ecu": "ecuador",
    "ecuador": "ecuador",
    "curacao": "curacao",
    "curaçao": "curacao",
    "tun": "tunisia",
    "tunisia": "tunisia",
    "jpn": "japan",
    "japan": "japan",
    "esp": "spain",
    "spain": "spain",
    "ksa": "saudi arabia",
    "saudi arabia": "saudi arabia",
    "bel": "belgium",
    "belgium": "belgium",
    "irn": "iran",
    "iran": "iran",
    "uru": "uruguay",
    "uruguay": "uruguay",
    "cpv": "cape verde",
    "cape verde": "cape verde",
    "new zealand": "new zealand",
    "egy": "egypt",
    "egypt": "egypt",
    "arg": "argentina",
    "argentina": "argentina",
    "aut": "austria",
    "austria": "austria",
    "mex": "mexico",
    "mexico": "mexico",
    "kor": "south korea",
    "south korea": "south korea",
    "korea republic": "south korea",
    "can": "canada",
    "canada": "canada",
    "fra": "france",
    "france": "france",
    "den": "denmark",
    "denmark": "denmark",
    "sui": "switzerland",
    "switzerland": "switzerland",
    "crc": "costa rica",
    "costa rica": "costa rica",
    "eng": "england",
    "england": "england",
    "por": "portugal",
    "portugal": "portugal",
    "cro": "croatia",
    "croatia": "croatia",
    "pol": "poland",
    "poland": "poland",
    "sen": "senegal",
    "senegal": "senegal",
    "gha": "ghana",
    "ghana": "ghana",
    "cam": "cameroon",
    "cameroon": "cameroon",
    "qat": "qatar",
    "qatar": "qatar",
    "wal": "wales",
    "wales": "wales",
    "srb": "serbia",
    "serbia": "serbia",
    "ukr": "ukraine",
    "ukraine": "ukraine",
    "nor": "norway",
    "norway": "norway",
    "jor": "jordan",
    "jordan": "jordan",
    "alg": "algeria",
    "algeria": "algeria",
    "uzb": "uzbekistan",
    "uzbekistan": "uzbekistan",
    "pan": "panama",
    "panama": "panama",
    "col": "colombia",
    "colombia": "colombia",
    "cod": "dr congo",
    "dr congo": "dr congo",
    "democratic republic of the congo": "dr congo",
    "drc": "dr congo",
    "bih": "bosnia & herzegovina",
    "bosnia and herzegovina": "bosnia & herzegovina",
    "bosnia & herzegovina": "bosnia & herzegovina",
    "rsa": "south africa",
    "south africa": "south africa",
    "cze": "czech republic",
    "czech republic": "czech republic",
    "czechia": "czech republic",
    "irq": "iraq",
    "iraq": "iraq",
    "bol": "bolivia",
    "bolivia": "bolivia",
}

MIN_BOOKMAKERS_FOR_ODDS = 1

# LLM (OpenRouter)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

# Per-run safety cap (in-memory only, resets each main.py start).
# Sized for ~24 win + ~69 LLM-worthy prop markets per full tournament pass + 20% headroom.
MAX_LLM_CALLS_PER_RUN = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "110"))

# ±0.05 = ±5 percentage points
LLM_MAX_ADJUSTMENT = 0.05

LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 150

LLM_REQUEST_TIMEOUT_SEC = 60

# shrinkage: shrunk = 0.5 + (p - 0.5) * factor
PROBABILITY_CENTER = 0.5
SHRINKAGE_FACTOR = 0.9

# Prop markets (corners, fouls, cards, etc.) — lower confidence, closer to 50%
PROP_SHRINKAGE_FACTOR = 0.55
PROP_LLM_MAX_ADJUSTMENT = 0.03
PROP_BASE_PROBABILITY = 0.5

# Win markets without Odds API line — save pending, submit when line appears
NO_ODDS_BASE_PROBABILITY = 0.5

MIN_PROBABILITY = 0.02
MAX_PROBABILITY = 0.98
API_PROB_MIN = 1
API_PROB_MAX = 99

# HTTP

HTTP_TIMEOUT_SEC = 30
HTTP_MAX_RETRIES = 5
HTTP_RETRY_BASE_DELAY_SEC = 1.0
HTTP_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

RATE_LIMIT_DELAY_SEC = 1.1  # ~60 req/min on SportsPredict
PREDICTIONS_BATCH_SIZE = 50

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

PREDICTION_LOG_FILENAME = "predictions.jsonl"
PENDING_DECISIONS_FILENAME = "pending_decisions.jsonl"
RESULTS_SYNC_FILENAME = "results_sync.jsonl"

TELEGRAM_API_BASE = "https://api.telegram.org"

BOT_VERSION = "1.2.0"
