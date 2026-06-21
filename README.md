# Probability Cup Bot

Script for the [Jump Trading Probability Cup](https://sportspredict.com) at the 2026 World Cup: pulls bookmaker lines, maps matches to tournament markets, optionally nudges probabilities via LLM, and submits predictions to the API.

Pipeline: odds → team name matching → optional LLM adjustment (±5 pp, news only) → shrinkage toward 50% → POST to SportsPredict.

## Layout

```
probability_cup_bot/
├── main.py                 # main run
├── config.py               # constants and team aliases
├── odds_client.py          # The Odds API
├── sportspredict_client.py # SportsPredict API
├── matcher.py              # match mapping
├── model.py                # probabilities and shrinkage
├── llm_client.py           # OpenRouter
├── notifier.py             # Telegram
├── sync_results.py         # Brier scores from /results
├── daily_report.py         # daily summary
├── pending_store.py        # отложенные решения без линии Odds API
├── utils.py
├── logs/                   # predictions.jsonl, pending_decisions.jsonl, results_sync.jsonl
├── .env.example
└── requirements.txt
```

## Setup

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `SPORTSPREDICT_API_KEY` | yes | Probability Cup / SportsPredict (`sp_live_...`) |
| `THE_ODDS_API_KEY` | yes | [the-odds-api.com](https://the-odds-api.com) |
| `OPENROUTER_API_KEY` | no | [openrouter.ai](https://openrouter.ai) |
| `OPENROUTER_MODEL` | no | default `anthropic/claude-haiku-4.5` |
| `MAX_LLM_CALLS_PER_RUN` | no | LLM calls per run, default `150` |
| `TELEGRAM_BOT_TOKEN` | no | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | no | your chat id |

Without OpenRouter the bot runs on odds only — `adjustment` stays 0.

```bash
cd probability_cup_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Dry-run without submission or LLM:

```bash
python main.py --dry-run --skip-llm
```

Test LLM on a single match:

```bash
python main.py --dry-run --max-matches 1
```

Live run:

```bash
python main.py
```

### Idempotent submission (manual check)

Run the same match twice — the second run should **update** existing predictions via `PATCH`, not fail with "already exists":

```bash
python main.py --max-matches 1 --skip-llm
python main.py --max-matches 1 --skip-llm
```

Expected in logs on the **second** run:
- `Submission plan: 0 create(s), N update(s), ...`
- `Updated market ... (update #1)`
- No batch of `success: false, error: already exists` without a follow-up PATCH

`logs/predictions.jsonl` will include `submission_action` (`create` / `update`) and `submission_update_count`.

Odds only, no LLM:

```bash
python main.py --skip-llm
```

Sync results and daily report (separate):

```bash
python sync_results.py
python daily_report.py
```

## How predictions are computed

1. **Odds** — average decimal h2h across bookmakers, margin removed: `fair = (1/odds) / Σ(1/odds)`.
2. **Matching** — normalize team names (`USA` → `usa`, `AUS` → `australia`, etc.) + date within ±1 day.
3. **LLM** — Claude Haiku via OpenRouter. Prompt in `llm_client.py`, max adjustment ±5 pp. Skipped if no API key or budget exhausted.
4. **Shrinkage** — `0.5 + (p - 0.5) * 0.9` для win-рынков; для prop-рынков (углы, фолы, карточки и т.д.) — factor `0.55` (ниже уверенность, ближе к 50%).
5. **Pending** — win-рынки без линии в The Odds API сохраняются в `logs/pending_decisions.jsonl` и пересчитываются при следующем запуске, когда линия появится.

Win-рынки (`Will <team> win the match?`) используют h2h-кэфы. Prop-рынки обрабатываются с нейтральной базой 50% и меньшим LLM-корridor (±3 pp).

## Logs

`logs/predictions.jsonl` — one JSON line per decision: base, adjustment, llm_reason, api_probability, market_kind, pending, skip_reason.

`logs/pending_decisions.jsonl` — win-рынки без линии Odds API; пересчитываются каждые 3 часа при появлении кэфов.

`logs/results_sync.jsonl` — settled markets with Brier score (via `sync_results.py`).

Example:

```json
{
  "match_name": "USA vs AUS",
  "question": "Will United States win the match?",
  "base_probability": 0.581,
  "adjustment": 0.0,
  "api_probability": 57,
  "skipped": false
}
```

## Telegram

After a real `main.py` run (not `--dry-run`) a short summary is sent. `daily_report.py` sends daily and tournament Brier stats. If Telegram is not configured, output goes to the console only.

## Cron

Автозапуск каждые 3 часа (Docker):

```cron
0 */3 * * * /root/projects/probability_cup_bot/run-cron.sh
```

Логи cron: `logs/cron.log`.

Ежедневный отчёт (опционально):

```cron
0 22 * * * cd /path/to/probability_cup_bot && docker compose run --rm probability-cup-bot python daily_report.py >> logs/cron_daily.log 2>&1
```

## Version

`config.BOT_VERSION` — `1.2.0`.
