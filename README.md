# Probability Cup Bot

Скрипт для [Jump Trading Probability Cup](https://sportspredict.com) на ЧМ-2026: тянет букмекерские линии, сопоставляет матчи с рынками турнира, при необходимости чуть подкручивает вероятность через LLM и шлёт прогнозы в API.

Модель простая: коэффициенты → матчинг по названиям команд → опциональная LLM-поправка (±5 п.п., только если есть конкретные новости) → shrinkage к 50% → POST в SportsPredict.

## Что внутри

```
probability_cup_bot/
├── main.py                 # основной прогон
├── config.py               # константы и алиасы команд
├── odds_client.py          # The Odds API
├── sportspredict_client.py # SportsPredict API
├── matcher.py              # сопоставление матчей
├── model.py                # вероятности и shrinkage
├── llm_client.py           # OpenRouter
├── notifier.py             # Telegram
├── sync_results.py         # подтягивание Brier из /results
├── daily_report.py         # ежедневная сводка
├── utils.py
├── logs/                   # predictions.jsonl, results_sync.jsonl (локально)
├── .env.example
└── requirements.txt
```

## Настройка

Скопируй `.env.example` в `.env` и заполни ключи:

| Переменная | Нужна? | Где взять |
|------------|--------|-----------|
| `SPORTSPREDICT_API_KEY` | да | Probability Cup / SportsPredict (`sp_live_...`) |
| `THE_ODDS_API_KEY` | да | [the-odds-api.com](https://the-odds-api.com) |
| `OPENROUTER_API_KEY` | нет | [openrouter.ai](https://openrouter.ai) |
| `OPENROUTER_MODEL` | нет | по умолчанию `anthropic/claude-haiku-4.5` |
| `MAX_LLM_CALLS_PER_RUN` | нет | лимит вызовов за прогон, дефолт `150` |
| `TELEGRAM_BOT_TOKEN` | нет | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | нет | свой chat id |

Без OpenRouter бот работает только на коэффициентах — `adjustment` будет 0.

```bash
cd probability_cup_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

Сначала dry-run без отправки и без LLM:

```bash
python main.py --dry-run --skip-llm
```

Проверить LLM на одном матче:

```bash
python main.py --dry-run --max-matches 1
```

Боевой прогон:

```bash
python main.py
```

Только коэффициенты, без LLM:

```bash
python main.py --skip-llm
```

Отдельно — синхронизация результатов и дневной отчёт:

```bash
python sync_results.py
python daily_report.py
```

## Как считается прогноз

1. **Коэффициенты** — средний decimal h2h по букмекерам, маржа убирается: `fair = (1/odds) / Σ(1/odds)`.
2. **Матчинг** — нормализация названий (`USA` → `usa`, `AUS` → `australia` и т.д.) + дата ±1 день.
3. **LLM** — Claude Haiku через OpenRouter. Промпт в `llm_client.py`, поправка не больше ±5 п.п. Если ключей нет или лимит исчерпан — шаг пропускается.
4. **Shrinkage** — `0.5 + (p - 0.5) * 0.9`, потом округление в 1–99 для API.

Сейчас обрабатываются только рынки вида `Will <team> win the match?`. Прочие рынки (угловые, фолы) скипаются — для них нет h2h.

## Логи

`logs/predictions.jsonl` — одна строка JSON на каждое решение: base, adjustment, llm_reason, api_probability, skip_reason.

`logs/results_sync.jsonl` — сыгранные рынки с Brier score (через `sync_results.py`).

Пример записи:

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

После реального `main.py` (не `--dry-run`) уходит короткая сводка. `daily_report.py` — раз в день Brier за день и за турнир. Если токен не задан, скрипт просто пишет в консоль.

## Cron

```cron
0 10 * * * cd /path/to/probability_cup_bot && .venv/bin/python main.py >> logs/cron.log 2>&1
0 22 * * * cd /path/to/probability_cup_bot && .venv/bin/python daily_report.py >> logs/cron_daily.log 2>&1
```

## Версия

`config.BOT_VERSION` — `1.1.0`.
