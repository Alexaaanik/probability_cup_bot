#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/probability_cup_bot
mkdir -p logs

docker compose run --rm --build probability-cup-bot >> logs/cron.log 2>&1
