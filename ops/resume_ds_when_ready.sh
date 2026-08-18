#!/usr/bin/env bash
# One-shot: wait until the opencode zen free tier (ds_gateway :10900) is usable
# again (FreeUsageLimitError resets ~daily), then clear DS pause markers and
# restart the DS supervisor so all worldlines resume with concurrency 3.
# Mirrors resume_terra_when_ready.sh.  Launched detached; logs to ops/logs.
set -u
ROOT=/home/lxx/trade-agent-benchmark
DS_RESULTS=$ROOT/AC-deepseek/results/ac9wl_deepseek
LOG=$ROOT/ops/logs/resume_ds_when_ready.log
KEY=$(grep -oE 'sk-[A-Za-z0-9]{20,}' /home/lxx/opencode-api/client.env | head -1)
mkdir -p "$(dirname "$LOG")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

probe() {
  curl -sS -m 60 -o /tmp/ds_probe.json -w '%{http_code}' \
    http://127.0.0.1:10900/v1/messages \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash-free","max_tokens":8,"messages":[{"role":"user","content":"say ok"}]}' \
    2>/dev/null
}

echo "$(ts) resume-ds watcher started; probing gateway every 30 min (up to 26h)" >> "$LOG"
for i in $(seq 1 52); do
  CODE=$(probe)
  if [ "$CODE" = "200" ]; then
    echo "$(ts) gateway OK (HTTP 200) on attempt $i; resuming DS" >> "$LOG"
    # 1) clear all active pause markers (stall/budget/operator kinds are all
    #    obsolete once the upstream is healthy again)
    rm -f "$DS_RESULTS"/pause_wl*_429
    # 2) restart supervisor: stop it + orphan workers gracefully first
    pkill -INT -f 'run_deepseek_ac9\.py' 2>/dev/null
    sleep 5
    for p in $(pgrep -f 'main\.py wl[0-9].*ac9wl_deepseek'); do kill -INT "$p" 2>/dev/null; done
    sleep 20
    for p in $(pgrep -f 'main\.py wl[0-9].*ac9wl_deepseek'); do kill -KILL "$p" 2>/dev/null; done
    # 3) relaunch (deepseek-sub2api.env now has AC_DEEPSEEK_CONCURRENCY=3)
    cd "$ROOT/AC-deepseek" || { echo "$(ts) cd failed" >> "$LOG"; exit 1; }
    AC_DEEPSEEK_CONCURRENCY=3 AC_DEEPSEEK_ONLINE_MAX_CYCLES=360 \
      setsid nohup bash run_deepseek_ac9.sh >> "$DS_RESULTS/supervisor.log" 2>&1 < /dev/null &
    echo "$(ts) DS supervisor relaunched; watcher done" >> "$LOG"
    exit 0
  fi
  echo "$(ts) attempt $i: gateway HTTP $CODE; retrying in 30 min" >> "$LOG"
  sleep 1800
done
echo "$(ts) gave up after 52 attempts; DS left in current state" >> "$LOG"
exit 1
