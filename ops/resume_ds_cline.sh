#!/usr/bin/env bash
# Wait until DS can actually be served through sub2api (any group-4 account:
# opencode WARP free :10900 / cline pool :10910), then clear DS pause markers
# and restart the supervisor so all worldlines resume. E2E probe replaces the
# old cline-only probe so the watcher also fires when just opencode is live.
# Mutual exclusion: only one instance (pgrep on this script's basename).
set -u
ROOT=/home/lxx/trade-agent-benchmark
DS_RESULTS=$ROOT/AC-deepseek/results/ac9wl_deepseek
LOG=$ROOT/ops/logs/resume_ds_cline.log
DS_ENV=/home/lxx/.config/alphacrafter/deepseek-sub2api.env
ACCESS=$(grep -oE 'OPENAI_API_KEY=[^ ]+' "$DS_ENV" | head -1 | cut -d= -f2)
[ -z "$ACCESS" ] && ACCESS=$(grep -oE 'sk-ac-[A-Za-z0-9]+' "$DS_ENV" | head -1)
mkdir -p "$(dirname "$LOG")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
# flock-based mutual exclusion (pgrep self-match on harness wrappers bit us)
LOCK="$ROOT/ops/logs/resume_ds_cline.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "$(ts) another instance running; exiting" >> "$LOG"; exit 0; }

probe() {
  curl -sS -m 60 -o /tmp/ds_cline_probe.json -w '%{http_code}' \
    http://127.0.0.1:8080/v1/chat/completions \
    -H "Authorization: Bearer $ACCESS" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","max_tokens":64,"messages":[{"role":"user","content":"say ok"}]}' \
    2>/dev/null
}

echo "$(ts) resume-ds-cline watcher started; probing sub2api E2E every 15 min (up to 30h)" >> "$LOG"
SUCC=0
for i in $(seq 1 120); do
  CODE=$(probe)
  if [ "$CODE" = "200" ]; then
    SUCC=$((SUCC+1))
    echo "$(ts) attempt $i: pool OK ($SUCC/2)" >> "$LOG"
    [ "$SUCC" -ge 2 ] && break
  else
    SUCC=0
    echo "$(ts) attempt $i: pool HTTP $CODE; retrying in 15 min" >> "$LOG"
  fi
  # after a success, confirm quickly; after a failure, back off 15 min
  if [ "$SUCC" -ge 1 ]; then sleep 60; else sleep 900; fi
done
if [ "$SUCC" -lt 2 ]; then echo "$(ts) gave up; DS left as-is" >> "$LOG"; exit 1; fi

echo "$(ts) upstream confirmed healthy; checking DS state" >> "$LOG"
# Guard: if DS has no pause markers and workers are alive, do nothing
# (makes the script safe to run from cron repeatedly).
MARKERS=$(ls "$DS_RESULTS"/pause_wl*_429 2>/dev/null | wc -l)
RUNNING=$(pgrep -cf 'main\.py wl[0-9].*ac9wl_deepseek' || true)
if [ "$MARKERS" -eq 0 ] && [ "${RUNNING:-0}" -gt 0 ]; then
  echo "$(ts) DS healthy ($RUNNING workers, no markers); nothing to do" >> "$LOG"
  exit 0
fi
echo "$(ts) DS needs resume (markers=$MARKERS workers=${RUNNING:-0}); resuming" >> "$LOG"
# 1) clear active pause markers (all kinds are obsolete once upstream is healthy)
rm -f "$DS_RESULTS"/pause_wl*_429
# 2) stop ALL supervisors + orphan workers gracefully, then hard
# (head -1 variant left stragglers -> duplicate wl workers on 8/21;
#  supervisors also survive SIGINT and keep respawning workers,
#  so they get the same INT-then-KILL treatment as the workers)
SUPS=$(pgrep -f 'AC-deepseek/run_deepseek_ac9\.py')
[ -n "$SUPS" ] && kill -INT $SUPS && sleep 5
for p in $(pgrep -f 'main\.py wl[0-9].*ac9wl_deepseek'); do kill -INT "$p" 2>/dev/null; done
sleep 20
for p in $(pgrep -f 'main\.py wl[0-9].*ac9wl_deepseek'); do kill -KILL "$p" 2>/dev/null; done
for p in $(pgrep -f 'AC-deepseek/run_deepseek_ac9\.py'); do kill -KILL "$p" 2>/dev/null; done
# 3) relaunch (deepseek-sub2api.env carries CONCURRENCY=3 + sub2api routing)
cd "$ROOT/AC-deepseek" || { echo "$(ts) cd failed" >> "$LOG"; exit 1; }
AC_DEEPSEEK_CONCURRENCY=3 AC_DEEPSEEK_ONLINE_MAX_CYCLES=360 \
  setsid nohup bash run_deepseek_ac9.sh >> "$DS_RESULTS/supervisor.log" 2>&1 < /dev/null 9>&- &
echo "$(ts) DS supervisor relaunched; watcher done" >> "$LOG"
