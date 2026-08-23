#!/usr/bin/env bash
# One-shot countdown: probe Plus relay A (:8787 -> sub2api account 1, ChatGPT
# Plus) until the subscription quota resets (~2026-08-22 16:24), then clear the
# plus-dir pause markers and immediately launch the wl5,7,9 supervisor on the
# Plus source.  The ac_supervisor_watch.sh watchdog (also repointed to :8787)
# is the safety net; both sides check "plus supervisor already alive" so they
# cannot double-launch (2026-08-18 lesson).
set -u
ROOT=/home/lxx/trade-agent-benchmark
PLUS_DIR=$ROOT/agent-framework/results/ac_luna_3wl_v5_plus
AF=$ROOT/agent-framework
LOG=$ROOT/ops/logs/resume_plus_when_ready.log
KEY=$(grep -oE 'UPSTREAM_API_KEY=[^ ]+' /home/lxx/ac-llm-relay/relay.env | head -1 | cut -d= -f2)
mkdir -p "$(dirname "$LOG")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
exec 9>"$ROOT/ops/logs/resume_plus_when_ready.lock"
flock -n 9 || { echo "$(ts) another instance running; exiting" >> "$LOG"; exit 0; }
[ -z "$KEY" ] && { echo "$(ts) no relay key found" >> "$LOG"; exit 1; }

probe() {
  curl -sS -m 120 -o /tmp/plus_ready_probe.json -w '%{http_code}' \
    http://127.0.0.1:8787/v1/responses \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"model":"gpt-5.6-terra","input":"Reply with: ok","max_output_tokens":16}' \
    2>/dev/null
}

plus_sup_pid() {
  for p in $(pgrep -f 'scheduler\.run_ac_luna_3' 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "AC_LUNA_RUN_DIR=$PLUS_DIR"; then
      echo "$p"; return 0
    fi
  done
  return 1
}

echo "$(ts) plus-resume watcher started; probing relay A :8787 every 5 min (up to 6h)" >> "$LOG"
SUCC=0
for i in $(seq 1 72); do
  CODE=$(probe)
  if [ "$CODE" = "200" ]; then
    SUCC=$((SUCC+1))
    echo "$(ts) attempt $i: Plus OK ($SUCC/2)" >> "$LOG"
    [ "$SUCC" -ge 2 ] && break
    sleep 60   # confirm quickly after a success
  else
    SUCC=0
    echo "$(ts) attempt $i: Plus HTTP $CODE; retrying in 5 min" >> "$LOG"
    sleep 300
  fi
done
if [ "$SUCC" -lt 2 ]; then echo "$(ts) gave up; plus dir left paused" >> "$LOG"; exit 1; fi

echo "$(ts) Plus quota reset confirmed; clearing pause markers" >> "$LOG"
rm -f "$PLUS_DIR"/pause_wl5_429 "$PLUS_DIR"/pause_wl7_429 "$PLUS_DIR"/pause_wl9_429

if plus_sup_pid >/dev/null; then
  echo "$(ts) plus supervisor already alive (pid $(plus_sup_pid), watchdog won the race); done" >> "$LOG"
  exit 0
fi
# stop any orphaned plus-dir workers before relaunch (same INT->KILL ladder)
ORPHANS=$(ps -eo pid,cmd | grep -E "main\.py wl[0-9].*ac_luna_3wl_v5_plus/run_config\.yaml" | grep -v grep | awk '{print $1}')
if [ -n "$ORPHANS" ]; then
  echo "$(ts) stopping orphaned plus workers: $ORPHANS" >> "$LOG"
  kill -INT $ORPHANS 2>/dev/null; sleep 20
  kill -KILL $ORPHANS 2>/dev/null
fi
cd "$AF" || { echo "$(ts) cd failed" >> "$LOG"; exit 1; }
OPENAI_API_URL=http://127.0.0.1:8787/v1 \
AC_LUNA_RUN_DIR="$PLUS_DIR" AC_LUNA_ONLY=5,7,9 AC_LUNA_WORLDLINES=9 \
AC_LUNA_CONCURRENCY=1 AC_LUNA_ONLINE_MAX_CYCLES=360 \
setsid nohup "$ROOT/.venv/bin/python" -u -m scheduler.run_ac_luna_3 \
  >> "$PLUS_DIR/supervisor.log" 2>&1 < /dev/null &
echo "$(ts) plus supervisor launched pid=$!; watcher done" >> "$LOG"
