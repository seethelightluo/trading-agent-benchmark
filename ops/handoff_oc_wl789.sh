#!/usr/bin/env bash
# Handoff rev2 (2026-08-23): when the running oc supervisor (ONLY=6,8, relay C
# :8789) exits after wl8 completes, immediately relaunch it as ONLY=6,8,9 so
# wl9 (moved here from the Plus dir; wl7 stays on Plus) starts without waiting
# for the 15-min watchdog tick.  wl5 keeps riding Plus as an orphan worker;
# the patched watchdog adopts it and then wl7 (serial, concurrency 1).
set -u
ROOT=/home/lxx/trade-agent-benchmark
OC_DIR=$ROOT/agent-framework/results/ac_luna_3wl_v5_oc
LOG=$ROOT/ops/logs/handoff_oc_wl789.log
mkdir -p "$(dirname "$LOG")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
exec 9>"$ROOT/ops/logs/handoff_oc_wl789.lock"
flock -n 9 || { echo "$(ts) another instance running; exiting" >> "$LOG"; exit 0; }

oc_sup_pid() {
  for p in $(pgrep -f 'scheduler\.run_ac_luna_3' 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "AC_LUNA_RUN_DIR=$OC_DIR"; then
      echo "$p"; return 0
    fi
  done
  return 1
}

echo "$(ts) rev2 watcher started: waiting for oc supervisor (ONLY=6,8) to exit" >> "$LOG"
while oc_sup_pid >/dev/null; do sleep 60; done
echo "$(ts) oc supervisor gone; confirming stable" >> "$LOG"
sleep 60
if oc_sup_pid >/dev/null; then
  echo "$(ts) oc supervisor reappeared (watchdog may have launched it); env:" >> "$LOG"
  tr '\0' '\n' < "/proc/$(oc_sup_pid)/environ" 2>/dev/null | grep AC_LUNA_ONLY >> "$LOG"
  exit 0
fi
cd "$ROOT/agent-framework" || { echo "$(ts) cd failed" >> "$LOG"; exit 1; }
OPENAI_API_URL=http://127.0.0.1:8789/v1 \
AC_LUNA_RUN_DIR="$OC_DIR" AC_LUNA_ONLY=6,8,9 AC_LUNA_WORLDLINES=9 \
AC_LUNA_CONCURRENCY=1 AC_LUNA_ONLINE_MAX_CYCLES=360 \
setsid nohup "$ROOT/.venv/bin/python" -u -m scheduler.run_ac_luna_3 \
  >> "$OC_DIR/supervisor.log" 2>&1 < /dev/null &
echo "$(ts) oc supervisor relaunched ONLY=6,8,9 pid=$!; watcher done" >> "$LOG"
