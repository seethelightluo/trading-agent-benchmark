#!/usr/bin/env bash
# Lightweight DS/Terra AC supervisor watchdog (cron, every 15 min).
# Restarts a dead supervisor after gracefully stopping orphaned WL workers,
# so queue rotation and failure recovery never stay down.
set -u

ROOT=/home/lxx/trade-agent-benchmark
LOG=$ROOT/ops/logs/ac_supervisor_watch.log
DS_SH=$ROOT/AC-deepseek/run_deepseek_ac9.sh
DS_RESULTS=$ROOT/AC-deepseek/results/ac9wl_deepseek
TERRA_RESULTS=$ROOT/agent-framework/results/ac_luna_3wl_v5

mkdir -p "$(dirname "$LOG")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

restart_ds() {
  # Stop any orphaned DS WL workers first (supervisor is dead, workers may be alive).
  local orphans
  orphans=$(ps -eo pid,cmd | rg 'main\.py wl[0-9].*ac9wl_deepseek' | rg -v rg | awk '{print $1}')
  if [ -n "$orphans" ]; then
    echo "$(ts) stopping orphaned DS workers: $orphans" >> "$LOG"
    for pid in $orphans; do kill -INT "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 12); do
      sleep 15
      still=""
      for pid in $orphans; do
        [ -n "$(ps -o pid= -p "$pid" 2>/dev/null)" ] && still="$still $pid"
      done
      [ -z "$still" ] && break
    done
    for pid in $orphans; do
      [ -n "$(ps -o pid= -p "$pid" 2>/dev/null)" ] && kill -KILL "$pid" 2>/dev/null || true
    done
  fi
  echo "$(ts) restarting DS supervisor" >> "$LOG"
  cd "$ROOT/AC-deepseek"
  setsid nohup bash "$DS_SH" >> "$DS_RESULTS/supervisor.log" 2>&1 &
}

restart_terra() {
  local orphans
  orphans=$(ps -eo pid,cmd | rg 'main\.py wl[0-9].*luna_3wl_v5' | rg -v rg | awk '{print $1}')
  if [ -n "$orphans" ]; then
    echo "$(ts) stopping orphaned Terra workers: $orphans" >> "$LOG"
    for pid in $orphans; do kill -INT "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 12); do
      sleep 15
      still=""
      for pid in $orphans; do
        [ -n "$(ps -o pid= -p "$pid" 2>/dev/null)" ] && still="$still $pid"
      done
      [ -z "$still" ] && break
    done
    for pid in $orphans; do
      [ -n "$(ps -o pid= -p "$pid" 2>/dev/null)" ] && kill -KILL "$pid" 2>/dev/null || true
    done
  fi
  echo "$(ts) restarting Terra supervisor" >> "$LOG"
  cd "$ROOT/agent-framework"
  AC_LUNA_WORLDLINES=9 AC_LUNA_CONCURRENCY=3 setsid nohup .venv/bin/python -u -m scheduler.run_ac_luna_3 >> "$TERRA_RESULTS/supervisor.log" 2>&1 &
}

DS_ALIVE=$(pgrep -f 'run_deepseek_ac9\.py' 2>/dev/null | head -1)
TE_ALIVE=$(pgrep -f 'run_ac_luna_3\.py' 2>/dev/null | head -1)

[ -z "$DS_ALIVE" ] && restart_ds
[ -z "$TE_ALIVE" ] && restart_terra

exit 0
