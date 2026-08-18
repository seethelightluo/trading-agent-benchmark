#!/usr/bin/env bash
# Wait for terra wl3 (Plus source, v5 supervisor) to finish, then launch
# supervisor P running wl5,7,9 on the Plus source (relay A :8787, sub2api).
# pgrep patterns are anchored to the agent-framework v5 run dir so the
# concurrent AC-deepseek wlN processes never trigger a false match.
set -u
AF=/home/lxx/trade-agent-benchmark/agent-framework
PLUS_DIR=$AF/results/ac_luna_3wl_v5_plus
LOG=$PLUS_DIR/chain_watcher.log
log() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }

log "watcher started: waiting for terra wl3 (v5) to end"
SETTLE=0
while :; do
  if pgrep -f "main\.py wl3 --config .*/results/ac_luna_3wl_v5/" >/dev/null 2>&1; then
    SETTLE=0
  else
    SETTLE=$((SETTLE + 1))
    log "terra wl3 not running (confirm $SETTLE/3)"
    [ "$SETTLE" -ge 3 ] && break   # ~3 min stable: supervisor acts within 10s
  fi
  sleep 60
done
log "terra wl3 ended. launching supervisor P (wl5,7,9 on Plus relay :8787)"

# 2026-08-18 教训：ac_supervisor_watch.sh 看门狗与本 watcher 在 wl3 结束后
# 45 秒内各拉起了一个 plus supervisor -> wl5 双实例同 sandbox 双写，全晚进度
# 作废（cycle 1-59 x3 重复）。launch 前必须确认没有已存在的 plus supervisor。
plus_sup_pid() {
  for p in $(pgrep -f 'run_ac_luna_3'); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q 'ac_luna_3wl_v5_plus'; then
      echo "$p"; return 0
    fi
  done
  return 1
}
if plus_sup_pid >/dev/null; then
  log "plus supervisor already running (pid $(plus_sup_pid), watchdog won the race); skipping launch"
  exit 0
fi

cd "$AF" || { log "cd failed"; exit 1; }
AC_LUNA_RUN_DIR="$PLUS_DIR" \
AC_LUNA_ONLY=5,7,9 \
AC_LUNA_CONCURRENCY=1 \
AC_LUNA_WORLDLINES=9 \
setsid nohup /home/lxx/trade-agent-benchmark/.venv/bin/python -u -m scheduler.run_ac_luna_3 \
  >>"$PLUS_DIR/supervisor.log" 2>&1 < /dev/null &
log "supervisor P launched pid=$!"
sleep 20
if pgrep -f "main\.py wl5 --config .*/results/ac_luna_3wl_v5_plus/" >/dev/null 2>&1; then
  log "terra wl5 agent process confirmed"
else
  log "NOTE: terra wl5 not yet visible (may still be seeding); check supervisor.log"
fi
log "watcher done"
