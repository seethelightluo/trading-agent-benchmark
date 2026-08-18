#!/usr/bin/env bash
# Lightweight DS/Terra AC supervisor watchdog (cron, every 15 min).
# Restarts a dead supervisor after gracefully stopping orphaned WL workers,
# so queue rotation and failure recovery never stay down.
# Terra is fork-aware since 2026-08-17: v5(wl3/Plus) / _oc(wl4,6,8/opencode
# key2 :8788) / _plus(wl5,7,9/Plus) each get their own supervisor, concurrency 1.
set -u

ROOT=/home/lxx/trade-agent-benchmark
ROOT_VENV=$ROOT/.venv/bin/python
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
  AC_DEEPSEEK_CONCURRENCY=3 AC_DEEPSEEK_ONLINE_MAX_CYCLES=360 setsid nohup bash "$DS_SH" >> "$DS_RESULTS/supervisor.log" 2>&1 &
}

terra_sup_pid() {
  # $1 = run dir -> supervisor pid for that dir (env match), else empty
  for p in $(pgrep -f 'scheduler\.run_ac_luna_3' 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "AC_LUNA_RUN_DIR=$1"; then
      echo "$p"; return 0
    fi
  done
}

terra_runnable_wls() {
  # $1 = run dir, $2 = comma WL list -> prints WLs that are incomplete AND not
  # pause-marked (i.e. supervisor for this dir still has real work)
  local dir="$1" list="$2" out=""
  for w in ${list//,/ }; do
    [ -f "$dir/pause_wl${w}_429" ] && continue
    f="$ROOT/agent-framework/AlphaCrafter/alphacrafter/sandbox/wl${w}/persistent/date.json"
    if [ -f "$f" ] && grep -q '"simulation_complete": *true' "$f"; then continue; fi
    out="${out:+$out,}$w"
  done
  echo "$out"
}

stop_terra_orphans() {
  # $1 = run dir name (e.g. ac_luna_3wl_v5_oc) -> stop workers of that dir only
  local orphans
  orphans=$(ps -eo pid,cmd | rg "main\.py wl[0-9].*$1/run_config.yaml" | rg -v rg | awk '{print $1}')
  [ -z "$orphans" ] && return 0
  echo "$(ts) stopping orphaned Terra workers ($1): $orphans" >> "$LOG"
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
}

launch_terra_sup() {
  # $1 = run dir, $2 = AC_LUNA_ONLY list, $3 = extra env (e.g. OPENAI_API_URL) or "-"
  local dir="$1" only="$2" extra="$3"
  echo "$(ts) restarting Terra supervisor ($dir, ONLY=$only)" >> "$LOG"
  cd "$ROOT/agent-framework"
  if [ "$extra" = "-" ]; then
    AC_LUNA_RUN_DIR="$dir" AC_LUNA_ONLY="$only" AC_LUNA_WORLDLINES=9 \
      AC_LUNA_CONCURRENCY=1 AC_LUNA_ONLINE_MAX_CYCLES=360 \
      setsid nohup "$ROOT_VENV" -u -m scheduler.run_ac_luna_3 >> "$dir/supervisor.log" 2>&1 &
  else
    env "$extra" AC_LUNA_RUN_DIR="$dir" AC_LUNA_ONLY="$only" AC_LUNA_WORLDLINES=9 \
      AC_LUNA_CONCURRENCY=1 AC_LUNA_ONLINE_MAX_CYCLES=360 \
      setsid nohup "$ROOT_VENV" -u -m scheduler.run_ac_luna_3 >> "$dir/supervisor.log" 2>&1 &
  fi
}

restart_terra() {
  # Fork-aware: v5 owns wl3 (Plus), _oc owns wl4,6,8 (opencode key2 relay :8788),
  # _plus owns wl5,7,9 (Plus relay :8787). Each dir is checked independently;
  # only dirs with runnable incomplete WLs get a supervisor, per-source concurrency 1.
  local base="$ROOT/agent-framework/results"

  if [ -f "$TERRA_RESULTS/.pause_terra_watch" ]; then
    echo "$(ts) Terra v5 restart skipped (pause guard)" >> "$LOG"
  elif [ -z "$(terra_sup_pid "$TERRA_RESULTS")" ]; then
    runnable=$(terra_runnable_wls "$TERRA_RESULTS" "3")
    [ -n "$runnable" ] && { stop_terra_orphans "ac_luna_3wl_v5"; launch_terra_sup "$TERRA_RESULTS" "3" "-"; }
  fi

  OC_DIR="$base/ac_luna_3wl_v5_oc"
  if [ -z "$(terra_sup_pid "$OC_DIR")" ]; then
    runnable=$(terra_runnable_wls "$OC_DIR" "4,6,8")
    [ -n "$runnable" ] && { stop_terra_orphans "ac_luna_3wl_v5_oc"; launch_terra_sup "$OC_DIR" "4,6,8" "OPENAI_API_URL=http://127.0.0.1:8788/v1"; }
  fi

  PLUS_DIR="$base/ac_luna_3wl_v5_plus"
  # Chaining (per-source concurrency=1): Plus runs wl3 first; wl5,7,9 must wait
  # until wl3 is complete or pause-marked (v5 has no runnable work left).
  # chain_wl3_to_plus.sh does the prompt handover; this is the safety net.
  if [ -n "$(terra_runnable_wls "$TERRA_RESULTS" "3")" ]; then
    echo "$(ts) Terra plus dir deferred (wl3 still runnable on Plus)" >> "$LOG"
  elif [ -z "$(terra_sup_pid "$PLUS_DIR")" ]; then
    runnable=$(terra_runnable_wls "$PLUS_DIR" "5,7,9")
    [ -n "$runnable" ] && { stop_terra_orphans "ac_luna_3wl_v5_plus"; launch_terra_sup "$PLUS_DIR" "5,7,9" "-"; }
  fi
}

DS_ALIVE=$(pgrep -f 'run_deepseek_ac9\.py' 2>/dev/null | head -1)
TE_ALIVE=$(pgrep -f 'scheduler\.run_ac_luna_3' 2>/dev/null | head -1)

[ -z "$DS_ALIVE" ] && restart_ds
restart_terra   # fork-aware: no-ops for dirs that are alive or have no runnable WL

exit 0
