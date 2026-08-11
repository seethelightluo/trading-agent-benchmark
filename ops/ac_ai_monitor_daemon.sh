#!/bin/bash
# AC AI Monitor daemon: adaptive interval, wakes Codex on anomaly.
# 30min baseline, ×2 on healthy (max 4h), reset to 30min on anomaly.
# Runs as systemd user service with linger=yes (survives SSH disconnect).

set -euo pipefail

STATE_DIR="/data/ac-monitor"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/state.json"
LOG_FILE="$STATE_DIR/monitor.log"
BASE=1800
MAXI=14400
CODEX_BIN="/snap/bin/codex"
WORKDIR="/home/lxx/trade-agent-benchmark"

[ ! -f "$STATE_FILE" ] && echo '{"interval":'$BASE',"healthy_streak":0}' > "$STATE_FILE"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

do_cycle() {
    ISSUES=""

    # 1. Supervisors alive?
    DS_ALIVE=$(pgrep -f "run_deepseek_ac9" 2>/dev/null | head -1)
    TE_ALIVE=$(pgrep -f "run_ac_luna_3" 2>/dev/null | head -1)
    [ -z "$DS_ALIVE" ] && ISSUES="${ISSUES}DS supervisor dead\n"
    [ -z "$TE_ALIVE" ] && ISSUES="${ISSUES}Terra supervisor dead\n"

    # 2. WL processes running?
    DS_WL=$(pgrep -f "main.py wl[0-9].*ac9wl_deepseek" 2>/dev/null | wc -l)
    TE_WL=$(pgrep -f "main.py wl[0-9].*luna_3wl_v5" 2>/dev/null | wc -l)
    [ "$DS_WL" -lt 1 ] && ISSUES="${ISSUES}DS no WL processes (${DS_WL})\n"
    [ "$TE_WL" -lt 1 ] && ISSUES="${ISSUES}Terra no WL processes (${TE_WL})\n"

    # 3. API health
    API_STATS=$(sudo docker logs sub2api --since 5m 2>&1 || true)
    API_OK=$(echo "$API_STATS" | grep -c '"status_code": 200' || true)
    API_ERR=$(echo "$API_STATS" | grep -cE '"status_code": (429|502|503)' || true)
    TOTAL=$((API_OK + API_ERR))
    if [ "$TOTAL" -gt 5 ]; then
        RATE=$((API_OK * 100 / TOTAL))
        [ "$RATE" -lt 70 ] && ISSUES="${ISSUES}API success rate ${RATE}% (<70%)\n"
    fi
    [ "$TOTAL" -eq 0 ] && ISSUES="${ISSUES}No API traffic\n"

    # 4. Framework errors
    DS_ERRS=0; TE_ERRS=0
    for wl in wl1 wl2 wl3; do
        DS_ERRS=$((DS_ERRS + $(tail -1000 "$WORKDIR/AC-deepseek/results/ac9wl_deepseek/logs/$wl.log" 2>/dev/null | grep -c "Traceback\|Unexpected error" || true)))
        TE_ERRS=$((TE_ERRS + $(tail -1000 "$WORKDIR/agent-framework/results/ac_luna_3wl_v5/logs/$wl.log" 2>/dev/null | grep -c "Traceback\|Unexpected error" || true)))
    done
    [ "$DS_ERRS" -gt 20 ] && ISSUES="${ISSUES}DS high errors ($DS_ERRS)\n"
    [ "$TE_ERRS" -gt 20 ] && ISSUES="${ISSUES}Terra high errors ($TE_ERRS)\n"

    # --- Report ---
    log "DS: sup=$([ -n "$DS_ALIVE" ] && echo ALIVE || echo DEAD) wls=$DS_WL errs=$DS_ERRS"
    log "TE: sup=$([ -n "$TE_ALIVE" ] && echo ALIVE || echo DEAD) wls=$TE_WL errs=$TE_ERRS"
    log "API: ${API_OK}ok/${API_ERR}err"

    if [ -z "$ISSUES" ]; then
        # Healthy
        STREAK=$(python3 -c "import json; s=json.load(open('$STATE_FILE')); s['healthy_streak']=s.get('healthy_streak',0)+1; s['interval']=min(s['interval']*2,$MAXI); json.dump(s,open('$STATE_FILE','w')); print(s['healthy_streak'])")
        NEW_INT=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['interval'])")
        log "All healthy (×${STREAK}), interval=${NEW_INT}s"
    else
        # Anomaly — wake Codex
        log "ANOMALY DETECTED:"
        echo -e "$ISSUES" | while read -r line; do [ -n "$line" ] && log "  - $line"; done
        log "Waking Codex for diagnosis + repair..."

        python3 -c "import json; s=json.load(open('$STATE_FILE')); s['healthy_streak']=0; s['interval']=$BASE; json.dump(s,open('$STATE_FILE','w'))"

        ISSUES_CLEAN=$(echo -e "$ISSUES" | tr '\n' ',' | sed 's/,$//;s/^,//')

        PROMPT="AC 实验监控检测到以下异常：${ISSUES_CLEAN}

请执行以下步骤：
1. 检查 DS (ac9wl_deepseek) 和 Terra (ac_luna_3wl_v5) 的 supervisor、WL 进程、日志尾部和 sub2api API 健康。
2. 如果是 supervisor 死亡，尝试从干净的 run_state 续跑（不要重跑 warmup）。
3. 如果是 API 503/429，检查 sub2api 账户冷却状态（psql UPDATE accounts SET temp_unschedulable_until=NULL）、redis 缓存（FLUSHALL）、代理端口（Clash 7897 + iptables NAT 172.18.0.1:7897）。
4. 如果是框架错误（Traceback），检查具体错误并修复代码。
5. 修复后验证 API 返回 200、WL 进程恢复运行。
6. 不要修改 warmup 数据或已有 online 结果，只做最小修复。
7. 修复完成后简短报告你做了什么。"

        cd "$WORKDIR"
        timeout 600 "$CODEX_BIN" exec --dangerously-bypass-approvals-and-sandbox "$PROMPT" >> "$LOG_FILE" 2>&1 || log "Codex repair timed out or failed (rc=$?)"
        log "Codex repair cycle done, interval reset to ${BASE}s"
    fi
}

log "=== AC AI Monitor daemon started (base=${BASE}s max=${MAXI}s) ==="

while true; do
    do_cycle || log "Cycle error (rc=$?)"
    SLEEP=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('interval',$BASE))")
    log "Sleeping ${SLEEP}s...\n"
    sleep "$SLEEP"
done
