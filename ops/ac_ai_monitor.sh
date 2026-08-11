#!/bin/bash
# AC AI Monitor: script does fast health check; on anomaly, wakes Codex for intelligent diagnosis+repair.
# Adaptive interval: 30min baseline, ×2 on healthy (max 4h), reset on anomaly.

set -euo pipefail

STATE_DIR="/data/ac-monitor"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/state.json"
LOG_FILE="$STATE_DIR/monitor.log"
BASE=1800
MAXI=14400
CODEX_BIN="/snap/bin/codex"
WORKDIR="/home/lxx/trade-agent-benchmark"

if [ ! -f "$STATE_FILE" ]; then
    echo '{"interval":'$BASE',"healthy_streak":0}' > "$STATE_FILE"
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

INTERVAL=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('interval',$BASE))")

log "=== AC AI Monitor cycle (interval=${INTERVAL}s) ==="

# --- Fast script-level health check ---
ISSUES=""

# 1. Supervisors alive?
DS_ALIVE=$(pgrep -f "run_deepseek_ac9" | head -1)
TE_ALIVE=$(pgrep -f "run_ac_luna_3" | head -1)
[ -z "$DS_ALIVE" ] && ISSUES="$ISSUES|DS supervisor dead"
[ -z "$TE_ALIVE" ] && ISSUES="$ISSUES|Terra supervisor dead"

# 2. WL processes running?
DS_WL=$(pgrep -f "main.py wl[0-9].*ac9wl_deepseek" | wc -l)
TE_WL=$(pgrep -f "main.py wl[0-9].*luna_3wl_v5" | wc -l)
[ "$DS_WL" -lt 1 ] && ISSUES="$ISSUES|DS no WL processes"
[ "$TE_WL" -lt 1 ] && ISSUES="$ISSUES|Terra no WL processes"

# 3. API health (last 5 min)
API_STATS=$(sudo docker logs sub2api --since 5m 2>&1)
API_OK=$(echo "$API_STATS" | grep -c '"status_code": 200' || true)
API_ERR=$(echo "$API_STATS" | grep -cE '"status_code": (429|502|503)' || true)
TOTAL=$((API_OK + API_ERR))
if [ "$TOTAL" -gt 5 ]; then
    RATE=$((API_OK * 100 / TOTAL))
    [ "$RATE" -lt 70 ] && ISSUES="$ISSUES|API success rate ${RATE}% (<70%)"
fi
[ "$TOTAL" -eq 0 ] && ISSUES="$ISSUES|no API traffic"

# 4. Recent framework errors in logs
DS_LOG_DIR="$WORKDIR/AC-deepseek/results/ac9wl_deepseek/logs"
TE_LOG_DIR="$WORKDIR/agent-framework/results/ac_luna_3wl_v5/logs"
DS_ERRS=0; TE_ERRS=0
for wl in wl1 wl2 wl3; do
    [ -f "$DS_LOG_DIR/$wl.log" ] && DS_ERRS=$((DS_ERRS + $(tail -1000 "$DS_LOG_DIR/$wl.log" 2>/dev/null | grep -c "Traceback\|Unexpected error" || true)))
    [ -f "$TE_LOG_DIR/$wl.log" ] && TE_ERRS=$((TE_ERRS + $(tail -1000 "$TE_LOG_DIR/$wl.log" 2>/dev/null | grep -c "Traceback\|Unexpected error" || true)))
done
[ "$DS_ERRS" -gt 20 ] && ISSUES="$ISSUES|DS high errors ($DS_ERRS)"
[ "$TE_ERRS" -gt 20 ] && ISSUES="$ISSUES|Terra high errors ($TE_ERRS)"

# --- Report current status ---
log "DS: supervisor=$([ -n "$DS_ALIVE" ] && echo ALIVE || echo DEAD) wls=$DS_WL api_errs=$DS_ERRS"
log "TE: supervisor=$([ -n "$TE_ALIVE" ] && echo ALIVE || echo DEAD) wls=$TE_WL api_errs=$TE_ERRS"
log "API: ${API_OK}ok/${API_ERR}err"

if [ -z "$ISSUES" ]; then
    # Healthy
    STREAK=$(python3 -c "import json; s=json.load(open('$STATE_FILE')); s['healthy_streak']=s.get('healthy_streak',0)+1; s['interval']=min(s['interval']*2,$MAXI); json.dump(s,open('$STATE_FILE','w')); print(s['healthy_streak'])")
    NEW_INT=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['interval'])")
    log "All healthy (×${STREAK}), next interval=${NEW_INT}s"
    exit 0
fi

# --- Anomaly detected: wake Codex for intelligent diagnosis + repair ---
log "ISSUES DETECTED: $ISSUES"
log "Waking Codex for intelligent diagnosis and repair..."

STREAK=0
python3 -c "import json; s=json.load(open('$STATE_FILE')); s['healthy_streak']=0; s['interval']=$BASE; json.dump(s,open('$STATE_FILE','w'))"

ISSUES_CLEAN=$(echo "$ISSUES" | sed 's/^|//;s/|/, /g')

PROMPT="AC 实验监控检测到以下异常：${ISSUES_CLEAN}

请执行以下步骤：
1. 检查 DS (ac9wl_deepseek) 和 Terra (ac_luna_3wl_v5) 的 supervisor、WL 进程、日志尾部和 sub2api API 健康。
2. 如果是 supervisor 死亡，尝试从干净的 run_state 续跑（不要重跑 warmup）。
3. 如果是 API 503/429，检查 sub2api 账户冷却状态、redis 缓存、代理端口（Clash 7897 + iptables NAT 172.18.0.1:7897）。
4. 如果是框架错误（Traceback），检查具体错误并修复代码。
5. 修复后验证 API 返回 200、WL 进程恢复运行。
6. 不要修改 warmup 数据或已有 online 结果，只做最小修复。
7. 修复完成后报告你做了什么。"

cd "$WORKDIR"
timeout 600 $CODEX_BIN exec --dangerously-bypass-approvals-and-sandbox "$PROMPT" >> "$LOG_FILE" 2>&1 || log "Codex repair timed out or failed (rc=$?)"

log "Codex repair cycle done"
NEW_INT=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['interval'])")
log "Next check in ${NEW_INT}s"

exit 0
