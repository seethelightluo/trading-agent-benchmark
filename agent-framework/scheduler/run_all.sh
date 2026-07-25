#!/usr/bin/env bash
# run_all.sh — 断电可恢复的 live 跑批守护
#
# 三层韧性：
#   1. setsid + nohup：脱离终端，关 SSH/关 IDE 不影响（进程继续）。
#   2. 内层 while：run_pipeline 崩溃/非零退出 → 10s 后自动重拉（state.json 跳过已完成 WL，
#      AC 内部 --resume 续跑未完成 cycle）。
#   3. 断电/重启：配 systemd（见 run_pipeline.service）或 cron @reboot 重新拉起本脚本。
#
# 启动（后台、断终端）：
#   setsid nohup bash agent-framework/scheduler/run_all.sh --mode both --cadence 10 \
#       > agent-framework/results/run_all.log 2>&1 &
#   disown
#
# 看进度：tail -f agent-framework/results/run_pipeline.log
# 停止：pkill -f run_pipeline.py ; pkill -f "main.py wl"
set -u
cd /home/lxx/trade-agent-benchmark/agent-framework
VENV=/home/lxx/trade-agent-benchmark/.venv/bin/python
LOG=results/run_pipeline.log
mkdir -p results

# 导出 LLM 凭证（AC 自带 load_dotenv；FM 读环境变量，故需在此 export）
if [ -f AlphaCrafter/.env ]; then
  set -a; . AlphaCrafter/.env; set +a
fi
if [ -z "${OPENAI_API_URL:-}" ] || [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "$(date '+%F %T') live LLM 凭证为空，未启动跑批" >> "$LOG"
  exit 1
fi

while true; do
  echo "$(date '+%F %T') === 拉起 run_pipeline: $* ===" >> "$LOG"
  setsid "$VENV" -m scheduler.run_pipeline "$@" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%F %T') run_pipeline 已成功完成；守护进程正常退出" >> "$LOG"
    exit 0
  fi
  echo "$(date '+%F %T') run_pipeline 退出 rc=$rc；10s 后重拉（state 跳过已完成 WL）" >> "$LOG"
  sleep 10
done
