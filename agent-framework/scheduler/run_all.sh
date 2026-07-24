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
# 停止：pkill -f run_pipeline.py ; pkill -f "main.py --session_id"
set -u
cd /home/lxx/trade-agent-benchmark
VENV=/home/lxx/trade-agent-benchmark/.venv/bin/python
LOG=agent-framework/results/run_pipeline.log
mkdir -p agent-framework/results

while true; do
  echo "$(date '+%F %T') === 拉起 run_pipeline: $* ===" >> "$LOG"
  setsid "$VENV" -m scheduler.run_pipeline "$@" >> "$LOG" 2>&1
  rc=$?
  echo "$(date '+%F %T') run_pipeline 退出 rc=$rc；10s 后重拉（state 跳过已完成 WL）" >> "$LOG"
  sleep 10
done
