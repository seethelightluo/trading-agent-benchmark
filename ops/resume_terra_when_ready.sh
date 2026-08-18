#!/usr/bin/env bash
# One-shot: wait until the terra (gpt-5.6-luna) upstream is usable again,
# then remove the pause guard so the 15-min watchdog launches the wl1-3 supervisor.
set -u
ROOT=/home/lxx/trade-agent-benchmark
TERRA_RESULTS=$ROOT/agent-framework/results/ac_luna_3wl_v5
LOG=$ROOT/ops/logs/resume_terra_when_ready.log
RELAY_KEY=$(grep UPSTREAM_API_KEY /home/lxx/ac-llm-relay/relay.env | cut -d= -f2)
mkdir -p "$(dirname "$LOG")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

probe() {
  curl -sS -m 60 -o /tmp/terra_probe.json -w '%{http_code}' \
    http://127.0.0.1:8787/v1/responses \
    -H "Authorization: Bearer $RELAY_KEY" -H "Content-Type: application/json" \
    -d '{"model":"gpt-5.6-terra","input":"Reply with: ok","max_output_tokens":8}' 2>/dev/null
}

# probe loop: try every 10 minutes for up to 8 hours
for i in $(seq 1 48); do
  code=$(probe)
  if [ "$code" = "200" ]; then
    echo "$(ts) terra upstream OK (HTTP 200); removing pause guard" >> "$LOG"
    rm -f "$TERRA_RESULTS/.pause_terra_watch"
    echo "$(ts) guard removed; watchdog will launch supervisor within 15 min" >> "$LOG"
    exit 0
  fi
  echo "$(ts) attempt $i: terra upstream probe HTTP $code; retrying in 10 min" >> "$LOG"
  sleep 600
done
echo "$(ts) gave up after 48 attempts; guard still in place" >> "$LOG"
exit 1
