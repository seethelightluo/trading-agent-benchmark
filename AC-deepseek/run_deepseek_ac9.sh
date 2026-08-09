#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lxx/trade-agent-benchmark/AC-deepseek
SECRET_ENV=${AC_DEEPSEEK_ENV_FILE:-/home/lxx/.config/alphacrafter/deepseek-sub2api.env}

if [[ -f "$SECRET_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$SECRET_ENV"
  set +a
fi
export OPENAI_API_URL=${OPENAI_API_URL:-http://127.0.0.1:8080/v1}
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "deepseek AC: sub2api API key is missing; configure $SECRET_ENV" >&2
  exit 1
fi

export AC_DATA_ROOT=${AC_DATA_ROOT:-/home/lxx/trade-agent-benchmark/WL-data-final}
cd "$ROOT"
exec /home/lxx/trade-agent-benchmark/.venv/bin/python -u "$ROOT/run_deepseek_ac9.py" "$@"
