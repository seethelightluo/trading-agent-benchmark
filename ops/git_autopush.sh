#!/usr/bin/env bash
set -Eeuo pipefail

exec "$(dirname "$0")/git_milestone_push.sh" "5h-autosave"
