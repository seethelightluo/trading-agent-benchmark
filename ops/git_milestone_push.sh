#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="${GIT_AUTOPUSH_REPO:-/home/lxx/trade-agent-benchmark}"
label="${1:-milestone}"
lock_wait="${GIT_MILESTONE_LOCK_WAIT:-120}"

case "$label" in
  (*[!A-Za-z0-9._-]*|'')
    echo "git-milestone: invalid label: $label" >&2
    exit 2
    ;;
esac

cd "$repo_dir"

# Periodic saves and milestone hooks share one Git index lock.
exec 9>"$repo_dir/.git/autopush.lock"
if ! flock -w "$lock_wait" 9; then
  echo "git-milestone: could not acquire Git lock within ${lock_wait}s" >&2
  exit 1
fi

if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
  echo "git-milestone: an unfinished rebase exists; refusing to continue" >&2
  exit 1
fi

compress_large_ac_logs() {
  local log_dir="$repo_dir/AC-deepseek/results/ac9wl_deepseek/logs"
  if [ ! -d "$log_dir" ]; then
    return 0
  fi

  while IFS= read -r -d '' source_log; do
    local compressed="${source_log}.gz"
    if [ ! -s "$compressed" ] || [ "$source_log" -nt "$compressed" ]; then
      echo "git-milestone: compressing oversized result log: $source_log" >&2
      gzip -c "$source_log" > "${compressed}.tmp"
      gzip -t "${compressed}.tmp"
      mv "${compressed}.tmp" "$compressed"
    fi
    if [ "$(stat -c '%s' "$compressed")" -gt 95000000 ]; then
      echo "git-milestone: compressed result log still exceeds GitHub limit: $compressed" >&2
      return 1
    fi
  done < <(find "$log_dir" -maxdepth 1 -type f -name 'wl*.log' -size +90M -print0)

  # .gitignore does not untrack files already present in an older index.
  while IFS= read -r -d '' tracked_log; do
    echo "git-milestone: untracking raw result log (gzip checkpoint retained): $tracked_log" >&2
    git rm --cached -- "$tracked_log" >/dev/null
  done < <(git ls-files -z -- "$log_dir/wl*.log")
}

# Commit locally before any network operation.  Thus a transient remote outage
# never removes the local recovery point.  .gitignore is the boundary for
# environments and credentials; every other result, log, script, and document
# is intentionally included.
compress_large_ac_logs
git add -A
if ! git diff --cached --quiet; then
  git commit -m "chore: milestone ${label}"
else
  echo "git-milestone: no local changes for ${label}"
fi

retry_git() {
  local attempt=1
  local max_attempts=3
  local delay
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      return 1
    fi
    delay=$((attempt * 10))
    echo "git-milestone: command failed; retrying in ${delay}s (${attempt}/${max_attempts})" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

if ! retry_git git fetch --prune origin main; then
  echo "git-milestone: local checkpoint kept; fetch origin/main failed" >&2
  exit 1
fi

remote_head="$(git rev-parse refs/remotes/origin/main)"
local_head="$(git rev-parse HEAD)"
if [ "$local_head" != "$remote_head" ]; then
  if git merge-base --is-ancestor "$remote_head" "$local_head"; then
    :
  elif git merge-base --is-ancestor "$local_head" "$remote_head"; then
    git merge --ff-only "$remote_head"
  else
    echo "git-milestone: local and origin/main diverged; manual merge required" >&2
    exit 1
  fi
fi

if ! retry_git git push origin HEAD:main; then
  echo "git-milestone: local checkpoint kept; push failed for ${label}" >&2
  exit 1
fi

echo "git-milestone: pushed ${label} at $(git rev-parse --short HEAD)"
