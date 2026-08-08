#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="${GIT_AUTOPUSH_REPO:-/home/lxx/trade-agent-benchmark}"
cd "$repo_dir"

# Prevent overlapping timer/manual runs from racing over the index.
exec 9>"$repo_dir/.git/autopush.lock"
if ! flock -n 9; then
  echo "git-autopush: another run is already active"
  exit 0
fi

if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
  echo "git-autopush: an unfinished rebase exists; refusing to continue"
  exit 1
fi

git fetch --prune origin main

# Track every non-ignored source/data/document change, including deletions.
git add -A
if ! git diff --cached --quiet; then
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git commit -m "chore: autosave workspace ${stamp}"
fi

remote_head="$(git rev-parse refs/remotes/origin/main)"
local_head="$(git rev-parse HEAD)"
if [ "$local_head" != "$remote_head" ]; then
  if git merge-base --is-ancestor "$local_head" "$remote_head"; then
    git merge --ff-only "$remote_head"
  else
    # Preserve remote commits and stop visibly if a real conflict needs review.
    git rebase "$remote_head"
  fi
fi

git push origin HEAD:main
echo "git-autopush: pushed $(git rev-parse --short HEAD)"
