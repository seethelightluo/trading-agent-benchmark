#!/usr/bin/env bash
set -euo pipefail

SOURCE=/data/trade-agent-benchmark/FM-live
SNAPSHOT_ROOT=/data/trade-agent-benchmark/FM-live-snapshots

if [[ ! -d "$SOURCE" ]]; then
  echo "fm-live snapshot: source is missing: $SOURCE" >&2
  exit 1
fi

install -d -m 0755 "$SNAPSHOT_ROOT"
exec 9>"$SNAPSHOT_ROOT/.snapshot.lock"
if ! flock -n 9; then
  echo "fm-live snapshot: another snapshot is already running" >&2
  exit 0
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
snapshot="$SNAPSHOT_ROOT/snapshot-$stamp"
temporary="$SNAPSHOT_ROOT/.snapshot-$stamp.tmp"

if [[ -e "$snapshot" || -e "$temporary" ]]; then
  echo "fm-live snapshot: destination already exists" >&2
  exit 1
fi

latest=$(find "$SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'snapshot-*' \
  -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; exit }')

echo "fm-live snapshot: creating $snapshot"
if [[ -n "$latest" ]]; then
  rsync -aHAX --numeric-ids --delete --link-dest="$latest" "$SOURCE/" "$temporary/"
else
  rsync -aHAX --numeric-ids --delete "$SOURCE/" "$temporary/"
fi

mv "$temporary" "$snapshot"

mapfile -t stale_snapshots < <(
  find "$SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'snapshot-*' \
    -printf '%T@ %p\n' | sort -nr | awk 'NR > 5 { sub(/^[^ ]+ /, ""); print }'
)
for stale in "${stale_snapshots[@]}"; do
  [[ -n "$stale" && -d "$stale" ]] || continue
  find "$stale" -depth -type f -delete
  find "$stale" -depth -type l -delete
  find "$stale" -depth -type d -empty -delete
done

count=$(find "$SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'snapshot-*' | wc -l)
echo "fm-live snapshot: complete; retained snapshots=$count"
