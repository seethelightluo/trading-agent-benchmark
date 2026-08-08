"""Offline, lossless window-state migration for the Windows FM bundle.

This helper exists because a P0/P1 performance-only source upgrade changes the
warm-up fingerprint while completed online ``window_state.json`` files retain
that fingerprint as part of their resume contract.  It changes *only* that
field after checking the persisted performance-equivalence certificate.

It deliberately does not modify scheduler state, seed manifests, libraries,
memory, checkpoints, combinations, forward results, market data, or runtime
credentials.  ``run_pipeline --fm-performance-equivalent-from`` performs the
scheduler/seed half of the audited migration immediately afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "bundle" / "agent-framework"
RUNTIME = ROOT / "runtime"
SOURCE = "b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c"
TARGET = "8410ae8bbd86fd8735de5ea4823e4924cebf977e51e2946854378fba46018c28"


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".migration.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_certificate() -> Path:
    path = (
        AGENT
        / "results"
        / "fm"
        / "performance_equivalence"
        / f"{SOURCE}_to_{TARGET}.json"
    )
    certificate = load_json(path)
    if (
        certificate.get("kind") != "fm_performance_equivalence"
        or certificate.get("source_warmup_fingerprint") != SOURCE
        or certificate.get("target_warmup_fingerprint") != TARGET
        or not isinstance(certificate.get("checks"), dict)
        or certificate["checks"].get("passed") is not True
    ):
        raise RuntimeError(f"invalid or unapproved performance certificate: {path}")
    return path


def migrate_worldline(worldline: int, certificate: Path) -> int:
    fm_root = AGENT / "results" / "fm" / f"WL{worldline}"
    if not fm_root.is_dir():
        print(f"WL{worldline}: no prior output; window migration not needed")
        return 0

    online_roots = sorted(path for path in fm_root.rglob("online_mining") if path.is_dir())
    if len(online_roots) > 1:
        raise RuntimeError(f"WL{worldline}: expected at most one online_mining root, got {online_roots}")
    if not online_roots:
        print(f"WL{worldline}: no online mining state; window migration not needed")
        return 0

    online_root = online_roots[0]
    seed_path = online_root / "seed_manifest.json"
    if seed_path.exists():
        seed = load_json(seed_path)
        seed_fingerprint = seed.get("warmup_fingerprint")
        if seed_fingerprint not in {SOURCE, TARGET}:
            raise RuntimeError(
                f"WL{worldline}: seed fingerprint is not part of this audited bridge: {seed_path}"
            )

    states = sorted((online_root / "windows").glob("*/window_state.json"))
    to_change: list[tuple[Path, dict, bytes]] = []
    for path in states:
        raw = path.read_bytes()
        state = load_json(path)
        name = path.parent.name
        try:
            index_text, cutoff = name.split("_", 1)
        except ValueError as exc:
            raise RuntimeError(f"WL{worldline}: malformed window directory: {path.parent}") from exc
        if state.get("window_index") != int(index_text) or state.get("decision_cutoff") != cutoff:
            raise RuntimeError(f"WL{worldline}: window contract/name mismatch: {path}")
        fingerprint = state.get("warmup_fingerprint")
        if fingerprint == TARGET:
            continue
        if fingerprint != SOURCE:
            raise RuntimeError(f"WL{worldline}: unexpected window fingerprint in {path}: {fingerprint!r}")
        to_change.append((path, state, raw))

    if not to_change:
        print(f"WL{worldline}: all {len(states)} window contracts already use {TARGET[:16]}")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = RUNTIME / "state" / "performance_equivalence_backups" / stamp / f"WL{worldline}"
    audit = {
        "schema_version": 1,
        "kind": "fm_performance_equivalence_window_state_correction",
        "source_warmup_fingerprint": SOURCE,
        "target_warmup_fingerprint": TARGET,
        "certificate": str(certificate),
        "corrected_at_utc": datetime.now(timezone.utc).isoformat(),
        "backup_root": str(backup_root),
        "states": [],
    }
    for path, state, raw in to_change:
        relative = path.relative_to(online_root)
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        state["warmup_fingerprint"] = TARGET
        state["performance_equivalence"] = {
            "kind": "fm_performance_equivalence",
            "source_warmup_fingerprint": SOURCE,
            "target_warmup_fingerprint": TARGET,
            "certificate": str(certificate),
            "corrected_at_utc": audit["corrected_at_utc"],
        }
        atomic_write_json(path, state)
        audit["states"].append({
            "path": str(path),
            "backup": str(backup),
            "sha256_before": hashlib.sha256(raw).hexdigest(),
        })
    atomic_write_json(online_root / "performance_equivalence_window_state_correction.json", audit)
    print(
        f"WL{worldline}: migrated {len(to_change)}/{len(states)} window contracts "
        f"{SOURCE[:16]} -> {TARGET[:16]}; originals backed up"
    )
    return len(to_change)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worldline", type=int, required=True, choices=range(1, 10))
    args = parser.parse_args()
    certificate = verify_certificate()
    migrate_worldline(args.worldline, certificate)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"WINDOW_MIGRATION_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
