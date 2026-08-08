#!/usr/bin/env python3
"""Merge a completed Windows FM WL4-WL9 export into this Linux checkout."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_WLS = [4, 5, 6, 7, 8, 9]
EXPECTED_FINGERPRINT = "b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c"


def rewrite_paths(value, source_agent: str, target_agent: str):
    if isinstance(value, dict):
        return {key: rewrite_paths(item, source_agent, target_agent) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, source_agent, target_agent) for item in value]
    if isinstance(value, str):
        variants = {
            source_agent,
            source_agent.replace("\\", "/"),
            source_agent.replace("/", "\\"),
        }
        for source in variants:
            value = value.replace(source, target_agent)
        return value
    return value


def rewrite_json_tree(root: Path, source_agent: str, target_agent: str) -> None:
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        path.write_text(
            json.dumps(
                rewrite_paths(payload, source_agent, target_agent),
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )


def load_export(root: Path) -> dict:
    manifest_path = root / "export_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"not an FM portable export: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("warmup_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError("export warm-up fingerprint does not match the current FM shared contract")
    if list(manifest.get("worldlines", [])) != EXPECTED_WLS:
        raise ValueError(f"export must contain exactly WL4-WL9, got {manifest.get('worldlines')}")
    for wl in EXPECTED_WLS:
        worldline_root = root / "fm" / f"WL{wl}"
        if not worldline_root.is_dir():
            raise FileNotFoundError(f"missing exported FM result for WL{wl}")
        seeds = list(worldline_root.rglob("online_mining/seed_manifest.json"))
        if len(seeds) != 1:
            raise ValueError(f"WL{wl} export must contain exactly one online seed manifest")
        seed = json.loads(seeds[0].read_text(encoding="utf-8"))
        if seed.get("warmup_fingerprint") != EXPECTED_FINGERPRINT:
            raise ValueError(f"WL{wl} seed was not derived from the verified shared warm-up")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="zip created by export_results.ps1")
    parser.add_argument("--source", type=Path, help="already-extracted export directory")
    parser.add_argument("--target-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if bool(args.archive) == bool(args.source):
        parser.error("specify exactly one of --archive or --source")

    temporary = None
    if args.archive:
        if not args.archive.is_file():
            raise FileNotFoundError(args.archive)
        temporary = tempfile.TemporaryDirectory(prefix="fm-windows-import-")
        source_root = Path(temporary.name)
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(source_root)
    else:
        source_root = args.source.resolve()
    try:
        manifest = load_export(source_root)
        target_root = args.target_root.resolve()
        target_agent = target_root / "agent-framework"
        target_fm = target_agent / "results" / "fm"
        if not target_agent.is_dir():
            raise FileNotFoundError(f"not a benchmark checkout: {target_agent}")
        destinations = [target_fm / f"WL{wl}" for wl in EXPECTED_WLS]
        nonempty = [str(path) for path in destinations if path.exists() and any(path.iterdir())]
        if nonempty:
            raise FileExistsError("refusing to overwrite existing WL outputs: " + "; ".join(nonempty))

        staging = target_fm.parent / f".fm_windows_import_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for wl in EXPECTED_WLS:
                shutil.copytree(source_root / "fm" / f"WL{wl}", staging / f"WL{wl}")
            rewrite_json_tree(
                staging,
                str(manifest["source_agent_framework"]),
                str(target_agent),
            )
            target_fm.mkdir(parents=True, exist_ok=True)
            for wl in EXPECTED_WLS:
                (staging / f"WL{wl}").rename(target_fm / f"WL{wl}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        imported = target_agent / "results" / "fm_imported_states" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if (source_root / "states").is_dir():
            shutil.copytree(source_root / "states", imported)
            rewrite_json_tree(
                imported,
                str(manifest["source_agent_framework"]),
                str(target_agent),
            )
        receipt = target_agent / "results" / "fm_windows_import_receipt.json"
        receipt.write_text(json.dumps({
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_archive": str(args.archive) if args.archive else str(source_root),
            "worldlines": EXPECTED_WLS,
            "warmup_fingerprint": EXPECTED_FINGERPRINT,
            "imported_state_dir": str(imported) if imported.exists() else None,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("MERGE OK: imported WL4-WL9; receipt=" + str(receipt))
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
