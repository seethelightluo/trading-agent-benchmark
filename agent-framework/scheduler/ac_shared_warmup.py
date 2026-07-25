"""Shared AlphaCrafter warm-up and worldline seeding helpers.

The nine crisis worldlines have byte-identical research history through
2026-07-15.  AlphaCrafter therefore performs one frozen-capital research cycle,
persists its factor workspace and registered strategy, and seeds that immutable
research result into each independent worldline account.  The seeded strategy
executes the first 10-day block locally before any WL-specific LLM cycle.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def workspace_digest(workspace: Path) -> str:
    """Hash persisted research files while ignoring interpreter caches."""
    digest = hashlib.sha256()
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(workspace).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def workflow_cycle_complete(session_dir: Path, miner_ids: list[str]) -> bool:
    """Verify one complete Miner/Screener/Trader warm-up cycle."""
    path = session_dir / "logs" / "workflow.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(entries, list):
        return False
    phases = {
        str(item.get("phase")): bool(item.get("success"))
        for item in entries
        if item.get("cycle") == 1
    }
    if not all(phases.get(name) for name in ("screener", "trader")):
        return False
    for miner_id in miner_ids:
        miner_id = str(miner_id)
        # Upstream configs commonly name an id ``miner_1`` while workflow
        # logging prepends ``miner_`` again.  Accept both the upstream double
        # prefix and the normalized form so older sessions remain resumable.
        candidates = {miner_id, f"miner_{miner_id}"}
        if not any(phases.get(candidate) for candidate in candidates):
            return False
    return True


def validate_warmup_workspace(session_dir: Path) -> dict:
    """Return auditable artifact metadata or raise on an unusable warm-up."""
    workspace = session_dir / "workspace"
    strategy = workspace / "strategy.py"
    if not strategy.exists():
        raise ValueError(f"shared AC warm-up did not persist strategy: {strategy}")
    strategy_text = strategy.read_text(encoding="utf-8")
    if "@register_hook" not in strategy_text:
        raise ValueError("shared AC warm-up strategy has no registered execution hook")
    factor_files = sorted((workspace / "factors").glob("*.json"))
    for factor_file in factor_files:
        try:
            json.loads(factor_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid factor artifact: {factor_file}") from exc
    return {
        "workspace_digest": workspace_digest(workspace),
        "strategy_sha256": hashlib.sha256(strategy.read_bytes()).hexdigest(),
        "factor_files": [path.name for path in factor_files],
        "factor_count": len(factor_files),
        "status": "ready" if factor_files else "cash_strategy_no_admitted_factors",
    }


def archive_session(session_dir: Path, archive_dir: Path, label: str) -> Path | None:
    """Preserve an old generated session before a contract-driven rebuild."""
    if not session_dir.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{session_dir.name}_{label}.tar.gz"
    if archive.exists():
        return archive
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(session_dir, arcname=session_dir.name)
    return archive


def seed_worldline_workspace(
    warmup_session_dir: Path,
    target_session_dir: Path,
    *,
    warmup_fingerprint: str,
    baseline_date: str,
    history_end: str,
    initial_capital: float,
) -> Path:
    """Copy frozen research artifacts into one untouched WL account."""
    marker = target_session_dir / "persistent" / "shared_warmup_seed.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("warmup_fingerprint") == warmup_fingerprint:
            return marker
        raise ValueError(f"worldline already seeded by another warm-up: {marker}")

    date_path = target_session_dir / "persistent" / "date.json"
    account_path = target_session_dir / "persistent" / "account.json"
    date_state = json.loads(date_path.read_text(encoding="utf-8"))
    account = json.loads(account_path.read_text(encoding="utf-8"))
    if (
        date_state.get("current_date") != baseline_date
        or date_state.get("visible_through") != history_end
        or date_state.get("simulation_complete")
    ):
        raise ValueError("AC worldline must be at the untouched forward boundary before seeding")
    if (
        float(account.get("initial_capital", 0.0)) != float(initial_capital)
        or float(account.get("available_cash", 0.0)) != float(initial_capital)
        or account.get("positions")
        or account.get("orders")
    ):
        raise ValueError("AC worldline account must be frozen 100% cash before seeding")

    logs_dir = target_session_dir / "logs"
    if logs_dir.exists() and any(path.is_file() and path.stat().st_size for path in logs_dir.rglob("*")):
        raise ValueError("AC worldline already has workflow logs; refusing to overwrite research state")

    source = warmup_session_dir / "workspace"
    target = target_session_dir / "workspace"
    if not source.is_dir():
        raise ValueError(f"shared AC warm-up workspace is missing: {source}")
    staged = target_session_dir / f"workspace.seed.{os.getpid()}.tmp"
    backup = target_session_dir / f"workspace.preseed.{os.getpid()}.bak"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(
        source,
        staged,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if target.exists():
        target.rename(backup)
    try:
        staged.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)
    payload = {
        "schema_version": 1,
        "warmup_fingerprint": warmup_fingerprint,
        "warmup_workspace_digest": workspace_digest(source),
        "seeded_workspace_digest": workspace_digest(target),
        "first_forward_block_complete": False,
        "first_forward_block_start_date": baseline_date,
    }
    _atomic_json(marker, payload)
    return marker


def execute_seeded_first_block(
    target_session_dir: Path,
    *,
    cadence: int,
    python: Path,
    ac_repo: Path,
    env: dict,
) -> dict:
    """Run the shared strategy locally for WL days 1..10, then mark resume."""
    marker_path = target_session_dir / "persistent" / "shared_warmup_seed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("first_forward_block_complete"):
        return marker

    date_path = target_session_dir / "persistent" / "date.json"
    date_state = json.loads(date_path.read_text(encoding="utf-8"))
    trading_days = list(date_state.get("trading_days") or [])
    start_date = str(marker.get("first_forward_block_start_date") or "")
    if start_date not in trading_days:
        raise ValueError(f"seeded first-block start date is not tradable: {start_date}")
    start_idx = trading_days.index(start_date)
    target_idx = min(start_idx + int(cadence) - 1, len(trading_days) - 1)
    target_date = trading_days[target_idx]
    saved_cadence = marker.get("first_forward_block_cadence")
    if saved_cadence is not None and int(saved_cadence) != int(cadence):
        raise ValueError(
            "cannot change AC cadence while the seeded first block is incomplete: "
            f"saved={saved_cadence}, requested={cadence}"
        )
    marker["first_forward_block_cadence"] = int(cadence)
    marker["first_forward_block_target_date"] = target_date

    visible_through = date_state.get("visible_through")
    if visible_through in trading_days and trading_days.index(visible_through) >= target_idx:
        account = json.loads(
            (target_session_dir / "persistent" / "account.json").read_text(
                encoding="utf-8"
            )
        )
        marker.update({
            "first_forward_block_complete": True,
            "completed_through": visible_through,
            "next_execution_date": date_state.get("current_date"),
            "nav_after_first_block": account.get("net_assets"),
            "cash_after_first_block": account.get("available_cash"),
            "positions_after_first_block": len(account.get("positions", [])),
        })
        _atomic_json(marker_path, marker)
        return marker

    current_date = date_state.get("current_date")
    if current_date not in trading_days:
        raise ValueError(f"seeded AC current date is not tradable: {current_date}")
    current_idx = trading_days.index(current_date)
    if current_idx < start_idx or current_idx > target_idx:
        raise ValueError(
            "seeded AC cursor escaped the first-block boundary: "
            f"current={current_date}, target={target_date}"
        )
    remaining_days = target_idx - current_idx + 1
    _atomic_json(marker_path, marker)

    command = [
        str(python),
        "-c",
        (
            "from agent.toolkit.step import StepTool; "
            f"result=StepTool().get_implementation()(days={remaining_days}); "
            "print(result); "
            "raise SystemExit(1 if result.startswith('Error during step execution:') else 0)"
        ),
    ]
    child_env = env.copy()
    child_env.pop("AC_WARMUP_ONLY", None)
    child_env["AC_CADENCE_DAYS"] = str(remaining_days)
    child_env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ac_repo),
            str(ac_repo.parent),
            child_env.get("PYTHONPATH"),
        )
        if part
    )
    completed = subprocess.run(
        command,
        cwd=target_session_dir / "workspace",
        env=child_env,
        text=True,
        capture_output=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "seeded AC first block failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )

    date_state = json.loads(date_path.read_text(encoding="utf-8"))
    account = json.loads(
        (target_session_dir / "persistent" / "account.json").read_text(encoding="utf-8")
    )
    if date_state.get("visible_through") != target_date:
        raise RuntimeError(
            "seeded AC first block stopped at the wrong boundary: "
            f"expected={target_date}, actual={date_state.get('visible_through')}"
        )
    marker.update({
        "first_forward_block_complete": True,
        "completed_through": date_state.get("visible_through"),
        "next_execution_date": date_state.get("current_date"),
        "nav_after_first_block": account.get("net_assets"),
        "cash_after_first_block": account.get("available_cash"),
        "positions_after_first_block": len(account.get("positions", [])),
    })
    _atomic_json(marker_path, marker)
    return marker
