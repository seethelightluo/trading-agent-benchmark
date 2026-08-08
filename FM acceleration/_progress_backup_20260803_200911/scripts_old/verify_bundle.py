"""Offline integrity check for the FM Windows payload; makes no API request."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from portable_runner import AGENT_FRAMEWORK, FACTOR_MINER, load_pipeline

EXPECTED_FINGERPRINT = "b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c"
PROFILE = "live_cap1000000_atomicxfer1_ic0p007_ir0p084_rho0p5_i200_t110_b40"


def main() -> int:
    p = load_pipeline()
    contract = p.load_benchmark_config()
    admission = p.factor_admission_contract(contract)
    panel_root = AGENT_FRAMEWORK.parent / "data-prepare" / "online-worldline"
    base_cfg = FACTOR_MINER / "factorminer" / "configs" / "fm_live.yaml"
    cadence, iterations, target, batch_size = 10, 200, 110, 40
    max_active = int(contract.get("max_active_factors", 10))
    initial_capital = float(contract.get("initial_capital_usd", 1_000_000.0))
    tradable_ids = {item["asset_id"] for item in contract["tradable"]}

    def profile_number(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    warmup_profile = "_".join((
        "live", f"cap{int(initial_capital)}", "atomicxfer1",
        f"ic{profile_number(admission['ic_threshold'])}",
        f"ir{profile_number(admission['icir_threshold'])}",
        f"rho{profile_number(admission['correlation_threshold'])}",
        f"i{iterations}", f"t{target}", f"b{batch_size}",
    ))
    if warmup_profile != PROFILE:
        raise RuntimeError(f"unexpected run profile: {warmup_profile}")
    research_digest = p._sha256_paths([
        path for path in (FACTOR_MINER / "factorminer").rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ])
    scheduler_digest = p._sha256_file(AGENT_FRAMEWORK / "scheduler" / "run_pipeline.py")
    manifest_path = (
        AGENT_FRAMEWORK / "results" / "fm" / "shared_warmup" / PROFILE
        / "warmup_2026-07-15" / EXPECTED_FINGERPRINT[:16]
        / "shared_warmup_manifest.json"
    )
    required = (
        manifest_path,
        manifest_path.parent / "factor_library.json",
        manifest_path.parent / "factor_library_signals.npz",
        manifest_path.parent / "checkpoint" / "library.json",
        manifest_path.parent / "checkpoint" / "memory.json",
        manifest_path.parent / "combination_results.json",
        manifest_path.parent / "window.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing bundle artifacts: " + "; ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("warmup_fingerprint") != EXPECTED_FINGERPRINT:
        raise RuntimeError("bundle shared warm-up manifest fingerprint is wrong")

    verified = []
    for wl in range(4, 10):
        panel = panel_root / f"WL{wl}_full.parquet"
        if not panel.exists():
            raise FileNotFoundError(panel)
        warmup_cutoff, _ = p.fm_window_cutoffs(
            panel, contract["baseline_date"], contract["online_end"], cadence
        )
        payload = {
            "history_digest": p.fm_history_digest(panel, warmup_cutoff, tradable_ids),
            "base_config_sha256": p._sha256_file(base_cfg),
            "research_code_sha256": research_digest,
            "scheduler_code_sha256": scheduler_digest,
            "assets_sha256": p._sha256_file(AGENT_FRAMEWORK / "ASSETS.yaml"),
            "warmup_cutoff": warmup_cutoff,
            "warmup_profile": warmup_profile,
            "cadence_days": cadence,
            "max_active": max_active,
            "initial_capital": initial_capital,
            "iterations": iterations,
            "target": target,
            "batch_size": batch_size,
            "factor_admission": admission,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if fingerprint != EXPECTED_FINGERPRINT:
            raise RuntimeError(
                f"WL{wl} warm-up fingerprint mismatch: {fingerprint} != {EXPECTED_FINGERPRINT}"
            )
        verified.append(wl)
    print(
        "OFFLINE VERIFY OK: WL%s share completed warm-up %s"
        % (verified, EXPECTED_FINGERPRINT[:16])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
