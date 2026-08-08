#!/usr/bin/env python
"""
run_pipeline.py — 断电可恢复的 live 跑批运行器（9 WL × AC [+FM]）

设计：长任务（数小时~数天）必须抗断电、抗崩溃、可续跑。
- **逐 WL 调 AC**：每条世界线一次 `python main.py wlN --config run_config.yaml --resume`。
  AC 自带 cycle 级 --resume（从 logs 续跑），崩在哪条 cycle 重启即续。
- **失败重试**：AC 非零退出 → 等待退避后重新 --resume，直到 --max-retries。
- **状态持久化**：`results/run_state.json` 原子写入每条 WL 的完成状态 → 断电重启后自动跳过已完成项。
- **nohup/setsid 友好**：本脚本可被 `setsid nohup ... &` 拉起，脱离终端；单 WL 失败不中断后续 WL。

用法（live，需 LLM key 已配）：
    # 前台试跑 1 条 WL（小样本）
    python -m scheduler.run_pipeline --only 1 --mode ac --max-cycles 30
    # 后台全量（nohup，断电可恢复）
    setsid nohup .venv/bin/python -m scheduler.run_pipeline --mode both \\
        > results/run_pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent  # agent-framework/
AC_REPO = HERE / "AlphaCrafter" / "alphacrafter"
FM_REPO = HERE / "FactorMiner"
RESULTS = HERE / "results"
VENV_PY = Path("/home/lxx/trade-agent-benchmark/.venv/bin/python")

ONLINE_DIR = HERE.parent / "data-prepare" / "online-worldline"
ASSETS_CONFIG = HERE / "ASSETS.yaml"


def load_benchmark_config() -> dict:
    """Load the benchmark contract shared by both agent frameworks."""
    return yaml.safe_load(ASSETS_CONFIG.read_text(encoding="utf-8"))


def load_llm_environment() -> None:
    """Load the ignored shared credential file for foreground and daemon runs."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(HERE / "AlphaCrafter" / ".env", override=False)


def llm_credentials_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_URL") and os.environ.get("OPENAI_API_KEY"))


def factor_admission_contract(contract: dict | None = None) -> dict:
    """Return and validate the factor gates shared by AlphaCrafter and FM."""
    benchmark = contract or load_benchmark_config()
    raw = benchmark.get("factor_admission") or {}
    required = (
        "universe_size",
        "reference_universe_size",
        "reference_ic_threshold",
        "reference_icir_threshold",
        "scaled_icir_threshold",
        "ic_threshold",
        "icir_threshold",
        "correlation_threshold",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"ASSETS.yaml factor_admission missing: {missing}")
    values = {
        **raw,
        "universe_size": int(raw["universe_size"]),
        "reference_universe_size": int(raw["reference_universe_size"]),
        "reference_ic_threshold": float(raw["reference_ic_threshold"]),
        "reference_icir_threshold": float(raw["reference_icir_threshold"]),
        "scaled_icir_threshold": float(raw["scaled_icir_threshold"]),
        "ic_threshold": float(raw["ic_threshold"]),
        "icir_threshold": float(raw["icir_threshold"]),
        "correlation_threshold": float(raw["correlation_threshold"]),
        "library_capacity": int(raw.get("library_capacity", 30)),
        "active_top_k": int(raw.get("active_top_k", 10)),
    }
    tradable_count = len(benchmark.get("tradable", []))
    if values["universe_size"] != tradable_count:
        raise ValueError(
            "factor_admission.universe_size does not match tradable universe: "
            f"{values['universe_size']} != {tradable_count}"
        )
    if values["icir_threshold"] < values["scaled_icir_threshold"]:
        raise ValueError(
            "adapted ICIR threshold must not be below the universe-scaled floor"
        )
    return values


def parse_cadence_days(value: str | int) -> int:
    """Accept the new integer form and the legacy ``10B`` CLI spelling."""
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    if text.endswith("B"):
        text = text[:-1]
    days = int(text)
    if days <= 0:
        raise ValueError("cadence must be a positive number of trading days")
    return days


def fm_window_stop_index(total_windows: int, max_online_windows: int) -> int:
    """Return the cumulative smoke ceiling, not an extra resume allowance."""
    if max_online_windows <= 0:
        return total_windows
    return min(total_windows, max_online_windows)


class EscalatingBackoff:
    """配额 5h 刷新模型下的重试退避：立即→1分→10分→1小时，封顶 1 小时。
    失败累加索引；**任一成功立即重置到立即**。共享于全 pipeline（一条 WL 成功后，下条 WL 的失败重新从立即开始）。"""
    SCHEDULE = [0, 60, 600, 3600]  # 秒：立即 / 1分 / 10分 / 1小时

    def __init__(self):
        self.idx = 0

    def on_fail(self) -> float:
        wait = self.SCHEDULE[min(self.idx, len(self.SCHEDULE) - 1)]
        self.idx += 1
        return wait

    def on_success(self):
        self.idx = 0


def ac_env(cadence: int, *, warmup_only: bool = False) -> dict:
    """构造 AC 子进程环境，并兼容仓库导入和虚拟环境命令。

    AC 从 ``alphacrafter/`` 目录启动，因此 ``from agent ...`` 依赖 cwd；而
    ``from alphacrafter ...`` 还要求父目录 ``AlphaCrafter/`` 在 ``PYTHONPATH``。
    Agent 的 shell 工具会执行 ``python``，所以还必须把 uv 环境的 ``bin``
    放在 PATH 首位，避免落到系统 Python 或在精简环境中找不到命令。
    """
    e = os.environ.copy()
    e["AC_CADENCE_DAYS"] = str(cadence)
    e["AC_REBALANCE_ONLY_ON_CYCLE_START"] = "1"
    admission = factor_admission_contract()
    e["AC_FACTOR_IC_THRESHOLD"] = str(admission["ic_threshold"])
    e["AC_FACTOR_ICIR_THRESHOLD"] = str(admission["icir_threshold"])
    benchmark = load_benchmark_config()
    e["AC_PORTFOLIO_MODE"] = "1"
    e["AC_REBALANCE_COST_BPS"] = str(float(benchmark.get("friction_bps", 3.0)))
    e["AC_INITIAL_ALLOCATION_COST_BPS"] = str(
        float(benchmark.get("initial_allocation_cost_bps", 0.0))
    )
    if warmup_only:
        e["AC_WARMUP_ONLY"] = "1"
    else:
        e.pop("AC_WARMUP_ONLY", None)
    inherited_path = e.get("PYTHONPATH")
    e["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(AC_REPO.parent), inherited_path) if part
    )
    inherited_executable_path = e.get("PATH")
    e["PATH"] = os.pathsep.join(
        part for part in (str(VENV_PY.parent), inherited_executable_path) if part
    )
    return e


def write_run_config(max_cycles: int, out: Path) -> Path:
    """生成 AC run 配置：复制 config.yaml，抬高 max_cycles（cadence-10 下每 WL ≈247 cycle）。"""
    src = AC_REPO / "config.yaml"
    text = src.read_text(encoding="utf-8")
    import re
    text = re.sub(r"max_cycles:\s*\d+", f"max_cycles: {max_cycles}", text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # 原子写（防断电半写）


def ac_command(session: str, cfg: Path) -> list[str]:
    """Build the AlphaCrafter CLI command (``session_id`` is positional)."""
    return [str(VENV_PY), "main.py", session, "--config", str(cfg), "--resume"]


def ac_session_complete(session: str) -> bool:
    """Return true only after the final online trading day was processed.

    AlphaCrafter exits successfully when ``max_cycles`` is reached, which is
    useful for a smoke test but must not be confused with completing the whole
    2026-07-16..2035 worldline.
    """
    date_path = AC_REPO / "sandbox" / session / "persistent" / "date.json"
    try:
        date_state = json.loads(date_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(date_state.get("simulation_complete"))


def _sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(HERE)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ac_warmup_fingerprint(panel: Path, warmup_cycles: int = 5) -> str:
    if warmup_cycles <= 0:
        raise ValueError("AC warm-up cycles must be positive")
    contract = load_benchmark_config()
    tradable_ids = {item["asset_id"] for item in contract["tradable"]}
    inputs = [
        ASSETS_CONFIG,
        AC_REPO / "config.yaml",
        AC_REPO / "main.py",
        AC_REPO / "sandbox" / "template_a" / "config" / "models.json",
        AC_REPO / "sandbox" / "template_a" / "workspace" / "strategy.py",
    ]
    inputs.extend((AC_REPO / "agent").rglob("*.py"))
    inputs.extend((AC_REPO / "agent" / "skills").glob("*.md"))
    inputs.extend((AC_REPO / "sim").rglob("*.py"))
    inputs.extend((AC_REPO / "utils").rglob("*.py"))
    payload = {
        "schema_version": 2,
        "warmup_cycles": int(warmup_cycles),
        "history_digest": fm_history_digest(
            panel, contract["history_end"], tradable_ids
        ),
        "research_code_digest": _sha256_paths(inputs),
        "history_end": contract["history_end"],
        "baseline_date": contract["baseline_date"],
        "initial_capital_usd": float(contract["initial_capital_usd"]),
        "max_active_factors": int(contract["max_active_factors"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_ac_session(session: str, panel: Path, stage_news_path: Path | None) -> Path:
    from adapters.build_inputs import (
        build_alpha_crafter,
        load_assets,
        load_panel,
    )

    stage_news = None
    if stage_news_path and stage_news_path.exists():
        stage_news = json.loads(stage_news_path.read_text(encoding="utf-8"))
    return build_alpha_crafter(
        load_panel(panel),
        load_assets(ASSETS_CONFIG),
        session,
        "template_a",
        stage_news=stage_news,
    )


def ensure_ac_worldline_session(wl: int, panel: Path) -> Path:
    session_dir = AC_REPO / "sandbox" / f"wl{wl}"
    if (session_dir / "persistent" / "date.json").exists():
        return session_dir
    return _build_ac_session(
        f"wl{wl}", panel, ONLINE_DIR / f"WL{wl}_stage_news.json"
    )


def ac_worldline_resume_ready(wl: int) -> bool:
    """Return whether a worldline already has a durable shared-warmup seed.

    Once online trading has started, resuming that session must not depend on
    rebuilding the mutable ``ws1`` source workspace.  Runtime-only code fixes
    legitimately change the warm-up fingerprint, but they do not invalidate
    the already-copied research artifacts, account, cursor, or workflow log in
    a seeded worldline.
    """
    session_dir = AC_REPO / "sandbox" / f"wl{wl}"
    marker = load_state(session_dir / "persistent" / "shared_warmup_seed.json")
    date_state = load_state(session_dir / "persistent" / "date.json")
    account = load_state(session_dir / "persistent" / "account.json")
    return bool(
        marker.get("warmup_fingerprint")
        and marker.get("first_forward_block_complete")
        and date_state.get("current_date")
        and date_state.get("visible_through")
        and account.get("portfolio_initialized")
    )


def _ensure_ac_shared_warmup_unlocked(
    panel: Path,
    cadence: int,
    bk: EscalatingBackoff,
    max_attempts: int,
    warmup_cycles: int = 5,
) -> tuple[bool, dict]:
    """Run or reuse the multi-cycle AC warm-up shared by all worldlines."""
    if warmup_cycles <= 0:
        raise ValueError("AC warm-up cycles must be positive")
    from scheduler.ac_shared_warmup import (
        archive_session,
        validate_warmup_workspace,
        workflow_cycle_complete,
        workspace_digest,
    )

    contract = load_benchmark_config()
    fingerprint = ac_warmup_fingerprint(panel, warmup_cycles)
    session = "ws1"
    session_dir = AC_REPO / "sandbox" / session
    result_dir = RESULTS / "ac" / "shared_warmup"
    manifest_path = result_dir / "manifest.json"
    contract_path = session_dir / "persistent" / "warmup_contract.json"
    manifest = load_state(manifest_path)
    reusable = (
        manifest.get("warmup_fingerprint") == fingerprint
        and int(manifest.get("warmup_cycles", 0)) == int(warmup_cycles)
        and session_dir.exists()
        and manifest.get("workspace_digest")
        == workspace_digest(session_dir / "workspace")
    )
    if reusable:
        date_state = load_state(session_dir / "persistent" / "date.json")
        account = load_state(session_dir / "persistent" / "account.json")
        miner_ids = list(yaml.safe_load((AC_REPO / "config.yaml").read_text(encoding="utf-8"))["miner"]["ids"])
        try:
            artifacts = validate_warmup_workspace(session_dir)
        except ValueError:
            reusable = False
        else:
            reusable = (
                all(
                    workflow_cycle_complete(session_dir, miner_ids, cycle=cycle)
                    for cycle in range(1, warmup_cycles + 1)
                )
                and date_state.get("current_date") == contract["baseline_date"]
                and date_state.get("visible_through") == contract["history_end"]
                and not date_state.get("simulation_complete")
                and not account.get("positions")
                and not account.get("orders")
                and float(account.get("initial_capital", 0.0))
                == float(contract["initial_capital_usd"])
                and float(account.get("available_cash", 0.0))
                == float(contract["initial_capital_usd"])
                and artifacts.get("workspace_digest") == manifest.get("workspace_digest")
            )
    if reusable:
        print(f"  ↷ AC 复用 9 条世界线共享 warm-up：{session_dir}", flush=True)
        return True, manifest

    saved_contract = load_state(contract_path)
    if session_dir.exists() and saved_contract.get("warmup_fingerprint") != fingerprint:
        archive_session(
            session_dir,
            RESULTS / "archive",
            f"contract_{saved_contract.get('warmup_fingerprint', 'unknown')[:12]}",
        )
        _build_ac_session(session, panel, None)
    elif not session_dir.exists():
        _build_ac_session(session, panel, None)
    save_state(contract_path, {"warmup_fingerprint": fingerprint})

    warmup_cfg = write_run_config(warmup_cycles, result_dir / "run_config.yaml")
    ok = run_ac_wl(
        0,
        session,
        warmup_cfg,
        cadence,
        bk,
        max_attempts,
        warmup_only=True,
    )
    if not ok:
        return False, {}
    miner_ids = list(yaml.safe_load((AC_REPO / "config.yaml").read_text(encoding="utf-8"))["miner"]["ids"])
    incomplete_cycles = [
        cycle
        for cycle in range(1, warmup_cycles + 1)
        if not workflow_cycle_complete(session_dir, miner_ids, cycle=cycle)
    ]
    if incomplete_cycles:
        print(
            f"  ❌ AC 共享 warm-up 缺少完整 Miner/Screener/Trader cycle: "
            f"{incomplete_cycles}",
            flush=True,
        )
        return False, {}
    try:
        artifacts = validate_warmup_workspace(session_dir)
    except ValueError as exc:
        print(f"  ❌ AC 共享 warm-up 产物无效：{exc}", flush=True)
        return False, {}
    date_state = load_state(session_dir / "persistent" / "date.json")
    account = load_state(session_dir / "persistent" / "account.json")
    if (
        date_state.get("current_date") != contract["baseline_date"]
        or date_state.get("visible_through") != contract["history_end"]
        or date_state.get("simulation_complete")
        or account.get("positions")
        or account.get("orders")
        or float(account.get("initial_capital", 0.0))
        != float(contract["initial_capital_usd"])
        or float(account.get("available_cash", 0.0))
        != float(contract["initial_capital_usd"])
    ):
        print("  ❌ AC warm-up 违反冻结资本/日期边界", flush=True)
        return False, {}
    manifest = {
        "schema_version": 2,
        "warmup_fingerprint": fingerprint,
        "warmup_cycles": int(warmup_cycles),
        "session": session,
        "history_end": contract["history_end"],
        "baseline_date": contract["baseline_date"],
        "initial_capital_usd": float(contract["initial_capital_usd"]),
        **artifacts,
    }
    save_state(manifest_path, manifest)
    return True, manifest


def ensure_ac_shared_warmup(
    panel: Path,
    cadence: int,
    bk: EscalatingBackoff,
    max_attempts: int,
    warmup_cycles: int = 5,
) -> tuple[bool, dict]:
    """Serialize shared warm-up validation/rebuild across concurrent runners."""
    lock_path = RESULTS / "ac" / "shared_warmup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _ensure_ac_shared_warmup_unlocked(
                panel, cadence, bk, max_attempts, warmup_cycles
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def prepare_ac_worldline(
    wl: int,
    panel: Path,
    warmup_manifest: dict,
    cadence: int,
) -> dict:
    from scheduler.ac_shared_warmup import (
        execute_seeded_first_block,
        seed_worldline_workspace,
    )

    contract = load_benchmark_config()
    target_session = ensure_ac_worldline_session(wl, panel)
    warmup_session = AC_REPO / "sandbox" / str(warmup_manifest["session"])
    seed_worldline_workspace(
        warmup_session,
        target_session,
        warmup_fingerprint=str(warmup_manifest["warmup_fingerprint"]),
        baseline_date=contract["baseline_date"],
        history_end=contract["history_end"],
        initial_capital=float(contract["initial_capital_usd"]),
    )
    return execute_seeded_first_block(
        target_session,
        cadence=cadence,
        python=VENV_PY,
        ac_repo=AC_REPO,
        env=ac_env(cadence),
    )


def ac_session_cursor(session: str) -> tuple[str | None, str | None, bool] | None:
    """Return the persisted execution/visibility cursor for progress checks."""
    date_path = AC_REPO / "sandbox" / session / "persistent" / "date.json"
    try:
        date_state = json.loads(date_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return (
        date_state.get("current_date"),
        date_state.get("visible_through"),
        bool(date_state.get("simulation_complete")),
    )


def run_ac_wl(wl: int, session: str, cfg: Path, cadence: int,
              bk: EscalatingBackoff, max_attempts: int = 0,
              warmup_only: bool = False) -> bool:
    """跑一条 WL 的 AC，失败按递增退避(立即→1分→10分→1h)重试（配额5h刷新模型）。
    AC --resume 从最近 cycle 续跑，不丢进度；成功则重置退避。
    max_attempts=0 无限重试（全量默认）；>0 限次（冒烟用，遇真 bug 不死循环）。"""
    cmd = ac_command(session, cfg)
    attempt = 0
    while True:
        attempt += 1
        cursor_before = ac_session_cursor(session)
        print(f"\n=== WL{wl} AC attempt {attempt}（退避档{min(bk.idx, len(bk.SCHEDULE)-1)+1}）===\n  $ {' '.join(cmd)}  (cwd={AC_REPO})", flush=True)
        t0 = time.time()
        rc = subprocess.call(
            cmd,
            cwd=str(AC_REPO),
            env=ac_env(cadence, warmup_only=warmup_only),
        )
        dur = time.time() - t0
        cursor_after = ac_session_cursor(session)
        stalled = (
            rc == 0
            and not warmup_only
            and cursor_before is not None
            and cursor_after == cursor_before
            and not cursor_after[2]
        )
        if rc == 0 and not stalled:
            bk.on_success()
            print(f"  ✅ WL{wl} AC 本次运行成功（{dur/3600:.1f}h），退避已重置", flush=True)
            return True
        if stalled:
            rc = 2
            print(
                f"  ❌ WL{wl} AC 进程虽返回 0，但交易游标未推进：{cursor_after}；"
                "判定为工具调用/step 链路失败",
                flush=True,
            )
        if max_attempts and attempt >= max_attempts:
            print(f"  ⛔ WL{wl} AC 达 max_attempts={max_attempts}，放弃（冒烟模式）", flush=True)
            return False
        wait = bk.on_fail()
        print(f"  ❌ WL{wl} AC rc={rc}（{dur/60:.1f}min）；{wait}s 后 --resume 重试（配额5h刷新，递增退避）", flush=True)
        if wait:
            time.sleep(wait)


def _read_panel(panel: Path):
    import pandas as pd

    df = pd.read_parquet(panel) if panel.suffix == ".parquet" else pd.read_csv(panel)
    df.columns = [c.lower() for c in df.columns]
    date_col = "datetime" if "datetime" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError(f"panel has no date/datetime column: {panel}")
    df[date_col] = pd.to_datetime(df[date_col])
    return df, date_col


def fm_window_cutoffs(panel: Path, baseline_date: str, online_end: str,
                      cadence_days: int) -> tuple[str, list[str]]:
    """Return the research-only cutoff and completed online 10-day cutoffs."""
    df, date_col = _read_panel(panel)
    dates = sorted(df[date_col].drop_duplicates())
    baseline = __import__("pandas").Timestamp(baseline_date)
    end = __import__("pandas").Timestamp(online_end)
    warmup_dates = [d for d in dates if d < baseline]
    if not warmup_dates:
        raise ValueError(f"no warm-up dates before {baseline_date}")
    online_dates = [d for d in dates if baseline <= d <= end]
    cutoffs = [
        online_dates[min(i + cadence_days, len(online_dates)) - 1].strftime("%Y-%m-%d")
        for i in range(0, len(online_dates), cadence_days)
    ]
    return warmup_dates[-1].strftime("%Y-%m-%d"), cutoffs


def _slice_fm_panel(panel: Path, cutoff: str, tradable_ids: set[str], out: Path) -> Path:
    """Create a no-future, tradable-only daily panel for one FM window."""
    import pandas as pd

    df, date_col = _read_panel(panel)
    sliced = df[(df[date_col] <= pd.Timestamp(cutoff)) & df["asset_id"].isin(tradable_ids)].copy()
    if sliced.empty:
        raise ValueError(f"FM slice is empty at cutoff={cutoff}")
    sliced = sliced.rename(columns={date_col: "datetime"})
    sliced = sliced.sort_values(["datetime", "asset_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_parquet(out, index=False)
    return out


def _write_fm_window_config(base_cfg: Path, panel: Path, cutoff: str, out: Path,
                            max_active_factors: int, initial_capital: float,
                            iterations: int | None = None,
                            target: int | None = None,
                            batch_size: int | None = None,
                            admission: dict | None = None,
                            evaluation_workers: int | None = None) -> Path:
    """Generate an internal visible-history split for FM self-validation.

    Both sides of this split are always ``<= cutoff``.  It is an Agent research
    aid inside one forward worldline, not a benchmark train/test partition and
    never a license to expose or tune against unrevealed worldline data.
    """
    df, date_col = _read_panel(panel)
    dates = sorted(df[date_col].drop_duplicates())
    if len(dates) < 20:
        raise ValueError(f"FM needs at least 20 dates, got {len(dates)}")
    test_size = min(252, max(20, len(dates) // 5))
    split_idx = len(dates) - test_size
    train_start = dates[0].strftime("%Y-%m-%d")
    train_end = dates[split_idx - 1].strftime("%Y-%m-%d")
    test_start = dates[split_idx].strftime("%Y-%m-%d")

    raw = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    raw["data_path"] = str(panel)
    raw.setdefault("data", {}).update({
        "market": "cross_asset",
        "universe": "benchmark_tradable_15",
        "frequency": "1d",
        "train_period": [train_start, train_end],
        "test_period": [test_start, cutoff],
    })
    raw.setdefault("benchmark", {})["freeze_top_k"] = max_active_factors
    raw.setdefault("research", {}).setdefault("capacity", {})["base_capital_usd"] = initial_capital
    mining = raw.setdefault("mining", {})
    if admission is not None:
        mining.update({
            "ic_threshold": float(admission["ic_threshold"]),
            "icir_threshold": float(admission["icir_threshold"]),
            "correlation_threshold": float(admission["correlation_threshold"]),
        })
    if iterations is not None:
        mining["max_iterations"] = iterations
    if target is not None:
        mining["target_library_size"] = target
    if batch_size is not None:
        mining["batch_size"] = batch_size
        raw.setdefault("llm", {})["batch_candidates"] = batch_size
    if evaluation_workers is not None:
        if evaluation_workers < 1:
            raise ValueError("FM evaluation_workers must be positive")
        # RalphLoop consumes cfg.evaluation.num_workers, not mining.num_workers.
        # This is a runtime parallelism setting and intentionally stays out of
        # the shared-warmup research fingerprint (which hashes base_cfg).
        raw.setdefault("evaluation", {})["num_workers"] = int(evaluation_workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


def _update_checkpoint_library_metadata(checkpoint_dir: Path, size: int) -> None:
    """Keep checkpoint progress metadata aligned with a post-mine library trim."""
    updates = {
        "loop_state.json": {"library_size": size},
        "session.json": {"last_library_size": size},
        "run_manifest.json": {"library_size": size},
    }
    for filename, values in updates.items():
        path = checkpoint_dir / filename
        if not path.exists():
            continue
        payload = load_state(path)
        payload.update(values)
        save_state(path, payload)


def _trim_factor_library(
    library_json_path: Path,
    cap: int,
    *,
    checkpoint_library_json_path: Path | None = None,
) -> dict:
    """Retain all factors up to ``cap`` and synchronize the checkpoint library.

    When the mined pool exceeds ``cap``, quality = ``|IC_mean| × |ICIR|`` and
    only the best ``cap`` factors are retained.  A pool at or below ``cap`` is
    kept in full.  The same in-memory library is saved to both the exported
    ``factor_library`` and checkpoint ``library`` bases so a later
    ``--resume-checkpoint`` cannot resurrect factors that were already evicted.
    """
    if cap <= 0:
        raise ValueError("factor library capacity must be positive")
    import sys
    fm_root = Path(__file__).resolve().parent.parent / "FactorMiner"
    if fm_root.is_dir() and str(fm_root) not in sys.path:
        sys.path.insert(0, str(fm_root))
    from factorminer.core.library_io import load_library, save_library

    base = Path(str(library_json_path))
    if base.suffix == ".json":
        base = base.with_suffix("")
    if not base.with_suffix(".json").exists():
        raise FileNotFoundError(f"FM factor library does not exist: {base.with_suffix('.json')}")

    library = load_library(base)
    original_size = library.size
    quality = lambda f: (
        abs(float(f.ic_mean)) * abs(float(getattr(f, "icir", 0.0) or 0.0)),
        int(f.id),
    )
    worst = sorted(library.list_factors(), key=quality)[: max(0, library.size - cap)]
    for factor in worst:
        library.remove_factor(factor.id)

    # Save even when size <= cap: this deliberately makes the exported library
    # the canonical source for the checkpoint copy and repairs old divergence.
    save_library(library, base)

    checkpoint_base = None
    if checkpoint_library_json_path is not None:
        checkpoint_base = Path(str(checkpoint_library_json_path))
        if checkpoint_base.suffix == ".json":
            checkpoint_base = checkpoint_base.with_suffix("")
        save_library(library, checkpoint_base)
        _update_checkpoint_library_metadata(checkpoint_base.parent, library.size)

        exported_ids = [factor.id for factor in load_library(base).list_factors()]
        checkpoint_ids = [factor.id for factor in load_library(checkpoint_base).list_factors()]
        if exported_ids != checkpoint_ids:
            raise RuntimeError(
                "FM library synchronization failed: exported/checkpoint factor IDs differ"
            )

    result = {
        "original_size": original_size,
        "retained_size": library.size,
        "evicted_ids": [factor.id for factor in worst],
        "retained_ids": [factor.id for factor in library.list_factors()],
        "checkpoint_synced": checkpoint_base is not None,
    }
    action = "淘汰" if worst else "全量保留"
    print(
        f"  ✂ factor_library: {original_size} -> {library.size} ({action}; cap={cap}; "
        f"checkpoint={'synced' if checkpoint_base is not None else 'not-requested'})",
        flush=True,
    )
    return result


def _library_contract_signature(library) -> str:
    """Stable signature for two FM library mirrors before one-refresh/two-save.

    The online checkpoint library is mine's canonical resume source while the
    exported library feeds combine/forward.  They must represent exactly the
    same factor contract before one evaluated in-memory library may safely be
    saved to both locations.
    """
    payload = {
        "library": {
            "correlation_threshold": library.correlation_threshold,
            "ic_threshold": library.ic_threshold,
            "dependence_metric": library.dependence_metric.name,
            "metric_version": getattr(library, "metric_version", ""),
            "next_id": library._next_id,
        },
        "factors": [factor.to_dict() for factor in library.list_factors()],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _refresh_library_signals(
    library_json_path,
    window_panel,
    config_path,
    *,
    mirror_library_json_path=None,
) -> bool:
    """Refresh one canonical library once, then safely save it to its mirror.

    Expanding online windows require a signal refresh before resume mining.
    When ``mirror_library_json_path`` is supplied, both locations must have an
    identical factor contract first; after one expensive evaluation the same
    in-memory library (including the new signal cache) is saved to both.  This
    replaces duplicate evaluation while preserving checkpoint/export parity.
    """
    import sys
    import numpy as np
    import pandas as pd
    fm_root = Path(__file__).resolve().parent.parent / "FactorMiner"
    if fm_root.is_dir() and str(fm_root) not in sys.path:
        sys.path.insert(0, str(fm_root))
    try:
        from factorminer.core.library_io import load_library, save_library
        from factorminer.evaluation.runtime import evaluate_factors, load_runtime_dataset
        from factorminer.utils.config import load_config
    except Exception as _exc:
        print(f"  ⚠ refresh_signals: FM import failed: {_exc}", flush=True)
        return False

    base = Path(str(library_json_path))
    if base.suffix == ".json":
        base = base.with_suffix("")
    if not base.with_suffix(".json").exists():
        # A fresh/empty checkpoint has nothing stale to refresh.
        return True
    mirror = None
    if mirror_library_json_path is not None:
        mirror = Path(str(mirror_library_json_path))
        if mirror.suffix == ".json":
            mirror = mirror.with_suffix("")
        if not mirror.with_suffix(".json").exists():
            print(
                f"  ⚠ refresh_signals: mirror missing: {mirror.with_suffix('.json')}",
                flush=True,
            )
            return False
    try:
        library = load_library(base)
        if library.size == 0:
            print(f"  ↻ refresh_signals: {base.name} empty, skip", flush=True)
            return True
        if mirror is not None:
            mirror_library = load_library(mirror)
            if _library_contract_signature(library) != _library_contract_signature(mirror_library):
                raise ValueError(
                    "FM checkpoint/exported library contract diverged before "
                    "refresh; refusing unsafe single-refresh synchronization"
                )
        cfg = load_config(config_path=config_path)
        panel_df = (
            pd.read_parquet(window_panel)
            if str(window_panel).lower().endswith((".parquet", ".pq"))
            else pd.read_csv(window_panel)
        )
        dataset = load_runtime_dataset(panel_df, cfg)
        factors = library.list_factors()
        artifacts = evaluate_factors(factors, dataset, signal_failure_policy="reject")
        refreshed = 0
        for factor, artifact in zip(factors, artifacts):
            if getattr(artifact, "succeeded", False) and getattr(artifact, "signals_full", None) is not None:
                factor.signals = np.asarray(artifact.signals_full)
                refreshed += 1
            else:
                factor.signals = None
        # save_library emits both .json and _signals.npz.  Save the same
        # evaluated object to the mine checkpoint first, then its forward
        # mirror; if interrupted before mine begins, next resume refreshes both.
        save_library(library, base)
        if mirror is not None:
            save_library(library, mirror)
            print(
                f"  ↻ refresh_signals: {base.name} refreshed {refreshed}/{library.size}; "
                f"{mirror.name} synchronized (single evaluation)",
                flush=True,
            )
        else:
            print(f"  ↻ refresh_signals: {base.name} refreshed {refreshed}/{library.size}", flush=True)
        return True
    except Exception as _exc:
        # Do not launch --resume-checkpoint with a stale or diverged mirror: it
        # can silently pair factors with incompatible signal shapes.  Preserve
        # the checkpoint for a later explicit retry instead.
        print(f"  ⚠ refresh_signals: {base} failed: {_exc}", flush=True)
        return False


def _fm_cli_script(argv: list[str]) -> str:
    return "from factorminer.cli import main; import sys; sys.argv=%r; main()" % argv


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_json_file(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return True


def fm_history_digest(panel: Path, cutoff: str, tradable_ids: set[str]) -> str:
    """Hash only the shared research history, never WL-specific future rows."""
    import pandas as pd

    df, date_col = _read_panel(panel)
    history = df[
        (df[date_col] <= pd.Timestamp(cutoff))
        & df["asset_id"].astype(str).isin(tradable_ids)
    ].copy()
    if history.empty:
        raise ValueError(f"no tradable FM history through {cutoff}: {panel}")
    history = history.sort_values([date_col, "asset_id"]).reset_index(drop=True)
    columns = sorted(history.columns)
    hashed = pd.util.hash_pandas_object(
        history[columns], index=False, categorize=True
    ).to_numpy()
    digest = hashlib.sha256()
    digest.update("\0".join(columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def load_fm_performance_equivalence(
    source_fingerprint: str,
    target_fingerprint: str,
    *,
    research_code_sha256: str,
    scheduler_code_sha256: str,
) -> dict:
    """Load an explicit, audited performance-only warm-up bridge.

    A code-hash change normally requires a new shared warm-up.  This opt-in
    bridge is allowed only after offline equality evidence has been persisted
    for a specific old/new fingerprint pair; it never silently reuses a stage.
    """
    source_fingerprint = str(source_fingerprint).strip()
    bridge_path = (
        RESULTS / "fm" / "performance_equivalence"
        / f"{source_fingerprint}_to_{target_fingerprint}.json"
    )
    certificate = load_state(bridge_path)
    expected = {
        "schema_version": 1,
        "kind": "fm_performance_equivalence",
        "source_warmup_fingerprint": source_fingerprint,
        "target_warmup_fingerprint": target_fingerprint,
        "target_research_code_sha256": research_code_sha256,
        "target_scheduler_code_sha256": scheduler_code_sha256,
    }
    if not certificate:
        raise ValueError(
            "FM performance-equivalence certificate is required but missing: "
            f"{bridge_path}"
        )
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(
                "FM performance-equivalence certificate does not match the "
                f"current contract at {bridge_path}: {key}"
            )
    checks = certificate.get("checks")
    if not isinstance(checks, dict) or not checks.get("passed", False):
        raise ValueError(
            "FM performance-equivalence certificate has no passed equality checks: "
            f"{bridge_path}"
        )
    certificate["path"] = str(bridge_path)
    return certificate


def promote_fm_performance_equivalent_warmup(
    *,
    shared_warmup_root: Path,
    warmup_cutoff: str,
    source_fingerprint: str,
    target_stage: Path,
    expected_manifest: dict,
    certificate: dict,
) -> None:
    """Create a new immutable warm-up stage from a certified equal parent."""
    manifest_path = target_stage / "shared_warmup_manifest.json"
    if manifest_path.exists():
        if load_state(manifest_path) != expected_manifest:
            raise ValueError(
                "existing promoted FM warm-up stage has a different contract: "
                f"{target_stage}"
            )
        return

    source_stage = (
        shared_warmup_root / f"warmup_{warmup_cutoff}" / source_fingerprint[:16]
    )
    source_manifest = source_stage / "shared_warmup_manifest.json"
    required = (
        source_manifest,
        source_stage / "factor_library.json",
        source_stage / "factor_library_signals.npz",
        source_stage / "checkpoint" / "library.json",
        source_stage / "checkpoint" / "memory.json",
        source_stage / "combination_results.json",
        source_stage / "window.yaml",
        source_stage / "panel_visible.parquet",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(
            "cannot promote incomplete FM warm-up stage: " + "; ".join(missing)
        )
    source = load_state(source_manifest)
    if source.get("warmup_fingerprint") != source_fingerprint:
        raise ValueError(f"source FM warm-up fingerprint mismatch: {source_manifest}")
    stable_keys = set(expected_manifest) - {
        "research_code_sha256", "scheduler_code_sha256", "warmup_fingerprint",
    }
    if any(source.get(key) != expected_manifest.get(key) for key in stable_keys):
        raise ValueError(
            "performance bridge may not change history/config/admission contract: "
            f"{source_stage}"
        )

    target_stage.parent.mkdir(parents=True, exist_ok=True)
    staged = target_stage.parent / f"{target_stage.name}.promote.{os.getpid()}.tmp"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(source_stage, staged)
    save_state(staged / "shared_warmup_manifest.json", expected_manifest)
    save_state(staged / "performance_equivalence_migration.json", certificate)
    staged.rename(target_stage)


def migrate_fm_online_progress_for_performance_equivalence(
    state: dict,
    *,
    online_root: Path,
    source_fingerprint: str,
    target_fingerprint: str,
    target_stage: Path,
    certificate: dict,
) -> None:
    """Retag an already independent online state after a certified bridge."""
    marker_path = online_root / "seed_manifest.json"
    marker = load_state(marker_path)
    if marker.get("warmup_fingerprint") != source_fingerprint:
        raise ValueError(
            "FM online seed does not match the requested performance bridge: "
            f"{marker_path}"
        )
    required = (
        Path(marker.get("library_path", "")),
        Path(marker.get("memory_path", "")),
        Path(marker.get("checkpoint_path", "")),
        Path(state.get("library_path", "")),
        Path(state.get("memory_path", "")),
    )
    if not all(path.exists() for path in required):
        raise ValueError("FM online progress is incomplete; refusing unsafe bridge migration")

    migration = {
        "schema_version": 1,
        "kind": "fm_performance_equivalence",
        "source_warmup_fingerprint": source_fingerprint,
        "target_warmup_fingerprint": target_fingerprint,
        "target_stage": str(target_stage),
        "certificate": certificate["path"],
        "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    marker.update({
        "warmup_fingerprint": target_fingerprint,
        "shared_stage": str(target_stage),
        "performance_equivalence": migration,
    })
    save_state(marker_path, marker)
    save_state(online_root / "performance_equivalence_migration.json", migration)
    state.update({
        "warmup_fingerprint": target_fingerprint,
        "stage_dir": str(target_stage),
        "checkpoint_path": str(target_stage / "checkpoint"),
        "combination_path": str(target_stage / "combination_results.json"),
        "performance_equivalence": migration,
    })


def seed_fm_online_state(
    shared_stage: Path,
    online_root: Path,
    *,
    warmup_fingerprint: str,
) -> dict:
    """Clone immutable shared FM research into one WL-specific evolution state."""
    marker_path = online_root / "seed_manifest.json"
    if marker_path.exists():
        marker = load_state(marker_path)
        if marker.get("warmup_fingerprint") != warmup_fingerprint:
            raise ValueError(f"FM online state was seeded by another warm-up: {online_root}")
        if not all(
            Path(marker.get(key, "")).exists()
            for key in ("library_path", "memory_path", "checkpoint_path")
        ):
            raise ValueError(f"FM online seed manifest points to missing artifacts: {marker_path}")
        return marker
    if online_root.exists() and any(online_root.iterdir()):
        raise ValueError(f"untracked FM online state exists without seed manifest: {online_root}")
    if online_root.exists():
        online_root.rmdir()
    required = (
        shared_stage / "checkpoint" / "memory.json",
        shared_stage / "checkpoint" / "loop_state.json",
        shared_stage / "factor_library.json",
        shared_stage / "window.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"shared FM warm-up is incomplete: {missing}")

    online_root.parent.mkdir(parents=True, exist_ok=True)
    staged = online_root.parent / f"{online_root.name}.seed.{os.getpid()}.tmp"
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir()
    shutil.copytree(
        shared_stage / "checkpoint",
        staged / "checkpoint",
        dirs_exist_ok=False,
    )
    shutil.copy2(shared_stage / "factor_library.json", staged / "factor_library.json")
    shutil.copy2(shared_stage / "window.yaml", staged / "warmup_window.yaml")
    marker = {
        "schema_version": 1,
        "warmup_fingerprint": warmup_fingerprint,
        "shared_stage": str(shared_stage),
        "library_path": str(online_root / "factor_library.json"),
        "memory_path": str(online_root / "checkpoint" / "memory.json"),
        "checkpoint_path": str(online_root / "checkpoint"),
    }
    save_state(staged / "seed_manifest.json", marker)
    staged.rename(online_root)
    return marker


def fm_checkpoint_iteration(online_root: Path) -> int:
    state = load_state(online_root / "checkpoint" / "loop_state.json")
    return int(state.get("iteration", 0))


def _run_fm_command(wl: int, stage: str, argv: list[str], env: dict,
                    bk: EscalatingBackoff, max_attempts: int) -> bool:
    attempt = 0
    while True:
        attempt += 1
        print(
            f"\n=== WL{wl} FM {stage} attempt {attempt} "
            f"（退避档{min(bk.idx, len(bk.SCHEDULE)-1)+1}）===",
            flush=True,
        )
        rc = subprocess.call(
            [str(VENV_PY), "-c", _fm_cli_script(argv)],
            cwd=str(FM_REPO), env=env,
        )
        if rc == 0:
            bk.on_success()
            return True
        if max_attempts and attempt >= max_attempts:
            print(f"  ⛔ WL{wl} FM {stage} 达 max_attempts={max_attempts}", flush=True)
            return False
        wait = bk.on_fail()
        retry_mode = "无限重试" if max_attempts == 0 else f"最多{max_attempts}次"
        print(
            f"  ❌ WL{wl} FM {stage} rc={rc}；{wait}s 后恢复重试"
            f"（{retry_mode}；退避封顶3600s）",
            flush=True,
        )
        if wait:
            time.sleep(wait)


def run_fm_wl(
    wl: int,
    panel: Path,
    bk: EscalatingBackoff,
    cadence_rule: str | int = 10,
    live: bool = True,
    max_attempts: int = 0,
    progress: dict | None = None,
    persist_progress=None,
    max_online_windows: int = 0,
    iterations: int | None = None,
    target: int | None = None,
    batch_size: int | None = None,
    online_iterations: int = 1,
    evaluation_workers: int = 4,
    warmup_only: bool = False,
    performance_equivalent_from: str | None = None,
) -> tuple[bool, bool]:
    """Mine once on shared history, then evolve per-WL every cadence block.

    FactorMiner's native Ralph loop is run exactly on the warm-up slice and its
    library, checkpoint, session, and experience memory are retained and shared.
    Each worldline clones that checkpoint, performs a small Ralph update every ten
    trading days using only then-visible bars, and keeps its account/library/memory
    independent. Daily mark-to-market remains local and makes no LLM call.
    """
    contract = load_benchmark_config()
    admission = factor_admission_contract(contract)
    if target is None:
        target = 110  # mining POOL size (paper default); library_capacity(30) is the retain cap — both warmup and online trim down to it after mining
    cadence_days = parse_cadence_days(cadence_rule)
    if evaluation_workers < 1:
        raise ValueError("FM evaluation_workers must be positive")
    baseline = contract["baseline_date"]
    online_end = contract["online_end"]
    max_active = int(contract.get("max_active_factors", 10))
    initial_capital = float(contract.get("initial_capital_usd", 1_000_000.0))
    tradable_ids = {item["asset_id"] for item in contract["tradable"]}
    warmup_cutoff, online_cutoffs = fm_window_cutoffs(
        panel, baseline, online_end, cadence_days
    )

    base_cfg = FM_REPO / "factorminer" / "configs" / (
        "fm_live.yaml" if live else "fm_mock_real.yaml"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(FM_REPO), env.get("PYTHONPATH")) if part
    )
    env["PATH"] = os.pathsep.join(
        part for part in (str(VENV_PY.parent), env.get("PATH")) if part
    )
    state = progress if progress is not None else {}

    def profile_number(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    warmup_parts = [
        "live" if live else "mock",
        f"cap{int(initial_capital)}",
        "atomicxfer1",
        f"ic{profile_number(admission['ic_threshold'])}",
        f"ir{profile_number(admission['icir_threshold'])}",
        f"rho{profile_number(admission['correlation_threshold'])}",
    ]
    if iterations is not None:
        warmup_parts.append(f"i{iterations}")
    if target is not None:
        warmup_parts.append(f"t{target}")
    if batch_size is not None:
        warmup_parts.append(f"b{batch_size}")
    warmup_profile = "_".join(warmup_parts)
    run_profile = f"{warmup_profile}_oi{online_iterations}"
    fm_root = RESULTS / "fm" / f"WL{wl}" / run_profile
    shared_warmup_root = RESULTS / "fm" / "shared_warmup" / warmup_profile
    pipeline_version = f"adaptive-shared-warmup-v6:{run_profile}"
    shared_pipeline_version = f"shared-warmup-v1:{warmup_profile}"
    forward_dir = fm_root / "forward_adaptive_v4"
    online_root = fm_root / "online_mining"
    history_digest = fm_history_digest(panel, warmup_cutoff, tradable_ids)
    fm_code_digest = _sha256_paths([
        path
        for path in (FM_REPO / "factorminer").rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ])
    scheduler_code_digest = _sha256_file(Path(__file__).resolve())
    warmup_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "history_digest": history_digest,
                "base_config_sha256": _sha256_file(base_cfg),
                "research_code_sha256": fm_code_digest,
                "scheduler_code_sha256": scheduler_code_digest,
                "assets_sha256": _sha256_file(ASSETS_CONFIG),
                "warmup_cutoff": warmup_cutoff,
                "warmup_profile": warmup_profile,
                "cadence_days": cadence_days,
                "max_active": max_active,
                "initial_capital": initial_capital,
                "iterations": iterations,
                "target": target,
                "batch_size": batch_size,
                "factor_admission": admission,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    performance_bridge = None
    if performance_equivalent_from:
        performance_bridge = load_fm_performance_equivalence(
            performance_equivalent_from,
            warmup_fingerprint,
            research_code_sha256=fm_code_digest,
            scheduler_code_sha256=scheduler_code_digest,
        )

    def persist() -> None:
        if persist_progress is not None:
            persist_progress(state)

    def run_warmup() -> tuple[str, str, str, str] | None:
        label = f"warmup_{warmup_cutoff}"
        cutoff = warmup_cutoff
        # Put every immutable research contract in its own directory.  This is
        # critical after a power loss between ``mine`` and ``combine``: a new
        # history/code/config fingerprint must never resume the old partial mine.
        stage_dir = shared_warmup_root / label / warmup_fingerprint[:16]
        manifest_path = stage_dir / "shared_warmup_manifest.json"
        library = stage_dir / "factor_library.json"
        memory = stage_dir / "checkpoint" / "memory.json"
        combination = stage_dir / "combination_results.json"
        cfg = stage_dir / "window.yaml"
        panel_slice = stage_dir / "panel_visible.parquet"
        expected_manifest = {
            "schema_version": 1,
            "pipeline_version": shared_pipeline_version,
            "run_profile": warmup_profile,
            "warmup_cutoff": warmup_cutoff,
            "history_digest": history_digest,
            "tradable_ids": sorted(tradable_ids),
            "max_active_factors": max_active,
            "initial_capital_usd": initial_capital,
            "iterations": iterations,
            "target_library_size": target,
            "batch_size": batch_size,
            "factor_admission": admission,
            "base_config_sha256": _sha256_file(base_cfg),
            "research_code_sha256": fm_code_digest,
            "scheduler_code_sha256": scheduler_code_digest,
            "warmup_fingerprint": warmup_fingerprint,
        }
        state.update({
            "warmup_stage": "mining",
            "stage_dir": str(stage_dir),
            "checkpoint_path": str(stage_dir / "checkpoint"),
            "mining_log_path": str(stage_dir / "mining.log"),
            "pipeline_log_path": state.get("pipeline_log_path"),
            "target_iterations": int(iterations or 200),
            "target_library_size": int(target),
            "batch_size": int(batch_size or 40),
            "library_capacity": int(admission.get("library_capacity", 30)),
        })
        persist()
        if not manifest_path.exists() and performance_bridge is not None:
            promote_fm_performance_equivalent_warmup(
                shared_warmup_root=shared_warmup_root,
                warmup_cutoff=warmup_cutoff,
                source_fingerprint=performance_bridge["source_warmup_fingerprint"],
                target_stage=stage_dir,
                expected_manifest=expected_manifest,
                certificate=performance_bridge,
            )
        if manifest_path.exists():
            saved_manifest = load_state(manifest_path)
            if saved_manifest != expected_manifest:
                raise ValueError(
                    "shared FM warm-up contract changed; use a new run profile "
                    f"instead of overwriting {manifest_path}"
                )
            if (
                all(path.exists() for path in (library, memory, combination, cfg, panel_slice))
                and all(_valid_json_file(path) for path in (library, memory, combination))
            ):
                trim_result = _trim_factor_library(
                    library,
                    int(admission.get("library_capacity", 30)),
                    checkpoint_library_json_path=stage_dir / "checkpoint" / "library.json",
                )
                state["trim_result"] = trim_result
                state["warmup_stage"] = "combined"
                persist()
                print(
                    f"  ↷ WL{wl} FM 复用 9 条世界线共享 warm-up：{stage_dir}",
                    flush=True,
                )
                return str(library), str(memory), str(cfg), str(combination)

        panel_slice = _slice_fm_panel(
            panel, cutoff, tradable_ids, panel_slice
        )
        cfg = _write_fm_window_config(
            base_cfg, panel_slice, cutoff, cfg,
            max_active, initial_capital, iterations, target, batch_size,
            admission, evaluation_workers,
        )
        can_resume_mined = _valid_json_file(library) and _valid_json_file(memory)
        if can_resume_mined:
            print(f"  ↷ FM 共享 warm-up 已完成 mine，继续 combine", flush=True)
        else:
            mine_argv = [
                "factorminer", "-c", str(cfg), "-o", str(stage_dir),
                "mine", "--data", str(panel_slice),
            ]
            if (stage_dir / "checkpoint" / "loop_state.json").exists():
                mine_argv.append("--resume-checkpoint")
            if not _run_fm_command(wl, f"{label}/mine", mine_argv, env, bk, max_attempts):
                return None
            if not _valid_json_file(library):
                print(f"  ❌ WL{wl} FM 未生成 {library}", flush=True)
                return None
            if not _valid_json_file(memory):
                print(f"  ❌ WL{wl} FM 未持久化 experience memory：{memory}", flush=True)
                return None
            state.update({
                "warmup_stage": "mined",
                "library_path": str(library),
                "memory_path": str(memory),
                "config_path": str(cfg),
            })
            persist()
        trim_result = _trim_factor_library(
            library,
            int(admission.get("library_capacity", 30)),
            checkpoint_library_json_path=stage_dir / "checkpoint" / "library.json",
        )
        state["trim_result"] = trim_result
        state["warmup_stage"] = "combining"
        persist()
        combine_argv = [
            "factorminer", "-c", str(cfg), "-o", str(stage_dir),
            "combine", str(library), "--data", str(panel_slice),
            "--fit-period", "train", "--eval-period", "test",
            "--method", "ic-weighted", "--top-k", str(max_active),
        ]
        if not _run_fm_command(wl, f"{label}/combine", combine_argv, env, bk, max_attempts):
            return None
        if not _valid_json_file(combination):
            print(f"  ❌ WL{wl} FM 未持久化组合结果：{combination}", flush=True)
            return None
        state["warmup_stage"] = "combined"
        state["combination_path"] = str(combination)
        persist()
        save_state(manifest_path, expected_manifest)
        return str(library), str(memory), str(cfg), str(combination)

    # An explicit equality certificate may retag an already independent online
    # checkpoint without rerunning the expensive shared warm-up.  This is
    # deliberately opt-in and preserves a migration audit beside the state.
    if (
        performance_bridge is not None
        and state.get("pipeline_version") == pipeline_version
        and state.get("warmup_fingerprint")
        == performance_bridge["source_warmup_fingerprint"]
    ):
        promoted = run_warmup()
        if promoted is None:
            return False, False
        migrate_fm_online_progress_for_performance_equivalence(
            state,
            online_root=online_root,
            source_fingerprint=performance_bridge["source_warmup_fingerprint"],
            target_fingerprint=warmup_fingerprint,
            target_stage=Path(state["stage_dir"]),
            certificate=performance_bridge,
        )
        state["phase"] = "online"
        persist()

    # Never reuse an older implementation's library that may have been mined on
    # an online/future slice.
    library_path = state.get("library_path")
    if (
        state.get("pipeline_version") != pipeline_version
        or state.get("warmup_fingerprint") != warmup_fingerprint
    ):
        library_path = None
        state.clear()
        state.update({
            "pipeline_version": pipeline_version,
            "warmup_fingerprint": warmup_fingerprint,
            "phase": "warmup",
            "warmup_cutoff": warmup_cutoff,
        })
        persist()
    config_path = state.get("config_path")
    memory_path = state.get("memory_path")
    combination_path = state.get("combination_path")
    artifacts_valid = all(
        value and Path(value).exists()
        for value in (library_path, memory_path, config_path, combination_path)
    )
    if not artifacts_valid:
        artifacts = run_warmup()
        if artifacts is None:
            return False, False
        library_path, memory_path, config_path, combination_path = artifacts
        state.update({
            "pipeline_version": pipeline_version,
            "phase": "online",
            "warmup_cutoff": warmup_cutoff,
            "completed_cutoff": warmup_cutoff,
            "next_window_index": 0,
            "library_path": library_path,
            "memory_path": memory_path,
            "config_path": config_path,
            "combination_path": combination_path,
            "max_active_factors": max_active,
            "initial_capital_usd": initial_capital,
        })
        persist()

    if warmup_only:
        state["phase"] = "warmup_done"
        state["warmup_stage"] = "warmup_done"
        state["completed_cutoff"] = warmup_cutoff
        persist()
        return True, False

    shared_stage = (
        shared_warmup_root
        / f"warmup_{warmup_cutoff}"
        / warmup_fingerprint[:16]
    )
    online_seed = seed_fm_online_state(
        shared_stage,
        online_root,
        warmup_fingerprint=warmup_fingerprint,
    )
    _trim_factor_library(
        Path(online_seed["library_path"]),
        int(admission.get("library_capacity", 30)),
        checkpoint_library_json_path=online_root / "checkpoint" / "library.json",
    )
    library_path = online_seed["library_path"]
    memory_path = online_seed["memory_path"]
    if int(state.get("next_window_index", 0)) == 0:
        state.update({
            "library_path": library_path,
            "memory_path": memory_path,
            "online_root": str(online_root),
            "online_iterations_per_window": int(online_iterations),
        })
        persist()

    next_idx = int(state.get("next_window_index", 0))
    stop_idx = fm_window_stop_index(len(online_cutoffs), max_online_windows)
    if str(FM_REPO) not in sys.path:
        sys.path.insert(0, str(FM_REPO))
    from scheduler.fm_walk_forward import run_forward

    for window_idx in range(next_idx, stop_idx):
        decision_cutoff = (
            warmup_cutoff if window_idx == 0 else online_cutoffs[window_idx - 1]
        )
        active_config = Path(config_path)
        if window_idx > 0 and online_iterations > 0:
            window_dir = online_root / "windows" / f"{window_idx:04d}_{decision_cutoff}"
            window_state_path = window_dir / "window_state.json"
            window_state = load_state(window_state_path)
            if window_state:
                if (
                    window_state.get("decision_cutoff") != decision_cutoff
                    or window_state.get("warmup_fingerprint") != warmup_fingerprint
                ):
                    raise ValueError(
                        f"FM online window contract mismatch: {window_state_path}"
                    )
            else:
                start_iteration = fm_checkpoint_iteration(online_root)
                window_state = {
                    "schema_version": 1,
                    "window_index": window_idx,
                    "decision_cutoff": decision_cutoff,
                    "warmup_fingerprint": warmup_fingerprint,
                    "start_iteration": start_iteration,
                    "target_iteration": start_iteration + int(online_iterations),
                    "mining_complete": False,
                    "combination_complete": False,
                }
                save_state(window_state_path, window_state)
            visible_panel = _slice_fm_panel(
                panel,
                decision_cutoff,
                tradable_ids,
                window_dir / "panel_visible.parquet",
            )
            # F4: grow the per-window target so online mining actually runs.
            # Ralph stops at min(library.size>=target, iter>=max_iter); warmup
            # already filled the library to ~target, so a static target makes
            # online mining a no-op. Iteration count is still capped by
            # max_iterations (= start_iter + online_iterations), so this only
            # removes the premature size-based stop.
            online_target = int(target) + (window_idx + 1) * int(batch_size or 40)
            absolute_max_iterations = int(window_state["target_iteration"])
            active_config = _write_fm_window_config(
                base_cfg,
                visible_panel,
                decision_cutoff,
                window_dir / "window.yaml",
                max_active,
                initial_capital,
                absolute_max_iterations,
                online_target,
                batch_size,
                admission,
                evaluation_workers,
            )
            if not window_state.get("mining_complete"):
                if not _refresh_library_signals(
                    online_root / "checkpoint" / "library.json",
                    visible_panel,
                    str(active_config),
                    mirror_library_json_path=online_root / "factor_library.json",
                ):
                    print(
                        f"  ❌ WL{wl} online_{window_idx:04d}: refresh failed; "
                        "checkpoint preserved for retry",
                        flush=True,
                    )
                    return False, False
                mine_argv = [
                    "factorminer", "-c", str(active_config), "-o", str(online_root),
                    "mine", "--data", str(visible_panel), "--resume-checkpoint",
                ]
                if not _run_fm_command(
                    wl, f"online_{window_idx:04d}/mine", mine_argv, env, bk, max_attempts
                ):
                    return False, False
                window_state["mining_complete"] = True
                window_state["completed_iteration"] = fm_checkpoint_iteration(online_root)
                save_state(window_state_path, window_state)
                _trim_factor_library(
                    online_root / "factor_library.json",
                    int(admission.get("library_capacity", 30)),
                    checkpoint_library_json_path=online_root / "checkpoint" / "library.json",
                )
            library_path = str(online_root / "factor_library.json")
            memory_path = str(online_root / "checkpoint" / "memory.json")
            if not window_state.get("combination_complete"):
                combine_argv = [
                    "factorminer", "-c", str(active_config), "-o", str(online_root),
                    "combine", library_path, "--data", str(visible_panel),
                    "--fit-period", "train", "--eval-period", "test",
                    "--method", "ic-weighted", "--top-k", str(max_active),
                ]
                if not _run_fm_command(
                    wl, f"online_{window_idx:04d}/combine", combine_argv, env, bk, max_attempts
                ):
                    return False, False
                window_state["combination_complete"] = True
                save_state(window_state_path, window_state)
            for artifact in (
                Path(library_path),
                Path(memory_path),
                online_root / "combination_results.json",
            ):
                if not artifact.exists():
                    raise FileNotFoundError(f"FM online update missing artifact: {artifact}")
                shutil.copy2(artifact, window_dir / artifact.name)

        forward_state = run_forward(
            panel,
            library_path=Path(library_path),
            config_path=active_config,
            output_dir=forward_dir,
            tradable_ids=sorted(tradable_ids),
            history_end=warmup_cutoff,
            baseline_date=baseline,
            online_end=online_cutoffs[window_idx],
            cadence=cadence_days,
            initial_capital=initial_capital,
            cost_bps=float(contract.get("friction_bps", 3.0)),
            initial_allocation_cost_bps=float(
                contract.get("initial_allocation_cost_bps", 0.0)
            ),
            max_factors=max_active,
        )
        state.update({
            "phase": "online",
            "completed_cutoff": forward_state["last_processed_date"],
            "next_window_index": window_idx + 1,
            "library_path": library_path,
            "memory_path": memory_path,
            "config_path": str(active_config),
            "forward_state_path": str(forward_dir / "forward_state.json"),
            "equity_path": str(forward_dir / "equity.csv"),
        })
        persist()

    complete = int(state.get("next_window_index", 0)) >= len(online_cutoffs)
    if complete:
        state["phase"] = "done"
        state["completed_cutoff"] = online_cutoffs[-1] if online_cutoffs else warmup_cutoff
        persist()
    return True, complete


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="逗号分隔 WL 号（默认 1-9 全跑）")
    ap.add_argument("--mode", default="ac", choices=["ac", "fm", "both"])
    ap.add_argument("--cadence", type=int, default=10, help="AC 再平衡频次（交易日/cycle）")
    ap.add_argument("--fm-cadence", default="10", help="FM 更新频次（交易日；兼容旧写法 10B）")
    ap.add_argument("--fm-max-windows", type=int, default=0, help="FM 在线窗口累计上限（0=跑完整时间线；烟测设5后重复命令仍停在5）")
    ap.add_argument("--fm-iterations", type=int, default=None, help="覆盖 FM 每窗口最大迭代数（烟测可设1）")
    ap.add_argument("--fm-target", type=int, default=None, help="覆盖 FM 因子库目标大小（烟测可设2；全量默认官方110）")
    ap.add_argument("--fm-batch-size", type=int, default=None, help="覆盖 FM candidate batch（烟测可设2）")
    ap.add_argument("--fm-online-iterations", type=int, default=1, help="每个前向10日窗口追加的 Ralph 迭代数（0=只重算冻结因子）")
    ap.add_argument("--fm-evaluation-workers", type=int, default=4, help="FM 候选评估并发 worker 数（写入 evaluation.num_workers；默认4）")
    ap.add_argument(
        "--fm-performance-equivalent-from", default="",
        help="经审计性能等价迁移的旧 shared warm-up fingerprint；仅在证书存在时允许复用在线进度",
    )
    ap.add_argument("--max-cycles", type=int, default=300, help="AC 前向 run config 的 max_cycles")
    ap.add_argument(
        "--ac-warmup-cycles",
        type=int,
        default=5,
        help="AC 共享历史 warm-up 的完整 Miner×3+Screener+Trader cycle 数（默认5）",
    )
    ap.add_argument("--fm-mock", action="store_true", help="FM 用 mock provider（默认 live：fm_live.yaml/gpt-5.6-terra）")
    ap.add_argument("--warmup-only", action="store_true", help="只运行/复用两框架共享历史 warm-up，不进入任何 WL 前向交易")
    ap.add_argument("--max-attempts", type=int, default=0, help="每 WL 最大尝试次数（0=无限重试，全量默认；冒烟设 3 遇真bug不死循环）")
    ap.add_argument("--state", default=str(RESULTS / "run_state.json"))
    args = ap.parse_args()
    load_llm_environment()

    requires_live_llm = args.mode in ("ac", "both") or (
        args.mode == "fm" and not args.fm_mock
    )
    if requires_live_llm and not llm_credentials_configured():
        print(
            "❌ live LLM 凭证未配置：AlphaCrafter/.env 中 "
            "OPENAI_API_URL 与 OPENAI_API_KEY 都必须为非空；未发起 API 请求。",
            flush=True,
        )
        return 1

    wls = [int(x) for x in args.only.split(",") if x.strip()] or list(range(1, 10))
    cfg = write_run_config(args.max_cycles, AC_REPO / "run_config.yaml")
    state_path = Path(args.state)
    state = load_state(state_path)
    bk = EscalatingBackoff()  # 全 pipeline 共享：立即→1分→10分→1h，成功重置
    print(f"运行器启动：WL={wls} mode={args.mode} cadence={args.cadence} "
          f"max_cycles={args.max_cycles} fm={'mock' if args.fm_mock else 'live'}", flush=True)
    admission = factor_admission_contract()
    print(
        "  因子门槛："
        f"|IC|>={admission['ic_threshold']:.4f} "
        f"|ICIR|>={admission['icir_threshold']:.4f} "
        f"|rho|<{admission['correlation_threshold']:.4f} "
        f"(15资产缩放底线={admission['scaled_icir_threshold']:.5f})",
        flush=True,
    )
    print(f"  退避档(秒)：{EscalatingBackoff.SCHEDULE}（配额5h刷新模型，成功重置）", flush=True)
    print(
        "  重试策略："
        + (
            "无限重试（0s→60s→600s→3600s，之后持续3600s）"
            if args.max_attempts == 0
            else f"最多 {args.max_attempts} 次（冒烟/限次模式）"
        ),
        flush=True,
    )
    print(
        f"  AC run_config → {cfg}；shared warm-up cycles={args.ac_warmup_cycles}；"
        f"状态 → {state_path}",
        flush=True,
    )

    all_ok = True
    shared_state = state.setdefault("shared_warmup", {})
    ac_warmup_manifest: dict = {}
    if args.mode in ("ac", "both"):
        seeded_resume = (
            not args.warmup_only
            and all(ac_worldline_resume_ready(wl) for wl in wls)
        )
        if seeded_resume:
            print(
                "  ↷ 所选 AC 世界线已有持久化 shared-warmup seed；"
                "直接从各自 account/date/workflow 断点续跑，不重建 ws1",
                flush=True,
            )
            ok_ac_warmup = True
        else:
            ok_ac_warmup, ac_warmup_manifest = ensure_ac_shared_warmup(
                ONLINE_DIR / "WL1_full.parquet",
                args.cadence,
                bk,
                args.max_attempts,
                args.ac_warmup_cycles,
            )
        shared_state["ac_done"] = ok_ac_warmup
        if ok_ac_warmup:
            shared_state["ac_manifest"] = str(
                RESULTS / "ac" / "shared_warmup" / "manifest.json"
            )
        state["shared_warmup"] = shared_state
        save_state(state_path, state)
        if not ok_ac_warmup:
            return 1

    if args.warmup_only and args.mode in ("fm", "both"):
        fm_progress = shared_state.setdefault("fm_progress", {})
        fm_progress["pipeline_log_path"] = os.environ.get("FM_PIPELINE_LOG_PATH")

        def persist_shared_fm(progress: dict) -> None:
            shared_state["fm_progress"] = progress
            state["shared_warmup"] = shared_state
            save_state(state_path, state)

        ok_fm_warmup, _ = run_fm_wl(
            1,
            ONLINE_DIR / "WL1_full.parquet",
            bk,
            args.fm_cadence,
            not args.fm_mock,
            args.max_attempts,
            fm_progress,
            persist_shared_fm,
            0,
            args.fm_iterations,
            args.fm_target,
            args.fm_batch_size,
            args.fm_online_iterations,
            args.fm_evaluation_workers,
            warmup_only=True,
            performance_equivalent_from=args.fm_performance_equivalent_from,
        )
        shared_state["fm_done"] = ok_fm_warmup
        state["shared_warmup"] = shared_state
        save_state(state_path, state)
        all_ok = all_ok and ok_fm_warmup

    if args.warmup_only:
        print(
            "\n========== 共享 warm-up 结束："
            f"AC={bool(shared_state.get('ac_done'))} "
            f"FM={bool(shared_state.get('fm_done'))} ==========",
            flush=True,
        )
        return 0 if all_ok else 1

    for wl in wls:
        key = f"wl{wl}"
        session = f"wl{wl}"
        panel = ONLINE_DIR / f"WL{wl}_full.parquet"
        st = state.get(key, {})
        if st.get("ac_done") and not ac_session_complete(session):
            print(
                f"\n⚠️  WL{wl} state 曾标记 AC 完成，但 session 尚未处理最终交易日；"
                "已自动恢复为可续跑状态",
                flush=True,
            )
            st["ac_done"] = False
        requested_done = (
            (args.mode not in ("ac", "both") or st.get("ac_done"))
            and (args.mode not in ("fm", "both") or st.get("fm_done"))
        )
        if requested_done:
            print(f"\n⏭️  WL{wl} 请求的组件已完成（state），跳过", flush=True)
            continue

        ok = True
        if args.mode in ("ac", "both") and not st.get("ac_done"):
            try:
                if ac_worldline_resume_ready(wl):
                    seed_state = load_state(
                        AC_REPO
                        / "sandbox"
                        / session
                        / "persistent"
                        / "shared_warmup_seed.json"
                    )
                    print(
                        f"  ↷ WL{wl} 已播种且已有在线账户；跳过重复播种，"
                        "直接 --resume",
                        flush=True,
                    )
                else:
                    seed_state = prepare_ac_worldline(
                        wl, panel, ac_warmup_manifest, args.cadence
                    )
                st["ac_shared_warmup_seed"] = seed_state
            except Exception as exc:
                print(f"  ❌ WL{wl} AC 共享 warm-up 播种/首块执行失败：{exc}", flush=True)
                ok_ac = False
            else:
                ok_ac = run_ac_wl(
                    wl, session, cfg, args.cadence, bk, args.max_attempts
                )
            st["ac_done"] = bool(ok_ac and ac_session_complete(session))
            st["ac_last_run_ok"] = bool(ok_ac)
            if ok_ac and not st["ac_done"]:
                print(
                    f"  ✅ WL{wl} AC 本次配额/周期烟测成功，但尚未走完整条世界线；"
                    "进度已持久化，未误标为完成",
                    flush=True,
                )
            state[key] = st
            save_state(state_path, state)
            ok = ok and ok_ac
        if args.mode in ("fm", "both") and ok and not st.get("fm_done"):
            if not panel.exists():
                print(f"  ❌ WL{wl} panel 不存在：{panel}", flush=True)
                ok_fm, fm_complete = False, False
            else:
                fm_progress = st.setdefault("fm_progress", {})

                def persist_fm_progress(progress: dict) -> None:
                    st["fm_progress"] = progress
                    state[key] = st
                    save_state(state_path, state)

                ok_fm, fm_complete = run_fm_wl(
                    wl, panel, bk, args.fm_cadence, not args.fm_mock,
                    args.max_attempts, fm_progress, persist_fm_progress,
                    args.fm_max_windows, args.fm_iterations,
                    args.fm_target, args.fm_batch_size,
                    args.fm_online_iterations, args.fm_evaluation_workers,
                    performance_equivalent_from=args.fm_performance_equivalent_from,
                )
            st["fm_done"] = fm_complete
            state[key] = st
            save_state(state_path, state)
            ok = ok and ok_fm

        # ``done`` means the full two-framework benchmark is complete.  Running an
        # AC-only smoke must not prevent a later --mode both run from entering FM.
        st["done"] = bool(st.get("ac_done") and st.get("fm_done"))
        st["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state[key] = st
        save_state(state_path, state)
        all_ok = all_ok and ok
        print(
            f"\n{'✅' if ok else '⚠️'} WL{wl} 本次结束"
            f"（ac_done={bool(st.get('ac_done'))}, fm_done={bool(st.get('fm_done'))}）",
            flush=True,
        )

    done = sum(1 for wl in wls if state.get(f"wl{wl}", {}).get("done"))
    print(f"\n========== 全部结束：{done}/{len(wls)} WL 完成 ==========", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
