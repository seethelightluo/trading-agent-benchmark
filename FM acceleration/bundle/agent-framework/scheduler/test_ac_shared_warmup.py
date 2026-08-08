import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scheduler.ac_shared_warmup import (
    execute_seeded_first_block,
    seed_worldline_workspace,
    validate_warmup_workspace,
    workflow_cycle_complete,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SharedWarmupTests(unittest.TestCase):
    def test_workflow_accepts_configured_miner_ids_without_double_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            phases = ["miner_1", "miner_2", "miner_3", "screener", "trader"]
            write_json(
                session / "logs" / "workflow.json",
                [{"cycle": 1, "phase": phase, "success": True} for phase in phases],
            )

            self.assertTrue(
                workflow_cycle_complete(session, ["miner_1", "miner_2", "miner_3"])
            )

    def test_workflow_accepts_upstream_double_prefixed_phase_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            phases = [
                "miner_miner_1",
                "miner_miner_2",
                "miner_miner_3",
                "screener",
                "trader",
            ]
            write_json(
                session / "logs" / "workflow.json",
                [{"cycle": 1, "phase": phase, "success": True} for phase in phases],
            )

            self.assertTrue(
                workflow_cycle_complete(session, ["miner_1", "miner_2", "miner_3"])
            )

    def test_workflow_can_validate_each_cycle_of_multi_cycle_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            phases = [
                "miner_miner_1",
                "miner_miner_2",
                "miner_miner_3",
                "screener",
                "trader",
            ]
            write_json(
                session / "logs" / "workflow.json",
                [
                    {"cycle": cycle, "phase": phase, "success": True}
                    for cycle in (1, 2)
                    for phase in phases
                ],
            )

            self.assertTrue(
                workflow_cycle_complete(
                    session, ["miner_1", "miner_2", "miner_3"], cycle=2
                )
            )
            self.assertFalse(
                workflow_cycle_complete(
                    session, ["miner_1", "miner_2", "miner_3"], cycle=3
                )
            )
            with self.assertRaises(ValueError):
                workflow_cycle_complete(
                    session, ["miner_1", "miner_2", "miner_3"], cycle=0
                )

    def test_zero_factor_fallback_strategy_is_a_valid_warmup_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            workspace = session / "workspace"
            workspace.mkdir()
            (workspace / "strategy.py").write_text(
                "@register_hook\ndef strategy_hook():\n    pass\n", encoding="utf-8"
            )

            result = validate_warmup_workspace(session)

            self.assertEqual(result["factor_count"], 0)
            self.assertEqual(result["status"], "fallback_allocation_no_admitted_factors")

    def _make_sessions(self, root: Path) -> tuple[Path, Path]:
        warmup = root / "ws1"
        target = root / "wl1"
        (warmup / "workspace" / "factors").mkdir(parents=True)
        (warmup / "workspace" / "strategy.py").write_text(
            "@register_hook\ndef strategy_hook():\n    pass\n", encoding="utf-8"
        )
        (warmup / "workspace" / "factors" / "alpha.json").write_text(
            '{"name": "alpha"}', encoding="utf-8"
        )
        (target / "workspace").mkdir(parents=True)
        (target / "workspace" / "stale.txt").write_text("stale", encoding="utf-8")
        write_json(
            target / "persistent" / "date.json",
            {
                "current_date": "2026-07-16",
                "visible_through": "2026-07-15",
                "simulation_complete": False,
                "trading_days": [
                    "2026-07-16",
                    "2026-07-17",
                    "2026-07-20",
                    "2026-07-21",
                    "2026-07-22",
                ],
            },
        )
        write_json(
            target / "persistent" / "account.json",
            {
                "initial_capital": 1_000_000.0,
                "available_cash": 1_000_000.0,
                "net_assets": 1_000_000.0,
                "positions": [],
                "orders": [],
            },
        )
        return warmup, target

    def test_seeding_replaces_workspace_exactly_and_refuses_used_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            warmup, target = self._make_sessions(Path(tmp))
            seed_worldline_workspace(
                warmup,
                target,
                warmup_fingerprint="abc",
                baseline_date="2026-07-16",
                history_end="2026-07-15",
                initial_capital=1_000_000.0,
            )
            self.assertFalse((target / "workspace" / "stale.txt").exists())
            self.assertTrue((target / "workspace" / "factors" / "alpha.json").exists())

            other = Path(tmp) / "wl2"
            _, other = self._make_sessions(Path(tmp) / "other")
            write_json(other / "logs" / "workflow.json", [{"cycle": 1}])
            with self.assertRaisesRegex(ValueError, "workflow logs"):
                seed_worldline_workspace(
                    warmup,
                    other,
                    warmup_fingerprint="abc",
                    baseline_date="2026-07-16",
                    history_end="2026-07-15",
                    initial_capital=1_000_000.0,
                )

    def test_interrupted_first_block_resumes_only_remaining_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            warmup, target = self._make_sessions(Path(tmp))
            marker_path = seed_worldline_workspace(
                warmup,
                target,
                warmup_fingerprint="abc",
                baseline_date="2026-07-16",
                history_end="2026-07-15",
                initial_capital=1_000_000.0,
            )
            date_path = target / "persistent" / "date.json"
            state = json.loads(date_path.read_text(encoding="utf-8"))
            state.update(current_date="2026-07-20", visible_through="2026-07-17")
            write_json(date_path, state)

            def finish_remaining(command, **kwargs):
                self.assertEqual(kwargs["env"]["AC_CADENCE_DAYS"], "2")
                self.assertIn("days=2", command[-1])
                resumed = json.loads(date_path.read_text(encoding="utf-8"))
                resumed.update(current_date="2026-07-22", visible_through="2026-07-21")
                write_json(date_path, resumed)
                return subprocess.CompletedProcess(command, 0, "ok", "")

            with patch("scheduler.ac_shared_warmup.subprocess.run", side_effect=finish_remaining):
                result = execute_seeded_first_block(
                    target,
                    cadence=4,
                    python=Path("/venv/python"),
                    ac_repo=Path("/repo/ac"),
                    env={},
                )

            self.assertTrue(result["first_forward_block_complete"])
            self.assertEqual(result["completed_through"], "2026-07-21")
            persisted = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["first_forward_block_target_date"], "2026-07-21")

    def test_completed_cursor_repairs_marker_without_replaying_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            warmup, target = self._make_sessions(Path(tmp))
            seed_worldline_workspace(
                warmup,
                target,
                warmup_fingerprint="abc",
                baseline_date="2026-07-16",
                history_end="2026-07-15",
                initial_capital=1_000_000.0,
            )
            date_path = target / "persistent" / "date.json"
            state = json.loads(date_path.read_text(encoding="utf-8"))
            state.update(current_date="2026-07-22", visible_through="2026-07-21")
            write_json(date_path, state)

            with patch("scheduler.ac_shared_warmup.subprocess.run") as run:
                result = execute_seeded_first_block(
                    target,
                    cadence=4,
                    python=Path("/venv/python"),
                    ac_repo=Path("/repo/ac"),
                    env={},
                )

            run.assert_not_called()
            self.assertTrue(result["first_forward_block_complete"])


if __name__ == "__main__":
    unittest.main()
