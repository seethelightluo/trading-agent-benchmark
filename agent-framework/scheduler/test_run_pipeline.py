import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))

from scheduler.run_pipeline import (
    AC_REPO,
    EscalatingBackoff,
    VENV_PY,
    _write_fm_window_config,
    ac_command,
    ac_env,
    ac_session_complete,
    factor_admission_contract,
    fm_window_stop_index,
    fm_history_digest,
    fm_checkpoint_iteration,
    main,
    run_ac_wl,
    run_fm_wl,
    seed_fm_online_state,
    write_run_config,
)


class AcEnvironmentTests(unittest.TestCase):
    def test_fm_smoke_window_limit_is_cumulative_across_resumes(self):
        self.assertEqual(fm_window_stop_index(100, 5), 5)
        self.assertEqual(fm_window_stop_index(3, 5), 3)
        self.assertEqual(fm_window_stop_index(100, 0), 100)

    def test_shared_factor_thresholds_match_15_asset_scaling_contract(self):
        admission = factor_admission_contract()
        scaled = admission["reference_icir_threshold"] * (
            (admission["universe_size"] - 1)
            / (admission["reference_universe_size"] - 1)
        ) ** 0.5

        self.assertAlmostEqual(admission["scaled_icir_threshold"], scaled, places=5)
        self.assertEqual(admission["ic_threshold"], 0.04)
        self.assertEqual(admission["icir_threshold"], 0.10)
        self.assertGreaterEqual(admission["icir_threshold"], scaled)

    def test_write_run_config_creates_nested_result_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "run_config.yaml"

            result = write_run_config(1, out)

            self.assertEqual(result, out)
            self.assertTrue(out.exists())
            self.assertIn("max_cycles: 1", out.read_text(encoding="utf-8"))

    def test_adds_package_and_venv_paths_while_preserving_existing_values(self):
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "already-present", "PATH": "/usr/local/bin:/usr/bin"},
        ):
            env = ac_env(cadence=7)

        self.assertEqual(env["AC_CADENCE_DAYS"], "7")
        self.assertEqual(env["AC_FACTOR_IC_THRESHOLD"], "0.04")
        self.assertEqual(env["AC_FACTOR_ICIR_THRESHOLD"], "0.1")
        self.assertNotIn("AC_WARMUP_ONLY", env)
        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep),
            [str(AC_REPO.parent), "already-present"],
        )
        self.assertEqual(
            env["PATH"].split(os.pathsep),
            [str(VENV_PY.parent), "/usr/local/bin", "/usr/bin"],
        )

    def test_warmup_environment_freezes_live_step(self):
        env = ac_env(cadence=10, warmup_only=True)
        self.assertEqual(env["AC_WARMUP_ONLY"], "1")

    def test_fm_window_config_receives_shared_factor_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel.parquet"
            base = root / "base.yaml"
            out = root / "window.yaml"
            dates = pd.bdate_range("2020-01-01", periods=30)
            pd.DataFrame({
                "datetime": dates,
                "asset_id": ["A"] * len(dates),
                "close": range(len(dates)),
            }).to_parquet(panel, index=False)
            base.write_text("llm:\n  provider: mock\n", encoding="utf-8")

            _write_fm_window_config(
                base,
                panel,
                dates[-1].strftime("%Y-%m-%d"),
                out,
                10,
                100_000_000.0,
                admission=factor_admission_contract(),
            )

            import yaml
            config = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertEqual(config["mining"]["ic_threshold"], 0.04)
            self.assertEqual(config["mining"]["icir_threshold"], 0.10)
            self.assertEqual(config["mining"]["correlation_threshold"], 0.5)

    def test_subprocess_can_resolve_both_import_styles(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import agent.openai.agent; import alphacrafter.sim.schemas",
            ],
            cwd=AC_REPO,
            env=ac_env(cadence=10),
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ac_command_uses_positional_session_id(self):
        config = Path("/tmp/run_config.yaml")

        command = ac_command("wl1", config)

        self.assertEqual(
            command,
            [
                str(VENV_PY),
                "main.py",
                "wl1",
                "--config",
                str(config),
                "--resume",
            ],
        )
        self.assertNotIn("--session_id", command)

    def test_successful_process_without_cursor_progress_is_a_failed_smoke(self):
        cursor = ("2026-07-16", "2026-07-15", False)
        with (
            patch("scheduler.run_pipeline.ac_session_cursor", return_value=cursor),
            patch("scheduler.run_pipeline.subprocess.call", return_value=0),
        ):
            ok = run_ac_wl(
                1,
                "wl1",
                Path("/tmp/run_config.yaml"),
                10,
                EscalatingBackoff(),
                max_attempts=1,
            )

        self.assertFalse(ok)

    def test_cursor_progress_accepts_successful_ac_process(self):
        with (
            patch(
                "scheduler.run_pipeline.ac_session_cursor",
                side_effect=[
                    ("2026-07-16", "2026-07-15", False),
                    ("2026-07-30", "2026-07-29", False),
                ],
            ),
            patch("scheduler.run_pipeline.subprocess.call", return_value=0),
        ):
            ok = run_ac_wl(
                1,
                "wl1",
                Path("/tmp/run_config.yaml"),
                10,
                EscalatingBackoff(),
                max_attempts=1,
            )

        self.assertTrue(ok)

    def test_shared_warmup_accepts_frozen_cursor_on_success(self):
        cursor = ("2026-07-16", "2026-07-15", False)
        with (
            patch("scheduler.run_pipeline.ac_session_cursor", return_value=cursor),
            patch("scheduler.run_pipeline.subprocess.call", return_value=0),
        ):
            ok = run_ac_wl(
                0,
                "ws1",
                Path("/tmp/warmup_config.yaml"),
                10,
                EscalatingBackoff(),
                max_attempts=1,
                warmup_only=True,
            )

        self.assertTrue(ok)

    def test_max_cycle_smoke_is_not_marked_as_full_worldline_completion(self):
        argv = [
            "run_pipeline",
            "--only",
            "1",
            "--mode",
            "ac",
            "--max-cycles",
            "2",
            "--max-attempts",
            "1",
        ]
        saved = []
        with (
            patch.object(sys, "argv", argv),
            patch("scheduler.run_pipeline.llm_credentials_configured", return_value=True),
            patch(
                "scheduler.run_pipeline.write_run_config",
                return_value=Path("/tmp/run_config.yaml"),
            ),
            patch("scheduler.run_pipeline.load_state", return_value={}),
            patch("scheduler.run_pipeline.save_state", side_effect=lambda _p, s: saved.append(dict(s))),
            patch(
                "scheduler.run_pipeline.ensure_ac_shared_warmup",
                return_value=(True, {"session": "ws1", "warmup_fingerprint": "abc"}),
            ),
            patch(
                "scheduler.run_pipeline.prepare_ac_worldline",
                return_value={"first_forward_block_complete": True},
            ),
            patch("scheduler.run_pipeline.run_ac_wl", return_value=True),
            patch("scheduler.run_pipeline.ac_session_complete", return_value=False),
        ):
            return_code = main()

        self.assertEqual(return_code, 0)
        self.assertFalse(saved[-1]["wl1"]["ac_done"])
        self.assertTrue(saved[-1]["wl1"]["ac_last_run_ok"])
        self.assertFalse(saved[-1]["wl1"]["done"])

    def test_pipeline_returns_failure_when_a_selected_worldline_fails(self):
        argv = [
            "run_pipeline",
            "--only",
            "1",
            "--mode",
            "ac",
            "--max-attempts",
            "1",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("scheduler.run_pipeline.llm_credentials_configured", return_value=True),
            patch(
                "scheduler.run_pipeline.write_run_config",
                return_value=Path("/tmp/run_config.yaml"),
            ),
            patch("scheduler.run_pipeline.load_state", return_value={}),
            patch("scheduler.run_pipeline.save_state"),
            patch(
                "scheduler.run_pipeline.ensure_ac_shared_warmup",
                return_value=(True, {"session": "ws1", "warmup_fingerprint": "abc"}),
            ),
            patch(
                "scheduler.run_pipeline.prepare_ac_worldline",
                return_value={"first_forward_block_complete": True},
            ),
            patch("scheduler.run_pipeline.run_ac_wl", return_value=False),
        ):
            return_code = main()

        self.assertEqual(return_code, 1)

    def test_fm_mock_and_live_profiles_use_separate_artifact_roots(self):
        # The full behavior is covered by run_fm_wl integration tests; this
        # source-level invariant prevents a cheap mock smoke from poisoning a
        # later live run's frozen library or forward account state.
        source = Path(run_fm_wl.__code__.co_filename).read_text(encoding="utf-8")
        self.assertIn('"live" if live else "mock"', source)
        self.assertIn('warmup_profile = "_".join(warmup_parts)', source)
        self.assertIn('run_profile = f"{warmup_profile}_oi{online_iterations}"', source)

    def test_fm_shared_history_digest_ignores_worldline_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_rows = [
                {
                    "date": day,
                    "asset_id": asset,
                    "open": value,
                    "high": value,
                    "low": value,
                    "close": value,
                    "volume": 1_000.0,
                    "amount": 100_000.0,
                }
                for day, value in (("2026-07-15", 100.0), ("2026-07-16", 101.0))
                for asset in ("A", "B")
            ]
            panel_a = pd.DataFrame(base_rows)
            panel_b = panel_a.copy()
            panel_b.loc[panel_b["date"] == "2026-07-16", "close"] = 999.0
            path_a = root / "wl1.parquet"
            path_b = root / "wl2.parquet"
            panel_a.to_parquet(path_a, index=False)
            panel_b.to_parquet(path_b, index=False)

            digest_a = fm_history_digest(path_a, "2026-07-15", {"A", "B"})
            digest_b = fm_history_digest(path_b, "2026-07-15", {"A", "B"})

            self.assertEqual(digest_a, digest_b)
            panel_b.loc[panel_b["date"] == "2026-07-15", "close"] = 88.0
            panel_b.to_parquet(path_b, index=False)
            self.assertNotEqual(
                digest_a,
                fm_history_digest(path_b, "2026-07-15", {"A", "B"}),
            )

    def test_fm_online_state_is_an_independent_atomic_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared"
            checkpoint = shared / "checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "memory.json").write_text('{"version": 1}', encoding="utf-8")
            (checkpoint / "loop_state.json").write_text(
                '{"iteration": 7}', encoding="utf-8"
            )
            (shared / "factor_library.json").write_text(
                '{"factors": []}', encoding="utf-8"
            )
            (shared / "window.yaml").write_text("data_path: panel.parquet\n", encoding="utf-8")
            online = root / "wl1" / "online_mining"

            marker = seed_fm_online_state(
                shared, online, warmup_fingerprint="fingerprint"
            )

            self.assertEqual(fm_checkpoint_iteration(online), 7)
            self.assertEqual(marker["warmup_fingerprint"], "fingerprint")
            self.assertTrue((online / "seed_manifest.json").exists())
            self.assertFalse(any(online.parent.glob("online_mining.seed.*.tmp")))
            (online / "factor_library.json").write_text(
                '{"factors": ["wl-only"]}', encoding="utf-8"
            )
            self.assertEqual(
                json.loads((shared / "factor_library.json").read_text())["factors"],
                [],
            )

            reused = seed_fm_online_state(
                shared, online, warmup_fingerprint="fingerprint"
            )
            self.assertEqual(reused, marker)


if __name__ == "__main__":
    unittest.main()
