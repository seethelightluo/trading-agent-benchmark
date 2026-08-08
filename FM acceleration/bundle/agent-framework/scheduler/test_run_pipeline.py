import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))
sys.path.insert(0, str(AGENT_FRAMEWORK / "FactorMiner"))

from scheduler.run_pipeline import (
    AC_REPO,
    EscalatingBackoff,
    VENV_PY,
    _run_fm_command,
    _refresh_library_signals,
    _trim_factor_library,
    _write_fm_window_config,
    ac_command,
    ac_env,
    ac_session_complete,
    ac_worldline_resume_ready,
    ac_warmup_fingerprint,
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
    def test_backoff_reaches_one_hour_and_stays_there_until_success(self):
        backoff = EscalatingBackoff()
        self.assertEqual(
            [backoff.on_fail() for _ in range(7)],
            [0, 60, 600, 3600, 3600, 3600, 3600],
        )
        backoff.on_success()
        self.assertEqual(backoff.on_fail(), 0)

    def test_fm_unlimited_retry_uses_full_backoff_then_repeats_hourly(self):
        backoff = EscalatingBackoff()
        with patch(
            "scheduler.run_pipeline.subprocess.call",
            side_effect=[1, 1, 1, 1, 1, 0],
        ), patch("scheduler.run_pipeline.time.sleep") as sleep:
            ok = _run_fm_command(
                1,
                "warmup/mine",
                ["factorminer", "mine"],
                {},
                backoff,
                max_attempts=0,
            )

        self.assertTrue(ok)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 600, 3600, 3600])
        self.assertEqual(backoff.idx, 0)

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
        self.assertEqual(admission["ic_threshold"], 0.007)
        self.assertEqual(admission["icir_threshold"], 0.084)
        self.assertGreaterEqual(admission["icir_threshold"], scaled)

    def test_ac_warmup_cycle_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "warm-up cycles must be positive"):
            ac_warmup_fingerprint(Path("/unused/panel.parquet"), warmup_cycles=0)

    def test_ac_warmup_cycle_count_is_part_of_fingerprint_contract(self):
        source = Path(ac_warmup_fingerprint.__code__.co_filename).read_text(
            encoding="utf-8"
        )
        self.assertIn('"warmup_cycles": int(warmup_cycles)', source)
        self.assertIn(
            'write_run_config(warmup_cycles, result_dir / "run_config.yaml")',
            source,
        )

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
        self.assertEqual(env["AC_FACTOR_IC_THRESHOLD"], "0.007")
        self.assertEqual(env["AC_FACTOR_ICIR_THRESHOLD"], "0.084")
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

    def test_seeded_online_worldline_is_resume_ready_without_ws1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = root / "sandbox" / "wl3" / "persistent"
            persistent.mkdir(parents=True)
            (persistent / "shared_warmup_seed.json").write_text(json.dumps({
                "warmup_fingerprint": "formal-warmup",
                "first_forward_block_complete": True,
            }), encoding="utf-8")
            (persistent / "date.json").write_text(json.dumps({
                "current_date": "2031-05-01",
                "visible_through": "2031-04-30",
            }), encoding="utf-8")
            (persistent / "account.json").write_text(json.dumps({
                "portfolio_initialized": True,
            }), encoding="utf-8")

            with patch("scheduler.run_pipeline.AC_REPO", root):
                self.assertTrue(ac_worldline_resume_ready(3))

            # The mutable shared warm-up source is deliberately absent; the
            # already-seeded online session remains independently resumable.
            self.assertFalse((root / "sandbox" / "ws1").exists())

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
                1_000_000.0,
                admission=factor_admission_contract(),
            )

            import yaml
            config = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertEqual(config["mining"]["ic_threshold"], 0.007)
            self.assertEqual(config["mining"]["icir_threshold"], 0.084)
            self.assertEqual(config["mining"]["correlation_threshold"], 0.5)

    def test_fm_window_config_sets_runtime_evaluation_workers_in_correct_schema(self):
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
            base.write_text(
                "evaluation:\n  num_workers: 40\nmining:\n  target_library_size: 110\n",
                encoding="utf-8",
            )

            _write_fm_window_config(
                base, panel, dates[-1].strftime("%Y-%m-%d"), out,
                10, 1_000_000.0, evaluation_workers=4,
            )

            import yaml
            config = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertEqual(config["evaluation"]["num_workers"], 4)
            self.assertNotIn("num_workers", config["mining"])

    def test_fm_window_config_rejects_nonpositive_evaluation_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel.parquet"
            dates = pd.bdate_range("2020-01-01", periods=30)
            pd.DataFrame({
                "datetime": dates,
                "asset_id": ["A"] * len(dates),
                "close": range(len(dates)),
            }).to_parquet(panel, index=False)
            base = root / "base.yaml"
            base.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluation_workers must be positive"):
                _write_fm_window_config(
                    base, panel, dates[-1].strftime("%Y-%m-%d"), root / "window.yaml",
                    10, 1_000_000.0, evaluation_workers=0,
                )

    def test_refresh_library_signals_evaluates_once_and_synchronizes_mirror(self):
        from factorminer.core.factor_library import Factor, FactorLibrary
        from factorminer.core.library_io import load_library, save_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "checkpoint" / "library"
            canonical.parent.mkdir(parents=True)
            mirror = root / "factor_library"
            library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.007)
            factor = Factor(
                id=1, name="f1", formula="Neg($close)", category="test",
                ic_mean=0.01, icir=0.1, ic_win_rate=0.5,
                max_correlation=0.0, batch_number=1,
                signals=np.zeros((2, 2)),
            )
            library.factors[1] = factor
            library._id_to_index[1] = 0
            library._next_id = 2
            library.correlation_matrix = np.zeros((1, 1))
            save_library(library, canonical)
            save_library(library, mirror)
            panel = root / "visible.parquet"
            pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=2), "asset_id": ["A", "A"]}).to_parquet(panel, index=False)
            artifacts = [SimpleNamespace(succeeded=True, signals_full=np.full((3, 4), 7.0))]

            with patch("factorminer.utils.config.load_config", return_value=object()), \
                 patch("factorminer.evaluation.runtime.load_runtime_dataset", return_value=object()), \
                 patch("factorminer.evaluation.runtime.evaluate_factors", return_value=artifacts) as evaluate:
                self.assertTrue(_refresh_library_signals(
                    canonical.with_suffix(".json"), panel, root / "window.yaml",
                    mirror_library_json_path=mirror.with_suffix(".json"),
                ))

            evaluate.assert_called_once()
            left = load_library(canonical).list_factors()[0]
            right = load_library(mirror).list_factors()[0]
            np.testing.assert_array_equal(left.signals, np.full((3, 4), 7.0))
            np.testing.assert_array_equal(left.signals, right.signals)

    def test_refresh_library_signals_refuses_divergent_mirror_before_evaluation(self):
        from factorminer.core.factor_library import Factor, FactorLibrary
        from factorminer.core.library_io import save_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "checkpoint" / "library"
            canonical.parent.mkdir(parents=True)
            mirror = root / "factor_library"
            def one_factor(factor_id):
                library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.007)
                library.factors[factor_id] = Factor(
                    id=factor_id, name=f"f{factor_id}", formula="Neg($close)", category="test",
                    ic_mean=0.01, icir=0.1, ic_win_rate=0.5,
                    max_correlation=0.0, batch_number=1, signals=np.zeros((2, 2)),
                )
                library._id_to_index[factor_id] = 0
                library._next_id = factor_id + 1
                library.correlation_matrix = np.zeros((1, 1))
                return library
            save_library(one_factor(1), canonical)
            save_library(one_factor(2), mirror)
            panel = root / "visible.parquet"
            pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=2), "asset_id": ["A", "A"]}).to_parquet(panel, index=False)

            with patch("factorminer.evaluation.runtime.evaluate_factors") as evaluate:
                self.assertFalse(_refresh_library_signals(
                    canonical.with_suffix(".json"), panel, root / "window.yaml",
                    mirror_library_json_path=mirror.with_suffix(".json"),
                ))
            evaluate.assert_not_called()

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

    def test_seeded_online_resume_skips_shared_warmup_rebuild_and_reseed(self):
        argv = [
            "run_pipeline",
            "--only",
            "1",
            "--mode",
            "ac",
            "--max-attempts",
            "1",
        ]
        marker = {
            "warmup_fingerprint": "formal-warmup",
            "first_forward_block_complete": True,
        }
        with (
            patch.object(sys, "argv", argv),
            patch("scheduler.run_pipeline.llm_credentials_configured", return_value=True),
            patch(
                "scheduler.run_pipeline.write_run_config",
                return_value=Path("/tmp/run_config.yaml"),
            ),
            patch("scheduler.run_pipeline.load_state", side_effect=[{}, marker]),
            patch("scheduler.run_pipeline.save_state"),
            patch(
                "scheduler.run_pipeline.ac_worldline_resume_ready",
                return_value=True,
            ),
            patch("scheduler.run_pipeline.ensure_ac_shared_warmup") as warmup,
            patch("scheduler.run_pipeline.prepare_ac_worldline") as seed,
            patch("scheduler.run_pipeline.run_ac_wl", return_value=True),
            patch("scheduler.run_pipeline.ac_session_complete", return_value=False),
        ):
            return_code = main()

        self.assertEqual(return_code, 0)
        warmup.assert_not_called()
        seed.assert_not_called()

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
        self.assertIn('"scheduler_code_sha256": scheduler_code_digest', source)

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

    def test_fm_capacity_trim_keeps_all_at_or_below_cap_and_syncs_checkpoint(self):
        from factorminer.core.factor_library import Factor, FactorLibrary
        from factorminer.core.library_io import load_library, save_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exported = root / "factor_library"
            checkpoint = root / "checkpoint" / "library"
            checkpoint.parent.mkdir(parents=True)

            library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.007)
            for factor_id in range(1, 31):
                factor = Factor(
                    id=factor_id,
                    name=f"f{factor_id}",
                    formula=f"Neg($close_{factor_id})",
                    category="test",
                    ic_mean=factor_id / 1000,
                    icir=0.1,
                    ic_win_rate=0.5,
                    max_correlation=0.0,
                    batch_number=1,
                    signals=np.full((2, 3), factor_id, dtype=float),
                )
                library.factors[factor_id] = factor
                library._id_to_index[factor_id] = factor_id - 1
            library._next_id = 31
            library.correlation_matrix = np.zeros((30, 30))
            save_library(library, exported)
            save_library(FactorLibrary(), checkpoint)
            (checkpoint.parent / "loop_state.json").write_text(
                '{"iteration": 20, "library_size": 999}', encoding="utf-8"
            )

            result = _trim_factor_library(
                exported.with_suffix(".json"),
                30,
                checkpoint_library_json_path=checkpoint.with_suffix(".json"),
            )

            self.assertEqual(result["original_size"], 30)
            self.assertEqual(result["retained_size"], 30)
            self.assertEqual(result["evicted_ids"], [])
            self.assertEqual(
                [f.id for f in load_library(exported).list_factors()],
                [f.id for f in load_library(checkpoint).list_factors()],
            )
            for left, right in zip(
                load_library(exported).list_factors(),
                load_library(checkpoint).list_factors(),
            ):
                np.testing.assert_array_equal(left.signals, right.signals)
            self.assertEqual(
                json.loads((checkpoint.parent / "loop_state.json").read_text())["library_size"],
                30,
            )

    def test_fm_capacity_trim_evicts_only_worst_and_syncs_checkpoint(self):
        from factorminer.core.factor_library import Factor, FactorLibrary
        from factorminer.core.library_io import load_library, save_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exported = root / "factor_library"
            checkpoint = root / "checkpoint" / "library"
            library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.007)
            for factor_id in range(1, 36):
                factor = Factor(
                    id=factor_id,
                    name=f"f{factor_id}",
                    formula=f"Neg($close_{factor_id})",
                    category="test",
                    ic_mean=factor_id / 1000,
                    icir=0.1,
                    ic_win_rate=0.5,
                    max_correlation=0.0,
                    batch_number=1,
                    signals=np.full((2, 3), factor_id, dtype=float),
                )
                library.factors[factor_id] = factor
                library._id_to_index[factor_id] = factor_id - 1
            library._next_id = 36
            library.correlation_matrix = np.zeros((35, 35))
            save_library(library, exported)

            result = _trim_factor_library(
                exported.with_suffix(".json"),
                30,
                checkpoint_library_json_path=checkpoint.with_suffix(".json"),
            )

            self.assertEqual(result["original_size"], 35)
            self.assertEqual(result["retained_size"], 30)
            self.assertEqual(result["evicted_ids"], [1, 2, 3, 4, 5])
            expected = list(range(6, 36))
            self.assertEqual([f.id for f in load_library(exported).list_factors()], expected)
            self.assertEqual([f.id for f in load_library(checkpoint).list_factors()], expected)
            for left, right in zip(
                load_library(exported).list_factors(),
                load_library(checkpoint).list_factors(),
            ):
                np.testing.assert_array_equal(left.signals, right.signals)


if __name__ == "__main__":
    unittest.main()
