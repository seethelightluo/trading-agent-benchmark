import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scheduler.monitor_fm_warmup import build_snapshot, format_snapshot, recent_problem_lines


class MonitorFmWarmupTests(unittest.TestCase):
    def test_build_snapshot_reports_progress_eta_and_recent_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage"
            checkpoint = stage / "checkpoint"
            checkpoint.mkdir(parents=True)
            state = root / "state.json"
            log = root / "run.log"
            pid_file = root / "run.pid"
            state.write_text(
                json.dumps(
                    {
                        "shared_warmup": {
                            "fm_progress": {
                                "warmup_stage": "mining",
                                "stage_dir": str(stage),
                                "checkpoint_path": str(checkpoint),
                                "target_iterations": 200,
                                "library_capacity": 30,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "loop_state.json").write_text(
                '{"iteration": 2, "library_size": 4}', encoding="utf-8"
            )
            (stage / "mining_batches.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"iteration": 1, "candidates": 40, "admitted": 1, "elapsed_seconds": 120}),
                        json.dumps({"iteration": 2, "candidates": 40, "admitted": 3, "elapsed_seconds": 180}),
                    ]
                ),
                encoding="utf-8",
            )
            log.write_text(
                "=== WL1 FM warmup attempt 2 ===\n"
                "normal line\n"
                "❌ WL1 FM warmup rc=1；60s 后恢复重试\n",
                encoding="utf-8",
            )
            pid_file.write_text(str(os.getpid()), encoding="utf-8")

            snapshot = build_snapshot(state, log, pid_file)

            self.assertTrue(snapshot["running"])
            self.assertEqual(snapshot["iteration"], 2)
            self.assertEqual(snapshot["library_size"], 4)
            self.assertEqual(snapshot["candidates"], 80)
            self.assertEqual(snapshot["admitted"], 4)
            self.assertEqual(snapshot["elapsed_seconds"], 300)
            self.assertEqual(snapshot["eta_seconds"], 29_700)
            self.assertEqual(len(snapshot["problems"]), 1)
            rendered = format_snapshot(snapshot)
            self.assertIn("iter 2/200", rendered)
            self.assertIn("library 4", rendered)
            self.assertIn("ETA 8h15m", rendered)

    def test_recent_problem_lines_filters_normal_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text(
                "ok\nWARNING without warning emoji is informational here\n"
                "Traceback (most recent call last):\nMining error: boom\n",
                encoding="utf-8",
            )
            self.assertEqual(
                recent_problem_lines(log),
                ["Traceback (most recent call last):", "Mining error: boom"],
            )

    def test_missing_pid_is_reported_as_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("scheduler.monitor_fm_warmup.datetime") as mocked_datetime:
                mocked_datetime.now.return_value.strftime.return_value = "2026-07-26 12:00:00"
                snapshot = build_snapshot(
                    root / "missing-state.json",
                    root / "missing.log",
                    root / "missing.pid",
                )
            self.assertFalse(snapshot["running"])
            self.assertEqual(snapshot["phase"], "not-started")


if __name__ == "__main__":
    unittest.main()
