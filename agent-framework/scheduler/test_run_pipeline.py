import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))

from scheduler.run_pipeline import AC_REPO, VENV_PY, ac_command, ac_env, main


class AcEnvironmentTests(unittest.TestCase):
    def test_adds_package_and_venv_paths_while_preserving_existing_values(self):
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "already-present", "PATH": "/usr/local/bin:/usr/bin"},
        ):
            env = ac_env(cadence=7)

        self.assertEqual(env["AC_CADENCE_DAYS"], "7")
        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep),
            [str(AC_REPO.parent), "already-present"],
        )
        self.assertEqual(
            env["PATH"].split(os.pathsep),
            [str(VENV_PY.parent), "/usr/local/bin", "/usr/bin"],
        )

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
            patch(
                "scheduler.run_pipeline.write_run_config",
                return_value=Path("/tmp/run_config.yaml"),
            ),
            patch("scheduler.run_pipeline.load_state", return_value={}),
            patch("scheduler.run_pipeline.save_state"),
            patch("scheduler.run_pipeline.run_ac_wl", return_value=False),
        ):
            return_code = main()

        self.assertEqual(return_code, 1)


if __name__ == "__main__":
    unittest.main()
