import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))

from scheduler.run_pipeline import AC_REPO, ac_env


class AcEnvironmentTests(unittest.TestCase):
    def test_adds_package_parent_and_preserves_existing_path(self):
        with patch.dict(os.environ, {"PYTHONPATH": "already-present"}):
            env = ac_env(cadence=7)

        self.assertEqual(env["AC_CADENCE_DAYS"], "7")
        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep),
            [str(AC_REPO.parent), "already-present"],
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


if __name__ == "__main__":
    unittest.main()
