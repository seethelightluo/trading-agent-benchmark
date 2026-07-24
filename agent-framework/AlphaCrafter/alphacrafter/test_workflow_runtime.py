import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

AC_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(AC_REPO))
sys.path.insert(0, str(AC_REPO.parent))

from agent.toolkit.step import StepTool
from main import Launcher


class WorkflowRuntimeTests(unittest.TestCase):
    def test_miners_are_logged_under_the_actual_cycle(self):
        launcher = Launcher.__new__(Launcher)
        launcher.miner_ids = ["miner_1"]
        miner_result = {
            "miner_id": "miner_1",
            "output_text": "factor ready",
            "success": True,
        }

        with (
            patch.object(launcher, "_run_single_miner", return_value=miner_result),
            patch.object(launcher, "_log_workflow_entry") as log_entry,
        ):
            outputs = launcher._run_all_miners_concurrently(cycle=4, context="")

        self.assertEqual(outputs, {"miner_1": miner_result})
        log_entry.assert_called_once_with(
            4,
            "miner_miner_1",
            {"success": True, "output_text": "factor ready"},
        )

    def test_step_reloads_strategy_hook_before_advancing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "stock_data"
            dataset.mkdir()
            date_file = root / "date.json"
            account_file = root / "account.json"
            strategy_file = root / "strategy.py"
            log_file = root / "logs" / "snapshot.json"

            start = date(2026, 7, 16)
            trading_days = [
                (start + timedelta(days=offset)).isoformat()
                for offset in range(20)
            ]
            date_file.write_text(
                json.dumps(
                    {"current_date": trading_days[0], "trading_days": trading_days}
                ),
                encoding="utf-8",
            )
            account_file.write_text(
                json.dumps(
                    {
                        "net_assets": 10_000_000.0,
                        "total_assets": 10_000_000.0,
                        "available_cash": 10_000_000.0,
                        "market_value": 0.0,
                        "gross_position_rate": 0.0,
                        "net_position_rate": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            strategy_file.write_text("# strategy placeholder\n", encoding="utf-8")

            initial_hook = MagicMock()
            reloaded_hook = MagicMock()
            with (
                patch("alphacrafter.sim.exchange_a.Exchange") as exchange_cls,
                patch(
                    "agent.toolkit.step.Hook",
                    side_effect=[initial_hook, reloaded_hook],
                ) as hook_cls,
                patch("agent.toolkit.step.sleep"),
            ):
                exchange_cls.return_value.post_tick.return_value = []
                tool = StepTool(
                    date_file_path=str(date_file),
                    dataset_dir_path=str(dataset),
                    account_file_path=str(account_file),
                    strategy_file_path=str(strategy_file),
                    log_file_path=str(log_file),
                )
                output = tool.get_implementation()(days=10)

            self.assertEqual(hook_cls.call_count, 2)
            self.assertIs(tool.hook, reloaded_hook)
            self.assertEqual(reloaded_hook.on_tick.call_count, 10)
            self.assertIn("Advanced 10 trading days", output)
