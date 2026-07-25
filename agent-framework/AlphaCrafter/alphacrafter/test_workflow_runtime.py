import json
import os
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
from agent.toolkit.backtest import BacktestTool
from agent.instructions.quantitative_trading_a import QUANTITATIVE_TRADING_INSTRUCTION_A
from agent.instructions.miner import MINER_INSTRUCTION
from main import CycleRecord, Launcher
from alphacrafter.sim.utils.finish_check import finish_check as simulation_finish_check


class WorkflowRuntimeTests(unittest.TestCase):
    def test_finish_check_does_not_skip_legacy_final_day_cursor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            persistent = root / "persistent"
            workspace = root / "workspace"
            persistent.mkdir()
            workspace.mkdir()
            (persistent / "date.json").write_text(
                json.dumps({
                    "current_date": "2026-07-20",
                    "trading_days": ["2026-07-16", "2026-07-17", "2026-07-20"],
                }),
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                self.assertFalse(simulation_finish_check())
            finally:
                os.chdir(previous_cwd)

    def test_finish_check_requires_persisted_completion_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            persistent = root / "persistent"
            workspace = root / "workspace"
            persistent.mkdir()
            workspace.mkdir()
            (persistent / "date.json").write_text(
                json.dumps({
                    "current_date": "2026-07-20",
                    "visible_through": "2026-07-20",
                    "simulation_complete": True,
                    "trading_days": ["2026-07-16", "2026-07-17", "2026-07-20"],
                }),
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                self.assertTrue(simulation_finish_check())
            finally:
                os.chdir(previous_cwd)

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

    def test_default_miner_run_uses_per_miner_context(self):
        launcher = Launcher.__new__(Launcher)
        launcher.miner_ids = ["miner_1", "miner_2"]

        def context_for(miner_id):
            return f"context for {miner_id}"

        with (
            patch.object(launcher, "_build_miner_context", side_effect=context_for),
            patch.object(launcher, "_run_single_miner") as run_miner,
            patch.object(launcher, "_log_workflow_entry"),
        ):
            run_miner.side_effect = lambda miner_id, context, resume: {
                "miner_id": miner_id,
                "output_text": context,
                "success": True,
            }
            outputs = launcher._run_all_miners_concurrently(cycle=2, context=None)

        self.assertEqual(outputs["miner_1"]["output_text"], "context for miner_1")
        self.assertEqual(outputs["miner_2"]["output_text"], "context for miner_2")

    def test_agent_instructions_describe_small_cross_asset_universe(self):
        self.assertIn("exactly 15 tradable", QUANTITATIVE_TRADING_INSTRUCTION_A)
        self.assertIn("observation-only", QUANTITATIVE_TRADING_INSTRUCTION_A)
        self.assertIn("Never impose a 50/80/300-instrument minimum", MINER_INSTRUCTION)
        self.assertIn("benchmark-wide admission gates", MINER_INSTRUCTION)
        self.assertNotIn("CSI 300 index constituent stocks", QUANTITATIVE_TRADING_INSTRUCTION_A)

    def test_miner_prompt_uses_shared_factor_thresholds(self):
        config = AC_REPO / "config.yaml"
        with patch.dict(
            os.environ,
            {
                "AC_FACTOR_IC_THRESHOLD": "0.04",
                "AC_FACTOR_ICIR_THRESHOLD": "0.10",
            },
        ):
            launcher = Launcher("template_a", str(config))

        with patch("main.Agent") as agent_cls:
            launcher._create_miner_agent("miner_1")

        instructions = agent_cls.call_args.kwargs["instructions"]
        self.assertIn("paper IC >= 0.0400", instructions)
        self.assertIn("paper ICIR >= 0.1000", instructions)

    def test_partial_first_cycle_resumes_from_cycle_zero_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "workflow.json"
            log_path.write_text(
                json.dumps(
                    [
                        {
                            "cycle": 1,
                            "phase": "miner_miner_1",
                            "success": True,
                            "output_text": "factor ready",
                        },
                        {
                            "cycle": 1,
                            "phase": "screener",
                            "success": True,
                            "output_text": "ensemble ready",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            launcher = Launcher.__new__(Launcher)
            launcher.log_path = str(log_path)
            launcher.cycle_records = []

            checkpoint = launcher._load_previous_workflow_state()

        self.assertEqual(checkpoint, 0)
        self.assertEqual(launcher.cycle_records, [])

    def test_only_interrupted_cycle_reuses_resume_inputs(self):
        launcher = Launcher.__new__(Launcher)
        launcher.resume = True
        launcher.max_cycles = 2
        launcher.miner_ids = ["miner_1"]
        launcher.miner_agents = {}
        launcher.cycle_records = []
        launcher.log_path = "workflow.json"
        launcher.original_sigint_handler = None
        launcher.stop_event = MagicMock()
        launcher.stop_event.is_set.return_value = False

        with (
            patch.object(launcher, "_setup_signal_handler"),
            patch.object(launcher, "_get_session_workspace", return_value="workspace"),
            patch.object(launcher, "_setup_workspace"),
            patch.object(launcher, "_load_resume_inputs"),
            patch.object(launcher, "_load_previous_workflow_state", return_value=0),
            patch.object(launcher, "_create_miner_agent", return_value=MagicMock()),
            patch.object(launcher, "_create_screener_agent", return_value=MagicMock()),
            patch.object(launcher, "_create_trader_agent", return_value=MagicMock()),
            patch.object(launcher, "_run_single_cycle", return_value=True) as run_cycle,
        ):
            result = launcher.run()

        self.assertTrue(result["success"])
        self.assertEqual(
            run_cycle.call_args_list,
            [
                unittest.mock.call(1, is_resume_cycle=True),
                unittest.mock.call(2, is_resume_cycle=False),
            ],
        )

    def test_resume_at_max_cycles_does_not_start_an_extra_cycle(self):
        launcher = Launcher.__new__(Launcher)
        launcher.resume = True
        launcher.max_cycles = 1
        launcher.miner_ids = ["miner_1"]
        launcher.miner_agents = {}
        launcher.cycle_records = [CycleRecord(cycle=1)]
        launcher.log_path = "workflow.json"
        launcher.original_sigint_handler = None
        launcher.stop_event = MagicMock()

        with (
            patch.object(launcher, "_setup_signal_handler"),
            patch.object(launcher, "_get_session_workspace", return_value="workspace"),
            patch.object(launcher, "_setup_workspace"),
            patch.object(launcher, "_load_resume_inputs"),
            patch.object(launcher, "_load_previous_workflow_state", return_value=1),
            patch.object(launcher, "_create_miner_agent") as create_miner,
            patch.object(launcher, "_run_single_cycle") as run_cycle,
        ):
            result = launcher.run()

        self.assertTrue(result["success"])
        self.assertEqual(result["total_cycles"], 1)
        create_miner.assert_not_called()
        run_cycle.assert_not_called()

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
                patch.dict(
                    "os.environ", {"AC_REBALANCE_ONLY_ON_CYCLE_START": "0"}
                ),
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

            state = json.loads(date_file.read_text(encoding="utf-8"))
            self.assertEqual(state["visible_through"], trading_days[9])
            self.assertEqual(state["current_date"], trading_days[10])

    def test_step_can_rebalance_only_once_while_advancing_daily_ticks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "stock_data"
            dataset.mkdir()
            date_file = root / "date.json"
            account_file = root / "account.json"
            strategy_file = root / "strategy.py"
            log_file = root / "logs" / "snapshot.json"
            trading_days = [
                (date(2026, 7, 16) + timedelta(days=offset)).isoformat()
                for offset in range(12)
            ]
            date_file.write_text(
                json.dumps(
                    {
                        "current_date": trading_days[0],
                        "visible_through": "2026-07-15",
                        "trading_days": trading_days,
                    }
                ),
                encoding="utf-8",
            )
            account_file.write_text(
                json.dumps(
                    {
                        "net_assets": 100_000_000.0,
                        "total_assets": 100_000_000.0,
                        "available_cash": 100_000_000.0,
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
                patch("agent.toolkit.step.Hook", side_effect=[initial_hook, reloaded_hook]),
                patch("agent.toolkit.step.sleep"),
                patch.dict(
                    "os.environ",
                    {
                        "AC_CADENCE_DAYS": "10",
                        "AC_REBALANCE_ONLY_ON_CYCLE_START": "1",
                    },
                ),
            ):
                exchange_cls.return_value.post_tick.return_value = []
                tool = StepTool(
                    date_file_path=str(date_file),
                    dataset_dir_path=str(dataset),
                    account_file_path=str(account_file),
                    strategy_file_path=str(strategy_file),
                    log_file_path=str(log_file),
                )
                tool.get_implementation()(days=10)

            self.assertEqual(reloaded_hook.on_tick.call_count, 1)
            self.assertEqual(exchange_cls.return_value.pre_tick.call_count, 10)
            self.assertEqual(exchange_cls.return_value.post_tick.call_count, 10)

    def test_step_processes_final_trading_day_once_and_marks_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "stock_data"
            dataset.mkdir()
            date_file = root / "date.json"
            account_file = root / "account.json"
            strategy_file = root / "strategy.py"
            log_file = root / "logs" / "snapshot.json"
            trading_days = ["2026-07-16", "2026-07-17", "2026-07-20"]
            date_file.write_text(
                json.dumps({
                    "current_date": trading_days[0],
                    "visible_through": "2026-07-15",
                    "simulation_complete": False,
                    "trading_days": trading_days,
                }),
                encoding="utf-8",
            )
            account_file.write_text(
                json.dumps({
                    "net_assets": 100_000_000.0,
                    "total_assets": 100_000_000.0,
                    "available_cash": 100_000_000.0,
                    "market_value": 0.0,
                    "gross_position_rate": 0.0,
                    "net_position_rate": 0.0,
                }),
                encoding="utf-8",
            )
            strategy_file.write_text("# strategy placeholder\n", encoding="utf-8")

            with (
                patch("alphacrafter.sim.exchange_a.Exchange") as exchange_cls,
                patch("agent.toolkit.step.Hook") as hook_cls,
                patch("agent.toolkit.step.sleep"),
                patch.dict(
                    "os.environ",
                    {
                        "AC_CADENCE_DAYS": "10",
                        "AC_REBALANCE_ONLY_ON_CYCLE_START": "1",
                    },
                ),
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

            self.assertEqual(exchange_cls.return_value.pre_tick.call_count, 3)
            self.assertEqual(exchange_cls.return_value.post_tick.call_count, 3)
            hook_cls.return_value.on_tick.assert_called_once_with()
            state = json.loads(date_file.read_text(encoding="utf-8"))
            self.assertEqual(state["current_date"], trading_days[-1])
            self.assertEqual(state["visible_through"], trading_days[-1])
            self.assertTrue(state["simulation_complete"])
            self.assertIn("Advanced 3 trading days", output)

    def test_shared_warmup_step_never_touches_exchange_or_account(self):
        tool = StepTool.__new__(StepTool)
        tool.strategy_file_path = "strategy.py"
        tool.exchange = MagicMock()
        with patch.dict(
            "os.environ",
            {
                "AC_CADENCE_DAYS": "10",
                "AC_REBALANCE_ONLY_ON_CYCLE_START": "1",
                "AC_WARMUP_ONLY": "1",
            },
        ):
            output = tool.get_implementation()(days=10)

        self.assertIn("capital remains frozen", output)
        tool.exchange.pre_tick.assert_not_called()
        tool.exchange.post_tick.assert_not_called()

    def test_backtest_reloads_strategy_hook_before_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "stock_data"
            dataset.mkdir()
            date_file = root / "date.json"
            account_file = root / "account.json"
            strategy_file = root / "strategy.py"
            log_file = root / "logs" / "backtest.json"

            trading_days = ["2026-07-15", "2026-07-16"]
            date_file.write_text(
                json.dumps(
                    {"current_date": trading_days[-1], "trading_days": trading_days}
                ),
                encoding="utf-8",
            )
            account_file.write_text(
                json.dumps(
                    {
                        "total_assets": 10_000_000.0,
                        "net_assets": 10_000_000.0,
                        "available_cash": 10_000_000.0,
                        "market_value": 0.0,
                        "gross_position_rate": 0.0,
                        "net_position_rate": 0.0,
                        "positions": [],
                        "orders": [],
                        "watch_list": [],
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
                    "agent.toolkit.backtest.Hook",
                    side_effect=[initial_hook, reloaded_hook],
                ) as hook_cls,
                patch("agent.toolkit.backtest.sleep"),
            ):
                tool = BacktestTool(
                    date_file_path=str(date_file),
                    dataset_dir_path=str(dataset),
                    account_file_path=str(account_file),
                    strategy_file_path=str(strategy_file),
                    log_file_path=str(log_file),
                )
                output = tool.get_implementation()(days=1)

            self.assertEqual(hook_cls.call_count, 2)
            self.assertIs(tool.hook, reloaded_hook)
            reloaded_hook.on_tick.assert_called_once_with()
            self.assertIn("Backtest completed", output)

    def test_backtest_ends_at_visible_through_not_next_execution_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "stock_data"
            dataset.mkdir()
            date_file = root / "date.json"
            account_file = root / "account.json"
            strategy_file = root / "strategy.py"
            log_file = root / "logs" / "backtest.json"
            trading_days = ["2026-07-14", "2026-07-15", "2026-07-16"]
            date_file.write_text(
                json.dumps({
                    "current_date": "2026-07-16",
                    "visible_through": "2026-07-15",
                    "trading_days": trading_days,
                }),
                encoding="utf-8",
            )
            account_file.write_text(
                json.dumps({
                    "initial_capital": 100_000_000.0,
                    "total_assets": 100_000_000.0,
                    "net_assets": 100_000_000.0,
                    "available_cash": 100_000_000.0,
                    "market_value": 0.0,
                    "gross_position_rate": 0.0,
                    "net_position_rate": 0.0,
                    "positions": [],
                    "orders": [],
                    "watch_list": [],
                }),
                encoding="utf-8",
            )
            strategy_file.write_text("# placeholder\n", encoding="utf-8")

            with (
                patch("alphacrafter.sim.exchange_a.Exchange") as exchange_cls,
                patch("agent.toolkit.backtest.Hook") as hook_cls,
                patch("agent.toolkit.backtest.sleep"),
            ):
                tool = BacktestTool(
                    date_file_path=str(date_file),
                    dataset_dir_path=str(dataset),
                    account_file_path=str(account_file),
                    strategy_file_path=str(strategy_file),
                    log_file_path=str(log_file),
                )
                output = tool.get_implementation()(days=2)

            self.assertEqual(exchange_cls.return_value.pre_tick.call_count, 2)
            self.assertEqual(hook_cls.return_value.on_tick.call_count, 1)
            self.assertIn("2026-07-14 → 2026-07-15", output)
            self.assertNotIn("→ 2026-07-16", output)
            restored = json.loads(date_file.read_text(encoding="utf-8"))
            self.assertEqual(restored["current_date"], "2026-07-16")
            self.assertEqual(restored["visible_through"], "2026-07-15")
