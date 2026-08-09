import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))

from adapters import build_inputs


class BuildAlphaCrafterTests(unittest.TestCase):
    def test_precreates_concurrent_miner_output_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = (
                root
                / "AlphaCrafter"
                / "alphacrafter"
                / "sandbox"
                / "template_a"
            )
            for relative in (
                "persistent/stock_data",
                "persistent/index_data",
                "persistent/stock_news",
                "workspace",
            ):
                (template / relative).mkdir(parents=True, exist_ok=True)
            (template / "workspace" / "strategy.py").write_text(
                "# placeholder\n", encoding="utf-8"
            )

            panel = pd.DataFrame(
                [
                    {
                        "date": day,
                        "asset_id": asset_id,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 1_000.0,
                    }
                    for day in ("2026-07-15", "2026-07-16")
                    for asset_id in ("SPX", "VIX")
                ]
            )
            assets = {
                "baseline_date": "2026-07-16",
                "history_end": "2026-07-15",
                "online_end": "2026-08-16",
                "initial_capital_usd": 100_000_000,
                "tradable": [{"asset_id": "SPX"}],
                "signals": [{"asset_id": "VIX"}],
            }

            with patch.object(build_inputs, "HERE", root):
                session = build_inputs.build_alpha_crafter(
                    panel, assets, "wl_test", stage_news=[]
                )

            self.assertTrue((session / "workspace" / "factors").is_dir())
            self.assertTrue((session / "workspace" / "scripts").is_dir())
            date_state = json.loads(
                (session / "persistent" / "date.json").read_text(encoding="utf-8")
            )
            self.assertEqual(date_state["current_date"], "2026-07-16")
            self.assertEqual(date_state["visible_through"], "2026-07-15")
            account = json.loads(
                (session / "persistent" / "account.json").read_text(encoding="utf-8")
            )
            self.assertEqual(account["initial_capital"], 100_000_000)
            self.assertEqual(account["available_cash"], 100_000_000)
            self.assertEqual(account["watch_list"], ["SPX"])
            self.assertEqual(account["positions"], [])
            self.assertEqual(account["orders"], [])

    def test_factor_miner_panel_excludes_observation_only_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configs = root / "FactorMiner" / "factorminer" / "configs"
            configs.mkdir(parents=True)
            (configs / "default.yaml").write_text(
                "execution:\n  cost_bps: 3.0\n", encoding="utf-8"
            )
            panel = pd.DataFrame(
                [
                    {
                        "date": day,
                        "asset_id": asset_id,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 1_000.0,
                        "amount": 100_000.0,
                    }
                    for day in ("2026-07-15", "2026-07-16")
                    for asset_id in ("SPX", "VIX")
                ]
            )
            assets = {
                "tradable": [{"asset_id": "SPX"}],
                "signals": [{"asset_id": "VIX"}],
            }

            with patch.object(build_inputs, "HERE", root):
                output = build_inputs.build_factor_miner(
                    panel, assets, root / "FactorMiner" / "data"
                )

            fm_panel = pd.read_parquet(output)
            self.assertEqual(set(fm_panel["asset_id"]), {"SPX"})
            self.assertEqual(len(fm_panel), 2)


if __name__ == "__main__":
    unittest.main()
