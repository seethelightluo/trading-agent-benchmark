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
                        "date": "2026-07-16",
                        "asset_id": asset_id,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 1_000.0,
                    }
                    for asset_id in ("SPX", "VIX")
                ]
            )
            assets = {
                "baseline_date": "2026-07-16",
                "online_end": "2026-08-16",
                "tradable": [{"asset_id": "SPX"}],
                "signals": [{"asset_id": "VIX"}],
            }

            with patch.object(build_inputs, "HERE", root):
                session = build_inputs.build_alpha_crafter(
                    panel, assets, "wl_test", stage_news=[]
                )

            self.assertTrue((session / "workspace" / "factors").is_dir())
            self.assertTrue((session / "workspace" / "scripts").is_dir())


if __name__ == "__main__":
    unittest.main()
