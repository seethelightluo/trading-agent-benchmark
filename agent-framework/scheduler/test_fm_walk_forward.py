import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

AGENT_FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_FRAMEWORK))

from scheduler.fm_walk_forward import run_forward, slice_panel
from scheduler.run_pipeline import fm_window_cutoffs


ASSETS = [f"A{i:02d}" for i in range(15)]


def _panel(path: Path, online_days: int = 12) -> tuple[Path, list[pd.Timestamp]]:
    days = list(pd.bdate_range("2026-07-16", periods=online_days))
    rows = []
    for day in [pd.Timestamp("2026-07-15"), *days]:
        for asset in ASSETS:
            is_first_online = day == days[0]
            rows.append({
                "date": day,
                "asset_id": asset,
                "open": 200.0 if is_first_online else 100.0,
                "high": 220.0 if is_first_online else 100.0,
                "low": 200.0 if is_first_online else 100.0,
                "close": 220.0 if is_first_online else 100.0,
                "volume": 1_000.0,
                "amount": 100_000.0,
            })
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path, days


def _strong_target(*_args, **_kwargs):
    selected = ASSETS[:3]
    return (
        {asset: 1 / 3 for asset in selected},
        [1, 2],
        {asset: (0.01 if asset in selected else -0.01) for asset in ASSETS},
    )


class FactorMinerForwardTests(unittest.TestCase):
    def test_window_cutoffs_start_after_frozen_history_in_ten_day_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel, days = _panel(Path(tmp) / "panel.parquet", online_days=21)

            warmup, cutoffs = fm_window_cutoffs(
                panel,
                baseline_date="2026-07-16",
                online_end=days[-1].strftime("%Y-%m-%d"),
                cadence_days=10,
            )

            self.assertEqual(warmup, "2026-07-15")
            self.assertEqual(cutoffs[0], days[9].strftime("%Y-%m-%d"))
            self.assertEqual(cutoffs[1], days[19].strftime("%Y-%m-%d"))
            self.assertEqual(cutoffs[2], days[20].strftime("%Y-%m-%d"))

    def test_slice_is_tradable_only_and_strictly_cut_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, _ = _panel(root / "panel.parquet", online_days=2)
            frame = pd.read_parquet(panel)
            frame = pd.concat([
                frame,
                pd.DataFrame([{
                    "date": pd.Timestamp("2026-07-15"),
                    "asset_id": "SIGNAL_ONLY",
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "volume": 1,
                    "amount": 1,
                }]),
            ], ignore_index=True)
            frame.to_parquet(panel, index=False)

            out = slice_panel(
                panel,
                cutoff="2026-07-15",
                tradable_ids=ASSETS,
                out=root / "visible.parquet",
            )
            visible = pd.read_parquet(out)

            self.assertEqual(set(visible["asset_id"]), set(ASSETS))
            self.assertEqual(visible["datetime"].max(), pd.Timestamp("2026-07-15"))

    def test_first_trade_uses_baseline_open_and_executes_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, days = _panel(root / "panel.parquet", online_days=1)
            with patch("scheduler.fm_walk_forward._target_weights", side_effect=_strong_target):
                state = run_forward(
                    panel,
                    library_path=root / "library.json",
                    config_path=root / "config.yaml",
                    output_dir=root / "out",
                    tradable_ids=ASSETS,
                    history_end="2026-07-15",
                    baseline_date="2026-07-16",
                    online_end=days[-1].strftime("%Y-%m-%d"),
                )

            # Initial allocation is explicitly free: 100M bought at 200 and
            # marked at 220 => 110M.
            self.assertAlmostEqual(state["nav"], 110_000_000.0, places=2)
            self.assertEqual(state["state_version"], 3)
            self.assertEqual(state["schema_version"], 1)
            self.assertEqual(state["initial_capital"], 100_000_000.0)
            self.assertEqual(state["contract"]["history_end"], "2026-07-15")
            self.assertEqual(len(state["shares"]), 3)
            decision = state["decisions"][0]
            self.assertEqual(decision["decision_date"], "2026-07-15")
            self.assertEqual(decision["execution_date"], "2026-07-16")
            self.assertEqual(decision["pre_trade_nav"], 100_000_000.0)
            self.assertTrue(decision["executed"])

    def test_missing_execution_open_uses_visible_previous_close_not_same_day_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, days = _panel(root / "panel.parquet", online_days=1)
            frame = pd.read_parquet(panel)
            selected = set(ASSETS[:3])
            date_col = "datetime" if "datetime" in frame.columns else "date"
            mask = (pd.to_datetime(frame[date_col]) == days[0]) & frame["asset_id"].isin(selected)
            frame.loc[mask, "open"] = float("nan")
            frame.loc[mask, "close"] = 220.0
            frame.to_parquet(panel, index=False)

            with patch("scheduler.fm_walk_forward._target_weights", side_effect=_strong_target):
                state = run_forward(
                    panel,
                    library_path=root / "library.json",
                    config_path=root / "config.yaml",
                    output_dir=root / "out",
                    tradable_ids=ASSETS,
                    history_end="2026-07-15",
                    baseline_date="2026-07-16",
                    online_end=days[-1].strftime("%Y-%m-%d"),
                )

            # The prior visible close is 100.  Filling from the unknown same-day
            # close (220) would leave NAV near 100M instead of earning the move.
            self.assertAlmostEqual(state["nav"], 220_000_000.0, places=2)

    def test_first_trade_without_factors_is_equal_weight_full_investment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, days = _panel(root / "panel.parquet", online_days=1)

            with patch("scheduler.fm_walk_forward._target_weights", return_value=({}, [], {})):
                state = run_forward(
                    panel,
                    library_path=root / "library.json",
                    config_path=root / "config.yaml",
                    output_dir=root / "out",
                    tradable_ids=ASSETS,
                    history_end="2026-07-15",
                    baseline_date="2026-07-16",
                    online_end=days[-1].strftime("%Y-%m-%d"),
                )

            self.assertEqual(len(state["shares"]), 15)
            self.assertEqual(state["cash"], 0.0)
            decision = state["decisions"][0]
            self.assertTrue(decision["executed"])
            self.assertEqual(decision["skip_reason"], "")
            self.assertAlmostEqual(decision["decision_edge_threshold_bps"], 1.5)
            self.assertAlmostEqual(sum(decision["executed_target_weights"].values()), 1.0)

    def test_resume_keeps_original_ten_day_cadence_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel, days = _panel(root / "panel.parquet", online_days=12)
            out = root / "out"
            with patch("scheduler.fm_walk_forward._target_weights", side_effect=_strong_target) as target:
                run_forward(
                    panel,
                    library_path=root / "library.json",
                    config_path=root / "config.yaml",
                    output_dir=out,
                    tradable_ids=ASSETS,
                    history_end="2026-07-15",
                    baseline_date="2026-07-16",
                    online_end=days[4].strftime("%Y-%m-%d"),
                )
                run_forward(
                    panel,
                    library_path=root / "library.json",
                    config_path=root / "config.yaml",
                    output_dir=out,
                    tradable_ids=ASSETS,
                    history_end="2026-07-15",
                    baseline_date="2026-07-16",
                    online_end=days[-1].strftime("%Y-%m-%d"),
                )

            state = json.loads((out / "forward_state.json").read_text())
            self.assertEqual(target.call_count, 2)
            self.assertEqual(
                [item["execution_date"] for item in state["decisions"]],
                [days[0].strftime("%Y-%m-%d"), days[10].strftime("%Y-%m-%d")],
            )
            equity = pd.read_csv(out / "equity.csv")
            self.assertEqual(len(equity), 12)
            self.assertEqual(equity["date"].nunique(), 12)


if __name__ == "__main__":
    unittest.main()
