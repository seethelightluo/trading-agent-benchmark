"""Cycle backtest runner (miner_3, 2026-08-11).

Per operator directive: after confirming factor_ensemble.json has
selected_factors and strategy.py matches it, run the backtest exactly once
before calling step exactly once.
"""
import sys
import traceback
from alphacrafter.agent.toolkit.backtest import BacktestTool

def main():
    bt = BacktestTool(
        date_file_path="../persistent/date.json",
        account_file_path="../persistent/account.json",
        dataset_dir_path="../persistent/stock_data",
        strategy_file_path="./strategy.py",
        log_file_path="../logs/backtest_results.json",
        mode="a",
    )
    fn = bt.get_implementation()
    out = fn(60)  # single backtest run over ~60 trading days (6 cadence blocks)
    print(out)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
