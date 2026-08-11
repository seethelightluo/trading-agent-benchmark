"""Cycle backtest runner (miner_1, 2026-10-08).

Per operator directive (2026-08-11) mandatory cycle flow step 3:
run backtest EXACTLY ONCE to verify strategy.py executes cleanly on the
current ensemble before advancing the worldline. Backtest restores the
original date/account state, so it does not advance the live date.
"""
import json
import traceback


def run_backtest_once():
    from alphacrafter.agent.toolkit.backtest import BacktestTool
    bt = BacktestTool(
        date_file_path="../persistent/date.json",
        dataset_dir_path="../persistent/stock_data",
        account_file_path="../persistent/account.json",
        strategy_file_path="./strategy.py",
        log_file_path="../logs/backtest_results.json",
        mode="a",
    )
    fn = bt.get_implementation()
    out = fn(60)  # EXACTLY ONE backtest call, 60 trading days (~2.4 blocks)
    print("=== BACKTEST OUTPUT ===")
    print(out)
    return out


if __name__ == "__main__":
    try:
        run_backtest_once()
        print("CYCLE_OK: backtest ran exactly once; date/account state restored.")
    except Exception:
        traceback.print_exc()
        raise
