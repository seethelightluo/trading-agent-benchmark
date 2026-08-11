"""Cycle runner (screener, 2026-08-11) per operator directive.

Mandatory cycle flow:
1) factor_ensemble.json exists with selected_factors -> confirmed (9 factors, sum=1.0).
2) strategy.py matches ensemble -> confirmed (loads factor_ensemble.json dynamically).
3) Run backtest EXACTLY ONCE.
4) Call step tool EXACTLY ONCE (advance one 10-trading-day block).
"""
import sys
import traceback
import json


def run_backtest():
    from alphacrafter.agent.toolkit.backtest import BacktestTool
    bt = BacktestTool(
        date_file_path="../persistent/date.json",
        account_file_path="../persistent/account.json",
        dataset_dir_path="../persistent/stock_data",
        strategy_file_path="./strategy.py",
        log_file_path="../logs/backtest_results.json",
        mode="a",
    )
    fn = bt.get_implementation()
    out = fn(60)  # single backtest run
    print("=== BACKTEST OUTPUT ===")
    print(out if isinstance(out, str) else json.dumps(out, indent=2, default=str)[:4000])
    return out


def run_step_once():
    from alphacrafter.agent.toolkit.step import StepTool
    st = StepTool(
        date_file_path="../persistent/date.json",
        dataset_dir_path="../persistent/stock_data",
        account_file_path="../persistent/account.json",
        strategy_file_path="./strategy.py",
        log_file_path="../logs/snapshot.json",
        mode="a",
    )
    fn = st.get_implementation()
    out = fn(10)  # EXACTLY ONE step call, one 10-trading-day block
    print("=== STEP OUTPUT ===")
    print(out if isinstance(out, str) else json.dumps(out, indent=2, default=str)[:4000])
    return out


if __name__ == "__main__":
    try:
        run_backtest()
        run_step_once()
        print("CYCLE_OK: backtest once + step once completed.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
