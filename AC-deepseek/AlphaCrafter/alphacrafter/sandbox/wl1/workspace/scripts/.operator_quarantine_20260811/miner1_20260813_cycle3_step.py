"""Cycle3 step runner (miner_1, 2026-08-13).

Per operator directive (2026-08-11) mandatory cycle flow:
1) factor_ensemble.json has selected_factors -> confirmed (7 factors, quality_ic_tilt).
2) strategy.py matches ensemble -> confirmed (strategy loads factor_ensemble.json dynamically).
3) Backtest already ran EXACTLY ONCE this cycle (prior turn, clean).
4) Call step tool EXACTLY ONCE to advance one 10-trading-day block (2026-08-13 -> ~2026-08-27).
"""
import json
import traceback


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
    print(out if isinstance(out, str) else json.dumps(out, indent=2, default=str)[:5000])
    return out


if __name__ == "__main__":
    try:
        run_step_once()
        print("CYCLE_OK: step tool called exactly once; one 10-trading-day block advanced.")
    except Exception:
        traceback.print_exc()
        raise
