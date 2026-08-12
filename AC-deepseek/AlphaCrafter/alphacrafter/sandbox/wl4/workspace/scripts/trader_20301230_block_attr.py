"""Trader block attribution for 2030-12-30..2031-01-13 (v36 first execution)."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]


def get(a, n=30):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        try:
            return get_index_daily_data(a, days=n)
        except Exception:
            return None


# executed target weights (from compute_target at 12-30) - reconstruct from cost basis
# Instead: compute asset returns over the block and combine with executed weights.
# Executed weights at 12-30 (target from v36 run):
tgt_w = {
    "000300.SH": 0.0913, "SPX": 0.0692, "HSI": 0.0913, "N225": 0.1075,
    "SX5E": 0.1200, "000688.SH": 0.0304, "SOX": 0.0248, "NDX": 0.0476,
    "XAU": 0.0492, "COPPER": 0.1061, "WTI": 0.0365, "BTC": 0.0137,
    "ETH": 0.0684, "US10Y": 0.0802, "CN10Y": 0.0639,
}

print(f"{'asset':10s} {'w_exec':>7s} {'ret10d%':>8s} {'contrib%':>9s}")
tot = 0.0
for a in assets:
    df = get(a)
    if df is None or len(df) < 12:
        print(a, "NO DATA")
        continue
    df = df.sort_values("date")
    p0 = float(df.iloc[-11]["close"])   # close 12-30 (10 trading days before last)
    p1 = float(df.iloc[-1]["close"])    # close 01-13
    ret = (p1 / p0 - 1.0) * 100.0
    contrib = tgt_w.get(a, 0.0) * ret
    tot += contrib
    print(f"{a:10s} {tgt_w.get(a,0):7.4f} {ret:8.2f} {contrib:9.2f}")
print(f"{'TOTAL':10s} {'1.0000':>7s} {'':>8s} {tot:9.2f}")
