"""Trader block review 2030-08-08 -> 2030-08-22.

Reconstruct per-asset returns, weight changes, and contribution.
"""
import json
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
          "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc_before = {
    "000300.SH": 127301.84, "SPX": 116101.22, "HSI": 5170.4, "N225": 27614.47,
    "SX5E": 125682.12, "000688.SH": 5170.4, "SOX": 26768.42, "NDX": 108391.32,
    "XAU": 125332.54, "COPPER": 27141.79, "WTI": 84691.45, "BTC": 49619.3,
    "ETH": 82524.21, "US10Y": 122146.16, "CN10Y": 5170.4,
}
NAV0 = 1038826.04

acc_after = {
    "000300.SH": 123752.13, "SPX": 98999.73, "HSI": 5193.74, "N225": 111995.67,
    "SX5E": 8968.12, "000688.SH": 5193.74, "SOX": 84426.33, "NDX": 105470.66,
    "XAU": 55386.74, "COPPER": 123828.72, "WTI": 92480.1, "BTC": 8729.51,
    "ETH": 96059.47, "US10Y": 122542.53, "CN10Y": 5193.74,
}
NAV1 = 1048220.94

# pull 60d of data to capture the block; compute close-to-close block return
# using the last 10 trading days of closes (visible at 08-07 close -> 08-21 close)
rets = {}
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=60)
    if df is None or len(df) < 15:
        rets[a] = None
        continue
    closes = df["close"].astype(float).tolist()
    r = closes[-1] / closes[-11] - 1.0  # 10-day block return
    rets[a] = r

print(f"{'asset':10s} {'w_old':>7s} {'w_new':>7s} {'dW':>7s} {'ret%':>7s} {'contrib%':>8s}")
total_contrib = 0.0
for a in ASSETS:
    wo = acc_before[a] / NAV0
    wn = acc_after[a] / NAV1
    r = rets.get(a)
    if r is None:
        continue
    contrib = wo * r * 100.0
    total_contrib += contrib
    print(f"{a:10s} {wo*100:6.2f} {wn*100:6.2f} {(wn-wo)*100:+6.2f} {r*100:+6.2f} {contrib:+8.2f}")

print(f"\nNAV: {NAV0:.0f} -> {NAV1:.0f}  block_return {(NAV1/NAV0-1)*100:+.2f}%")
print(f"sum contrib (approx, ignores intra-block trades): {total_contrib:+.2f}%")

# turnover estimate: sum of |w_new - w_old| / 2 (approximate one-way migration)
turn = sum(abs(acc_after[a]/NAV1 - acc_before[a]/NAV0) for a in ASSETS) / 2.0
print(f"approx one-way turnover: {turn:.3f}")
print(f"3bp cost estimate on migrated notional: {turn*0.0003*NAV0:.1f}")
