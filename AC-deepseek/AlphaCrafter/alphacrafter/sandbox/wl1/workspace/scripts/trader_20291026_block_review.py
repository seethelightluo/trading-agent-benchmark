"""Trader block review: 2029-10-26 -> 2029-11-09.

Compute per-asset returns over the block (decision close 10-25 -> last
completed close 11-08) and combine with the executed 10-26 target weights to
estimate per-asset PnL contribution.
"""
import json
from alphacrafter.sim.utils import get_stock_daily_data

TARGETS = {
    "000300.SH": 0.10663132690334808, "SPX": 0.04122088315557638,
    "HSI": 0.07376368564682087, "N225": 0.06105417316187245,
    "SX5E": 0.09222168813262538, "000688.SH": 0.08432307132495537,
    "SOX": 0.06870768774625993, "NDX": 0.03037328232516154,
    "XAU": 0.08461128647723572, "COPPER": 0.06105417316187245,
    "WTI": 0.06105417316187245, "BTC": 0.06674329722118276,
    "ETH": 0.053256702778819345, "US10Y": 0.06291608481640605,
    "CN10Y": 0.05206848398599122,
}
START = "2029-10-25"
END = "2029-11-08"

rows = []
for sym, w in TARGETS.items():
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None or len(df) == 0:
        rows.append((sym, w, None, None, None))
        continue
    df = df.copy()
    df["date"] = df["date"].astype(str)
    s = df.set_index("date")["close"].astype(float)
    if START in s.index and END in s.index:
        r = s[END] / s[START] - 1.0
        rows.append((sym, w, r, r * w, s[START]))
    else:
        rows.append((sym, w, None, None, s.index[-1]))

rows.sort(key=lambda x: -(x[3] if x[3] is not None else -9))
tot_contrib = sum(r[3] for r in rows if r[3] is not None)
print(f"{'asset':10s} {'weight':>7s} {'ret%':>8s} {'contrib%':>9s}  start_px")
for sym, w, r, c, px in rows:
    if r is None:
        print(f"{sym:10s} {w*100:6.2f}% {'NA':>8s} {'NA':>9s}")
    else:
        print(f"{sym:10s} {w*100:6.2f}% {r*100:7.2f}% {c*100:8.2f}%  {px:.2f}")
print(f"\nsum contrib ~ {tot_contrib*100:.2f}% (ignores intra-block drift/cost)")
