"""Trader cycle129 (2034-08-31 -> 2034-09-14) per-asset in-block returns and drift attribution."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

a = json.load(open("../persistent/account.json"))
na = a["net_assets"]
pos = {p["symbol"]: p for p in a["positions"]}

print(f"NAV end: {na:,.2f}")
print(f"{'symbol':10s} {'close 08-30':>12s} {'close 09-13':>12s} {'ret%':>8s} {'wt_now%':>8s} {'contrib%':>8s}")
tot = 0.0
rows = []
for s in WATCH:
    df = get_stock_daily_data(s, days=60)
    if df is None or len(df) < 30:
        print(s, "no data")
        continue
    c = df["close"].astype(float)
    dts = df["date"].astype(str)
    # last close is 09-13 (visible_through); find 08-30 close
    mask = dts <= "2034-08-30"
    c0 = float(c[mask].iloc[-1]) if mask.any() else float("nan")
    c1 = float(c.iloc[-1])
    ret = c1 / c0 - 1.0 if c0 == c0 and c0 > 0 else float("nan")
    wt = pos[s]["market_value"] / na if s in pos else 0.0
    contrib = wt * ret if ret == ret else 0.0
    rows.append((s, c0, c1, ret, wt, contrib))
    tot += contrib
    print(f"{s:10s} {c0:12.4f} {c1:12.4f} {ret*100:8.2f} {wt*100:8.2f} {contrib*100:8.2f}")

print(f"\nSum of contrib (drift approx): {tot*100:.2f}%")
print(f"NAV change: {(na/1473424.4737 - 1)*100:.2f}%  (from account.json.bak pre-step NAV)")
