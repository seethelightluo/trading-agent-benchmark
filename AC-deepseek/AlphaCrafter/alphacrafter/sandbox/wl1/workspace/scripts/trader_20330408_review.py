"""Trader block review for 2033-04-08 -> 2033-04-22 (decision 04-08, data through 04-07).

Computes per-asset block returns for the memory log + feedback line.
Uses the same market-data API as the live strategy.
"""
import json
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc = json.load(open("../persistent/account.json"))
net = acc["net_assets"]
pos = {p["symbol"]: p for p in acc["positions"]}

# cost-basis weights at the 04-08 rebalance
cb = sum(p["quantity"] * p["cost_price"] for p in acc["positions"])
print(f"net_assets={net:,.2f} cash={acc['available_cash']} costbasis_total={cb:,.2f}")
print(f"{'asset':9s} {'cb_w%':>7s} {'mv_w%':>7s} {'blk_ret%':>9s} {'contrib_pp':>10s}")

rows = []
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=60)
    if df is None or len(df) < 20:
        print(f"{a:9s} NO DATA")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # block: decision close 04-07 -> last close 04-21 (data lag) or use last
    pre = df.loc[df.index <= "2033-04-07", "close"]
    post = df.loc[df.index > "2033-04-07", "close"]
    if len(pre) == 0 or len(post) == 0:
        print(f"{a:9s} WINDOW MISSING pre={len(pre)} post={len(post)}")
        continue
    r = float(post.iloc[-1] / pre.iloc[-1] - 1.0)
    p = pos.get(a)
    w_cb = (p["quantity"] * p["cost_price"] / cb * 100) if p else 0.0
    w_mv = (p["market_value"] / net * 100) if p else 0.0
    contrib = w_cb * r
    rows.append((a, w_cb, w_mv, r * 100, contrib))
    print(f"{a:9s} {w_cb:7.2f} {w_mv:7.2f} {r*100:9.2f} {contrib:10.3f}")

tot = sum(r for _, _, _, _, r in rows)
print(f"\nsum contrib (cb-weighted): {tot:.3f}pp  (period ret {net/ (net/(1-0.0043)) -1:.4%})")
