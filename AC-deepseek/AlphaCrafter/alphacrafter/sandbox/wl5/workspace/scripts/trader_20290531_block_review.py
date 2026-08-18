"""Trader block review: 2029-05-17 -> 2029-05-31 (10 trading days).

Inspect account state and per-asset block returns for memory logging.
Read-only: no backtest/step/rebalance imports.
"""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acc = get_account_dict()
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 4))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("watch_list:", acc.get("watch_list", []))
print("pending orders:", len(acc.get("orders", [])))
print("\nPositions:")
pos_map = {}
for p in acc.get("positions", []):
    sym = p["symbol"]
    pos_map[sym] = p
    plr = p.get("profit_loss_rate", 0.0)
    print(f"  {sym}: qty={p.get('quantity'):.4f} cost={p.get('cost_price')} "
          f"cur={p.get('current_price')} mv={p.get('market_value'):,.0f} plr={plr*100:.1f}%")

assets = acc.get("watch_list", [])
def closes(sym, days=250):
    df = get_stock_daily_data(sym, days=days)
    if df is None or len(df) < 20:
        df = get_index_daily_data(sym, days=days)
    if df is None or len(df) < 20:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]

print("\nBlock return 2029-05-16 close -> latest close (approx per asset):")
blk = {}
for a in assets:
    c = closes(a)
    if c is None:
        continue
    # last close before block start (05-16) and latest close
    pre = c[c.index <= "2029-05-16"]
    post = c[c.index >= "2029-05-16"]
    if len(pre) == 0 or len(post) == 0:
        print(f"  {a}: insufficient data"); continue
    p0 = pre.iloc[-1]
    p1 = post.iloc[-1]
    r = p1 / p0 - 1.0
    blk[a] = r
    mv = pos_map.get(a, {}).get("market_value", 0.0)
    contrib = mv / acc.get("total_assets", 1.0) * r
    print(f"  {a}: {r*100:+.2f}%  (approx contrib {contrib*100:+.2f}pp, mv {mv:,.0f})")

tot = sum(abs(v) for v in blk.values())
print("\nsum abs block ret:", round(tot, 4))
