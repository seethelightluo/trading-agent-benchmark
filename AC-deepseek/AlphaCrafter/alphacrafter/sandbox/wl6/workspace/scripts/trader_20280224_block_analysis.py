"""Block analysis 2028-02-10 -> 2028-02-24: per-asset returns, PnL contrib, regime."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

WATCH = get_account_dict()["watch_list"]


def series(a):
    for fn in (get_stock_daily_data, get_index_daily_data):
        try:
            df = fn(a, days=300)
            if df is not None and len(df):
                return df
        except Exception:
            continue
    return None


START = pd.Timestamp("2028-02-10")

print("=== per-asset block returns (close 2028-02-10 -> close 2028-02-24) ===")
block_ret = {}
for a in WATCH:
    df = series(a)
    if df is None or len(df) < 30:
        print(f"{a:10s} no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    try:
        idx_start = dates[dates <= START].index[-1]
    except IndexError:
        print(f"{a:10s} start missing")
        continue
    p0 = float(df.loc[idx_start, "close"])
    p1 = float(df.iloc[-1]["close"])
    r = p1 / p0 - 1.0
    block_ret[a] = r
    print(f"{a:10s} {p0:12.4f} -> {p1:12.4f}  {r*100:7.2f}%")

print("\n=== regime drift at 2028-02-10 decision (cross-asset 20d t-stat) ===")
closes = {}
for a in WATCH:
    df = series(a)
    if df is None:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    closes[a] = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
panel = pd.concat(closes, axis=1).dropna()
pre = panel[panel.index <= START]
if len(pre) >= 30:
    rets = pre.pct_change().dropna()
    mkt = rets.mean(axis=1)
    r20 = float(mkt.tail(20).mean())
    v20 = float(mkt.tail(20).std())
    trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
    regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")
    print(f"trend_t={trend:.3f} regime={regime}")

print("\n=== rebalance + cost from account.json ===")
acc = json.load(open('../persistent/account.json'))
hist = acc.get('rebalance_history', [])
for h in hist[-3:]:
    print(json.dumps({k: h.get(k) for k in (
        'date', 'initial_allocation', 'pre_trade_nav', 'post_trade_nav',
        'transferred_notional', 'cost', 'cost_bps', 'edge_bps', 'gross_edge',
        'turnover')}, default=str)[:400])
print('cumulative_transaction_cost', acc.get('cumulative_transaction_cost'))
print('net_assets', acc.get('net_assets'))
