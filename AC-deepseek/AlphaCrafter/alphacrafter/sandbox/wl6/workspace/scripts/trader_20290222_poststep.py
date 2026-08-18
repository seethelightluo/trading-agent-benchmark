"""Post-step analysis for the 2029-02-22 -> 2029-03-08 live block."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ACC_PRE = {  # read from ../persistent/account.json before step (2029-02-22)
    '000300.SH': 27.22, 'SPX': 18.39, 'HSI': 0.21, 'N225': 0.52, 'SX5E': 23.21,
    '000688.SH': 2.87, 'SOX': 2.91, 'NDX': 7.01, 'XAU': 24.53, 'COPPER': 3520.28,
    'WTI': 1396.19, 'BTC': 0.21, 'ETH': 33.23, 'US10Y': 18478.59, 'CN10Y': 3040.81}

def getdf(sym, n=40):
    df = get_stock_daily_data(sym, days=n)
    if df is None or len(df) == 0:
        df = get_index_daily_data(sym, days=n)
    return df

assets = list(ACC_PRE.keys())
px_pre, px_post = {}, {}
for a in assets:
    df = getdf(a)
    if df is None or len(df) == 0:
        continue
    df = df.sort_values('date')
    d0 = df[df['date'] <= '2029-02-21']
    d1 = df[df['date'] <= '2029-03-07']
    if len(d0):
        px_pre[a] = float(d0.iloc[-1]['close'])
    if len(d1):
        px_post[a] = float(d1.iloc[-1]['close'])

acc = json.load(open('../persistent/account.json'))
pos = {p['symbol']: p for p in acc['positions']}
net = acc['net_assets']

print(f"block 2029-02-22 -> 2029-03-08  net_assets={net:.2f}")
qty_chg = {}
for a in assets:
    q_pre = ACC_PRE[a]
    q_post = pos[a]['quantity']
    qty_chg[a] = q_post - q_pre

turnover_notional = sum(abs(qty_chg[a]) * px_pre.get(a, 0) for a in assets)
turnover = turnover_notional / 1_069_918.50
print(f"approx one-way turnover: {turnover:.3f}  notional {turnover_notional:.0f}  cost_3bp={turnover*0.0003*1_069_918.50:.1f}")

print(f"\n{'sym':10s} {'ret_blk':>8s} {'mv_end':>10s} {'w_end':>6s} {'qty_pre':>9s} {'qty_post':>9s} {'qty_chg':>9s}")
tot = 0.0
for a in assets:
    r = px_post.get(a, 0) / px_pre.get(a, 1) - 1 if (a in px_post and a in px_pre) else float('nan')
    mv = pos[a]['market_value']
    w = mv / net
    tot += w
    print(f"{a:10s} {r*100:7.2f}% {mv:10.0f} {w*100:5.2f}% {ACC_PRE[a]:9.2f} {pos[a]['quantity']:9.2f} {qty_chg[a]:9.2f}")
print(f"sum weights: {tot:.4f}")
