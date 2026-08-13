"""Trader block analysis: 2032-08-13 rebalance -> 2032-08-27 block end."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

acct = json.load(open('../persistent/account.json'))
rh = acct['rebalance_history']
last = rh[-1]
assert last['date'] == '2032-08-13', last['date']
exec_w = last['executed_target_weights']
pre_trade_nav = last['pre_trade_nav']
post_trade_nav = last['post_trade_nav']
cost = last['cost']
print("== 08-13 rebalance ==")
print(f"pre_trade_nav {pre_trade_nav:.2f} post_trade_nav {post_trade_nav:.2f} cost {cost:.2f} "
      f"turnover {last['one_way_turnover']:.4f} gross_edge_bps {last['gross_edge_bps']:.2f}")
print("executed weights:")
for a, w in sorted(exec_w.items(), key=lambda x: -x[1]):
    print(f"  {a:10s} {w*100:6.2f}%")

frames = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=60)
    if df is None or len(df) < 30:
        frames[a] = None
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    frames[a] = df

t0 = pd.Timestamp('2032-08-13')
last_date = max(df.index.max() for df in frames.values() if df is not None)
print("\nLatest data date:", last_date.date())

print("\n== Per-asset block return 08-13 ->", last_date.date(), "==")
contrib = {}
for a in WATCH:
    df = frames.get(a)
    if df is None or t0 not in df.index or last_date not in df.index:
        continue
    p0 = float(df.loc[t0, 'close'])
    p1 = float(df.loc[last_date, 'close'])
    r = p1 / p0 - 1.0
    w = exec_w[a]
    contrib[a] = (r, w, r * w)
    print(f"  {a:10s} ret {r*100:7.2f}%  w {w*100:6.2f}%  contrib {r*w*100:7.3f}pp")

tot_contrib = sum(v[2] for v in contrib.values())
print(f"\nSum contrib (approx block PnL): {tot_contrib*100:.3f}%")

nav0 = post_trade_nav
nav1 = acct['total_assets']
print(f"\nAccount NAV: {nav0:.2f} -> {nav1:.2f}  block PnL {nav1/nav0 - 1:.4%} "
      f"({nav1-nav0:,.0f} USD)")

# daily path estimate using executed weights
dates = frames['SPX'].index
block_days = [d for d in dates if t0 <= d <= last_date]
rets = {}
for d in block_days:
    r = 0.0
    for a in WATCH:
        df = frames.get(a)
        if df is None or d not in df.index:
            continue
        idx = df.index.get_loc(d)
        if idx == 0:
            continue
        p0 = float(df['close'].iloc[idx-1])
        p1 = float(df['close'].iloc[idx])
        r += exec_w[a] * (p1/p0 - 1.0)
    rets[d] = r

path = []
cum = 1.0
peak = 1.0
mdd = 0.0
for d in block_days:
    cum *= (1 + rets.get(d, 0.0))
    peak = max(peak, cum)
    mdd = max(mdd, (peak - cum) / peak)
    path.append((d, rets.get(d, 0.0), cum))

rarr = np.array([r for _, r, _ in path])
cum_end = path[-1][2]
ann = (cum_end ** (252/len(rarr)) - 1) if len(rarr) > 0 else 0.0
sharpe = (rarr.mean() / rarr.std() * np.sqrt(252)) if rarr.std() > 0 else 0.0
calmar = ann / mdd if mdd > 0 else 0.0
print(f"\nDaily-path estimate: {len(rarr)} days, cum {cum_end-1:.4%}, "
      f"ann {ann:.4%}, Sharpe {sharpe:.2f}, MaxDD {mdd:.4%}, Calmar {calmar:.2f}")
print("\nDaily returns:")
for d, r, c in path:
    print(f"  {d.date()} {r*100:+7.3f}%  cum {c-1:+.4%}")

# drift: current weights vs executed
print("\n== Current weights (drift @", last_date.date(), ") ==")
mv = {p['symbol']: p['market_value'] for p in acct['positions']}
tot = sum(mv.values())
for a in sorted(mv, key=lambda x: -mv[x]):
    print(f"  {a:10s} {mv[a]/tot*100:6.2f}%  (exec {exec_w[a]*100:5.2f}%, "
          f"drift {(mv[a]/tot - exec_w[a])*100:+5.2f}pp)")

# regime at decision (data through 08-12 close)
print("\n== Regime at 08-13 decision (data through 08-12) ==")
t1 = pd.Timestamp('2032-08-12')
above20 = 0
above60 = 0
dly = []
for a in WATCH:
    df = frames.get(a)
    if df is None or t1 not in df.index:
        continue
    idx = df.index.get_loc(t1)
    if idx < 60:
        continue
    px = float(df['close'].iloc[idx])
    ma20 = float(df['close'].iloc[idx-20:idx].mean())
    ma60 = float(df['close'].iloc[idx-60:idx].mean())
    above20 += (px > ma20)
    above60 += (px > ma60)
    dly.append(float(df['pct_change'].iloc[idx-20:idx].mean()) if 'pct_change' in df else 0.0)
print(f"breadth above MA20: {above20}/15, above MA60: {above60}/15")
if dly:
    print(f"20d mean daily ret: {np.mean(dly)*100:.3f}%")
