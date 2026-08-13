"""Reconstruct executed weights & block PnL for the 2031-08-01 -> 2031-08-15 block."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

acc = get_account_dict()
nav = acc['total_assets']
pos = {p['symbol']: p['quantity'] for p in acc['positions']}

frames = {}
for a in WATCH:
    df = get_stock_daily_data(symbol=a, days=170)
    if df is None or len(df) < 30:
        frames[a] = None
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    frames[a] = df

# Proposal fired 2031-08-01 using data through previous completed day (0731 close).
# Executed at 0801 close. Estimate executed weights via current qty x 0801 close.
target_date = pd.Timestamp('2031-08-01')
exec_weights = {}
for a in WATCH:
    df = frames.get(a)
    if df is None or target_date not in df.index:
        exec_weights[a] = None
        continue
    px = float(df.loc[target_date, 'close'])
    q = pos.get(a, 0.0)
    exec_weights[a] = q * px

tot = sum(v for v in exec_weights.values() if v is not None and v > 0)
print("== Executed weights (est @0801 close via current qty x 0801 px) ==")
for a in sorted(exec_weights, key=lambda x: -(exec_weights[x] or 0)):
    if exec_weights[a] is not None:
        print(f"  {a:10s} {exec_weights[a]/tot*100:6.2f}%")

# Block returns: 0801 close -> latest close (0814/0815), per asset
last_date = max(df.index.max() for df in frames.values() if df is not None)
print("\nLatest data date:", last_date.date())
print("\n== Per-asset block return 0801->", last_date.date(), " ==")
contrib = {}
for a in WATCH:
    df = frames.get(a)
    if df is None or target_date not in df.index:
        continue
    p0 = float(df.loc[target_date, 'close'])
    p1 = float(df['close'].iloc[-1])
    r = p1 / p0 - 1.0
    w = (exec_weights[a] or 0) / tot
    contrib[a] = (r, w, r * w)
    print(f"  {a:10s} ret {r*100:7.2f}%  w {w*100:6.2f}%  contrib {r*w*100:7.3f}pp")

print("\nSum contrib (approx block PnL):", sum(v[2] for v in contrib.values()) * 100, "%")

# Regime at decision (data through 0731 close)
print("\n== Regime @0731 close (decision recon) ==")
rets = []
for a in WATCH:
    df = frames.get(a)
    if df is None or len(df) < 25:
        continue
    sub = df[df.index <= pd.Timestamp('2031-07-31')]
    if len(sub) < 25:
        continue
    rets.append(float(sub['close'].pct_change().tail(20).mean()))
m = float(np.mean(rets)) if rets else 0
print("20d mean daily ret:", round(m, 5), "-> regime:", "bull" if m > 0.010 else ("bear" if m < -0.010 else "side"))

# MA20 breadth at 0731
above = 0
for a in WATCH:
    df = frames.get(a)
    if df is None:
        continue
    sub = df[df.index <= pd.Timestamp('2031-07-31')]
    if len(sub) < 25:
        continue
    c = float(sub['close'].iloc[-1])
    ma = float(sub['close'].rolling(20).mean().iloc[-1])
    if c >= ma:
        above += 1
print("Breadth above MA20 @0731:", above, "/15")

# Current NAV for reference
print("\nCurrent NAV:", round(nav, 2))
