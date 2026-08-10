"""Market regime assessment for factor screening - data through visible date only."""
import csv, os, math
from datetime import datetime

VISIBLE = '2026-08-12'
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['VIX','DXY','USDJPY','EURUSD','USDCNY']

def load(path):
    rows = {}
    with open(path) as f:
        r = list(csv.DictReader(f))
    for x in r:
        d = x['date']
        if d > VISIBLE:
            continue
        try:
            c = float(x['close'])
        except (TypeError, ValueError):
            continue
        if c and c > 0:
            rows[d] = c
    return rows

def rets(series, n):
    ds = sorted(series.keys())
    out = {}
    for i in range(n, len(ds)):
        out[ds[i]] = series[ds[i]] / series[ds[i-n]] - 1.0
    return out

def daily_rets(series):
    ds = sorted(series.keys())
    out = {}
    for i in range(1, len(ds)):
        out[ds[i]] = series[ds[i]] / series[ds[i-1]] - 1.0
    return out

def vol_ann(series, n=20):
    dr = daily_rets(series)
    ds = sorted(dr.keys())
    out = {}
    for i in range(n, len(ds)):
        win = [dr[ds[j]] for j in range(i-n, i)]
        m = sum(win)/len(win)
        v = sum((x-m)**2 for x in win)/(len(win)-1)
        out[ds[i]] = math.sqrt(v)*math.sqrt(252)
    return out

def ma(series, n):
    ds = sorted(series.keys())
    out = {}
    s = 0.0
    for i, d in enumerate(ds):
        s += series[d]
        if i >= n:
            s -= series[ds[i-n]]
        if i >= n-1:
            out[d] = s/n
    return out

data = {}
for s in TRADABLE:
    data[s] = load(f'../persistent/stock_data/{s}.csv')
for s in OBS:
    data[s] = load(f'../persistent/index_data/{s}.csv')

print('=== PRICE LEVELS & TREND (through', VISIBLE, ') ===')
print(f'{"sym":8s} {"last":>12s} {"ret20":>9s} {"ret60":>9s} {"ret5":>8s} {"vol20_ann":>9s} {"above_ma60":>10s}')
closes = {}
for s in TRADABLE + OBS:
    closes[s] = data[s]

last_dates = {}
for s in TRADABLE + OBS:
    last_dates[s] = max(data[s].keys())

# alignment: use last common date per series
def last_ret(s, n):
    r = rets(data[s], n)
    ds = sorted(r.keys())
    return r[ds[-1]] if ds else float('nan')

def last_vol(s, n=20):
    v = vol_ann(data[s], n)
    ds = sorted(v.keys())
    return v[ds[-1]] if ds else float('nan')

def above_ma(s, n=60):
    m = ma(data[s], n)
    ds = sorted(m.keys())
    if not ds:
        return float('nan')
    d = ds[-1]
    return 1.0 if data[s][d] > m[d] else 0.0

for s in TRADABLE:
    print(f'{s:8s} {data[s][last_dates[s]]:12.1f} {last_ret(s,20)*100:8.1f}% {last_ret(s,60)*100:8.1f}% {last_ret(s,5)*100:7.1f}% {last_vol(s)*100:8.1f}% {above_ma(s):10.0f}')

print()
print('=== OBSERVATION SIGNALS ===')
for s in OBS:
    print(f'{s:8s} last={data[s][last_dates[s]]:10.2f} ret20={last_ret(s,20)*100:7.1f}% ret60={last_ret(s,60)*100:7.1f}%')

# Cross-sectional stats on common dates
common_dates = set(data['SPX'].keys())
for s in TRADABLE:
    common_dates &= set(data[s].keys())
common_dates = sorted(common_dates)
print()
print('n_common_dates:', len(common_dates), 'last:', common_dates[-1])

# cross-sectional dispersion of 20d returns and mean pairwise corr of daily returns
r20_by_date = {}
dr_by_date = {}
for d in common_dates:
    r20_by_date[d] = {}
    dr_by_date[d] = {}
for s in TRADABLE:
    r20 = rets(data[s], 20)
    dr = daily_rets(data[s])
    for d in common_dates:
        if d in r20: r20_by_date[d][s] = r20[d]
        if d in dr: dr_by_date[d][s] = dr[d]

import statistics
disp = []
for d in common_dates:
    vals = list(r20_by_date[d].values())
    if len(vals) >= 10:
        disp.append((d, statistics.pstdev(vals)))
print('cross-sectional dispersion of 20d rets (last 5):', [(d, round(v*100,1)) for d,v in disp[-5:]])

# pairwise correlation of daily returns over last 60 common days
def corr(a, b):
    n = min(len(a), len(b))
    a = a[-n:]; b = b[-n:]
    ma_ = sum(a)/n; mb = sum(b)/n
    cov = sum((a[i]-ma_)*(b[i]-mb) for i in range(n))/(n-1)
    va = sum((x-ma_)**2 for x in a)/(n-1)
    vb = sum((x-mb)**2 for x in b)/(n-1)
    if va == 0 or vb == 0: return 0.0
    return cov/math.sqrt(va*vb)

# build last-60d daily return aligned series per asset
import collections
dr_series = {s: [] for s in TRADABLE}
for d in common_dates[-60:]:
    for s in TRADABLE:
        if d in dr_by_date[d]:
            dr_series[s].append(dr_by_date[d][s])
pairs = []
for i in range(len(TRADABLE)):
    for j in range(i+1, len(TRADABLE)):
        pairs.append(abs(corr(dr_series[TRADABLE[i]], dr_series[TRADABLE[j]])))
print(f'mean pairwise |20d-ish corr| (last 60d): {sum(pairs)/len(pairs):.3f}')

# VIX regime
vix = data['VIX']
vix_dates = sorted(vix.keys())
vix_last = vix[vix_dates[-1]]
vix_30d_ago = vix[vix_dates[-31]] if len(vix_dates) > 31 else vix[vix_dates[0]]
vix_60d_ago = vix[vix_dates[-61]] if len(vix_dates) > 61 else vix[vix_dates[0]]
print(f'VIX last={vix_last:.2f} 30d ago={vix_30d_ago:.2f} ({100*(vix_last/vix_30d_ago-1):+.1f}%) 60d ago={vix_60d_ago:.2f}')
