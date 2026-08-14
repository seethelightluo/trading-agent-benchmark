"""Screener regime analysis - computes cross-sectional metrics as of visible_through date.
Data source: ../persistent/stock_data/*.csv and ../persistent/index_data/*.csv (observation-only).
No future data used: all series are sliced at the visible_through date from ../persistent/date.json.
This is a read-only analysis script (no account/date mutation, no backtest/step imports).
"""
import csv, json, math, os

VISIBLE = json.load(open('../persistent/date.json'))['visible_through']

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

def load(fn):
    rows = list(csv.reader(open(fn)))
    header, data = rows[0], rows[1:]
    out = []
    for r in data:
        try:
            d = dict(zip(header, r))
            out.append((d['date'], float(d['close'])))
        except Exception:
            pass
    out.sort()
    return out

def slice_to(series, cutoff=VISIBLE):
    return [x for x in series if x[0] <= cutoff]

def ret(series, n):
    if len(series) < n + 1:
        return None
    return series[-1][1] / series[-1 - n][1] - 1.0

def range_pos(series, n=252):
    if len(series) < n:
        return None
    window = [c for _, c in series[-n:]]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 0.5
    return (series[-1][1] - lo) / (hi - lo)

def max_consec_gain(series, n=20):
    """max consecutive up-close days within last n days"""
    closes = [c for _, c in series[-(n+1):]]
    streak, best = 0, 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best

def daily_ret_series(series):
    out = []
    for i in range(1, len(series)):
        out.append(series[i][1] / series[i-1][1] - 1.0)
    return out

def vol20(series):
    r = daily_ret_series(series)[-20:]
    if len(r) < 5:
        return None
    m = sum(r)/len(r)
    return math.sqrt(sum((x-m)**2 for x in r)/len(r))

def corr60_spx(series, spx_ret):
    r = daily_ret_series(series)[-60:]
    n = min(len(r), len(spx_ret))
    r, s = r[-n:], spx_ret[-n:]
    mr, ms = sum(r)/n, sum(s)/n
    cov = sum((a-mr)*(b-ms) for a, b in zip(r, s)) / n
    vr = sum((a-mr)**2 for a in r)/n
    vs = sum((b-ms)**2 for b in s)/n
    if vr == 0 or vs == 0:
        return None
    return cov / math.sqrt(vr*vs)

def downbeta60(series, spx_ret):
    """downside beta: beta of asset returns on SPX returns in days when SPX < 0"""
    r = daily_ret_series(series)[-60:]
    n = min(len(r), len(spx_ret))
    r, s = r[-n:], spx_ret[-n:]
    pairs = [(a, b) for a, b in zip(r, s) if b < 0]
    if len(pairs) < 5:
        return None
    mr = sum(a for a, _ in pairs)/len(pairs)
    ms = sum(b for _, b in pairs)/len(pairs)
    cov = sum((a-mr)*(b-ms) for a, b in pairs)/len(pairs)
    vs = sum((b-ms)**2 for b, _ in pairs)/len(pairs)
    if vs == 0:
        return None
    return cov/vs

def mom_skip5(series, n=180):
    """momentum over n days skipping the most recent 5"""
    if len(series) < n + 6:
        return None
    return series[-6][1] / series[-1 - n][1] - 1.0

print('visible_through:', VISIBLE)
print('=' * 110)
print(f"{'asset':>10} {'last':>10} {'20d%':>8} {'60d%':>8} {'180d%':>9} {'rng252':>7} {'streak20':>8} {'vol20':>7} {'corr60':>7} {'dnbeta':>7} {'mom180s5':>9}")
print('=' * 110)

spx_series = slice_to(load('../persistent/stock_data/SPX.csv'))
spx_ret = daily_ret_series(spx_series)
results = {}
for a in ASSETS:
    s = slice_to(load(f'../persistent/stock_data/{a}.csv'))
    r20 = ret(s, 20); r60 = ret(s, 60); r180 = ret(s, 180)
    rp = range_pos(s, 252); sc = max_consec_gain(s, 20)
    v = vol20(s); c = corr60_spx(s, spx_ret) if a != 'SPX' else 1.0
    db = downbeta60(s, spx_ret) if a != 'SPX' else 1.0
    m5 = mom_skip5(s, 180)
    results[a] = dict(r20=r20, r60=r60, r180=r180, rng=rp, streak=sc, vol=v, corr=c, dnbeta=db, mom180=m5)
    f=lambda x,fmt: ('n/a' if x is None else format(x,fmt))
    print(f"{a:>10} {s[-1][1]:>10.2f} {f(r20*100,'7.2f')}% {f(r60*100,'7.2f')}% {f(r180*100,'8.2f')}% {f(rp,'7.3f')} {sc:>8} {f(v*100,'6.2f')}% {f(c,'7.3f')} {f(db,'7.3f')} {f(m5*100,'8.2f')}%")

print('=' * 110)
print('OBSERVATION-ONLY (index_data):')
for a in OBS:
    s = slice_to(load(f'../persistent/index_data/{a}.csv'))
    r20 = ret(s, 20); r60 = ret(s, 60)
    print(f"{a:>10} last={s[-1][1]:>10.2f}  20d={r20*100:>7.2f}%  60d={r60*100:>7.2f}%")

# cross-sectional dispersion
r20s = sorted([results[a]['r20'] for a in ASSETS if results[a]['r20'] is not None])
r180s = sorted([results[a]['r180'] for a in ASSETS if results[a]['r180'] is not None])
if len(r20s)<2 or len(r180s)<2:
    print('insufficient data for dispersion'); raise SystemExit
print('=' * 110)
print(f"20d dispersion: min {r20s[0]*100:.2f}% max {r20s[-1]*100:.2f}% spread {(r20s[-1]-r20s[0])*100:.2f}pp")
print(f"180d dispersion: min {r180s[0]*100:.2f}% max {r180s[-1]*100:.2f}% spread {(r180s[-1]-r180s[0])*100:.2f}pp")

import pickle
with open('/tmp/screener_results.pkl','wb') as f:
    pickle.dump(results, f)
print('saved to /tmp/screener_results.pkl')
