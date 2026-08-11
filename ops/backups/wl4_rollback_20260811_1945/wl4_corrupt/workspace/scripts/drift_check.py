"""Screener drift-check: recompute 6 ensemble factor rank-ICs on live data through
the current visible date, compare vs persisted validation metrics (drift detection).
h=10 forward returns, 15-asset cross-section, >=8 valid names per date.
"""
import csv, math, statistics, sys
from collections import defaultdict

CUT = '2026-08-26'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
          'COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_close(name, path):
    rows = [r for r in csv.DictReader(open(path)) if r['date'] <= CUT]
    return [(r['date'], float(r['close'])) for r in rows]

def series_map(name, path):
    d = load_close(name, path)
    return {dt: c for dt, c in d}, [dt for dt, _ in d]

# Build aligned date-indexed close dicts
close = {}
dates_all = None
for a in ASSETS:
    m, ds = series_map(a, f'../persistent/stock_data/{a}.csv')
    close[a] = m
    dates_all = ds if dates_all is None else [d for d in dates_all if d in m]
dates_all = sorted(dates_all)
# only dates where all assets have a close
dates = [d for d in dates_all if all(d in close[a] for a in ASSETS)]
pos = {d: i for i, d in enumerate(dates)}

# Macro series
eur = series_map('EURUSD', '../persistent/index_data/EURUSD.csv')[0]
vix = series_map('VIX', '../persistent/index_data/VIX.csv')[0]

def pct(d0, d1):
    """return d1/d0 - 1"""
    return d1 / d0 - 1.0

def rolling_beta(y, x, win=60, min_obs=40):
    # y,x aligned lists of (date, value) restricted to win
    n = min(len(y), len(x))
    if n < min_obs:
        return None
    ys = [v for _, v in y[-win:]]
    xs = [v for _, v in x[-win:]]
    mx = statistics.mean(xs); my = statistics.mean(ys)
    vx = sum((xi - mx) ** 2 for xi in xs)
    if vx <= 1e-14:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    return cov / vx

def rank(vals):
    valid = sorted((v, a) for a, v in vals.items() if v is not None and math.isfinite(v))
    out = {a: 0.5 for a in ASSETS}
    n = len(valid)
    for i, (_, a) in enumerate(valid):
        out[a] = i / (n - 1) if n > 1 else 0.5
    return out

def spearman(x, y):
    n = len(x)
    if n < 8:
        return None
    rx = {v: i for i, v in enumerate(sorted(set(x)))}
    ry = {v: i for i, v in enumerate(sorted(set(y)))}
    xs = [rx[v] for v in x]; ys = [ry[v] for v in y]
    mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    vy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0

# Factor value function per date index i (>= 130)
def factor_values(i):
    d = dates[i]
    vals = {fid: {} for fid in FACTOR_IDS}
    # momentum
    for a in ASSETS:
        c = close[a]
        if d in c:
            i0 = pos[d]
            # need 126 history
            pass
    # simpler: precompute close lists
    return vals

# Precompute close lists aligned to dates
cl = {a: [close[a][d] for d in dates] for a in ASSETS}
ret = {a: [None] + [pct(cl[a][i-1], cl[a][i]) for i in range(1, len(dates))] for a in ASSETS}
mkt = [statistics.mean([ret[a][i] for a in ASSETS if ret[a][i] is not None]) if any(ret[a][i] is not None for a in ASSETS) else None for i in range(len(dates))]
eur_ret = [None] + [pct(eur.get(dates[i-1]), eur.get(dates[i])) if eur.get(dates[i-1]) and eur.get(dates[i]) else None for i in range(1, len(dates))]
vix_ret = [None] + [pct(vix.get(dates[i-1]), vix.get(dates[i])) if vix.get(dates[i-1]) and vix.get(dates[i]) else None for i in range(1, len(dates))]

# 20d realized vol per asset (sample std, ddof=1), then 60d std of that (vol-of-vol)
import statistics as _st
vol20 = {a: [None]*60 for a in ASSETS}
for a in ASSETS:
    r = ret[a]
    for i in range(60, len(dates)):
        w = [r[j] for j in range(i-19, i+1) if r[j] is not None]
        if len(w) >= 15:
            vol20[a].append(_st.stdev(w))
        else:
            vol20[a].append(None)
vov = {a: [None]*120 for a in ASSETS}
for a in ASSETS:
    v = vol20[a]
    for i in range(120, len(dates)):
        w = [v[j] for j in range(i-59, i+1) if v[j] is not None]
        if len(w) >= 40:
            vov[a].append(_st.stdev(w))
        else:
            vov[a].append(None)

def vix_move(i):
    if i < 21:
        return None
    v0 = vix.get(dates[i-21]); v1 = vix.get(dates[i])
    return pct(v0, v1) if v0 and v1 else None

def factor_at(i):
    """cross-sectional raw factor values at date index i"""
    out = {}
    for a in ASSETS:
        c = cl[a]
        vals = {}
        if i >= 126:
            vals['mom_120d_skip5'] = pct(c[i-126], c[i-6])
        if i >= 16:
            vals['mom_10d_skip5'] = pct(c[i-16], c[i-6])
        # betas: need 60 prior daily returns
        if i >= 60:
            y = [(dates[j], ret[a][j]) for j in range(i-59, i+1) if ret[a][j] is not None]
            dn = [(dates[j], min(mkt[j], 0.0)) for j in range(i-59, i+1) if mkt[j] is not None]
            eu = [(dates[j], eur_ret[j]) for j in range(i-59, i+1) if eur_ret[j] is not None]
            cn = [(dates[j], ret['CN10Y'][j]) for j in range(i-59, i+1) if ret['CN10Y'][j] is not None]
            vi = [(dates[j], vix_ret[j]) for j in range(i-59, i+1) if vix_ret[j] is not None]
            vals['dn_mkt_beta_60d'] = rolling_beta(y, dn)
            vals['eurusd_beta_60d'] = rolling_beta(y, eu)
            vals['rate_beta_cn10y_60d'] = rolling_beta(y, cn)
            bv = rolling_beta(y, vi)
            vm = vix_move(i)
            vals['vix_beta_cond_60x20'] = -bv * vm if (bv is not None and vm is not None) else None
        if i >= 120:
            vals['vol_of_vol20x60'] = vov[a][i]
        out[a] = vals
    return out

FACTOR_IDS = ['eurusd_beta_60d','rate_beta_cn10y_60d','dn_mkt_beta_60d',
              'mom_120d_skip5','mom_10d_skip5','vix_beta_cond_60x20','vol_of_vol20x60']

# IC series per factor
ic_series = {f: [] for f in FACTOR_IDS}
for i in range(126, len(dates) - 10):
    fv = factor_at(i)
    fr = [pct(cl[a][i], cl[a][i+10]) for a in ASSETS]
    for f in FACTOR_IDS:
        x = [fv[a][f] for a in ASSETS if fv[a][f] is not None]
        y = [fr[k] for k, a in enumerate(ASSETS) if fv[a][f] is not None]
        if len(x) >= 8:
            r = spearman(x, y)
            if r is not None:
                ic_series[f].append((dates[i], r))

print(f"IC window: {len(ic_series['mom_120d_skip5'])} dates, "
      f"{dates[126]} .. {ic_series['mom_120d_skip5'][-1][0] if ic_series['mom_120d_skip5'] else 'NA'}")
print(f"{'factor':<24} {'IC_all':>8} {'IC_60d':>8} {'hit60':>6} {'IC_120d':>8} {'hit120':>7} {'IC_180d':>8} {'hit180':>7} {'persist':>8}")
persist = {'eurusd_beta_60d': -0.0551, 'rate_beta_cn10y_60d': -0.052, 'dn_mkt_beta_60d': 0.0554,
           'mom_120d_skip5': 0.0521, 'mom_10d_skip5': 0.0409, 'vix_beta_cond_60x20': -0.0382,
           'vol_of_vol20x60': 0.0424}
for f in FACTOR_IDS:
    s = ic_series[f]
    if len(s) < 30:
        print(f"{f:<24} insufficient data ({len(s)})"); continue
    ics = [r for _, r in s]
    ic_all = statistics.mean(ics)
    sd = statistics.pstdev(ics)
    icir = ic_all / sd if sd > 0 else 0.0
    hit_all = sum(1 for r in ics if (r > 0) == (persist[f] > 0)) / len(ics)
    def win_stats(n):
        sn = s[-n:]
        icn = statistics.mean([r for _, r in sn])
        hitn = sum(1 for r in [r for _, r in sn] if (r > 0) == (persist[f] > 0)) / len(sn)
        return icn, hitn
    ic60, hit60 = win_stats(60)
    ic120, hit120 = win_stats(120)
    ic180, hit180 = win_stats(180)
    print(f"{f:<24} {ic_all:>8.4f} {ic60:>8.4f} {hit60:>6.3f} {ic120:>8.4f} {hit120:>7.3f} {ic180:>8.4f} {hit180:>7.3f} {persist[f]:>8.4f}")
