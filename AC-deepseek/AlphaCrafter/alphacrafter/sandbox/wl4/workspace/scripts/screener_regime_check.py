import csv, json, math
import numpy as np

vt = json.load(open('../persistent/date.json'))['visible_through']
assets = ['SPX','NDX','SOX','000300.SH','000688.SH','HSI','N225','SX5E','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(fn, col='close'):
    rows = {}
    for r in csv.DictReader(open(fn)):
        d = r['date']
        if d > vt: continue
        try: rows[d] = float(r[col])
        except: pass
    return rows

px = {a: load(f'../persistent/stock_data/{a}.csv') for a in assets}
eur = load('../persistent/index_data/EURUSD.csv')
vols = {a: load(f'../persistent/stock_data/{a}.csv', 'volume') for a in assets}

all_dates = sorted(set.intersection(*[set(px[a].keys()) for a in assets]) & set(eur.keys()))
print('common dates:', len(all_dates), '| first:', all_dates[0], '| last:', all_dates[-1])

R = np.full((len(all_dates), len(assets)), np.nan)
for i, d in enumerate(all_dates):
    for j, a in enumerate(assets):
        R[i, j] = px[a][d]
Ret = np.diff(R, axis=0) / R[:-1]
dates = all_dates[1:]

eur_arr = np.array([eur[d] for d in all_dates])
eur_ret = np.diff(eur_arr) / eur_arr[:-1]
mkt_ret = np.nanmean(Ret, axis=1)

cn10y_arr = np.array([px['CN10Y'][d] for d in all_dates])
cn10y_ret = np.diff(cn10y_arr) / cn10y_arr[:-1]

def rolling_beta(y, x, win=60, min_obs=40):
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(win - 1, n):
        xs = x[i - win + 1:i + 1]
        ys = y[i - win + 1:i + 1]
        m = ~(np.isnan(xs) | np.isnan(ys))
        if m.sum() < min_obs: continue
        xv = xs[m]; yv = ys[m]
        cov = np.cov(xv, yv)
        if cov[0, 0] == 0 or np.isnan(cov[0, 0]): continue
        out[i] = cov[0, 1] / cov[0, 0]
    return out

i = len(dates) - 1
print('last return date:', dates[i])

dmkt = np.minimum(mkt_ret, 0.0)
fvals = {}
fvals['dn_mkt_beta_60d'] = {a: rolling_beta(Ret[:, j], dmkt)[i] for j, a in enumerate(assets)}
fvals['eurusd_beta_60d'] = {a: rolling_beta(Ret[:, j], eur_ret)[i] for j, a in enumerate(assets)}
fvals['rate_beta_cn10y_60d'] = {a: rolling_beta(Ret[:, j], cn10y_ret)[i] for j, a in enumerate(assets)}

vpc = {}
for a in assets:
    ds = [d for d in all_dates if d in vols[a]]
    rv = []
    for k in range(len(ds) - 1, max(len(ds) - 22, 0), -1):
        d0, d1 = ds[k - 1], ds[k]
        rr = px[a][d1] / px[a][d0] - 1
        vv = vols[a][d1]
        rv.append((rr, vv))
    rv = rv[:20]
    if len(rv) >= 10:
        rr = np.array([x[0] for x in rv]); vv = np.array([x[1] for x in rv])
        if np.std(rr) > 0 and np.std(vv) > 0:
            vpc[a] = float(np.corrcoef(rr, vv)[0, 1])
        else:
            vpc[a] = np.nan
    else:
        vpc[a] = np.nan
fvals['vol_price_corr_20'] = vpc

print('\n=== Current factor values (as of %s) ===' % dates[i])
for f in fvals:
    vals = fvals[f]
    print('\n' + f)
    for a in assets:
        v = vals.get(a)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            print('  %-10s %+.4f' % (a, v))

# Pairwise Spearman corr of the 4 factors cross-sectionally (last date)
print('\n=== Cross-sectional Spearman correlation of factors (last date) ===')
from scipy.stats import spearmanr
names = list(fvals.keys())
for x in range(len(names)):
    for y in range(x + 1, len(names)):
        vx = [fvals[names[x]][a] for a in assets if fvals[names[x]].get(a) is not None and not (isinstance(fvals[names[x]][a], float) and math.isnan(fvals[names[x]][a]))]
        # build aligned pairs
        pairs = []
        for a in assets:
            v1 = fvals[names[x]].get(a); v2 = fvals[names[y]].get(a)
            if v1 is None or v2 is None: continue
            if isinstance(v1, float) and math.isnan(v1): continue
            if isinstance(v2, float) and math.isnan(v2): continue
            pairs.append((v1, v2))
        if len(pairs) >= 5:
            rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
            print('  %s vs %s: rho=%.3f (n=%d)' % (names[x], names[y], rho, len(pairs)))
