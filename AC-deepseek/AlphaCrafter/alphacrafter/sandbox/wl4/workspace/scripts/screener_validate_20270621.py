"""Screener cycle 2027-06-21: recompute the 4 active factor signals from persistent
price data (through 2027-06-18 only) and evaluate recent cross-sectional rank IC,
ICIR and pairwise factor correlation to inform the quality_ic_tilt ensemble."""
import csv, json, math
import numpy as np

DATA = '../persistent/stock_data/'
IDX = '../persistent/index_data/'
ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU',
          'COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = '2027-06-18'
START = '2026-01-01'  # buffer for warm-up windows

def load_close(sym):
    path = DATA + sym + '.csv'
    with open(path) as f:
        rows = list(csv.DictReader(f))
    dates, closes, vols = [], [], []
    for r in rows:
        d = r['date']
        if d > END or d < START:
            continue
        try:
            c = float(r['close'])
        except (TypeError, ValueError):
            continue
        dates.append(d); closes.append(c)
        try:
            vols.append(float(r['volume']) if r.get('volume') not in (None, '') else np.nan)
        except (TypeError, ValueError):
            vols.append(np.nan)
    return np.array(dates), np.array(closes, dtype=float), np.array(vols, dtype=float)

def load_close_idx(sym):
    path = IDX + sym + '.csv'
    with open(path) as f:
        rows = list(csv.DictReader(f))
    dates, closes = [], []
    for r in rows:
        d = r['date']
        if d > END or d < START:
            continue
        try:
            c = float(r['close'])
        except (TypeError, ValueError):
            continue
        dates.append(d); closes.append(c)
    return np.array(dates), np.array(closes, dtype=float)

# ---- build aligned panel ----
panel = {}
for a in ASSETS:
    d, c, v = load_close(a)
    panel[a] = (d, c, v)

# common date grid (union of all dates, sorted)
all_dates = sorted(set().union(*[set(d) for d, _, _ in panel.values()]))
didx = {dt: i for i, dt in enumerate(all_dates)}
n = len(all_dates)

close_mat = np.full((len(ASSETS), n), np.nan)
vol_mat = np.full((len(ASSETS), n), np.nan)
for i, a in enumerate(ASSETS):
    d, c, v = panel[a]
    for j, dt in enumerate(d):
        k = didx[dt]
        close_mat[i, k] = c[j]
        vol_mat[i, k] = v[j]

ret = np.full_like(close_mat, np.nan)
ret[:, 1:] = close_mat[:, 1:] / close_mat[:, :-1] - 1.0

mkt_ret = np.nanmean(ret, axis=0)          # equal-weight cross-asset market
down_mkt = np.where(mkt_ret < 0, mkt_ret, 0.0)

# EURUSD, CN10Y pct changes (CN10Y from stock_data already in panel; use its own series)
d_eur, c_eur = load_close_idx('EURUSD')
eur_map = dict(zip(d_eur, c_eur))
eur_pct = np.full(n, np.nan)
for k in range(1, n):
    dt_prev, dt_cur = all_dates[k-1], all_dates[k]
    if dt_prev in eur_map and dt_cur in eur_map and eur_map[dt_prev] > 0:
        eur_pct[k] = eur_map[dt_cur] / eur_map[dt_prev] - 1.0

cn10y = close_mat[ASSETS.index('CN10Y'), :]
cn_pct = np.full(n, np.nan)
cn_pct[1:] = cn10y[1:] / cn10y[:-1] - 1.0

def rolling_beta(x, y, w, min_obs=40):
    """rolling beta of y on x (x is regressor), y = a + b*x."""
    out = np.full(n, np.nan)
    for k in range(w, n):
        xw, yw = x[k-w+1:k+1], y[k-w+1:k+1]
        m = ~(np.isnan(xw) | np.isnan(yw))
        if m.sum() < min_obs:
            continue
        xv, yv = xw[m], yw[m]
        sx = xv - xv.mean()
        denom = np.sum(sx * sx)
        if abs(denom) < 1e-12:
            continue
        out[k] = np.sum(sx * (yv - yv.mean())) / denom
    return out

def rolling_corr(x, y, w, min_obs=15):
    out = np.full(n, np.nan)
    for k in range(w, n):
        xw, yw = x[k-w+1:k+1], y[k-w+1:k+1]
        m = ~(np.isnan(xw) | np.isnan(yw))
        if m.sum() < min_obs:
            continue
        c = np.corrcoef(xw[m], yw[m])
        if np.isfinite(c[0, 1]):
            out[k] = c[0, 1]
    return out

# ---- factor signals (cross-sectional: rows=assets, cols=dates) ----
sig = {}
# dn_mkt_beta_60d: beta of asset ret on down-market returns
dn = np.full_like(ret, np.nan)
for i in range(len(ASSETS)):
    dn[i] = rolling_beta(down_mkt, ret[i], 60, min_obs=40)
sig['dn_mkt_beta_60d'] = dn
# vol_price_corr_20: rolling corr(ret, volume, 20)
vc = np.full_like(ret, np.nan)
for i in range(len(ASSETS)):
    vc[i] = rolling_corr(ret[i], vol_mat[i], 20, min_obs=15)
sig['vol_price_corr_20'] = vc
# eurusd_beta_60d
eb = np.full_like(ret, np.nan)
for i in range(len(ASSETS)):
    eb[i] = rolling_beta(eur_pct, ret[i], 60, min_obs=40)
sig['eurusd_beta_60d'] = eb
# rate_beta_cn10y_60d
rb = np.full_like(ret, np.nan)
for i in range(len(ASSETS)):
    rb[i] = rolling_beta(cn_pct, ret[i], 60, min_obs=40)
sig['rate_beta_cn10y_60d'] = rb

DIR = {'dn_mkt_beta_60d': 1, 'vol_price_corr_20': 1, 'eurusd_beta_60d': -1, 'rate_beta_cn10y_60d': -1}

# ---- rank IC vs forward 10d returns ----
fwd10 = np.full_like(ret, np.nan)
fwd10[:, :-10] = close_mat[:, 10:] / close_mat[:, :-10] - 1.0

def spearman(xs, ys):
    xr = np.argsort(np.argsort(xs))
    yr = np.argsort(np.argsort(ys))
    xr = xr - xr.mean(); yr = yr - yr.mean()
    denom = np.sqrt(np.sum(xr**2) * np.sum(yr**2))
    return float(np.sum(xr * yr) / denom) if denom > 0 else np.nan

def ic_series(sig_mat, fwd, use_assets=None):
    """daily cross-sectional IC series (raw signal vs fwd ret), sign NOT applied."""
    out = []
    for k in range(n):
        x = sig_mat[:, k]; y = fwd[:, k]
        m = ~(np.isnan(x) | np.isnan(y))
        if use_assets is not None:
            mask = np.zeros(len(ASSETS), dtype=bool)
            for a in use_assets:
                mask[ASSETS.index(a)] = True
            m = m & mask
        if m.sum() < 8:
            continue
        out.append((all_dates[k], spearman(x[m], y[m])))
    return out

ALL_A = set(ASSETS)
LIVE_A = [a for a in ASSETS if a not in ('HSI', 'ETH')]

windows = {'last_60d': 60, 'last_120d': 120, 'last_260d': 260}
print('=== Rank IC (raw signal) by window ===')
print(f'{"factor":22s}', end='')
for wname in windows:
    print(f'{wname:>12s}', end='')
print(f'{"icir_120d":>12s} {"q_120d":>10s}')
for fname in sig:
    ics_all = ic_series(sig[fname], fwd10, use_assets=None)
    ics_live = ic_series(sig[fname], fwd10, use_assets=LIVE_A)
    dates_all = [d for d, _ in ics_all]
    vals_all = np.array([v for _, v in ics_all])
    dates_live = [d for d, _ in ics_live]
    vals_live = np.array([v for _, v in ics_live])
    line = f'{fname:22s}'
    for wname, wd in windows.items():
        # last wd dates by index on all-asset series
        sub = vals_all[-wd:]
        line += f'{np.nanmean(sub):12.4f}'
    # icir over last 120d (live)
    sub120 = vals_live[-120:]
    icir = (np.nanmean(sub120) / np.nanstd(sub120)) if np.nanstd(sub120) > 0 else np.nan
    q = abs(np.nanmean(sub120)) * abs(icir) if np.isfinite(icir) else np.nan
    line += f'{icir:12.3f} {q:10.5f}'
    print(line)
    # also 60d live
    sub60 = vals_live[-60:]
    ic60 = np.nanmean(sub60)
    print(f'{"":22s}   live(no HSI/ETH) 60d IC={ic60:.4f}  n60={np.sum(~np.isnan(sub60))}')

print()
print('=== pairwise factor correlation (last 120d, live assets) ===')
names = list(sig.keys())
corr_mat = np.full((len(names), len(names)), np.nan)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j < i:
            continue
        xs, ys = [], []
        for k in range(n - 120, n):
            xi = [sig[a][ii, k] for ii in range(len(ASSETS)) if ASSETS[ii] in LIVE_A]
            yi = [sig[b][ii, k] for ii in range(len(ASSETS)) if ASSETS[ii] in LIVE_A]
            m = ~(np.isnan(xi) | np.isnan(yi))
            if m.sum() < 5:
                continue
            xs += list(np.array(xi)[m]); ys += list(np.array(yi)[m])
        if len(xs) >= 30:
            corr_mat[i, j] = np.corrcoef(xs, ys)[0, 1]
print('        ' + ''.join(f'{nm[:10]:>12s}' for nm in names))
for i, a in enumerate(names):
    row = f'{a[:10]:>10s}'
    for j in range(len(names)):
        v = corr_mat[i, j] if j >= i else corr_mat[j, i]
        row += f'{v:12.2f}' if np.isfinite(v) else f'{"":12s}'
    print(row)

print()
print('=== recent cross-sectional signal snapshot (2027-06-18) ===')
k_last = n - 1
print(f'{"asset":10s}', *[f'{nm[:12]:>14s}' for nm in names])
for i, a in enumerate(ASSETS):
    vals = []
    for nm in names:
        v = sig[nm][i, k_last]
        vals.append(f'{v:14.4f}' if np.isfinite(v) else f'{"":14s}')
    print(f'{a:10s}', *vals)
print('as-of date:', all_dates[k_last])
