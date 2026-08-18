"""Screener: compute current factor values + recent cross-sectional IC on the 15-asset universe.
Data through 2029-01-29 only (visible_through). Read-only analysis; does NOT touch live account."""
import csv, math, numpy as np, statistics

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_close(path, max_date='2029-01-29'):
    rows = list(csv.reader(open(path)))
    out = {}
    for r in rows[1:]:
        if len(r) < 2 or not r[0] or r[0] > max_date:
            continue
        try:
            v = float(r[1])
        except:
            continue
        if math.isnan(v):
            continue
        out[r[0]] = v
    return out

# Build aligned date-indexed price matrix
closes = {}
for a in ASSETS:
    closes[a] = load_close('../persistent/stock_data/%s.csv' % a)

dates = sorted(set.intersection(*[set(closes[a].keys()) for a in ASSETS]))
print('aligned dates:', len(dates), dates[0], '->', dates[-1])

P = np.full((len(dates), len(ASSETS)), np.nan)
for j, a in enumerate(ASSETS):
    for i, d in enumerate(dates):
        P[i, j] = closes[a][d]

R = np.diff(P, axis=0) / P[:-1]  # daily returns, shape (T-1, 15)
D = dates[1:]

# EW market return
mkt = np.nanmean(R, axis=1)

def rolling(arr, w, func, minp=None):
    T = len(arr)
    out = np.full(T, np.nan)
    for i in range(w - 1, T):
        win = arr[i - w + 1: i + 1]
        if minp is not None:
            if np.sum(~np.isnan(win)) < minp:
                continue
        out[i] = func(win)
    return out

def beta_win(x, m):
    # slope of x on m
    mask = ~(np.isnan(x) | np.isnan(m))
    if mask.sum() < 30:
        return np.nan
    xx = x[mask]; mm = m[mask]
    if np.std(mm) < 1e-12:
        return np.nan
    return np.cov(xx, mm)[0, 1] / np.var(mm)

def corr_win(x, m):
    mask = ~(np.isnan(x) | np.isnan(m))
    if mask.sum() < 30:
        return np.nan
    return np.corrcoef(x[mask], m[mask])[0, 1]

# Factor computations on aligned daily returns. Return arrays indexed by D (length T-1).
T = len(R)
F = {fid: np.full((T, len(ASSETS)), np.nan) for fid in
     ['rel_mom_20d_skip5','downside_vol_ratio_20','beta_ew_60d','dxy_beta_cond_60x20',
      'max_ret_20d','kurt_20d_skip5','corr_ew_60']}

# --- rel_mom_20d_skip5: 20d momentum skipping last 5d, cross-sectionally demeaned ---
for i in range(5, T):
    mom = P[i, :] / P[i - 20, :] - 1  # using price at t-5 relative to t-25
    med = np.nanmedian(mom)
    F['rel_mom_20d_skip5'][i, :] = mom - med

# --- max_ret_20d ---
for i in range(19, T):
    win = R[i - 19: i + 1, :]
    F['max_ret_20d'][i, :] = np.nanmax(win, axis=0)

# --- kurt_20d_skip5: kurtosis of daily returns over t-25..t-5 (skip last 5d) ---
for i in range(25, T):
    win = R[i - 25: i - 5, :]  # 20 obs
    for j in range(len(ASSETS)):
        x = win[:, j]
        x = x[~np.isnan(x)]
        if len(x) < 12:
            continue
        mu = np.mean(x)
        sd = np.std(x)
        if sd < 1e-12:
            continue
        k = np.mean((x - mu) ** 4) / sd ** 4
        F['kurt_20d_skip5'][i, j] = k

# --- downside_vol_ratio_20 (flipped) ---
for i in range(19, T):
    win = R[i - 19: i + 1, :]
    for j in range(len(ASSETS)):
        x = win[:, j]
        x = x[~np.isnan(x)]
        if len(x) < 12:
            continue
        sd = np.std(x)
        if sd < 1e-12:
            continue
        ds = np.sqrt(np.mean(np.minimum(x - np.mean(x), 0) ** 2))
        F['downside_vol_ratio_20'][i, j] = -(ds / sd)

# --- beta_ew_60d ---
for i in range(59, T):
    for j in range(len(ASSETS)):
        F['beta_ew_60d'][i, j] = beta_win(R[i - 59: i + 1, j], mkt[i - 59: i + 1])

# --- corr_ew_60: mean pairwise 60d correlation ---
for i in range(59, T):
    win = R[i - 59: i + 1, :]
    for j in range(len(ASSETS)):
        cs = []
        for k in range(len(ASSETS)):
            if k == j:
                continue
            c = corr_win(win[:, j], win[:, k])
            if not np.isnan(c):
                cs.append(c)
        if len(cs) >= 8:
            F['corr_ew_60'][i, j] = np.mean(cs)

# --- dxy_beta_cond_60x20: -beta(asset,DXY,60)*(DXY20d move) ---
dxy = load_close('../persistent/index_data/DXY.csv')
dxy_dates = sorted(dxy.keys())
dxy_idx = {d: i for i, d in enumerate(dxy_dates)}
dxy_arr = np.array([dxy[d] for d in dxy_dates])
dxy_R = np.diff(dxy_arr) / dxy_arr[:-1]
# map to our dates
for i in range(59, T):
    d = D[i]
    # DXY move over 20 trading days ending at t (approx using last 20 dxy obs <= d)
    obs = [x for x in dxy_dates if x <= d]
    if len(obs) < 21:
        continue
    dxy20 = dxy[obs[-1]] / dxy[obs[-21]] - 1
    for j in range(len(ASSETS)):
        # beta over 60d: use our returns and dxy returns aligned by date
        # approximate: use the same window over our D and dxy mapped series
        xs = R[i - 59: i + 1, j]
        # dxy returns over matching dates
        ds = []
        for k in range(i - 59, i + 1):
            dd = D[k]
            # find dxy return for date dd (need previous dxy date)
            oi = dxy_idx.get(dd)
            if oi is None or oi == 0:
                ds.append(np.nan)
            else:
                ds.append(dxy_R[oi - 1])
        ds = np.array(ds)
        b = beta_win(xs, ds)
        if not np.isnan(b):
            F['dxy_beta_cond_60x20'][i, j] = -b * dxy20

# --- Forward 10d returns ---
fwd = np.full((T, len(ASSETS)), np.nan)
for i in range(T - 10):
    fwd[i, :] = P[i + 10, :] / P[i, :] - 1

# --- Cross-sectional Spearman IC per date ---
from scipy.stats import spearmanr

def ic_series(fid):
    ics = []
    for i in range(0, T - 10):
        f = F[fid][i, :]
        r = fwd[i, :]
        mask = ~(np.isnan(f) | np.isnan(r))
        if mask.sum() < 8:
            continue
        ic, _ = spearmanr(f[mask], r[mask])
        if not np.isnan(ic):
            ics.append((D[i], ic))
    return ics

print('\n=== Recent IC (Spearman, fwd 10d) by factor ===')
print('%-24s %8s %8s %8s %8s %8s %8s' % ('factor','IC_20d','IC_60d','IC_120d','IC_250d','n','hit60'))
for fid in F:
    ics = ic_series(fid)
    if not ics:
        print(fid, 'no IC data')
        continue
    ics = np.array([x[1] for x in ics])
    def mean_last(n):
        return np.mean(ics[-n:]) if len(ics) >= n else np.nan
    hit60 = np.mean(ics[-60:] > 0) if len(ics) >= 60 else np.nan
    print('%-24s %8.4f %8.4f %8.4f %8.4f %8d %8.2f' % (fid, mean_last(20), mean_last(60), mean_last(120), mean_last(250), len(ics), hit60))

# --- Current factor ranks (last date) ---
print('\n=== Current factor values (2029-01-29) ===')
print('%-10s ' % 'asset', end='')
for fid in F:
    print('%-10s' % fid[:10], end='')
print()
last_i = T - 1
for j, a in enumerate(ASSETS):
    print('%-10s ' % a, end='')
    for fid in F:
        v = F[fid][last_i, j]
        print('%10.4f' % v if not np.isnan(v) else '%10s' % 'nan', end=' ')
    print()
