"""Screener IC check - compute recent cross-sectional IC for candidate factors (data <= 2027-06-16).

Replicates strategy.py factor logic on local CSV data, computes IC vs forward 10d returns.
"""
import csv, math
import numpy as np
import pandas as pd

CUR = '2027-06-16'
ASSETS = ['SPX','NDX','SOX','HSI','N225','SX5E','000300.SH','000688.SH',
          'BTC','ETH','XAU','COPPER','WTI','US10Y','CN10Y']

def load_close(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['date'] <= CUR:
                try:
                    rows.append((r['date'], float(r['close'])))
                except ValueError:
                    pass
    s = pd.Series({d: c for d, c in rows})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

closes = {}
for a in ASSETS:
    closes[a] = load_close(f'../persistent/stock_data/{a}.csv')
dxy_c = load_close('../persistent/index_data/DXY.csv')
vix_c = load_close('../persistent/index_data/VIX.csv')

panel = pd.DataFrame(closes).sort_index()
rets = panel.pct_change()
dxy_r = dxy_c.pct_change()
vix_r = vix_c.pct_change()

def trend_r2(c):
    y = np.log(c.values.astype(float))
    x = np.arange(len(y))
    cov = np.cov(y, x)[0, 1]
    vy, vx = np.var(y), np.var(x)
    if vy <= 0 or vx <= 0:
        return np.nan
    return np.copysign(cov * cov / (vy * vx), cov)

def semi_down_ratio(r):
    down = float((r.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((r.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return np.nan
    return down / up - 1.0

def mom_120(c):
    p0 = float(c.iloc[-126])
    if p0 <= 0:
        return np.nan
    return float(c.iloc[-6]) / p0 - 1.0

def mom_10(c):
    p0 = float(c.iloc[-16])
    if p0 <= 0:
        return np.nan
    return float(c.iloc[-6]) / p0 - 1.0

def underwater(c):
    w = c.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))

def vol_of_vol(r):
    v = r.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return out if np.isfinite(out) else np.nan

def kurt_20(r):
    k = r.rolling(20, min_periods=8).kurt().iloc[-1]
    return k if np.isfinite(k) else np.nan

def tail_ratio(r):
    q95 = np.percentile(r.values, 95)
    q05 = np.percentile(r.values, 5)
    if abs(q05) < 1e-12:
        return np.nan
    return q95 / abs(q05)

def dxy_beta(r, dr):
    z = pd.concat([r.rename('a'), dr.rename('d')], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vd = float(z['d'].var())
    if vd < 1e-14:
        return np.nan
    return float(z['a'].cov(z['d']) / vd)

def vix_beta_cond(r, vr, vc):
    z = pd.concat([r.rename('a'), vr.rename('v')], axis=1).dropna().tail(60)
    if len(z) < 30:
        return np.nan
    vv = float(z['v'].var())
    if vv < 1e-14:
        return np.nan
    beta = float(z['a'].cov(z['v']) / vv)
    if len(vc) < 22:
        return np.nan
    v0 = float(vc.iloc[-21])
    if v0 <= 0:
        return np.nan
    vmove = float(vc.iloc[-1]) / v0 - 1.0
    return -beta * vmove

# Precompute factor time series per asset
FUNCS = {
    'trend_r2_30_signed': ('close', 31),
    'semi_down_ratio_20': ('ret', 21),
    'mom_120d_skip5': ('close', 127),
    'mom_10d_skip5': ('close', 17),
    'vol_of_vol20x60': ('ret', 121),
    'time_under_water_120': ('close', 121),
    'tail_ratio_20': ('ret', 21),
    'kurt_20': ('ret', 41),
    'dxy_beta_60': ('ret', 61),
    'vix_beta_cond_60x20': ('ret', 61),
}

factor_series = {}
for fid, (kind, minlen) in FUNCS.items():
    ser = {}
    for a in ASSETS:
        if kind == 'close':
            c = closes[a]
        else:
            c = rets[a]
        vals = []
        idxs = []
        for i in range(len(c)):
            w = c.iloc[max(0, i - 200):i + 1]
            if len(w) < minlen:
                vals.append(np.nan); idxs.append(c.index[i]); continue
            try:
                if fid == 'trend_r2_30_signed':
                    v = trend_r2(w.tail(30))
                elif fid == 'semi_down_ratio_20':
                    v = semi_down_ratio(w.tail(20))
                elif fid == 'mom_120d_skip5':
                    v = mom_120(w)
                elif fid == 'mom_10d_skip5':
                    v = mom_10(w)
                elif fid == 'vol_of_vol20x60':
                    v = vol_of_vol(w)
                elif fid == 'time_under_water_120':
                    v = underwater(w)
                elif fid == 'tail_ratio_20':
                    v = tail_ratio(w.tail(20))
                elif fid == 'kurt_20':
                    v = kurt_20(w.tail(40))
                elif fid == 'dxy_beta_60':
                    v = dxy_beta(w, dxy_r.reindex(w.index))
                elif fid == 'vix_beta_cond_60x20':
                    v = vix_beta_cond(w, vix_r.reindex(w.index), vix_c.reindex(w.index))
            except Exception:
                v = np.nan
            vals.append(v); idxs.append(c.index[i])
        ser[a] = pd.Series(vals, index=idxs)
    factor_series[fid] = pd.DataFrame(ser)

# Forward 10d returns
fwd = panel.shift(-10) / panel - 1.0

# IC computation
def ic_stats(fvals, fwd10, start):
    dates = fvals.index
    ics = []
    for t in dates:
        fv = fvals.loc[t]
        fr = fwd10.loc[t]
        pair = pd.concat([fv.rename('f'), fr.rename('r')], axis=1).dropna()
        if len(pair) >= 8:
            ic = pair['f'].corr(pair['r'], method='spearman')
            if np.isfinite(ic):
                ics.append((t, ic))
    s = pd.Series({t: ic for t, ic in ics})
    s = s[s.index >= start]
    return s

full_start = pd.Timestamp('2020-01-01')
recent_start = pd.Timestamp('2027-01-01')
last120_start = pd.Timestamp('2026-12-01')

print(f"{'factor':24s} {'dir':>3s} {'ic_all':>8s} {'icir_all':>8s} {'ic_120d':>8s} {'icir_120d':>8s} {'ic_60d':>8s} {'hit120':>6s}")
results = {}
for fid in FUNCS:
    s_all = ic_stats(factor_series[fid], fwd, full_start)
    s120 = s_all[s_all.index >= last120_start]
    s60 = s_all[s_all.index >= pd.Timestamp('2027-03-01')]
    def stats(s):
        if len(s) < 20:
            return (np.nan, np.nan)
        m = s.mean(); sd = s.std(ddof=1)
        return (m, m / sd * math.sqrt(len(s)) if sd > 0 else np.nan)
    m_all, ir_all = stats(s_all)
    m120, ir120 = stats(s120)
    m60, _ = stats(s60)
    hit120 = (s120 > 0).mean() if len(s120) else np.nan
    results[fid] = (m_all, ir_all, m120, ir120, m60, hit120, len(s120))
    print(f"{fid:24s} {'':3s} {m_all:8.4f} {ir_all:8.2f} {m120:8.4f} {ir120:8.2f} {m60:8.4f} {hit120:6.2f}  n120={len(s120)}")

print("\nQuality q=|ic|*|icir| (full sample):")
for fid, (m_all, ir_all, m120, ir120, m60, hit120, n) in results.items():
    q = abs(m_all) * abs(ir_all) if (m_all == m_all and ir_all == ir_all) else 0
    print(f"  {fid:24s} q={q:.5f}  ic120={m120:.4f} ir120={ir120:.2f} ic60={m60:.4f}")
