"""miner_3 2026-07-16: screen structurally-novel factor families vs library.

Goal: find candidates passing |IC1|>=0.007, |ICIR1|>=0.084 with max abs
Spearman rho < 0.5 vs the existing factor library, on the 15-name
cross-asset panel (2020-01-01..2026-07-15, cut 2026-07-15).
"""
import numpy as np
import pandas as pd
import json, os, base64, gzip, glob, pickle

panel = pd.read_pickle('scripts/panel_cache.pkl')
close, open_, high, low, vol, ret = (panel[k] for k in
                                     ['close', 'open', 'high', 'low', 'vol', 'ret'])
macro = panel['macro'].reindex(close.index, method='ffill')
idx = close.index
SYMBOLS = list(close.columns)
N = len(idx)

def fdf(series_dict):
    return pd.DataFrame(series_dict, index=idx, columns=SYMBOLS)

F = {}
# ---------------- A. Overnight gap vs intraday split (open-based, absent from library) ----
gap = open_ / close.shift(1) - 1.0
intra = close / open_ - 1.0
F['gap_rev_1d'] = -gap                                   # gap reversal
F['intra_rev_1d'] = -intra                               # intraday reversal
F['gap_intra_corr_20'] = gap.rolling(20).corr(intra)     # return-composition persistence
F['overnight_share_20'] = gap.abs().rolling(20).mean() / (gap.abs() + intra.abs()).rolling(20).mean()
F['gap_streak_20'] = (gap > 0).rolling(20).sum() - (gap < 0).rolling(20).sum()
F['gap_mom_5d'] = gap.rolling(5).mean()                  # persistent gap direction

# ---------------- B. serial dependence / autocorrelation --------------------------------
def roll_ac(y, w=20):
    out = {}
    for s in y.columns:
        r = y[s]
        m = r.rolling(w).mean()
        num = ((r - m) * (r.shift(1) - m.shift(1))).rolling(w).mean()
        den = ((r - m) ** 2).rolling(w).mean()
        out[s] = num / den
    return fdf(out)
F['ac1_20'] = roll_ac(ret, 20)
F['sign_persist_20'] = (np.sign(ret) * np.sign(ret.shift(1))).rolling(20).mean()

# ---------------- C. candle-shape / range structure ------------------------------------
rng = (high - low).replace(0, np.nan)
F['upper_shadow_20'] = ((high - np.maximum(open_, close)) / rng).rolling(20).mean()
F['lower_shadow_20'] = ((np.minimum(open_, close) - low) / rng).rolling(20).mean()
F['body_ratio_20'] = ((close - open_).abs() / rng).rolling(20).mean()
F['range_std_20'] = (rng / close).rolling(20).std() / (rng / close).rolling(20).mean()

# ---------------- D. volume-price co-movement / liquidity -------------------------------
F['vp_corr_20'] = ret.rolling(20).corr(vol.pct_change())
illq = (ret.abs() / vol.replace(0, np.nan))
F['amihud_20'] = -illq.rolling(20).mean()               # high illiquidity -> reversal (neg)
F['vol_trend_5_60'] = vol.rolling(5).mean() / vol.rolling(60).mean() - 1.0

# ---------------- E. return distribution shape -----------------------------------------
F['skew_20'] = ret.rolling(20).skew()
F['kurt_20'] = ret.rolling(20).kurt()
F['down_vol_ratio_20'] = np.sqrt((ret.where(ret < 0, 0) ** 2).rolling(20).mean()) / ret.rolling(20).std()
F['max_dd_20'] = (close / close.rolling(20).max() - 1.0)

# ---------------- F. cross-asset beta shift / regime -----------------------------------
eq = SYMBOLS[:8]
basket_all = ret.mean(axis=1)
def roll_beta(y, x, w=60):
    out = {}
    for s in y.columns:
        df = pd.concat([y[s].rename('y'), x.rename('x')], axis=1)
        out[s] = df['y'].rolling(w).cov(df['x']) / df['x'].rolling(w).var()
    return fdf(out)
F['beta_all_60'] = roll_beta(ret, basket_all, 60)
spx = ret['SPX']
F['corr_spx_shift_20'] = ret.rolling(20).corr(spx) - ret.rolling(60).corr(spx)
m_ret = macro.pct_change()
F['beta_vix_60'] = roll_beta(ret, m_ret['VIX'], 60)
F['beta_dxy_60'] = roll_beta(ret, m_ret['DXY'], 60)
# conditional: asset sensitivity to VIX only when VIX rising
vix_up = m_ret['VIX'].where(m_ret['VIX'] > 0)
F['beta_vix_up_60'] = roll_beta(ret, vix_up, 60)
us10y, cn10y = close['US10Y'], close['CN10Y']
spread_chg = (us10y - cn10y).diff()
F['beta_yspread_60'] = roll_beta(ret, spread_chg, 60)

# ---------------- G. conditional / interaction ------------------------------------------
std20 = ret.rolling(20).std()
F['gap_rev_x_vol'] = -gap / std20
F['intra_rev_x_vol'] = -intra / std20
F['rev1_x_gapdir'] = -ret * np.sign(gap.fillna(0))      # reversal stronger after same-direction gap
F['range_loc_20'] = ((close - low) / rng).rolling(20).mean()

# ---------------- metrics ----------------------------------------------------------------
fwd = {}
for h in (1, 2, 3, 5, 10):
    fwd[h] = close.shift(-h) / close - 1.0

def fast_ic(fdf_, fwd_, min_names=8):
    ics, obs = [], []
    common_dates = fdf_.index.intersection(fwd_.index)
    fv = fdf_.loc[common_dates]
    fr = fwd_.loc[common_dates]
    for dt in common_dates:
        f = fv.loc[dt].dropna()
        r = fr.loc[dt].reindex(f.index).dropna()
        c = f.index.intersection(r.index)
        if len(c) < min_names:
            continue
        x = f[c].astype(float).rank(); y = r[c].astype(float).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic); obs.append(len(c))
    ics = np.array(ics)
    if len(ics) == 0:
        return {'n_dates': 0, 'ic': np.nan, 'icir': np.nan, 'hit': np.nan}
    return {'n_dates': int(len(ics)), 'ic': float(ics.mean()),
            'icir': float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan,
            'hit': float((ics > 0).mean())}

results = {}
for nm, p in F.items():
    ic1 = fast_ic(p, fwd[1])
    decay = {h: fast_ic(p, fwd[h])['ic'] for h in (1, 2, 3, 5, 10)}
    # coverage: fraction of symbol-days with valid values
    cov = float(p.notna().sum().sum() / (N * len(SYMBOLS)))
    # turnover: mean daily cross-sectional rank displacement / (n_names-1)
    rk = p.rank(axis=1)
    to = float(rk.diff().abs().mean().mean() / (len(SYMBOLS) - 1)) if len(SYMBOLS) > 1 else np.nan
    yr = {}
    for y in range(2020, 2027):
        m = (idx >= pd.Timestamp(f'{y}-01-01')) & (idx <= pd.Timestamp(f'{y}-12-31'))
        r = fast_ic(p.reindex(idx[m]), fwd[1].reindex(idx[m]))
        yr[y] = round(r['ic'], 4) if np.isfinite(r['ic']) else None
    results[nm] = {'ic1': ic1['ic'], 'icir1': ic1['icir'], 'hit1': ic1['hit'],
                   'n_dates': ic1['n_dates'], 'decay': decay, 'coverage': cov,
                   'turnover': to, 'by_year': yr}

# ---------------- print ---------------------------------------------------------------
print(f'panel: {N} dates x {len(SYMBOLS)} symbols  {idx[0].date()}..{idx[-1].date()}')
print(f'{"name":22s} {"ic1":>7s} {"icir1":>7s} {"hit":>5s} {"cov":>5s} {"to":>5s} | decay1/2/3/5/10 | by_year')
for nm, r in sorted(results.items(), key=lambda kv: -abs(kv[1]['icir1'])):
    d = r['decay']
    print(f"{nm:22s} {r['ic1']:7.4f} {r['icir1']:7.3f} {r['hit1']:5.2f} {r['coverage']:5.2f} {r['turnover']:5.2f} | "
          f"{d[1]:+.3f}/{d[2]:+.3f}/{d[3]:+.3f}/{d[5]:+.3f}/{d[10]:+.3f} | {r['by_year']}")

with open('scripts/_miner3_novel_v2.pkl', 'wb') as fh:
    pickle.dump({'panels': {nm: p for nm, p in F.items()}, 'results': results}, fh)
print('\nsaved scripts/_miner3_novel_v2.pkl')
