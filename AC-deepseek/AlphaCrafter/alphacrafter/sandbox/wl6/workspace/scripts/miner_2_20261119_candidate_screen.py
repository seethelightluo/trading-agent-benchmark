"""miner_2 screen 2026-11-19 (v4, fixed): re-validate 4 live ensemble factors
+ screen 8 new candidates. Rank IC via numpy row-wise Pearson on row-ranks.
Data window: 2020-01-01 .. 2026-11-04 (sim visible cutoff; NO lookahead).
Gate (shared): |IC| >= 0.007, |ICIR| >= 0.084 at h=10.
Fix: explicit axis=0 alignment in rolling_beta; min_periods variants reported.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-11-04')
ASSETS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225',
          'NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
HORIZONS = [1, 2, 3, 5, 10, 20]
MIN_VALID = 8

def load_close(sym, base):
    df = pd.read_csv(base / f'{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    return df['close'].astype(float)

px = pd.DataFrame({a: load_close(a, Path('../persistent/stock_data')) for a in ASSETS}).sort_index()
vix = load_close('VIX', Path('../persistent/index_data'))
dxy = load_close('DXY', Path('../persistent/index_data'))
ret = px.pct_change()
vix_ret = vix.pct_change()
dxy_ret = dxy.pct_change()
print(f'price panel: {px.shape[0]} dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()}')

def rolling_beta(x_df, y_s, w, mp=None):
    """beta of each asset column vs y_s, rolling w, aligned on index (dates)."""
    mp = mp if mp is not None else w
    my = y_s.rolling(w, min_periods=mp).mean()
    cov = x_df.mul(y_s, axis=0).rolling(w, min_periods=mp).mean() - x_df.rolling(w, min_periods=mp).mean().mul(my, axis=0)
    var = (y_s ** 2).rolling(w, min_periods=mp).mean() - my ** 2
    return cov.div(var, axis=0)

panels = {}
s5, s15, s125 = px.shift(5), px.shift(15), px.shift(125)
panels['mom_10d_skip5'] = s5 / s15 - 1.0
panels['mom_120d_skip5'] = s5 / s125 - 1.0
panels['vol_of_vol20x60'] = ret.rolling(20, min_periods=10).std(ddof=0).rolling(60, min_periods=20).std(ddof=0)
v20 = rolling_beta(ret, vix_ret, 60, mp=30)
vix_move = vix / vix.shift(20) - 1.0
panels['vix_beta_cond_60x20'] = -v20.mul(vix_move, axis=0)
rsi = 100.0 - 100.0 / (1.0 + ret.clip(lower=0).rolling(14, min_periods=7).mean() / (-ret.clip(upper=0).rolling(14, min_periods=7).mean()))
panels['rsi14_rev'] = 50.0 - rsi
panels['dd_60d'] = 1.0 - px / px.rolling(60, min_periods=30).max()
vol5 = ret.rolling(5, min_periods=3).std(ddof=0); vol60 = ret.rolling(60, min_periods=30).std(ddof=0)
panels['vol_term_5x60'] = (vol5 - vol60) / vol60
panels['risk_mom_20x60'] = (px.shift(5) / px.shift(25) - 1.0) / vol60
panels['usd_beta_60x20'] = rolling_beta(ret, dxy_ret, 60, mp=30)
hi20 = px.rolling(20, min_periods=10).max(); lo20 = px.rolling(20, min_periods=10).min()
panels['hl_pos_20d'] = (px - lo20) / (hi20 - lo20)
# new candidates
panels['vol_ratio_5x60'] = vol5 / vol60
d60 = ret.rolling(60, min_periods=30).std(ddof=0)
panels['inv_vol_60d'] = -1.0 / d60
# 52w-high proximity (trend persistence) with 10d skip
panels['hh_skip5_120d'] = px.shift(5) / px.rolling(120, min_periods=60).max() - 1.0
# correlation of asset with VIX level change over 20d (risk-on/off tilt)
cov20 = ret.mul(vix_ret, axis=0).rolling(20, min_periods=10).mean() - ret.rolling(20, min_periods=10).mean().mul(vix_ret.rolling(20, min_periods=10).mean(), axis=0)
v20v = vix_ret.rolling(20, min_periods=10).std(ddof=0)
a20v = ret.rolling(20, min_periods=10).std(ddof=0)
panels['vix_corr_20d_neg'] = -(cov20.div(v20v, axis=0).div(a20v, axis=0))

# ---------- vectorized rank-IC engine ----------
fwd_rank = {}
for h in HORIZONS:
    fr = (px.shift(-h) / px - 1.0)
    fwd_rank[h] = fr.rank(axis=1).values

idx = px.index
def ic_series_all_horizons(fac_panel):
    fac_r = fac_panel.reindex(idx).rank(axis=1).values
    pos = {h: [] for h in HORIZONS}
    val = {h: [] for h in HORIZONS}
    for i in range(len(idx)):
        xi = fac_r[i]
        for h in HORIZONS:
            yi = fwd_rank[h][i]
            m = ~(np.isnan(xi) | np.isnan(yi))
            if m.sum() < MIN_VALID:
                continue
            xm, ym = xi[m], yi[m]
            xm = xm - xm.mean(); ym = ym - ym.mean()
            denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
            if denom > 0:
                pos[h].append(i); val[h].append(float((xm * ym).sum() / denom))
    return {h: pd.Series(val[h], index=idx[pos[h]]) for h in HORIZONS}

def summarize(ic):
    if len(ic) < 30:
        return None
    m, s = ic.mean(), ic.std(ddof=0)
    return {'n': int(len(ic)), 'ic': float(m), 'abs_ic': abs(float(m)),
            'icir': float(m / s) if s > 0 else 0.0,
            'hit': float((np.sign(ic) == np.sign(m)).mean())}

def turnover_10d(fac):
    r = fac.rank(axis=1, pct=True)
    d = r.diff(10).abs().mean().mean()
    return float(d) if np.isfinite(d) else float('nan')

rows = []
for fid, panel in panels.items():
    ic_all = ic_series_all_horizons(panel)
    ic10 = ic_all[10]
    ic_recent = ic10.loc[ic10.index >= '2025-06-01'].values
    st_full = summarize(ic10.values); st_recent = summarize(ic_recent)
    cov_ad = float(panel.notna().mean().mean())
    cov_d8 = float((panel.notna().sum(axis=1) >= MIN_VALID).mean())
    to = turnover_10d(panel)
    decay = {str(h): round(float(ic_all[h].mean()), 3) for h in HORIZONS}
    row = {'factor': fid,
           'ic10_full': round(st_full['ic'], 4) if st_full else None,
           'icir10_full': round(st_full['icir'], 4) if st_full else None,
           'hit_full': round(st_full['hit'], 3) if st_full else None,
           'n_full': st_full['n'] if st_full else 0,
           'ic10_recent': round(st_recent['ic'], 4) if st_recent else None,
           'icir10_recent': round(st_recent['icir'], 4) if st_recent else None,
           'n_recent': st_recent['n'] if st_recent else 0,
           'cov_asset_day': round(cov_ad, 3), 'cov_dates_ge8': round(cov_d8, 3),
           'turnover_10d': round(to, 3), 'decay_ic': decay}
    rows.append(row)
    print(f"{fid:22s} full ic={row['ic10_full']} icir={row['icir10_full']} hit={row['hit_full']} n={row['n_full']} | "
          f"recent ic={row['ic10_recent']} icir={row['icir10_recent']} n={row['n_recent']} | "
          f"cov_ad={row['cov_asset_day']} cov_d8={row['cov_dates_ge8']} to={row['turnover_10d']} | "
          f"decay={row['decay_ic']}")

json.dump(rows, open('scripts/miner_2_20261119_screen_results.json', 'w'), indent=1)
print('\nsaved scripts/miner_2_20261119_screen_results.json')
