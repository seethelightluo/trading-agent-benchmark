"""
miner_1 screen 2026-12-31 (visible cutoff 2026-12-30).
Motivation: live block 20261203-20261217 lost -4.61% in a commodity/crypto
selloff (BTC -13.9%, WTI -7.5%, SX5E -7.8%); beta_vix_60d_neg overweighted
negative-VIX-beta assets (BTC/WTI/COPPER) which hurt. Feedback: recheck factor
directions. Explore NEW differentiated candidates:
  - mom_20d_skip5        : intermediate momentum (5..25d)
  - vol_term_5x60        : short/long vol term structure (stress gauge)
  - dd_60d               : distance from 60d rolling max (drawdown depth)
  - tail_20d             : 5th percentile daily return over 20d (tail risk)
  - beta_btc_60d_neg     : negative beta to BTC (crypto sensitivity hedge)
  - beta_wti_60d_neg     : negative beta to WTI (commodity sensitivity hedge)
  - corr_ew_60d_neg      : negative correlation to EW portfolio
  - asym_beta_60d        : down-beta - up-beta vs EW (asymmetric sensitivity)
  - rsi14_rev            : RSI-14 reversal (100 - rsi)
  - vol_adj_mom_20d      : 20d mom / 20d vol (risk-adjusted momentum)
  - range_pos_60d        : position within 60d high-low range
  - ma_slope_60d         : 60d MA slope normalized by vol
  - skew_60d             : 60d realized skewness
  - kurt_20d             : 20d realized kurtosis (fat tails)
  - vix_calm_mom20       : 20d momentum scaled by calm-VIX condition
Also re-validate the 8 existing effective factors (drift check).
Gate: |IC| >= 0.007, |ICIR| >= 0.084 at h=10 (shared benchmark admission).
Data: 2020-01-01 .. 2026-12-30.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-12-30')
ASSETS = ['000300.SH', '000688.SH', 'BTC', 'CN10Y', 'COPPER', 'ETH', 'HSI', 'N225',
          'NDX', 'SOX', 'SPX', 'SX5E', 'US10Y', 'WTI', 'XAU']
HORIZONS = [1, 2, 3, 5, 10, 20]
MIN_VALID = 8


def load_close(sym, base):
    df = pd.read_csv(base / f'{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    return df['close'].astype(float)


px = pd.DataFrame({a: load_close(a, Path('../persistent/stock_data')) for a in ASSETS}).sort_index()
vix = load_close('VIX', Path('../persistent/index_data'))
vix_ret = vix.pct_change()
print(f'price panel: {px.shape[0]} union dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()}')

ret_union = px.pct_change()
ew_ret = ret_union.mean(axis=1, skipna=True)
# equal-weight down-day mask (EW return < 0)
ew_down = (ew_ret < 0).astype(float)


def panel_from_func(func):
    out = {}
    for a in ASSETS:
        c = px[a].dropna()
        if len(c) < 160:
            out[a] = pd.Series(index=c.index, dtype=float)
        else:
            out[a] = func(c)
    return pd.DataFrame(out).reindex(px.index).sort_index()


def ew_down_ret_series():
    # down-day EW return, NaN on up days (for conditional beta)
    return ew_ret.where(ew_down)


def ew_up_ret_series():
    return ew_ret.where(ew_ret > 0)


# ---------------- new candidate definitions (per-asset own calendar) ----------------
def f_mom20(c):
    return c.shift(5) / c.shift(25) - 1.0


def f_vol_term(c):
    r = c.pct_change()
    return r.rolling(5, min_periods=4).std(ddof=0) / r.rolling(60, min_periods=30).std(ddof=0)


def f_dd60(c):
    return c / c.rolling(60, min_periods=30).max() - 1.0


def f_tail20(c):
    return c.pct_change().rolling(20, min_periods=10).quantile(0.05)


def f_beta_btc_neg(c):
    btc = px['BTC'].pct_change().rename('v')
    r = c.pct_change().rename('a')
    z = pd.concat([r, btc], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_beta_wti_neg(c):
    wti = px['WTI'].pct_change().rename('v')
    r = c.pct_change().rename('a')
    z = pd.concat([r, wti], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_corr_ew_neg(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, ew_ret.rename('m')], axis=1).dropna()
    corr = z['a'].rolling(60, min_periods=30).corr(z['m'])
    return -corr.reindex(z.index)


def f_asym_beta(c):
    r = c.pct_change().rename('a')
    zd = pd.concat([r, ew_down_ret_series().rename('m')], axis=1).dropna()
    zu = pd.concat([r, ew_up_ret_series().rename('m')], axis=1).dropna()
    bd = zd['a'].rolling(60, min_periods=30).cov(zd['m']) / zd['m'].rolling(60, min_periods=30).var()
    bu = zu['a'].rolling(60, min_periods=30).cov(zu['m']) / zu['m'].rolling(60, min_periods=30).var()
    return (bd - bu).reindex(zd.index)


def f_rsi14_rev(c):
    r = c.pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + up / dn)
    return 100.0 - rsi


def f_vol_adj_mom20(c):
    r = c.pct_change()
    mom = c.shift(5) / c.shift(25) - 1.0
    vol = r.rolling(20, min_periods=10).std(ddof=0)
    return mom / vol


def f_range_pos(c):
    hi = c.rolling(60, min_periods=30).max()
    lo = c.rolling(60, min_periods=30).min()
    return (c - lo) / (hi - lo)


def f_ma_slope(c):
    ma = c.rolling(60, min_periods=30).mean()
    slope = ma - ma.shift(10)
    vol = c.pct_change().rolling(60, min_periods=30).std(ddof=0)
    return slope / (vol * c)


def f_skew60(c):
    return c.pct_change().rolling(60, min_periods=30).skew()


def f_kurt20(c):
    return c.pct_change().rolling(20, min_periods=10).kurt()


def f_vix_calm_mom20(c):
    mom = c.shift(5) / c.shift(25) - 1.0
    calm = (vix / vix.rolling(120, min_periods=60).mean() - 1.0).reindex(mom.index)
    return mom * calm


# ---------------- existing effective factors (re-validation & library corr) ----------------
def f_mom10(c):
    return c.shift(5) / c.shift(15) - 1.0


def f_mom120(c):
    return c.shift(5) / c.shift(125) - 1.0


def f_volofvol(c):
    return c.pct_change().rolling(20).std(ddof=0).rolling(60).std(ddof=0)


def f_lowvol20(c):
    return -c.pct_change().rolling(20, min_periods=10).std(ddof=0)


def f_beta_vix_neg(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, vix_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_beta_cn10y(c):
    cn = px['CN10Y'].pct_change().rename('v')
    r = c.pct_change().rename('a')
    z = pd.concat([r, cn], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return beta.reindex(z.index)


def f_dvr(c):
    r = c.pct_change()
    dn = (-r.clip(upper=0))
    return -dn.rolling(20, min_periods=10).std(ddof=0) / dn.rolling(120, min_periods=60).std(ddof=0)


def f_vixbeta_cond(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, vix_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    vmove = vix / vix.shift(20) - 1.0
    return -beta.reindex(z.index) * vmove.reindex(z.index)


candidates = {
    'mom_20d_skip5': f_mom20, 'vol_term_5x60': f_vol_term, 'dd_60d': f_dd60,
    'tail_20d': f_tail20, 'beta_btc_60d_neg': f_beta_btc_neg, 'beta_wti_60d_neg': f_beta_wti_neg,
    'corr_ew_60d_neg': f_corr_ew_neg, 'asym_beta_60d': f_asym_beta, 'rsi14_rev': f_rsi14_rev,
    'vol_adj_mom_20d': f_vol_adj_mom20, 'range_pos_60d': f_range_pos, 'ma_slope_60d': f_ma_slope,
    'skew_60d': f_skew60, 'kurt_20d': f_kurt20, 'vix_calm_mom20': f_vix_calm_mom20,
}
existing = {
    'mom_10d_skip5': f_mom10, 'mom_120d_skip5': f_mom120, 'vol_of_vol20x60': f_volofvol,
    'low_vol_20d': f_lowvol20, 'beta_vix_60d_neg': f_beta_vix_neg, 'beta_cn10y_60d': f_beta_cn10y,
    'down_vol_ratio_20x120': f_dvr, 'vix_beta_cond_60x20': f_vixbeta_cond,
}

panels = {fid: panel_from_func(func) for fid, func in {**candidates, **existing}.items()}
for fid, p in panels.items():
    print(f'  panel {fid:22s} shape={p.shape} cov_d8={float((p.notna().sum(axis=1) >= MIN_VALID).mean()):.3f}')

# ---------------- forward returns (rank) ----------------
fwd = {}
for h in HORIZONS:
    fr = pd.DataFrame({a: (px[a].shift(-h) / px[a] - 1.0) for a in ASSETS}).reindex(px.index)
    fwd[h] = fr.rank(axis=1).values

idx = px.index


def ic_series_all_horizons(fac_panel):
    fac_r = fac_panel.rank(axis=1).values
    pos = {h: [] for h in HORIZONS}
    val = {h: [] for h in HORIZONS}
    for i in range(len(idx)):
        xi = fac_r[i]
        for h in HORIZONS:
            yi = fwd[h][i]
            m = ~(np.isnan(xi) | np.isnan(yi))
            if m.sum() < MIN_VALID:
                continue
            xm, ym = xi[m], yi[m]
            xm = xm - xm.mean(); ym = ym - ym.mean()
            denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
            if denom > 0:
                pos[h].append(i)
                val[h].append(float((xm * ym).sum() / denom))
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


def max_lib_corr(fid_panel, others):
    """mean over dates of |Spearman| between candidate ranks and each library factor ranks."""
    best = 0.0
    for ofid, op in others.items():
        rs = []
        both = fid_panel.notna() & op.notna()
        for dt in both.index[both.sum(axis=1) >= MIN_VALID]:
            a = fid_panel.loc[dt]; b = op.loc[dt]
            m = a.notna() & b.notna()
            if m.sum() < MIN_VALID:
                continue
            x = a[m].rank(); y = b[m].rank()
            x = x - x.mean(); y = y - y.mean()
            den = np.sqrt((x * x).sum() * (y * y).sum())
            if den > 0:
                rs.append(abs(float((x * y).sum() / den)))
        if rs:
            best = max(best, float(np.mean(rs)))
    return best


rows = []
for fid, panel in panels.items():
    ic_all = ic_series_all_horizons(panel)
    ic10 = ic_all[10]
    st_full = summarize(ic10.values)
    st_2024 = summarize(ic10.loc[(ic10.index >= '2024-01-01') & (ic10.index < '2025-06-01')].values)
    st_recent = summarize(ic10.loc[ic10.index >= '2025-06-01'].values)
    cov_d8 = float((panel.notna().sum(axis=1) >= MIN_VALID).mean())
    to = turnover_10d(panel)
    decay = {str(h): round(float(ic_all[h].mean()), 3) for h in HORIZONS}
    row = {'factor': fid,
           'ic10_full': round(st_full['ic'], 4) if st_full else None,
           'icir10_full': round(st_full['icir'], 4) if st_full else None,
           'hit_full': round(st_full['hit'], 3) if st_full else None,
           'n_full': st_full['n'] if st_full else 0,
           'ic10_2024': round(st_2024['ic'], 4) if st_2024 else None,
           'icir10_2024': round(st_2024['icir'], 4) if st_2024 else None,
           'ic10_recent': round(st_recent['ic'], 4) if st_recent else None,
           'icir10_recent': round(st_recent['icir'], 4) if st_recent else None,
           'n_recent': st_recent['n'] if st_recent else 0,
           'cov_dates_ge8': round(cov_d8, 3),
           'turnover_10d': round(to, 3),
           'decay_ic': decay}
    if fid in candidates:
        row['max_abs_library_correlation'] = round(max_lib_corr(panel, {k: v for k, v in panels.items() if k in existing}), 3)
    rows.append(row)
    print(f"{fid:22s} full ic={row['ic10_full']} icir={row['icir10_full']} hit={row['hit_full']} n={row['n_full']} | "
          f"2024 ic={row['ic10_2024']} icir={row['icir10_2024']} | "
          f"recent ic={row['ic10_recent']} icir={row['icir10_recent']} n={row['n_recent']} | "
          f"cov_d8={row['cov_dates_ge8']} to={row['turnover_10d']} | decay={row['decay_ic']}"
          + (f" | maxlibcorr={row['max_abs_library_correlation']}" if 'max_abs_library_correlation' in row else ''))

json.dump(rows, open('scripts/miner_1_20261231_screen_v3_results.json', 'w'), indent=1)
print('\nsaved scripts/miner_1_20261231_screen_v3_results.json')
