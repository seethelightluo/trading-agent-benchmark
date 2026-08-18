"""
miner_2 screen 2026-12-17: explore regime-robust cross-sectional factors.
Motivation: last live block (20261203-20261217) lost -4.61% in a commodity/crypto
selloff; beta_vix_60d_neg overweighted negative-VIX-beta assets (BTC/WTI/COPPER)
which hurt. Screen candidates that differentiate WITHIN the 15-asset cross-section:
  - rs_20d / rs_60d        : relative strength vs equal-weight cross-section
  - ew_beta_60d_neg        : negative beta to EW portfolio (risk-off robustness)
  - skew_20d_neg           : negative realized skewness (crash-risk premium)
  - eff_ratio_20d          : Kaufman efficiency ratio (trend quality)
  - usd_beta_cond          : DXY-conditional USD sensitivity
  - vix_stress_rev         : short-term reversal scaled by VIX stress
Also re-validate existing effective factors on the new cutoff (drift check).
Gate: |IC| >= 0.007, |ICIR| >= 0.084 at h=10 (shared benchmark admission).
Data: 2020-01-01 .. 2026-12-16 (visible cutoff).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-12-16')
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
dxy = load_close('DXY', Path('../persistent/index_data'))
vix_ret = vix.pct_change()
dxy_ret = dxy.pct_change()
print(f'price panel: {px.shape[0]} union dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()}')

# equal-weight portfolio return on union grid (per-asset ret reindexed to union)
ret_union = px.pct_change()
ew_ret = ret_union.mean(axis=1, skipna=True)
print(f'EW portfolio ret series: n={ew_ret.notna().sum()}')


def panel_from_func(func):
    out = {}
    for a in ASSETS:
        c = px[a].dropna()
        if len(c) < 160:
            out[a] = pd.Series(index=c.index, dtype=float)
        else:
            out[a] = func(c)
    return pd.DataFrame(out).reindex(px.index).sort_index()


# ---------------- candidate factor definitions (per-asset own calendar) ----------------
def f_rs20(c):
    r = c.pct_change().rolling(20).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    return r


def f_rs60(c):
    r = c.pct_change().rolling(60).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    return r


def f_ew_beta_neg(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, ew_ret.rename('m')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['m']) / z['m'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_skew_neg(c):
    r = c.pct_change()
    sk = r.rolling(20, min_periods=10).skew()
    return -sk


def f_eff_ratio(c):
    r = c.pct_change().abs()
    return (c - c.shift(20)).abs() / r.rolling(20).sum()


def f_usd_beta_cond(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, dxy_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    dmom = dxy / dxy.shift(20) - 1.0
    return (-beta.reindex(z.index) * dmom.reindex(z.index)).reindex(z.index)


def f_vix_stress_rev(c):
    r = c.pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + up / dn)
    rev = 50.0 - rsi
    stress = (vix / vix.rolling(120).mean() - 1.0).reindex(rev.index)
    return rev * stress


# ---------------- existing effective factors (for re-validation & library corr) ----------------
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
    'rs_20d': f_rs20, 'rs_60d': f_rs60, 'ew_beta_60d_neg': f_ew_beta_neg,
    'skew_20d_neg': f_skew_neg, 'eff_ratio_20d': f_eff_ratio,
    'usd_beta_cond': f_usd_beta_cond, 'vix_stress_rev': f_vix_stress_rev,
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

json.dump(rows, open('scripts/miner_2_20261217_screen_results.json', 'w'), indent=1)
print('\nsaved scripts/miner_2_20261217_screen_results.json')
