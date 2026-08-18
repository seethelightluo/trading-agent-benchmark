"""
miner_1 deep validation 2027-01-14 (visible cutoff 2027-01-13).
Focus A: deep-validate skew_60d (lottery/MAX effect): high positive skew assets
underperform over 10d. Check sub-period stability, per-asset contribution,
decay, turnover, and correlation vs each library factor.
Focus B: new regime-aware candidates not yet screened:
  - dxy_beta_60d_neg   : -60d beta to DXY (defensive vs USD strength)
  - usdcny_beta_60d_neg: -60d beta to USDCNY (defensive vs CNY weakness)
  - xau_beta_60d_neg   : -60d beta to XAU (safe-haven linkage)
  - vix_scaled_vol_20d : asset 20d vol / VIX (relative overextension)
  - ew_gate_mom20      : 20d momentum * sign(EW 60d trend) (trend-follow in up regime)
  - ew_gate_rev20      : -20d momentum * sign(EW 60d trend) (contrarian in up regime, momentum in down)
  - mom_30d_skip10     : intermediate momentum 10..40d
  - skew_40d           : 40d realized skewness (shorter lottery window)
Gate: |IC|>=0.007, |ICIR|>=0.084 at h=10; library pairwise rho < 0.5 preferred.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2027-01-13')
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
usdcny = load_close('USDCNY', Path('../persistent/index_data'))
vix_ret = vix.pct_change()
dxy_ret = dxy.pct_change()
usdcny_ret = usdcny.pct_change()
xau = px['XAU']
print(f'price panel: {px.shape[0]} union dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()}')

ret_union = px.pct_change()
ew_ret = ret_union.mean(axis=1, skipna=True)
ew_trend60 = (ew_ret.rolling(60, min_periods=30).mean() * 100.0)  # EW MA slope proxy
ew_ma60 = ew_ret.rolling(60, min_periods=30).mean()
ew_trend60 = ew_ma60  # level of EW 60d mean return


def panel_from_func(func):
    out = {}
    for a in ASSETS:
        c = px[a].dropna()
        if len(c) < 160:
            out[a] = pd.Series(index=c.index, dtype=float)
        else:
            out[a] = func(c)
    return pd.DataFrame(out).reindex(px.index).sort_index()


def f_skew60(c):
    return c.pct_change().rolling(60, min_periods=30).skew()


def f_skew40(c):
    return c.pct_change().rolling(40, min_periods=20).skew()


def f_dxy_beta_neg(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, dxy_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_usdcny_beta_neg(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, usdcny_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_xau_beta_neg(c):
    r = c.pct_change().rename('a')
    z = pd.concat([r, xau.pct_change().rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return -beta.reindex(z.index)


def f_vix_scaled_vol(c):
    r = c.pct_change()
    vol = r.rolling(20, min_periods=10).std(ddof=0)
    scaled = vol / vix.reindex(vol.index)  # asset vol vs VIX (both daily)
    return scaled


def f_ew_gate_mom20(c):
    mom = c.shift(5) / c.shift(25) - 1.0
    gate = np.sign(ew_ma60.reindex(mom.index))
    return mom * gate


def f_ew_gate_rev20(c):
    mom = c.shift(5) / c.shift(25) - 1.0
    gate = np.sign(ew_ma60.reindex(mom.index))
    return -mom * gate


def f_mom30_skip10(c):
    return c.shift(10) / c.shift(40) - 1.0


candidates = {
    'skew_60d': f_skew60, 'skew_40d': f_skew40,
    'dxy_beta_60d_neg': f_dxy_beta_neg, 'usdcny_beta_60d_neg': f_usdcny_beta_neg,
    'xau_beta_60d_neg': f_xau_beta_neg, 'vix_scaled_vol_20d': f_vix_scaled_vol,
    'ew_gate_mom20': f_ew_gate_mom20, 'ew_gate_rev20': f_ew_gate_rev20,
    'mom_30d_skip10': f_mom30_skip10,
}

# library panels for correlation (recompute faithfully)
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


library = {
    'mom_10d_skip5': f_mom10, 'mom_120d_skip5': f_mom120, 'vol_of_vol20x60': f_volofvol,
    'low_vol_20d': f_lowvol20, 'beta_vix_60d_neg': f_beta_vix_neg, 'beta_cn10y_60d': f_beta_cn10y,
    'down_vol_ratio_20x120': f_dvr, 'vix_beta_cond_60x20': f_vixbeta_cond,
}

panels = {fid: panel_from_func(func) for fid, func in {**candidates, **library}.items()}
for fid, p in panels.items():
    print(f'  panel {fid:22s} shape={p.shape} cov_d8={(p.notna().sum(axis=1) >= MIN_VALID).mean():.3f}')

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


def pairwise_corr(fid_panel, op_panel):
    """mean over dates of |Spearman| between candidate and library factor ranks."""
    rs = []
    both = fid_panel.notna() & op_panel.notna()
    for dt in both.index[both.sum(axis=1) >= MIN_VALID]:
        a = fid_panel.loc[dt]; b = op_panel.loc[dt]
        m = a.notna() & b.notna()
        if m.sum() < MIN_VALID:
            continue
        x = a[m].rank(); y = b[m].rank()
        x = x - x.mean(); y = y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den > 0:
            rs.append(abs(float((x * y).sum() / den)))
    return float(np.mean(rs)) if rs else float('nan')


rows = []
for fid, panel in panels.items():
    ic_all = ic_series_all_horizons(panel)
    ic10 = ic_all[10]
    st_full = summarize(ic10.values)
    st_h1 = summarize(ic10.loc[ic10.index < '2023-07-01'].values)
    st_h2 = summarize(ic10.loc[(ic10.index >= '2023-07-01') & (ic10.index < '2025-06-01')].values)
    st_recent = summarize(ic10.loc[ic10.index >= '2025-06-01'].values)
    cov_d8 = float((panel.notna().sum(axis=1) >= MIN_VALID).mean())
    to = turnover_10d(panel)
    decay = {str(h): round(float(ic_all[h].mean()), 3) for h in HORIZONS}
    row = {'factor': fid,
           'ic10_full': round(st_full['ic'], 4) if st_full else None,
           'icir10_full': round(st_full['icir'], 4) if st_full else None,
           'hit_full': round(st_full['hit'], 3) if st_full else None,
           'n_full': st_full['n'] if st_full else 0,
           'ic10_h1': round(st_h1['ic'], 4) if st_h1 else None,
           'icir10_h1': round(st_h1['icir'], 4) if st_h1 else None,
           'ic10_h2': round(st_h2['ic'], 4) if st_h2 else None,
           'icir10_h2': round(st_h2['icir'], 4) if st_h2 else None,
           'ic10_recent': round(st_recent['ic'], 4) if st_recent else None,
           'icir10_recent': round(st_recent['icir'], 4) if st_recent else None,
           'n_recent': st_recent['n'] if st_recent else 0,
           'cov_dates_ge8': round(cov_d8, 3),
           'turnover_10d': round(to, 3),
           'decay_ic': decay}
    if fid in candidates:
        corrs = {lfid: round(pairwise_corr(panel, lp), 3) for lfid, lp in library.items()}
        row['max_abs_library_correlation'] = round(max(corrs.values()), 3)
        row['lib_corr_detail'] = corrs
    rows.append(row)
    extra = f" | maxlib={row['max_abs_library_correlation']}" if 'max_abs_library_correlation' in row else ''
    print(f"{fid:22s} full ic={row['ic10_full']} icir={row['icir10_full']} hit={row['hit_full']} n={row['n_full']} | "
          f"h1 ic={row['ic10_h1']} icir={row['icir10_h1']} | h2 ic={row['ic10_h2']} icir={row['icir10_h2']} | "
          f"recent ic={row['ic10_recent']} icir={row['icir10_recent']} n={row['n_recent']} | "
          f"cov={row['cov_dates_ge8']} to={row['turnover_10d']} | decay={row['decay_ic']}{extra}")

json.dump(rows, open('scripts/miner_1_20270114_deep_skew_regime_results.json', 'w'), indent=1)
print('\nsaved scripts/miner_1_20270114_deep_skew_regime_results.json')
