"""
miner_2 screen 2026-12-17 (round 2): regime-gated / risk-off robust candidates.
Motivation: last block was a bull-call-then-riskoff regime mismatch; commodity/crypto
selloff hurt negative-VIX-beta longs. Test:
  - reg_mom10        : 10d momentum gated ON by positive 20d EW drift (trend-confirmed)
  - reg_rev10        : 10d reversal gated ON by negative 20d EW drift (fade downtrends)
  - down_beta_ew_neg : downside (EW-down days only) beta, negated (defensive)
  - dist_1y_high     : closeness to 1y high (trend quality)
  - amihud_20d_neg   : -mean(|ret|/volume,20) (liquidity premium)
  - vol_scaled_rev5  : -5d return / 20d vol (risk-adjusted short reversal)
Gate: |IC| >= 0.007, |ICIR| >= 0.084 at h=10. Cutoff 2026-12-16.
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


def load_vol(sym, base):
    df = pd.read_csv(base / f'{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    return df['volume'].astype(float) if 'volume' in df else None


px = pd.DataFrame({a: load_close(a, Path('../persistent/stock_data')) for a in ASSETS}).sort_index()
vol = {a: load_vol(a, Path('../persistent/stock_data')) for a in ASSETS}
vix = load_close('VIX', Path('../persistent/index_data'))
ret_union = px.pct_change()
ew_ret = ret_union.mean(axis=1, skipna=True)
ew_drift20 = ew_ret.rolling(20).mean()
print(f'price panel: {px.shape[0]} union dates, {px.index[0].date()}..{px.index[-1].date()}')


def panel_from_func(func):
    out = {}
    for a in ASSETS:
        c = px[a].dropna()
        if len(c) < 300:
            out[a] = pd.Series(index=c.index, dtype=float)
        else:
            out[a] = func(c, a)
    return pd.DataFrame(out).reindex(px.index).sort_index()


def f_reg_mom10(c, a):
    m = c.shift(5) / c.shift(15) - 1.0
    gate = (ew_drift20.reindex(c.index) > 0).astype(float)
    return m * gate


def f_reg_rev10(c, a):
    m = c.shift(5) / c.shift(15) - 1.0
    gate = (ew_drift20.reindex(c.index) < 0).astype(float)
    return -m * gate


def f_down_beta_ew(c, a):
    r = c.pct_change().rename('a')
    z = pd.concat([r, ew_ret.rename('m')], axis=1).dropna()
    mneg = z['m'] < 0
    b = z['a'].where(mneg).rolling(60, min_periods=30).cov(z['m'].where(mneg)) / \
        z['m'].where(mneg).rolling(60, min_periods=30).var()
    return -b.reindex(z.index)


def f_dist_1y_high(c, a):
    return c / c.rolling(250).max()


def f_amihud_neg(c, a):
    v = vol[a].reindex(c.index)
    r = c.pct_change().abs()
    ill = (r / v).rolling(20, min_periods=10).mean()
    return -ill


def f_vol_scaled_rev5(c, a):
    r = c.pct_change()
    v20 = r.rolling(20, min_periods=10).std(ddof=0)
    m5 = c.shift(1) / c.shift(6) - 1.0
    return -m5 / v20


candidates = {
    'reg_mom10': f_reg_mom10,
    'reg_rev10': f_reg_rev10,
    'down_beta_ew_neg': f_down_beta_ew,
    'dist_1y_high': f_dist_1y_high,
    'amihud_20d_neg': f_amihud_neg,
    'vol_scaled_rev5': f_vol_scaled_rev5,
}

panels = {fid: panel_from_func(func) for fid, func in candidates.items()}
for fid, p in panels.items():
    print(f'  panel {fid:22s} shape={p.shape} cov_d8={float((p.notna().sum(axis=1) >= MIN_VALID).mean()):.3f}')

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
    rows.append(row)
    print(f"{fid:22s} full ic={row['ic10_full']} icir={row['icir10_full']} hit={row['hit_full']} n={row['n_full']} | "
          f"2024 ic={row['ic10_2024']} icir={row['icir10_2024']} | "
          f"recent ic={row['ic10_recent']} icir={row['icir10_recent']} n={row['n_recent']} | "
          f"cov_d8={row['cov_dates_ge8']} to={row['turnover_10d']} | decay={row['decay_ic']}")

json.dump(rows, open('scripts/miner_2_20261217_screen_results_r2.json', 'w'), indent=1)
print('\nsaved scripts/miner_2_20261217_screen_results_r2.json')
