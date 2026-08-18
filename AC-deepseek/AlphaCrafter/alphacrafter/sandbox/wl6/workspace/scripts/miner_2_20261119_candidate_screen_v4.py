"""
miner_2 screen 2026-11-19 (v4, per-asset calendar): re-validate 4 live ensemble factors
+ screen 6 new candidates. Factors computed on each asset's OWN trading calendar
(matching strategy.py compute_raw_factors), then reindexed to the union grid for
cross-sectional rank IC. Data window: 2020-01-01 .. 2026-11-04 (sim visible cutoff).
Gate (shared): |IC| >= 0.007, |ICIR| >= 0.084 at h=10.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-11-04')
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


# ---------- factor panels built per-asset on own calendar ----------
def panel_from_func(func):
    """func(close: Series) -> Series of factor values on that asset's own dates."""
    out = {}
    for a in ASSETS:
        c = px[a].dropna()
        if len(c) < 140:
            out[a] = pd.Series(index=c.index, dtype=float)
        else:
            out[a] = func(c)
    return pd.DataFrame(out).reindex(px.index).sort_index()


def f_mom10(c):
    return c.shift(5) / c.shift(15) - 1.0


def f_mom120(c):
    return c.shift(5) / c.shift(125) - 1.0


def f_volofvol(c):
    return c.pct_change().rolling(20).std(ddof=0).rolling(60).std(ddof=0)


def f_vixbeta(c):
    r = c.pct_change()
    z = pd.concat([r.rename('a'), vix_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    vmove = vix / vix.shift(20) - 1.0
    return -beta.reindex(z.index) * vmove.reindex(z.index)


def f_usdbeta(c):
    r = c.pct_change()
    z = pd.concat([r.rename('a'), dxy_ret.rename('v')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['v']) / z['v'].rolling(60, min_periods=30).var()
    return beta.reindex(z.index)


def f_rsi14_rev(c):
    r = c.pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + up / dn)
    return 50.0 - rsi


def f_dd60(c):
    return 1.0 - c / c.rolling(60).max()


def f_volterm(c):
    r = c.pct_change()
    v5 = r.rolling(5).std(ddof=0)
    v60 = r.rolling(60).std(ddof=0)
    return (v5 - v60) / v60


def f_riskmom(c):
    r = c.pct_change()
    v60 = r.rolling(60).std(ddof=0)
    return (c.shift(5) / c.shift(25) - 1.0) / v60


def f_hlpos(c):
    hi = c.rolling(20).max()
    lo = c.rolling(20).min()
    return (c - lo) / (hi - lo)


panels = {
    'mom_10d_skip5': panel_from_func(f_mom10),
    'mom_120d_skip5': panel_from_func(f_mom120),
    'vol_of_vol20x60': panel_from_func(f_volofvol),
    'vix_beta_cond_60x20': panel_from_func(f_vixbeta),
    'rsi14_rev': panel_from_func(f_rsi14_rev),
    'dd_60d': panel_from_func(f_dd60),
    'vol_term_5x60': panel_from_func(f_volterm),
    'risk_mom_20x60': panel_from_func(f_riskmom),
    'usd_beta_60x20': panel_from_func(f_usdbeta),
    'hl_pos_20d': panel_from_func(f_hlpos),
}
for fid, p in panels.items():
    print(f'  panel {fid:20s} shape={p.shape} cov_d8={float((p.notna().sum(axis=1) >= MIN_VALID).mean()):.3f}')


# ---------- forward returns per-asset on own calendar, reindexed to union ----------
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
            xm = xm - xm.mean()
            ym = ym - ym.mean()
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
    ic_recent = ic10.loc[ic10.index >= '2025-06-01'].values
    ic_2024 = ic10.loc[(ic10.index >= '2024-01-01') & (ic10.index < '2025-06-01')].values
    st_full = summarize(ic10.values)
    st_recent = summarize(ic_recent)
    st_2024 = summarize(ic_2024)
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
           'n_2024': st_2024['n'] if st_2024 else 0,
           'ic10_recent': round(st_recent['ic'], 4) if st_recent else None,
           'icir10_recent': round(st_recent['icir'], 4) if st_recent else None,
           'n_recent': st_recent['n'] if st_recent else 0,
           'cov_dates_ge8': round(cov_d8, 3),
           'turnover_10d': round(to, 3),
           'decay_ic': decay}
    rows.append(row)
    print(f"{fid:22s} full ic={row['ic10_full']} icir={row['icir10_full']} hit={row['hit_full']} n={row['n_full']} | "
          f"2024 ic={row['ic10_2024']} icir={row['icir10_2024']} n={row['n_2024']} | "
          f"recent ic={row['ic10_recent']} icir={row['icir10_recent']} n={row['n_recent']} | "
          f"cov_d8={row['cov_dates_ge8']} to={row['turnover_10d']} | decay={row['decay_ic']}")

json.dump(rows, open('scripts/miner_2_20261119_screen_results_v4.json', 'w'), indent=1)
print('\nsaved scripts/miner_2_20261119_screen_results_v4.json')
