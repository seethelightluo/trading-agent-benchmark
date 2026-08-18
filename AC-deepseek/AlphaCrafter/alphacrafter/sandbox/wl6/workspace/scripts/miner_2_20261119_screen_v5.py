"""
miner_2 screen 2026-11-19 (v5): updated cutoff 2026-11-18.
Re-validate 3 passers from v4 (risk_mom_20x60, usd_beta_60x20, rsi14_rev) and
screen ~13 NEW candidates aimed at low correlation with the 8-factor library.
Per-asset calendar factor computation, union reindex, rank IC at h=10.
Gate: |IC| >= 0.007, |ICIR| >= 0.084.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2026-11-18')
ASSETS = ['000300.SH', '000688.SH', 'BTC', 'CN10Y', 'COPPER', 'ETH', 'HSI', 'N225',
          'NDX', 'SOX', 'SPX', 'SX5E', 'US10Y', 'WTI', 'XAU']
HORIZONS = [1, 2, 3, 5, 10, 20]
MIN_VALID = 8


def load(sym, base, cols=None):
    df = pd.read_csv(base / f'{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    if cols is None:
        cols = ['open', 'close', 'high', 'low', 'volume']
    return {c: df[c].astype(float) for c in cols if c in df.columns}


stock = {a: load(a, Path('../persistent/stock_data')) for a in ASSETS}
px = pd.DataFrame({a: stock[a]['close'] for a in ASSETS}).sort_index()
vix = load('VIX', Path('../persistent/index_data'), cols=['close'])['close']
dxy = load('DXY', Path('../persistent/index_data'), cols=['close'])['close']
vix_ret = vix.pct_change()
dxy_ret = dxy.pct_change()
print(f'price panel: {px.shape[0]} union dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()}')


def panel_from_func(func):
    out = {}
    for a in ASSETS:
        c = px[a].dropna()
        o = stock[a]['open'].reindex(c.index)
        h = stock[a]['high'].reindex(c.index)
        l = stock[a]['low'].reindex(c.index)
        v = stock[a]['volume'].reindex(c.index)
        if len(c) < 140:
            out[a] = pd.Series(index=c.index, dtype=float)
        else:
            out[a] = func(c, o, h, l, v)
    return pd.DataFrame(out).reindex(px.index).sort_index()


# ---------------- library factors (8) ----------------
def f_beta_vix(c, o, h, l, v):
    r = c.pct_change()
    z = pd.concat([r.rename('a'), vix_ret.rename('x')], axis=1).dropna()
    return -z['a'].rolling(60, min_periods=30).cov(z['x']) / z['x'].rolling(60, min_periods=30).var()


def f_down_vol(c, o, h, l, v):
    r = c.pct_change()
    d = -r.clip(upper=0)
    return -(d.rolling(20).std(ddof=0) / d.rolling(120, min_periods=60).std(ddof=0))


def f_low_vol(c, o, h, l, v):
    return -c.pct_change().rolling(20).std(ddof=0)


def f_mom10(c, o, h, l, v):
    return c.shift(5) / c.shift(15) - 1.0


def f_mom120(c, o, h, l, v):
    return c.shift(5) / c.shift(125) - 1.0


def f_vixbeta_cond(c, o, h, l, v):
    r = c.pct_change()
    z = pd.concat([r.rename('a'), vix_ret.rename('x')], axis=1).dropna()
    beta = z['a'].rolling(60, min_periods=30).cov(z['x']) / z['x'].rolling(60, min_periods=30).var()
    vmove = vix / vix.shift(20) - 1.0
    return -beta.reindex(z.index) * vmove.reindex(z.index)


def f_vol_imb(c, o, h, l, v):
    r = c.pct_change()
    vol = r.abs()
    up = vol.where(r > 0, 0.0).rolling(20).sum()
    dn = vol.where(r <= 0, 0.0).rolling(20).sum()
    return (up - dn) / vol.rolling(20).sum()


def f_vol_of_vol(c, o, h, l, v):
    return c.pct_change().rolling(20).std(ddof=0).rolling(60).std(ddof=0)


LIBRARY = {
    'beta_vix_60d_neg': f_beta_vix,
    'down_vol_ratio_20x120': f_down_vol,
    'low_vol_20d': f_low_vol,
    'mom_10d_skip5': f_mom10,
    'mom_120d_skip5': f_mom120,
    'vix_beta_cond_60x20': f_vixbeta_cond,
    'vol_imb_20d': f_vol_imb,
    'vol_of_vol20x60': f_vol_of_vol,
}
lib_panels = {fid: panel_from_func(fn) for fid, fn in LIBRARY.items()}

# ---------------- candidates ----------------
basket_ret = px.pct_change().mean(axis=1)


def f_risk_mom_20x60(c, o, h, l, v):
    r = c.pct_change()
    v60 = r.rolling(60, min_periods=30).std(ddof=0)
    return (c.shift(5) / c.shift(25) - 1.0) / v60


def f_risk_mom_60x60(c, o, h, l, v):
    r = c.pct_change()
    v60 = r.rolling(60, min_periods=30).std(ddof=0)
    return (c.shift(5) / c.shift(65) - 1.0) / v60


def f_usd_beta(c, o, h, l, v, win=60):
    r = c.pct_change()
    z = pd.concat([r.rename('a'), dxy_ret.rename('x')], axis=1).dropna()
    return z['a'].rolling(win, min_periods=30).cov(z['x']) / z['x'].rolling(win, min_periods=30).var()


def f_rsi14_rev(c, o, h, l, v):
    r = c.pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + up / dn)
    return 50.0 - rsi


def f_skew_60(c, o, h, l, v):
    return c.pct_change().rolling(60, min_periods=30).skew()


def f_kurt_60(c, o, h, l, v):
    return c.pct_change().rolling(60, min_periods=30).kurt()


def f_basket_beta_60(c, o, h, l, v):
    r = c.pct_change()
    z = pd.concat([r.rename('a'), basket_ret.rename('x')], axis=1).dropna()
    return z['a'].rolling(60, min_periods=30).cov(z['x']) / z['x'].rolling(60, min_periods=30).var()


def f_vol_ratio_20x120(c, o, h, l, v):
    r = c.pct_change()
    v20 = r.rolling(20).std(ddof=0)
    v120 = r.rolling(120, min_periods=60).std(ddof=0)
    return v20 / v120 - 1.0


def f_vol_trend(c, o, h, l, v):
    vm = v.rolling(20).mean() / v.rolling(60, min_periods=30).mean() - 1.0
    return vm


def f_ret_ac_20(c, o, h, l, v):
    r = c.pct_change()
    return r.rolling(20).corr(r.shift(1))


def f_mom_accel(c, o, h, l, v):
    m20 = c.shift(5) / c.shift(25) - 1.0
    m60 = c.shift(5) / c.shift(65) - 1.0
    return m20 - m60


def f_high52w(c, o, h, l, v):
    return c / c.rolling(250, min_periods=120).max() - 1.0


def f_dd_cummax(c, o, h, l, v):
    return 1.0 - c / c.cummax()


def f_body_pos_20(c, o, h, l, v):
    body = (c - o) / (h - l)
    return body.rolling(20, min_periods=10).mean()


def f_ema_cross(c, o, h, l, v):
    e20 = c.ewm(span=20, adjust=False).mean()
    e60 = c.ewm(span=60, adjust=False).mean()
    return e20 / e60 - 1.0


CAND = {
    'risk_mom_20x60': f_risk_mom_20x60,
    'risk_mom_60x60': f_risk_mom_60x60,
    'usd_beta_60x20': lambda c, o, h, l, v: f_usd_beta(c, o, h, l, v, 60),
    'usd_beta_120': lambda c, o, h, l, v: f_usd_beta(c, o, h, l, v, 120),
    'rsi14_rev': f_rsi14_rev,
    'skew_60d': f_skew_60,
    'kurt_60d': f_kurt_60,
    'basket_beta_60': f_basket_beta_60,
    'vol_ratio_20x120': f_vol_ratio_20x120,
    'vol_trend_20x60': f_vol_trend,
    'ret_ac_20': f_ret_ac_20,
    'mom_accel_20x60': f_mom_accel,
    'high52w': f_high52w,
    'dd_cummax': f_dd_cummax,
    'body_pos_20': f_body_pos_20,
    'ema_cross_20x60': f_ema_cross,
}
cand_panels = {fid: panel_from_func(fn) for fid, fn in CAND.items()}

# ---------------- forward returns ----------------
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
    return {'n': int(len(ic)), 'ic': float(m), 'icir': float(m / s) if s > 0 else 0.0,
            'hit': float((np.sign(ic) == np.sign(m)).mean())}


def turnover_10d(fac):
    r = fac.rank(axis=1, pct=True)
    d = r.diff(10).abs().mean().mean()
    return float(d) if np.isfinite(d) else float('nan')


def max_abs_lib_corr(fac, libs):
    best = 0.0
    for fid, lp in libs.items():
        df = pd.concat([fac.stack().rename('a'), lp.stack().rename('b')], axis=1).dropna()
        if len(df) < 200:
            continue
        rho = df['a'].corr(df['b'], method='spearman')
        best = max(best, abs(rho))
    return best


rows = []
for fid, panel in {**CAND}.items():
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
    rho = max_abs_lib_corr(panel, lib_panels)
    row = {'factor': fid,
           'ic10_full': round(st_full['ic'], 4) if st_full else None,
           'icir10_full': round(st_full['icir'], 4) if st_full else None,
           'hit_full': round(st_full['hit'], 3) if st_full else None,
           'n_full': st_full['n'] if st_full else 0,
           'ic10_2024': round(st_2024['ic'], 4) if st_2024 else None,
           'icir10_2024': round(st_2024['icir'], 4) if st_2024 else None,
           'ic10_recent': round(st_recent['ic'], 4) if st_recent else None,
           'icir10_recent': round(st_recent['icir'], 4) if st_recent else None,
           'cov_dates_ge8': round(cov_d8, 3),
           'turnover_10d': round(to, 3),
           'max_abs_library_correlation': round(rho, 3),
           'decay_ic': decay}
    rows.append(row)
    print(f"{fid:22s} full ic={row['ic10_full']} icir={row['icir10_full']} hit={row['hit_full']} n={row['n_full']} | "
          f"2024 ic={row['ic10_2024']} icir={row['icir10_2024']} | "
          f"recent ic={row['ic10_recent']} icir={row['icir10_recent']} | "
          f"cov_d8={row['cov_dates_ge8']} to={row['turnover_10d']} rho_lib={row['max_abs_library_correlation']} | "
          f"decay={row['decay_ic']}")

json.dump(rows, open('scripts/miner_2_20261119_screen_results_v5.json', 'w'), indent=1)
print('\nsaved scripts/miner_2_20261119_screen_results_v5.json')
