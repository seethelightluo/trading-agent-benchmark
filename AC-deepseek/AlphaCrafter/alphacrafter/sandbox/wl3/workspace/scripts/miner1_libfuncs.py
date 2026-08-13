"""miner_1 shared library factor functions (2033-05-12 refresh).

Every persisted factor's signal is recomputed from price data via these
functions so drift re-validation can cover the full history including the
online period (2026-07-16 .. now). Frozen assets (HSI,SX5E,BTC,US10Y,CN10Y)
are excluded from OOS/recent cross-sections.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from factor_common import WATCHLIST, load_prices, factor_to_panel

FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]


def load_index_csv(symbol, prices):
    path = Path('../persistent/index_data') / f'{symbol}.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    visible = max(dd.index.max() for dd in prices.values())
    return df[df.index <= visible]


def rb(r, m, w, cond=None, min_obs=0.5):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    if cond is not None:
        z = z[cond.reindex(z.index).astype(bool)]
    if len(z) < 30:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w, min_periods=int(w * min_obs)).cov(z['m']) / \
        z['m'].rolling(w, min_periods=int(w * min_obs)).var().replace(0, np.nan)
    return b.reindex(r.index)


def f_cn10y_beta_60(df, s, ref=None): return rb(df['close'].pct_change(), ref['cn10y_d'], 60)
def f_comm_basket_beta_60(df, s, ref=None): return rb(df['close'].pct_change(), ref['comm_r'], 60)
def f_copper_gold_beta_20(df, s, ref=None): return rb(df['close'].pct_change(), ref['cg_spread'], 20)
def f_down_beta_60(df, s, ref=None): return rb(df['close'].pct_change(), ref['spx_r'], 60, cond=ref['spx_r'] < 0)
def f_dxy_beta_cond_60x20(df, s, ref=None):
    r = df['close'].pct_change()
    return rb(r, ref['dxy_r'], 60) * (ref['dxy']['close'] / ref['dxy']['close'].shift(20) - 1.0)
def f_eurusd_beta_cond_60x20(df, s, ref=None):
    r = df['close'].pct_change()
    return rb(r, ref['eur_r'], 60) * (ref['eur']['close'] / ref['eur']['close'].shift(20) - 1.0)
def f_gap_freq_60(df, s, ref=None):
    gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
    return (gap > 0.01).astype(float).rolling(60, min_periods=30).mean()
def f_hilo_pos_60(df, s, ref=None):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return ((df['close'] - lo) / (hi - lo).replace(0, np.nan))
def f_hilo_vol_ratio_20(df, s, ref=None):
    c = df['close']
    rng = (c.rolling(20).max() - c.rolling(20).min()) / c
    v = c.pct_change().rolling(20).std()
    return (rng / v).replace([np.inf, -np.inf], np.nan)
def f_hs300_beta_60(df, s, ref=None): return rb(df['close'].pct_change(), ref['hs300_r'], 60)
def f_intraday_ret_skew_20(df, s, ref=None):
    return (df['close'] / df['open'] - 1.0).rolling(20, min_periods=12).skew()
def f_mom_accel_60_120(df, s, ref=None):
    c = df['close']
    return c.shift(5) / c.shift(65) - c.shift(5) / c.shift(125)
def f_range_amplitude_60(df, s, ref=None):
    c = df['close']
    return ((c.rolling(60).max() - c.rolling(60).min()) / c.rolling(60).mean())
def f_range_skew_20(df, s, ref=None):
    return ((df['high'] - df['low']) / df['close']).rolling(20, min_periods=12).skew()
def f_sign_persist_20(df, s, ref=None):
    r = df['close'].pct_change()
    same = (np.sign(r) == np.sign(r.shift(1))).astype(float)
    same[r == 0] = np.nan
    return same.rolling(20, min_periods=8).mean()
def f_spx_beta_60(df, s, ref=None): return rb(df['close'].pct_change(), ref['spx_r'], 60)
def f_streak_60(df, s, ref=None):
    r = df['close'].pct_change()
    up = (r > 0).astype(int); dn = (r < 0).astype(int)
    up_s = up.groupby((up != up.shift()).cumsum()).cumsum()
    dn_s = dn.groupby((dn != dn.shift()).cumsum()).cumsum()
    net = up_s - dn_s
    return net.rolling(60).max() / 60.0
def f_vix_beta_cond_60x20(df, s, ref=None):
    r = df['close'].pct_change()
    return -rb(r, ref['vix_r'], 60) * (ref['vix']['close'] / ref['vix']['close'].shift(20) - 1.0)
def f_vol_adj_mom_20_60(df, s, ref=None):
    r = df['close'].pct_change()
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    return (mom / r.rolling(60).std()).replace([np.inf, -np.inf], np.nan)
def f_vol_of_vol20x60(df, s, ref=None):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()
def f_vol_regime_switch_20x60(df, s, ref=None):
    rvol = df['close'].pct_change().rolling(20).std()
    med = rvol.rolling(60).median()
    state = (rvol > med).astype(float)
    switch = (state != state.shift()).astype(float)
    return switch.rolling(60).mean()
def f_dd_raw(df, s, ref=None):
    c = df['close']
    run_max = c.rolling(120, min_periods=60).max()
    below = (c < run_max).astype(int)
    grp = (below != below.shift()).cumsum()
    days_since = below.groupby(grp).cumsum()
    return np.log1p(days_since)
def f_dd_duration_120_resid(df, s, ref=None):
    dd = f_dd_raw(df, s)
    return dd - dd.rolling(120).mean()


def build_refs(prices):
    spx_r = prices['SPX']['close'].pct_change()
    hs300_r = prices['000300.SH']['close'].pct_change()
    cn10y_d = prices['CN10Y']['close'].diff()
    dxy = load_index_csv('DXY', prices); dxy_r = dxy['close'].pct_change()
    eur = load_index_csv('EURUSD', prices); eur_r = eur['close'].pct_change()
    vix = load_index_csv('VIX', prices); vix_r = vix['close'].pct_change()
    comm_r = pd.concat([prices[s]['close'].pct_change().rename(s)
                        for s in ['XAU', 'COPPER', 'WTI']], axis=1).mean(axis=1)
    cg_spread = prices['COPPER']['close'].pct_change() - prices['XAU']['close'].pct_change()
    return dict(spx_r=spx_r, hs300_r=hs300_r, cn10y_d=cn10y_d, dxy=dxy, dxy_r=dxy_r,
                eur=eur, eur_r=eur_r, vix=vix, vix_r=vix_r, comm_r=comm_r, cg_spread=cg_spread)


FUNCS = {
    'cn10y_beta_60': f_cn10y_beta_60,
    'comm_basket_beta_60': f_comm_basket_beta_60,
    'copper_gold_beta_20': f_copper_gold_beta_20,
    'dd_duration_120_resid': f_dd_duration_120_resid,
    'down_beta_60': f_down_beta_60,
    'dxy_beta_cond_60x20': f_dxy_beta_cond_60x20,
    'eurusd_beta_cond_60x20': f_eurusd_beta_cond_60x20,
    'gap_freq_60': f_gap_freq_60,
    'hilo_pos_60': f_hilo_pos_60,
    'hilo_vol_ratio_20': f_hilo_vol_ratio_20,
    'hs300_beta_60': f_hs300_beta_60,
    'intraday_ret_skew_20': f_intraday_ret_skew_20,
    'mom_accel_60_120': f_mom_accel_60_120,
    'range_amplitude_60': f_range_amplitude_60,
    'range_skew_20': f_range_skew_20,
    'sign_persist_20': f_sign_persist_20,
    'spx_beta_60': f_spx_beta_60,
    'streak_60': f_streak_60,
    'vix_beta_cond_60x20': f_vix_beta_cond_60x20,
    'vol_adj_mom_20_60': f_vol_adj_mom_20_60,
    'vol_of_vol20x60': f_vol_of_vol20x60,
    'vol_regime_switch_20x60': f_vol_regime_switch_20x60,
}


def library_panels(prices, refs=None):
    """Recompute all 22 persisted library factor panels on full history."""
    if refs is None:
        refs = build_refs(prices)
    out = {}
    for name, fn in FUNCS.items():
        out[name] = factor_to_panel(lambda df, s, fn=fn: fn(df, s, ref=refs), prices)
    return out
