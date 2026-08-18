"""Shared helpers for miner_3 cycles: load 15-asset universe + macro, compute 10d forward returns, IC stats.
Uses ../persistent CSV files as-of visible_through date from ../persistent/date.json.
"""
import json, os, math
import numpy as np
import pandas as pd

WATCH = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
MACRO = {'VIX':'../persistent/index_data/VIX.csv','DXY':'../persistent/index_data/DXY.csv',
         'USDCNY':'../persistent/index_data/USDCNY.csv','USDJPY':'../persistent/index_data/USDJPY.csv',
         'EURUSD':'../persistent/index_data/EURUSD.csv'}

def load_visible_through():
    dj = json.load(open('../persistent/date.json'))
    return dj.get('visible_through', dj['current_date'])

def load_prices(asof=None):
    if asof is None:
        asof = load_visible_through()
    closes = {}
    for s in WATCH:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.Timestamp(asof)].set_index('date')['close']
        closes[s] = df
    px = pd.DataFrame(closes).sort_index()
    return px

def load_macro(asof=None):
    if asof is None:
        asof = load_visible_through()
    out = {}
    for name, path in MACRO.items():
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.Timestamp(asof)].set_index('date')['close']
        out[name] = df
    return pd.DataFrame(out).sort_index()

def zscore_series(x, winsor=5.0):
    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median(); mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return x * 0.0
    z = (x - med) / (1.4826 * mad)
    return z.clip(-winsor, winsor)

def cross_sectional_ic(factor_df, fwd_ret_df, min_assets=8):
    """Date-wise Spearman IC between factor values and forward returns. Returns DataFrame of daily ICs."""
    recs = []
    common = factor_df.index.intersection(fwd_ret_df.index)
    for d in common:
        f = factor_df.loc[d]
        r = fwd_ret_df.loc[d]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(m.sum())
        if n >= min_assets:
            ic = f[m].corr(r[m], method='spearman')
            if not np.isnan(ic):
                recs.append((d, ic, n))
    out = pd.DataFrame(recs, columns=['date', 'ic', 'n']).set_index('date')
    return out

def ic_stats(icdf, label=''):
    if len(icdf) == 0:
        return {'label': label, 'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n_dates': 0}
    ic = icdf['ic'].mean()
    sd = icdf['ic'].std(ddof=1)
    icir = ic / sd if sd and not np.isnan(sd) and sd > 0 else 0.0
    hit = (icdf['ic'] > 0).mean()
    avg_n = icdf['n'].mean()
    return {'label': label, 'ic': ic, 'icir': icir, 'hit': hit, 'n_dates': len(icdf), 'avg_n': avg_n}

def spearman_panel_rho(a_df, b_df):
    """Date-wise spearman corr of two factor panels, averaged, then max abs across library."""
    common = a_df.index.intersection(b_df.index)
    rhos = []
    for d in common:
        a = a_df.loc[d]; b = b_df.loc[d]
        m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 6:
            r = a[m].corr(b[m], method='spearman')
            if not np.isnan(r):
                rhos.append(r)
    if not rhos:
        return np.nan
    return float(np.mean(rhos))

def regime_split(icdf, breaks=(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-01-01'))):
    labels = ['2020-2021 COVID/recovery', '2022-2023 tightening/AI', '2024+ crypto/commodity']
    out = {}
    segs = [(icdf.index < breaks[0], labels[0]), ((icdf.index >= breaks[0]) & (icdf.index < breaks[1]), labels[1]), (icdf.index >= breaks[1], labels[2])]
    for mask, lab in segs:
        sub = icdf[mask]
        if len(sub):
            st = ic_stats(sub)
            out[lab] = [st['ic'], st['icir'], int(st['n_dates'])]
    return out