"""miner_1 shared factor-validation library.

Loads synthetic benchmark data through visible_through (2035-04-13), computes
cross-sectional rank IC panels for candidate factors, and reports the shared
admission metrics (|IC|>=0.0070, |ICIR|>=0.0840 at h=10).
"""
import json
import base64
import zlib
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VIS = '2035-04-13'
WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']


def calendar():
    d = json.load(open('../persistent/date.json'))
    cal = pd.Index([x for x in d['trading_days'] if x <= VIS])
    return cal


def load_closes():
    cal = calendar()
    out = {}
    for s in WATCH:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df[df['date'] <= VIS].set_index('date')['close']
        out[s] = df.reindex(cal)
    return pd.DataFrame(out)


def load_volumes():
    cal = calendar()
    out = {}
    for s in WATCH:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df[df['date'] <= VIS].set_index('date')['volume']
        out[s] = df.reindex(cal)
    return pd.DataFrame(out)


def load_macro():
    cal = calendar()
    out = {}
    for m in MACRO:
        df = pd.read_csv(f'../persistent/index_data/{m}.csv')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df[df['date'] <= VIS]
        col = 'close' if 'close' in df.columns else df.columns[1]
        out[m] = df.set_index('date')[col].reindex(cal)
    return pd.DataFrame(out)


def load_library_signals():
    """Load signal panels from effective factor JSONs (base64:zlib:csv)."""
    sigs = {}
    for f in ['vol_adj_mom_accel_20x60', 'dn_mkt_beta_60d']:
        try:
            d = json.load(open(f'factors/{f}.json'))
            sa = d['validation']['signal_artifact']
            raw = base64.b64decode(sa['data'])
            csv_txt = zlib.decompress(raw).decode('utf-8')
            panel = pd.read_csv(io.StringIO(csv_txt), index_col=0)
            panel.index = [str(x)[:10] for x in panel.index]
            panel = panel.reindex(calendar())
            sigs[f] = panel
        except Exception as e:
            print(f'  [warn] cannot load library signal {f}: {e}')
    return sigs


def forward_returns(closes, horizons=(1, 2, 3, 5, 10, 20)):
    """Forward close-to-close returns at each horizon (calendar days)."""
    out = {}
    for h in horizons:
        out[h] = closes.shift(-h) / closes - 1.0
    return out


def rank_ic_panel(factor, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman IC between factor and forward return."""
    idx = factor.index.intersection(fwd_ret.index)
    ics = {}
    for t in idx:
        f = factor.loc[t]
        r = fwd_ret.loc[t]
        m = f.notna() & r.notna()
        if m.sum() >= min_valid:
            ics[t] = spearmanr(f[m], r[m]).statistic
    s = pd.Series(ics)
    return s


def summarize_ic(ic_series, label=''):
    ic = ic_series.mean()
    std = ic_series.std(ddof=1)
    icir = ic / std if std > 0 else np.nan
    hit = (ic_series > 0).mean()
    return {
        'ic': round(float(ic), 5),
        'ic_std': round(float(std), 5),
        'icir': round(float(icir), 5),
        'ic_hit_ratio': round(float(hit), 4),
        'n_ic_dates': int(len(ic_series)),
        'first_ic_date': str(ic_series.index.min())[:10],
        'last_ic_date': str(ic_series.index.max())[:10],
    }


def coverage_metrics(factor):
    valid = factor.notna()
    asset_days = float(valid.sum().sum()) / float(valid.shape[0] * valid.shape[1])
    dates_ge8 = float((valid.sum(axis=1) >= 8).mean())
    return {
        'coverage_asset_days': round(asset_days, 4),
        'coverage_dates_ge8': round(dates_ge8, 4),
    }


def turnover_10d(factor, step=10):
    """Mean abs rank change at 10-day spacing, normalized by n-1."""
    ranks = factor.rank(axis=1)
    valid = factor.notna()
    out = []
    prev_ts = None
    prev_row = None
    for t in factor.index:
        ts = pd.Timestamp(t)
        if prev_ts is not None and (ts - prev_ts).days >= step:
            a, b = ranks.loc[prev_row], ranks.loc[t]
            m = valid.loc[prev_row] & valid.loc[t]
            n = int(m.sum())
            if n >= 8:
                chg = (a[m] - b[m]).abs().mean() / (n - 1)
                out.append(chg)
            prev_ts, prev_row = ts, t
        elif prev_ts is None:
            prev_ts, prev_row = ts, t
    return round(float(np.mean(out)), 4) if out else np.nan


def library_corr(factor, lib_signals, max_lag_days=0):
    """Max abs cross-sectional correlation with library factor signals."""
    best = 0.0
    best_f = None
    for name, sig in lib_signals.items():
        common = factor.index.intersection(sig.index)
        cors = []
        for t in common:
            a, b = factor.loc[t], sig.loc[t]
            m = a.notna() & b.notna()
            if m.sum() >= 8:
                cors.append(spearmanr(a[m], b[m]).statistic)
        if cors:
            mc = max(abs(x) for x in cors)
            if mc > best:
                best = mc
                best_f = name
    return best, best_f


def evaluate(factor, closes, horizons=(1, 2, 3, 5, 10, 20), min_valid=8,
             lib_signals=None):
    """Full evaluation: decay IC by horizon + headline metrics at h=10."""
    fwd = forward_returns(closes, horizons)
    res = {'metrics_by_horizon': {}}
    for h in horizons:
        ic = rank_ic_panel(factor, fwd[h], min_valid)
        res['metrics_by_horizon'][str(h)] = summarize_ic(ic)
    h10 = res['metrics_by_horizon']['10']
    res['coverage'] = coverage_metrics(factor)
    res['turnover_10d_rank'] = turnover_10d(factor)
    if lib_signals:
        mc, mf = library_corr(factor, lib_signals)
        res['max_abs_library_correlation'] = round(mc, 4)
        res['max_corr_factor'] = mf
    res['headline_h10'] = h10
    return res
