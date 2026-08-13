"""Shared validation harness for miner_2 factor research (2033-11-28 cycle).

Loads the 15 tradable instruments + 5 macro observation series, truncates to
the visible horizon (date.json visible_through), computes forward returns and
per-date cross-sectional Spearman IC metrics. No future data is used.
"""
import json
import numpy as np
import pandas as pd

WATCH = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

DATA_DIR = '../persistent/stock_data'
MACRO_DIR = '../persistent/index_data'


def get_visible_through():
    d = json.load(open('../persistent/date.json'))
    return pd.Timestamp(d['visible_through'])


def load_series(name, macro=False):
    path = f'{MACRO_DIR}/{name}.csv' if macro else f'{DATA_DIR}/{name}.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').drop_duplicates('date')
    df = df[df['date'] <= get_visible_through()]
    df = df.set_index('date')
    return df


def load_prices():
    px = {}
    for s in WATCH:
        df = load_series(s)
        px[s] = df['close'].rename(s)
    return pd.concat(px.values(), axis=1).sort_index()


def load_macro():
    mx = {}
    for m in MACRO:
        df = load_series(m, macro=True)
        mx[m] = df['close'].rename(m)
    return pd.concat(mx.values(), axis=1).sort_index()


def forward_returns(px, horizons=(1, 2, 3, 5, 10, 20)):
    """Forward return over h trading days, using each asset's own series."""
    out = {}
    for h in horizons:
        fr = px.shift(-h) / px - 1.0
        out[h] = fr
    return out


def factor_ics(factor_df, fwd_ret, min_valid=8):
    """Per-date Spearman IC between factor and forward return (h=10 default).

    Returns dict with mean IC, std, ICIR, hit ratio, n dates, coverage.
    """
    res = {}
    for h, fr in fwd_ret.items():
        ics = []
        dates = []
        valid = 0
        for dt in factor_df.index:
            if dt not in fr.index:
                continue
            f = factor_df.loc[dt]
            r = fr.loc[dt]
            mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
            if mask.sum() < min_valid:
                continue
            ic = f[mask].corr(r[mask], method='spearman')
            if not np.isfinite(ic):
                continue
            ics.append(ic)
            dates.append(dt)
            valid += mask.sum()
        ics = np.array(ics)
        if len(ics) == 0:
            res[h] = {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n': 0,
                      'ic_std': np.nan, 'coverage_asset_days': np.nan}
            continue
        ic_mean = ics.mean()
        ic_std = ics.std(ddof=1) if len(ics) > 1 else np.nan
        icir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        hit = (np.sign(ics) == np.sign(ic_mean)).mean()
        res[h] = {'ic': float(ic_mean), 'icir': float(icir), 'hit': float(hit),
                  'n': int(len(ics)), 'ic_std': float(ic_std),
                  'coverage_asset_days': float(valid) / (len(ics) * 15) if len(ics) else np.nan}
    return res


def summarize(factor_df, fwd_ret, label, min_valid=8, recent=250):
    res = factor_ics(factor_df, fwd_ret, min_valid=min_valid)
    main = res[10]
    # recent-window (last `recent` dates with valid IC)
    fr10 = fwd_ret[10]
    ics_all = []
    dates_all = []
    for dt in factor_df.index:
        if dt not in fr10.index:
            continue
        f = factor_df.loc[dt]
        r = fr10.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            ics_all.append(ic)
            dates_all.append(dt)
    ics_all = np.array(ics_all)
    rec = {}
    if len(ics_all) >= 20:
        rics = ics_all[-recent:]
        rec = {'ic': float(rics.mean()), 'icir': float(rics.mean() / (rics.std(ddof=1) if rics.std(ddof=1) > 0 else np.nan)),
               'n': int(len(rics))}
    # turnover: mean abs change in cross-sectional rank (per 10d) normalized
    rdf = factor_df.rank(axis=1)
    turn = None
    if len(rdf) > 10:
        d = rdf.diff(10).abs().mean().mean() / (len(rdf.columns) - 1)
        turn = float(d)
    cov_dates = float((factor_df.notna().sum(axis=1) >= min_valid).mean())
    print(f'=== {label} ===')
    print(f'  H10 IC={main["ic"]:.4f} ICIR={main["icir"]:.4f} hit={main["hit"]:.3f} '
          f'n_dates={main["n"]} cov_asset_days={main["coverage_asset_days"]:.3f} '
          f'cov_dates_ge8={cov_dates:.3f} turnover10d={turn if turn is None else round(turn,3)}')
    if rec:
        print(f'  recent{recent}d IC={rec["ic"]:.4f} ICIR={rec["icir"]:.4f} n={rec["n"]}')
    for h in (1, 2, 5, 20):
        print(f'    H{h}: IC={res[h]["ic"]:.4f} ICIR={res[h]["icir"]:.4f}')
    return res
