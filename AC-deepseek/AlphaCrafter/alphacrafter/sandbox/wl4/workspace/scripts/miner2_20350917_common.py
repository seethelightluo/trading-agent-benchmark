"""Shared harness for miner_2 factor research (2035-09-17 cycle).

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
    out = {}
    for h in horizons:
        fr = px.shift(-h) / px - 1.0
        out[h] = fr
    return out


def factor_ics(factor_df, fwd_ret, min_valid=8):
    """Per-date Spearman IC between factor and forward returns (all horizons)."""
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
    rics = None
    fr10 = fwd_ret[10]
    ics_all = []
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
    ics_all = np.array(ics_all)
    rec = {}
    if len(ics_all) >= 20:
        rics = ics_all[-recent:]
        rec = {'ic': float(rics.mean()),
               'icir': float(rics.mean() / (rics.std(ddof=1) if rics.std(ddof=1) > 0 else np.nan)),
               'n': int(len(rics))}
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


def rank_ic_series(factor_df, fwd10, min_valid=8):
    ics, dates = [], []
    for dt in factor_df.index:
        if dt not in fwd10.index:
            continue
        f = factor_df.loc[dt]
        r = fwd10.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name='ic')


def max_library_corr(candidate, library):
    best, best_key = 0.0, None
    for name, lib_sig in library.items():
        both = pd.concat([candidate.stack().rename('cand'), lib_sig.stack().rename('lib')], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both['cand'].corr(both['lib']))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key


def library_signals(px, ret, macro):
    """Recompute existing library factor panels from stored definitions."""
    sig = {}
    # vol_adj_mom_accel_20x60: (mom20 - mom60)/vol20
    sig['vol_adj_mom_accel_20x60'] = (
        (px / px.shift(20) - 1.0) - (px / px.shift(60) - 1.0)
    ) / ret.rolling(20).std()
    # dn_mkt_beta_60d: beta(asset, min(mkt,0), 60), mkt = ew mean of 15
    mkt = ret.mean(axis=1)
    mkt_dn = mkt.where(mkt < 0, 0.0)
    beta = {}
    for a in ret.columns:
        z = pd.concat([ret[a].rename('a'), mkt_dn.rename('m')], axis=1).dropna()
        b = z['a'].rolling(60).cov(z['m']) / z['m'].rolling(60).var()
        beta[a] = b
    sig['dn_mkt_beta_60d'] = pd.DataFrame(beta, index=ret.index)
    # rate_beta_cn10y_60d: beta(asset, CN10Y ret, 60)
    r_cn = px['CN10Y'].pct_change()
    beta_cn = {}
    for a in ret.columns:
        z = pd.concat([ret[a].rename('a'), r_cn.rename('m')], axis=1).dropna()
        b = z['a'].rolling(60).cov(z['m']) / z['m'].rolling(60).var()
        beta_cn[a] = b
    sig['rate_beta_cn10y_60d'] = pd.DataFrame(beta_cn, index=ret.index)
    return sig
