"""Shared factor-IC validation library for miner_1 (15-asset cross-asset universe).

Data window: <= CUT (previous completed trading day before the decision date).
No backtest/step usage; pure factor analytics. Anti-leakage: future rows are
filtered out before any computation.
"""
import pandas as pd
import numpy as np
import glob
import os

CUT = '2029-06-17'          # last completed trading day before 2029-06-18
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX',
          'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'VIX', 'EURUSD', 'USDJPY', 'USDCNY']


def load_prices(cut=CUT):
    """Return close panel (dates x 15 assets), rows <= cut."""
    files = sorted(glob.glob('../persistent/stock_data/*.csv'))
    px = {}
    for f in files:
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        px[sym] = df.set_index('date').sort_index()['close'].astype(float)
    panel = pd.DataFrame(px).sort_index()
    panel = panel[panel.index <= pd.Timestamp(cut)]
    panel = panel[ASSETS]
    return panel


def load_macro(cut=CUT):
    """Return dict of observation-only macro closes, rows <= cut."""
    out = {}
    for s in MACRO:
        df = pd.read_csv(f'../persistent/index_data/{s}.csv')
        df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        out[s] = df.set_index('date').sort_index()['close'].astype(float)
        out[s] = out[s][out[s].index <= pd.Timestamp(cut)]
    return out


def rolling_beta(y, x, win=60, min_obs=40):
    """Rolling beta of y on x over trailing win (expanding window style per date)."""
    z = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    out = pd.Series(index=z.index, dtype=float)
    for i in range(len(z)):
        w = z.iloc[max(0, i - win + 1):i + 1]
        if len(w) < min_obs:
            continue
        var = float(w.x.var())
        if var <= 1e-14:
            continue
        out.iloc[i] = float(w.y.cov(w.x) / var)
    return out


def rolling_beta_fast(y, x, win=60, min_obs=40):
    """Vectorized rolling beta via rolling cov/var (uses full trailing window)."""
    var = x.rolling(win, min_periods=min_obs).var()
    cov = y.rolling(win, min_periods=min_obs).cov(x)
    return cov / var


def rank_ic_series(fval, fwd, min_valid=8):
    """Per-date Spearman IC between factor values and forward returns.

    fval: DataFrame dates x assets (raw factor values).
    fwd : DataFrame dates x assets (forward h-day returns).
    Returns Series of IC per date (dates with >= min_valid valid pairs).
    """
    ics = {}
    idx = fval.index.intersection(fwd.index)
    for d in idx:
        a = fval.loc[d]
        b = fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < min_valid:
            continue
        aa = a[m].rank()
        bb = b[m].rank()
        if aa.nunique() < 2 or bb.nunique() < 2:
            continue
        ics[d] = np.corrcoef(aa, bb)[0, 1]
    return pd.Series(ics).sort_index()


def summarize_ic(ic, name='factor', horizons=(1, 2, 3, 5, 10, 20), fval=None,
                 fwd_map=None, min_valid=8):
    """Full metric summary: IC, ICIR, hit, coverage, turnover, decay."""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {'name': name, 'n_ic_dates': 0}
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if n > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((np.sign(ic) == np.sign(ic_mean)).mean())
    out = {
        'name': name,
        'n_ic_dates': n,
        'ic': round(ic_mean, 6),
        'icir': round(icir, 6),
        'ic_hit_ratio': round(hit, 4),
        'ic_std': round(ic_std, 6),
        'ic_first': str(ic.index[0].date()),
        'ic_last': str(ic.index[-1].date()),
    }
    # coverage
    if fval is not None:
        cov = float(fval.notna().mean().mean())
        dates_ge8 = float((fval.notna().sum(axis=1) >= min_valid).mean())
        out['coverage_asset_days'] = round(cov, 4)
        out['coverage_dates_ge8'] = round(dates_ge8, 4)
        # turnover: mean absolute cross-sectional rank change over 10 trading days
        rk = fval.rank(axis=1)
        rk10 = rk.shift(10)
        trn = (rk - rk10).abs().mean()
        out['turnover_10d_rank'] = round(float(trn), 4)
    # decay
    if fwd_map is not None and fval is not None:
        decay = {}
        for h in horizons:
            if h in fwd_map:
                ic_h = rank_ic_series(fval, fwd_map[h], min_valid=min_valid)
                decay[str(h)] = round(float(ic_h.mean()), 4) if len(ic_h) else None
        out['decay_ic_by_horizon'] = decay
    return out


def library_corr(fval, lib_factors, min_valid=8):
    """Max absolute cross-sectional Spearman correlation with library factors.

    lib_factors: dict factor_id -> DataFrame (dates x assets) of raw values.
    Computed per date then averaged; returns (max_abs_mean, factor, mean_abs_map).
    """
    means = {}
    for fid, lf in lib_factors.items():
        corrs = []
        idx = fval.index.intersection(lf.index)
        for d in idx:
            a = fval.loc[d]
            b = lf.loc[d]
            m = a.notna() & b.notna()
            if m.sum() < min_valid:
                continue
            aa = a[m].rank()
            bb = b[m].rank()
            if aa.nunique() < 2 or bb.nunique() < 2:
                continue
            corrs.append(abs(np.corrcoef(aa, bb)[0, 1]))
        means[fid] = float(np.mean(corrs)) if corrs else 0.0
    best = max(means, key=means.get) if means else None
    return (means.get(best, 0.0) if best else 0.0), best, means
