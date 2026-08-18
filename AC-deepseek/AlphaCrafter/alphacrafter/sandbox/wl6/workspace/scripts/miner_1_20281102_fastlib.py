"""miner_1 2028-11-02: fast vectorized validation machinery (replaces slow per-date loops).

Loads the 15-asset tradable universe + 5 macro observation signals on a common
trading calendar (macro index), truncates to visible_through (2028-11-01), and
computes cross-sectional rank IC / ICIR / hit ratio / coverage / turnover /
decay using vectorized row-wise Pearson on ranks.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'VIX', 'USDCNY', 'USDJPY', 'EURUSD']

DATE_JSON = Path('../persistent/date.json')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')

IC_TH, ICIR_TH = 0.0070, 0.0840
MIN_ASSETS_PER_DATE = 8


def visible_through():
    d = json.load(open(DATE_JSON))
    return pd.to_datetime(d.get('visible_through', d.get('current_date')))


def load_macro_panel():
    vth = visible_through()
    frames = {}
    for s in MACRO:
        df = pd.read_csv(INDEX_DIR / f'{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= vth].sort_values('date')
        frames[s] = df.set_index('date')['close'].astype(float)
    return pd.DataFrame(frames).sort_index()


def load_close_panel():
    vth = visible_through()
    macro = load_macro_panel()
    cal = macro.index
    closes = {}
    for s in WATCH:
        df = pd.read_csv(STOCK_DIR / f'{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= vth].sort_values('date')
        srs = df.set_index('date')['close'].astype(float)
        closes[s] = srs.reindex(cal)
    return pd.DataFrame(closes).sort_index()


def forward_returns(panel, horizons=(1, 2, 3, 5, 10, 20)):
    fwd = {}
    for h in horizons:
        fwd[h] = panel.shift(-h) / panel - 1.0
    return fwd


def _row_pearson_on_ranks(F, R, mask, min_assets=MIN_ASSETS_PER_DATE):
    """Vectorized Spearman (rank) IC per date row."""
    n = mask.sum(axis=1)
    valid = n >= min_assets
    out = np.full(len(F), np.nan)
    if not valid.any():
        return out, n
    x = np.where(mask, F, np.nan)
    y = np.where(mask, R, np.nan)
    xm = np.nanmean(x, axis=1, keepdims=True)
    ym = np.nanmean(y, axis=1, keepdims=True)
    xc = np.where(mask, x - xm, 0.0)
    yc = np.where(mask, y - ym, 0.0)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        ic = num / den
    out[valid] = ic[valid]
    return out, n


def rank_ic_series(factor_df, fwd_df, min_assets=MIN_ASSETS_PER_DATE):
    """Return (ic_series, n_series) with pandas index."""
    F = factor_df.rank(axis=1).values
    R = fwd_df.rank(axis=1).values
    mask = factor_df.notna().values & fwd_df.notna().values
    ic, n = _row_pearson_on_ranks(F, R, mask, min_assets)
    ic_s = pd.Series(ic, index=factor_df.index)
    n_s = pd.Series(n, index=factor_df.index)
    return ic_s, n_s


def summarize(ic_series, n_series, name, fwd=None, factor_df=None, label=''):
    ic = ic_series.dropna()
    out = {'factor': name}
    if len(ic) == 0:
        out.update({'ic': np.nan, 'icir': np.nan, 'hit': np.nan,
                    'n_dates': 0, 'status': 'NO_DATA'})
        return out
    out.update({
        'ic': float(ic.mean()),
        'icir': float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else 0.0,
        'hit': float((ic > 0).mean()),
        'n_dates': int(len(ic)),
        'mean_n_assets': float(n_series.reindex(ic.index).mean()),
    })
    for lo, hi, tag in [('2020-01-01', '2022-12-31', 'r2020_22'),
                        ('2023-01-01', '2026-12-31', 'r2023_26'),
                        ('2027-01-01', '2028-11-01', 'r2027_28')]:
        sub = ic[(ic.index >= lo) & (ic.index <= hi)]
        out[f'{tag}_ic'] = float(sub.mean()) if len(sub) else np.nan
        out[f'{tag}_n'] = int(len(sub))
    if fwd is not None and factor_df is not None:
        dec = {}
        for h in sorted(fwd.keys()):
            ic_h, _ = rank_ic_series(factor_df, fwd[h])
            ic_h = ic_h.dropna()
            dec[str(h)] = float(ic_h.mean()) if len(ic_h) else np.nan
        out['decay_ic'] = dec
    if factor_df is not None and len(factor_df) > 2:
        rk = factor_df.rank(axis=1)
        chg = rk.diff().abs().mean(axis=1).dropna()
        out['turnover_rank_abs'] = float(chg.mean()) if len(chg) else np.nan
    cov = factor_df.notna().mean()
    out['coverage_asset_days'] = float(cov.mean())
    out['coverage_dates_ge8'] = float((n_series >= 8).mean()) if len(n_series) else np.nan
    out['label'] = label
    return out


def max_abs_library_correlation(factor_df, lib_signals, label=''):
    """Max absolute pairwise time-series correlation vs existing library signals (row-wise on common dates)."""
    best = {}
    common_idx = factor_df.index
    f = factor_df.reindex(common_idx)
    for fid, sig in lib_signals.items():
        s = sig.reindex(common_idx)
        both = f.notna() & s.notna()
        if both.sum().sum() < 200:
            continue
        a = f[both].values.ravel()
        b = s[both].values.ravel()
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        best[fid] = r
    if not best:
        return None, {}
    fid = max(best, key=lambda k: abs(best[k]))
    return abs(best[fid]), {k: round(v, 3) for k, v in best.items()}
