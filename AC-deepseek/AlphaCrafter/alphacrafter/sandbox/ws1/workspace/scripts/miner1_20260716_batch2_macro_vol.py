"""miner_1 batch-2: explore macro-conditional (DXY/VIX beta), vol term structure,
skewness, z-score reversion, close-position, autocorrelation, volume trend.

Admission gate at h=10: |IC| >= 0.007 and |ICIR| >= 0.084.
Also prints coverage, 10-day-rebalance rank turnover, and pairwise correlation
with already-passing batch-1 candidates.
"""
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic, summarize)

H = 10
MIN_VALID = 8


def _ret(close):
    return close.pct_change()


def vol_ratio_short_long(short=20, long=120):
    def fn(sym, close, volume):
        vs = _ret(close).rolling(short).std()
        vl = _ret(close).rolling(long).std()
        return (1.0 - vs / vl).replace([np.inf, -np.inf], np.nan)  # high=calm
    return fn


def drawdown_dist(win=60):
    def fn(sym, close, volume):
        return 1.0 - close / close.rolling(win).max()
    return fn


def ret_skew(win=60):
    def fn(sym, close, volume):
        return _ret(close).rolling(win).skew()
    return fn


def autocorr_ret(win=60, lag=5):
    def fn(sym, close, volume):
        r = _ret(close)
        a = r.rolling(win).apply(lambda x: x.autocorr(lag) if len(x) == win else np.nan, raw=False)
        return a
    return fn


def zscore_rev(win=20):
    def fn(sym, close, volume):
        ma = close.rolling(win).mean()
        sd = _ret(close).rolling(win).std() * close
        return ((close - ma) / sd).replace([np.inf, -np.inf], np.nan)
    return fn


def inv_vol(win=60):
    def fn(sym, close, volume):
        sd = _ret(close).rolling(win).std()
        return (-sd).replace([np.inf, -np.inf], np.nan)
    return fn


def downside_ratio(win=60):
    def fn(sym, close, volume):
        r = _ret(close)
        dd = r.clip(upper=0).rolling(win).std()
        v = r.rolling(win).std()
        return (dd / v).replace([np.inf, -np.inf], np.nan)
    return fn


def volume_trend(short=5, long=60):
    def fn(sym, close, volume):
        if volume is None or volume.dropna().empty:
            return None
        v = volume.astype(float)
        return (v.rolling(short).mean() / v.rolling(long).mean()).replace([np.inf, -np.inf], np.nan)
    return fn


def macro_beta_cond(macro_key, sign=-1.0, win=60, mom=20):
    """factor = sign * rolling_beta(asset_ret, macro_ret) * macro_momentum."""
    def fn(sym, close, volume, panel=None):
        if panel is None:
            return None
        macro = panel['macro'].get(macro_key)
        if macro is None:
            return None
        grid = panel['grid']
        r_a = close.pct_change().reindex(grid)
        r_m = macro.pct_change().reindex(grid)
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = (macro.reindex(grid) / macro.shift(mom).reindex(grid) - 1.0)
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn


CANDIDATES = [
    ('mom_60d_skip5', lambda sym, c, v: c.shift(5) / c.shift(65) - 1.0),
    ('vol_term_20_120', vol_ratio_short_long(20, 120)),
    ('drawdown_60d', drawdown_dist(60)),
    ('ret_skew_60d', ret_skew(60)),
    ('zscore_rev_20d', zscore_rev(20)),
    ('inv_vol_60d', inv_vol(60)),
    ('downside_ratio_60d', downside_ratio(60)),
    ('autocorr_60d_5lag', autocorr_ret(60, 5)),
    ('volume_trend_5_60', volume_trend(5, 60)),
    ('dxy_beta_cond_60x20', macro_beta_cond('DXY', sign=-1.0)),
    ('vix_beta_cond_60x20', macro_beta_cond('VIX', sign=-1.0)),
]


def with_panel(fn, panel):
    """Wrap factor fn to receive panel for macro-conditional candidates."""
    def wrapped(sym, close, volume):
        return fn(sym, close, volume, panel=panel)
    return wrapped


if __name__ == '__main__':
    panel = build_panel()
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    ret = forward_returns(closes, grid, H)

    # batch-1 passing library for correlation reference
    lib1 = {
        'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
        'mom_20d_skip5': lambda s, c, v: c.shift(5) / c.shift(25) - 1.0,
        'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
        'trend_sma60': lambda s, c, v: c / c.rolling(60).mean() - 1.0,
        'trend_sma120': lambda s, c, v: c / c.rolling(120).mean() - 1.0,
        'risk_adj_trend20': lambda s, c, v: (c.pct_change().rolling(20).mean()
                                             / c.pct_change().rolling(20).std()
                                             ).replace([np.inf, -np.inf], np.nan),
        'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    }
    lib_frames = {}
    for lbl, fn in lib1.items():
        lib_frames[lbl] = factor_values(closes, volumes, grid, fn)

    results = {}
    new_frames = {}
    print(f'=== BATCH-2 VALIDATION h={H} ===')
    for label, fn in CANDIDATES:
        ffn = with_panel(fn, panel) if label in ('dxy_beta_cond_60x20', 'vix_beta_cond_60x20') else fn
        fac = factor_values(closes, volumes, grid, ffn)
        new_frames[label] = fac
        cov_assets = float(fac.notna().mean().mean())
        cov_dates = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
        f10 = fac.iloc[::10]
        turn = float(f10.rank(axis=1).diff().abs().mean().mean()) if len(f10) > 2 else np.nan
        ics = daily_ic(fac, ret, min_valid=MIN_VALID)
        m = summarize(ics, label, H)
        m['coverage_assets'] = cov_assets
        m['coverage_dates8'] = cov_dates
        m['turnover'] = turn
        results[label] = m
        print(f'   cov_assets={cov_assets:.3f} cov_dates8+={cov_dates:.3f} turn={turn:.3f}')

    print('\n=== YEARLY IC (h=10) for batch-2 ===')
    for label, fac in new_frames.items():
        ics = daily_ic(fac, ret, min_valid=MIN_VALID)
        if len(ics) == 0:
            continue
        s = ics['ic']
        print(f'[{label}]', {int(y): round(float(v), 4) for y, v in s.groupby(s.index.year).mean().items()})

    print('\n=== PAIRWISE CORR (new vs batch-1 library) ===')
    all_labels = list(lib_frames.keys()) + list(new_frames.keys())
    corr = pd.DataFrame(index=all_labels, columns=all_labels, dtype=float)
    cd = None
    for l in all_labels:
        d = lib_frames[l].dropna(how='all').index if l in lib_frames else new_frames[l].dropna(how='all').index
        cd = d if cd is None else cd.intersection(d)
    for i, a in enumerate(all_labels):
        for j, b in enumerate(all_labels):
            if i > j:
                continue
            fa = lib_frames[a] if a in lib_frames else new_frames[a]
            fb = lib_frames[b] if b in lib_frames else new_frames[b]
            cs = []
            for t in cd:
                x, y = fa.loc[t], fb.loc[t]
                mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 8:
                    r = pd.Series(x[mask]).corr(pd.Series(y[mask]), method='spearman')
                    if np.isfinite(r):
                        cs.append(r)
            v = float(np.mean(cs)) if cs else np.nan
            corr.loc[a, b] = v
            corr.loc[b, a] = v
    pd.set_option('display.width', 250)
    print(corr.round(2))
