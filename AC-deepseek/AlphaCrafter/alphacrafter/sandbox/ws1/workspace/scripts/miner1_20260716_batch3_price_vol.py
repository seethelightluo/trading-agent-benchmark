"""miner_1 batch-3b: price-shape / volume-liquidity factors orthogonal to library.

Candidates:
  - trend_eff_20d / trend_eff_60d: Kaufman efficiency ratio |P_t/P_{t-n}-1| / sum(|r|)
  - amihud_60d: |ret| / volume (illiquidity), sign flipped so high=liquid
  - vol_zscore_20_120: z-score of 20d realized vol vs trailing 120d distribution
  - candle_body_20d: mean(|close-open| / (high-low)) body dominance
  - stoch_k_14d: stochastic %K position in 14d range
  - obv_trend_20d: 20d slope of OBV (volume-confirmed price trend)

Admission gate at h=10: |IC| >= 0.007 and |ICIR| >= 0.084, library corr < 0.5.
"""
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic, summarize)

H = 10
MIN_VALID = 8


def trend_eff(n=20):
    def fn(sym, close, volume):
        r = close.pct_change().abs()
        net = (close / close.shift(n) - 1.0).abs()
        path = r.rolling(n).sum()
        return (net / path).replace([np.inf, -np.inf], np.nan)
    return fn


def amihud(win=60):
    def fn(sym, close, volume):
        if volume is None or volume.dropna().empty:
            return None
        v = volume.astype(float).reindex(close.index)
        illiq = (close.pct_change().abs() / v).rolling(win).mean()
        return (-illiq).replace([np.inf, -np.inf], np.nan)  # high = liquid
    return fn


def vol_zscore(short=20, long=120):
    def fn(sym, close, volume):
        r = close.pct_change()
        vs = r.rolling(short).std()
        vl_mean = vs.rolling(long).mean()
        vl_std = vs.rolling(long).std()
        return ((vs - vl_mean) / vl_std).replace([np.inf, -np.inf], np.nan)
    return fn


def candle_body(win=20):
    def fn(sym, close, volume, ohlc=None):
        if ohlc is None:
            return None
        d = ohlc
        body = (d['close'] - d['open']).abs()
        rng = (d['high'] - d['low']).replace(0, np.nan)
        return (body / rng).rolling(win).mean()
    return fn


def stoch_k(n=14):
    def fn(sym, close, volume, ohlc=None):
        if ohlc is None:
            return None
        d = ohlc
        ll = d['low'].rolling(n).min()
        hh = d['high'].rolling(n).max()
        return ((close - ll) / (hh - ll)).replace([np.inf, -np.inf], np.nan)
    return fn


def obv_trend(n=20):
    def fn(sym, close, volume):
        if volume is None or volume.dropna().empty:
            return None
        v = volume.astype(float).reindex(close.index)
        obv = (np.sign(close.diff()) * v).fillna(0.0).cumsum()
        return (obv / obv.shift(n) - 1.0).replace([np.inf, -np.inf], np.nan)
    return fn


def lib_frames(panel, ohlc):
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    lib = {
        'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
        'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
        'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    }
    # vix_beta_cond needs macro panel
    macro = panel['macro'].get('VIX')
    def vix_fn(sym, close, volume):
        if macro is None:
            return None
        grid = panel['grid']
        r_a = close.pct_change().reindex(grid)
        r_m = macro.pct_change().reindex(grid)
        beta = r_a.rolling(60, min_periods=30).cov(r_m) / r_m.rolling(60, min_periods=30).var()
        mm = (macro.reindex(grid) / macro.shift(20).reindex(grid) - 1.0)
        return (-beta * mm).replace([np.inf, -np.inf], np.nan)
    lib['vix_beta_cond_60x20'] = vix_fn
    out = {}
    for lbl, fn in lib.items():
        out[lbl] = factor_values(closes, volumes, grid, fn)
    return out


def max_lib_corr(fac, libs, cd):
    best = 0.0
    for lbl, lf in libs.items():
        cs = []
        for t in cd:
            x, y = fac.loc[t], lf.loc[t]
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= MIN_VALID:
                r = pd.Series(x[mask]).corr(pd.Series(y[mask]), method='spearman')
                if np.isfinite(r):
                    cs.append(r)
        if cs:
            best = max(best, abs(float(np.mean(cs))))
    return best


if __name__ == '__main__':
    panel = build_panel()
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    ret = forward_returns(closes, grid, H)

    # OHLC panel per asset from API
    from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
    WATCH = panel['closes'].keys()
    ohlc = {}
    for sym in WATCH:
        try:
            df = get_stock_daily_data(symbol=sym, days=4000)
        except Exception:
            df = get_index_daily_data(symbol=sym, days=4000)
        if df is None:
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= pd.Timestamp('2020-01-01')) & (df['date'] <= pd.Timestamp('2026-07-15'))]
        ohlc[sym] = df.set_index('date')[['open', 'high', 'low', 'close']]

    libs = lib_frames(panel, ohlc)
    cd = grid

    CANDIDATES = [
        ('trend_eff_20d', trend_eff(20), 'none'),
        ('trend_eff_60d', trend_eff(60), 'none'),
        ('amihud_60d', amihud(60), 'none'),
        ('vol_zscore_20_120', vol_zscore(20, 120), 'none'),
        ('candle_body_20d', candle_body(20), 'ohlc'),
        ('stoch_k_14d', stoch_k(14), 'ohlc'),
        ('obv_trend_20d', obv_trend(20), 'none'),
    ]

    print(f'=== BATCH-3b PRICE/VOLUME h={H} | gate |IC|>=0.007 |ICIR|>=0.084 corr<0.5 ===')
    for label, fn, needs in CANDIDATES:
        if needs == 'ohlc':
            fac = factor_values(closes, volumes, grid, lambda s, c, v: fn(s, c, v, ohlc=ohlc.get(s)))
        else:
            fac = factor_values(closes, volumes, grid, fn)
        cov = float(fac.notna().mean().mean())
        f10 = fac.iloc[::10]
        turn = float(f10.rank(axis=1).diff().abs().mean().mean()) if len(f10) > 2 else np.nan
        ics = daily_ic(fac, ret, min_valid=MIN_VALID)
        m = summarize(ics, label, H)
        corr = max_lib_corr(fac, libs, cd)
        print(f'   cov={cov:.3f} turn={turn:.3f} max_lib_corr={corr:.3f}')
        if len(ics) > 0:
            s = ics['ic']
            print('   yearly:', {int(y): round(float(v), 4) for y, v in s.groupby(s.index.year).mean().items()})
