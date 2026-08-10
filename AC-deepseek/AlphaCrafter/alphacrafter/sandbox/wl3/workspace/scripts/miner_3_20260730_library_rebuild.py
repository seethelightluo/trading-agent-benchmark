"""Rebuild library factor panels on the canonical grid.

Used for candidate auditing so every pairwise rho uses the same date-axis.
Does NOT overwrite persistent artifacts unless --write is passed.
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel)

np.seterr(all='ignore')

def build_library_panels(prices, vix=None, dxy=None, eurusd=None):
    out = {}

    # 1. dd_duration_120_resid
    def f_dd(df, s):
        hi120 = df['close'].rolling(120).max()
        mark = (df['close'] >= hi120).astype(int)
        days = mark.groupby((~mark.astype(bool)).cumsum()).cumcount() + 1
        mom = df['close'].shift(5) / df['close'].shift(125) - 1.0
        ref = pd.concat([np.log1p(days.rename('d')), mom.rename('m')], axis=1).dropna()
        if len(ref) < 60:
            return None
        z = pd.concat([np.log1p(days.rename('d')), mom.rename('m')], axis=1)
        b = z['d'].rolling(60).cov(z['m']) / z['m'].rolling(60).var()
        return (np.log1p(days) - b * (mom - mom.rolling(60).mean()) / mom.rolling(60).std()).reindex(z.index)
    out['dd_duration_120_resid'] = factor_to_panel(f_dd, prices)

    # 2. down_beta_60
    spx = prices['SPX']['close']
    def f_downbeta(df, s):
        r = df['close'].pct_change()
        rs = spx.reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
        z = z[z['s'] < 0]
        if len(z) < 30:
            return pd.Series(np.nan, index=df.index)
        b = z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()
        return b
    out['down_beta_60'] = factor_to_panel(f_downbeta, prices)

    # 3. spx_beta_60
    def f_spxbeta(df, s):
        r = df['close'].pct_change()
        rs = spx.reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
        return z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()
    out['spx_beta_60'] = factor_to_panel(f_spxbeta, prices)

    # 4. hs300_beta_60
    hs300 = prices['000300.SH']['close']
    def f_hsbeta(df, s):
        r = df['close'].pct_change()
        rs = hs300.reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
        return z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()
    out['hs300_beta_60'] = factor_to_panel(f_hsbeta, prices)

    # 5. dxy_beta_cond_60x20
    def f_dxy(df, s):
        if dxy is None:
            return None
        r = df['close'].pct_change()
        rd = dxy['close'].reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rd.rename('d')], axis=1).dropna()
        beta = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
        mom_d = dxy['close'].reindex(z.index) / dxy['close'].reindex(z.index).shift(20) - 1.0
        return (beta * mom_d).reindex(z.index)
    out['dxy_beta_cond_60x20'] = factor_to_panel(f_dxy, prices)

    # 6. eurusd_beta_cond_60x20
    def f_eur(df, s):
        if eurusd is None:
            return None
        r = df['close'].pct_change()
        re = eurusd['close'].reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), re.rename('e')], axis=1).dropna()
        beta = z['r'].rolling(60).cov(z['e']) / z['e'].rolling(60).var()
        mom_e = eurusd['close'].reindex(z.index) / eurusd['close'].reindex(z.index).shift(20) - 1.0
        return (beta * mom_e).reindex(z.index)
    out['eurusd_beta_cond_60x20'] = factor_to_panel(f_eur, prices)

    # 7. vix_beta_cond_60x20
    def f_vix(df, s):
        if vix is None:
            return None
        r = df['close'].pct_change()
        rv = vix['close'].reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rv.rename('v')], axis=1).dropna()
        beta = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
        mom_v = vix['close'].reindex(z.index) / vix['close'].reindex(z.index).shift(20) - 1.0
        return (-beta * mom_v).reindex(z.index)
    out['vix_beta_cond_60x20'] = factor_to_panel(f_vix, prices)

    # 8. max_ret_20d
    def f_maxr(df, s):
        return df['close'].pct_change().rolling(20).max()
    out['max_ret_20d'] = factor_to_panel(f_maxr, prices)

    # 9. vol_adj_mom_20_60
    def f_vam(df, s):
        mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
        v = df['close'].pct_change().rolling(60).std()
        return mom / v
    out['vol_adj_mom_20_60'] = factor_to_panel(f_vam, prices)

    # 10. vol_of_vol20x60
    def f_vov(df, s):
        return df['close'].pct_change().rolling(20).std().rolling(60).std()
    out['vol_of_vol20x60'] = factor_to_panel(f_vov, prices)

    # 11. skew_term_20_60
    def f_skew(df, s):
        r = df['close'].pct_change()
        return r.rolling(20).skew() - r.rolling(60).skew()
    out['skew_term_20_60'] = factor_to_panel(f_skew, prices)

    # 12. hilo_pos_60
    def f_hilo(df, s):
        hi = df['high'].rolling(60).max()
        lo = df['low'].rolling(60).min()
        return (df['close'] - lo) / (hi - lo)
    out['hilo_pos_60'] = factor_to_panel(f_hilo, prices)

    return out


if __name__ == '__main__':
    prices = load_prices(days=2500)
    grid = canonical_grid(prices)
    vix = load_index('VIX', prices=prices)
    dxy = load_index('DXY', prices=prices)
    eurusd = load_index('EURUSD', prices=prices)
    print(f"grid {len(grid)} dates; VIX {False if vix is None else len(vix)}; "
          f"DXY {False if dxy is None else len(dxy)}; EURUSD {False if eurusd is None else len(eurusd)}")
    panels = build_library_panels(prices, vix, dxy, eurusd)
    write = '--write' in sys.argv
    for fid, p in panels.items():
        arr = signal_matrix(p, grid)
        print(f"{fid:28s} panel {p.shape}  matrix {arr.shape}  coverage {np.isfinite(arr).mean():.3f}")
        if write:
            np.save(Path('factors') / f'{fid}_signal.npy', arr)
    if write:
        print("wrote all artifacts to factors/")
