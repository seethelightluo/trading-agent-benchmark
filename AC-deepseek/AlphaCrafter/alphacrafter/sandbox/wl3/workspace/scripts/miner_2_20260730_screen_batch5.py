"""miner_2 2026-07-30 batch-5 screen: structural / macro-rate / idiosyncratic candidates.

Gate context (from factors/*.reason.json): trend family (vol_adj_mom_20_60) and beta
family (spx_beta_60) are crowded; new candidates must keep max_abs_library_correlation
< 0.5 against the CURRENT 11-factor effective library.

Candidates (all per-asset, cross-sectional, h=10 admission):
 1. overnight_ratio_20  : share of total 20d absolute move realized in overnight gaps
 2. parkinson_ratio_20  : range-based vol / close-to-close realized vol (20d)
 3. ret_skew_20         : skewness of 20d daily returns (lottery-demand)
 4. us10y_beta_60       : rolling 60d beta of asset returns to US10Y yield changes
 5. usdjpy_beta_cond_60x20 : 60d beta to USDJPY x 20d USDJPY trend (new macro driver)
 6. downside_beta_60_ndx   : downside beta vs NDX (full-series rolling, boolean weights)
 7. idio_mom_20_resid      : 20d cumulative residual return vs EW cross-market factor
 8. rel_strength_20        : (ret20 - cross-sectional median ret20) / cross-sectional MAD
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           validate_factor, max_library_correlation, canonical_grid,
                           signal_matrix, VAL_START, VAL_END)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"assets={len(prices)} grid_dates={len(grid)} grid={grid.min().date()}..{grid.max().date()}")

# ---------- rebuild current 11-factor effective library panels ----------
vix = load_index('VIX', prices=prices)
dxy = load_index('DXY', prices=prices)
eur = load_index('EURUSD', prices=prices)
jpy = load_index('USDJPY', prices=prices)

def lib_panels():
    out = {}
    def f_dd(df, s):
        r = (df['high'].rolling(120).max() - df['close']) / df['high'].rolling(120).max()
        dd = r.clip(lower=0)
        days = dd.groupby((dd == 0).cumsum()).cumcount() + 1
        days = days.where(dd > 0, 0)
        base = np.log1p(days)
        mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
        z = pd.concat([base.rename('b'), mom.rename('m')], axis=1)
        beta = z['b'].rolling(120).cov(z['m']) / z['m'].rolling(120).var()
        return (base - beta * (mom - mom.rolling(120).mean())).reindex(z.index)
    def f_dxy(df, s):
        if dxy is None: return None
        r = df['close'].pct_change(); rd = dxy['close'].pct_change()
        z = pd.concat([r.rename('r'), rd.rename('d')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
        return (b * (dxy['close'] / dxy['close'].shift(20) - 1.0)).reindex(z.index)
    def f_eur(df, s):
        if eur is None: return None
        r = df['close'].pct_change(); rd = eur['close'].pct_change()
        z = pd.concat([r.rename('r'), rd.rename('d')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
        return (b * (eur['close'] / eur['close'].shift(20) - 1.0)).reindex(z.index)
    def f_hilo(df, s):
        hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
        return ((df['close'] - lo) / (hi - lo)).reindex(df.index)
    def f_hs(df, s):
        ref = prices['000300.SH']['close'].pct_change()
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), ref.rename('d')], axis=1).dropna()
        return (z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()).reindex(z.index)
    def f_maxret(df, s): return df['close'].pct_change().rolling(20).max()
    def f_skewt(df, s):
        r = df['close'].pct_change()
        return (r.rolling(20).skew() - r.rolling(60).skew()).reindex(df.index)
    def f_spx(df, s):
        ref = prices['SPX']['close'].pct_change()
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), ref.rename('d')], axis=1).dropna()
        return (z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()).reindex(z.index)
    def f_vix(df, s):
        if vix is None: return None
        r = df['close'].pct_change(); vr = vix['close'].pct_change()
        z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
        return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
    def f_vam(df, s):
        mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
        vol = df['close'].pct_change().rolling(60).std()
        return (mom / vol).reindex(df.index)
    def f_vov(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
    for fid, fn in [('dd_duration_120_resid', f_dd), ('dxy_beta_cond_60x20', f_dxy),
                    ('eurusd_beta_cond_60x20', f_eur), ('hilo_pos_60', f_hilo),
                    ('hs300_beta_60', f_hs), ('max_ret_20d', f_maxret),
                    ('skew_term_20_60', f_skewt), ('spx_beta_60', f_spx),
                    ('vix_beta_cond_60x20', f_vix), ('vol_adj_mom_20_60', f_vam),
                    ('vol_of_vol20x60', f_vov)]:
        p = factor_to_panel(fn, prices)
        out[fid] = p
    return out

lib = lib_panels()
print("library panels rebuilt:", {k: v.shape for k, v in lib.items()})

# ---------- candidate factor functions ----------
def make_overnight_ratio(n=20):
    def f(df, s):
        o = (df['open'] / df['close'].shift(1) - 1.0).abs()
        i = (df['close'] / df['open'] - 1.0).abs()
        so = o.rolling(n).sum(); si = i.rolling(n).sum()
        return (so / (so + si)).reindex(df.index)
    return f

def make_parkinson_ratio(n=20):
    def f(df, s):
        rng = (df['high'] / df['low'] - 1.0)
        rv = df['close'].pct_change().rolling(n).std()
        return (rng.rolling(n).mean() / rv).reindex(df.index)
    return f

def make_ret_skew(n=20):
    def f(df, s): return df['close'].pct_change().rolling(n).skew().reindex(df.index)
    return f

def make_us10y_beta(n=60):
    def f(df, s):
        if s == 'US10Y':
            return None  # self-row degenerate
        ref = prices['US10Y']['close'].pct_change()
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), ref.rename('d')], axis=1).dropna()
        return (z['r'].rolling(n).cov(z['d']) / z['d'].rolling(n).var()).reindex(z.index)
    return f

def make_usdjpy_beta_cond(n=60, t=20):
    def f(df, s):
        if jpy is None: return None
        r = df['close'].pct_change(); rd = jpy['close'].pct_change()
        z = pd.concat([r.rename('r'), rd.rename('d')], axis=1).dropna()
        b = z['r'].rolling(n).cov(z['d']) / z['d'].rolling(n).var()
        return (b * (jpy['close'] / jpy['close'].shift(t) - 1.0)).reindex(z.index)
    return f

def make_downside_beta(ref_sym, n=60):
    def f(df, s):
        ref = prices[ref_sym]['close'].pct_change()
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), ref.rename('d')], axis=1).dropna()
        w = (z['d'] < 0).astype(float)
        mu_r = z['r'].rolling(n).mean(); mu_d = z['d'].rolling(n).mean()
        num = ((z['r'] - mu_r) * (z['d'] - mu_d) * w).rolling(n).sum()
        den = ((z['d'] - mu_d) ** 2 * w).rolling(n).sum()
        return (num / den).reindex(z.index)
    return f

def make_idio_mom(n=20, b=60):
    def f(df, s):
        r = df['close'].pct_change()
        panel = pd.DataFrame({k: v['close'].pct_change() for k, v in prices.items()})
        fct = panel.mean(axis=1)
        z = pd.concat([r.rename('r'), fct.rename('f')], axis=1).dropna()
        beta = z['r'].rolling(b).cov(z['f']) / z['f'].rolling(b).var()
        resid = (z['r'] - beta * z['f'])
        return resid.rolling(n).sum().reindex(z.index)
    return f

def make_rel_strength(n=20):
    def f(df, s):
        ret20 = df['close'] / df['close'].shift(n) - 1.0
        panel = pd.DataFrame({k: v['close'] / v['close'].shift(n) - 1.0 for k, v in prices.items()})
        med = panel.median(axis=1)
        mad = (panel.sub(med, axis=0)).abs().median(axis=1)
        return ((ret20 - med) / mad).reindex(df.index)
    return f

cands = [
    ('overnight_ratio_20', make_overnight_ratio(20), 'structural'),
    ('parkinson_ratio_20', make_parkinson_ratio(20), 'structural'),
    ('ret_skew_20', make_ret_skew(20), 'distribution'),
    ('us10y_beta_60', make_us10y_beta(60), 'macro-beta'),
    ('usdjpy_beta_cond_60x20', make_usdjpy_beta_cond(60, 20), 'macro-beta'),
    ('downside_beta_60_ndx', make_downside_beta('NDX', 60), 'beta-asymmetry'),
    ('idio_mom_20_resid', make_idio_mom(20, 60), 'idiosyncratic'),
    ('rel_strength_20', make_rel_strength(20), 'cross-sectional'),
]

results = {}
for fid, fn, fam in cands:
    panel = factor_to_panel(fn, prices)
    if panel.empty or len(panel) < 200:
        print(f"{fid}: panel too small -> skip")
        continue
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None")
        continue
    rho, rid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rid
    m['family'] = fam
    results[fid] = (m, panel)
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f"{fid} [{fam}]: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rid}) -> {'PASS' if ok else 'FAIL'}")

print("\n--- decay (best-of-family) ---")
for fid in ['overnight_ratio_20', 'ret_skew_20', 'us10y_beta_60', 'usdjpy_beta_cond_60x20',
            'downside_beta_60_ndx', 'idio_mom_20_resid', 'rel_strength_20']:
    if fid in results:
        m, _ = results[fid]
        print(fid, {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()})

# save screening results for reference
json.dump({fid: {k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}
           for fid, (m, _) in results.items()},
          open('scripts/miner_2_20260730_batch5_results.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_2_20260730_batch5_results.json")
