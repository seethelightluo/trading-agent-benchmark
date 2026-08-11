"""miner_1 2026-08-27 exploration batch A: novel conditional-beta and risk-structure factors.

Motivation: library already has strong plain betas (spx, cn10y, hs300) and conditional
betas (vix/dxy/eurusd). This batch explores UNTESTED signal exposures:
  - USDJPY conditional beta (risk-sentiment FX, observation-only)
  - US10Y-CN10Y yield-curve-slope conditional beta
  - CN10Y conditional beta (rates level x trend interaction)
  - Equal-weight commodity basket beta (XAU/COPPER/WTI)
  - Cross-sectional vol rank (asset vol vs universe median)
  - Drawdown depth 120d (depth, not duration)
  - Parkinson range/vol ratio (efficiency of range information)
  - Volume skew 20d
  - Downside beta vs HS300 (China risk-off)
  - Crypto basket beta (BTC/ETH EW)
  - Momentum x efficiency-ratio interaction
  - Vol-normalized skew 20d

Validation battery: shared factor_common (IC/ICIR at 10d horizon, coverage,
turnover, decay, max library correlation). Admission gate |IC|>=0.007, |ICIR|>=0.084.
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, build_library_panels,
                           max_library_correlation, WATCHLIST)

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")
usdjpy = load_index('USDJPY', prices=prices)
print(f"USDJPY len={0 if usdjpy is None else len(usdjpy)}")

# observation signals (returns)
def sig_ret(sig, name):
    if sig is None:
        return None
    s = sig['close'].copy()
    return s.pct_change().rename(name)

usdjpy_r = sig_ret(usdjpy, 'usdjpy')

# tradable yield series
us10y = prices.get('US10Y')
cn10y = prices.get('CN10Y')
us10y_r = us10y['close'].pct_change().rename('us10y') if us10y is not None else None
cn10y_r = cn10y['close'].pct_change().rename('cn10y') if cn10y is not None else None
# spread = US10Y close - CN10Y close (yield differential level)
if us10y is not None and cn10y is not None:
    spread = (us10y['close'] - cn10y['close']).rename('spread')
    spread_r = spread.pct_change().rename('spread_r')
else:
    spread = spread_r = None

# equal-weight baskets
def ew_ret(symbols, prices):
    df = None
    for s in symbols:
        r = prices[s]['close'].pct_change().rename(s)
        df = r if df is None else pd.concat([df, r], axis=1)
    return df.mean(axis=1).rename('ew')

comm_r = ew_ret(['XAU', 'COPPER', 'WTI'], prices)
crypto_r = ew_ret(['BTC', 'ETH'], prices)
hs300_r = prices['000300.SH']['close'].pct_change().rename('hs300') if '000300.SH' in prices else None

# ---------- candidate factor functions ----------

def f_usdjpy_beta_cond_60x20(df, s):
    if usdjpy_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), usdjpy_r.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    return (b * y_move).reindex(z.index)

def f_spread_cond_60x20(df, s):
    if spread_r is None or spread is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spread_r.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = spread / spread.shift(20) - 1.0
    return (b * y_move).reindex(z.index)

def f_cn10y_beta_cond_60x20(df, s):
    if cn10y_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), cn10y_r.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = cn10y['close'] / cn10y['close'].shift(20) - 1.0
    return (b * y_move).reindex(z.index)

def f_comm_basket_beta_60(df, s):
    if comm_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), comm_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_cross_sec_vol_rank_20(df, s):
    v = df['close'].pct_change().rolling(20).std()
    # cross-sectional median across the universe panel is applied after; here
    # return per-asset vol; the panel-level normalization is done in the wrapper
    return v

def f_dd_depth_120(df, s):
    c = df['close']
    peak = c.rolling(120, min_periods=20).max()
    return (c / peak - 1.0)

def f_range_vol_ratio_20(df, s):
    rng = ((df['high'] - df['low']) / df['close']).rolling(20).mean()
    vol = df['close'].pct_change().rolling(20).std()
    return rng / vol

def f_volume_skew_20(df, s):
    if 'volume' not in df.columns or df['volume'].isna().all():
        return None
    lv = np.log(df['volume'].replace(0, np.nan))
    return lv.rolling(20).skew()

def f_down_beta_hs300_60(df, s):
    if hs300_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), hs300_r.rename('y')], axis=1).dropna()
    z = z[z['y'] < 0]
    if len(z) < 30:
        return pd.Series(np.nan, index=df.index)
    return (z['r'].rolling(60, min_periods=20).cov(z['y']) / z['y'].rolling(60, min_periods=20).var()).reindex(df.index)

def f_crypto_basket_beta_60(df, s):
    if crypto_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), crypto_r.rename('y')], axis=1).dropna()
    return (z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()).reindex(z.index)

def f_mom_x_er_20(df, s):
    c = df['close']
    mom = c / c.shift(20) - 1.0
    path = c.diff().abs().rolling(20).sum()
    net = (c - c.shift(20)).abs()
    er = net / path
    return mom * er

def f_vol_adj_skew_20(df, s):
    r = df['close'].pct_change()
    sk = r.rolling(20).skew()
    v = r.rolling(20).std()
    return sk / v

def f_cs_vol_rank_panel(panel):
    """Normalize each asset's 20d vol by the cross-sectional median vol per date."""
    med = panel.median(axis=1)
    out = panel.div(med, axis=0)
    return out

# ---------- evaluate ----------
lib = build_library_panels(prices)
print("library panels built:", list(lib.keys()))

candidates = [
    ('usdjpy_beta_cond_60x20', f_usdjpy_beta_cond_60x20),
    ('us10y_cn10y_spread_cond_60x20', f_spread_cond_60x20),
    ('cn10y_beta_cond_60x20', f_cn10y_beta_cond_60x20),
    ('comm_basket_beta_60', f_comm_basket_beta_60),
    ('dd_depth_120', f_dd_depth_120),
    ('range_vol_ratio_20', f_range_vol_ratio_20),
    ('volume_skew_20', f_volume_skew_20),
    ('down_beta_hs300_60', f_down_beta_hs300_60),
    ('crypto_basket_beta_60', f_crypto_basket_beta_60),
    ('mom_x_er_20', f_mom_x_er_20),
    ('vol_adj_skew_20', f_vol_adj_skew_20),
]

results = {}
for fid, fn in candidates:
    try:
        panel = factor_to_panel(fn, prices)
        if fid == 'cross_sec_vol_rank_20':
            panel = f_cs_vol_rank_panel(panel)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: insufficient -> None")
            results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
            continue
        rho, rho_id = max_library_correlation(panel, lib)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"--- {fid}: panel {panel.shape} | IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
              f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
              f"turn={m['turnover_10d_rank']:.3f} rho={rho:.3f}({rho_id}) -> {'PASS' if ok else 'FAIL'}")
        print("    decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
        results[fid] = {'ok': ok, 'metrics': m}
    except Exception as e:
        print(f"{fid}: ERROR {e}")
        results[fid] = {'ok': False, 'metrics': {'error': str(e)}}

with open('scripts/miner_1_20260827_results_batchA.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE. Saved scripts/miner_1_20260827_results_batchA.json")
