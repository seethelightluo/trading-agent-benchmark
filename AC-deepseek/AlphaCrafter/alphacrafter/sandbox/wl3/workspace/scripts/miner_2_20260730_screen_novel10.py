"""miner_2 2026-07-30 round: novel factor screen (9 ideas).

Focus: constructs NOT already in the 12-factor library and not rejected before:
 1. amihud_illiq_20          : Amihud illiquidity (|ret|/volume, 20d mean)
 2. lower_wick_10            : lower-wick position (close-low)/(high-low), 10d mean
 3. volume_flow_imb_20       : signed volume-flow imbalance over 20d
 4. gk_vol_ratio_20x60       : Garman-Klass vol term ratio 20d/60d (OHLC-based)
 5. kurt_60                  : excess kurtosis of daily returns, 60d
 6. usdjpy_beta_cond_60x20   : beta(asset,USDJPY chg,60d) * USDJPY 20d move
 7. btc_corr_60              : 60d correlation with BTC returns (crypto channel)
 8. rsi_60                   : RSI over 60d
 9. uscn_spread_beta_cond_60x20 : beta(asset, US10Y-CN10Y spread chg) * spread 20d move

Correlation audit is done against ALL 12 effective library factors using their
real .npy signal artifacts on the canonical grid.
"""
import sys, json, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, VAL_START, VAL_END, WATCHLIST)

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {len(grid)} dates {grid.min().date()}..{grid.max().date()}")

# ---- load library artifacts (12 effective factors) for correlation audit ----
EFFECTIVE_IDS = ['dd_duration_120_resid','down_beta_60','dxy_beta_cond_60x20','eurusd_beta_cond_60x20',
                 'hilo_pos_60','hs300_beta_60','max_ret_20d','skew_term_20_60','spx_beta_60',
                 'vix_beta_cond_60x20','vol_adj_mom_20_60','vol_of_vol20x60']
lib_mats = {}
for fid in EFFECTIVE_IDS:
    p = Path('factors') / f'{fid}_signal.npy'
    if p.exists():
        arr = np.load(p, allow_pickle=False)
        lib_mats[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
print(f"library artifacts loaded: {len(lib_mats)}")

def lib_max_corr(panel):
    """mean daily cross-sectional Spearman |rho| vs each library artifact."""
    best, best_id = 0.0, None
    for fid, lp in lib_mats.items():
        idx = panel.index.intersection(lp.index)
        corrs = []
        for d in idx:
            x, y = panel.loc[d], lp.loc[d]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---- index signals ----
usdjpy = load_index('USDJPY', prices=prices)
us10y = prices.get('US10Y')
cn10y = prices.get('CN10Y')

# ---- candidate definitions ----
def amihud_illiq_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume']
    amihud = (r / v.replace(0, np.nan))
    return amihud.rolling(20).mean()

def lower_wick_10(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    lw = (df['close'] - df['low']) / rng
    return lw.rolling(10).mean()

def volume_flow_imb_20(df, s):
    r = df['close'].pct_change()
    v = df['volume']
    num = (np.sign(r) * v).rolling(20).sum()
    den = v.rolling(20).sum()
    return (num / den.replace(0, np.nan))

def gk_vol_ratio_20x60(df, s):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    hl = np.log(h / l).pow(2)
    co = np.log(c / o).pow(2)
    gk = np.sqrt(0.5 * hl - (2 * np.log(2) - 1) * co)
    v20 = gk.rolling(20).mean()
    v60 = gk.rolling(60).mean()
    return (v20 / v60.replace(0, np.nan))

def kurt_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).kurt()

def usdjpy_beta_cond_60x20(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    rd = usdjpy['close'].pct_change()
    z = pd.concat([r.rename('r'), rd.rename('d')], axis=1, sort=False).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
    d_move = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    return (b * d_move).reindex(z.index)

def btc_corr_60(df, s):
    btc = prices.get('BTC')
    if btc is None:
        return None
    r = df['close'].pct_change()
    rb = btc['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1, sort=False).dropna()
    return z['r'].rolling(60).corr(z['b'])

def rsi_60(df, s):
    d = df['close'].diff()
    up = d.clip(lower=0).rolling(60).mean()
    dn = (-d.clip(upper=0)).rolling(60).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def uscn_spread_beta_cond_60x20(df, s):
    if us10y is None or cn10y is None:
        return None
    spread = us10y['close'] - cn10y['close']
    r = df['close'].pct_change()
    ds = spread.diff()
    z = pd.concat([r.rename('r'), ds.rename('s')], axis=1, sort=False).dropna()
    b = z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()
    s_move = spread / spread.shift(20) - 1.0
    return (b * s_move).reindex(z.index)

candidates = {
    'amihud_illiq_20': amihud_illiq_20,
    'lower_wick_10': lower_wick_10,
    'volume_flow_imb_20': volume_flow_imb_20,
    'gk_vol_ratio_20x60': gk_vol_ratio_20x60,
    'kurt_60': kurt_60,
    'usdjpy_beta_cond_60x20': usdjpy_beta_cond_60x20,
    'btc_corr_60': btc_corr_60,
    'rsi_60': rsi_60,
    'uscn_spread_beta_cond_60x20': uscn_spread_beta_cond_60x20,
}

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel.shape[0] < 100:
            print(f"{fid}: insufficient panel {panel.shape} -> skip", flush=True)
            continue
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        rho, rid = lib_max_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rid
        ok_ic = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        ok_corr = rho < 0.5
        print(f"{fid}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rid}) "
              f"decay1={m['decay_ic_by_horizon']['1']:+.4f} decay5={m['decay_ic_by_horizon']['5']:+.4f} "
              f"decay10={m['decay_ic_by_horizon']['10']:+.4f} decay20={m['decay_ic_by_horizon']['20']:+.4f} "
              f"-> {'PASS' if (ok_ic and ok_corr) else 'skip'} [{time.time()-t1:.1f}s]", flush=True)
        results[fid] = (m, panel)
    except Exception as e:
        print(f"{fid}: ERROR {type(e).__name__}: {e}", flush=True)

print(f"\nTOTAL {time.time()-t0:.1f}s")
print('SUMMARY:')
for fid, (m, _) in sorted(results.items()):
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5
    print(f"  {fid:28s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} rho={m['max_abs_library_correlation']:.3f} -> {'ADMIT' if ok else 'skip'}")
