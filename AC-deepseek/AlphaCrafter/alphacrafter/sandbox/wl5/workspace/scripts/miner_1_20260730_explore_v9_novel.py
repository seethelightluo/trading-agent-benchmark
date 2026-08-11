"""miner_1 cycle 2026-07-30 (v9): novel structurally-distinct factor candidates.

Focus: dimensions NOT already covered by the library (trend_r2, semi-vol ratio,
mom 10/120, DXY/VIX/WTI betas, time_under_water, vol-of-vol).
Candidates use raw ingredients the library does not use:
  - 4th moment (kurtosis)                -> kurt_60
  - conditional correlation asymmetry   -> down_corr_asym_60
  - high/low range data                 -> range_vol_ratio_20
  - order statistics (max daily ret)    -> max_ret_60 (lottery/MAX effect)
  - volatility term-structure slope     -> vol_ratio_20x60
  - momentum acceleration gap           -> mom_accel_20x120
All research restricted to visible window <= 2026-07-29.
"""
import json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split)

VIS = '2026-07-29'
H = 10
t0 = time.time()
close = closes_panel(VIS)
ret = close.pct_change()
lp = np.log(close)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS} load={time.time()-t0:.1f}s", flush=True)


def fast_ic_series(factor, fwd_ret, min_valid=8):
    """Spearman rank IC per date, vectorized via ranks."""
    f_rank = factor.rank(axis=1)
    r_rank = fwd_ret.rank(axis=1)
    fa = f_rank.values
    ra = r_rank.values
    idx = factor.index
    out = {}
    for i in range(len(factor)):
        frow = fa[i]
        rrow = ra[i]
        mask = ~(np.isnan(frow) | np.isnan(rrow))
        n = int(mask.sum())
        if n < min_valid:
            continue
        fv = frow[mask]
        rv = rrow[mask]
        if len(np.unique(fv)) < 3 or len(np.unique(rv)) < 2:
            continue
        fm = fv - fv.mean()
        rm = rv - rv.mean()
        denom = np.sqrt((fm * fm).sum() * (rm * rm).sum())
        if denom <= 1e-15:
            continue
        ic = float((fm * rm).sum() / denom)
        if np.isfinite(ic):
            out[idx[i]] = ic
    return pd.Series(out, dtype=float)


def rolling_beta(asset_ret, mkt_ret, win, mp=36):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename('a'), mkt_ret.rename('m')], axis=1).dropna()
        b = pair['a'].rolling(win, min_periods=mp).cov(pair['m']) / pair['m'].rolling(win, min_periods=mp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


# ---------------- candidate signals ----------------
# 1) kurt_60: rolling excess kurtosis of daily returns (tail fatness)
mu = ret.rolling(60, min_periods=36).mean()
m2 = ((ret - mu) ** 2).rolling(60, min_periods=36).mean()
m4 = ((ret - mu) ** 4).rolling(60, min_periods=36).mean()
kurt_60 = m4 / (m2 ** 2) - 3.0

# 2) down_corr_asym_60: corr(asset, EW market | market down) - corr(asset, EW | market up)
ew_ret = ret.mean(axis=1)
down_mask = (ew_ret < 0)
up_mask = (ew_ret > 0)
down_corr = {}
up_corr = {}
for a in ret.columns:
    pair = pd.concat([ret[a].rename('a'), ew_ret.rename('m')], axis=1).dropna()
    cd = pair[pair['m'] < 0]['a'].rolling(60, min_periods=18).corr(pair[pair['m'] < 0]['m'])
    cu = pair[pair['m'] > 0]['a'].rolling(60, min_periods=18).corr(pair[pair['m'] > 0]['m'])
    down_corr[a] = cd
    up_corr[a] = cu
down_corr = pd.DataFrame(down_corr).reindex(ret.index)
up_corr = pd.DataFrame(up_corr).reindex(ret.index)
down_corr_asym_60 = down_corr - up_corr

# 3) range_vol_ratio_20: avg daily (high-low)/close over 20d vs std of close-to-close ret 20d
hl = close.copy()
for c in close.columns:
    pass
# build high/low panels from raw csv
import os
def ohlc_panel(symbols, vis):
    out = {}
    for s in symbols:
        fp = os.path.join('../persistent/stock_data', s + '.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=['date'])
        df = df[df['date'] <= pd.Timestamp(vis)].set_index('date')
        out[s] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return pd.concat(out, axis=1)
ohlc = ohlc_panel(close.columns, VIS)
hi = ohlc.xs('high', axis=1, level=1)
lo = ohlc.xs('low', axis=1, level=1)
rng = (hi - lo) / close
avg_range_20 = rng.rolling(20, min_periods=12).mean()
vol20 = ret.rolling(20, min_periods=12).std()
range_vol_ratio_20 = avg_range_20 / vol20

# 4) max_ret_60: max single-day return over 60d (lottery/MAX effect)
max_ret_60 = ret.rolling(60, min_periods=36).max()

# 5) vol_ratio_20x60: short vol / long vol - 1 (vol term-structure slope)
vol20b = ret.rolling(20, min_periods=12).std()
vol60 = ret.rolling(60, min_periods=36).std()
vol_ratio_20x60 = vol20b / vol60 - 1.0

# 6) mom_accel_20x120: short-term trend gap vs long-term trend (acceleration)
mom20 = lp.diff(20)
mom120 = lp.diff(120)
mom_accel_20x120 = mom20 - mom120

# 7) drawdown_depth_ratio_60: current depth vs 60d range (position within drawdown cycle)
roll_max60 = close.rolling(60, min_periods=36).max()
roll_min60 = close.rolling(60, min_periods=36).min()
dd_pos_60 = (close - roll_min60) / (roll_max60 - roll_min60)  # 1 = at high, 0 = at low

cands = {
    'kurt_60': kurt_60,
    'down_corr_asym_60': down_corr_asym_60,
    'range_vol_ratio_20': range_vol_ratio_20,
    'max_ret_60': max_ret_60,
    'vol_ratio_20x60': vol_ratio_20x60,
    'mom_accel_20x120': mom_accel_20x120,
    'dd_pos_60': dd_pos_60,
}

fr = forward_returns(close, H)
t0 = time.time()
lib = library_ic_series_map(close, h=H)
print(f"library_ic_series_map: {time.time()-t0:.1f}s lib_size={len(lib)}", flush=True)

results = {}
panels_dir = 'scripts/_panels'
for fid, sig in cands.items():
    t0 = time.time()
    ic_s = fast_ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    dt = time.time() - t0
    if m is None:
        print(f"{fid}: INSUFFICIENT ({len(ic_s)} ic dates) [{dt:.1f}s]", flush=True)
        results[fid] = {"gate_pass": False, "reason": "insufficient IC dates",
                        "n_ic_dates": len(ic_s), "valid_entries": int(sig.notna().sum().sum())}
        continue
    m['max_abs_library_correlation'] = max_abs_library_corr(ic_s, lib)
    m['regime'] = regime_split(ic_s)
    gate = abs(m['ic']) >= 0.007 and abs(m['icir'] or 0) >= 0.084
    m['gate_pass'] = bool(gate)
    results[fid] = m
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} max_rho_lib={m['max_abs_library_correlation']} GATE={gate} [{dt:.1f}s]", flush=True)
    print("  decay:", m['decay_ic_by_horizon'], flush=True)
    print("  regimes:", m['regime'], flush=True)
    if gate:
        sig.index = sig.index.strftime('%Y-%m-%d')
        sig.to_csv(f"{panels_dir}/{fid}.csv")

with open('scripts/miner_1_20260730_explore_v9_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE saved scripts/miner_1_20260730_explore_v9_results.json")
