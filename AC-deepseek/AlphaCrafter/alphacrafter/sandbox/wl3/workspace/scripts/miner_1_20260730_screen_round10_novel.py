"""Round 10 (miner_1): novel factor candidates vs the 12-factor library artifacts.

Fresh ideas (single-idea constructs, not previously screened in the repo):
 1. usdjpy_beta_cond_60x20  - conditional beta to USDJPY * 20d USDJPY move (risk-on funding)
 2. usdcny_beta_cond_60x20  - conditional beta to USDCNY * 20d USDCNY move (China risk)
 3. us10y_beta_60           - plain rolling beta of asset returns to US10Y yield changes
 4. cn10y_beta_60           - plain rolling beta of asset returns to CN10Y yield changes
 5. down_vol_ratio_60       - downside vol / upside vol asymmetry (60d)
 6. low_wick_ratio_10       - lower wick / upper wick ratio (support vs supply pressure)
 7. ret_autocorr_20         - lag-1 daily return autocorrelation over 20d (persistence)
 8. vol_volume_20           - volume volatility: std(volume)/mean(volume) over 20d
 9. overnight_share_20      - gap-return share of total 2-day return variance
10. kurt_term_20_60         - excess-kurtosis term spread (20d vs 60d)
11. usdjpy_gate_mom_20      - 20d momentum gated by sign of 20d USDJPY move
12. hi_lo_vol_ratio_20      - range-based (Parkinson) vol / close-based vol ratio
13. drawdown_rec_120        - drawdown recovery speed proxy (120d)
14. vol_skew_60             - realized-vol skew: down-move vol contribution ratio

Admission gate: |IC|>=0.007, |ICIR|>=0.084, max library rho < 0.5 (artifact-based).
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor)

np.seterr(all='ignore')

prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})")

idx_sigs = {}
for nm in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    idx_sigs[nm] = load_index(nm, prices=prices)
    print(f"{nm} loaded: {idx_sigs[nm] is not None}, rows={0 if idx_sigs[nm] is None else len(idx_sigs[nm])}")

# ---------- library artifacts (real signal matrices) ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
    except Exception as e:
        print("lib skip", f, e)
print(f"library artifacts loaded: {len(lib)} -> {sorted(lib)}")


def max_lib_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        if la.shape[0] < arr.shape[0]:
            arr_use = arr[-la.shape[0]:]
        else:
            arr_use = arr
        corrs = []
        for i in range(arr_use.shape[0]):
            x, y = arr_use[i], la[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def rolling_beta(r_asset, r_sig, win=60):
    b = r_asset.rolling(win).cov(r_sig) / r_sig.rolling(win).var().replace(0, np.nan)
    return b


def f_usdjpy_beta_cond(df, s):
    sig = idx_sigs['USDJPY']
    if sig is None:
        return None
    r = df['close'].pct_change()
    sr = sig['close'].pct_change()
    z = pd.concat([r.rename('r'), sr.rename('s')], axis=1).dropna()
    b = rolling_beta(z['r'], z['s'], 60)
    mv = sig['close'] / sig['close'].shift(20) - 1.0
    return (b * mv).reindex(z.index)


def f_usdcny_beta_cond(df, s):
    sig = idx_sigs['USDCNY']
    if sig is None:
        return None
    r = df['close'].pct_change()
    sr = sig['close'].pct_change()
    z = pd.concat([r.rename('r'), sr.rename('s')], axis=1).dropna()
    b = rolling_beta(z['r'], z['s'], 60)
    mv = sig['close'] / sig['close'].shift(20) - 1.0
    return (b * mv).reindex(z.index)


def f_us10y_beta(df, s):
    u = prices['US10Y']['close'].pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), u.rename('u')], axis=1).dropna()
    return rolling_beta(z['r'], z['u'], 60).reindex(z.index)


def f_cn10y_beta(df, s):
    u = prices['CN10Y']['close'].pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), u.rename('u')], axis=1).dropna()
    return rolling_beta(z['r'], z['u'], 60).reindex(z.index)


def f_down_vol_ratio(df, s):
    r = df['close'].pct_change()
    dn = r[r < 0].rolling(60).std()
    up = r[r > 0].rolling(60).std()
    return (dn / up.replace(0, np.nan)).reindex(df.index)


def f_low_wick_ratio(df, s):
    lo_w = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    hi_w = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    mlo = lo_w.rolling(10).mean()
    mhi = hi_w.rolling(10).mean()
    return (mlo / mhi.replace(0, np.nan)).reindex(df.index)


def f_ret_autocorr(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False)


def f_vol_volume(df, s):
    v = df['volume'].replace(0, np.nan)
    return (v.rolling(20).std() / v.rolling(20).mean().replace(0, np.nan)).reindex(df.index)


def f_overnight_share(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    intr = df['close'] / df['open'] - 1.0
    two = df['close'] / df['close'].shift(2) - 1.0
    vg = gap.rolling(20).var()
    vi = intr.rolling(20).var()
    vt = two.rolling(20).var()
    return (vg / vt.replace(0, np.nan)).reindex(df.index)


def f_kurt_term(df, s):
    r = df['close'].pct_change()
    k20 = r.rolling(20).kurt()
    k60 = r.rolling(60).kurt()
    return (k20 - k60).reindex(df.index)


def f_usdjpy_gate_mom(df, s):
    sig = idx_sigs['USDJPY']
    if sig is None:
        return None
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    sm = sig['close'].shift(5) / sig['close'].shift(25) - 1.0
    g = np.sign(sm).reindex(df.index).fillna(0.0)
    return mom * g


def f_hi_lo_vol_ratio(df, s):
    r = df['close'].pct_change()
    cv = r.rolling(20).std()
    pv = (np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))).rolling(20).mean() ** 0.5
    return (pv / cv.replace(0, np.nan)).reindex(df.index)


def f_drawdown_rec(df, s):
    c = df['close']
    roll_max = c.rolling(120, min_periods=20).max()
    dd = c / roll_max - 1.0
    # recovery speed: rate of climb out of drawdown over last 10d (positive=recovering)
    rec = dd - dd.shift(10)
    return rec


def f_vol_skew_60(df, s):
    r = df['close'].pct_change()
    dn = r[r < 0].rolling(60).std()
    tot = r.rolling(60).std()
    return (dn / tot.replace(0, np.nan)).reindex(df.index)


candidates = {
    'usdjpy_beta_cond_60x20': f_usdjpy_beta_cond,
    'usdcny_beta_cond_60x20': f_usdcny_beta_cond,
    'us10y_beta_60': f_us10y_beta,
    'cn10y_beta_60': f_cn10y_beta,
    'down_vol_ratio_60': f_down_vol_ratio,
    'low_wick_ratio_10': f_low_wick_ratio,
    'ret_autocorr_20': f_ret_autocorr,
    'vol_volume_20': f_vol_volume,
    'overnight_share_20': f_overnight_share,
    'kurt_term_20_60': f_kurt_term,
    'usdjpy_gate_mom_20': f_usdjpy_gate_mom,
    'hi_lo_vol_ratio_20': f_hi_lo_vol_ratio,
    'drawdown_rec_120': f_drawdown_rec,
    'vol_skew_60': f_vol_skew_60,
}

results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT DATA (panel {panel.shape})")
        continue
    rho, fid_lib = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid_lib
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = dict(ok=ok, metrics=m)
    print(f"\n=== {fid} === panel {panel.shape} "
          f"range {panel.index.min().date()}..{panel.index.max().date()}")
    print(f"IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"maxlibrho={rho:.3f}({fid_lib})")
    print("decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f} {'PASS' if abs(m['ic'])>=0.007 else 'FAIL'} | "
          f"|ICIR|={abs(m['icir']):.4f} {'PASS' if abs(m['icir'])>=0.084 else 'FAIL'} | "
          f"rho={rho:.3f} {'PASS' if rho<0.5 else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")

json.dump({k: v for k, v in results.items()},
          open('scripts/miner_1_20260730_results_round10.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_1_20260730_results_round10.json")
