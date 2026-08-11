"""Round 26 (2027-02-25) novel factor screen for the 15-asset cross-asset universe.

Novel candidates vs the 20-factor EFFECTIVE library and prior rejected sets:
 1. us10y_beta_60        : rolling 60d beta of asset returns vs US10Y yield change (US rates beta)
 2. btc_beta_60          : rolling 60d beta of asset returns vs BTC returns (crypto risk exposure)
 3. close_loc_20         : mean((close-low)/(high-low)) over 20d (intraday close-location pressure)
 4. overnight_mom_20     : 20d sum of overnight returns open/prev_close-1 (overnight momentum)
 5. kurt_20              : rolling 20d excess kurtosis of daily returns (fat-tail risk)
 6. downup_vol_ratio_20  : std(neg rets)/std(pos rets) over 20d (volatility asymmetry)
 7. usdjpy_beta_60       : rolling 60d beta vs USDJPY change (carry/risk proxy)
 8. max_dd_60            : 60d max drawdown depth  rolling_min(close,60)/rolling_max(close,60)-1
 9. wick_ratio_20        : mean((upper_wick-lower_wick)/range) over 20d (candle rejection asymmetry)
10. recovery_dur_60      : log1p(days since 60d low) (recovery duration, complement of dd_duration)

Gate (benchmark-wide): |IC10| >= 0.007, |ICIR10| >= 0.084 on warm-up
2020-01-01..2026-07-15, max_abs_library_correlation < 0.5 vs 20 library
factors computed from REAL persisted signal artifacts on the canonical grid.
Additional out-of-sample robustness: IC10 on 2026-07-16..2027-02-24.
"""
import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series, signal_matrix,
                           VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2500)
print(f"assets loaded: {len(prices)} | {time.time()-t0:.1f}s", flush=True)

# ---------------- macro / market inputs ----------------
spx_r = prices['SPX']['close'].pct_change()
btc_r = prices['BTC']['close'].pct_change()
us10y_c = prices['US10Y']['close']
us10y_d = us10y_c.diff()          # yield change (level series)
jpy = load_index('USDJPY', prices=prices)
jpy_r = jpy['close'].pct_change() if jpy is not None else None


def rb(r, m, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)


# ---------------- candidate factor functions ----------------
def f_us10y_beta(df, s):
    r = df['close'].pct_change()
    return rb(r, us10y_d, 60)

def f_btc_beta(df, s):
    r = df['close'].pct_change()
    return rb(r, btc_r, 60)

def f_close_loc(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    loc = (df['close'] - df['low']) / rng
    return loc.rolling(20, min_periods=10).mean()

def f_overnight_mom(df, s):
    ov = df['open'] / df['close'].shift(1) - 1.0
    return ov.rolling(20, min_periods=10).sum()

def f_kurt(df, s):
    r = df['close'].pct_change()
    return r.rolling(20, min_periods=10).kurt()

def f_downup_vol(df, s):
    r = df['close'].pct_change()
    pos = r.where(r > 0)
    neg = r.where(r < 0)
    sp = pos.rolling(20, min_periods=8).std()
    sn = neg.rolling(20, min_periods=8).std()
    return (sn / sp).replace([np.inf, -np.inf], np.nan)

def f_usdjpy_beta(df, s):
    if jpy_r is None:
        return None
    r = df['close'].pct_change()
    return rb(r, jpy_r, 60)

def f_max_dd(df, s):
    c = df['close']
    return (c.rolling(60, min_periods=30).min() / c.rolling(60, min_periods=30).max() - 1.0)

def f_wick_ratio(df, s):
    hi, lo, op, cl = df['high'], df['low'], df['open'], df['close']
    rng = (hi - lo).replace(0, np.nan)
    uw = (hi - np.maximum(op, cl)) / rng
    lw = (np.minimum(op, cl) - lo) / rng
    return (uw - lw).rolling(20, min_periods=10).mean()

def f_recovery_dur(df, s):
    c = df['close'].values
    ll = df['close'].rolling(60, min_periods=30).min().values
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(ll[i]):
            continue
        k = 0
        while (i - k) >= 0 and c[i - k] > ll[i] and k < 300:
            k += 1
        out[i] = np.log1p(k)
    return pd.Series(out, index=df.index)

cands = {
    'us10y_beta_60':      (f_us10y_beta, 'US 10Y rates beta (60d)', 'beta/rates'),
    'btc_beta_60':        (f_btc_beta, 'BTC crypto beta (60d)', 'beta/crypto'),
    'close_loc_20':       (f_close_loc, 'intraday close location (20d)', 'microstructure'),
    'overnight_mom_20':   (f_overnight_mom, 'overnight return momentum (20d)', 'momentum'),
    'kurt_20':            (f_kurt, 'excess kurtosis of daily rets (20d)', 'volatility/tails'),
    'downup_vol_ratio_20': (f_downup_vol, 'downside/upside vol asymmetry (20d)', 'volatility/asymmetry'),
    'usdjpy_beta_60':     (f_usdjpy_beta, 'USDJPY beta (60d)', 'beta/fx'),
    'max_dd_60':          (f_max_dd, 'max drawdown depth (60d)', 'drawdown'),
    'wick_ratio_20':      (f_wick_ratio, 'upper-lower wick asymmetry (20d)', 'microstructure'),
    'recovery_dur_60':    (f_recovery_dur, 'days since 60d low (log1p)', 'drawdown/recovery'),
}

# ---------------- library correlation gate via REAL signal artifacts ----------------
import os
from pathlib import Path
lib_arts = {}
for p in sorted(Path('factors').glob('*.json')):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    fid = d.get('factor_id')
    art = d.get('signal_artifact')
    if not fid or not art or fid in ('factor_ensemble',):
        continue
    ap = Path('factors') / art
    if not ap.exists():
        continue
    g = d.get('signal_artifact_grid', {})
    arr = np.load(ap, allow_pickle=False)
    if arr.shape[1] != len(WATCHLIST):
        continue
    lib_arts[fid] = (arr, g)
print(f"library artifacts loaded: {len(lib_arts)}", flush=True)

# reconstruct canonical date index from API data within artifact grid range
_g0 = next(iter(lib_arts.values()))[1]
g_start, g_end, g_n = pd.Timestamp(_g0['start']), pd.Timestamp(_g0['end']), int(_g0['n_dates'])
all_dates = sorted(set().union(*[set(df.index) for df in prices.values()]))
grid_idx = pd.DatetimeIndex([d for d in all_dates if g_start <= d <= g_end])
print(f"artifact grid: {g_start.date()}..{g_end.date()} n={g_n} | reconstructed={len(grid_idx)}", flush=True)
assert len(grid_idx) == g_n, "grid mismatch - cannot align artifacts"

lib_panels = {}
for fid, (arr, g) in lib_arts.items():
    lib_panels[fid] = pd.DataFrame(arr, index=grid_idx, columns=WATCHLIST)

fwd = forward_returns(prices, 10)
ic10_full = rank_ic_series  # alias

def lib_corr(panel, grid_idx):
    """max abs mean daily Spearman corr vs library artifact panels."""
    best = 0.0; best_id = None; per = {}
    for fid, lp in lib_panels.items():
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
            per[fid] = r
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id, per

# ---------------- run validation ----------------
def validate(factor_panel):
    """Admission metrics on warm-up window + extended OOS stats."""
    ic = rank_ic_series(factor_panel, fwd, 8)
    ic10 = ic[(ic.index >= VAL_START) & (ic.index <= VAL_END)]
    if len(ic10) < 100:
        return None
    mean = float(ic10.mean()); sd = float(ic10.std(ddof=1))
    icir = mean / sd if sd > 0 else 0.0
    hit = float((ic10 > 0).mean()) if mean >= 0 else float((ic10 < 0).mean())
    fac = factor_panel[(factor_panel.index >= VAL_START) & (factor_panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum() / (fac.shape[0] * fac.shape[1])) if fac.shape[0] else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    # extended OOS (post warm-up)
    oos = ic[(ic.index >= pd.Timestamp('2026-07-16')) & (ic.index <= pd.Timestamp('2027-02-24'))]
    oos_ic = float(oos.mean()) if len(oos) > 20 else float('nan')
    oos_sd = float(oos.std(ddof=1)) if len(oos) > 20 else float('nan')
    oos_icir = oos_ic / oos_sd if oos_sd and oos_sd > 0 else float('nan')
    return {'ic': mean, 'icir': icir, 'ic_hit_ratio': hit,
            'n_ic_dates': int(len(ic10)), 'coverage_asset_days': cov,
            'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
            'oos_ic': oos_ic, 'oos_icir': oos_icir, 'n_oos_dates': int(len(oos))}

results = {}
for fid, (fn, desc, tag) in cands.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel is None or len(panel) == 0:
            print(f"{fid}: EMPTY panel -> skip", flush=True)
            continue
        m = validate(panel)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        rho, rho_id, per = lib_corr(panel, grid_idx)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        m['per_factor_rho'] = {k: round(v, 3) for k, v in sorted(per.items(), key=lambda kv: -abs(kv[1]))[:4]}
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = {'ok': bool(ok), 'metrics': m, 'desc': desc, 'tag': tag}
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id}) [{time.time()-t1:.1f}s]", flush=True)
        print(f"  top rho: {m['per_factor_rho']}", flush=True)
        print(f"  OOS(2026-07-16..2027-02-24): ic={m['oos_ic']:.4f} icir={m['oos_icir']:.4f} n={m['n_oos_dates']}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_3_20270225_results_round26.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:22s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')}) oos_ic={m.get('oos_ic', float('nan')):.4f}")
    else:
        print(f"{fid:22s} ERROR {r.get('error', '')[:70]}")
print(f"total time {time.time()-t0:.1f}s")
