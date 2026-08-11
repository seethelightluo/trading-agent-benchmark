"""miner_1 2026-10-08: screen novel non-beta factor batch B.

Prior state: coexceed_down_60 passed IC/ICIR but was gate-evicted for rho=0.64
with down_beta_60. updown_volume_* failed ICIR (see results json). This batch
focuses on price/OHLC-only constructions (all 15 assets covered) with low
expected overlap with the beta/regime library:

  1. min_ret_20         - worst single-day return over 20d            [tail resilience]
  2. vol_term_5_60      - vol(5d)/vol(60d) ratio                      [vol term structure]
  3. updown_vol_asym_20 - downside semidev / upside semidev (20d)     [vol asymmetry]
  4. btc_tail_coexceed_60 - joint co-crash with BTC (60d)             [crypto tail]
  5. range_trend_20_60  - avg daily range 20d / 60d                   [range expansion]
  6. realized_kurt_60   - excess kurtosis 60d (batch A retest)        [tail thickness]
  7. ar1_60             - lag-1 autocorr of returns 60d (batch A)     [serial dependence]
  8. gap_vol_ratio_20   - std(overnight gap)/std(intraday) (batch A)  [gap share]

Gate: |IC10|>=0.007, |ICIR10|>=0.084, rho<0.5 vs full 18-factor library.
Validation window 2020-01-01..2026-07-15, canonical grid, min 8 valid assets.
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, factor_to_panel,
                           validate_factor, signal_matrix, VAL_START, VAL_END,
                           forward_returns, rank_ic_series, load_index)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)

# ---------- full 18-factor library panels (artifacts + reconstruct missing 3) ----------
lib_panels = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = d.get('signal_artifact')
        if art:
            arr = np.load('factors/' + art, allow_pickle=False)
            if arr.shape == (len(grid), len(WATCHLIST)):
                lib_panels[d['factor_id']] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)

# reconstruct: hilo_pos_60, vix_beta_cond_60x20, eurusd_beta_cond_60x20
vix = load_index('VIX', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
dxy = load_index('DXY', prices=prices)

def f_hilo(df, s):
    return (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())

def f_vixb(df, s):
    if vix is None:
        return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)

def f_eurusd_b(df, s):
    if eurusd is None:
        return None
    r = df['close'].pct_change(); er = eurusd['close'].pct_change()
    z = pd.concat([r.rename('r'), er.rename('e')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['e']) / z['e'].rolling(60).var()
    return (b * (eurusd['close'] / eurusd['close'].shift(20) - 1.0)).reindex(z.index)

for fid, fn in [('hilo_pos_60', f_hilo), ('vix_beta_cond_60x20', f_vixb), ('eurusd_beta_cond_60x20', f_eurusd_b)]:
    if fid in lib_panels:
        continue
    p = factor_to_panel(fn, prices)
    if len(p):
        lib_panels[fid] = pd.DataFrame(signal_matrix(p, grid), index=grid, columns=WATCHLIST)
print(f"library panels for rho: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)

def max_lib_corr(panel):
    best, best_id = 0.0, None
    pm = signal_matrix(panel, grid)
    for fid, lp in lib_panels.items():
        lm = lp.values
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xv = x[m]; yv = y[m]
                xc = xv - xv.mean(); yc = yv - yv.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------- candidates ----------
btc_ret = prices['BTC']['close'].pct_change()

def make_min_ret(w):
    def f(df, s):
        return df['close'].pct_change().rolling(w).min()
    return f

def make_vol_term(ws, wl):
    def f(df, s):
        r = df['close'].pct_change()
        return r.rolling(ws).std() / r.rolling(wl).std().replace(0, np.nan)
    return f

def make_updown_vol_asym(w):
    def f(df, s):
        r = df['close'].pct_change()
        mu = r.rolling(w).mean()
        down = (r - mu).where(r < 0)
        up = (r - mu).where(r > 0)
        dsd = (down ** 2).rolling(w).mean().apply(np.sqrt)
        usd = (up ** 2).rolling(w).mean().apply(np.sqrt)
        return dsd / usd.replace(0, np.nan)
    return f

def make_btc_coexceed(w, thresh=0.01):
    def f(df, s):
        r = df['close'].pct_change()
        joint = (r < -thresh) & (btc_ret < -thresh)
        return joint.rolling(w).mean()
    return f

def make_range_trend(ws, wl):
    def f(df, s):
        rg = (df['high'] - df['low']) / df['close']
        return rg.rolling(ws).mean() / rg.rolling(wl).mean().replace(0, np.nan)
    return f

def make_kurt(w):
    def f(df, s):
        r = df['close'].pct_change()
        mu = r.rolling(w).mean()
        sd = r.rolling(w).std()
        k = ((r - mu) ** 4).rolling(w).mean() / (sd ** 4).replace(0, np.nan) - 3.0
        return k
    return f

def make_ar1(w):
    def f(df, s):
        r = df['close'].pct_change()
        return r.rolling(w).corr(r.shift(1))
    return f

def make_gap_vol_ratio(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        intra = df['close'] / df['open'] - 1.0
        gs = gap.rolling(w).std()
        iv = intra.rolling(w).std()
        return gs / iv.replace(0, np.nan)
    return f

cands = {
    'min_ret_20':          (make_min_ret(20), 'worst single-day return over 20d', 'tail resilience'),
    'vol_term_5_60':       (make_vol_term(5, 60), 'vol(5d)/vol(60d) ratio', 'vol term structure'),
    'updown_vol_asym_20':  (make_updown_vol_asym(20), 'downside semidev / upside semidev (20d)', 'vol asymmetry'),
    'btc_tail_coexceed_60': (make_btc_coexceed(60), 'joint co-crash with BTC (60d)', 'crypto tail'),
    'range_trend_20_60':   (make_range_trend(20, 60), 'avg range 20d / 60d', 'range expansion'),
    'realized_kurt_60':    (make_kurt(60), 'excess kurtosis of 60d returns', 'tail thickness'),
    'ar1_60':              (make_ar1(60), 'lag-1 autocorr of returns (60d)', 'serial dependence'),
    'gap_vol_ratio_20':    (make_gap_vol_ratio(20), 'std(gap)/std(intraday) (20d)', 'gap share'),
}

fwd10 = forward_returns(prices, 10)
results = {}
for fid, (fn, desc, tag) in cands.items():
    panel = factor_to_panel(fn, prices)
    if panel is None or len(panel) == 0:
        print(f"{fid}: EMPTY panel -> skip", flush=True)
        continue
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None", flush=True)
        continue
    rho, rho_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}, 'desc': desc, 'tag': tag}
    dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
    print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"coverage={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id})", flush=True)
    print(f"  decay: {dec}", flush=True)
    ic10 = rank_ic_series(panel, fwd10, 8)
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(ic10.index >= a) & (ic10.index <= b)]
        print(f"  {nm}: {sub.mean():.4f} (n={len(sub)})", flush=True)
    rec = ic10[ic10.index >= '2025-07-16']
    if len(rec) > 30:
        r_ic = rec.mean(); r_icir = r_ic / rec.std(ddof=1) if rec.std(ddof=1) > 0 else 0.0
        print(f"  recent_1y: ic={r_ic:.4f} icir={r_icir:.4f} (n={len(rec)})", flush=True)
    print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007, "
          f"|ICIR|={abs(m['icir']):.4f}/0.084, rho={rho:.3f}/0.5)", flush=True)

with open('scripts/miner_1_20261008_results_batchB.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print("\n=== SUMMARY ===")
for fid, r in results.items():
    m = r['metrics']
    print(f"{fid:22s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} "
          f"rho={m['max_abs_library_correlation']:.3f} ({m.get('max_corr_library_id')}) "
          f"cov={m['coverage_asset_days']:.3f}", flush=True)
