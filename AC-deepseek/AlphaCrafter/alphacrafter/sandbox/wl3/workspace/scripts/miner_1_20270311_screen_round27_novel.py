"""Round 27 (2027-03-11) novel factor screen for the 15-asset cross-asset universe.

Novel candidates vs the 21-factor EFFECTIVE library and prior rejected/evicted sets:
 1. usdcny_beta_60       : rolling 60d beta of asset returns vs USDCNY daily change (China FX beta)
 2. gold_beta_60         : rolling 60d beta vs XAU returns (safe-haven beta)
 3. wti_beta_60          : rolling 60d beta vs WTI returns (energy beta)
 4. overnight_intraday_corr_20 : rolling 20d corr(overnight ret, intraday ret) (overnight/intraday consistency)
 5. vol_term_5_60        : std(ret,5)/std(ret,60) (vol term-structure slope)
 6. gap_fade_20          : 20d mean of gap * sign(prev close-to-close ret) (gap continuation vs fade)
 7. updown_volume_20     : 20d mean volume on up days / mean volume on down days (volume asymmetry)
 8. body_pos_20          : 20d mean of (close-open)/(high-low) (candle body direction/conviction)
 9. corr_spx_20          : rolling 20d correlation of asset ret vs SPX ret (market co-movement)

Gate (benchmark-wide): |IC10| >= 0.007, |ICIR10| >= 0.084 on warm-up
2020-01-01..2026-07-15, max_abs_library_correlation < 0.5 vs library
factors computed from REAL persisted signal artifacts on the canonical grid.
Additional out-of-sample robustness: IC10 on 2026-07-16..2027-03-10.
"""
import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series,
                           VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2600)
print(f"assets loaded: {len(prices)} | {time.time()-t0:.1f}s", flush=True)

# ---------------- macro / market inputs ----------------
spx_r = prices['SPX']['close'].pct_change()
xau_r = prices['XAU']['close'].pct_change()
wti_r = prices['WTI']['close'].pct_change()
cny = load_index('USDCNY', prices=prices)
cny_d = cny['close'].diff() if cny is not None else None
print(f"USDCNY loaded: {cny is not None} | SPX rows {len(spx_r)}", flush=True)


def rb(r, m, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)


def rc(r, m, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    c = z['r'].rolling(w).corr(z['m'])
    return c.reindex(r.index)


# ---------------- candidate factor functions ----------------
def f_usdcny_beta(df, s):
    if cny_d is None:
        return None
    r = df['close'].pct_change()
    return rb(r, cny_d, 60)

def f_gold_beta(df, s):
    r = df['close'].pct_change()
    return rb(r, xau_r, 60)

def f_wti_beta(df, s):
    r = df['close'].pct_change()
    return rb(r, wti_r, 60)

def f_ovint_corr(df, s):
    ov = df['open'] / df['close'].shift(1) - 1.0
    intr = df['close'] / df['open'] - 1.0
    z = pd.concat([ov.rename('ov'), intr.rename('in')], axis=1)
    return z['ov'].rolling(20, min_periods=10).corr(z['in'])

def f_vol_term(df, s):
    r = df['close'].pct_change()
    v5 = r.rolling(5, min_periods=3).std()
    v60 = r.rolling(60, min_periods=30).std()
    return (v5 / v60).replace([np.inf, -np.inf], np.nan)

def f_gap_fade(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    prev_ret = df['close'].pct_change().shift(1)
    gs = gap * np.sign(prev_ret)
    return gs.rolling(20, min_periods=10).mean()

def f_updown_volume(df, s):
    r = df['close'].pct_change()
    vol = df['volume'].astype(float)
    up = vol.where(r > 0)
    dn = vol.where(r < 0)
    mu = up.rolling(20, min_periods=6).mean()
    md = dn.rolling(20, min_periods=6).mean()
    return (mu / md).replace([np.inf, -np.inf], np.nan)

def f_body_pos(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    body = (df['close'] - df['open']) / rng
    return body.rolling(20, min_periods=10).mean()

def f_corr_spx(df, s):
    r = df['close'].pct_change()
    return rc(r, spx_r, 20)

cands = {
    'usdcny_beta_60':         (f_usdcny_beta, 'USDCNY rates beta (60d)', 'beta/fx'),
    'gold_beta_60':           (f_gold_beta, 'XAU safe-haven beta (60d)', 'beta/commodity'),
    'wti_beta_60':            (f_wti_beta, 'WTI energy beta (60d)', 'beta/commodity'),
    'overnight_intraday_corr_20': (f_ovint_corr, 'overnight/intraday ret corr (20d)', 'microstructure'),
    'vol_term_5_60':          (f_vol_term, 'vol term structure 5d/60d', 'volatility'),
    'gap_fade_20':            (f_gap_fade, 'gap continuation/fade (20d)', 'microstructure'),
    'updown_volume_20':       (f_updown_volume, 'up/down day volume ratio (20d)', 'volume'),
    'body_pos_20':            (f_body_pos, 'candle body direction (20d)', 'microstructure'),
    'corr_spx_20':            (f_corr_spx, '20d correlation vs SPX', 'beta/market'),
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
    # decay by horizon
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        fh = forward_returns(prices, h)
        ih = rank_ic_series(factor_panel, fh, 8)
        ihw = ih[(ih.index >= VAL_START) & (ih.index <= VAL_END)]
        decay[str(h)] = float(ihw.mean()) if len(ihw) else float('nan')
    # extended OOS (post warm-up)
    oos = ic[(ic.index >= pd.Timestamp('2026-07-16')) & (ic.index <= pd.Timestamp('2027-03-10'))]
    oos_ic = float(oos.mean()) if len(oos) > 20 else float('nan')
    oos_sd = float(oos.std(ddof=1)) if len(oos) > 20 else float('nan')
    oos_icir = oos_ic / oos_sd if oos_sd and oos_sd > 0 else float('nan')
    return {'ic': mean, 'icir': icir, 'ic_hit_ratio': hit,
            'n_ic_dates': int(len(ic10)), 'coverage_asset_days': cov,
            'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
            'decay_ic_by_horizon': decay,
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
        print(f"  decay: { {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} }", flush=True)
        print(f"  OOS(2026-07-16..2027-03-10): ic={m['oos_ic']:.4f} icir={m['oos_icir']:.4f} n={m['n_oos_dates']}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_1_20270311_results_round27.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:26s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')}) oos_ic={m.get('oos_ic', float('nan')):.4f}")
    else:
        print(f"{fid:26s} ERROR {r.get('error', '')[:70]}")
print(f"total time {time.time()-t0:.1f}s")
