"""miner_1 2026-07-30 batch-4 screening: NEW orthogonal factor families.

Previously covered by library (11 EFFECTIVE): spx/hs300/dxy/eurusd/vix beta,
vol-adj momentum, hilo range position, skew term, vol-of-vol, max ret, drawdown
duration residual. Prior batches covered: autocorr, variance ratio, dd_60d,
close_loc, us10y_beta_cond, vam variants, kurt_term, obv_slope, lower_wick,
gap_10, ret5_rev, volz_volume, amihud, vol_term_20_60, eff_ratio, downside vol
ratio, CCI-like, stoch, aroon, etc.

THIS batch (all NEW):
  1. vwap_bias_20        : close vs 20d VWAP distance (volume-weighted anchor)
  2. mfi_14              : Money Flow Index (volume-weighted RSI)
  3. usdcny_beta_60      : beta to USDCNY (China FX) changes
  4. cn10y_beta_60       : beta to CN10Y (China yield) changes
  5. gc_ratio_beta_60    : beta to XAU/COPPER ratio (growth vs safe-haven)
  6. resid_mom_60_20     : 60d momentum residualized on basket beta
  7. calmar_60           : 60d return / 60d max-drawdown depth
  8. vol_ratio_5_60      : 5d/60d realized vol ratio (short vol burst)
  9. cci_20              : Commodity Channel Index 20d
 10. force_index_20      : EMA20 of (close change x volume)
 11. updown_capture_20   : mean up-day ret / |mean down-day ret|
 12. hilo_pos_10         : 10d range position (short-horizon)
 13. ndx_beta_60         : beta to NDX (tech anchor)
 14. us10y_beta_60       : plain beta to US10Y changes
 15. vix_corr_60         : 60d corr of asset ret with VIX changes

Gate: |IC(h=10)| >= 0.007 AND |ICIR| >= 0.084 on 2020-01-01..2026-07-15;
rho vs extended artifact library < 0.5 recommended for gate survival.
"""
import sys, json, glob
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel, WATCHLIST,
                           canonical_grid, VAL_START, VAL_END, build_library_panels)

prices = load_prices(days=2200)
dxy = load_index('DXY', prices=prices)
usdcny = load_index('USDCNY', prices=prices)
vix = load_index('VIX', prices=prices)
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; grid n={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---------- fast rank IC (vectorized, h=10 admission) ----------
def forward_returns_fast(prices, horizon):
    return pd.DataFrame({s: df['close'].shift(-horizon) / df['close'] - 1.0
                         for s, df in prices.items()}).sort_index()

def rank_ic_series_fast(factor_panel, fwd_ret, min_valid=8):
    df = pd.concat({'x': factor_panel, 'y': fwd_ret}, axis=1).sort_index()
    x = df['x'].rank(axis=1); y = df['y'].rank(axis=1)
    valid = df['x'].notna() & df['y'].notna() & np.isfinite(df['x']) & np.isfinite(df['y'])
    n = valid.sum(axis=1)
    x = x.where(valid); y = y.where(valid)
    mx = x.mean(axis=1); my = y.mean(axis=1)
    cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
    vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
    vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(vx * vy)
    return ic.where((n >= min_valid) & (vx > 0) & (vy > 0)).dropna()

def validate_fast(factor_panel, prices, horizons=(1, 2, 3, 5, 10, 20), min_valid=8,
                  start=VAL_START, end=VAL_END):
    fwd = {h: forward_returns_fast(prices, h) for h in horizons}
    ic_s = {h: rank_ic_series_fast(factor_panel, fwd[h], min_valid) for h in horizons}
    ic10 = ic_s[10][(ic_s[10].index >= start) & (ic_s[10].index <= end)]
    if len(ic10) < 60:
        return None
    ic_mean = float(ic10.mean()); ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = factor_panel[(factor_panel.index >= start) & (factor_panel.index <= end)]
    total = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total if total else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {str(h): (float(ic_s[h].mean()) if len(ic_s[h]) else float('nan')) for h in horizons}
    rec = ic10[ic10.index >= pd.Timestamp('2025-07-01')]
    ic_rec = float(rec.mean()) if len(rec) else float('nan')
    icir_rec = (ic_rec / rec.std(ddof=1)) if len(rec) > 30 and rec.std(ddof=1) > 0 else float('nan')
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
            'turnover_10d_rank': turn, 'decay_ic_by_horizon': decay,
            'recent_1y_ic': ic_rec, 'recent_1y_icir': icir_rec}

# ---------- extended library: artifacts + reconstructed non-artifact effective factors ----------
def load_all_artifact_panels():
    out = {}
    for jp in sorted(glob.glob('factors/*.json')):
        if 'ensemble' in jp:
            continue
        d = json.load(open(jp))
        art = d.get('signal_artifact')
        if not art or not (Path_ := __import__('pathlib').Path('factors') / art).exists():
            continue
        arr = np.load(Path_, allow_pickle=False)
        if arr.shape[0] != len(grid):
            continue
        out[d['factor_id']] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    return out

def f_hilo(df, s, w):
    hi = df['high'].rolling(w).max(); lo = df['low'].rolling(w).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)

def _beta_cond(anchor_close, sign=1.0):
    def f(df, s):
        a = anchor_close.reindex(df.index)
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var().replace(0, np.nan)
        mom = (anchor_close / anchor_close.shift(20) - 1.0).reindex(df.index)
        return (sign * b * mom).reindex(z.index)
    return f

eurusd = load_index('EURUSD', prices=prices)
lib = build_library_panels(prices)
lib.update(load_all_artifact_panels())
lib['eurusd_beta_cond_60x20'] = factor_to_panel(_beta_cond(eurusd['close']), prices)
lib['hilo_pos_60'] = factor_to_panel(lambda df, s: f_hilo(df, s, 60), prices)
lib['vix_beta_cond_60x20'] = factor_to_panel(_beta_cond(vix['close'], sign=-1.0), prices)
lib['vol_of_vol20x60'] = factor_to_panel(
    lambda df, s: df['close'].pct_change().rolling(20).std().rolling(60).std(), prices)
print(f'[lib] extended library: {len(lib)} -> {sorted(lib.keys())}', flush=True)

# ---------- candidate definitions ----------
ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)
basket_level = (1 + basket.fillna(0.0)).cumprod()
copper = prices.get('COPPER')
xau = prices.get('XAU')
ndx = prices.get('NDX')

def f_vwap_bias_20(df, s):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).rolling(20).sum()
    vv = df['volume'].rolling(20).sum().replace(0, np.nan)
    vwap = pv / vv
    return df['close'] / vwap - 1.0

def f_mfi_14(df, s):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    d = tp.diff()
    pos = d.clip(lower=0.0) * df['volume']
    neg = (-d.clip(upper=0.0)) * df['volume']
    pr = pos.rolling(14).sum(); nr = neg.rolling(14).sum()
    mr = pr / nr.replace(0, np.nan)
    return 100 - 100 / (1 + mr)

def f_beta(anchor_close):
    def f(df, s):
        a = anchor_close.reindex(df.index)
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var().replace(0, np.nan)
        return b.reindex(z.index)
    return f

def f_gc_ratio_beta(df, s):
    if xau is None or copper is None:
        return None
    ratio = xau['close'] / copper['close']
    return f_beta(ratio)(df, s)

def f_resid_mom_60_20(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var().replace(0, np.nan)
    mom = df['close'].shift(5) / df['close'].shift(65) - 1.0
    bm = basket_level.shift(5) / basket_level.shift(65) - 1.0
    return (mom - b * bm.reindex(df.index)).reindex(df.index)

def f_calmar_60(df, s):
    c = df['close']
    mom60 = c / c.shift(60) - 1.0
    dd = 1.0 - c / c.rolling(60).max()
    return mom60 / (dd + 1e-4)

def f_vol_ratio_5_60(df, s):
    r = df['close'].pct_change()
    v5 = r.rolling(5).std(); v60 = r.rolling(60).std()
    return (v5 / v60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_cci_20(df, s):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def f_force_index_20(df, s):
    fi = df['close'].diff() * df['volume']
    return fi.ewm(span=20, adjust=False).mean()

def f_updown_capture_20(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0); dn = r.where(r < 0)
    mu = up.rolling(20).mean()
    md = dn.rolling(20).mean()
    return mu / md.abs().replace(0, np.nan)

CANDIDATES = [
    ('vwap_bias_20', f_vwap_bias_20, 'volume-weighted anchor distance'),
    ('mfi_14', f_mfi_14, 'money flow index'),
    ('usdcny_beta_60', f_beta(usdcny['close']), 'China FX beta'),
    ('cn10y_beta_60', f_beta(prices['CN10Y']['close']), 'China yield beta'),
    ('gc_ratio_beta_60', f_gc_ratio_beta, 'gold/copper ratio beta'),
    ('resid_mom_60_20', f_resid_mom_60_20, 'idiosyncratic momentum vs basket'),
    ('calmar_60', f_calmar_60, 'reward per drawdown unit'),
    ('vol_ratio_5_60', f_vol_ratio_5_60, 'short vol burst ratio'),
    ('cci_20', f_cci_20, 'commodity channel index'),
    ('force_index_20', f_force_index_20, 'volume-force momentum'),
    ('updown_capture_20', f_updown_capture_20, 'up/down capture asymmetry'),
    ('hilo_pos_10', lambda df, s: f_hilo(df, s, 10), '10d range position'),
    ('ndx_beta_60', f_beta(ndx['close']), 'NDX tech beta'),
    ('us10y_beta_60', f_beta(prices['US10Y']['close']), 'US yield beta'),
    ('vix_corr_60', lambda df, s: df['close'].pct_change().rolling(60).corr(
        vix['close'].pct_change()) if vix is not None else None, 'VIX correlation'),
]

def max_lib_corr(panel, lib_panels, min_valid=8):
    best, best_id = 0.0, None
    idx = panel.index.intersection(grid)
    for fid, lp in lib_panels.items():
        idxc = idx.intersection(lp.index)
        corrs = []
        for dd in idxc:
            x = panel.loc[dd]; y = lp.loc[dd]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= min_valid:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

print('\n=== BATCH-4 CANDIDATE VALIDATION ===', flush=True)
results = {}
for fid, fn, idea in CANDIDATES:
    try:
        panel = factor_to_panel(fn, prices)
    except Exception as exc:
        print(f'  {fid:24s} ERROR {exc}', flush=True)
        continue
    m = validate_fast(panel, prices)
    if m is None:
        print(f'  {fid:24s} insufficient data', flush=True)
        continue
    rho, rho_id = max_lib_corr(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    gate_ok = ok and rho < 0.5
    results[fid] = {'idea': idea, 'metrics': m, 'panel': panel, 'ok': ok, 'gate_ok': gate_ok}
    print(f'  {fid:24s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'cov={m["coverage_asset_days"]:.2f} turn={m["turnover_10d_rank"]:.2f} '
          f'rho={rho:.3f}({rho_id}) rec1y_ICIR={m["recent_1y_icir"]:+.3f} -> {"PASS" if ok else "FAIL"} '
          f'{"+RHO<0.5" if gate_ok else ""}', flush=True)
    print(f'      decay={json.dumps(m["decay_ic_by_horizon"])}', flush=True)

print('\n=== PASS SUMMARY (IC/ICIR gate) ===', flush=True)
for fid, r in results.items():
    if r['ok']:
        m = r['metrics']
        print(f'  {fid:24s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} rho={m["max_abs_library_correlation"]:.3f} '
              f'vs {m["max_corr_library_id"]} gate_ok={r["gate_ok"]}', flush=True)

with open('scripts/miner_1_20260730_results_batch4.json', 'w') as fh:
    json.dump({k: {'idea': v['idea'], 'metrics': v['metrics'], 'ok': v['ok'], 'gate_ok': v['gate_ok']}
               for k, v in results.items()}, fh, indent=1, default=str)
print('[saved] scripts/miner_1_20260730_results_batch4.json', flush=True)
