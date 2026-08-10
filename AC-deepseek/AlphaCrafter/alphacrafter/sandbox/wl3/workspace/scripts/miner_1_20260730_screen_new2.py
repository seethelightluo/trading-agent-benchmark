"""miner_1 2026-07-30: drift re-validation of artifact-bearing library + screen NEW orthogonal factors.

1) Drift check: for every factor with a recoverable .npy artifact, reconstruct the
   signal panel from the artifact + declared grid, compute IC/ICIR on the FULL
   window and the RECENT 1y window (drift detection).
2) New candidates (not previously screened): lower_wick_10, obv_slope_20,
   amihud_illiq_20, volz_volume_20x60, ret5_rev_skip1, gap_10, kurt_term_20_60,
   copper_beta_cond_60x20.
3) Admission gate: |IC(h=10)|>=0.007 & |ICIR|>=0.084, rho vs artifact library < 0.5.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel, forward_returns,
                           canonical_grid, WATCHLIST, VAL_START, VAL_END)

prices = load_prices(days=2200)
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; canonical grid n={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---------- fast rank IC (vectorized) ----------
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

def forward_returns_fast(prices, horizon):
    return pd.DataFrame({s: df['close'].shift(-horizon) / df['close'] - 1.0
                         for s, df in prices.items()}).sort_index()

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
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
            'turnover_10d_rank': turn, 'decay_ic_by_horizon': decay}

# ---------- A. DRIFT RE-VALIDATION of artifact-bearing library ----------
import glob
lib_panels = {}
for p in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in p:
        continue
    d = json.load(open(p))
    art = d.get('signal_artifact')
    if not art:
        continue
    arr = np.load('factors/' + art, allow_pickle=False)
    g = d['signal_artifact_grid']
    idx = pd.date_range(pd.Timestamp(g['start']), pd.Timestamp(g['end']),
                        periods=g['n_dates'])
    if len(idx) != arr.shape[0]:
        print(f'  WARN grid mismatch {d["factor_id"]}', flush=True)
        continue
    lib_panels[d['factor_id']] = pd.DataFrame(arr, index=idx, columns=WATCHLIST)

print('\n=== A. DRIFT RE-VALIDATION (artifact library) ===')
RECENT_START = pd.Timestamp('2025-07-01')
for fid, panel in sorted(lib_panels.items()):
    m_full = validate_fast(panel, prices)
    m_rec = validate_fast(panel, prices, start=RECENT_START, end=VAL_END)
    if m_full is None or m_rec is None:
        print(f'  {fid:24s} insufficient', flush=True)
        continue
    flag = ''
    if abs(m_rec['ic']) < 0.007 or abs(m_rec['icir']) < 0.084:
        flag = '  <-- DRIFT (fails gate on recent 1y)'
    print(f'  {fid:24s} FULL  IC={m_full["ic"]:+.4f} ICIR={m_full["icir"]:+.4f} | '
          f'RECENT IC={m_rec["ic"]:+.4f} ICIR={m_rec["icir"]:+.4f}{flag}', flush=True)

# ---------- B. NEW CANDIDATES ----------
dxy = load_index('DXY', prices=prices)
copper = prices.get('COPPER')

def f_lower_wick_10(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    lw = (np.minimum(df['open'], df['close']) - df['low']) / rng
    return lw.rolling(10).mean()

def f_obv_slope_20(df, s):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0.0).cumsum()
    return (obv / df['volume'].rolling(20).mean().replace(0, np.nan)).rolling(20).mean()

def f_amihud_20(df, s):
    illiq = (df['close'].pct_change().abs() / df['volume'].replace(0, np.nan))
    return illiq.rolling(20).mean()

def f_volz_volume_20x60(df, s):
    v = df['volume'].rolling(20).mean()
    mu = v.rolling(60).mean(); sd = v.rolling(60).std()
    return (v - mu) / sd.replace(0, np.nan)

def f_ret5_rev_skip1(df, s):
    return df['close'].shift(2) / df['close'].shift(7) - 1.0

def f_gap_10(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    return gap.rolling(10).mean()

def f_kurt_term_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).kurt() - r.rolling(60).kurt()

def f_copper_beta_cond(df, s):
    if copper is None:
        return None
    r = df['close'].pct_change(); rc = copper['close'].pct_change()
    z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()
    move = copper['close'] / copper['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)

CANDIDATES = [
    ('lower_wick_10', f_lower_wick_10, 'buying-pressure microstructure'),
    ('obv_slope_20', f_obv_slope_20, 'volume-flow trend'),
    ('amihud_illiq_20', f_amihud_20, 'illiquidity premium'),
    ('volz_volume_20x60', f_volz_volume_20x60, 'volume expansion regime'),
    ('ret5_rev_skip1', f_ret5_rev_skip1, 'short-term reversal (5d skip1)'),
    ('gap_10', f_gap_10, 'opening-gap tendency'),
    ('kurt_term_20_60', f_kurt_term_20_60, 'kurtosis term structure'),
    ('copper_beta_cond_60x20', f_copper_beta_cond, 'copper (global growth) beta x 20d move'),
]

def max_lib_corr(panel, lib_panels, min_valid=8):
    best, best_id = 0.0, None
    for fid, lp in lib_panels.items():
        idx = panel.index.intersection(lp.index)
        corrs = []
        for dd in idx:
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

print('\n=== B. NEW CANDIDATE VALIDATION ===')
results = {}
for fid, fn, idea in CANDIDATES:
    panel = factor_to_panel(fn, prices)
    m = validate_fast(panel, prices)
    if m is None:
        print(f'  {fid:24s} insufficient data', flush=True)
        continue
    rho, rho_id = max_lib_corr(panel, lib_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'idea': idea, 'metrics': m, 'ok': ok, 'panel': panel}
    print(f'  {fid:24s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'cov={m["coverage_asset_days"]:.2f} turn={m["turnover_10d_rank"]:.2f} '
          f'rho={rho:.3f}({rho_id}) -> {"PASS" if ok else "FAIL"}', flush=True)

print('\n=== SUMMARY PASS ===')
for fid, r in results.items():
    if r['ok']:
        m = r['metrics']
        print(f'  {fid:24s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} rho={m["max_abs_library_correlation"]:.3f}')
