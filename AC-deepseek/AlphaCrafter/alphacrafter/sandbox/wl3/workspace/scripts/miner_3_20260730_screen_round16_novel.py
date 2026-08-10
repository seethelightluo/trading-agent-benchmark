"""Round-16 screen: novel OHLC-only candidates with vectorized numpy IC engine.

Motivation: round-15 candidates (MFI, CCI, reversal, XAU-relative momentum) all
failed -- mostly on |max library correlation| >= 0.5. The effective library is
dominated by macro-beta (SPX/HS300/DXY/EURUSD/VIX/CN10Y/down), volatility
(vol_of_vol, vol_adj_mom, dd_duration), trend position (hilo_pos_60), intraday
skew and momentum acceleration. Round-16 targets orthogonal microstructure:
overnight-gap structure, candle body geometry, range-vs-close vol ratio,
run-streak asymmetry, downside semivariance, gap consistency, open positioning
and close-close skew. All use OHLC only -> full 15-asset coverage.

Admission gate (shared): |IC10| >= 0.007, |ICIR10| >= 0.084, rho < 0.5.
"""
import sys, json, glob, time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, forward_returns,
                           VAL_START, VAL_END)
from miner_3_20260730_library_rebuild import build_library_panels

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2500)
grid = canonical_grid(prices)
vix = load_index('VIX', prices=prices)
dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
print(f"prices {len(prices)} assets; grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------------- library panels (effective factors only) ----------------
lib = build_library_panels(prices, vix, dxy, eurusd)
spx = prices['SPX']['close']
hs300 = prices['000300.SH']['close']
cn10y = prices['CN10Y']['close']

def f_cn10ybeta(df, s):
    r = df['close'].pct_change()
    rc = cn10y.reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
    return z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()

def f_intraskew(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20).skew()

def f_momaccel(df, s):
    return df['close'].shift(5) / df['close'].shift(65) - df['close'].shift(5) / df['close'].shift(125)

lib['cn10y_beta_60'] = factor_to_panel(f_cn10ybeta, prices)
lib['intraday_ret_skew_20'] = factor_to_panel(f_intraskew, prices)
lib['mom_accel_60_120'] = factor_to_panel(f_momaccel, prices)

eff = set()
for f in glob.glob('factors/*.json'):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            eff.add(d['factor_id'])
    except Exception:
        pass
lib = {k: v for k, v in lib.items() if k in eff}
print(f"library panels rebuilt: {len(lib)} -> {sorted(lib)}", flush=True)

# ---------------- fast rank-matrix infrastructure ----------------
def to_rank_matrix(panel):
    m = signal_matrix(panel, grid)
    out = np.full(m.shape, np.nan)
    for i in range(m.shape[0]):
        row = m[i]
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            r = pd.Series(row[valid]).rank().values
            out[i, valid] = r
    return out

lib_ranks = {fid: to_rank_matrix(p) for fid, p in lib.items()}
print("library rank matrices done", flush=True)

def fast_max_lib_corr(rank_m):
    best, best_id = 0.0, None
    for fid, lr in lib_ranks.items():
        corrs = []
        for t in range(len(grid)):
            x = rank_m[t]; y = lr[t]
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

# ---------------- precomputed forward-return rank matrices ----------------
fwd_panels = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
fwd_ranks = {h: to_rank_matrix(fwd_panels[h]) for h in fwd_panels}
print("forward rank matrices done", flush=True)

def fast_ic_series_from_ranks(fac_rank, fwd_rank, min_valid=8):
    ic, dates = [], []
    for t in range(len(grid)):
        x = fac_rank[t]; y = fwd_rank[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            xv = x[m]; yv = y[m]
            xc = xv - xv.mean(); yc = yv - yv.mean()
            den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
            if den > 0:
                ic.append((xc * yc).sum() / den)
                dates.append(t)
    return pd.Series(ic, index=grid[dates])

def validate(fid, panel):
    if panel is None or len(panel) == 0:
        return None
    rank_m = to_rank_matrix(panel)
    ic_series = {h: fast_ic_series_from_ranks(rank_m, fwd_ranks[h]) for h in fwd_ranks}
    ic10 = ic_series[10]
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    if len(ic10) < 100:
        return None
    ic_mean = float(ic10.mean()); ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    m = {
        'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
        'n_ic_dates': int(len(ic10)), 'coverage_asset_days': coverage,
        'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
        'decay_ic_by_horizon': {str(h): (float(ic_series[h].mean()) if len(ic_series[h]) else float('nan')) for h in fwd_ranks},
    }
    for name, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                       ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                       ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(ic10.index >= pd.Timestamp(a)) & (ic10.index <= pd.Timestamp(b))]
        m[name] = float(sub.mean()) if len(sub) > 30 else float('nan')
    recent = ic10[(ic10.index >= pd.Timestamp('2025-07-15')) & (ic10.index <= pd.Timestamp('2026-07-15'))]
    if len(recent) > 30:
        m['recent_1y_ic'] = float(recent.mean())
        m['recent_1y_icir'] = float(recent.mean() / recent.std(ddof=1)) if recent.std(ddof=1) > 0 else 0.0
    m['_rank_matrix'] = rank_m
    return m

# ---------------- Round-16 candidates (all OHLC-only) ----------------
def make_ovn_share(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        intra = df['close'] / df['open'] - 1.0
        g = gap.abs().rolling(w).sum()
        i = intra.abs().rolling(w).sum()
        return g / (g + i).replace(0, np.nan)
    return f

def make_gap_cont(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        intra = df['close'] / df['open'] - 1.0
        agree = (np.sign(gap) == np.sign(intra)).astype(float)
        valid = gap.notna() & intra.notna() & (gap.abs() > 0)
        return (agree * valid).rolling(w).sum() / valid.rolling(w).sum()
    return f

def make_ovn_mom(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        return gap.shift(1).rolling(w).sum()
    return f

def make_body_ratio(w):
    def f(df, s):
        rng = (df['high'] - df['low']).replace(0, np.nan)
        return (df['close'] - df['open']).abs() / rng
    return f

def make_gk_ratio(w):
    def f(df, s):
        park = (np.log(df['high'] / df['low']) ** 2).rolling(w).mean() / (4 * np.log(2))
        cc = df['close'].pct_change().rolling(w).var()
        return park / cc.replace(0, np.nan)
    return f

def make_down_vol_ratio(w):
    def f(df, s):
        r = df['close'].pct_change()
        neg = r.where(r < 0, 0.0)
        dvol = (neg ** 2).rolling(w).mean().apply(np.sqrt)
        tot = r.rolling(w).std()
        return dvol / tot.replace(0, np.nan)
    return f

def make_streak(w):
    def f(df, s):
        r = df['close'].pct_change()
        up = (r > 0).astype(int)
        dn = (r < 0).astype(int)
        ups = up.groupby((~up.astype(bool)).cumsum()).cumsum()
        dns = dn.groupby((~dn.astype(bool)).cumsum()).cumsum()
        return (ups - dns).rolling(w).max() / w
    return f

def make_gap_z(wm, ws):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        mg = gap.abs().rolling(wm).mean()
        sg = gap.rolling(ws).std()
        return mg / sg.replace(0, np.nan)
    return f

def make_open_pos(w):
    def f(df, s):
        rng = (df['high'] - df['low']).replace(0, np.nan)
        return ((df['open'] - df['low']) / rng).rolling(w).mean()
    return f

def make_cc_skew(w):
    def f(df, s):
        return df['close'].pct_change().rolling(w).skew()
    return f

def make_up_vol_ratio(w):
    def f(df, s):
        r = df['close'].pct_change()
        pos = r.where(r > 0, np.nan).rolling(w).std()
        neg = r.where(r < 0, np.nan).rolling(w).std()
        return pos / neg.replace(0, np.nan)
    return f

def make_range_ratio(ws, wl):
    def f(df, s):
        a = (df['high'] - df['low']).rolling(ws).mean()
        b = (df['high'] - df['low']).rolling(wl).mean()
        return a / b.replace(0, np.nan)
    return f

def make_gap_intra_corr(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        intra = df['close'] / df['open'] - 1.0
        return gap.rolling(w).corr(intra)
    return f

candidates = {
    'ovn_share_20': make_ovn_share(20),
    'gap_cont_20': make_gap_cont(20),
    'ovn_mom_10': make_ovn_mom(10),
    'body_ratio_20': make_body_ratio(20),
    'gk_ratio_20': make_gk_ratio(20),
    'down_vol_ratio_20': make_down_vol_ratio(20),
    'streak_60': make_streak(60),
    'gap_z_20x60': make_gap_z(20, 60),
    'open_pos_20': make_open_pos(20),
    'cc_ret_skew_20': make_cc_skew(20),
    'up_vol_ratio_60': make_up_vol_ratio(60),
    'range_ratio_5_60': make_range_ratio(5, 60),
    'gap_intra_corr_20': make_gap_intra_corr(20),
}

results = {}
for fid, fn in candidates.items():
    ta = time.time()
    panel = factor_to_panel(fn, prices)
    m = validate(fid, panel)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        json.dump(results, open('scripts/miner_3_20260730_results_round16.json', 'w'), indent=1, default=str)
        continue
    rank_m = m.pop('_rank_matrix')
    rho, lib_id = fast_max_lib_corr(rank_m)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': ok, 'metrics': m}
    print(f"{fid}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({lib_id}) "
          f"p1={m['ic_2020_2022']:+.3f} p2={m['ic_2023_2024']:+.3f} p3={m['ic_2025_2026']:+.3f} "
          f"1y={m.get('recent_1y_ic', float('nan')):+.4f} t={time.time()-ta:.1f}s -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)
    json.dump(results, open('scripts/miner_3_20260730_results_round16.json', 'w'), indent=1, default=str)

print(f"\ndone in {time.time()-t0:.1f}s; saved scripts/miner_3_20260730_results_round16.json", flush=True)
