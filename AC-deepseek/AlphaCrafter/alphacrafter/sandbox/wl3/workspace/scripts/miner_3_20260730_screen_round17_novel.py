"""Round-17 screen: orthogonal microstructure/liquidity/cross-asset candidates.

Motivation: rounds 15-16 failed mostly on ICIR or |max library corr| >= 0.5.
Library is dominated by macro-beta (SPX/HS300/DXY/EURUSD/VIX/CN10Y/down),
volatility (vol_of_vol, vol_adj_mom, dd_duration), trend position (hilo_pos_60),
intraday skew, mom acceleration. Round-17 targets orthogonal information:
liquidity/impact (Amihud), volume flow (trend, HHI concentration), sign
randomness (runs test), autocorrelation, return kurtosis, VWAP deviation,
cross-sectional demeaned relative strength, and conditional beta to
crypto (BTC/ETH) / energy (WTI) / gold (XAU) drivers.

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

# ---------------- Round-17 candidates ----------------
def make_amihud(w):
    def f(df, s):
        r = df['close'].pct_change().abs()
        v = df['volume']
        return (r / v.replace(0, np.nan)).rolling(w).mean() * 1e9
    return f

def make_vol_trend(ws, wl):
    def f(df, s):
        r = df['close'].pct_change()
        return r.rolling(ws).std() / r.rolling(wl).std().replace(0, np.nan)
    return f

def make_runs_z(w):
    def f(df, s):
        r = df['close'].pct_change()
        up = (r > 0).astype(float)
        dn = (r < 0).astype(float)
        n = up + dn
        runs = (up != up.shift(1)).astype(float)
        runs = (runs * n).rolling(w).sum()
        n1 = up.rolling(w).sum()
        n2 = dn.rolling(w).sum()
        nn = n.rolling(w).sum()
        exp = 1.0 + 2.0 * n1 * n2 / nn.replace(0, np.nan)
        var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - nn)) / (nn ** 2 * (nn - 1.0)).replace(0, np.nan)
        return (runs - exp) / np.sqrt(var.clip(lower=0))
    return f

def make_kurt(w):
    def f(df, s):
        return df['close'].pct_change().rolling(w).kurt()
    return f

def make_ac1(w):
    def f(df, s):
        r = df['close'].pct_change()
        return r.rolling(w).corr(r.shift(1))
    return f

def make_beta_cond(mkt, wb, wm):
    mktc = mkt['close']
    def f(df, s):
        r = df['close'].pct_change()
        rm = mktc.reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
        beta = z['r'].rolling(wb).cov(z['m']) / z['m'].rolling(wb).var()
        mom = mktc.reindex(z.index) / mktc.reindex(z.index).shift(wm) - 1.0
        return (beta * mom).reindex(z.index)
    return f

def make_xau_beta(w):
    xau = prices['XAU']['close']
    def f(df, s):
        r = df['close'].pct_change()
        rx = xau.reindex(df.index).pct_change()
        z = pd.concat([r.rename('r'), rx.rename('x')], axis=1).dropna()
        return z['r'].rolling(w).cov(z['x']) / z['x'].rolling(w).var()
    return f

def make_vwap_dev(w):
    def f(df, s):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        pv = (tp * df['volume']).rolling(w).sum()
        vv = df['volume'].rolling(w).sum().replace(0, np.nan)
        vwap = pv / vv
        return df['close'] / vwap - 1.0
    return f

def make_basket_demeaned(w):
    def f(df, s):
        return df['close'].shift(1) / df['close'].shift(1 + w) - 1.0
    # post-process: demean across assets per date
    return f

def make_vol_hhi(w):
    def f(df, s):
        v = df['volume']
        tot = v.rolling(w).sum().replace(0, np.nan)
        sh = v / tot
        return (sh ** 2).rolling(w).mean()
    return f

def make_depth(w):
    def f(df, s):
        hi = df['close'].rolling(w).max()
        return df['close'] / hi - 1.0
    return f

def make_sign_persist(w):
    def f(df, s):
        r = df['close'].pct_change()
        same = (np.sign(r) == np.sign(r.shift(1))).astype(float)
        valid = (r.notna() & r.shift(1).notna()).astype(float)
        return (same * valid).rolling(w).sum() / valid.rolling(w).sum().replace(0, np.nan)
    return f

def make_volume_mom(ws, wl):
    def f(df, s):
        return df['volume'].rolling(ws).mean() / df['volume'].rolling(wl).mean().replace(0, np.nan)
    return f

candidates = {
    'amihud_20': make_amihud(20),
    'vol_trend_20_60': make_vol_trend(20, 60),
    'runs_z_20': make_runs_z(20),
    'kurt_20': make_kurt(20),
    'ac1_20': make_ac1(20),
    'btc_beta_cond_60x20': make_beta_cond(prices['BTC'], 60, 20),
    'eth_beta_cond_60x20': make_beta_cond(prices['ETH'], 60, 20),
    'xau_beta_60': make_xau_beta(60),
    'vwap_dev_20': make_vwap_dev(20),
    'xret_basket_20': make_basket_demeaned(20),
    'vol_hhi_20': make_vol_hhi(20),
    'depth_120': make_depth(120),
    'sign_persist_20': make_sign_persist(20),
    'wti_beta_cond_60x20': make_beta_cond(prices['WTI'], 60, 20),
    'volume_mom_20_60': make_volume_mom(20, 60),
}

results = {}
for fid, fn in candidates.items():
    ta = time.time()
    panel = factor_to_panel(fn, prices)
    # cross-sectional demean for basket-relative factor
    if fid == 'xret_basket_20' and len(panel):
        panel = panel.sub(panel.mean(axis=1), axis=0)
    m = validate(fid, panel)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        json.dump(results, open('scripts/miner_3_20260730_results_round17.json', 'w'), indent=1, default=str)
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
    json.dump(results, open('scripts/miner_3_20260730_results_round17.json', 'w'), indent=1, default=str)

print(f"\ndone in {time.time()-t0:.1f}s; saved scripts/miner_3_20260730_results_round17.json", flush=True)
