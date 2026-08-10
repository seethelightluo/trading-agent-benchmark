"""Round-15 screen: optimized validation (numpy rank matrices) for round-14
candidates + a few new ideas. Saves results incrementally so a timeout still
yields partial output.

Motivation: round-14 timed out inside pandas max_library_correlation
(17 candidates x 13 library panels x ~1650 dates of Series rank corr) and
rolling().apply() autocorr. Here we precompute library rank matrices once and
vectorize daily Spearman via numpy.
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
usdjpy = load_index('USDJPY', prices=prices)
print(f"prices {len(prices)} assets; grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------------- library panels ----------------
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
    """Panel -> (n_dates, 15) rank matrix on canonical grid (NaN preserved)."""
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
    """Mean daily Spearman vs each library panel (Pearson on ranks)."""
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

# ---------------- validation (precomputed forward returns) ----------------
fwd = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}

def fast_ic_series(panel, fwd_ret, min_valid=8):
    common = panel.index.intersection(fwd_ret.index)
    ic = {}
    for d in common:
        x = panel.loc[d]; y = fwd_ret.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    return pd.Series(ic).sort_index()

def validate(fid, panel):
    if panel is None or len(panel) == 0:
        return None
    ic_series = {h: fast_ic_series(panel, fwd[h]) for h in fwd}
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
        'decay_ic_by_horizon': {str(h): (float(ic_series[h].mean()) if len(ic_series[h]) else float('nan')) for h in fwd},
    }
    ic_s = ic10
    for name, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                       ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                       ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic_s[(ic_s.index >= pd.Timestamp(a)) & (ic_s.index <= pd.Timestamp(b))]
        m[name] = float(sub.mean()) if len(sub) > 30 else float('nan')
    recent = ic_s[(ic_s.index >= pd.Timestamp('2025-07-15')) & (ic_s.index <= pd.Timestamp('2026-07-15'))]
    if len(recent) > 30:
        m['recent_1y_ic'] = float(recent.mean())
        m['recent_1y_icir'] = float(recent.mean() / recent.std(ddof=1)) if recent.std(ddof=1) > 0 else 0.0
    return m

# ---------------- candidates ----------------
def make_mfi(w):
    def mfi_w(df, s):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        mf = tp * df['volume'].astype(float)
        d = tp.diff()
        pos = mf.where(d > 0, 0.0).rolling(w).sum()
        neg = mf.where(d < 0, 0.0).rolling(w).sum()
        ratio = pos / neg.replace(0, np.nan)
        return 100.0 - 100.0 / (1.0 + ratio)
    return mfi_w

def make_cci(w):
    def cci_w(df, s):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        sma = tp.rolling(w).mean()
        md = (tp - sma).abs().rolling(w).mean().replace(0, np.nan)
        return (tp - sma) / (0.015 * md)
    return cci_w

def rev_5(df, s):
    return -(df['close'].shift(1) / df['close'].shift(6) - 1.0)

def rev_5_vol(df, s):
    c = df['close']
    r5 = -(c.shift(1) / c.shift(6) - 1.0)
    vol20 = c.pct_change().rolling(20).std()
    return r5 / vol20.replace(0, np.nan)

def _xau_ret():
    return prices['XAU']['close'].pct_change()

def corr_xau_60(df, s):
    xr = _xau_ret()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xr.rename('x')], axis=1)
    return z['r'].rolling(60).corr(z['x'])

def rel_mom_xau_20(df, s):
    c = df['close']
    xau = prices['XAU']['close']
    return (c.shift(1) / c.shift(21) - 1.0 - (xau.shift(1) / xau.shift(21) - 1.0)).reindex(c.index)

def updown_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    up = r[r > 0].rolling(20).std()
    dn = r[r < 0].rolling(20).std()
    return up / dn.replace(0, np.nan)

def usdjpy_beta_cond(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    rj = usdjpy['close'].reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rj.rename('j')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['j']) / z['j'].rolling(60).var()
    mom_j = usdjpy['close'].reindex(z.index) / usdjpy['close'].reindex(z.index).shift(20) - 1.0
    return (beta * mom_j).reindex(z.index)

def yieldspread_beta_60(df, s):
    r = df['close'].pct_change()
    sp = (cn10y - prices['US10Y']['close']).reindex(df.index)
    dsp = sp.diff()
    z = pd.concat([r.rename('r'), dsp.rename('s')], axis=1).dropna()
    return z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var()

def wti_beta_cond(df, s):
    r = df['close'].pct_change()
    rw = prices['WTI']['close'].reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['w']) / z['w'].rolling(60).var()
    mom_w = prices['WTI']['close'].reindex(z.index) / prices['WTI']['close'].reindex(z.index).shift(20) - 1.0
    return (beta * mom_w).reindex(z.index)

def xau_beta_cond(df, s):
    r = df['close'].pct_change()
    rx = prices['XAU']['close'].reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rx.rename('x')], axis=1).dropna()
    beta = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var()
    mom_x = prices['XAU']['close'].reindex(z.index) / prices['XAU']['close'].reindex(z.index).shift(20) - 1.0
    return (beta * mom_x).reindex(z.index)

def autocorr_20(df, s):
    r = df['close'].pct_change().values.astype(float)
    n = len(r)
    out = np.full(n, np.nan)
    if n >= 21:
        for t in range(20, n):
            w = r[t - 20:t + 1]
            if np.isfinite(w).all():
                x = w[:-1]; y = w[1:]
                xc = x - x.mean(); yc = y - y.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                out[t] = (xc * yc).sum() / den if den > 0 else np.nan
    return pd.Series(out, index=df.index)

def maxdd_60(df, s):
    c = df['close']
    roll_max = c.rolling(60).max()
    return (c / roll_max - 1.0)

def ret_skew_20(df, s):
    return df['close'].pct_change().rolling(20).skew()

def corr_btc_30(df, s):
    rb = prices['BTC']['close'].reindex(df.index).pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1)
    return z['r'].rolling(30).corr(z['b'])

def beta_asym_60(df, s):
    r = df['close'].pct_change()
    rs = spx.reindex(df.index).pct_change()
    z = pd.concat([r.rename('r'), rs.rename('s')], axis=1).dropna()
    up = z[z['s'] > 0]; dn = z[z['s'] < 0]
    bu = up['r'].rolling(60).cov(up['s']) / up['s'].rolling(60).var()
    bd = dn['r'].rolling(60).cov(dn['s']) / dn['s'].rolling(60).var()
    return (bu - bd).reindex(z.index)

def vwap_dev_20(df, s):
    vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    return (df['close'] - vwap) / vwap.replace(0, np.nan)

def vol_ratio_10_30(df, s):
    r = df['close'].pct_change()
    v10 = r.rolling(10).std(); v30 = r.rolling(30).std()
    return v10 / v30.replace(0, np.nan)

def downside_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0, 0.0)
    dvol = (neg ** 2).rolling(20).mean().apply(np.sqrt)
    tot = r.rolling(20).std()
    return dvol / tot.replace(0, np.nan)

candidates = {
    'mfi_7': make_mfi(7), 'mfi_21': make_mfi(21), 'cci_7': make_cci(7),
    'rev_5': rev_5, 'rev_5_vol': rev_5_vol,
    'corr_xau_60': corr_xau_60, 'rel_mom_xau_20': rel_mom_xau_20,
    'updown_vol_ratio_20': updown_vol_ratio_20,
    'usdjpy_beta_cond_60x20': usdjpy_beta_cond,
    'yieldspread_beta_60': yieldspread_beta_60,
    'wti_beta_cond_60x20': wti_beta_cond,
    'xau_beta_cond_60x20': xau_beta_cond,
    'autocorr_20': autocorr_20, 'maxdd_60': maxdd_60,
    'ret_skew_20': ret_skew_20, 'corr_btc_30': corr_btc_30,
    'beta_asym_60': beta_asym_60, 'vwap_dev_20': vwap_dev_20,
    'vol_ratio_10_30': vol_ratio_10_30, 'downside_vol_ratio_20': downside_vol_ratio_20,
}

results = {}
for fid, fn in candidates.items():
    ta = time.time()
    panel = factor_to_panel(fn, prices)
    m = validate(fid, panel)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        json.dump(results, open('scripts/miner_3_20260730_results_round15.json', 'w'), indent=1, default=str)
        continue
    rm = to_rank_matrix(panel)
    rho, lib_id = fast_max_lib_corr(rm)
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
    json.dump(results, open('scripts/miner_3_20260730_results_round15.json', 'w'), indent=1, default=str)

print(f"\ndone in {time.time()-t0:.1f}s; saved scripts/miner_3_20260730_results_round15.json", flush=True)
