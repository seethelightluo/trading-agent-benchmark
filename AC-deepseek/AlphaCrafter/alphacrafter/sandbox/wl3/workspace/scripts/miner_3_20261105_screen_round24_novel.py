"""Round 24 (2026-11-05) novel factor screen for the 15-asset cross-asset universe.

Key fix vs round 22/23: the persisted library signal artifacts were built on a
2388-date grid, while the current API data (BTC/ETH history now starts
2021-04-13) yields a 2260-date canonical grid. Shape-exact artifact matching
therefore silently yields empty libraries (rho=0 false PASS). Here we REBUILD
all 18 EFFECTIVE library factor panels from their documented formulas on the
current data for a real per-date Spearman correlation gate (rho < 0.5).

Candidates (all novel vs the 18-factor library and prior rejected/evicted sets):
 1. mkt_corr_60            : mean pairwise 60d return correlation (systemic connectivity)
 2. volume_flow_ratio_20   : up-volume / down-volume over 20d (volume-confirmed flow)
 3. eth_btc_ratio_beta_60  : beta to ETH/BTC ratio return (crypto rotation)
 4. beta_asym_60           : down_beta / spx_beta (downside beta asymmetry ratio)
 5. raw_autocorr_20        : lag-1 autocorrelation of daily returns (20d)
 6. skew_20                : rolling 20d skewness of daily returns (vs intraday variant)
 7. sma_cross_20_60        : (MA20-MA60)/MA60 (moving-average crossover distance)
 8. cvar_term_20_60        : CVaR5(20d)/CVaR5(60d) (tail-risk term structure)
 9. ndx_spx_ratio_beta_60  : beta to NDX/SPX ratio return (tech rotation)
10. body_ratio_20          : mean |close-open|/(high-low) over 20d (candle conviction)

Gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max_abs_library_correlation < 0.5.
"""
import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           factor_to_panel, validate_factor, forward_returns,
                           rank_ic_series, signal_matrix, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2500)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
print(f"grid {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)

# ---------------- macro / market inputs ----------------
spx_c = prices['SPX']['close']; spx_r = spx_c.pct_change()
ndx_c = prices['NDX']['close']; ndx_r = ndx_c.pct_change()
hs300_r = prices['000300.SH']['close'].pct_change()
btc_r = prices['BTC']['close'].pct_change(); eth_r = prices['ETH']['close'].pct_change()
us10y_c = prices['US10Y']['close']; cn10y_c = prices['CN10Y']['close']
xau_r = prices['XAU']['close'].pct_change()
cop_r = prices['COPPER']['close'].pct_change()
wti_r = prices['WTI']['close'].pct_change()
dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
vix = load_index('VIX', prices=prices)
jpy = load_index('USDJPY', prices=prices)

def rb(r, m, w):
    """rolling beta of r on m with window w (series aligned via concat+dropna)"""
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)

def cond_beta(r, m, cond, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    z = z[cond(z['m'])]
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)

# ---------------- rebuild 18 library panels ----------------
def lib_dd_duration(df, s):
    c = df['close'].values
    hh = df['close'].rolling(120, min_periods=60).max().values
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(hh[i]):
            continue
        k = 0
        while (i - k) >= 0 and c[i - k] < hh[i] and k < 300:
            k += 1
        out[i] = np.log1p(k)
    return pd.Series(out, index=df.index)

def lib_streak(df, s):
    r = df['close'].pct_change()
    sg = np.sign(r.values)
    out = np.zeros(len(r))
    up = dn = 0
    for i in range(len(sg)):
        if np.isnan(sg[i]):
            up = dn = 0; out[i] = np.nan; continue
        if sg[i] > 0:
            up += 1; dn = 0
        elif sg[i] < 0:
            dn += 1; up = 0
        else:
            up = dn = 0
        out[i] = up - dn
    return pd.Series(out, index=r.index).rolling(60).max() / 60.0

def lib_sign_persist(df, s):
    r = df['close'].pct_change()
    same = (np.sign(r) == np.sign(r.shift(1))).astype(float)
    return same.rolling(20, min_periods=8).mean()

def lib_mom_accel(df, s):
    c = df['close']
    return (c.shift(5) / c.shift(65) - 1.0) - (c.shift(5) / c.shift(125) - 1.0)

def lib_intraday_skew(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20).skew()

def lib_hilo_pos(df, s):
    return (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())

def lib_hilo_vol(df, s):
    return ((df['close'].rolling(20).max() - df['close'].rolling(20).min()) / df['close']) / df['close'].pct_change().rolling(20).std()

def lib_vol_adj_mom(df, s):
    c = df['close']
    return (c / c.shift(25) - 1.0) / df['close'].pct_change().rolling(60).std()

def lib_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

def lib_vix_cond(df, s):
    if vix is None: return None
    b = rb(df['close'].pct_change(), vix['close'].pct_change(), 60)
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0))

def lib_dxy_cond(df, s):
    if dxy is None: return None
    b = rb(df['close'].pct_change(), dxy['close'].pct_change(), 60)
    return b * (dxy['close'] / dxy['close'].shift(20) - 1.0)

def lib_eurusd_cond(df, s):
    if eurusd is None: return None
    b = rb(df['close'].pct_change(), eurusd['close'].pct_change(), 60)
    return b * (eurusd['close'] / eurusd['close'].shift(20) - 1.0)

def lib_spx_beta(df, s):
    return rb(df['close'].pct_change(), spx_r, 60)

def lib_hs300_beta(df, s):
    return rb(df['close'].pct_change(), hs300_r, 60)

def lib_cn10y_beta(df, s):
    return rb(df['close'].pct_change(), cn10y_c.diff(), 60)

def lib_down_beta(df, s):
    return cond_beta(df['close'].pct_change(), spx_r, lambda m: m < 0, 60)

def lib_comm_basket(df, s):
    basket = (xau_r + cop_r + wti_r) / 3.0
    return rb(df['close'].pct_change(), basket, 60)

def lib_copper_gold(df, s):
    return rb(df['close'].pct_change(), cop_r - xau_r, 20)

LIB_BUILDERS = {
    'cn10y_beta_60': lib_cn10y_beta, 'comm_basket_beta_60': lib_comm_basket,
    'copper_gold_beta_20': lib_copper_gold, 'down_beta_60': lib_down_beta,
    'dxy_beta_cond_60x20': lib_dxy_cond, 'eurusd_beta_cond_60x20': lib_eurusd_cond,
    'hilo_pos_60': lib_hilo_pos, 'hilo_vol_ratio_20': lib_hilo_vol,
    'hs300_beta_60': lib_hs300_beta, 'intraday_ret_skew_20': lib_intraday_skew,
    'mom_accel_60_120': lib_mom_accel, 'sign_persist_20': lib_sign_persist,
    'spx_beta_60': lib_spx_beta, 'streak_60': lib_streak,
    'vix_beta_cond_60x20': lib_vix_cond, 'vol_adj_mom_20_60': lib_vol_adj_mom,
    'vol_of_vol20x60': lib_vov,
    'dd_duration_120_resid': lib_dd_duration,
}

lib_panels = {}
for fid, fn in LIB_BUILDERS.items():
    try:
        p = factor_to_panel(fn, prices)
        if len(p) > 0:
            lib_panels[fid] = p
            print(f"  lib {fid}: {p.shape}", flush=True)
        else:
            print(f"  lib {fid}: EMPTY", flush=True)
    except Exception as e:
        print(f"  lib {fid}: ERR {e}", flush=True)

# dd_duration_120_resid: per-date cross-sectional orthogonalization vs z-scored mom120
def mom120_panel():
    cols = {}
    for s, df in prices.items():
        c = df['close']
        cols[s] = (c.shift(5) / c.shift(125) - 1.0).astype(float)
    return pd.DataFrame(cols).sort_index()
mp = mom120_panel()
ddp = lib_panels.get('dd_duration_120_resid')
if ddp is not None:
    common = ddp.index.intersection(mp.index)
    resid = pd.DataFrame(index=common, columns=WATCHLIST)
    for t in common:
        x = ddp.loc[t].astype(float)
        z = mp.loc[t].astype(float)
        m = x.notna() & z.notna() & np.isfinite(x) & np.isfinite(z)
        if m.sum() >= 8:
            xv = x[m].values; zv = (z[m] - z[m].mean()) / (z[m].std(ddof=0) + 1e-12)
            b = np.cov(xv, zv)[0, 1] / (np.var(zv) + 1e-12)
            rv = xv - b * zv
            resid.loc[t, WATCHLIST[:len(m)]] = np.nan
            resid.loc[t, np.array(WATCHLIST)[m.values]] = rv
    lib_panels['dd_duration_120_resid'] = resid.dropna(how='all')
    print(f"  lib dd_duration_120_resid orthogonalized: {resid.shape}", flush=True)
print(f"rebuilt library panels: {len(lib_panels)}", flush=True)

# ---------------- candidate constructions ----------------
def f_mkt_corr_60(df, s):
    """mean pairwise correlation of this asset's returns vs all other assets (60d)"""
    r = df['close'].pct_change()
    corrs = []
    for s2, df2 in prices.items():
        if s2 == s:
            continue
        r2 = df2['close'].pct_change()
        z = pd.concat([r.rename('a'), r2.rename('b')], axis=1)
        corrs.append(z['a'].rolling(60).corr(z['b']))
    m = pd.concat(corrs, axis=1)
    return m.mean(axis=1, skipna=True)

def f_volume_flow_ratio_20(df, s):
    v = df['volume']; r = df['close'].pct_change()
    up = (v * (r > 0).astype(float)).rolling(20).sum()
    dn = (v * (r < 0).astype(float)).rolling(20).sum()
    return up / dn.replace(0, np.nan)

def f_eth_btc_ratio_beta_60(df, s):
    spread = eth_r - btc_r
    return rb(df['close'].pct_change(), spread, 60)

def f_beta_asym_60(df, s):
    r = df['close'].pct_change()
    db = cond_beta(r, spx_r, lambda m: m < 0, 60)
    tb = rb(r, spx_r, 60)
    return db / tb.replace(0, np.nan)

def f_raw_autocorr_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).corr(r.shift(1))

def f_skew_20(df, s):
    return df['close'].pct_change().rolling(20).skew()

def f_sma_cross_20_60(df, s):
    c = df['close']
    return (c.rolling(20).mean() - c.rolling(60).mean()) / c.rolling(60).mean()

def f_cvar_term_20_60(df, s):
    r = df['close'].pct_change()
    def cvar(win):
        q = r.rolling(win).quantile(0.05)
        return r.where(r <= q).rolling(win, min_periods=5).mean().abs()
    return cvar(20) / cvar(60).replace(0, np.nan)

def f_ndx_spx_ratio_beta_60(df, s):
    spread = ndx_r - spx_r
    return rb(df['close'].pct_change(), spread, 60)

def f_body_ratio_20(df, s):
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return (body / rng).rolling(20).mean()

cands = {
    'mkt_corr_60': (f_mkt_corr_60, 'mean pairwise 60d return correlation', 'systemic connectivity'),
    'volume_flow_ratio_20': (f_volume_flow_ratio_20, 'up-volume/down-volume 20d', 'volume-confirmed flow'),
    'eth_btc_ratio_beta_60': (f_eth_btc_ratio_beta_60, 'beta to ETH/BTC ratio ret 60d', 'crypto rotation'),
    'beta_asym_60': (f_beta_asym_60, 'down_beta/spx_beta 60d', 'downside beta asymmetry'),
    'raw_autocorr_20': (f_raw_autocorr_20, 'lag-1 autocorr of daily rets 20d', 'return serial dependence'),
    'skew_20': (f_skew_20, 'rolling 20d skewness of daily rets', 'return asymmetry'),
    'sma_cross_20_60': (f_sma_cross_20_60, '(MA20-MA60)/MA60', 'MA crossover distance'),
    'cvar_term_20_60': (f_cvar_term_20_60, 'CVaR5(20d)/CVaR5(60d)', 'tail-risk term structure'),
    'ndx_spx_ratio_beta_60': (f_ndx_spx_ratio_beta_60, 'beta to NDX/SPX ratio ret 60d', 'tech rotation'),
    'body_ratio_20': (f_body_ratio_20, 'mean |body|/range 20d', 'candle conviction'),
}

fwd_ret = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}

def max_lib_corr(panel):
    pm = signal_matrix(panel, grid)
    best, best_id, per_id = 0.0, None, {}
    for fid, lp in lib_panels.items():
        lm = signal_matrix(lp, grid)
        corrs = []
        for t in range(T):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values; yr = pd.Series(y[m]).rank().values
                xc = xr - xr.mean(); yc = yr - yr.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            per_id[fid] = r
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id, per_id

results = {}
for fid, (fn, desc, tag) in cands.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel is None or len(panel) == 0:
            print(f"{fid}: EMPTY panel -> skip", flush=True)
            continue
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        ic10 = rank_ic_series(panel, fwd_ret[10], 8)
        ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
        for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                         ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                         ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
            sub = ic10[(ic10.index >= pd.Timestamp(a)) & (ic10.index <= pd.Timestamp(b))]
            m[nm] = float(sub.mean()) if len(sub) > 30 else float('nan')
        recent = ic10[(ic10.index >= pd.Timestamp('2025-07-15')) & (ic10.index <= pd.Timestamp('2026-07-15'))]
        if len(recent) > 30:
            m['recent_1y_ic'] = float(recent.mean())
            sd = float(recent.std(ddof=1))
            m['recent_1y_icir'] = float(recent.mean() / sd) if sd > 0 else 0.0
        rho, rho_id, per_id = max_lib_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        m['per_factor_rho'] = {k: round(v, 3) for k, v in sorted(per_id.items(), key=lambda kv: -abs(kv[1]))[:4]}
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}, 'desc': desc, 'tag': tag}
        dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id}) [{time.time()-t1:.1f}s]", flush=True)
        print(f"  decay: {dec}", flush=True)
        print(f"  top rho: {m['per_factor_rho']}", flush=True)
        for nm in ['ic_2020_2022', 'ic_2023_2024', 'ic_2025_2026']:
            print(f"  {nm}: {m.get(nm, float('nan')):.4f}", flush=True)
        if 'recent_1y_ic' in m:
            print(f"  recent_1y: ic={m['recent_1y_ic']:.4f} icir={m['recent_1y_icir']:.4f}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_3_20261105_results_round24.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:24s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')})")
    else:
        print(f"{fid:24s} ERROR {r.get('error', '')[:80]}")
print(f"total time {time.time()-t0:.1f}s")
