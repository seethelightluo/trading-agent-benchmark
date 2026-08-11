"""Round 28 (2027-08-26) novel factor screen for the 15-asset cross-asset universe.

Novel candidates vs the EFFECTIVE library and prior rejected/evicted/quarantined sets:
 1. return_autocorr_20     : 20d lag-1 autocorrelation of daily returns (trend persistence)
 2. updown_vol_ratio_20    : 20d upside semi-dev / downside semi-dev (volatility asymmetry; mean-based
                             gain_loss_asym_20 was evicted, this is vol-based)
 3. sharpe_20              : 20d mean/std of daily returns (return-to-risk; distinct from vol_adj_mom_20_60
                             which is momentum/vol on different lookbacks)
 4. kurtosis_20_60         : 20d excess-kurtosis / 60d excess-kurtosis (tail-regime change)
 5. gap_vol_20             : 20d std of overnight gaps (gap volatility; gap_dir_20 was direction-based)
 6. range_ratio_5_60       : mean (high-low)/close 5d vs 60d (activity burst; vol_term_5_60 in round 27
                             used return std, this uses range)
 7. vol_volume_corr_20     : 20d rolling corr(|ret|, volume) (volume-volatility coupling;
                             vol_price_corr_60 was corr(vol, price level), evicted)
 8. ndx_spx_ratio_beta_60  : 60d beta of asset rets to NDX/SPX ratio daily change (growth/tech rotation)
 9. eth_btc_ratio_beta_60  : 60d beta of asset rets to ETH/BTC ratio daily change (altcoin risk appetite)
10. rates_spread_beta_60   : 60d beta of asset rets to (US10Y-CN10Y) spread daily change (rates divergence)
11. mom_ratio_5_20         : 5d momentum (skip1) / 20d momentum (skip5) (short vs medium momentum tilt)
12. corr_ew_basket_20      : 20d correlation of asset rets with equal-weight basket of the other 14 assets

Gate (benchmark-wide): |IC10| >= 0.007, |ICIR10| >= 0.084 on warm-up 2020-01-01..2026-07-15,
max_abs_library_correlation < 0.5 vs library computed from REAL persisted signal artifacts.
Extended OOS robustness: IC10 on 2026-07-16..2027-08-25.
"""
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, factor_to_panel, forward_returns,
                           rank_ic_series, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2800)
print(f"assets loaded: {len(prices)} | {time.time()-t0:.1f}s", flush=True)

rets = {s: df['close'].pct_change() for s, df in prices.items()}
spx_r = rets['SPX']; xau_r = rets['XAU']; wti_r = rets['WTI']

# ---- cross-asset ratio series (from tradable closes) ----
ndx_spx = prices['NDX']['close'] / prices['SPX']['close']
eth_btc = prices['ETH']['close'] / prices['BTC']['close']
rates_spread = prices['US10Y']['close'] - prices['CN10Y']['close']
ndx_spx_chg = ndx_spx.pct_change()
eth_btc_chg = eth_btc.pct_change()
rates_spread_chg = rates_spread.diff()


def rb(r, m, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)


def rc(r, m, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    c = z['r'].rolling(w).corr(z['m'])
    return c.reindex(r.index)


# ---------------- candidate factor functions ----------------
def f_autocorr(df, s):
    r = df['close'].pct_change()
    a = r; b = r.shift(1)
    z = pd.concat([a.rename('a'), b.rename('b')], axis=1)
    c = z['a'].rolling(20, min_periods=12).corr(z['b'])
    return c


def f_updown_vol(df, s):
    r = df['close'].pct_change()
    m = r.rolling(20, min_periods=10).mean()
    up = r.where(r > m, 0.0) - m.where(r > m, np.nan)
    dn = r.where(r < m, 0.0) - m.where(r < m, np.nan)
    usd = (up.pow(2).rolling(20, min_periods=10).mean()).pow(0.5)
    dsd = (dn.pow(2).rolling(20, min_periods=10).mean()).pow(0.5)
    return (usd / dsd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_sharpe(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(20, min_periods=10).mean()
    sd = r.rolling(20, min_periods=10).std()
    return (mu / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_kurt_ratio(df, s):
    r = df['close'].pct_change()
    k20 = r.rolling(20, min_periods=12).kurt()
    k60 = r.rolling(60, min_periods=30).kurt()
    return (k20 / k60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_gap_vol(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    return gap.rolling(20, min_periods=10).std()


def f_range_ratio(df, s):
    rng = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    r5 = rng.rolling(5, min_periods=3).mean()
    r60 = rng.rolling(60, min_periods=30).mean()
    return (r5 / r60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_vol_vol_corr(df, s):
    r = df['close'].pct_change()
    av = r.abs()
    vol = df['volume'].astype(float)
    z = pd.concat([av.rename('a'), vol.rename('v')], axis=1)
    return z['a'].rolling(20, min_periods=10).corr(z['v'])


def f_ndxspx_beta(df, s):
    return rb(df['close'].pct_change(), ndx_spx_chg, 60)


def f_ethbtc_beta(df, s):
    return rb(df['close'].pct_change(), eth_btc_chg, 60)


def f_ratespread_beta(df, s):
    return rb(df['close'].pct_change(), rates_spread_chg, 60)


def f_mom_ratio(df, s):
    r = df['close'].pct_change()
    m5 = r.rolling(5).sum().shift(1)
    m20 = r.rolling(20).sum().shift(5)
    return (m5 / m20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_corr_ew(df, s):
    r = df['close'].pct_change()
    others = [rs for ss, rs in rets.items() if ss != s]
    ew = pd.concat(others, axis=1).mean(axis=1, skipna=True)
    return rc(r, ew, 20)


cands = {
    'return_autocorr_20':     (f_autocorr, '20d lag-1 autocorr of daily returns', 'microstructure/trend'),
    'updown_vol_ratio_20':    (f_updown_vol, 'upside/downside semi-dev ratio (20d)', 'volatility'),
    'sharpe_20':              (f_sharpe, '20d mean/std daily return', 'return-to-risk'),
    'kurtosis_20_60':         (f_kurt_ratio, '20d/60d excess kurtosis ratio', 'volatility/tail'),
    'gap_vol_20':             (f_gap_vol, '20d std of overnight gaps', 'microstructure'),
    'range_ratio_5_60':       (f_range_ratio, '5d/60d mean range ratio', 'volatility/activity'),
    'vol_volume_corr_20':     (f_vol_vol_corr, '20d corr(|ret|, volume)', 'volume'),
    'ndx_spx_ratio_beta_60':  (f_ndxspx_beta, '60d beta vs NDX/SPX ratio chg', 'beta/rotation'),
    'eth_btc_ratio_beta_60':  (f_ethbtc_beta, '60d beta vs ETH/BTC ratio chg', 'beta/crypto'),
    'rates_spread_beta_60':   (f_ratespread_beta, '60d beta vs US10Y-CN10Y spread chg', 'beta/rates'),
    'mom_ratio_5_20':         (f_mom_ratio, '5d(skip1)/20d(skip5) momentum ratio', 'momentum'),
    'corr_ew_basket_20':      (f_corr_ew, '20d corr vs EW basket of other assets', 'beta/market'),
}

# ---------------- library correlation gate via REAL persisted signal artifacts ----------------
lib_arts = {}
for p in sorted(Path('factors').glob('*.json')):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    fid = d.get('factor_id')
    art = d.get('signal_artifact')
    if not fid or not art or fid == 'factor_ensemble':
        continue
    ap = Path('factors') / art
    if not ap.exists():
        continue
    g = d.get('signal_artifact_grid', {})
    try:
        arr = np.load(ap, allow_pickle=False)
    except Exception:
        continue
    if arr.shape[1] != len(WATCHLIST):
        continue
    lib_arts[fid] = (arr, g)
print(f"library artifacts loaded: {len(lib_arts)}", flush=True)

all_dates = sorted(set().union(*[set(df.index) for df in prices.values()]))
lib_panels = {}
for fid, (arr, g) in lib_arts.items():
    g_start, g_end, g_n = pd.Timestamp(g['start']), pd.Timestamp(g['end']), int(g['n_dates'])
    gidx = pd.DatetimeIndex([d for d in all_dates if g_start <= d <= g_end])
    if len(gidx) != g_n or arr.shape[0] != g_n:
        print(f"  skip artifact {fid}: grid n={g_n} arr n={arr.shape[0]} reconstructed={len(gidx)}", flush=True)
        continue
    lib_panels[fid] = pd.DataFrame(arr, index=gidx, columns=WATCHLIST)
print(f"library panels usable: {len(lib_panels)}", flush=True)


def lib_corr(panel):
    best, best_id, per = 0.0, None, {}
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


# precompute forward returns once
fwd = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}


def validate(panel):
    """Admission metrics on warm-up window + extended OOS stats."""
    ic = rank_ic_series(panel, fwd[10], 8)
    ic10 = ic[(ic.index >= VAL_START) & (ic.index <= VAL_END)]
    if len(ic10) < 100:
        return None
    mean = float(ic10.mean()); sd = float(ic10.std(ddof=1))
    icir = mean / sd if sd > 0 else 0.0
    hit = float((ic10 > 0).mean()) if mean >= 0 else float((ic10 < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum() / (fac.shape[0] * fac.shape[1])) if fac.shape[0] else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        ih = rank_ic_series(panel, fwd[h], 8)
        ihw = ih[(ih.index >= VAL_START) & (ih.index <= VAL_END)]
        decay[str(h)] = float(ihw.mean()) if len(ihw) else float('nan')
    oos = ic[(ic.index >= pd.Timestamp('2026-07-16')) & (ic.index <= pd.Timestamp('2027-08-25'))]
    oos_ic = float(oos.mean()) if len(oos) > 20 else float('nan')
    oos_sd = float(oos.std(ddof=1)) if len(oos) > 20 else float('nan')
    oos_icir = oos_ic / oos_sd if oos_sd and oos_sd > 0 else float('nan')
    # recent 1y (last 252 trading dates of full sample)
    rec = ic[ic.index >= pd.Timestamp('2026-08-26')]
    rec_ic = float(rec.mean()) if len(rec) > 40 else float('nan')
    rec_sd = float(rec.std(ddof=1)) if len(rec) > 40 else float('nan')
    rec_icir = rec_ic / rec_sd if rec_sd and rec_sd > 0 else float('nan')
    return {'ic': mean, 'icir': icir, 'ic_hit_ratio': hit,
            'n_ic_dates': int(len(ic10)), 'coverage_asset_days': cov,
            'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
            'decay_ic_by_horizon': decay,
            'oos_ic': oos_ic, 'oos_icir': oos_icir, 'n_oos_dates': int(len(oos)),
            'recent_1y_ic': rec_ic, 'recent_1y_icir': rec_icir, 'n_recent_dates': int(len(rec))}


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
        rho, rho_id, per = lib_corr(panel)
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
        print(f"  OOS(2026-07-16..2027-08-25): ic={m['oos_ic']:.4f} icir={m['oos_icir']:.4f} n={m['n_oos_dates']}", flush=True)
        print(f"  recent-1y: ic={m['recent_1y_ic']:.4f} icir={m['recent_1y_icir']:.4f} n={m['n_recent_dates']}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_1_20270826_results_round28.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:24s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')}) oos_ic={m.get('oos_ic', float('nan')):.4f} rec1y_ic={m.get('recent_1y_ic', float('nan')):.4f}")
    else:
        print(f"{fid:24s} ERROR {r.get('error', '')[:70]}")
print(f"total time {time.time()-t0:.1f}s")
