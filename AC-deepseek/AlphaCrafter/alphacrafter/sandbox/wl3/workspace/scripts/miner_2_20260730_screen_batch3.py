"""miner_2 exploration: screen novel factor candidates (batch 3) + gate-style rho audit for batch-1 winners.

Batch3 ideas:
 1. us10y_beta_60          : rolling 60d beta of asset ret to US10Y ret (untapped US yield beta)
 2. updown_vol_asym_60     : 60d downside-vol / upside-vol ratio (fixed implementation)
 3. vol_beta_60            : beta of asset 20d realized vol to SPX 20d realized vol (vol co-movement)
 4. xsec_beta_60           : rolling 60d beta to breadth (longer window variant)
 5. breadth_mom_20         : 20d asset return minus cross-sectional mean 20d return (relative strength)
 6. skew_60                : rolling 60d skewness of daily returns
 7. corr_xau_20            : rolling 20d corr of asset ret with XAU ret (safe-haven linkage)
 8. maxdd_recovery_60      : 60d max drawdown depth residual vs 60d vol (risk-adjusted drawdown)
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import factor_common as fc

prices = fc.load_prices(days=2000)

lib_ids = ['cn10y_beta_60','copper_gold_beta_20','dd_duration_120_resid','down_beta_60',
           'dxy_beta_cond_60x20','eurusd_beta_cond_60x20','hilo_pos_60','hs300_beta_60',
           'intraday_ret_skew_20','mom_accel_60_120','sign_persist_20','spx_beta_60',
           'streak_60','vix_beta_cond_60x20','vol_adj_mom_20_60','vol_of_vol20x60']
lib_mats = {}
for fid in lib_ids:
    p = f'factors/{fid}_signal.npy'
    try:
        lib_mats[fid] = np.load(p)
    except Exception:
        pass
grid = fc.canonical_grid(prices)

def max_rho(panel):
    m = fc.signal_matrix(panel, grid)
    best = 0.0; best_id = None; allr = {}
    for fid, lm in lib_mats.items():
        if lm.shape != m.shape:
            continue
        corrs = []
        for i in range(len(grid)):
            x, y = m[i], lm[i]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 8:
                r = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            r = float(np.mean(corrs))
            allr[fid] = r
            if abs(r) > best:
                best = abs(r); best_id = fid
    return best, best_id, allr

# ---------- 1: US10Y beta 60 ----------
def f_us10y_beta(df, s):
    r = df['close'].pct_change(); ur = prices['US10Y']['close'].pct_change()
    z = pd.concat([r.rename('r'), ur.rename('u')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['u']) / z['u'].rolling(60).var()
    return b

# ---------- 2: down/up vol asym ----------
def f_ud_vol(df, s):
    r = df['close'].pct_change()
    dn = r.clip(upper=0).rolling(60, min_periods=20).std()
    up = r.clip(lower=0).rolling(60, min_periods=20).std()
    return dn / up

# ---------- 3: vol beta to SPX vol ----------
spx_vol = prices['SPX']['close'].pct_change().rolling(20).std().sort_index()
def f_vol_beta(df, s):
    v = df['close'].pct_change().rolling(20).std()
    z = pd.concat([v.rename('v'), spx_vol.rename('sv')], axis=1).dropna()
    b = z['v'].rolling(60).cov(z['sv']) / z['sv'].rolling(60).var()
    return b

# ---------- 4: xsec beta 60 ----------
xret = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
breadth = xret.mean(axis=1)
def f_xsec_beta60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), breadth.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()
    return b

# ---------- 5: breadth-relative momentum ----------
breadth_ret20 = xret.rolling(20).mean().mean(axis=1)
def f_breadth_mom(df, s):
    r20 = df['close'] / df['close'].shift(20) - 1.0
    return r20 - breadth_ret20.reindex(df.index)

# ---------- 6: skew 60 ----------
def f_skew60(df, s):
    return df['close'].pct_change().rolling(60, min_periods=30).skew()

# ---------- 7: corr with XAU ----------
xau_ret = prices['XAU']['close'].pct_change().sort_index()
def f_corr_xau(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xau_ret.rename('x')], axis=1).dropna()
    return z['r'].rolling(20).corr(z['x'])

# ---------- 8: drawdown depth risk-adjusted ----------
def f_dd_risk(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    v = df['close'].pct_change().rolling(60, min_periods=30).std()
    return dd / v

cands = {
    'us10y_beta_60': f_us10y_beta,
    'updown_vol_asym_60': f_ud_vol,
    'vol_beta_60': f_vol_beta,
    'xsec_beta_60': f_xsec_beta60,
    'breadth_mom_20': f_breadth_mom,
    'skew_60': f_skew60,
    'corr_xau_20': f_corr_xau,
    'dd_risk_60': f_dd_risk,
}

results = {}
for fid, fn in cands.items():
    try:
        panel = fc.factor_to_panel(fn, prices)
        m = fc.validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: INSUFFICIENT DATA"); continue
        rho, rid, allr = max_rho(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rid
        results[fid] = m
        print(f"\n=== {fid} === panel {panel.shape}")
        print(f"  IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f}")
        print(f"  max_rho={rho:.3f} vs {rid}")
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'}  rho_ok={rho < 0.5}")
    except Exception as e:
        print(f"{fid}: ERROR {e}")

json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'decay_ic_by_horizon'} for k, v in results.items()},
          open('scripts/miner_2_20260730_screen_batch3.json', 'w'), indent=1, default=str)
print("\nSaved screening results.")
