"""miner_2 exploration: screen novel factor candidates (batch 1).

Ideas (all use only data visible at t; validation on warm-up 2020-01-01..2026-07-15):
 1. usdjpy_beta_cond_60x20 : rolling 60d beta of asset ret to USDJPY ret, conditioned on 20d USDJPY trend (untapped macro signal)
 2. usdcny_beta_cond_60x20 : same with USDCNY
 3. yldspread_beta_60       : rolling 60d beta of asset ret to d(US10Y - CN10Y) yield spread
 4. xsec_beta_20            : rolling 20d beta of asset ret to cross-sectional equal-weight avg ret (breadth co-movement)
 5. updown_vol_asym_60      : downside vol / upside vol ratio over 60d (asymmetry)
 6. kurt_60                 : rolling 60d excess kurtosis of daily returns
 7. amihud_illiq_20         : mean(|ret|/volume) over 20d, log-transformed
 8. corr_breadth_20         : rolling 20d corr of asset ret with cross-sectional avg ret (co-movement strength)
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import factor_common as fc

prices = fc.load_prices(days=2000)
print(f"Loaded {len(prices)} assets; date range {min(d.index.min() for d in prices.values())}..{max(d.index.max() for d in prices.values())}")

idx = {s: fc.load_index(s, prices=prices) for s in fc.INDEX_SIGNALS}
for s, df in idx.items():
    print(f"index {s}: {None if df is None else (df.index.min(), df.index.max(), len(df))}")

# --- library signal matrices (16 effective factors) for correlation audit ---
lib_ids = ['cn10y_beta_60','copper_gold_beta_20','dd_duration_120_resid','down_beta_60',
           'dxy_beta_cond_60x20','eurusd_beta_cond_60x20','hilo_pos_60','hs300_beta_60',
           'intraday_ret_skew_20','mom_accel_60_120','sign_persist_20','spx_beta_60',
           'streak_60','vix_beta_cond_60x20','vol_adj_mom_20_60','vol_of_vol20x60']
lib_mats = {}
for fid in lib_ids:
    p = f'factors/{fid}_signal.npy'
    try:
        lib_mats[fid] = np.load(p)
    except Exception as e:
        print(f"missing library artifact {fid}: {e}")
grid = fc.canonical_grid(prices)
print(f"canonical grid: {len(grid)} dates {grid.min()}..{grid.max()}")

def max_rho(panel):
    m = fc.signal_matrix(panel, grid)
    best = 0.0; best_id = None
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
            if abs(r) > best:
                best = abs(r); best_id = fid
    return best, best_id

# ---------- candidate 1: USDJPY conditional beta ----------
def f_usdjpy(df, s):
    if idx['USDJPY'] is None:
        return None
    r = df['close'].pct_change(); jr = idx['USDJPY']['close'].pct_change()
    z = pd.concat([r.rename('r'), jr.rename('j')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['j']) / z['j'].rolling(60).var()
    trend = idx['USDJPY']['close'] / idx['USDJPY']['close'].shift(20) - 1.0
    return (b * np.sign(trend)).reindex(z.index)

# ---------- candidate 2: USDCNY conditional beta ----------
def f_usdcny(df, s):
    if idx['USDCNY'] is None:
        return None
    r = df['close'].pct_change(); cr = idx['USDCNY']['close'].pct_change()
    z = pd.concat([r.rename('r'), cr.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()
    trend = idx['USDCNY']['close'] / idx['USDCNY']['close'].shift(20) - 1.0
    return (b * np.sign(trend)).reindex(z.index)

# ---------- candidate 3: US10Y-CN10Y yield spread beta ----------
def f_yldspread(df, s):
    r = df['close'].pct_change()
    us = prices['US10Y']['close']; cn = prices['CN10Y']['close']
    spread = us - cn
    dspread = spread.diff()
    z = pd.concat([r.rename('r'), dspread.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
    return b

# ---------- candidate 4: cross-sectional (breadth) beta ----------
xret = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
breadth = xret.mean(axis=1)
def f_xsec_beta(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), breadth.rename('b')], axis=1).dropna()
    b = z['r'].rolling(20).cov(z['b']) / z['b'].rolling(20).var()
    return b

# ---------- candidate 5: downside/upside vol asymmetry ----------
def f_ud_vol(df, s):
    r = df['close'].pct_change()
    dn = r[r < 0].rolling(60).std()
    up = r[r >= 0].rolling(60).std()
    return dn / up

# ---------- candidate 6: kurtosis 60d ----------
def f_kurt(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).kurt()

# ---------- candidate 7: Amihud illiquidity ----------
def f_amihud(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume']
    ill = (r / v).rolling(20).mean()
    return np.log1p(ill)

# ---------- candidate 8: corr with breadth (co-movement strength) ----------
def f_corr_breadth(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), breadth.rename('b')], axis=1).dropna()
    return z['r'].rolling(20).corr(z['b'])

cands = {
    'usdjpy_beta_cond_60x20': f_usdjpy,
    'usdcny_beta_cond_60x20': f_usdcny,
    'yldspread_beta_60': f_yldspread,
    'xsec_beta_20': f_xsec_beta,
    'updown_vol_asym_60': f_ud_vol,
    'kurt_60': f_kurt,
    'amihud_illiq_20': f_amihud,
    'corr_breadth_20': f_corr_breadth,
}

results = {}
for fid, fn in cands.items():
    try:
        panel = fc.factor_to_panel(fn, prices)
        m = fc.validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: INSUFFICIENT DATA"); continue
        rho, rid = max_rho(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rid
        results[fid] = m
        print(f"\n=== {fid} === panel {panel.shape} dates {panel.index.min()}..{panel.index.max()}")
        print(f"  IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f}")
        print(f"  max_rho={rho:.3f} vs {rid}")
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'}  rho_ok={rho < 0.5}")
    except Exception as e:
        print(f"{fid}: ERROR {e}")

json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'decay_ic_by_horizon'} for k, v in results.items()},
          open('scripts/miner_2_20260730_screen_batch1.json', 'w'), indent=1, default=str)
print("\nSaved screening results.")
