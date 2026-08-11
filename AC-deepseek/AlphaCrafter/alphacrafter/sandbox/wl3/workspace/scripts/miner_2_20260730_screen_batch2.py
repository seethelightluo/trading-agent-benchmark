"""miner_2 exploration: screen novel factor candidates (batch 2).

Ideas (novel vs current 16-factor library):
 1. days_since_60d_high  : age since last 60d high (fresh-high recency)
 2. gain_loss_ratio_60   : mean(up rets)/|mean(down rets)| over 60d (asymmetric payoff)
 3. vol_percentile_252   : 20d realized vol percentile rank within trailing 252d (regime-relative vol)
 4. parkinson_vr_20_60   : Parkinson(high-low) vol ratio 20d/60d (intraday vol term structure)
 5. max_dd_20            : max drawdown over trailing 20d (downside depth)
 6. serial_corr_5        : autocorrelation of daily returns at lag 5 (weekly echo)
 7. crypto_beta_20       : rolling 20d beta of asset ret to crypto (BTC+ETH avg) ret
 8. wti_gold_beta_20     : rolling 20d beta of asset ret to (WTI - XAU) ret spread
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

# ---------- 1: days since 60d high ----------
def f_age_high(df, s):
    hi = df['close'].rolling(60).max()
    age = pd.Series(index=df.index, dtype=float)
    # count days since close last equaled rolling 60d high
    hit = (df['close'] >= hi)
    last_hit = hit.groupby((~hit).cumsum()).cumcount()
    # simpler: for each date, days since last date where close==rolling max
    recent_highs = df['close'].where(hit)
    days_since = (df.index - recent_highs.ffill()).days
    return days_since

# ---------- 2: gain/loss ratio 60d ----------
def f_gl_ratio(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(60).mean()
    dn = r.where(r < 0).rolling(60).mean()
    return up / dn.abs()

# ---------- 3: vol percentile 252 ----------
def f_vol_pct(df, s):
    v = df['close'].pct_change().rolling(20).std()
    pct = v.rolling(252).rank(pct=True)
    return pct

# ---------- 4: Parkinson vol ratio 20/60 ----------
def f_park_vr(df, s):
    hl = np.log(df['high'] / df['low'])
    pv = np.sqrt(hl**2 / (4 * np.log(2)))
    s20 = pv.rolling(20).mean()
    s60 = pv.rolling(60).mean()
    return s20 / s60

# ---------- 5: max drawdown 20 ----------
def f_maxdd(df, s):
    c = df['close']
    roll_max = c.rolling(20, min_periods=5).max()
    dd = c / roll_max - 1.0
    return dd

# ---------- 6: serial correlation lag 5 ----------
def f_serial5(df, s):
    r = df['close'].pct_change()
    a = r.rolling(60).corr(r.shift(5))
    return a

# ---------- 7: crypto beta 20 ----------
crypto = pd.DataFrame({s: prices[s]['close'].pct_change() for s in ['BTC', 'ETH']}).mean(axis=1).sort_index()
def f_crypto_beta(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), crypto.rename('c')], axis=1).dropna()
    b = z['r'].rolling(20).cov(z['c']) / z['c'].rolling(20).var()
    return b

# ---------- 8: WTI-Gold beta 20 ----------
def f_wti_gold(df, s):
    r = df['close'].pct_change()
    w = prices['WTI']['close'].pct_change(); g = prices['XAU']['close'].pct_change()
    spread = (w - g)
    z = pd.concat([r.rename('r'), spread.rename('s')], axis=1).dropna()
    b = z['r'].rolling(20).cov(z['s']) / z['s'].rolling(20).var()
    return b

cands = {
    'days_since_high_60': f_age_high,
    'gain_loss_ratio_60': f_gl_ratio,
    'vol_percentile_252': f_vol_pct,
    'parkinson_vr_20_60': f_park_vr,
    'max_dd_20': f_maxdd,
    'serial_corr_5': f_serial5,
    'crypto_beta_20': f_crypto_beta,
    'wti_gold_beta_20': f_wti_gold,
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
          open('scripts/miner_2_20260730_screen_batch2.json', 'w'), indent=1, default=str)
print("\nSaved screening results.")
