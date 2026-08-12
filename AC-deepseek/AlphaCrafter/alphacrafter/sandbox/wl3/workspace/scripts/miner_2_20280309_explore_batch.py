"""miner_2 2028-03-09: novel factor screen (batch).
Candidates (cross-asset, interpretable, not in current library / not covered by
recent miner_1/miner_3 screens):
 A. hurst_60           - trend persistence via variance-ratio Hurst estimate (60d)
 B. updown_beta_asym_60- SPX up-beta minus down-beta (crash-sensitivity asymmetry)
 C. yldspread_beta_60  - rolling beta vs CN10Y-US10Y spread daily change
 D. rv_term_10_60      - vol term structure RV10/RV60 (short vs long vol)
 E. pain_index_60      - mean drawdown depth over 60d (area under water)
 F. overnight_ret_20   - mean overnight return (open/prev_close-1) over 20d
Admission: |IC10|>=0.007 and |ICIR10|>=0.084 on 2020-01-01..2026-07-15.
Also reports online-window (2026-07-16..2028-03-08) IC for drift/timeliness.
"""
import sys, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START, VAL_END,
                           factor_to_panel, forward_returns, rank_ic_series)

t0 = time.time()
prices = load_prices(days=2600)
vix = load_index('VIX', days=2600, prices=prices)
dxy = load_index('DXY', days=2600, prices=prices)
print(f"data loaded {time.time()-t0:.1f}s; max visible: {max(d.index.max() for d in prices.values()).date()}", flush=True)
for s in WATCHLIST:
    df = prices.get(s)
    print(f"  {s}: n={len(df) if df is not None else 0} vol_frac={df['volume'].notna().mean():.2f}" if df is not None else f"  {s}: MISSING")

R = {s: df['close'].pct_change() for s, df in prices.items()}
RV10 = {s: df['close'].pct_change().rolling(10).std() for s, df in prices.items()}
RV60 = {s: df['close'].pct_change().rolling(60).std() for s, df in prices.items()}

# ---------- candidate factor definitions ----------
def f_hurst(df, s, w=60):
    r = df['close'].pct_change()
    r1 = r.rolling(w).var()
    r2 = (r + r.shift(1)).rolling(w).var()
    vr = r2 / (2.0 * r1)
    hurst = 0.5 * (1.0 + np.log2(vr.clip(lower=1e-12)))
    return hurst.clip(0, 1)

def f_updown_beta(df, s, w=60):
    spx = R['SPX']; r = R[s]
    z = pd.concat([r.rename('r'), spx.rename('m')], axis=1).dropna()
    up = z[z['m'] > 0]
    dn = z[z['m'] < 0]
    b_up = (up['r'].rolling(w, min_periods=20).cov(up['m']) / up['m'].rolling(w, min_periods=20).var()).reindex(z.index)
    b_dn = (dn['r'].rolling(w, min_periods=20).cov(dn['m']) / dn['m'].rolling(w, min_periods=20).var()).reindex(z.index)
    return (b_up - b_dn)

def f_yldspread_beta(df, s, w=60):
    sp = prices['CN10Y']['close'] - prices['US10Y']['close']
    dsp = sp.diff()
    r = R[s]
    z = pd.concat([r.rename('r'), dsp.rename('m')], axis=1).dropna()
    return (z['r'].rolling(w, min_periods=40).cov(z['m']) / z['m'].rolling(w, min_periods=40).var()).reindex(z.index)

def f_rv_term(df, s):
    return RV10[s] / RV60[s]

def f_pain_index(df, s, w=60):
    roll_max = df['close'].rolling(w).max()
    dd = df['close'] / roll_max - 1.0
    return dd.rolling(w).mean()

def f_overnight(df, s, w=20):
    on = df['open'] / df['close'].shift(1) - 1.0
    return on.rolling(w).mean()

CANDIDATES = {
    'hurst_60': f_hurst,
    'updown_beta_asym_60': f_updown_beta,
    'yldspread_beta_60': f_yldspread_beta,
    'rv_term_10_60': f_rv_term,
    'pain_index_60': f_pain_index,
    'overnight_ret_20': f_overnight,
}

WARM = (VAL_START, VAL_END)
ONLN = (pd.Timestamp('2026-07-16'), pd.Timestamp('2028-03-08'))

def ic_stats(panel, start, end, h=10, min_valid=8):
    fwd = forward_returns(prices, h)
    ic = rank_ic_series(panel, fwd, min_valid)
    ic = ic[(ic.index >= start) & (ic.index <= end)]
    if len(ic) < 60:
        return None
    mu = float(ic.mean()); sd = float(ic.std(ddof=1))
    return dict(ic=mu, icir=mu / sd if sd > 0 else 0.0, n=len(ic),
                hit=float((ic > 0).mean()))

print(f"\n{'factor':24s} | {'warm_ic':>8s} {'warm_icir':>9s} {'warm_hit':>7s} | {'onln_ic':>8s} {'onln_icir':>9s} | gate")
out = {}
for fid, fn in CANDIDATES.items():
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid:24s} EMPTY"); continue
    w = ic_stats(panel, *WARM); o = ic_stats(panel, *ONLN)
    out[fid] = {'warm': w, 'online': o, 'n_cols': panel.shape[1]}
    ww = f"{w['ic']:+.4f}" if w else "n/a"; wi = f"{w['icir']:+.3f}" if w else "n/a"
    wh = f"{w['hit']:.2f}" if w else "n/a"
    oo = f"{o['ic']:+.4f}" if o else "n/a"; oi = f"{o['icir']:+.3f}" if o else "n/a"
    gate = ""
    if w:
        gate = "PASS" if (abs(w['ic']) >= 0.007 and abs(w['icir']) >= 0.084) else "fail"
    print(f"{fid:24s} | {ww:>8s} {wi:>9s} {wh:>7s} | {oo:>8s} {oi:>9s} | {gate}")

import json
json.dump(out, open('scripts/miner_2_20280309_explore_batch.json', 'w'), indent=1, default=str)
print(f"\ndone {time.time()-t0:.1f}s")
