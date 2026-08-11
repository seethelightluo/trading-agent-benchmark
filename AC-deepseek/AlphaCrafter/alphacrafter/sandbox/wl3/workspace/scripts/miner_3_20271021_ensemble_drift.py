"""Re-validation / drift check of the 10 currently-selected ensemble factors.

Window A (warm-up admission): 2020-01-01..2026-07-15
Window B (live OOS drift)   : 2026-07-16..2027-10-20 (current date minus 1 day)

Trader flagged 3 consecutive 120d backtest Sharpe < 0.4 (2027-09-23, 10-07, 10-21);
this checks whether the selected factor signals have decayed in the live period.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2500)
print(f"assets loaded: {len(prices)} | {time.time()-t0:.1f}s", flush=True)

spx_r = prices['SPX']['close'].pct_change()
hs300_r = prices['000300.SH']['close'].pct_change()
cn10y_d = prices['CN10Y']['close'].diff()
dxy = load_index('DXY', prices=prices)
dxy_r = dxy['close'].pct_change() if dxy is not None else None
comm_r = pd.concat([prices[s]['close'].pct_change().rename(s) for s in ['XAU', 'COPPER', 'WTI']], axis=1).mean(axis=1)


def rb(r, m, w, cond=None):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    if cond is not None:
        z = z[cond.reindex(z.index).astype(bool)]
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)


def f_down_beta_60(df, s):
    r = df['close'].pct_change()
    return rb(r, spx_r, 60, cond=spx_r < 0)

def f_cn10y_beta_60(df, s):
    r = df['close'].pct_change()
    return rb(r, cn10y_d, 60)

def f_spx_beta_60(df, s):
    r = df['close'].pct_change()
    return rb(r, spx_r, 60)

def f_vol_adj_mom_20_60(df, s):
    r = df['close'].pct_change()
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = r.rolling(60).std()
    return (mom / v).replace([np.inf, -np.inf], np.nan)

def f_dxy_beta_cond_60x20(df, s):
    if dxy_r is None:
        return None
    r = df['close'].pct_change()
    b = rb(r, dxy_r, 60)
    return b * (dxy['close'] / dxy['close'].shift(20) - 1.0)

def f_hs300_beta_60(df, s):
    r = df['close'].pct_change()
    return rb(r, hs300_r, 60)

def f_hilo_vol_ratio_20(df, s):
    c = df['close']
    rng = (c.rolling(20).max() - c.rolling(20).min()) / c
    v = c.pct_change().rolling(20).std()
    return (rng / v).replace([np.inf, -np.inf], np.nan)

def f_intraday_ret_skew_20(df, s):
    ir = df['close'] / df['open'] - 1.0
    return ir.rolling(20, min_periods=12).skew()

def f_comm_basket_beta_60(df, s):
    r = df['close'].pct_change()
    return rb(r, comm_r, 60)

def f_vol_of_vol20x60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).std().rolling(60).std()

FUNCS = {
    'down_beta_60': f_down_beta_60,
    'cn10y_beta_60': f_cn10y_beta_60,
    'spx_beta_60': f_spx_beta_60,
    'vol_adj_mom_20_60': f_vol_adj_mom_20_60,
    'dxy_beta_cond_60x20': f_dxy_beta_cond_60x20,
    'hs300_beta_60': f_hs300_beta_60,
    'hilo_vol_ratio_20': f_hilo_vol_ratio_20,
    'intraday_ret_skew_20': f_intraday_ret_skew_20,
    'comm_basket_beta_60': f_comm_basket_beta_60,
    'vol_of_vol20x60': f_vol_of_vol20x60,
}

fwd10 = forward_returns(prices, 10)


def stats(ic, lo, hi):
    ic = ic[(ic.index >= lo) & (ic.index <= hi)]
    if len(ic) < 30:
        return None
    mean = float(ic.mean())
    sd = float(ic.std(ddof=1))
    icir = mean / sd if sd > 0 else 0.0
    hit = float((ic > 0).mean())
    return {'ic': mean, 'icir': icir, 'hit': hit, 'n': len(ic)}


print(f"\n{'factor':22s} {'warmIC':>8s} {'warmICIR':>9s} {'oosIC':>8s} {'oosICIR':>9s} {'oosHit':>7s} {'nOOS':>5s}  verdict", flush=True)
out = {}
for fid, fn in FUNCS.items():
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    ic = rank_ic_series(panel, fwd10, 8)
    w = stats(ic, VAL_START, VAL_END)
    o = stats(ic, pd.Timestamp('2026-07-16'), pd.Timestamp('2027-10-20'))
    if w is None or o is None:
        print(f"{fid:22s} insufficient", flush=True)
        continue
    drift = (o['icir'] - w['icir'])
    verdict = 'DECAYED' if (abs(o['ic']) < 0.007 or abs(o['icir']) < 0.084) else ('OK' if o['icir'] * w['icir'] > 0 else 'SIGN_FLIP')
    out[fid] = {'warm': w, 'oos': o, 'verdict': verdict}
    print(f"{fid:22s} {w['ic']:8.4f} {w['icir']:9.4f} {o['ic']:8.4f} {o['icir']:9.4f} {o['hit']:7.3f} {o['n']:5d}  {verdict} (drift {drift:+.3f}) [{time.time()-t1:.1f}s]", flush=True)

import json
with open('scripts/miner_3_20271021_ensemble_drift.json', 'w') as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\ntotal {time.time()-t0:.1f}s")
