"""Drill-down on ensemble factor drift: half-year IC path, recent coverage,
direction-consistent edge, and composite (weighted) ensemble IC.

Current date 2027-11-04; last completed trading day 2027-11-03.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2800)
max_date = max(dd.index.max() for dd in prices.values())

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

ENSEMBLE = {
    'down_beta_60': 0.2573213343, 'cn10y_beta_60': 0.131536378397,
    'spx_beta_60': 0.127590944702, 'vol_adj_mom_20_60': 0.114799296145,
    'dxy_beta_cond_60x20': 0.068016626906, 'hs300_beta_60': 0.063631685676,
    'hilo_vol_ratio_20': 0.060882870695, 'intraday_ret_skew_20': 0.059388262744,
    'comm_basket_beta_60': 0.059039461323, 'vol_of_vol20x60': 0.057793139111,
}
DIR = {
    'down_beta_60': 1, 'cn10y_beta_60': -1, 'spx_beta_60': 1,
    'vol_adj_mom_20_60': 1, 'dxy_beta_cond_60x20': 1, 'hs300_beta_60': -1,
    'hilo_vol_ratio_20': 1, 'intraday_ret_skew_20': 1,
    'comm_basket_beta_60': 1, 'vol_of_vol20x60': 1,
}

fwd10 = forward_returns(prices, 10)
OOS_END = pd.Timestamp('2027-11-03')

# ---- 1. half-year OOS IC path + recent coverage ----
print("\n=== half-year OOS IC (h=10) ===")
print(f"{'factor':22s} {'26H2':>8s} {'27H1':>8s} {'27H2':>8s} | {'n26H2':>5s} {'n27H1':>5s} {'n27H2':>5s} | recCoverage")
panels = {}
for fid, fn in FUNCS.items():
    panel = factor_to_panel(fn, prices)
    panels[fid] = panel
    ic = rank_ic_series(panel, fwd10, 8)
    seg = {}
    for lo, hi, lab in [(pd.Timestamp('2026-07-16'), pd.Timestamp('2026-12-31'), '26H2'),
                        (pd.Timestamp('2027-01-01'), pd.Timestamp('2027-06-30'), '27H1'),
                        (pd.Timestamp('2027-07-01'), OOS_END, '27H2')]:
        s = ic[(ic.index >= lo) & (ic.index <= hi)]
        seg[lab] = (float(s.mean()) if len(s) else np.nan, len(s))
    # recent-window coverage (2027-08-04..OOS_END): dates with >=8 valid factor rows
    sub = panel[(panel.index >= pd.Timestamp('2027-08-04')) & (panel.index <= OOS_END)]
    valid_dates = int((sub.notna().sum(axis=1) >= 8).sum())
    tot_dates = len(sub)
    print(f"{fid:22s} {seg['26H2'][0]:8.4f} {seg['27H1'][0]:8.4f} {seg['27H2'][0]:8.4f} | {seg['26H2'][1]:5d} {seg['27H1'][1]:5d} {seg['27H2'][1]:5d} | {valid_dates}/{tot_dates}", flush=True)

# ---- 2. direction-consistent IC and composite ensemble IC ----
print("\n=== direction-consistent edge + composite ensemble ===")
def zscore_panel(panel):
    z = panel.rank(axis=1, pct=True).apply(lambda r: r - 0.5, axis=1)
    return z

comp = pd.Series(0.0, index=next(iter(panels.values())).index)
for fid, w in ENSEMBLE.items():
    z = zscore_panel(panels[fid])
    comp = comp.add(w * DIR[fid] * z.reindex(comp.index), fill_value=0.0)
comp = comp.replace(0, np.nan)

def ic_stats(ic, lo, hi, label):
    s = ic[(ic.index >= lo) & (ic.index <= hi)]
    if len(s) < 30:
        print(f"{label:12s} insufficient n={len(s)}")
        return
    m = float(s.mean()); sd = float(s.std(ddof=1))
    print(f"{label:12s} IC={m:8.4f} ICIR={m/sd if sd>0 else 0:9.4f} hit={(s>0).mean():6.3f} n={len(s)}")

comp_ic = rank_ic_series(comp.to_frame('c'), fwd10, 8)
print("composite ensemble (weights+directions from factor_ensemble.json):")
ic_stats(comp_ic, VAL_START, VAL_END, 'warm-up')
ic_stats(comp_ic, pd.Timestamp('2026-07-16'), OOS_END, 'OOS')
ic_stats(comp_ic, pd.Timestamp('2027-08-04'), OOS_END, 'recent3m')

print("\nper-factor direction-consistent OOS edge (IC * sign(warm_ic)):")
warm_ic = {}
for fid in FUNCS:
    ic = rank_ic_series(panels[fid], fwd10, 8)
    w = ic[(ic.index >= VAL_START) & (ic.index <= VAL_END)]
    warm_ic[fid] = float(w.mean())
    o = ic[(ic.index >= pd.Timestamp('2026-07-16')) & (ic.index <= OOS_END)]
    om = float(o.mean())
    print(f"{fid:22s} warmIC={warm_ic[fid]:+8.4f} oosIC={om:+8.4f} dirConsist={om*np.sign(warm_ic[fid]):+8.4f}")

print(f"\ntotal {time.time()-t0:.1f}s")
