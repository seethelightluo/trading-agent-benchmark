"""miner_1 (2034-02-06): candidate factor screen.

Batch-evaluate novel candidate factors across the 15-asset tradable universe.
Admission gate (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Also check CN10Y data flatness (rate_beta_cn10y has 0 recent IC dates).
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

log("loading panels...")
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt = rets.mean(axis=1)
log(f"closes {closes.shape} {closes.index.min().date()} -> {closes.index.max().date()}")

# ---- CN10Y flatness check ----
cn = closes["CN10Y"]
cn_chg = cn.diff().abs()
print("=" * 72)
print("CN10Y FLATNESS CHECK")
print("=" * 72)
print(f"CN10Y last={cn.iloc[-1]:.6f}; last change >1e-8 on: "
      f"{cn.index[cn_chg > 1e-8][-1].date() if (cn_chg > 1e-8).any() else 'never'}")
print(f"n_days with |change|>1e-8 in last 5y: {int((cn_chg[cn_chg.index >= '2029-01-01'] > 1e-8).sum())}")
print(f"n_days with |change|>1e-8 in last 3y: {int((cn_chg[cn_chg.index >= '2031-01-01'] > 1e-8).sum())}")

# ---- build candidate factors ----
log("building candidates...")
C = {}

# B. short-term reversal 5d (negated 5d return)
C["short_rev_5d"] = -rets.rolling(5).sum()

# C. vol ratio 10/60 (short-term vol spike)  [expect negative IC]
vol10 = rets.rolling(10).std(ddof=0)
vol60 = rets.rolling(60).std(ddof=0)
vol20 = rets.rolling(20).std(ddof=0)
C["vol_ratio_10_60"] = vol10 / vol60

# D. downside vol ratio 20d: std of negative returns / total std
def downside_ratio(r, win=20):
    neg = r.where(r < 0, 0.0)
    pos = r.where(r > 0, 0.0)
    dv = neg.rolling(win).std(ddof=0)
    pv = pos.rolling(win).std(ddof=0)
    return dv / (dv + pv + 1e-12)
C["downside_ratio_20d"] = downside_ratio(rets, 20)

# E. skewness 60d
C["skew_60d"] = rets.rolling(60).skew()

# F. 20d high-low range / close (range proxy)
hl = pd.DataFrame({a: (panels[a]["high"].rolling(20).max() - panels[a]["low"].rolling(20).min())
                   for a in closes.columns}, index=closes.index)
C["range_hl_20d"] = hl / closes

# G. up/down asymmetry 20d: mean(up)/mean(|down|)
def updown_asym(r, win=20):
    up = r.where(r > 0, np.nan).rolling(win).mean()
    dn = r.where(r < 0, np.nan).rolling(win).mean()
    return up / (-dn + 1e-12)
C["updown_asym_20d"] = updown_asym(rets, 20)

# H. vol-of-vol ratio: vol20 / vol60 of vol
C["vol_of_vol_ratio_20_60"] = vol10.rolling(20).std(ddof=0) / (vol60.rolling(20).std(ddof=0) + 1e-12)

# I. max drawdown 20d (negative)
C["max_dd_20d"] = closes / closes.rolling(20).max() - 1.0

# J. trend strength: close / SMA60 - 1
C["trend_strength_60d"] = closes / closes.rolling(60).mean() - 1.0

# M. volume shock: vol_ma5 / vol_ma20 (per asset, using volume)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()
vol_ma5 = vol_panel.rolling(5).mean()
vol_ma20 = vol_panel.rolling(20).mean()
C["volume_shock_5_20"] = vol_ma5 / vol_ma20

# N. Amihud illiquidity 20d: mean(|ret|/volume)
amihud = (rets.abs() / (vol_panel + 1e-9)).rolling(20).mean()
C["amihud_illiquidity_20d"] = amihud

# O. price vs VWAP-20d deviation
vwap = (closes * vol_panel).rolling(20).sum() / vol_panel.rolling(20).sum()
C["price_vs_vwap_20d"] = closes / vwap - 1.0

# P. volume-confirmed momentum: mom20 * (vol_ma5/vol_ma20)
mom20 = closes / closes.shift(20) - 1.0
C["vol_confirmed_mom_20d"] = mom20 * (vol_ma5 / vol_ma20)

# Q. momentum quality: mom20 * (1 + rank of down-side ratio) -> skip, keep simple

# R. correlation with market 10d (recent regime comovement)
corr_mkt_10 = pd.DataFrame({a: rets[a].rolling(10).corr(mkt) for a in rets.columns}, index=rets.index)
C["corr_mkt_10d"] = corr_mkt_10

# S. downside beta 20d (short-window tail beta)
dn20 = np.minimum(mkt, 0.0)
C["dn_mkt_beta_20d"] = pd.DataFrame({a: (rets[a].rolling(20, min_periods=14).cov(dn20)
                                         / dn20.rolling(20, min_periods=14).var())
                                     for a in rets.columns}, index=rets.index)

# T. wti_beta_60d: beta vs WTI (energy beta)
wti_ret = rets["WTI"]
C["wti_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(wti_ret)
                                      / wti_ret.rolling(60, min_periods=40).var())
                                  for a in rets.columns}, index=rets.index)

# U. xau_beta_60d: beta vs XAU (safe-haven beta)
xau_ret = rets["XAU"]
C["xau_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(xau_ret)
                                      / xau_ret.rolling(60, min_periods=40).var())
                                  for a in rets.columns}, index=rets.index)

log("building library signals...")
library = library_signals(panels, closes, rets)
library["vol_adj_mom_accel_20x60"] = (mom20 - (closes / closes.shift(60) - 1.0)) / vol20
library["dn_mkt_beta_60d"] = C["dn_mkt_beta_20d"]  # placeholder replaced below
library.pop("dn_mkt_beta_60d")
library["dn_mkt_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn)
                                               / dn.rolling(60, min_periods=40).var())
                                           for a in rets.columns}, index=rets.index)
cn_ret = closes["CN10Y"].pct_change()
library["rate_beta_cn10y_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret)
                                                   / cn_ret.rolling(60, min_periods=40).var())
                                               for a in rets.columns}, index=rets.index)

fwd10 = forward_returns(closes, 10)
ADM_IC, ADM_ICIR = 0.0070, 0.0840

print("=" * 120)
print("CANDIDATE SCREEN (h=10, full history 2020..2034-02-03)")
print("=" * 120)
hdr = f"{'candidate':26s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covA':>5s} {'covD':>5s} {'turn':>5s} {'libCorr':>7s} {'decay5':>7s} {'decay10':>7s} {'decay20':>7s} {'pass':>4s}"
print(hdr)
print("-" * 120)

results = {}
for name, fp in C.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    if len(ics) == 0:
        print(f"{name:26s} no IC dates")
        continue
    m = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    dec = decay_profile(fp, closes)
    corr, key = max_library_corr(fp, library)
    p = abs(m["ic"]) >= ADM_IC and abs(m["icir"]) >= ADM_ICIR
    results[name] = {"metrics": m, "coverage": cov, "turnover": turn,
                     "decay": dec, "corr": corr, "corr_key": key, "pass": p}
    print(f"{name:26s} {m['ic']:8.4f} {m['icir']:7.3f} {m['ic_hit_ratio']:5.2f} "
          f"{m['n_ic_dates']:5d} {cov['coverage_asset_days']:5.2f} {cov['coverage_dates_ge8']:5.2f} "
          f"{turn:5.2f} {corr:7.3f} {dec.get('5', float('nan')):7.4f} {dec.get('10', float('nan')):7.4f} "
          f"{dec.get('20', float('nan')):7.4f} {'YES' if p else ''}")

print("-" * 120)
print("PASSING CANDIDATES (full history):")
for name, r in results.items():
    if r["pass"]:
        print(f"  {name}: IC={r['metrics']['ic']:.4f} ICIR={r['metrics']['icir']:.3f} "
              f"hit={r['metrics']['ic_hit_ratio']:.2f} n={r['metrics']['n_ic_dates']} "
              f"libCorr={r['corr']:.3f}({r['corr_key']})")

# recent-window robustness for candidates with |IC|>=0.005
print("=" * 120)
print("RECENT WINDOW CHECK (2031-01-01..2034-02-03) for candidates |IC|>=0.005 full")
print("=" * 120)
for name, r in results.items():
    if abs(r["metrics"]["ic"]) < 0.005:
        continue
    fp = C[name].replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    sub = ics[ics.index >= pd.Timestamp("2031-01-01")]
    if len(sub) == 0:
        print(f"{name:26s} recent3y: no IC dates")
        continue
    rm = summarize_ic(sub, expected_sign=1)
    rp = abs(rm["ic"]) >= ADM_IC and abs(rm["icir"]) >= ADM_ICIR
    print(f"{name:26s} recent3y IC={rm['ic']:+.4f} ICIR={rm['icir']:+.3f} hit={rm['ic_hit_ratio']:.2f} n={rm['n_ic_dates']:4d} pass={'YES' if rp else ''}")

log("done")
