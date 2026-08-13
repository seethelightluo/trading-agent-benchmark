"""miner_1 (2034-03-20): candidate factor screen through visible 2034-03-17.

Batch-evaluate novel candidate factors across the 15-asset tradable universe.
Admission gate (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Also prints recent-window (2031+, 2033+) robustness and max library correlation.
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

C = {}

# ---- price/momentum family ----
mom5 = closes / closes.shift(5) - 1.0
mom10 = closes / closes.shift(10) - 1.0
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
mom120 = closes / closes.shift(120) - 1.0
vol10 = rets.rolling(10).std(ddof=0)
vol20 = rets.rolling(20).std(ddof=0)
vol60 = rets.rolling(60).std(ddof=0)

C["short_rev_5d"] = -mom5                                   # B. short-term reversal
C["trend_strength_60d"] = closes / closes.rolling(60).mean() - 1.0   # J. trend strength
C["trend_strength_60d_voladj"] = (closes / closes.rolling(60).mean() - 1.0) / vol20   # vol-adj trend
C["hi_lo_pos_60d"] = ((closes - closes.rolling(60).min()) /
                      (closes.rolling(60).max() - closes.rolling(60).min() + 1e-12))  # range position 60d
C["pct_off_52w_high"] = closes / closes.rolling(250).max() - 1.0   # 52w-high proximity
C["mom_quality_20d"] = mom20 * (1.0 - closes.rolling(20).std(ddof=0).rank(axis=1) / 15.0)  # low-vol x mom

# ---- volatility family ----
C["vol_ratio_10_60"] = vol10 / vol60                         # C. vol term structure
C["rel_vol_20d"] = vol20 / vol20.median(axis=1)              # relative vol vs cross-sectional median

def downside_ratio(r, win=20):
    neg = r.where(r < 0, 0.0)
    pos = r.where(r > 0, 0.0)
    dv = neg.rolling(win).std(ddof=0)
    pv = pos.rolling(win).std(ddof=0)
    return dv / (dv + pv + 1e-12)

C["downside_ratio_10d"] = downside_ratio(rets, 10)           # short-window downside ratio
C["skew_60d"] = rets.rolling(60).skew()                      # E. skewness 60d
C["range_hl_20d"] = (pd.DataFrame({a: (panels[a]["high"].rolling(20).max() - panels[a]["low"].rolling(20).min())
                                   for a in closes.columns}, index=closes.index) / closes)  # F. range proxy
C["intraday_range_ratio_20d"] = (pd.DataFrame({a: (panels[a]["high"] - panels[a]["low"]) / panels[a]["close"]
                                               for a in closes.columns}, index=closes.index)).rolling(20).mean()
C["updown_asym_20d"] = (rets.where(rets > 0, np.nan).rolling(20).mean() /
                        (-rets.where(rets < 0, np.nan).rolling(20).mean() + 1e-12))  # G. up/down asym
C["max_dd_20d"] = closes / closes.rolling(20).max() - 1.0    # I. max drawdown 20d
C["ret_consistency_20d"] = (rets > 0).rolling(20).mean()     # fraction of positive days
C["autocorr_5_20d"] = rets.rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) > 3 else np.nan, raw=False)  # mean-reversion tendency

# ---- volume family ----
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()
vol_ma5 = vol_panel.rolling(5).mean()
vol_ma20 = vol_panel.rolling(20).mean()
vol_ma60 = vol_panel.rolling(60).mean()
C["volume_shock_5_20"] = vol_ma5 / vol_ma20                  # M. volume shock
C["volume_trend_20_60"] = vol_ma20 / vol_ma60                # volume trend
C["amihud_illiquidity_20d"] = (rets.abs() / (vol_panel + 1e-9)).rolling(20).mean()  # N. Amihud
C["price_vs_vwap_20d"] = closes / ((closes * vol_panel).rolling(20).sum() / vol_panel.rolling(20).sum()) - 1.0  # O.
C["vol_confirmed_mom_20d"] = mom20 * (vol_ma5 / vol_ma20)    # P. volume-confirmed momentum

# ---- cross-asset beta family ----
C["corr_mkt_10d"] = pd.DataFrame({a: rets[a].rolling(10).corr(mkt) for a in rets.columns}, index=rets.index)  # R.
dn = np.minimum(mkt, 0.0)
C["dn_mkt_beta_20d"] = pd.DataFrame({a: (rets[a].rolling(20, min_periods=14).cov(dn)
                                         / dn.rolling(20, min_periods=14).var()) for a in rets.columns}, index=rets.index)  # S.
C["wti_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(rets["WTI"])
                                      / rets["WTI"].rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)  # T.
C["xau_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(rets["XAU"])
                                      / rets["XAU"].rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)  # U.
C["btc_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(rets["BTC"])
                                      / rets["BTC"].rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)  # crypto beta
C["spx_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(rets["SPX"])
                                      / rets["SPX"].rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)  # equity beta
C["us10y_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(rets["US10Y"])
                                        / rets["US10Y"].rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)  # bond beta
if "DXY" in panels:
    dxy_ret = panels["DXY"]["close"].astype(float).pct_change()
    C["dxy_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dxy_ret)
                                          / dxy_ret.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)
if "USDJPY" in panels:
    jpy_ret = panels["USDJPY"]["close"].astype(float).pct_change()
    C["usdjpy_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(jpy_ret)
                                             / jpy_ret.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)
if "EURUSD" in panels:
    eur_ret = panels["EURUSD"]["close"].astype(float).pct_change()
    C["eurusd_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(eur_ret)
                                             / eur_ret.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)

log("building library signals...")
library = library_signals(panels, closes, rets)
library["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20
library["dn_mkt_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn)
                                               / dn.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)
cn_ret = closes["CN10Y"].pct_change()
library["rate_beta_cn10y_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret)
                                                   / cn_ret.rolling(60, min_periods=40).var()) for a in rets.columns}, index=rets.index)

fwd10 = forward_returns(closes, 10)
ADM_IC, ADM_ICIR = 0.0070, 0.0840

print("=" * 130)
print("CANDIDATE SCREEN (h=10, full history 2020..2034-03-17)")
print("=" * 130)
hdr = f"{'candidate':28s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covA':>5s} {'covD':>5s} {'turn':>5s} {'libCorr':>7s} {'decay5':>7s} {'decay10':>7s} {'decay20':>7s} {'pass':>4s}"
print(hdr)
print("-" * 130)

results = {}
for name, fp in C.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    if len(ics) == 0:
        print(f"{name:28s} no IC dates")
        continue
    m = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    dec = decay_profile(fp, closes)
    corr, key = max_library_corr(fp, library)
    p = abs(m["ic"]) >= ADM_IC and abs(m["icir"]) >= ADM_ICIR
    results[name] = {"metrics": m, "coverage": cov, "turnover": turn,
                     "decay": dec, "corr": corr, "corr_key": key, "pass": p}
    print(f"{name:28s} {m['ic']:8.4f} {m['icir']:7.3f} {m['ic_hit_ratio']:5.2f} "
          f"{m['n_ic_dates']:5d} {cov['coverage_asset_days']:5.2f} {cov['coverage_dates_ge8']:5.2f} "
          f"{turn:5.2f} {corr:7.3f} {dec.get('5', float('nan')):7.4f} {dec.get('10', float('nan')):7.4f} "
          f"{dec.get('20', float('nan')):7.4f} {'YES' if p else ''}")

print("-" * 130)
print("PASSING CANDIDATES (full history):")
for name, r in results.items():
    if r["pass"]:
        print(f"  {name}: IC={r['metrics']['ic']:.4f} ICIR={r['metrics']['icir']:.3f} "
              f"hit={r['metrics']['ic_hit_ratio']:.2f} n={r['metrics']['n_ic_dates']} "
              f"libCorr={r['corr']:.3f}({r['corr_key']})")

print("=" * 130)
print("RECENT WINDOW CHECK (2031-01-01..2034-03-17) for candidates |IC|>=0.005 full")
print("=" * 130)
for name, r in results.items():
    if abs(r["metrics"]["ic"]) < 0.005:
        continue
    fp = C[name].replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    sub = ics[ics.index >= pd.Timestamp("2031-01-01")]
    sub3 = ics[ics.index >= pd.Timestamp("2033-01-01")]
    if len(sub) == 0:
        print(f"{name:28s} recent3y: no IC dates")
        continue
    rm = summarize_ic(sub, expected_sign=1)
    rp = abs(rm["ic"]) >= ADM_IC and abs(rm["icir"]) >= ADM_ICIR
    extra = ""
    if len(sub3) > 20:
        rm3 = summarize_ic(sub3, expected_sign=1)
        extra = f" | 1y3 IC={rm3['ic']:+.4f} ICIR={rm3['icir']:+.3f} n={rm3['n_ic_dates']}"
    print(f"{name:28s} recent3y IC={rm['ic']:+.4f} ICIR={rm['icir']:+.3f} hit={rm['ic_hit_ratio']:.2f} n={rm['n_ic_dates']:4d} pass={'YES' if rp else ''}{extra}")

log("done")
