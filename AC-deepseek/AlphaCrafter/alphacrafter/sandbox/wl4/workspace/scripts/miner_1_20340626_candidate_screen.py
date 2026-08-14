"""miner_1 (2034-06-26): revalidate current library + screen novel candidate factors
through the latest visible trading day (2034-06-23).

Admission gate (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840 (15-asset cross-section).
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

# high/low/volume panels
highs = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).sort_index()
lows = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).sort_index()
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()

mom5 = closes / closes.shift(5) - 1.0
mom10 = closes / closes.shift(10) - 1.0
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol10 = rets.rolling(10).std(ddof=0)
vol20 = rets.rolling(20).std(ddof=0)
vol60 = rets.rolling(60).std(ddof=0)
sma20 = closes.rolling(20).mean()
sma60 = closes.rolling(60).mean()
sma200 = closes.rolling(200).mean()

C = {}

# ---- oscillator / mean-reversion family (novel) ----
# 1. MACD histogram (12,26) normalized by close
ema12 = closes.ewm(span=12, adjust=False).mean()
ema26 = closes.ewm(span=26, adjust=False).mean()
C["macd_hist_12_26"] = (ema12 - ema26) / closes

# 2. RSI 14 (simple mean-based)
gain = rets.clip(lower=0.0)
loss = (-rets).clip(lower=0.0)
ag = gain.rolling(14).mean()
al = loss.rolling(14).mean()
C["rsi_14"] = 100.0 - 100.0 / (1.0 + ag / (al + 1e-12))

# 3. Bollinger %B (20,2)
C["bollinger_pctb_20_2"] = (closes - sma20) / (2.0 * vol20 * np.sqrt(20) + 1e-12)

# 4. Kaufman efficiency ratio 20d
C["efficiency_ratio_20"] = (closes - closes.shift(20)).abs() / (rets.abs().rolling(20).sum() + 1e-12)

# 5. Stochastic %K 14d
ll14 = lows.rolling(14).min()
hh14 = highs.rolling(14).max()
C["stoch_k_14"] = (closes - ll14) / (hh14 - ll14 + 1e-12)

# 6. Vol-scaled short reversal 5d
C["voladj_short_rev_5d"] = -mom5 / (vol10 + 1e-12)

# ---- trend/price-level family (novel) ----
# 7. 60d high proximity (breakout distance)
C["pct_off_60d_high"] = closes / closes.rolling(60).max() - 1.0

# 8. 250d low proximity (deep-value / beaten-down)
C["pct_off_250d_low"] = closes / closes.rolling(250).min() - 1.0

# 9. 200d MA distance
C["price_vs_sma200"] = closes / sma200 - 1.0

# 10. ATR-relative trend 20d
atr20 = (highs - lows).rolling(20).mean()
C["trend_vs_atr_20"] = (closes - sma20) / (atr20 + 1e-12)

# 11. Sharpe-style momentum 20d (mom/vol)
C["sharpe_mom_20d"] = mom20 / (vol20 + 1e-12)

# 12. Trend strength 120d
C["trend_strength_120d"] = closes / closes.rolling(120).mean() - 1.0

# ---- risk/asymmetry family (novel) ----
# 13. Upside beta 60d (high upside beta = risk-seeking / lottery)
up = np.maximum(mkt, 0.0)
C["up_mkt_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(up)
                                         / up.rolling(60, min_periods=40).var()) for a in rets.columns},
                                    index=rets.index)

# 14. Max daily return 60d (lottery preference)
C["max_daily_ret_60d"] = rets.rolling(60).max()

# 15. Idiosyncratic vol 60d (residual std vs market)
def idio_vol(r, m, win=60):
    out = {}
    for a in r.columns:
        z = pd.concat([r[a].rename("a"), m.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win, min_periods=40).cov(z["m"])
        var = z["m"].rolling(win, min_periods=40).var()
        beta = cov / var
        resid_var = z["a"].rolling(win, min_periods=40).var() - beta**2 * var
        out[a] = np.sqrt(resid_var.clip(lower=0.0))
    return pd.DataFrame(out, index=r.index)
C["idio_vol_60d"] = idio_vol(rets, mkt)

# 16. Residual momentum 60d (beta-adjusted)
beta60 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(mkt)
                           / mkt.rolling(60, min_periods=40).var()) for a in rets.columns},
                      index=rets.index)
mkt_mom60 = (mkt + 1.0).rolling(60).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1.0, raw=False)
C["resid_mom_60d"] = mom60 - beta60 * mkt_mom60

# 17. Downside-risk-weighted momentum (momentum scaled by downside share)
dn20 = rets.where(rets < 0, 0.0)
downside_share = dn20.rolling(20).std(ddof=0) / (vol20 + 1e-12)
C["dn_risk_weighted_mom_20d"] = mom20 * (1.0 - downside_share)

# ---- macro-conditional family (novel) ----
vix = panels["VIX"]["close"].astype(float)
vix_ret = vix.pct_change()
vix_pctile = vix.rolling(250).rank(pct=True)
# 18. Defensive tilt when VIX elevated: -beta(asset,VIX,60) * VIX percentile
beta_vix = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(vix_ret)
                             / vix_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                        index=rets.index)
C["vix_elev_beta_tilt"] = -beta_vix * vix_pctile

# 19. VIX momentum regime interaction: -beta(asset,VIX,60)*sign(VIX 20d trend)*|VIX trend|
vix_trend = vix / vix.shift(20) - 1.0
C["vix_trend_beta_tilt"] = -beta_vix * vix_trend

# 20. Rate-regime bond momentum tilt: beta(asset,US10Y,60) * US10Y 20d momentum
us10y_ret = rets["US10Y"]
beta_us10y = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(us10y_ret)
                               / us10y_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                          index=rets.index)
us10y_trend = closes["US10Y"] / closes["US10Y"].shift(20) - 1.0
C["rate_mom_beta_tilt"] = beta_us10y * us10y_trend

# 21. Equity-regime factor: beta(asset,SPX,60) * SPX 20d momentum (risk-on tilt)
spx_ret = rets["SPX"]
beta_spx = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(spx_ret)
                             / spx_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                        index=rets.index)
spx_trend = closes["SPX"] / closes["SPX"].shift(20) - 1.0
C["equity_mom_beta_tilt"] = beta_spx * spx_trend

# 22. Cross-asset carry proxy: vol-ratio trend (vol term structure momentum)
C["vol_ratio_trend_10_60"] = (vol10 / vol60) / (vol10 / vol60).shift(20) - 1.0

# 23. Price acceleration (3rd derivative proxy): (mom5 - mom20) scaled
C["mom_accel_5_20"] = (mom5 - mom20) / (vol10 + 1e-12)

# 24. Drawdown depth 120d
C["drawdown_depth_120d"] = closes / closes.rolling(120).max() - 1.0

log("building library signals...")
library = library_signals(panels, closes, rets, vix)
library["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20
dn = np.minimum(mkt, 0.0)
library["dn_mkt_beta_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn)
                                               / dn.rolling(60, min_periods=40).var()) for a in rets.columns},
                                          index=rets.index)
cn_ret = closes["CN10Y"].pct_change()
library["rate_beta_cn10y_60d"] = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret)
                                                   / cn_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                                              index=rets.index)

fwd10 = forward_returns(closes, 10)
ADM_IC, ADM_ICIR = 0.0070, 0.0840

print("=" * 140)
print("CANDIDATE SCREEN (h=10, full history 2020..2034-06-23)")
print("=" * 140)
hdr = (f"{'candidate':30s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covA':>5s} "
       f"{'covD':>5s} {'turn':>5s} {'libCorr':>7s} {'d5':>7s} {'d10':>7s} {'d20':>7s} {'pass':>4s}")
print(hdr)
print("-" * 140)

results = {}
for name, fp in C.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    if len(ics) == 0:
        print(f"{name:30s} no IC dates")
        continue
    m = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    dec = decay_profile(fp, closes)
    corr, key = max_library_corr(fp, library)
    p = abs(m["ic"]) >= ADM_IC and abs(m["icir"]) >= ADM_ICIR
    results[name] = {"metrics": m, "coverage": cov, "turnover": turn,
                     "decay": dec, "corr": corr, "corr_key": key, "pass": p}
    print(f"{name:30s} {m['ic']:8.4f} {m['icir']:7.3f} {m['ic_hit_ratio']:5.2f} "
          f"{m['n_ic_dates']:5d} {cov['coverage_asset_days']:5.2f} {cov['coverage_dates_ge8']:5.2f} "
          f"{turn:5.2f} {corr:7.3f} {dec.get('5', float('nan')):7.4f} {dec.get('10', float('nan')):7.4f} "
          f"{dec.get('20', float('nan')):7.4f} {'YES' if p else ''}")

print("-" * 140)
print("PASSING CANDIDATES (full history):")
for name, r in results.items():
    if r["pass"]:
        print(f"  {name}: IC={r['metrics']['ic']:.4f} ICIR={r['metrics']['icir']:.3f} "
              f"hit={r['metrics']['ic_hit_ratio']:.2f} n={r['metrics']['n_ic_dates']} "
              f"libCorr={r['corr']:.3f}({r['corr_key']})")

print("=" * 140)
print("RECENT WINDOW CHECK (2031-01-01.. and 2033-01-01..) for candidates |IC|>=0.005 full")
print("=" * 140)
for name, r in results.items():
    if abs(r["metrics"]["ic"]) < 0.005:
        continue
    fp = C[name].replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    sub = ics[ics.index >= pd.Timestamp("2031-01-01")]
    sub3 = ics[ics.index >= pd.Timestamp("2033-01-01")]
    if len(sub) == 0:
        print(f"{name:30s} recent3y: no IC dates")
        continue
    rm = summarize_ic(sub, expected_sign=1)
    rp = abs(rm["ic"]) >= ADM_IC and abs(rm["icir"]) >= ADM_ICIR
    extra = ""
    if len(sub3) > 20:
        rm3 = summarize_ic(sub3, expected_sign=1)
        extra = f" | 2033+ IC={rm3['ic']:+.4f} ICIR={rm3['icir']:+.3f} n={rm3['n_ic_dates']}"
    print(f"{name:30s} recent3y IC={rm['ic']:+.4f} ICIR={rm['icir']:+.3f} hit={rm['ic_hit_ratio']:.2f} "
          f"n={rm['n_ic_dates']:4d} pass={'YES' if rp else ''}{extra}")

log("done")
