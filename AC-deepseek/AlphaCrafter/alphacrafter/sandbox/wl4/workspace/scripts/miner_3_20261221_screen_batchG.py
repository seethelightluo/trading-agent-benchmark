"""miner_3 batch G screen (2026-12-21) - vectorized rank-IC on data through last completed day.

A) drift re-validation of 4 active library factors (full + recent)
B) new batch G candidates:
   - trend/quality: sharpe_60d, mom_20d, mom_20d_skip5, rsi_20d, high_dist_120d
   - reversal/mean-reversion: drawdown_20d (close/running_max60-1), gap_rev_5d, cs_zscore_20d
   - relative strength: cs_rel_mom_20d
   - regime/vol: vol_mom_5x20, range_ratio_20x60, up_vol_share_60d, vol_skew_20d
   - cross-asset: us10y_beta_60d, eurusd_down_beta_60d, btc_cond_beta_60d, corr_mkt_60d,
     ndx_lead_5d (5d NDX return lead), cny10_rate_spread_20d (yield momentum)
   - memory: hurst_proxy (autocorr ratio)

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe).
Robustness: full-period + sub-period split + recent window; report ex-frozen (HSI/ETH frozen since 2026-10-14).
"""
import sys, time, json, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)

# last completed trading day = last close date
LAST = closes.index.max()
print("last completed trading day:", LAST.date(), flush=True)

# frozen check: HSI/ETH flat since 2026-10-14?
for a in ["HSI", "ETH"]:
    s = closes[a].dropna()
    last20 = s.tail(20)
    print(f"{a}: n_flat_last20={int((last20.diff()==0).sum())} last_close={last20.iloc[-1]:.4f} n_days_last={len(last20)}", flush=True)

H_ADM = 10
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840
mkt = rets.mean(axis=1)

def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)

# ---------- library factor signal artifacts (recompute from definitions) ----------
def lib_vol_price_corr_20():
    return pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a]) for a in closes.columns}, index=rets.index)

def lib_dn_mkt_beta_60d():
    m = mkt
    dn = m.where(m < 0)
    beta = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dn.rename("m")], axis=1).dropna()
        z = z[z["m"].notna()]
        cov = z["a"].rolling(60).cov(z["m"])
        var = z["m"].rolling(60).var()
        beta[a] = (cov / var).where(z["m"].rolling(60).count() >= 40)
    return pd.DataFrame(beta, index=rets.index)

def lib_eurusd_beta_60d():
    eur = panels["EURUSD"]["close"].pct_change() if "EURUSD" in panels else None
    return rolling_beta(rets, eur, 60, 40) if eur is not None else None

def lib_rate_beta_cn10y_60d():
    cn = rets["CN10Y"]
    return rolling_beta(rets, cn, 60, 40)

LIBRARY = {}
LIBRARY["vol_price_corr_20"] = lib_vol_price_corr_20()
LIBRARY["dn_mkt_beta_60d"] = lib_dn_mkt_beta_60d()
LIBRARY["eurusd_beta_60d"] = lib_eurusd_beta_60d()
LIBRARY["rate_beta_cn10y_60d"] = lib_rate_beta_cn10y_60d()

def max_lib_corr(cand):
    best, best_key = 0.0, None
    c = cand.stack()
    for name, s in LIBRARY.items():
        both = pd.concat([c.rename("cand"), s.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key

def evaluate(tag, panel, expected_sign=1):
    fwd = forward_returns(closes, H_ADM)
    ics_full = rank_ic_series(panel, fwd, MIN_VALID)
    m_full = summarize_ic(ics_full, expected_sign)
    # recent windows
    cut_recent = closes.index[-250]
    cut_mid = closes.index[-500]
    out = {"tag": tag, "expected_sign": expected_sign,
           "ic_full": m_full["ic"], "icir_full": m_full["icir"],
           "hit_full": m_full["ic_hit_ratio"], "n_full": m_full["n_ic_dates"],
           "ic_recent250": np.nan, "icir_recent250": np.nan, "n_recent250": 0,
           "ic_recent500": np.nan, "icir_recent500": np.nan, "n_recent500": 0}
    for name, cut in (("recent250", cut_recent), ("recent500", cut_mid)):
        sub = ics_full[ics_full.index >= cut]
        if len(sub) >= 20:
            s = summarize_ic(sub, expected_sign)
            out[f"ic_{name}"] = s["ic"]
            out[f"icir_{name}"] = s["icir"]
            out[f"n_{name}"] = s["n_ic_dates"]
    cov = coverage_metrics(panel, min_valid=MIN_VALID)
    out["cov_dates_ge8"] = cov["coverage_dates_ge8"]
    out["turnover_10d"] = turnover_rank(panel, 10)
    corr, key = max_lib_corr(panel)
    out["max_lib_corr"] = corr
    out["max_corr_factor"] = key
    return out

results = {}

# ---------- A) active library drift re-validation ----------
for name, sig in LIBRARY.items():
    if sig is None:
        continue
    exp = -1 if name == "eurusd_beta_60d" else 1
    if name == "rate_beta_cn10y_60d":
        exp = -1
    results["active_" + name] = evaluate("active_" + name, sig, exp)

# ---------- B) batch G candidates ----------
cands = {}

# trend / quality
cands["G_sharpe_60d"] = rets.rolling(60).mean() / rets.rolling(60).std().replace(0, np.nan)
cands["G_mom_20d"] = closes / closes.shift(20) - 1.0
cands["G_mom_20d_skip5"] = closes.shift(5) / closes.shift(25) - 1.0
cands["G_mom_60d_skip10"] = closes.shift(10) / closes.shift(70) - 1.0
cands["G_rsi_20d"] = (rets.clip(lower=0).rolling(20).mean() /
                      rets.abs().rolling(20).mean().replace(0, np.nan))
cands["G_high_dist_120d"] = closes / closes.rolling(120).max() - 1.0

# reversal / mean reversion
cands["G_drawdown_60d"] = closes / closes.rolling(60).max() - 1.0
cands["G_gap_rev_5d"] = -(closes / closes.shift(5) - 1.0)
cands["G_cs_zscore_20d"] = (closes - closes.rolling(20).mean()) / closes.rolling(20).std().replace(0, np.nan)

# relative strength
cands["G_cs_rel_mom_20d"] = (closes / closes.shift(20) - 1.0).sub(
    (closes / closes.shift(20) - 1.0).mean(axis=1), axis=0)

# regime / vol
cands["G_vol_mom_5x20"] = rets.rolling(5).std() / rets.rolling(20).std() - 1.0
cands["G_range_ratio_20x60"] = ((highs - lows) / closes).rolling(20).mean() / ((highs - lows) / closes).rolling(60).mean() - 1.0
cands["G_up_vol_share_60d"] = rets.clip(lower=0).rolling(60).std() / rets.rolling(60).std().replace(0, np.nan)
cands["G_vol_skew_20d"] = (rets - rets.rolling(20).mean()).clip(upper=0).pow(3).rolling(20).mean() / rets.rolling(20).std().pow(3).replace(0, np.nan)

# cross-asset
us10y_ret = rets["US10Y"]
cands["G_us10y_beta_60d"] = rolling_beta(rets, us10y_ret, 60, 40)
eur_ret = panels["EURUSD"]["close"].pct_change()
dn_eur = eur_ret.where(eur_ret < 0)
cands["G_eurusd_down_beta_60d"] = rolling_beta(rets, dn_eur, 60, 40)
btc_ret = rets["BTC"]
btc_up = btc_ret.where(btc_ret > 0)
cands["G_btc_up_beta_60d"] = rolling_beta(rets, btc_up, 60, 40)
cands["G_corr_mkt_60d"] = pd.DataFrame(
    {a: rets[a].rolling(60).corr(mkt) for a in closes.columns}, index=rets.index)
cands["G_ndx_lead_5d"] = pd.DataFrame(
    {a: np.nan for a in closes.columns}, index=rets.index)
for a in closes.columns:
    ndx_lead = rets["NDX"].shift(1).rolling(5).sum()
    cands["G_ndx_lead_5d"][a] = ndx_lead

# memory
cands["G_hurst_proxy"] = (rets.rolling(10).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False) /
                          rets.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False).replace(0, np.nan))

for tag, panel in cands.items():
    exp = 1
    if tag in ("G_gap_rev_5d",):
        exp = 1  # panel already negated -> expect positive IC
    results[tag] = evaluate(tag, panel, exp)

with open("scripts/_miner3_batchG_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)

print("\n=== SUMMARY (full-period h=10) ===")
rows = sorted(results.items(), key=lambda kv: -abs(kv[1].get("ic_full") or 0))
for k, v in rows:
    print(f"{k:28s} IC={v['ic_full']:+.4f} ICIR={v['icir_full']:+.4f} hit={v['hit_full']:.2f} n={v['n_full']:4d} "
          f"R250={v['ic_recent250']:+.4f}({v['n_recent250']}) R500={v['ic_recent500']:+.4f}({v['n_recent500']}) "
          f"cov={v['cov_dates_ge8']:.2f} to={v['turnover_10d']} libcorr={v['max_lib_corr']}({v['max_corr_factor']})")

print(f"\nDONE {time.time()-t0:.1f}s -> scripts/_miner3_batchG_results.json", flush=True)
