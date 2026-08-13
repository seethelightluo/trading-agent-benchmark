"""miner_3 batch AG (2033-07-25) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2033-07-22). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift and max-abs library
correlation for passers. No live-account interaction.

Candidates (batch AG - families NOT covered by batches Z/AA/AB/AD/AE or the
evicted list):
  A. usdjpy_beta_60d   - carry/beta to USDJPY returns (60d)
  B. dxy_beta_60d      - dollar-sensitivity beta to DXY returns (60d)
  C. skew_60d          - rolling 60d return skewness (crash-risk)
  D. rev5_voladj       - 5d short-term reversal scaled by 20d vol
  E. mom30_voladj      - 30d momentum scaled by 30d vol
  F. vol_z_120         - vol timing: 20d vol vs its 120d rolling mean
  G. park_vol_ratio_10_60 - range-based (Parkinson) vol ratio 10/60
  H. gold_beta_60d     - safe-haven spillover beta to XAU returns
  I. btc_beta_60d      - crypto-spillover beta to BTC returns
  J. dn_up_beta_diff_60d - downside-beta minus upside-beta (asymmetry)
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE,
                                 max_library_corr)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

H = 10
fwd = forward_returns(closes, H)


def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)


# ---------------- 1) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = ((closes / closes.shift(20) - 1) - (closes / closes.shift(60) - 1)) / rets.rolling(20).std()
dn_mask = (mkt_ret < 0).astype(float)
mr_dn = mkt_ret.where(dn_mask > 0)
existing["dn_mkt_beta_60d"] = (rets.where(dn_mask > 0)).rolling(60, min_periods=40).cov(mr_dn) / mr_dn.rolling(60, min_periods=40).var()
existing["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)


def report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm / ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) "
          f"r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics


for name, sig in existing.items():
    report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)

# ---------------- 2) CANDIDATE SCREEN (batch AG) ----------------
print("\n=== CANDIDATE SCREEN (batch AG, full history) ===", flush=True)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
usdjpy_ret = panels["USDJPY"]["close"].pct_change()
dxy_ret = panels["DXY"]["close"].pct_change()
xau_ret = rets["XAU"]
btc_ret = rets["BTC"]

C = {}
# A. USDJPY carry beta
C["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy_ret, 60)
# B. DXY beta
C["dxy_beta_60d"] = rolling_beta(rets, dxy_ret, 60)
# C. 60d skewness
C["skew_60d"] = rets.rolling(60).skew()
# D. 5d vol-adj reversal (skip 1)
C["rev5_voladj"] = -(closes.shift(1) / closes.shift(6) - 1.0) / vol20
# E. 30d vol-adj momentum
C["mom30_voladj"] = (closes / closes.shift(30) - 1.0) / rets.rolling(30).std()
# F. vol timing z: 20d vol vs 120d rolling mean
C["vol_z_120"] = vol20 / vol20.rolling(120).mean() - 1.0
# G. Parkinson range-vol ratio 10/60
pk10 = np.sqrt(np.log(2.0) * ((np.log(hi / lo)) ** 2).rolling(10).mean())
pk60 = np.sqrt(np.log(2.0) * ((np.log(hi / lo)) ** 2).rolling(60).mean())
C["park_vol_ratio_10_60"] = pk10 / pk60 - 1.0
# H. gold beta
C["gold_beta_60d"] = rolling_beta(rets, xau_ret, 60)
# I. BTC beta
C["btc_beta_60d"] = rolling_beta(rets, btc_ret, 60)
# J. downside minus upside beta (60d, split on asset own sign)
up_mask = (rets > 0).astype(float)
up_beta = (rets.where(up_mask > 0)).rolling(60, min_periods=40).cov(mkt_ret.where(up_mask > 0)) / mkt_ret.where(up_mask > 0).rolling(60, min_periods=40).var()
dn_beta = (rets.where(dn_mask > 0)).rolling(60, min_periods=40).cov(mr_dn) / mr_dn.rolling(60, min_periods=40).var()
C["dn_up_beta_diff_60d"] = dn_beta - up_beta

lib = {k: v for k, v in existing.items()}

print(f"{len(C)} candidates; time {time.time()-t0:.1f}s", flush=True)
results = {}
for i, (name, sig) in enumerate(C.items()):
    s, ics = report(name, sig, expected_sign=1)
    m = dict(s)
    m["turnover_10d_rank"] = turnover_rank(sig, 10)
    m["coverage_asset_days"] = coverage_metrics(sig)["coverage_asset_days"]
    m["coverage_dates_ge8"] = coverage_metrics(sig)["coverage_dates_ge8"]
    m["decay_ic_by_horizon"] = decay_profile(sig, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(sig, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    for w in (63, 252):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            m[f"ic_last{w}d"] = round(float(mm), 4)
            m[f"icir_last{w}d"] = round(float(mm / ss), 3) if ss and ss > 0 else None
        else:
            m[f"ic_last{w}d"] = None
            m[f"icir_last{w}d"] = None
    results[name] = m
    passed = abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840
    print(f"   decay: { {k: v for k, v in m['decay_ic_by_horizon'].items()} }"
          f" | libcorr={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']})"
          f" | cov_asset={m['coverage_asset_days']:.3f} cov_d8={m['coverage_dates_ge8']:.3f}"
          f" | 1y_IC={m['ic_last252d']} 1y_ICIR={m['icir_last252d']} 63d_IC={m['ic_last63d']}", flush=True)

print("\n--- summary table ---")
for name, m in results.items():
    passed = abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} {'PASS' if passed else '---'}")
print(f"total time {time.time()-t0:.1f}s", flush=True)
