"""miner_3 batch AE (2033-01-24) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2033-01-21). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift and max-abs library
correlation for passers. No live-account interaction.

Candidates (batch AE - families not covered by batches Z/AA/AB/AD):
  A. resid_mom_60d       - idiosyncratic momentum: t-stat of 60d regression alpha on equal-w mkt
  B. upday_ratio_60d     - trend consistency: fraction of up days over 60d
  C. hi_250_prox         - 52-week high proximity: close/rollmax(close,250)-1
  D. sma200_dist         - long-term trend: close/SMA200 - 1
  E. idio_vol_20d        - idiosyncratic vol: resid std from 20d mkt regression (low = stable)
  F. corr_delta_60       - correlation momentum: 60d mkt-corr minus its 20d-ago value
  G. vol_conf_mom_20x60  - volume-confirmed momentum: 20d return * volume-trend z
  H. max_dd_250d         - long-horizon drawdown depth over 250d
  I. ndx_beta_60d        - tech-spillover beta: beta of asset ret on NDX ret, 60d
  J. vol_adj_reversal_20d- 20d short reversal scaled by 20d vol
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

# high/low/volume panels
hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

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


def rolling_corr(y, x, win=20, min_obs=15):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        c = z["y"].rolling(win).corr(z["x"])
        out[a] = c.where(z["x"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=y.index)


# ---------------- 1) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = ((closes / closes.shift(20) - 1) - (closes / closes.shift(60) - 1)) / rets.rolling(20).std()
dn_mask = (mkt_ret < 0).astype(float)
existing["dn_mkt_beta_60d"] = (rets.where(dn_mask > 0)).rolling(60, min_periods=40).cov(mkt_ret.where(dn_mask > 0)) / mkt_ret.where(dn_mask > 0).rolling(60, min_periods=40).var()
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
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics


for name, sig in existing.items():
    report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)

# ---------------- 2) CANDIDATE SCREEN (batch AE) ----------------
print("\n=== CANDIDATE SCREEN (batch AE, full history) ===", flush=True)
C = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

# A. idiosyncratic momentum: 60d regression alpha t-stat vs equal-w market
beta60 = rolling_beta(rets, mkt_ret, 60)
alpha_mean = rets.rolling(60).mean() - beta60 * mkt_ret.rolling(60).mean()
resid_var = rets.rolling(60).var() - beta60 ** 2 * mkt_ret.rolling(60).var()
resid_var = resid_var.clip(lower=1e-14)
C["resid_mom_60d"] = alpha_mean * np.sqrt(60) / np.sqrt(resid_var)

# B. trend consistency: fraction of up days over 60d
C["upday_ratio_60d"] = (rets > 0).rolling(60).mean()

# C. 52-week high proximity
C["hi_250_prox"] = closes / closes.rolling(250).max() - 1.0

# D. long-term trend vs 200d SMA
C["sma200_dist"] = closes / closes.rolling(200).mean() - 1.0

# E. idiosyncratic vol (20d residual std from mkt regression)
beta20 = rolling_beta(rets, mkt_ret, 20, min_obs=15)
resid_var20 = rets.rolling(20).var() - beta20 ** 2 * mkt_ret.rolling(20).var()
C["idio_vol_20d"] = np.sqrt(resid_var20.clip(lower=1e-14))

# F. correlation momentum: 60d mkt-corr minus its value 20d ago
corr60 = rolling_corr(rets, mkt_ret, 60, min_obs=40)
C["corr_delta_60"] = corr60 - corr60.shift(20)

# G. volume-confirmed momentum: 20d return * volume trend z (20/60 means)
vol_tr = vol_panel.rolling(20).mean() / vol_panel.rolling(60).mean() - 1.0
mom20 = closes / closes.shift(20) - 1.0
C["vol_conf_mom_20x60"] = mom20 * vol_tr

# H. long-horizon drawdown depth (250d): min(close/rollmax-1) over 250d
roll_max250 = closes.rolling(250).max()
dd = closes / roll_max250 - 1.0
C["max_dd_250d"] = dd.rolling(250).min()

# I. tech-spillover beta (60d)
C["ndx_beta_60d"] = rolling_beta(rets, rets["NDX"], 60)

# J. 20d vol-adj reversal
C["vol_adj_reversal_20d"] = -(closes / closes.shift(20) - 1.0) / vol20

# library signals for correlation
lib = {}
lib["vol_adj_mom_accel_20x60"] = existing["vol_adj_mom_accel_20x60"]
lib["dn_mkt_beta_60d"] = existing["dn_mkt_beta_60d"]
lib["rate_beta_cn10y_60d"] = existing["rate_beta_cn10y_60d"]

from factor_research_lib import max_library_corr

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
