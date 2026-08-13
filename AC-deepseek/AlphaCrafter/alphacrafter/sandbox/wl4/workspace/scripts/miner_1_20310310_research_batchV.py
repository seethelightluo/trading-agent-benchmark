"""MINER_1 2031-03-10: cross-asset factor exploration + re-validation (batch V).

Visible data through 2031-03-07 (previous completed trading day before current
date 2031-03-10). Data loaded through the simulator API (no lookahead beyond the
simulator's visible window). No live-account interaction.

Screens novel, interpretable cross-asset factors on the 15-instrument universe
(min_valid=8). Admission gates: |IC|>=0.0070 and |ICIR|>=0.0840 at h=10 (daily
rank IC). Reports decay profile and max-abs library correlation for passers.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = pd.Timestamp.now()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {pd.Timestamp.now()-t0}", flush=True)

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)

H = 10
fwd = forward_returns(closes, H)

# ---------- NaN-aware rolling beta (vectorized cov/var per asset) ----------
def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

def rolling_corr(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        c = z["y"].rolling(win).corr(z["x"]).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = c
    return pd.DataFrame(out, index=y.index)

# ---------- 1) RE-VALIDATE current effective factors ----------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
existing["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
existing["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

def report(name, sig):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm/ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to:.2f}", flush=True)
    return ics, s

for name, sig in existing.items():
    report(name, sig)

# ---------- 2) CANDIDATE SCREEN (batch V: novel cross-asset ideas) ----------
print("\n=== CANDIDATE SCREEN (full history + recent) ===", flush=True)
C = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

# V1: cross-sectional dispersion of 20d returns (breadth)
C["dispersion_20d"] = (closes/closes.shift(20)-1.0).std(axis=1)
# V2: avg pairwise return correlation 60d (regime cohesion)
def pairwise_corr(win=60):
    out = {}
    cols = list(closes.columns)
    for a in cols:
        corrs = []
        for b in cols:
            if b <= a:
                continue
            z = pd.concat([rets[a].rename("a"), rets[b].rename("b")], axis=1).dropna()
            corrs.append(z["a"].rolling(win).corr(z["b"]))
        out[a] = pd.concat(corrs, axis=1).mean(axis=1)
    return pd.DataFrame(out, index=rets.index)
C["avg_pair_corr_60d"] = pairwise_corr(60)
# V3: DXY beta 60d (USD strength linkage)
C["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60)
# V4: USDJPY beta 60d (carry/risk linkage)
C["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy.pct_change(), 60)
# V5: gold-copper ratio momentum 20d (inflation vs growth)
gold_cop = closes["XAU"] / closes["COPPER"]
C["gold_cop_ratio_mom_20d"] = gold_cop / gold_cop.shift(20) - 1.0
# V6: crypto minus equity spread momentum 20d (risk appetite)
crypto = closes[["BTC", "ETH"]].mean(axis=1)
equity = closes[["SPX", "NDX"]].mean(axis=1)
C["crypto_eq_spread_20d"] = (crypto / crypto.shift(20) - 1.0) - (equity / equity.shift(20) - 1.0)
# V7: VIX z-score vs 252d (fear gauge)
vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
C["vix_zscore_252d"] = -vix_z
# V8: AR(1) autocorrelation of daily returns 20d (reversal vs continuation)
def ar1_20d():
    out = {}
    for a in rets.columns:
        r = rets[a]
        r1 = r.shift(1)
        z = pd.concat([r.rename("r"), r1.rename("r1")], axis=1).dropna()
        out[a] = z["r"].rolling(20).corr(z["r1"])
    return pd.DataFrame(out, index=rets.index)
C["ar1_20d"] = ar1_20d()
# V9: range compression 20/60 (volatility squeeze)
hl = pd.DataFrame({a: (panels[a]["high"].astype(float) - panels[a]["low"].astype(float)) / panels[a]["close"].astype(float)
                   for a in TRADABLE}, index=closes.index)
C["range_compress_20_60"] = -hl.rolling(20).mean() / hl.rolling(60).mean()
# V10: Kaufman trend consistency 20d (directional efficiency, daily)
C["trend_consistency_20d"] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
# V11: lottery - max daily gain 20d
C["max_gain_20d"] = rets.rolling(20).max()
# V12: lottery - max daily loss 20d
C["max_loss_20d"] = rets.rolling(20).min()
# V13: downside momentum 60d (momentum on down days only)
down_ret = rets.where(rets < 0, 0.0)
C["downside_mom_60d"] = (closes / closes.shift(60) - 1.0) * (down_ret.rolling(60).sum() / (rets.abs().rolling(60).sum() + 1e-9))
# V14: volatility change ratio 5/60 (volatility trend)
C["vol_chg_ratio_5_60"] = -rets.rolling(5).std() / rets.rolling(60).std()
# V15: position within 20d range (close-low)/(high-low)
C["hl_pos_20d"] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min() + 1e-9)
# V16: US10Y-CN10Y spread change 20d (yield slope move)
spread = closes["US10Y"] - closes["CN10Y"]
C["yield_spread_chg_20d"] = spread / spread.shift(20) - 1.0
# V17: drawup recovery 60d (px / 60d min - 1)
C["drawup_recovery_60d"] = closes / closes.rolling(60).min() - 1.0
# V18: XAU beta 60d (safe-haven linkage)
C["xau_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60)
# V19: NDX/SPX ratio momentum 20d (tech risk appetite)
ndx_spx = closes["NDX"] / closes["SPX"]
C["ndx_spx_ratio_mom_20d"] = ndx_spx / ndx_spx.shift(20) - 1.0
# V20: 000688/000300 relative momentum 20d (China tech vs broad)
ch = closes["000688.SH"] / closes["000300.SH"]
C["cn_tech_rel_mom_20d"] = ch / ch.shift(20) - 1.0

results = {}
for name, sig in C.items():
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm/ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to:.2f}{flag}", flush=True)
    results[name] = (s, ics, sig)

# ---------- 3) decay + library correlation for full-pass candidates ----------
print("\n=== DECAY (h=1,3,5,10,20) + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
lib = dict(existing)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["rsi_14"] = closes.rolling(14).apply(lambda w: 100 - 100/(1 + (w.diff().clip(lower=0).mean() / (-w.diff().clip(upper=0).mean() + 1e-9))), raw=False)
vix_ret = vix.pct_change()
vix_beta = rolling_beta(rets, vix_ret, 60)
lib["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
lib["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
lib["vol_ratio_20_60"] = vol20 / vol60
lib["volume_z_20"] = None  # volume-based; skip in corr (not used for these candidates)

def max_lib_corr(sig):
    best, key = 0.0, None
    for lname, lsig in lib.items():
        if lsig is None:
            continue
        both = pd.concat([sig.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rr = float(both["c"].corr(both["l"]))
        if abs(rr) > best:
            best, key = abs(rr), lname
    return round(best, 4), key

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    corr, key = max_lib_corr(sig)
    print(f"{name:26s} decay={dec} max_abs_lib_corr={corr:.4f} (vs {key})", flush=True)

print("\ndone", flush=True)
