"""miner_2 batch1 screen (2032-03-08) - novel cross-asset factor candidates.

Visible data through the previous completed trading day (API-driven). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates at h=10: |IC| >= 0.0070 and
|ICIR| >= 0.0840. Reports recent-window drift, coverage, turnover, decay and
library correlation for full-passers. No live-account interaction.

Candidate families (distinct from existing library and miner_3 batch AA):
  rev_5d / rev_vol_5d      : short-term reversal (classic, not in library)
  gain_loss_ratio_20/60d   : profit-factor style return asymmetry
  trend_consistency_60d    : fraction of positive days (trend quality)
  variance_ratio_10d       : VR>1 trending / VR<1 mean-reverting
  intraday_pos_10d         : mean daily close position within high-low range
  efficiency_ratio_60d     : Kaufman efficiency at 60d (AA tested 20d)
  mom_eff_20x60            : momentum x efficiency interaction
  us_tech_beta_60d         : rolling beta to NDX (tech transmission)
  dxy_beta_60d             : rolling beta to DXY (USD sensitivity)
  sma_cross_50_200         : (sma50-sma200)/sma200 golden-cross distance
  ret_autocorr_5d          : 5d-lag return autocorrelation
  skew_60d                 : rolling skewness 60d (crash risk)
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, max_library_corr,
                                 TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

# macro signals
def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)

# high/low/open panels
hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
op = pd.concat({a: panels[a]["open"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

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


vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

C = {}
# 1. short-term reversal (5d)
C["rev_5d"] = -rets.rolling(5).mean()
# 2. vol-adjusted reversal
C["rev_vol_5d"] = -rets.rolling(5).mean() / (vol20 + 1e-12)
# 3. gain/loss profit factor 20d
pos = rets.clip(lower=0)
neg = rets.clip(upper=0)
C["gain_loss_ratio_20d"] = pos.rolling(20).sum() / (neg.rolling(20).sum().abs() + 1e-12)
# 4. gain/loss profit factor 60d
C["gain_loss_ratio_60d"] = pos.rolling(60).sum() / (neg.rolling(60).sum().abs() + 1e-12)
# 5. trend consistency: fraction of positive days over 60d
C["trend_consistency_60d"] = (rets > 0).rolling(60).mean()
# 6. variance ratio 10d (trend persistence)
C["variance_ratio_10d"] = rets.rolling(10).var() / (rets.rolling(1).var() * 10 + 1e-14)
# 7. intraday position: mean (close-low)/(high-low) over 10d
rng = (hi - lo).replace(0, np.nan)
C["intraday_pos_10d"] = ((closes - lo) / rng).rolling(10).mean()
# 8. Kaufman efficiency ratio 60d
C["efficiency_ratio_60d"] = (closes - closes.shift(60)).abs() / (rets.abs().rolling(60).sum() + 1e-12)
# 9. momentum x efficiency interaction (20d)
eff20 = (closes - closes.shift(20)).abs() / (rets.abs().rolling(20).sum() + 1e-12)
C["mom_eff_20x60"] = (closes / closes.shift(20) - 1.0) * eff20
# 10. tech beta: rolling beta to NDX (60d)
C["us_tech_beta_60d"] = rolling_beta(rets, rets["NDX"], 60)
# 11. DXY beta 60d
C["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60)
# 12. golden cross distance (sma50-sma200)/sma200
sma50 = closes.rolling(50).mean()
sma200 = closes.rolling(200).mean()
C["sma_cross_50_200"] = (sma50 - sma200) / (sma200 + 1e-12)
# 13. 5d-lag return autocorrelation over 20d window
C["ret_autocorr_5d"] = rets.rolling(20).apply(lambda s: s.autocorr(lag=5), raw=False)
# 14. rolling skewness 60d
C["skew_60d"] = rets.rolling(60).skew()

print(f"{len(C)} candidates; time {time.time()-t0:.1f}s", flush=True)

results = {}
for i, (name, sig) in enumerate(C.items()):
    ics = rank_ic_series(sig, fwd, min_valid=8)
    s = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252):
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
          f"cov={cov['coverage_dates_ge8']:.2f} to={to:.2f}{flag}", flush=True)
    results[name] = (s, ics, sig)
    print(f"  [{i+1}/{len(C)}] {name} done {time.time()-t0:.1f}s", flush=True)

# decay + library correlation for full-pass candidates
print("\n=== DECAY + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
# reference library signals (effective factors + historical library defs)
lib = {}
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / vol20
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = vol20.rolling(60).std()
lib["vol_ratio_20_60"] = vol20 / (vol60 + 1e-12)
lib["vix_beta_cond_60x20"] = rolling_beta(rets, vix.pct_change(), 60) * (vix.pct_change().rolling(20).mean() > 0).astype(float)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).rolling(60).median()
lib["hl_pos_20d"] = (closes - closes.rolling(20).min()) / ((closes.rolling(20).max() - closes.rolling(20).min()) + 1e-12)
lib["max_dd_60d"] = closes.rolling(60).max() / closes - 1.0
lib["kurt_60d"] = rets.rolling(60).kurt()
lib["rsi_14"] = closes.rolling(14).apply(lambda s: (s.diff().clip(lower=0).sum() / (s.diff().abs().sum() + 1e-12)), raw=False)
lib["vol_price_corr_20"] = rets.rolling(20).corr(vol20)

passers = [k for k, (s, _, _) in results.items() if abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840]
for name in passers:
    sig = results[name][2]
    dec = decay_profile(sig, closes, horizons=(1, 2, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    corr, key = max_library_corr(sig, lib)
    print(f"{name:24s} decay={dec} max_lib_corr={corr} (vs {key})", flush=True)

print(f"\nTotal runtime {time.time()-t0:.1f}s", flush=True)
