"""miner_2 batch Y (2031-08-11) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2031-08-08). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift and max-abs library
correlation for passers. No live-account interaction.
"""
import sys, time, warnings, json
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

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)

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

def rolling_corr(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        c = z["y"].rolling(win).corr(z["x"]).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = c
    return pd.DataFrame(out, index=y.index)

# ---------------- 1) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
existing["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
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
            recent[w] = (mm, mm/ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics, sig

for name, sig in existing.items():
    report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)

# ---------------- 2) CANDIDATE SCREEN (batch Y) ----------------
print("\n=== CANDIDATE SCREEN (batch Y, full history) ===", flush=True)
C = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol120 = rets.rolling(120).std()

# Y1: rolling skewness 20d (crash-risk asymmetry)
C["skew_20d"] = rets.rolling(20).skew()
# Y2: downside vol ratio 60d = semi-dev / total vol (asymmetry)
def downside_ratio(r, win=60):
    out = {}
    for a in r.columns:
        v = r[a]
        down = v.clip(upper=0)
        sd = down.rolling(win).std()
        tot = v.rolling(win).std()
        out[a] = sd / (tot + 1e-12)
    return pd.DataFrame(out, index=r.index)
C["downside_ratio_60d"] = downside_ratio(rets, 60)
# Y3: 5d return autocorrelation (reversal tendency)
C["autocorr_5d"] = rets.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=5) if len(x) == 20 else np.nan, raw=False)
# Y4: Parkinson ratio: (high-low)/close vol vs close-based vol (intraday efficiency)
if "high" in panels["SPX"].columns and "low" in panels["SPX"].columns:
    hl = pd.concat({a: (panels[a]["high"] - panels[a]["low"]) / panels[a]["close"] for a in TRADABLE if a in panels}, axis=1).sort_index()
    park = hl.rolling(20).std()
    C["parkinson_ratio_20d"] = park / (vol20 + 1e-12)
# Y5: beta to DXY 60d (USD-strength sensitivity)
C["beta_dxy_60d"] = rolling_beta(rets, dxy.pct_change(), 60)
# Y6: beta to USDJPY 60d (risk-on currency sensitivity)
C["beta_usdjpy_60d"] = rolling_beta(rets, usdjpy.pct_change(), 60)
# Y7: beta to BTC 60d (crypto sentiment sensitivity)
C["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
# Y8: drawdown from 60d high (negative distance from high)
C["drawdown_60d"] = (closes - closes.rolling(60).max()) / closes.rolling(60).max()
# Y9: Kaufman efficiency ratio 20d
def efficiency_ratio(px, win=20):
    out = {}
    for a in px.columns:
        v = px[a]
        num = (v - v.shift(win)).abs()
        den = v.diff().abs().rolling(win).sum()
        out[a] = num / (den + 1e-12)
    return pd.DataFrame(out, index=px.index)
C["eff_ratio_20d"] = efficiency_ratio(closes, 20)
# Y10: excess 20d momentum vs equal-weight market (relative strength)
C["excess_mom_20d"] = (closes/closes.shift(20)-1) - mkt_ret.rolling(20).mean().to_frame(0).values
# Y11: 20d range position (close position in 20d high-low range)
C["hl_pos_20d"] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())
# Y12: vol-regime conditional momentum: mom20 * (vol20<vol60) [trend works in calm regimes]
calm = (vol20 < vol60).astype(float)
C["calm_mom20"] = (closes/closes.shift(20)-1) * calm
# Y13: second-order vol-of-vol ratio: vol20 vol-of-vol vs vol60 vol-of-vol
C["vov_ratio_20x60"] = vol20.rolling(60).std() / (vol60.rolling(120).std() + 1e-12)
# Y14: downside beta 20d (fast downside sensitivity)
C["dn_beta_20d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 20)
# Y15: lottery proxy: max 1d gain over 20d
C["max_gain_20d"] = rets.rolling(20).max()
# Y16: relative momentum vs XAU (risk-off anchor)
C["rel_mom_xau_20d"] = (closes/closes.shift(20)-1) - (closes["XAU"]/closes["XAU"].shift(20)-1).to_frame(0).values
# Y17: EURUSD beta 60d (anti-USD carry)
C["beta_eurusd_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
# Y18: VIX level interaction: -beta(asset,VIX) * VIX level (fear sensitivity scaled by level)
vix_ret = vix.pct_change()
C["vix_beta_x_level_60d"] = -rolling_beta(rets, vix_ret, 60) * (vix / vix.shift(60)).to_frame(0).values

results = {}
for name, sig in C.items():
    s, ics, _ = report(name, sig, expected_sign=1)
    results[name] = (s, ics, sig)

# ---------------- 3) decay + library correlation for full-pass ----------------
print("\n=== DECAY + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
lib = dict(existing)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
vix_beta = rolling_beta(rets, vix_ret, 60)
lib["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
lib["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
lib["vol_ratio_20_60"] = vol20 / vol60
lib["rsi_14"] = 100 - 100/(1 + (rets.clip(lower=0).rolling(14).mean())/((-rets.clip(upper=0)).rolling(14).mean()+1e-9))
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["volume_z_20"] = pd.concat({a: (panels[a]["volume"].astype(float) - panels[a]["volume"].astype(float).rolling(60).mean()) / panels[a]["volume"].astype(float).rolling(60).std() for a in TRADABLE if a in panels}, axis=1).sort_index()
for name, sig in C.items():
    lib.setdefault(name, sig)

def max_lib_corr(sig, exclude):
    best, key = 0.0, None
    for lname, lsig in lib.items():
        if lname == exclude or lsig is None:
            continue
        both = pd.concat([sig.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rr = float(both["c"].corr(both["l"]))
        if abs(rr) > best:
            best, key = abs(rr), lname
    return round(best, 4), key

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"\nFull-pass count: {len(passing)}", flush=True)
for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    corr, key = max_lib_corr(sig, exclude=name)
    print(f"{name:26s} decay={dec} max_abs_lib_corr={corr:.4f} (vs {key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
