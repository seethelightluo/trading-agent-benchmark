"""miner_3 batch W (2031-06-16) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2031-06-13). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift and max-abs library
correlation for passers. No live-account interaction.
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

# volume availability probe
vol_ok = {}
for a in TRADABLE:
    df = panels.get(a)
    vol_ok[a] = (df is not None and "volume" in df.columns and df["volume"].notna().sum() > 100)
print("volume available:", {a: vol_ok[a] for a in TRADABLE}, flush=True)

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
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}", flush=True)
    return ics, s

for name, sig in existing.items():
    report(name, sig)

# ---------------- 2) CANDIDATE SCREEN (batch W) ----------------
print("\n=== CANDIDATE SCREEN (batch W, full history) ===", flush=True)
C = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

# W1: Kaufman efficiency ratio 60d (trend quality, long)
C["er_60d"] = (closes - closes.shift(60)).abs() / rets.abs().rolling(60).sum()
# W2: vol-scaled MA slope 20x60 (trend strength normalized)
C["ma_slope_20x60_vol"] = (closes.rolling(20).mean() / closes.rolling(60).mean() - 1.0) / vol20
# W3: stochastic position 5d (short-range mean reversion)
C["stoch_5d"] = (closes - closes.rolling(5).min()) / (closes.rolling(5).max() - closes.rolling(5).min() + 1e-9)
# W4: Donchian breakout 20d (close vs 20d high)
C["donchian_20d"] = closes / closes.rolling(20).max() - 1.0
# W5: days since 60d high (time-under-water proxy; lower=longer since peak -> bearish)
def days_since_high(px, win=60):
    out = {}
    for a in px.columns:
        arr = px[a].values
        ds = np.full(len(arr), np.nan)
        for t in range(win, len(arr)):
            w = arr[t-win:t+1]
            ds[t] = win - 1 - int(np.argmax(w[::-1])) if np.isfinite(w).all() else np.nan
        out[a] = ds
    return pd.DataFrame(out, index=px.index)
C["days_since_high_60d"] = -days_since_high(closes, 60)  # negate: higher = fresher high
# W6: return skewness 20d (lottery/left-tail)
C["skew_20d"] = rets.rolling(20).skew()
# W7: up/down volatility ratio 20d (asymmetric risk)
up_ret = rets.where(rets > 0, np.nan)
dn_ret = rets.where(rets < 0, np.nan)
C["up_down_vol_ratio_20d"] = up_ret.rolling(20).std() / (dn_ret.rolling(20).std() + 1e-9)
# W8: cross-sectional relative-strength z-score (20d return vs cross-section)
cs_mean = (closes/closes.shift(20)-1).mean(axis=1)
cs_std = (closes/closes.shift(20)-1).std(axis=1)
C["rel_strength_z_20d"] = ((closes/closes.shift(20)-1) - cs_mean) / (cs_std + 1e-9)
# W9: momentum conditioned on market 20d trend
C["mkt_cond_mom20"] = (closes/closes.shift(20)-1) * np.sign(mkt_ret.rolling(20).mean()).to_frame(0).values
# W10: energy beta 60d (WTI linkage)
C["wti_beta_60d"] = rolling_beta(rets, closes["WTI"].pct_change(), 60)
# W11: copper/cyclical beta 60d
C["copper_beta_60d"] = rolling_beta(rets, closes["COPPER"].pct_change(), 60)
# W12: correlation with SPX 60d (market cohesion)
C["spx_corr_60d"] = rolling_corr(rets, closes["SPX"].pct_change(), 60)
# W13: drawup recovery 20d (close vs 20d min)
C["drawup_20d"] = closes / closes.rolling(20).min() - 1.0
# W14: 1y price z-score (52-week position)
C["zscore_252d"] = (closes - closes.rolling(252).mean()) / closes.rolling(252).std()
# W15: vol-scaled 5d reversal (short-term contrarian)
C["reversal_5d_vol20"] = -(closes/closes.shift(5)-1) / vol20
# W16: high-low position 60d
C["hl_pos_60d"] = (closes - closes.rolling(60).min()) / (closes.rolling(60).max() - closes.rolling(60).min() + 1e-9)
# W17: max consecutive up days over 10d (streak)
def max_streak(px, win=10):
    out = {}
    for a in px.columns:
        r = px[a].pct_change()
        s = (r > 0).astype(int)
        streak = np.zeros(len(s))
        cnt = 0
        for i in range(len(s)):
            cnt = cnt + 1 if s.iloc[i] else 0
            streak[i] = cnt
        st = pd.Series(streak, index=px.index)
        out[a] = st.rolling(win).max()
    return pd.DataFrame(out, index=px.index)
C["max_streak_10d"] = max_streak(closes, 10)
# W18: vol-scaled ER 20d (trend quality per unit risk)
C["er20_vol20"] = ((closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()) / (vol20 + 1e-9)
# W19: overnight gap 5d avg (open/prev close - 1)
gap = pd.DataFrame({a: panels[a]["open"].astype(float) / panels[a]["close"].astype(float).shift(1) - 1.0
                    for a in TRADABLE}, index=closes.index)
C["gap_avg_5d"] = gap.rolling(5).mean()
# W20: conditional gold beta: beta to XAU * sign(XAU 20d trend)
C["xau_cond_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60) * np.sign(closes["XAU"]/closes["XAU"].shift(20)-1).to_frame(0).values
# W21: z-score of 10d return (short RS)
cs_mean10 = (closes/closes.shift(10)-1).mean(axis=1)
cs_std10 = (closes/closes.shift(10)-1).std(axis=1)
C["rel_strength_z_10d"] = ((closes/closes.shift(10)-1) - cs_mean10) / (cs_std10 + 1e-9)

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
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    results[name] = (s, ics, sig)

# ---------------- 3) decay + library correlation for full-pass ----------------
print("\n=== DECAY + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
lib = dict(existing)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
vix_ret = vix.pct_change()
vix_beta = rolling_beta(rets, vix_ret, 60)
lib["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
lib["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
lib["vol_ratio_20_60"] = vol20 / vol60

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

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
