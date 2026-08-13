"""miner_3 batch X (2031-07-28) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2031-07-25). Uses the
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

# ---------------- 2) CANDIDATE SCREEN (batch X) ----------------
print("\n=== CANDIDATE SCREEN (batch X, full history) ===", flush=True)
C = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

# X1: batch W passers re-test with fresh data
C["ma_slope_20x60_vol"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)) / rets.rolling(60).std()
# days since rolling high (60d)
C["days_since_high_60d"] = pd.DataFrame({a: (closes.index.to_series() - closes[a].rolling(60).apply(lambda x: closes[a].loc[x.index].idxmax() if False else np.nan, raw=False)).days if False else np.nan for a in closes.columns}, index=closes.index)
# simpler days_since_high: count trading days since 60d rolling max
def days_since_high(px, win=60):
    out = {}
    for a in px.columns:
        v = px[a]
        rmax = v.rolling(win).max()
        ds = pd.Series(np.nan, index=v.index)
        cnt = 0
        # vectorized: for each date, find last date where rolling max was achieved
        # use expanding count of days since the max within window
        ismax = (v == rmax).astype(int)
        # days since last 1 in ismax
        grp = ismax.groupby((ismax != ismax.shift()).cumsum())
        out[a] = grp.cumcount() + 1
    return pd.DataFrame(out, index=px.index)
C["days_since_high_60d"] = days_since_high(closes, 60)
C["copper_beta_60d"] = rolling_beta(rets, closes["COPPER"].pct_change(), 60)
# drawup 20d: (close - rolling_min(close,20)) / close
C["drawup_20d"] = (closes - closes.rolling(20).min()) / closes
# zscore_252d
C["zscore_252d"] = (closes - closes.rolling(252).mean()) / closes.rolling(252).std()
# hl_pos_60d: (close - min)/(max - min) over 60d
C["hl_pos_60d"] = (closes - closes.rolling(60).min()) / (closes.rolling(60).max() - closes.rolling(60).min())

# X2: NEW candidates - batch X novel ideas
# X-a: risk-adjusted 20d momentum (Sharpe-like trend quality)
C["mom20_vol20"] = (closes/closes.shift(20)-1) / (vol20 + 1e-9)
# X-b: 60d momentum scaled by 60d vol
C["mom60_vol60"] = (closes/closes.shift(60)-1) / (vol60 + 1e-9)
# X-c: vol term structure 5d/60d (short-term vol expansion vs long)
C["vol_ratio_5_60"] = rets.rolling(5).std() / (vol60 + 1e-9)
# X-d: US10Y-CN10Y spread beta (yield-curve/global-rates exposure)
spread = closes["US10Y"] - closes["CN10Y"]
C["spread_beta_60d"] = rolling_beta(rets, spread.pct_change(), 60)
# X-e: USDJPY beta (yen carry / risk sentiment)
C["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy.pct_change(), 60)
# X-f: DXY beta (dollar direction exposure)
C["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60)
# X-g: correlation with market * market trend (regime-conditional commonality)
mkt_trend = np.sign(mkt_ret.rolling(20).mean())
C["mkt_corr_trend_60"] = rolling_corr(rets, mkt_ret, 60) * mkt_trend.to_frame(0).values
# X-h: 20d momentum x vol-ratio regime (momentum gated by vol expansion)
C["mom20_x_volregime"] = (closes/closes.shift(20)-1) * np.sign(rets.rolling(5).std() / (vol20 + 1e-9) - 1.0)
# X-i: distance from 252d high (long-term trend position)
C["dist_252d_high"] = closes / closes.rolling(252).max() - 1.0
# X-j: 20d return consistency (fraction of up days * magnitude)
up_frac = (rets > 0).rolling(20).mean()
C["up_frac_20d"] = up_frac * np.sign(closes/closes.shift(20)-1)
# X-k: XAU-conditional beta (gold trend regime)
C["xau_cond_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60) * np.sign(closes["XAU"]/closes["XAU"].shift(20)-1).to_frame(0).values
# X-l: volume z-score 20d (liquidity pulse) - requires volume panels
vols = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in TRADABLE}, index=closes.index).sort_index()
C["volume_z_20"] = (vols - vols.rolling(20).mean()) / (vols.rolling(20).std() + 1e-9)
# X-m: 5d reversal / 20d momentum hybrid (short-term mean reversion within trend)
C["rev5_mom20"] = -np.sign(closes/closes.shift(5)-1) * (closes/closes.shift(20)-1)
# X-n: EURUSD beta
C["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
# X-o: VIX beta (volatility regime exposure)
C["vix_beta_60d"] = rolling_beta(rets, vix.pct_change(), 60)
# X-p: down-market beta 20d (faster defensive beta)
C["dn_mkt_beta_20d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 20)
# X-q: RSI-like 14d
delta = rets
up = delta.clip(lower=0).rolling(14).mean()
dn = (-delta.clip(upper=0)).rolling(14).mean()
C["rsi_14"] = 100 - 100/(1 + up/(dn+1e-9))
# X-r: range position 10d (short-term stochastic)
C["range_pos_10d"] = (closes - closes.rolling(10).min()) / (closes.rolling(10).max() - closes.rolling(10).min())
# X-s: momentum acceleration 10x60 (faster accel)
C["mom_accel_10x60"] = (closes/closes.shift(10)-1) - (closes/closes.shift(60)-1)
# X-t: cross-asset dispersion regime: asset beta to equal-weight mkt * mkt 60d momentum
C["mkt_beta_x_mom60"] = rolling_beta(rets, mkt_ret, 60) * np.sign(mkt_ret.rolling(60).mean()).to_frame(0).values

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
vix_ret = vix.pct_change()
vix_beta = rolling_beta(rets, vix_ret, 60)
lib["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
lib["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
lib["vol_ratio_20_60"] = vol20 / vol60
# also include batchW passers already in C as lib proxies (to avoid self-correlation exclude self)
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
for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    corr, key = max_lib_corr(sig, exclude=name)
    print(f"{name:26s} decay={dec} max_abs_lib_corr={corr:.4f} (vs {key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
