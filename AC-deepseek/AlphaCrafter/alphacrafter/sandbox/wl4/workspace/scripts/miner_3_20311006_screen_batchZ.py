"""miner_3 batch Z (2031-10-06) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2031-10-03). Uses the
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
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)

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

def rolling_r2(logp, win=60, min_obs=40):
    """R2 of rolling linear trend fit on log price."""
    out = {}
    for a in logp.columns:
        s = logp[a].dropna()
        x = np.arange(len(s))
        res = {}
        for i in range(win - 1, len(s)):
            y = s.iloc[i - win + 1:i + 1].values
            if np.isnan(y).any() or len(y) < min_obs:
                res[s.index[i]] = np.nan
                continue
            xv = x[i - win + 1:i + 1]
            xm, ym = xv.mean(), y.mean()
            ssxy = ((xv - xm) * (y - ym)).sum()
            ssxx = ((xv - xm) ** 2).sum()
            ssyy = ((y - ym) ** 2).sum()
            r2 = (ssxy ** 2 / (ssxx * ssyy)) if ssxx > 0 and ssyy > 0 else np.nan
            res[s.index[i]] = r2
        out[a] = pd.Series(res)
    return pd.DataFrame(out, index=logp.index)

def rolling_coskew(y, mkt, win=60, min_obs=40):
    """Co-skewness of each asset with the market: E[(y-ym)(m-mbar)^2]/(sy*sm^2)."""
    out = {}
    for a in y.columns:
        df = pd.concat([y[a].rename("y"), mkt.rename("m")], axis=1).dropna()
        vals = df.values
        n = len(vals)
        res = np.full(n, np.nan)
        for i in range(win - 1, n):
            w = vals[i - win + 1:i + 1]
            yy, mm = w[:, 0], w[:, 1]
            sy, sm = yy.std(ddof=0), mm.std(ddof=0)
            if sy == 0 or sm == 0:
                continue
            res[i] = float(np.mean((yy - yy.mean()) * (mm - mm.mean()) ** 2) / (sy * sm * sm))
        out[a] = pd.Series(res, index=df.index)
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

# ---------------- 2) CANDIDATE SCREEN (batch Z - novel) ----------------
print("\n=== CANDIDATE SCREEN (batch Z, full history) ===", flush=True)
C = {}
logc = np.log(closes)
sma60 = closes.rolling(60).mean()
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
C["ma_dist_z_60"] = (closes - sma60) / (vol20 * np.sqrt(60))
C["trend_r2_60"] = rolling_r2(logc, 60)
C["semi_dev_20"] = rets.rolling(20).mean() / rets.clip(upper=0).rolling(20).std()
C["co_skew_mkt_60"] = rolling_coskew(rets, mkt_ret, 60)
C["vix_beta_60"] = rolling_beta(rets, vix.pct_change(), 60)
C["dxy_beta_60"] = rolling_beta(rets, dxy.pct_change(), 60)
C["usdjpy_beta_60"] = rolling_beta(rets, usdjpy.pct_change(), 60)
C["breadth_rs_60"] = (closes/closes.shift(60)-1) - mkt_ret.rolling(60).sum()
spread = closes["CN10Y"] - closes["US10Y"]
C["spread_beta_60"] = rolling_beta(rets, spread.diff(), 60)
C["rev_5d"] = -(closes.shift(5)/closes - 1.0)
hi20 = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
C["range_break_20"] = closes / hi20.rolling(20).max() - 1.0
C["corr_mkt_20"] = rets.rolling(20).corr(mkt_ret)
C["kelly_frac_60"] = rets.rolling(60).mean() / (rets.rolling(60).var() + 1e-12)
C["dn_rate_beta_60"] = rolling_beta(rets, closes["US10Y"].pct_change().clip(upper=0), 60)
C["upside_beta_mkt_60"] = rolling_beta(rets, mkt_ret.clip(lower=0), 60)
# volume trend (volume column; coverage may be low for some index assets)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
C["volume_trend_20"] = vol_panel.rolling(20).mean() / vol_panel.rolling(60).mean()

print(f"{len(C)} candidates + 3 re-validation factors; time {time.time()-t0:.1f}s", flush=True)
results = {}
for i, (name, sig) in enumerate(C.items()):
    s, ics = report(name, sig, expected_sign=1)
    results[name] = (s, ics, sig)
    print(f"  [{i+1}/{len(C)}] {name} done {time.time()-t0:.1f}s", flush=True)

# ---------------- 3) decay + library correlation for full-pass ----------------
print("\n=== DECAY + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
lib = dict(existing)
lib.update({k: v[2] for k, v in results.items()})
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std() / rets.rolling(60).std()
lib["vol_ratio_20_60"] = vol20 / vol60
lib["volume_z_20"] = (vol_panel - vol_panel.rolling(60).mean()) / vol_panel.rolling(60).std()
lib["rsi_14"] = closes.rolling(14).apply(lambda s: (s.diff().clip(lower=0).sum() / (s.diff().abs().sum() + 1e-12)), raw=False)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["vix_beta_cond_60x20"] = rolling_beta(rets, vix.pct_change(), 60) * (vix.pct_change().rolling(20).mean() > 0).astype(float)
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).rolling(60).median()
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60) * (closes["US10Y"].pct_change().rolling(60).mean() > 0).astype(float)

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
print(f"FULL-PASS count: {len(passing)}", flush=True)
for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20))
    corr, key = max_lib_corr(sig, exclude=name)
    print(f"{name:26s} decay={dec} max_abs_lib_corr={corr:.4f} (vs {key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
