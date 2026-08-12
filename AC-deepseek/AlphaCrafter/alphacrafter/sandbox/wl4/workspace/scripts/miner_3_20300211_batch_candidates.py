"""miner_3 batch exploration (2030-02-11) - candidate factor screen.

Tests novel, interpretable cross-asset factors on the 15-instrument universe.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (paper, daily rank IC).
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
dxy = panels["DXY"]["close"].astype(float) if "DXY" in panels else None
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s")

H = 10  # admission horizon

# ---------------- library signal recomputation (3 effective factors) ----------------
def rolling_beta(y, x, win=60, min_obs=40):
    """y: asset ret panel, x: common factor series -> beta panel."""
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

mkt_ret = rets.mean(axis=1)
lib = {}
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].pct_change(), 60)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()

def max_lib_corr(cand):
    best, bestk = 0.0, None
    for k, s in lib.items():
        both = pd.concat([cand.stack().rename("c"), s.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["c"].corr(both["l"]))
        if abs(r) > best:
            best, bestk = abs(r), k
    return round(best, 4), bestk

# ---------------- candidate factors ----------------
C = {}
roll_std20 = rets.rolling(20).std()

# 1. realized skewness 60d (crash-risk proxy)
C["skew_60d"] = rets.rolling(60).skew()
# 2. sortino-like: mean / downside-std 60d
def downside_std(r, win=60):
    neg = r.clip(upper=0)
    return np.sqrt((neg**2).rolling(win).mean())
C["sortino_60d"] = rets.rolling(60).mean() / (downside_std(rets, 60) + 1e-9)
# 3. range ratio 20d: mean((high-low)/close)
hl = {}
for a in TRADABLE:
    df = panels[a]
    hl[a] = (df["high"].astype(float) - df["low"].astype(float)) / df["close"].astype(float)
hl = pd.DataFrame(hl, index=closes.index)
C["range_ratio_20d"] = hl.rolling(20).mean()
# 4. overnight gap 20d: mean(open[t]/close[t-1]-1)
gap = {}
for a in TRADABLE:
    df = panels[a]
    gap[a] = df["open"].astype(float) / df["close"].astype(float).shift(1) - 1.0
gap = pd.DataFrame(gap, index=closes.index)
C["gap_20d"] = gap.rolling(20).mean()
# 5. intraday return 20d: cumprod(close/open)
intra = {}
for a in TRADABLE:
    df = panels[a]
    intra[a] = (df["close"].astype(float) / df["open"].astype(float)).rolling(20).apply(lambda x: np.prod(x), raw=True) - 1.0
C["intraday_20d"] = pd.DataFrame(intra, index=closes.index)
# 6. DXY-conditioned momentum acceleration: (r20-r60)*sign(DXY 60d chg)
if dxy is not None:
    dxy_trend = np.sign(dxy.pct_change(60))
    C["dxy_cond_momaccel"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)) * dxy_trend.values.reshape(-1, 1)
# 7. max drawdown 60d (negative depth)
def mdd(r, win=60):
    c = (1 + r).cumprod()
    return c / c.rolling(win).max() - 1.0
C["max_dd_60d"] = (1 + rets).cumprod().div((1 + rets).cumprod().rolling(60).max()) - 1.0
# 8. rolling sharpe 60d
C["sharpe_60d"] = rets.rolling(60).mean() / (rets.rolling(60).std() + 1e-9)
# 9. autocorr lag5 over 60d
def autocorr_lag5(r):
    r = r.dropna()
    if len(r) < 70:
        return np.nan
    x = r.iloc[-60:]
    if x.std() < 1e-12:
        return np.nan
    return x.autocorr(lag=5)
C["autocorr5_60d"] = rets.rolling(60).apply(lambda r: autocorr_lag5(pd.Series(r)), raw=False)
# 10. mom 10d skip5 / vol20
C["mom10skip5_vol20"] = (closes.shift(5) / closes.shift(15) - 1.0) / roll_std20
# 11. up-day hit rate 60d
C["up_hit_60d"] = (rets > 0).rolling(60).mean()
# 12. downside vol ratio: vol(neg rets)/total vol
C["downside_vol_ratio_60d"] = downside_std(rets, 60) / (rets.rolling(60).std() + 1e-9)
# 13. half-life: 3d momentum / vol20 (short-term)
C["mom3_vol20"] = (closes/closes.shift(3)-1.0) / roll_std20

fwd10 = forward_returns(closes, H)
rows = []
for name, panel in C.items():
    ics = rank_ic_series(panel, fwd10)
    if len(ics) < 100:
        print(f"{name:24s} SKIP (n_ic={len(ics)})")
        continue
    m = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(panel)
    tov = turnover_rank(panel, 10)
    dec = decay_profile(panel, closes, horizons=(1, 3, 5, 10, 20))
    corr, ck = max_lib_corr(panel)
    # recent-window checks
    ic_r500 = round(float(ics.tail(500).mean()), 4) if len(ics) >= 500 else None
    icir_r500 = round(float(ics.tail(500).mean() / ics.tail(500).std(ddof=1)), 4) if len(ics) >= 500 and ics.tail(500).std(ddof=1) > 0 else None
    gate = (abs(m["ic"]) >= 0.0070) and (abs(m["icir"]) >= 0.0840)
    rows.append((name, m["ic"], m["icir"], m["ic_hit_ratio"], len(ics), cov["coverage_asset_days"],
                 cov["coverage_dates_ge8"], tov, dec, corr, ck, ic_r500, icir_r500, gate))
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={len(ics):4d} "
          f"cov={cov['coverage_asset_days']:.2f} ge8={cov['coverage_dates_ge8']:.2f} tov={tov} "
          f"decay={dec} libcorr={corr}({ck}) r500_ic={ic_r500} r500_icir={icir_r500} GATE={gate}")

print("\n=== summary sorted by |ICIR| ===")
rows.sort(key=lambda r: -abs(r[2]))
for r in rows:
    print(f"{r[0]:24s} IC={r[1]:+.4f} ICIR={r[2]:+.4f} n={r[4]:4d} cov={r[5]:.2f} ge8={r[6]:.2f} r500_ic={r[11]} r500_icir={r[12]} GATE={r[13]}")
print(f"elapsed {time.time()-t0:.1f}s")
