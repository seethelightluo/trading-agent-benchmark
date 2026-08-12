"""miner_3 batch exploration (2030-02-25) - candidate factor screen batch S (vectorized v2).

Tests novel, interpretable cross-asset factors on the 15-instrument universe.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (paper, daily rank IC).
Uses data through the previous completed trading day (2030-02-22).
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=3200)
closes = close_panel(panels)
rets = closes.pct_change()

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

H = 10

def rolling_beta(y, x, win=60, min_obs=40):
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

# ---------------- candidates (vectorized) ----------------
C = {}
roll_std20 = rets.rolling(20).std()
roll_std60 = rets.rolling(60).std()

def downside_std(r, win=60):
    return np.sqrt((r.clip(upper=0)**2).rolling(win).mean())

C["skew_60d"] = rets.rolling(60).skew()
C["sortino_60d"] = rets.rolling(60).mean() / (downside_std(rets, 60) + 1e-9)
hl = {}
for a in TRADABLE:
    df = panels[a]
    hl[a] = (df["high"].astype(float) - df["low"].astype(float)) / df["close"].astype(float)
hl = pd.DataFrame(hl, index=closes.index)
C["range_ratio_20d"] = hl.rolling(20).mean()
gap = {}
for a in TRADABLE:
    df = panels[a]
    gap[a] = df["open"].astype(float) / df["close"].astype(float).shift(1) - 1.0
gap = pd.DataFrame(gap, index=closes.index)
C["gap_20d"] = gap.rolling(20).mean()
log_intra = np.log(closes / np.vstack([panels[a]["open"].astype(float) for a in TRADABLE]).T)
log_intra = pd.DataFrame(log_intra, index=closes.index, columns=closes.columns)
C["intraday_20d"] = np.exp(log_intra.rolling(20).sum()) - 1.0
dxy_trend = np.sign(dxy.pct_change(60))
C["dxy_cond_momaccel"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)).mul(dxy_trend, axis=0)
C["max_dd_60d"] = (1 + rets).cumprod().div((1 + rets).cumprod().rolling(60).max()) - 1.0
C["sharpe_60d"] = rets.rolling(60).mean() / (roll_std60 + 1e-9)
# autocorr lag5: rolling corr(ret, ret.shift(5)) over 60d window
C["autocorr5_60d"] = rets.rolling(60).corr(rets.shift(5))
C["mom10skip5_vol20"] = (closes.shift(5) / closes.shift(15) - 1.0) / roll_std20
C["up_hit_60d"] = (rets > 0).rolling(60).mean()
C["downside_vol_ratio_60d"] = downside_std(rets, 60) / (roll_std60 + 1e-9)
C["mom3_vol20"] = (closes/closes.shift(3)-1.0) / roll_std20
C["up_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(lower=0), 60)
C["beta_asym_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60) - rolling_beta(rets, mkt_ret.clip(lower=0), 60)
C["btc_beta_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
C["xau_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60)
C["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60)
C["vix_beta_60d"] = rolling_beta(rets, vix.pct_change(), 60)
vix_trend = np.sign(vix.pct_change(20))
C["vix_cond_mom5"] = (closes/closes.shift(5)-1.0).mul(vix_trend, axis=0)
usdjpy_trend = np.sign(usdjpy.pct_change(60))
C["usdjpy_cond_momaccel"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)).mul(usdjpy_trend, axis=0)
eurusd_trend = np.sign(eurusd.pct_change(60))
C["eurusd_cond_momaccel"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)).mul(eurusd_trend, axis=0)
C["volreg_cond_mom20"] = (closes/closes.shift(20)-1.0).mul(np.sign(roll_std20 - roll_std60), axis=0)
C["recovery_60d"] = closes / closes.rolling(60).max()
us10y = panels["US10Y"]["close"].astype(float)
cn10y = panels["CN10Y"]["close"].astype(float)
C["rate_spread_beta_60d"] = rolling_beta(rets, (us10y - cn10y).pct_change(), 60)

fwd10 = forward_returns(closes, H)
rows = []
for name, panel in C.items():
    ics = rank_ic_series(panel, fwd10)
    if len(ics) < 100:
        print(f"{name:26s} SKIP (n_ic={len(ics)})", flush=True)
        continue
    m = summarize_ic(ics, expected_sign=1)
    cov = coverage_metrics(panel)
    tov = turnover_rank(panel, 10)
    dec = decay_profile(panel, closes, horizons=(1, 3, 5, 10, 20))
    corr, ck = max_lib_corr(panel)
    ic_r500 = round(float(ics.tail(500).mean()), 4) if len(ics) >= 500 else None
    icir_r500 = round(float(ics.tail(500).mean() / ics.tail(500).std(ddof=1)), 4) if len(ics) >= 500 and ics.tail(500).std(ddof=1) > 0 else None
    gate = (abs(m["ic"]) >= 0.0070) and (abs(m["icir"]) >= 0.0840)
    rows.append((name, m["ic"], m["icir"], m["ic_hit_ratio"], len(ics), cov["coverage_asset_days"],
                 cov["coverage_dates_ge8"], tov, dec, corr, ck, ic_r500, icir_r500, gate))
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={len(ics):4d} "
          f"cov={cov['coverage_asset_days']:.2f} ge8={cov['coverage_dates_ge8']:.2f} tov={tov} "
          f"decay={dec} libcorr={corr}({ck}) r500_ic={ic_r500} r500_icir={icir_r500} GATE={gate}", flush=True)

print("\n=== summary sorted by |ICIR| ===", flush=True)
rows.sort(key=lambda r: -abs(r[2]))
for r in rows:
    print(f"{r[0]:26s} IC={r[1]:+.4f} ICIR={r[2]:+.4f} n={r[4]:4d} cov={r[5]:.2f} ge8={r[6]:.2f} "
          f"r500_ic={r[11]} r500_icir={r[12]} libcorr={r[9]}({r[10]}) GATE={r[13]}", flush=True)
print(f"elapsed {time.time()-t0:.1f}s", flush=True)
