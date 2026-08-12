"""miner_2 batch exploration (2030-05-06) - candidate factor screen batch U.

Theme: drawdown/recovery dynamics, trend breadth, EWMA momentum, systematic
beta/correlation, time-series z-score momentum, regime-conditional momentum,
volume-confirmation and price-range position on the 15-instrument tradable
cross-asset universe. Data through previous completed trading day (2030-05-03).

Admission gates (benchmark-wide): |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10
(paper, daily Spearman rank IC). Max abs library correlation reported for audit.
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
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
us10y = align(panels["US10Y"]["close"].astype(float), closes.index)
cn10y = align(panels["CN10Y"]["close"].astype(float), closes.index)

H = 10
mkt_ret = rets.mean(axis=1)

# ---------------- library reference signals for correlation audit ----------------
def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win, min_periods=min_obs).cov(z["x"])
        var = z["x"].rolling(win, min_periods=min_obs).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

lib = {}
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y.pct_change(), 60)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["vol_adj_mom_accel_20x60"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["vix_beta_cond_60x20"] = -rolling_beta(rets, vix.pct_change(), 60) * (vix / vix.shift(20) - 1.0)
lib["vol_price_corr_20"] = rets.rolling(20).corr((closes/closes.shift(20)-1).abs())
lib["volume_z_20"] = (panels["BTC"]["volume"].astype(float) / 1)  # placeholder replaced below
lib["us10y_cond_beta_60d"] = rolling_beta(rets, us10y.pct_change(), 60) * np.sign(us10y.pct_change(20))
lib["eurusd_beta_60d"] = rolling_beta(rets, align(panels["EURUSD"]["close"].astype(float), closes.index).pct_change(), 60)
lib["usdcny_beta_60d"] = rolling_beta(rets, align(panels["USDCNY"]["close"].astype(float), closes.index).pct_change(), 60)
lib["rsi_14"] = closes / closes.rolling(14).mean()  # proxy construction, audit only
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).median(axis=1)
lib["recovery_60d"] = closes / closes.rolling(60).max()
lib["skew_60d"] = rets.rolling(60).skew()
lib["efficiency_ratio_20d"] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()

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

# ---------------- derived panels ----------------
roll_max60 = closes.rolling(60, min_periods=30).max()
roll_max120 = closes.rolling(120, min_periods=60).max()
roll_min250 = closes.rolling(250, min_periods=120).min()
roll_max250 = closes.rolling(250, min_periods=120).max()
roll_std20 = rets.rolling(20).std()
roll_std60 = rets.rolling(60).std()
roll_std250 = rets.rolling(250).std()
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
downside_vol20 = np.sqrt((rets.clip(upper=0) ** 2).rolling(20).mean())
up_ret = rets.clip(lower=0)
dn_ret = rets.clip(upper=0).abs()

# days since 60d high (recency of new high): count consecutive days where close < rolling max
def days_since_high(px, win=60, min_periods=30):
    rmax = px.rolling(win, min_periods=min_periods).max()
    is_high = (px >= rmax - 1e-12)
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    counter = np.zeros(px.shape[1])
    vals = is_high.values
    outv = out.values
    for i in range(px.shape[0]):
        row = vals[i]
        for j in range(px.shape[1]):
            if not np.isfinite(px.values[i, j]):
                counter[j] = np.nan
            elif row[j]:
                counter[j] = 0
            elif np.isfinite(counter[j]):
                counter[j] += 1
        outv[i] = counter
    return pd.DataFrame(outv, index=px.index, columns=px.columns)

dsh60 = days_since_high(closes, 60)

# EWMA momentum
def ewma_mom(px, half_life=40):
    lam = 1.0 - 0.5 ** (1.0 / half_life)
    r = px.pct_change()
    out = r.ewm(alpha=lam, min_periods=half_life // 2).mean()
    return out

# cross-sectional dispersion of 20d returns
disp20 = mom20.std(axis=1)
disp_med60 = disp20.rolling(60, min_periods=30).median()
vix_med120 = vix.rolling(120, min_periods=60).median()

mom20_ts_mean = mom20.rolling(250, min_periods=120).mean()
mom20_ts_std = mom20.rolling(250, min_periods=120).std()

# volume confirmation: mean volume up days / mean volume down days over 20d
vol_p = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in TRADABLE}, index=closes.index).sort_index()
up_vol = (vol_p * (rets > 0)).rolling(20).sum()
dn_vol = (vol_p * (rets < 0)).rolling(20).sum()
up_cnt = (rets > 0).rolling(20).sum()
dn_cnt = (rets < 0).rolling(20).sum()

# ---------------- candidates ----------------
C = {}
# A. drawdown / recovery dynamics
C["dd_depth_60d"] = closes / roll_max60 - 1.0
C["dd_depth_120d"] = closes / roll_max120 - 1.0
C["dd_change_20d"] = (closes / roll_max60 - 1.0) - (closes.shift(20) / roll_max60.shift(20) - 1.0)
C["days_since_high_60d"] = -dsh60  # fewer days since high = fresher trend

# B. trend breadth
C["up_ratio_20d"] = (rets > 0).rolling(20).mean()
C["up_ratio_60d"] = (rets > 0).rolling(60).mean()

# C. EWMA momentum
C["ewma_mom_20d"] = ewma_mom(closes, 20)
C["ewma_mom_40d"] = ewma_mom(closes, 40)

# D. risk-adjusted momentum
C["mom20_sortino"] = mom20 / (downside_vol20 + 1e-9)
C["mom20_vol60"] = mom20 / (roll_std60 + 1e-9)

# E. systematic beta / correlation
C["beta_mkt_20d"] = rolling_beta(rets, mkt_ret, 20, min_obs=14)
C["beta_mkt_60d"] = rolling_beta(rets, mkt_ret, 60, min_obs=40)
C["corr_mkt_20d"] = rets.rolling(20).corr(mkt_ret)
C["beta_dn_20d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 20, min_obs=10)
C["beta_up_20d"] = rolling_beta(rets, mkt_ret.clip(lower=0), 20, min_obs=10)
C["beta_asym_20d"] = C["beta_dn_20d"] - C["beta_up_20d"]

# F. time-series z-score momentum (acceleration vs own history)
C["mom20_tsz_250"] = (mom20 - mom20_ts_mean) / (mom20_ts_std + 1e-9)

# G. price position in 250d range
C["range_pos_250d"] = (closes - roll_min250) / (roll_max250 - roll_min250 + 1e-9)

# H. regime-conditional momentum
C["mom20_lowdisp"] = mom20.where(disp20 <= disp_med60, 0.0)
C["mom20_calmvix"] = mom20.where(vix <= vix_med120, 0.0)

# I. volume confirmation
C["up_vol_ratio_20d"] = (up_vol + 1e-9) / (dn_vol + 1e-9)
C["up_vol_per_day_20d"] = (up_vol / (up_cnt + 1e-9)) / (dn_vol / (dn_cnt + 1e-9) + 1e-9)

# J. momentum variants with skip / longer window
C["mom40_skip5_vol20"] = (closes.shift(5) / closes.shift(45) - 1.0) / (roll_std20 + 1e-9)
C["mom20_skip2"] = closes.shift(2) / closes.shift(22) - 1.0

fwd10 = forward_returns(closes, H)
rows = []
for name, panel in C.items():
    ics = rank_ic_series(panel, fwd10)
    if len(ics) < 100:
        print(f"{name:24s} SKIP (n_ic={len(ics)})", flush=True)
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
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={len(ics):4d} "
          f"cov={cov['coverage_asset_days']:.2f} ge8={cov['coverage_dates_ge8']:.2f} tov={tov} "
          f"decay={dec} libcorr={corr}({ck}) r500_ic={ic_r500} r500_icir={icir_r500} GATE={gate}", flush=True)

print("\n=== summary sorted by |ICIR| ===", flush=True)
rows.sort(key=lambda r: -abs(r[2]))
for r in rows:
    print(f"{r[0]:24s} IC={r[1]:+.4f} ICIR={r[2]:+.4f} n={r[4]:4d} cov={r[5]:.2f} ge8={r[6]:.2f} "
          f"r500_ic={r[11]} r500_icir={r[12]} libcorr={r[9]}({r[10]}) GATE={r[13]}", flush=True)
print(f"elapsed {time.time()-t0:.1f}s", flush=True)
