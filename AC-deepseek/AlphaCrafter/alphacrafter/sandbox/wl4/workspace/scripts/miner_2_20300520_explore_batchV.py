"""miner_2 batch exploration (2030-05-20) - candidate factor screen batch V.

Theme: trend efficiency (Kaufman), OHLC volatility structure (Parkinson/GK vs
close-close), volatility-regime shift speed, up/down vol asymmetry, rate/dollar/
risk-off conditional momentum, rotation betas, close-location value, drawdown
z-score. Data through previous completed trading day (2030-05-17).

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
roll_std20 = rets.rolling(20).std()
roll_std60 = rets.rolling(60).std()

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win, min_periods=min_obs).cov(z["x"])
        var = z["x"].rolling(win, min_periods=min_obs).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

# ---------------- library reference signals for correlation audit ----------------
lib = {}
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y.pct_change(), 60)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["vol_adj_mom_accel_20x60"] = ((closes/closes.shift(20)-1) - (closes/closes.shift(60)-1)) / roll_std20
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["vol_of_vol20x60"] = roll_std20.rolling(60).std()
lib["vix_beta_cond_60x20"] = -rolling_beta(rets, vix.pct_change(), 60) * (vix / vix.shift(20) - 1.0)
lib["vol_price_corr_20"] = rets.rolling(20).corr((closes/closes.shift(20)-1).abs())
lib["us10y_cond_beta_60d"] = rolling_beta(rets, us10y.pct_change(), 60) * np.sign(us10y.pct_change(20))
lib["eurusd_beta_60d"] = rolling_beta(rets, align(panels["EURUSD"]["close"].astype(float), closes.index).pct_change(), 60)
lib["usdcny_beta_60d"] = rolling_beta(rets, align(panels["USDCNY"]["close"].astype(float), closes.index).pct_change(), 60)
lib["rsi_14"] = closes / closes.rolling(14).mean()

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

# ---------------- OHLC panels ----------------
hi, lo, op = {}, {}, {}
for a in TRADABLE:
    df = panels[a]
    hi[a] = df["high"].astype(float)
    lo[a] = df["low"].astype(float)
    op[a] = df["open"].astype(float)
hi = pd.DataFrame(hi, index=closes.index)
lo = pd.DataFrame(lo, index=closes.index)
op = pd.DataFrame(op, index=closes.index)

log_hl = np.log(hi / lo)
log_co = np.log(closes / op)
parkinson = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2)))
gk = np.sqrt((0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).rolling(20).mean())
cc_vol20 = roll_std20

# ---------------- candidates ----------------
C = {}
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0

# A. trend efficiency (Kaufman)
C["eff_ratio_60d"] = (closes / closes.shift(60) - 1.0).abs() / (rets.abs().rolling(60).sum() + 1e-9)
C["eff_ratio_20d"] = (closes / closes.shift(20) - 1.0).abs() / (rets.abs().rolling(20).sum() + 1e-9)

# B. OHLC vol structure
C["parkinson_cc_ratio_20d"] = parkinson / (cc_vol20 + 1e-9)   # >1 => intraday range larger than close moves
C["gk_cc_ratio_20d"] = gk / (cc_vol20 + 1e-9)
C["parkinson_z_120d"] = (parkinson - parkinson.rolling(120).mean()) / (parkinson.rolling(120).std() + 1e-9)

# C. vol-regime shift speed & asymmetry
C["vol_regime_shift_5d"] = (roll_std20 / (roll_std60 + 1e-9)) / ((roll_std20.shift(5) / (roll_std60.shift(5) + 1e-9))) - 1.0
up_ret = rets.where(rets > 0, 0.0)
dn_ret = rets.where(rets < 0, 0.0)
up_vol60 = np.sqrt((up_ret ** 2).rolling(60).mean())
dn_vol60 = np.sqrt((dn_ret ** 2).rolling(60).mean())
C["updown_vol_asym_60d"] = up_vol60 / (dn_vol60 + 1e-9)

# D. conditional momentum on macro regime
C["cn10y_cond_mom20"] = mom20 * np.sign(cn10y.pct_change(20))
C["btc_cond_mom20"] = mom20 * np.sign(closes["BTC"].pct_change(20))
spread = us10y - cn10y
C["ycspread_cond_mom20"] = mom20 * np.sign(spread.pct_change(20))

# E. risk-off composite beta
riskoff_ret = 0.5 * vix.pct_change() + 0.5 * dxy.pct_change()
C["riskoff_beta_60d"] = rolling_beta(rets, riskoff_ret, 60)
C["eq_gold_beta_60d"] = rolling_beta(rets, (closes["SPX"] / closes["XAU"]).pct_change(), 60)
C["wti_xau_beta_60d"] = rolling_beta(rets, (closes["WTI"] / closes["XAU"]).pct_change(), 60)
C["eth_beta_60d"] = rolling_beta(rets, closes["ETH"].pct_change(), 60)

# F. drawdown risk-adjusted & close-location
roll_max60 = closes.rolling(60).max()
dd60 = closes / roll_max60 - 1.0
C["drawdown_z_60d"] = dd60 / (roll_std60 + 1e-9)
roll_min20 = closes.rolling(20).min()
roll_max20 = closes.rolling(20).max()
C["range_pos_20d_vol"] = ((closes - roll_min20) / (roll_max20 - roll_min20 + 1e-9)) / (roll_std20 + 1e-9)
C["close_loc_20d"] = (closes - lo) / (hi - lo + 1e-9)          # avg close location in daily range
C["max_loss_20d"] = rets.rolling(20).min()

# G. breadth momentum & skew of aggregated returns
up_ratio20 = (rets > 0).rolling(20).mean()
C["up_ratio_chg_10d"] = up_ratio20 - up_ratio20.shift(10)
r5 = closes / closes.shift(5) - 1.0
C["skew5_20d"] = r5.rolling(20).skew()

# H. composite time-series-z momentum blend (mean of z-scored mom10/20/60)
def tsz(p, win=250):
    m = p.rolling(win).mean()
    s = p.rolling(win).std()
    return (p - m) / (s + 1e-9)
z10 = tsz(mom20.rolling(10).mean())
z20 = tsz(mom20)
z60 = tsz(mom60)
C["mom_blend_tsz"] = (z10 + z20 + z60) / 3.0

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
