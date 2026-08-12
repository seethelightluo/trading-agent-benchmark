"""miner_2 batch exploration (2030-03-25) - candidate factor screen batch T.

Theme: trend-quality / efficiency, candle geometry, liquidity participation,
and conditional cross-asset betas on the 15-instrument tradable universe.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (paper, daily rank IC).
Data through previous completed trading day (2030-03-22).
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
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
us10y = align(panels["US10Y"]["close"].astype(float), closes.index)
cn10y = align(panels["CN10Y"]["close"].astype(float), closes.index)

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
# library reference signals (the 3 ensemble factors + previously screened ones) for max corr
lib = {}
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y.pct_change(), 60)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vix_beta_cond_60x20"] = -rolling_beta(rets, vix.pct_change(), 60) * (vix / vix.shift(20) - 1.0)
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["recovery_60d"] = closes / closes.rolling(60).max()
lib["skew_60d"] = rets.rolling(60).skew()

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

# ---------------- data structures for OHLC ---------------- 
open_p = pd.DataFrame({a: panels[a]["open"].astype(float) for a in TRADABLE}, index=closes.index).sort_index()
high_p = pd.DataFrame({a: panels[a]["high"].astype(float) for a in TRADABLE}, index=closes.index).sort_index()
low_p = pd.DataFrame({a: panels[a]["low"].astype(float) for a in TRADABLE}, index=closes.index).sort_index()
vol_p = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in TRADABLE}, index=closes.index).sort_index()
hl = (high_p - low_p) / closes
body = (closes - open_p).abs() / (high_p - low_p).replace(0, np.nan)
upper_sh = (high_p - pd.concat([open_p, closes], axis=1).max(axis=1)) / (high_p - low_p).replace(0, np.nan)
lower_sh = (pd.concat([open_p, closes], axis=1).min(axis=1) - low_p) / (high_p - low_p).replace(0, np.nan)
close_pos = (closes - low_p) / (high_p - low_p).replace(0, np.nan)

roll_std5 = rets.rolling(5).std()
roll_std20 = rets.rolling(20).std()
roll_std60 = rets.rolling(60).std()
abs_ret = rets.abs()

# ---------------- candidates ----------------
C = {}
# 1) trend quality / efficiency
C["efficiency_ratio_20d"] = (closes - closes.shift(20)).abs() / abs_ret.rolling(20).sum()
C["efficiency_ratio_60d"] = (closes - closes.shift(60)).abs() / abs_ret.rolling(60).sum()
# R^2 of log-price linear trend over 60d
def r2_trend(px, win=60):
    lpx = np.log(px)
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    x = np.arange(win, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    for col in px.columns:
        y = lpx[col].values
        n = len(y)
        r2s = np.full(n, np.nan)
        for i in range(win - 1, n):
            w = y[i - win + 1:i + 1]
            if np.isfinite(w).sum() < win * 0.7:
                continue
            ym = np.nanmean(w)
            syy = np.nansum((w - ym) ** 2)
            if syy < 1e-15:
                continue
            sxy = np.nansum((x - xm) * (w - ym))
            r2s[i] = (sxy ** 2) / (sxx * syy)
        out[col] = r2s
    return out
C["r2_trend_60d"] = r2_trend(closes, 60)
C["trend_consistency_20d"] = (np.sign(rets) == np.sign(closes / closes.shift(20) - 1)).rolling(20).mean()

# 2) volatility term structure / vol regime
C["vol_term_5_60"] = roll_std5 / roll_std60
C["vol_change_20_60"] = roll_std20 / roll_std60 - 1.0
C["downside_vol_ratio_20d"] = np.sqrt((rets.clip(upper=0) ** 2).rolling(20).mean()) / (roll_std20 + 1e-9)

# 3) higher moments (short window)
C["skew_20d"] = rets.rolling(20).skew()
C["kurt_20d"] = rets.rolling(20).kurt()
C["max_gain_20d"] = rets.rolling(20).max()
C["max_loss_20d"] = rets.rolling(20).min()
C["win_loss_ratio_20d"] = rets.clip(lower=0).rolling(20).mean() / (rets.clip(upper=0).abs().rolling(20).mean() + 1e-9)

# 4) liquidity / participation (volume based)
C["amihud_20d"] = (abs_ret / (vol_p + 1e-9)).rolling(20).mean()
C["volume_trend_20d"] = vol_p.rolling(20).mean() / vol_p.rolling(60).mean() - 1.0
C["volume_z_60d"] = (vol_p - vol_p.rolling(60).mean()) / (vol_p.rolling(60).std() + 1e-9)

# 5) candle geometry
C["body_ratio_20d"] = body.rolling(20).mean()
C["upper_shadow_20d"] = upper_sh.rolling(20).mean()
C["lower_shadow_20d"] = lower_sh.rolling(20).mean()
C["close_pos_20d"] = close_pos.rolling(20).mean()
C["range_compression_20d"] = -(hl.rolling(20).mean() / hl.rolling(60).mean() - 1.0)  # tight range -> expansion

# 6) conditional cross-asset betas
C["wti_beta_60d"] = rolling_beta(rets, closes["WTI"].pct_change(), 60)
C["copper_beta_60d"] = rolling_beta(rets, closes["COPPER"].pct_change(), 60)
C["us10y_beta_60d"] = rolling_beta(rets, us10y.pct_change(), 60)
C["spx_beta_60d"] = rolling_beta(rets, closes["SPX"].pct_change(), 60)
C["xau_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60)
dxy_ret = dxy.pct_change()
C["dxy_up_beta_60d"] = rolling_beta(rets, dxy_ret.clip(lower=0), 60)
C["dxy_dn_beta_60d"] = rolling_beta(rets, dxy_ret.clip(upper=0), 60)
C["dxy_beta_asym_60d"] = rolling_beta(rets, dxy_ret.clip(upper=0), 60) - rolling_beta(rets, dxy_ret.clip(lower=0), 60)

# 7) conditional momentum on macro regimes
us10y_up = np.sign(us10y.pct_change(20))
dxy_dn = np.sign(-dxy.pct_change(20))
vix_dn = np.sign(-vix.pct_change(20))
C["mom20_cond_us10y_up"] = (closes / closes.shift(20) - 1.0).mul(us10y_up, axis=0)
C["mom20_cond_dxy_dn"] = (closes / closes.shift(20) - 1.0).mul(dxy_dn, axis=0)
C["mom20_cond_vix_dn"] = (closes / closes.shift(20) - 1.0).mul(vix_dn, axis=0)

# 8) momentum variants
C["mom_accel_5x20_vol60"] = ((closes / closes.shift(5) - 1.0) - (closes / closes.shift(20) - 1.0)) / roll_std60
C["mom60_vol20"] = (closes / closes.shift(60) - 1.0) / roll_std20
C["autocorr1_20d"] = rets.rolling(20).corr(rets.shift(1))

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
