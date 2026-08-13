"""miner_1 (2033-09-05): batch AG v3 - vectorized evaluation of cross-asset factor candidates.

Same candidates as batchAG_v2, but the IC engine is fully vectorized:
  - rank_ic_fast: Spearman IC = Pearson corr of cross-sectional ranks, computed
    per-date over the whole panel via DataFrame.rank(axis=1) and row-wise algebra
    (no per-date python loop, no pandas rolling().apply()).
  - decay_profile_fast: same horizons (1,2,3,5,10,20) using the fast IC.

Universe: 15 tradable cross-asset instruments; macro (DXY, USDCNY, USDJPY,
EURUSD, VIX) observation-only. Data visible through previous completed trading day.

Admission gates (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import sys, time, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (
    load_panels, close_panel, forward_returns, coverage_metrics, turnover_rank,
    max_library_corr, library_signals,
)

t_start = time.time()

def log(msg):
    print(f"[{time.time()-t_start:6.1f}s] {msg}", flush=True)

panels = load_panels(days=3200)
closes = close_panel(panels)
rets = closes.pct_change()
log(f"closes shape={closes.shape} range={closes.index.min().date()} -> {closes.index.max().date()} assets={closes.shape[1]}")

lib_sig = library_signals(panels, closes, rets)
# add the 3 currently-effective library factors (as stored in factor_ensemble.json)
lib_sig["vol_adj_mom_accel_20x60"] = ((closes / closes.shift(20) - 1.0) - (closes / closes.shift(60) - 1.0)) / rets.rolling(20).std()
dn_mask = (rets < 0).astype(float)
mr_dn = (rets * dn_mask).sum(axis=1)
lib_sig["dn_mkt_beta_60d"] = (rets.where(dn_mask > 0)).rolling(60, min_periods=40).cov(mr_dn) / mr_dn.rolling(60, min_periods=40).var()
cn10y_ret = closes["CN10Y"].pct_change()
cn_beta = {a: (rets[a].rolling(60, min_periods=40).cov(cn10y_ret) / cn10y_ret.rolling(60, min_periods=40).var()) for a in rets.columns}
lib_sig["rate_beta_cn10y_60d"] = pd.DataFrame(cn_beta, index=rets.index)
log(f"lib_sig keys={list(lib_sig.keys())}")

H = 10
fwd = forward_returns(closes, H)


def clean(fp):
    return fp.replace([np.inf, -np.inf], np.nan)


def rolling_beta(a, b, win, min_obs):
    cov = a.rolling(win, min_periods=min_obs).cov(b)
    var = b.rolling(win, min_periods=min_obs).var()
    return (cov / var).where(b.rolling(win, min_periods=min_obs).count() >= min_obs)


def rank_ic_fast(factor_panel, fwd_panel, min_valid=8):
    """Vectorized daily Spearman IC between factor and forward-return cross-sections."""
    f = factor_panel.rank(axis=1)
    r = fwd_panel.rank(axis=1)
    m = f.notna() & r.notna()
    n = m.sum(axis=1)
    f = f.where(m)
    r = r.where(m)
    fmean = f.mean(axis=1)
    rmean = r.mean(axis=1)
    fc = f.sub(fmean, axis=0)
    rc = r.sub(rmean, axis=0)
    num = (fc * rc).mean(axis=1)
    den = fc.pow(2).mean(axis=1).pow(0.5) * rc.pow(2).mean(axis=1).pow(0.5)
    ic = num / den
    ic = ic.where(n >= min_valid)
    ok = (n >= min_valid) & (den.abs() > 1e-14) & ic.notna()
    return ic[ok].dropna()


def decay_profile_fast(factor_panel, closes_p, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    out = {}
    for h in horizons:
        fh = forward_returns(closes_p, h)
        ics = rank_ic_fast(factor_panel, fh, min_valid)
        if len(ics):
            out[str(h)] = round(float(ics.mean()), 4)
    return out


def summarize_ic(ic_series, expected_sign=1):
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series * expected_sign > 0).mean())
    return {"ic": ic, "icir": icir, "ic_hit_ratio": hit, "n_ic_dates": int(len(ic_series))}


# ---------------- sanity check fast IC vs reference loop ----------------
def rank_ic_ref(factor_panel, fwd_panel, min_valid=8):
    dates, ics = [], []
    for dt in factor_panel.index:
        pair = pd.concat([factor_panel.loc[dt].rename("f"), fwd_panel.loc[dt].rename("r")], axis=1).dropna()
        if len(pair) < min_valid or pair["r"].std() < 1e-14 or pair["f"].std() < 1e-14:
            continue
        ic = pair["f"].corr(pair["r"], method="spearman")
        if not math.isnan(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

smoke = closes.pct_change().iloc[-250:]
smoke_fwd = forward_returns(closes, 5).iloc[-250:]
a_fast = rank_ic_fast(smoke, smoke_fwd)
a_ref = rank_ic_ref(smoke, smoke_fwd)
aligned = pd.concat([a_fast, a_ref], axis=1, join="inner").dropna()
maxdiff = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().max())
log(f"SMOKE TEST fast-vs-ref: n={len(aligned)} max_abs_diff={maxdiff:.6f} {'OK' if maxdiff < 1e-8 else 'MISMATCH'}")

# ---------------- candidate construction ----------------
open_p = panels  # panels contain open/high/low/close/volume
hi = closes.copy()
lo = closes.copy()
vol_panel = closes.copy()
for a in closes.columns:
    hi[a] = panels[a]["high"]
    lo[a] = panels[a]["low"]
    vol_panel[a] = panels[a].get("volume", np.nan) if "volume" in panels[a].columns else np.nan

cands = {}
# A. gap ratio 20d
gaps = (open_p  # placeholder replaced below
        )
# proper gap: open/prev_close - 1
gap_df = pd.DataFrame({a: panels[a]["open"] / panels[a]["close"].shift(1) - 1.0 for a in closes.columns}, index=closes.index)
cands["gap_ratio_20d"] = clean(gap_df.abs().rolling(20, min_periods=15).mean() / rets.rolling(20, min_periods=15).std())
# B. close range position 20d
rng_hi = hi.rolling(20, min_periods=15).max()
rng_lo = lo.rolling(20, min_periods=15).min()
cands["close_range_pos_20d"] = clean((closes - rng_lo) / (rng_hi - rng_lo))
# C. lag-1 autocorrelation 20d (vectorized rolling cov)
xs = rets.shift(1)
cov1 = rets.rolling(20, min_periods=15).cov(xs)
var1 = rets.rolling(20, min_periods=15).var()
cands["autocorr_ret_20d"] = clean(cov1 / var1)
# D. relative strength 20d
m20 = closes / closes.shift(20) - 1.0
cands["rel_strength_20d"] = clean(m20 - m20.mean(axis=1))
# E. DXY beta 60d
dxy_ret = panels["DXY"]["close"].pct_change()
cands["dxy_beta_60d"] = clean(rolling_beta(rets, dxy_ret, 60, 40))
# F. USDJPY beta 60d
usdjpy_ret = panels["USDJPY"]["close"].pct_change()
cands["usdjpy_beta_60d"] = clean(rolling_beta(rets, usdjpy_ret, 60, 40))
# G. trend R2 20d (signed) - vectorized via rolling corr(close, time)^2 * sign(slope)
t_idx = np.arange(len(closes))
time_panel = pd.DataFrame(np.tile(t_idx[:, None], (1, closes.shape[1])), index=closes.index, columns=closes.columns)
rho = closes.rolling(20, min_periods=15).corr(time_panel)
slope_sign = np.sign(closes.diff(10))
cands["trend_r2_20d"] = clean((rho ** 2) * slope_sign)
# H. updown vol spread 20d
up = rets.where(rets > 0, 0.0)
dn = rets.where(rets < 0, 0.0)
up_vol = up.rolling(20, min_periods=10).std()
dn_vol = dn.rolling(20, min_periods=10).std()
cands["updown_vol_spread_20d"] = clean((up_vol - dn_vol) / rets.rolling(20, min_periods=15).std())
# I. candle body position 20d
body = ((closes - gap_df * 0)  # placeholder
        )
body = pd.DataFrame({a: ((panels[a]["close"] - panels[a]["open"]) / (panels[a]["high"] - panels[a]["low"])) for a in closes.columns}, index=closes.index)
cands["candle_body_pos_20d"] = clean(body.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean())
# J. VIX-regime conditional momentum 20d
vix = panels["VIX"]["close"].astype(float)
vix_med60 = vix.rolling(60, min_periods=40).median()
calm = (vix < vix_med60).astype(float)
cands["vix_regime_mom20"] = clean(m20.mul(calm, axis=0))
# K. short risk-adjusted momentum 5/20
m5 = closes / closes.shift(5) - 1.0
cands["mom5_vol20"] = clean(m5 / rets.rolling(20, min_periods=15).std())
# L. BTC beta 60d
btc_ret = closes["BTC"].pct_change()
cands["btc_beta_60d"] = clean(rolling_beta(rets, btc_ret, 60, 40))
# M. volume-trend correlation 20d
cands["volume_trend_corr_20d"] = clean(vol_panel.rolling(20, min_periods=15).corr(closes))
log(f"candidates computed: {list(cands.keys())}")

# ---------------- library drift re-validation (h=10) ----------------
print("\n=== LIBRARY FACTORS (drift re-validation, h=10) ===", flush=True)
for name, fp in lib_sig.items():
    ics = rank_ic_fast(fp, fwd, min_valid=8)
    if len(ics) < 30:
        print(f"{name:26s} INSUFFICIENT dates ({len(ics)})", flush=True)
        continue
    m = summarize_ic(ics)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={m['n_ic_dates']} "
          f"cov_days={cov['coverage_asset_days']:.2f} cov8={cov['coverage_dates_ge8']:.2f} turn={turn}", flush=True)

# ---------------- candidate evaluation ----------------
print("\n=== CANDIDATE FACTORS (h=10, gates |IC|>=0.0070, |ICIR|>=0.0840) ===", flush=True)
rows = []
for name, fp in cands.items():
    ics = rank_ic_fast(fp, fwd, min_valid=8)
    if len(ics) < 30:
        print(f"{name:26s} INSUFFICIENT dates ({len(ics)})", flush=True)
        continue
    m = summarize_ic(ics)
    m.update(coverage_metrics(fp, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(fp, 10)
    m["decay_ic_by_horizon"] = decay_profile_fast(fp, closes, (1, 2, 3, 5, 10, 20), 8)
    corr, key = max_library_corr(fp, lib_sig)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    yr = ics.groupby(ics.index.year).mean()
    yr_str = " ".join(f"{y}:{v:+.3f}" for y, v in yr.items())
    gate = "PASS" if (abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840) else "FAIL"
    rows.append((name, m, gate))
    dec = m["decay_ic_by_horizon"]
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} n={m['n_ic_dates']} "
          f"cov_days={m['coverage_asset_days']:.2f} cov8={m['coverage_dates_ge8']:.2f} turn={m['turnover_10d_rank']} "
          f"maxcorr={corr:.3f}({key}) decay1/10/20={dec.get('1')}/{dec.get('10')}/{dec.get('20')} => {gate}", flush=True)
    print(f"{'':26s} yearly_IC: {yr_str}", flush=True)

print("\n=== PASS SUMMARY ===", flush=True)
for name, m, gate in rows:
    if gate == "PASS":
        print(f"{name} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} maxcorr={m['max_abs_library_correlation']:.3f} turn={m['turnover_10d_rank']}", flush=True)
log("DONE")
