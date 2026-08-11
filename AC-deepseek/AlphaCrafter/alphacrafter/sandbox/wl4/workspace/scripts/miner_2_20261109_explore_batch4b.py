"""miner_2 2026-11-09: explore fresh factor families (batch4b, updated window) (batch 4 - trend/relative-strength,
liquidity, lead-lag beta, regression R2/idio, convexity, overnight-intraday split).
Vectorized NaN-safe rank-IC. Validation window 2020-01-01..2026-11-06 (last completed day).

Candidates (direction exploratory, sign decided by data):
  A sma_trend_20_60      : SMA20/SMA60 - 1  (trend following)                      [+1]
  B rel_strength_20d     : 20d ret - cross-section median 20d ret (rel strength)   [+1]
  C amihud_illiq_20d     : -mean(|ret|/vol, 20)  (liquidity premium)               [+1]
  D boll_bandwidth_20d   : 4*std20/SMA20  (vol squeeze)                            [-1]
  E leadlag_beta_spx_5d  : rolling 60d beta of asset ret on SPX ret lagged 5d      [+/-]
  F r2_spx_60d           : R^2 of asset ret ~ SPX ret over 60d                     [+/-]
  G idio_vol_60d         : residual std from SPX regression 60d                    [-1]
  H upside_downside_60d  : mean(pos ret)/|mean(neg ret)| over 60d (convexity)      [+1]
  I mom_accel_20_60      : mom20 - mom60  (momentum acceleration)                  [+1]
  J vol_trend_5_60       : SMA(vol,5)/SMA(vol,60)  (volume flow)                   [-1]
  K beta_us10y_60d       : rolling 60d beta of asset ret to US10Y ret              [+/-]
  L spread_beta_uscn_60d : beta to (US10Y ret - CN10Y ret) 60d                     [+/-]
  M ts_momentum_20d      : close/SMA20 - 1                                         [+1]
  N ewma_vol_inv_20d     : -EWMA(span=20) vol of ret                               [+1]
  O dist_250d_high       : close/rolling_max(close,250) - 1  (52w high distance)   [-1]
  P gap_20d              : mean(open/prev_close - 1, 20d)  (overnight gap)         [+/-]
  Q intraday_ret_20d     : mean((close-open)/open, 20d)  (intraday component)      [+/-]
  R coskew_60d           : coskewness with market ret (60d)                        [+/-]

Gate (h=10): |IC|>=0.0070, |ICIR|>=0.0840, max_abs_library_correlation < 0.5.
Also re-validates the 4 currently-effective library factors (report only, no rewrite).
"""
from __future__ import annotations
import sys, json, base64, zlib, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, TRADABLE,
                                 coverage_metrics, turnover_rank)

HORIZON = 10
MIN_VALID = 8
END = pd.Timestamp("2026-11-06")
WINDOW = (pd.Timestamp("2020-01-01"), END)
HORIZONS = (1, 2, 3, 5, 10, 20)
SUB_PERIODS = {"full": (pd.Timestamp("2020-01-01"), END),
               "p23": (pd.Timestamp("2023-01-01"), END),
               "p25": (pd.Timestamp("2025-01-01"), END),
               "online": (pd.Timestamp("2026-07-16"), END)}


def rank_ic_vec(F: pd.DataFrame, R: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    common = F.index.intersection(R.index)
    Fr = F.loc[common].rank(axis=1, method="average")
    Rr = R.loc[common].rank(axis=1, method="average")
    X = Fr.values.astype(float)
    Y = Rr.values.astype(float)
    valid = ~(np.isnan(X) | np.isnan(Y))
    n = valid.sum(axis=1)
    keep = n >= min_valid
    X, Y, V, N = X[keep], Y[keep], valid[keep], n[keep]
    Xv = np.where(V, X, np.nan)
    Yv = np.where(V, Y, np.nan)
    Xc = X - np.nanmean(Xv, axis=1, keepdims=True)
    Yc = Y - np.nanmean(Yv, axis=1, keepdims=True)
    Xc = np.where(V, Xc, 0.0)
    Yc = np.where(V, Yc, 0.0)
    xy = (Xc * Yc).sum(axis=1)
    xx = (Xc * Xc).sum(axis=1)
    yy = (Yc * Yc).sum(axis=1)
    denom = np.sqrt(xx * yy)
    ok = (xx > 1e-14) & (yy > 1e-14) & (denom > 0)
    ic = np.full(len(X), np.nan)
    ic[ok] = xy[ok] / denom[ok]
    return pd.Series(ic, index=Fr.index[keep], name="ic").dropna()


def summarize_ic(ic_series: pd.Series, expected_sign: int = 1):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean()) if expected_sign else float((np.sign(ic_series) != 0).mean())
    return {"ic": round(ic, 4), "icir": round(icir, 4),
            "ic_hit_ratio": round(hit, 3), "n_ic_dates": int(len(ic_series)),
            "ic_std": round(sd, 4)}


def rolling_beta(asset_ret: pd.Series, driver_ret: pd.Series, win: int, min_obs: int = 40):
    z = pd.concat([asset_ret.rename("a"), driver_ret.rename("m")], axis=1).dropna()
    cov = z["a"].rolling(win).cov(z["m"])
    var = z["m"].rolling(win).var()
    return (cov / var).where(z["m"].rolling(win).count() >= min_obs)


t0 = time.time()
panels = load_panels(3000)
closes_all = close_panel(panels)
rets_all = closes_all.pct_change()
print(f"loaded panels {time.time()-t0:.1f}s | closes {closes_all.shape} | last {closes_all.index[-1]}")

clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}


def asset_wide(func):
    out = {}
    for a, s in clean.items():
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes_all.index)


cand = {}
# A sma trend 20/60
cand["sma_trend_20_60"] = asset_wide(lambda s: s.rolling(20).mean() / s.rolling(60).mean() - 1.0)
# B relative strength vs cross-section median 20d ret
mom20_cs = (closes_all.shift(2) / closes_all.shift(22) - 1.0)
cand["rel_strength_20d"] = (mom20_cs - mom20_cs.median(axis=1)).reindex(closes_all.index)
# C amihud illiquidity 20d (negated: illiquid -> higher fwd ret expected [+1])
amihud = {}
for a in TRADABLE:
    if a not in clean:
        continue
    v = panels[a]["volume"].astype(float).dropna() if "volume" in panels[a] else None
    if v is None:
        continue
    r = clean[a].pct_change()
    z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
    amihud[a] = -(z["r"].abs() / z["v"]).rolling(20).mean()
cand["amihud_illiq_20d"] = pd.DataFrame(amihud).reindex(closes_all.index)
# D bollinger bandwidth 20d
cand["boll_bandwidth_20d"] = asset_wide(lambda s: 4.0 * s.rolling(20).std() / s.rolling(20).mean())
# E lead-lag beta: asset ret on SPX ret lagged 5d
spx_ret = clean["SPX"].pct_change()
spx_lag5 = spx_ret.shift(5)
cand["leadlag_beta_spx_5d"] = asset_wide(lambda s: rolling_beta(s.pct_change(), spx_lag5, 60))
# F R2 of asset ~ SPX over 60d
def r2_spx(s):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), spx_ret.rename("m")], axis=1).dropna()
    corr = z["a"].rolling(60).corr(z["m"])
    return corr ** 2
cand["r2_spx_60d"] = asset_wide(r2_spx)
# G idio vol 60d (residual std from SPX regression) negated
def idio_vol(s):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), spx_ret.rename("m")], axis=1).dropna()
    b = rolling_beta(r, spx_ret, 60)
    resid = z["a"] - b * z["m"]
    return resid.rolling(60).std()
cand["idio_vol_60d"] = -asset_wide(idio_vol)
# H upside/downside convexity 60d
def ud_ratio(s):
    r = s.pct_change()
    pos = r.where(r > 0, np.nan).rolling(60).mean()
    neg = r.where(r < 0, np.nan).rolling(60).mean()
    return pos / neg.abs()
cand["upside_downside_60d"] = asset_wide(ud_ratio)
# I momentum acceleration mom20 - mom60
mom60 = closes_all.shift(5) / closes_all.shift(65) - 1.0
cand["mom_accel_20_60"] = mom20_cs - mom60
# J volume trend 5/60
vt = {}
for a in TRADABLE:
    if a not in clean:
        continue
    v = panels[a]["volume"].astype(float).dropna() if "volume" in panels[a] else None
    if v is None:
        continue
    vt[a] = v.rolling(5).mean() / v.rolling(60).mean()
cand["vol_trend_5_60"] = pd.DataFrame(vt).reindex(closes_all.index)
# K beta to US10Y ret
us10y_ret = clean["US10Y"].pct_change()
cand["beta_us10y_60d"] = asset_wide(lambda s: rolling_beta(s.pct_change(), us10y_ret, 60))
# L beta to (US10Y - CN10Y) spread ret
cn10y_ret = clean["CN10Y"].pct_change()
spread_ret = (us10y_ret - cn10y_ret).dropna()
cand["spread_beta_uscn_60d"] = asset_wide(lambda s: rolling_beta(s.pct_change(), spread_ret, 60))
# M time-series momentum 20d (close vs SMA20)
cand["ts_momentum_20d"] = asset_wide(lambda s: s / s.rolling(20).mean() - 1.0)
# N EWMA vol inverse
cand["ewma_vol_inv_20d"] = asset_wide(lambda s: -s.pct_change().ewm(span=20, adjust=False).std())
# O distance from 250d high
cand["dist_250d_high"] = asset_wide(lambda s: s / s.rolling(250).max() - 1.0)
# P overnight gap 20d
gap = {}
for a in TRADABLE:
    if a not in clean:
        continue
    o = panels[a]["open"].astype(float).dropna()
    c = clean[a]
    idx = c.index.intersection(o.index)
    g = o.loc[idx] / c.loc[idx].shift(1) - 1.0
    gap[a] = g.rolling(20).mean()
cand["gap_20d"] = pd.DataFrame(gap).reindex(closes_all.index)
# Q intraday ret 20d
intra = {}
for a in TRADABLE:
    if a not in clean:
        continue
    o = panels[a]["open"].astype(float).dropna()
    c = clean[a]
    idx = c.index.intersection(o.index)
    ir = c.loc[idx] / o.loc[idx] - 1.0
    intra[a] = ir.rolling(20).mean()
cand["intraday_ret_20d"] = pd.DataFrame(intra).reindex(closes_all.index)
# R coskewness with market 60d
mkt_ret = rets_all.mean(axis=1)
def coskew(s):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), mkt_ret.rename("m")], axis=1).dropna()
    e = z - z.mean()
    num = (e["a"] * e["m"] ** 2).rolling(60).mean()
    den = e["a"].rolling(60).std() * e["m"].rolling(60).std() ** 2
    return num / den
cand["coskew_60d"] = asset_wide(coskew)

print(f"candidates built {time.time()-t0:.1f}s | n={len(cand)}")
print("candidate list:", list(cand.keys()))

idx = (closes_all.index >= WINDOW[0]) & (closes_all.index <= WINDOW[1])
closes = closes_all.loc[idx]
cand = {k: v.loc[idx] for k, v in cand.items()}

fwd_by_h = {h: closes.shift(-h) / closes - 1.0 for h in HORIZONS}

# library signals from persisted JSON signal artifacts (both formats)
lib = {}
for p in sorted(Path("factors").glob("*.json")):
    if p.name == "factor_ensemble.json":
        continue
    try:
        d = json.loads(p.read_text())
        sa = d.get("validation", {}).get("signal_artifact")
        if not sa:
            continue
        fmt = sa.get("format")
        if fmt == "base64:zlib:csv":
            raw = zlib.decompress(base64.b64decode(sa["data"]))
            df = pd.read_csv(io.BytesIO(raw), index_col=0)
            df.index = pd.to_datetime(df.index)
        elif fmt == "panel_json_v1":
            df = pd.DataFrame(sa["values"], index=pd.to_datetime(sa["dates"]), columns=sa["assets"])
        else:
            continue
        lib[d["factor_id"]] = df.reindex(closes_all.index[idx])
    except Exception as e:
        print(f"skip lib {p.name}: {e}")
print(f"library factors loaded: {list(lib.keys())}  ({time.time()-t0:.1f}s)")


def max_lib_corr(cand_df, lib):
    best, best_key = 0.0, None
    cstack = cand_df.stack().rename("cand")
    for name, lib_df in lib.items():
        both = pd.concat([cstack, lib_df.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key


rows = []
for name, panel in cand.items():
    panel = panel.reindex(closes.index)
    ics = rank_ic_vec(panel, fwd_by_h[HORIZON], MIN_VALID)
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {str(h): round(float(rank_ic_vec(panel, fwd_by_h[h], MIN_VALID).mean()), 4)
                                for h in HORIZONS}
    sub = {}
    for sname, (s0, s1) in SUB_PERIODS.items():
        sub_ics = rank_ic_vec(panel.loc[s0:s1], fwd_by_h[HORIZON].loc[s0:s1], MIN_VALID)
        sub[sname] = round(float(sub_ics.mean()), 4) if len(sub_ics) else None
    m["subperiod_ic"] = sub
    corr, key = max_lib_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    pi = abs(m["ic"]) >= 0.007
    pir = abs(m["icir"]) >= 0.084
    pc = corr < 0.5
    rows.append((name, m, pi, pir, pc))
    print(f"\n=== {name} === IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} "
          f"cov_asset={m['coverage_asset_days']:.3f} cov_ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']} "
          f"decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} } sub={sub} lib_corr={corr:.3f}({key}) "
          f"GATES IC={pi} ICIR={pir} CORR={pc}")
    sys.stdout.flush()

print("\n===== SUMMARY =====")
for name, m, pi, pir, pc in rows:
    flag = "PASS" if (pi and pir and pc) else "FAIL"
    print(f"{flag:4s} {name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} corr={m['max_abs_library_correlation']:.3f} n={m['n_ic_dates']}")

# ---- re-validation of currently-effective library factors (report only) ----
print("\n===== LIBRARY RE-VALIDATION (report only, window 2020-01-01..2026-11-06) =====")
for name, lib_df in lib.items():
    lib_df = lib_df.reindex(closes.index)
    ics = rank_ic_vec(lib_df, fwd_by_h[HORIZON], MIN_VALID)
    m = summarize_ic(ics, 1)
    sub = {}
    for sname, (s0, s1) in SUB_PERIODS.items():
        sub_ics = rank_ic_vec(lib_df.loc[s0:s1], fwd_by_h[HORIZON].loc[s0:s1], MIN_VALID)
        sub[sname] = round(float(sub_ics.mean()), 4) if len(sub_ics) else None
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} sub={sub}")

print(f"\ntotal time {time.time()-t0:.1f}s")
