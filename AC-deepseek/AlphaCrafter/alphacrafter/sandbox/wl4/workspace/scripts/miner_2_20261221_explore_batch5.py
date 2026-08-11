"""miner_2 2026-12-21: explore fresh factor families (batch5, updated window).
Validation window 2020-01-01..2026-12-18 (last completed trading day).
Vectorized NaN-safe rank-IC; gates h=10: |IC|>=0.0070, |ICIR|>=0.0840,
max_abs_library_correlation < 0.5 (self-reported provenance only).

Candidates (direction exploratory, sign decided by data; expected_sign used for hit-ratio only):
  A trend_eff_20d        : Kaufman efficiency |close[t]-close[t-20]|/sum(|ret|,20)   [+1]
  B downside_dev_ratio_60d: downside semi-dev / total vol 60d (crash-risk asymmetry) [-1]
  C range_pos_20d        : (close - min(low,20))/(max(high,20)-min(low,20))          [+1]
  D vol_zscore_60_250    : (vol20 - mean(vol20,250))/std(vol20,250)                  [-1]
  E upday_ratio_60d      : fraction of up days over 60d                              [+1]
  F ret_autocorr_60d     : rolling corr(ret, ret.shift(1), 60) (trend persistence)   [+1]
  G crypto_beta_60d      : rolling 60d beta of asset ret on BTC ret (risk-on beta)   [+1]
  H tech_beta_spread_60d : beta(asset,NDX,60) - beta(asset,SPX,60)                   [+1]
  I overnight_ratio_60d  : sum(log(open/pc),60) / sum(|log(close/open)|,60)         [+1]
  J dayclose_pos_20d     : mean((close-open)/(high-low),20) (intraday close pos)     [+1]
  K max_drawdown_60d     : -max drawdown over 60d (drawdown depth)                   [-1]
  L skew_60d             : return skewness over 60d                                  [-1]
  M hi_lo_range_20d      : mean((high-low)/close,20)/vol20 (gap vs realized)         [+/-]
  N vol_of_vol_60_250    : std(vol20,250)/mean(vol20,250)  (vol regime instability)  [-1]
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
END = pd.Timestamp("2026-12-18")
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
print(f"loaded panels {time.time()-t0:.1f}s | closes {closes_all.shape} | last {closes_all.index[-1]}")

clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}
assert closes_all.index[-1] <= END, f"data beyond END: {closes_all.index[-1]}"


def asset_wide(func):
    out = {}
    for a, s in clean.items():
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes_all.index)


cand = {}
# A trend efficiency (Kaufman ER) 20d
cand["trend_eff_20d"] = asset_wide(lambda s: (s - s.shift(20)).abs() / s.pct_change().abs().rolling(20).sum())
# B downside deviation ratio 60d (negated: high downside share => crash risk, expect -1)
def downside_ratio(s):
    r = s.pct_change()
    dd = r.where(r < 0, 0.0).rolling(60).std()
    tot = r.rolling(60).std()
    return dd / tot
cand["downside_dev_ratio_60d"] = asset_wide(downside_ratio)
# C range position 20d
def range_pos(s):
    hi = s.rolling(20).max()
    lo = s.rolling(20).min()
    return (s - lo) / (hi - lo)
cand["range_pos_20d"] = asset_wide(range_pos)
# D vol z-score: (vol20 - mean(vol20,250))/std(vol20,250)
def vol_zscore(s):
    v = s.pct_change().rolling(20).std()
    mu = v.rolling(250).mean()
    sd = v.rolling(250).std()
    return (v - mu) / sd
cand["vol_zscore_60_250"] = asset_wide(vol_zscore)
# E up-day ratio 60d
cand["upday_ratio_60d"] = asset_wide(lambda s: (s.pct_change() > 0).rolling(60).mean())
# F return autocorrelation 60d (trend persistence)
def autocorr(s):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), r.shift(1).rename("b")], axis=1)
    return z["a"].rolling(60).corr(z["b"])
cand["ret_autocorr_60d"] = asset_wide(autocorr)
# G crypto beta 60d (BTC driver)
btc_ret = clean["BTC"].pct_change()
cand["crypto_beta_60d"] = asset_wide(lambda s: rolling_beta(s.pct_change(), btc_ret, 60))
# H tech beta spread: beta(NDX) - beta(SPX) 60d
ndx_ret = clean["NDX"].pct_change()
spx_ret = clean["SPX"].pct_change()
def tech_spread(s):
    return rolling_beta(s.pct_change(), ndx_ret, 60) - rolling_beta(s.pct_change(), spx_ret, 60)
cand["tech_beta_spread_60d"] = asset_wide(tech_spread)
# I overnight ratio 60d
overnight = {}
intraday = {}
for a in TRADABLE:
    if a not in clean:
        continue
    o = panels[a]["open"].astype(float).dropna()
    c = clean[a]
    idx = c.index.intersection(o.index)
    gap = (o.loc[idx] / c.loc[idx].shift(1)).pipe(np.log)
    intra = (c.loc[idx] / o.loc[idx]).pipe(np.log)
    overnight[a] = gap.rolling(60).sum()
    intraday[a] = intra.abs().rolling(60).sum()
cand["overnight_ratio_60d"] = (pd.DataFrame(overnight).reindex(closes_all.index)
                               / pd.DataFrame(intraday).reindex(closes_all.index))
# J day close position 20d (mean of (close-open)/(high-low))
def dayclose_pos(s):
    pass  # built below per-asset (needs OHLC)
dcp = {}
for a in TRADABLE:
    if a not in clean:
        continue
    df = panels[a]
    o, h, l, c = df["open"].astype(float), df["high"].astype(float), df["low"].astype(float), clean[a]
    idx = c.index.intersection(o.index)
    rng = (h.loc[idx] - l.loc[idx]).replace(0, np.nan)
    dcp[a] = ((c.loc[idx] - o.loc[idx]) / rng).rolling(20).mean()
cand["dayclose_pos_20d"] = pd.DataFrame(dcp).reindex(closes_all.index)
# K max drawdown depth 60d (negated: deeper drawdown => expect -1)
def max_dd(s):
    roll_max = s.rolling(60).max()
    dd = s / roll_max - 1.0
    return dd.rolling(60).min()
cand["max_drawdown_60d"] = asset_wide(max_dd)
# L return skewness 60d
cand["skew_60d"] = asset_wide(lambda s: s.pct_change().rolling(60).skew())
# M hi-lo range ratio 20d: mean((high-low)/close,20) / vol20
hlr = {}
for a in TRADABLE:
    if a not in clean:
        continue
    df = panels[a]
    h, l, c = df["high"].astype(float), df["low"].astype(float), clean[a]
    idx = c.index.intersection(h.index)
    hlr[a] = ((h.loc[idx] - l.loc[idx]) / c.loc[idx]).rolling(20).mean() / c.loc[idx].pct_change().rolling(20).std()
cand["hi_lo_range_20d"] = pd.DataFrame(hlr).reindex(closes_all.index)
# N vol-of-vol ratio 60/250
cand["vol_of_vol_60_250"] = asset_wide(lambda s: s.pct_change().rolling(20).std().rolling(60).std()
                                       / s.pct_change().rolling(20).std().rolling(250).mean())

print(f"candidates built {time.time()-t0:.1f}s | n={len(cand)}")
print("candidate list:", list(cand.keys()))

idx = (closes_all.index >= WINDOW[0]) & (closes_all.index <= WINDOW[1])
closes = closes_all.loc[idx]
cand = {k: v.loc[idx] for k, v in cand.items()}

fwd_by_h = {h: closes.shift(-h) / closes - 1.0 for h in HORIZONS}

# library signals from persisted JSON signal artifacts
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
print("\n===== LIBRARY RE-VALIDATION (report only, window 2020-01-01..2026-12-18) =====")
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
