"""miner_2 2026-09-14: explore fresh factor families (batch 3 - oscillators, macro betas,
conditional momentum, volume-price hybrids). Vectorized NaN-safe rank-IC.

Fresh candidates vs all prior batches (mom/skew/range/vol/beta-to-EURUSD-CN10Y-USDCNY-VIX already done):
  A stoch_14d          : %K stochastic (c-LL14)/(HH14-LL14)                      [+1]
  B cci_20d            : Commodity Channel Index 20d (TP=(h+l+c)/3)              [+1]
  C macd_hist_norm     : (EMA12-EMA26)/close                                     [+/-]
  D ts_zscore_60d      : (c-SMA60)/std60 time-series z-score                     [+1]
  E vol_percentile_60d : 20d vol / rolling max(20d vol,60d)  (vol compression)   [-1]
  F beta_dxy_60d       : rolling beta of asset ret to DXY ret (60d)              [-1]
  G beta_usdjpy_60d    : rolling beta to USDJPY ret (60d)                        [+/-]
  H beta_xau_60d       : rolling beta to XAU ret (60d)                           [+/-]
  I beta_btc_60d       : rolling beta to BTC ret (60d)                           [+1]
  J corr_btc_60d       : rolling 60d corr of asset ret with BTC ret              [+1]
  K up_beta_60d        : beta on up-market days only (60d)                       [+1]
  L beta_asym_60d      : up_beta - dn_beta (60d)                                 [+/-]
  M obv_slope_20d      : OBV 20d linear slope / close                            [+1]
  N vwap_dist_20d      : close / rolling VWAP(20) - 1  (volume-weighted price)   [+1]
  O mom20_x_vixz       : mom_20d_skip2 * (-vix_z60)  (conditional momentum)      [+1]
  P kurtosis_60d       : excess kurtosis of 60d returns                          [-1]
  Q max_loss_20d       : min daily return over 20d                               [-1]
  R avg_corr_60d       : mean pairwise corr of asset ret vs all other assets     [-1]
  S atr_ratio_20d      : ATR(20)/close                                           [-1]
  T range_breakout_10d : close / max(high,10d prior) - 1  (breakout)             [+1]

Gate (h=10): |IC|>=0.0070, |ICIR|>=0.0840, max_abs_library_correlation < 0.5.
Validation window: 2020-01-01 .. 2026-09-11 (last completed day). Also reports
sub-period IC (2023-01-01.., 2025-01-01.., online 2026-07-16..) for drift check.
"""
from __future__ import annotations
import sys, json, base64, zlib, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, TRADABLE, MACRO,
                                 coverage_metrics, turnover_rank)

HORIZON = 10
MIN_VALID = 8
WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-09-11"))
HORIZONS = (1, 2, 3, 5, 10, 20)
SUB_PERIODS = {"full": (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-09-11")),
               "p23": (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-09-11")),
               "p25": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-11")),
               "online": (pd.Timestamp("2026-07-16"), pd.Timestamp("2026-09-11"))}


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
    b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return b


t0 = time.time()
panels = load_panels(3000)
closes_all = close_panel(panels)
rets_all = closes_all.pct_change()
print(f"loaded panels {time.time()-t0:.1f}s | closes {closes_all.shape}")

clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}


def asset_wide(func):
    out = {}
    for a, s in clean.items():
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes_all.index)


cand = {}
# A stochastic %K 14d
cand["stoch_14d"] = asset_wide(lambda s: (s - s.rolling(14).min()) / (s.rolling(14).max() - s.rolling(14).min()))
# D ts zscore 60d
cand["ts_zscore_60d"] = asset_wide(lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std())
# E vol percentile: 20d vol / rolling max of 20d vol over 60d
cand["vol_percentile_60d"] = asset_wide(lambda s: s.pct_change().rolling(20).std() / s.pct_change().rolling(20).std().rolling(60).max())
# C MACD hist normalized
cand["macd_hist_norm"] = asset_wide(lambda s: (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()) / s)
# K up beta 60d: beta of asset ret on market ret where market ret > 0
mkt_ret = rets_all.mean(axis=1)
up_mkt = mkt_ret.where(mkt_ret > 0)
cand["up_beta_60d"] = asset_wide(lambda s: rolling_beta(s.pct_change(), up_mkt, 60))
# L beta asymmetry
dn_mkt = mkt_ret.where(mkt_ret < 0)
ub = {a: rolling_beta(clean[a].pct_change(), up_mkt, 60) for a in clean}
db = {a: rolling_beta(clean[a].pct_change(), dn_mkt, 60) for a in clean}
cand["beta_asym_60d"] = pd.DataFrame(ub).reindex(closes_all.index) - pd.DataFrame(db).reindex(closes_all.index)
# M OBV slope 20d / close
def obv_slope(s):
    v = panels[s.name]["volume"].astype(float).dropna() if "volume" in panels[s.name] else None
    if v is None:
        return pd.Series(np.nan, index=s.index)
    r = s.pct_change()
    obv = (np.sign(r) * v).cumsum()
    return obv.rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / len(x), raw=True) / s
cand["obv_slope_20d"] = asset_wide(obv_slope)
# N vwap distance 20d: sum(close*vol)/sum(vol) over 20d
def vwap_dist(s):
    v = panels[s.name]["volume"].astype(float).dropna() if "volume" in panels[s.name] else None
    if v is None:
        return pd.Series(np.nan, index=s.index)
    z = pd.concat([s.rename("c"), v.rename("v")], axis=1).dropna()
    vwap = (z["c"] * z["v"]).rolling(20).sum() / z["v"].rolling(20).sum()
    return z["c"] / vwap - 1.0
cand["vwap_dist_20d"] = asset_wide(vwap_dist)
# O conditional momentum: mom_20d_skip2 * (-vix_z60)
vix = panels["VIX"]["close"].astype(float).dropna()
vix_z = ((vix - vix.rolling(60).mean()) / vix.rolling(60).std()).reindex(closes_all.index)
mom20 = closes_all.shift(2) / closes_all.shift(22) - 1.0
cand["mom20_x_vixz"] = mom20 * (-vix_z)
# P kurtosis 60d
cand["kurtosis_60d"] = asset_wide(lambda s: s.pct_change().rolling(60).kurt())
# Q max loss 20d
cand["max_loss_20d"] = asset_wide(lambda s: s.pct_change().rolling(20).min())
# R avg corr 60d (mean pairwise correlation with other assets)
avg_corr = {}
for a in TRADABLE:
    if a not in clean:
        continue
    others = [b for b in TRADABLE if b in clean and b != a]
    ra = clean[a].pct_change()
    cols = {}
    for b in others:
        z = pd.concat([ra.rename("a"), clean[b].pct_change().rename("b")], axis=1).dropna()
        cols[b] = z["a"].rolling(60).corr(z["b"])
    avg_corr[a] = pd.DataFrame(cols).mean(axis=1)
cand["avg_corr_60d"] = pd.DataFrame(avg_corr).reindex(closes_all.index)

# high/low based candidates
for name, build in [
    ("cci_20d", lambda h, l, c: (lambda tp: (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean()))((h + l + c) / 3.0)),
    ("atr_ratio_20d", lambda h, l, c: (lambda pc: (lambda tr: tr.rolling(20).mean() / c)(pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)))(c.shift(1))),
    ("range_breakout_10d", lambda h, l, c: c / h.rolling(10).max().shift(1) - 1.0),
]:
    out = {}
    for a in TRADABLE:
        if a not in clean:
            continue
        h = panels[a]["high"].astype(float).dropna()
        l = panels[a]["low"].astype(float).dropna()
        c = clean[a]
        idx = c.index.intersection(h.index).intersection(l.index)
        out[a] = build(h.loc[idx], l.loc[idx], c.loc[idx])
    cand[name] = pd.DataFrame(out).reindex(closes_all.index)

# macro beta candidates
for name, driver in [("beta_dxy_60d", "DXY"), ("beta_usdjpy_60d", "USDJPY"),
                     ("beta_xau_60d", "XAU"), ("beta_btc_60d", "BTC")]:
    dr = panels[driver]["close"].astype(float).dropna().pct_change()
    out = {a: rolling_beta(clean[a].pct_change(), dr, 60) for a in clean}
    cand[name] = pd.DataFrame(out).reindex(closes_all.index)

# J corr with BTC
btc_r = clean["BTC"].pct_change()
cand["corr_btc_60d"] = asset_wide(lambda s: pd.concat([s.pct_change().rename("a"), btc_r.rename("b")], axis=1).dropna().pipe(
    lambda z: z["a"].rolling(60).corr(z["b"])))

print(f"candidates built {time.time()-t0:.1f}s | n={len(cand)}")
print("candidate list:", list(cand.keys()))

idx = (closes_all.index >= WINDOW[0]) & (closes_all.index <= WINDOW[1])
closes = closes_all.loc[idx]
cand = {k: v.loc[idx] for k, v in cand.items()}

fwd_by_h = {h: closes.shift(-h) / closes - 1.0 for h in HORIZONS}

# library signals from persisted JSON artifacts
lib = {}
for p in sorted(Path("factors").glob("*.json")):
    if p.name == "factor_ensemble.json":
        continue
    try:
        d = json.loads(p.read_text())
        sa = d.get("validation", {}).get("signal_artifact")
        if not sa:
            continue
        raw = zlib.decompress(base64.b64decode(sa["data"]))
        df = pd.read_csv(io.BytesIO(raw), index_col=0)
        df.index = pd.to_datetime(df.index)
        lib[d["factor_id"]] = df.loc[idx]
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
print(f"total time {time.time()-t0:.1f}s")
