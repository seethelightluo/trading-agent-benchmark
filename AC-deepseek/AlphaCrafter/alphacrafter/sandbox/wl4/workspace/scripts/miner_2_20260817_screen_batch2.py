"""miner_2 exploration batch 2 (2026-08-17): NaN-safe screening of new factor families.

Fixes the NaN-propagation bug of batch1 (cross-asset calendars -> rolling stats on the
union calendar produced NaN). All rolling stats are computed per-asset on each asset's
own complete series, then reindexed to the union calendar.

Candidates:
  A risk_adj_mom_60d    : 60d momentum / 20d vol (retest, NaN-safe)
  B bollinger_pos_20d   : (close-sma20)/std20 (retest, NaN-safe)
  C max_dd_60d          : close/60d max - 1 (retest, NaN-safe)
  D skew_60d            : realized skew 60d (retest, NaN-safe)
  E range_ratio_20d     : mean((h-l)/c) 20d, negated (retest, NaN-safe)
  F vol_surprise_5d     : volume/60d mean vol, 5d avg, negated (retest, NaN-safe)
  G parkinson_vol_inv_20d: -sqrt(mean(ln(h/l)^2)) 20d (retest, NaN-safe)
  H downside_vol_ratio_60d: downside std / total std 60d, negated (retest)
  I eff_ratio_20d       : |c_t-c_{t-20}| / sum(|daily ret|) 20d (trend efficiency)
  J var_ratio_20d       : var(20d ret) / (20*var(1d ret)) (variance ratio)
  K up_day_ratio_60d    : fraction of positive days over 60d (momentum consistency)
  L intraday_pos_20d    : mean((c-l)/(h-l)) 20d (close-in-range position)
  M vol_term_60_20      : 60d vol / 20d vol (volatility term structure)
  N autocorr_20d        : lag-1 autocorrelation of returns over 20d
  O corr_us10y_20d      : rolling 20d corr of asset ret with US10Y yield change
  P crypto_spillover_10d: avg(BTC,ETH) 10d return (crypto spillover)
  Q max_gain_20d        : max daily return over 20d
  R skew_term_20_60     : skew20 - skew60

Gate: |IC|>=0.007, |ICIR|>=0.084 at h=10; library corr < 0.5 vs persisted artifacts.
Uses a fast vectorized rank-IC (pandas rank + numpy per-date Pearson).
"""
from __future__ import annotations
import sys, json, base64, zlib, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, TRADABLE)

HORIZON = 10
MIN_VALID = 8
WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-07-15"))

def rank_ic_vec(factor_panel, fwd, min_valid=8):
    """Fast daily Spearman rank IC (Pearson on cross-sectional ranks)."""
    common = factor_panel.index.intersection(fwd.index)
    F = factor_panel.loc[common]
    R = fwd.loc[common]
    Fr = F.rank(axis=1)
    Rr = R.rank(axis=1)
    Fv = Fr.values.astype(float)
    Rv = Rr.values.astype(float)
    dates, ics = [], []
    for i in range(Fv.shape[0]):
        x = Fv[i]
        y = Rv[i]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < min_valid:
            continue
        xm = x[mask]; ym = y[mask]
        if xm.std() < 1e-14 or ym.std() < 1e-14:
            continue
        xc = xm - xm.mean(); yc = ym - ym.mean()
        ic = float((xc * yc).sum() / np.sqrt((xc * xc).sum() * (yc * yc).sum()))
        dates.append(common[i]); ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")

def summarize_ic(ic_series, expected_sign=1):
    ic = ic_series.mean()
    sd = ic_series.std(ddof=1) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean()) if expected_sign else float((np.sign(ic_series) != 0).mean())
    return {"ic": round(float(ic), 4), "icir": round(float(icir), 4),
            "ic_hit_ratio": round(float(hit), 3), "n_ic_dates": int(len(ic_series)),
            "ic_std": round(float(sd), 4)}

def decay_profile(factor_panel, closes, horizons=(1, 3, 5, 10, 20), min_valid=8):
    out = {}
    for h in horizons:
        fwd = forward_returns(closes, h)
        ics = rank_ic_vec(factor_panel, fwd, min_valid)
        if len(ics):
            out[str(h)] = round(float(ics.mean()), 4)
    return out

t0 = time.time()
panels = load_panels(3000)
closes_all = close_panel(panels)
rets_all = closes_all.pct_change()
print(f"loaded panels {time.time()-t0:.1f}s")

clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}

def asset_wide(func):
    out = {}
    for a, s in clean.items():
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes_all.index)

cand = {}
# A
def f_risk_mom(s):
    r = s.pct_change()
    return (s.shift(5) / s.shift(65) - 1.0) / r.rolling(20).std()
cand["risk_adj_mom_60d"] = asset_wide(f_risk_mom)
# B
def f_boll(s):
    return (s - s.rolling(20).mean()) / s.rolling(20).std()
cand["bollinger_pos_20d"] = asset_wide(f_boll)
# C
def f_maxdd(s):
    return s / s.rolling(60).max() - 1.0
cand["max_dd_60d"] = asset_wide(f_maxdd)
# D
def f_skew(s):
    return s.pct_change().rolling(60).skew()
cand["skew_60d"] = asset_wide(f_skew)
# E range ratio 20d negated
rng = {}
for a in TRADABLE:
    if a not in clean: continue
    h = panels[a]["high"].astype(float).dropna(); l = panels[a]["low"].astype(float).dropna(); c = clean[a]
    idx = c.index.intersection(h.index).intersection(l.index)
    rng[a] = -(((h.loc[idx] - l.loc[idx]) / c.loc[idx]).rolling(20).mean())
cand["range_ratio_20d"] = pd.DataFrame(rng).reindex(closes_all.index)
# F volume surprise 5d negated
vs = {}
for a in TRADABLE:
    if a not in clean: continue
    v = panels[a]["volume"].astype(float).dropna(); c = clean[a]
    idx = c.index.intersection(v.index)
    vs[a] = -((v.loc[idx] / v.loc[idx].rolling(60).mean()).rolling(5).mean())
cand["vol_surprise_5d"] = pd.DataFrame(vs).reindex(closes_all.index)
# G parkinson vol inv 20d
pk = {}
for a in TRADABLE:
    if a not in clean: continue
    h = panels[a]["high"].astype(float).dropna(); l = panels[a]["low"].astype(float).dropna(); c = clean[a]
    idx = c.index.intersection(h.index).intersection(l.index)
    pk[a] = -np.sqrt((np.log(h.loc[idx] / l.loc[idx]) ** 2).rolling(20).mean())
cand["parkinson_vol_inv_20d"] = pd.DataFrame(pk).reindex(closes_all.index)
# H
def f_downside(s):
    r = s.pct_change()
    neg = r.where(r < 0, 0.0)
    return -(neg.rolling(60).std() / r.rolling(60).std())
cand["downside_vol_ratio_60d"] = asset_wide(f_downside)
# I efficiency ratio
def f_eff(s):
    return (s - s.shift(20)).abs() / s.pct_change().abs().rolling(20).sum()
cand["eff_ratio_20d"] = asset_wide(f_eff)
# J variance ratio
def f_var_ratio(s):
    r = s.pct_change()
    return r.rolling(20).var() / r.rolling(20).var(ddof=1) * 0 + r.rolling(20).var() / (20.0 * r.var())
cand["var_ratio_20d"] = asset_wide(f_var_ratio)
# K up-day ratio
def f_upday(s):
    return (s.pct_change() > 0).rolling(60).mean()
cand["up_day_ratio_60d"] = asset_wide(f_upday)
# L intraday position
ip = {}
for a in TRADABLE:
    if a not in clean: continue
    h = panels[a]["high"].astype(float).dropna(); l = panels[a]["low"].astype(float).dropna(); c = clean[a]
    idx = c.index.intersection(h.index).intersection(l.index)
    ip[a] = ((c.loc[idx] - l.loc[idx]) / (h.loc[idx] - l.loc[idx])).rolling(20).mean()
cand["intraday_pos_20d"] = pd.DataFrame(ip).reindex(closes_all.index)
# M vol term 60/20
def f_volterm(s):
    r = s.pct_change()
    return r.rolling(60).std() / r.rolling(20).std()
cand["vol_term_60_20"] = asset_wide(f_volterm)
# N autocorr
def f_autocorr(s):
    r = s.pct_change()
    return r.rolling(20).corr(r.shift(1))
cand["autocorr_20d"] = asset_wide(f_autocorr)
# O corr us10y 20d
us10y_r = panels["US10Y"]["close"].astype(float).dropna().pct_change()
def f_corrus10y(s):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), us10y_r.rename("u")], axis=1).dropna()
    return z["a"].rolling(20).corr(z["u"])
cand["corr_us10y_20d"] = asset_wide(f_corrus10y)
# P crypto spillover
cr = ((clean["BTC"].pct_change() + clean["ETH"].pct_change()) / 2.0).reindex(closes_all.index)
cand["crypto_spillover_10d"] = pd.DataFrame({a: cr.rolling(10).mean() for a in TRADABLE})
# Q max gain
def f_maxgain(s):
    return s.pct_change().rolling(20).max()
cand["max_gain_20d"] = asset_wide(f_maxgain)
# R skew term
def f_skewterm(s):
    r = s.pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
cand["skew_term_20_60"] = asset_wide(f_skewterm)
print(f"candidates built {time.time()-t0:.1f}s")

idx = (closes_all.index >= WINDOW[0]) & (closes_all.index <= WINDOW[1])
closes = closes_all.loc[idx]
cand = {k: v.loc[idx] for k, v in cand.items()}

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

fwd = forward_returns(closes, HORIZON)
rows = []
for name, panel in cand.items():
    panel = panel.reindex(closes.index)
    ics = rank_ic_vec(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 3, 5, 10, 20), MIN_VALID)
    corr, key = max_lib_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    pi = abs(m["ic"]) >= 0.007
    pir = abs(m["icir"]) >= 0.084
    pc = corr < 0.5
    rows.append((name, m, pi, pir, pc))
    print(f"\n=== {name} === IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} "
          f"cov_asset={m['coverage_asset_days']:.3f} cov_ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']} "
          f"decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} } lib_corr={corr:.3f}({key}) "
          f"GATES IC={pi} ICIR={pir} CORR={pc}")
    sys.stdout.flush()

print("\n===== SUMMARY =====")
for name, m, pi, pir, pc in rows:
    flag = "PASS" if (pi and pir and pc) else "FAIL"
    print(f"{flag:4s} {name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} corr={m['max_abs_library_correlation']:.3f} n={m['n_ic_dates']}")
print(f"total time {time.time()-t0:.1f}s")
