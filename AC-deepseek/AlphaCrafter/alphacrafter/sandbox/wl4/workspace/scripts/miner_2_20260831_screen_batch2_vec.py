"""miner_2 (2026-08-31): vectorized NaN-safe screen of factor families (batch 2 + extras).

Fully vectorized daily Spearman rank-IC (numpy row-wise Pearson on cross-sectional ranks,
no per-date Python loop). Precomputes forward returns for all horizons once.

Candidates:
  A risk_adj_mom_60d    : 60d momentum / 20d vol
  B bollinger_pos_20d   : (close-sma20)/std20
  C max_dd_60d          : close/60d max - 1
  D skew_60d            : realized skew 60d
  E range_ratio_20d     : mean((h-l)/c) 20d, negated
  F vol_surprise_5d     : volume/60d mean vol, 5d avg, negated
  G parkinson_vol_inv_20d: -sqrt(mean(ln(h/l)^2)) 20d
  H downside_vol_ratio_60d: downside std / total std 60d, negated
  I eff_ratio_20d       : |c_t-c_{t-20}| / sum(|daily ret|) 20d
  J var_ratio_20d       : var(20d ret)/(20*var(1d ret))
  K up_day_ratio_60d    : fraction of positive days over 60d
  L intraday_pos_20d    : mean((c-l)/(h-l)) 20d
  M vol_term_60_20      : 60d vol / 20d vol
  N autocorr_20d        : lag-1 autocorrelation of returns over 20d
  O corr_us10y_20d      : rolling 20d corr with US10Y yield change
  P crypto_spillover_10d: avg(BTC,ETH) 10d return
  Q max_gain_20d        : max daily return over 20d
  R skew_term_20_60     : skew20 - skew60
  S mom_60d_skip5       : c[t-5]/c[t-65]-1
  T drawdown_recovery_20d: c/rolling_max(c,20)-1
  U hilo_pos_20d        : (c-min_low20)/(max_high20-min_low20)
  V corr_spx_20d        : rolling 20d corr of asset ret with SPX ret

Gate (h=10, shared benchmark admission): |IC|>=0.007, |ICIR|>=0.084, lib corr<0.5.
Validation window: 2020-01-01 .. 2026-07-15 (research warm-up).
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
WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-07-15"))
HORIZONS = (1, 2, 3, 5, 10, 20)


def rank_ic_vec(F: pd.DataFrame, R: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    """Fully vectorized daily Spearman rank IC."""
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


t0 = time.time()
panels = load_panels(3000)
closes_all = close_panel(panels)
print(f"loaded panels {time.time()-t0:.1f}s | closes {closes_all.shape}")

clean = {a: closes_all[a].dropna() for a in TRADABLE if len(closes_all[a].dropna()) > 300}


def asset_wide(func):
    out = {}
    for a, s in clean.items():
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes_all.index)


cand = {}
cand["risk_adj_mom_60d"] = asset_wide(lambda s: (s.shift(5) / s.shift(65) - 1.0) / s.pct_change().rolling(20).std())
cand["bollinger_pos_20d"] = asset_wide(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
cand["max_dd_60d"] = asset_wide(lambda s: s / s.rolling(60).max() - 1.0)
cand["skew_60d"] = asset_wide(lambda s: s.pct_change().rolling(60).skew())
cand["mom_60d_skip5"] = asset_wide(lambda s: s.shift(5) / s.shift(65) - 1.0)
cand["drawdown_recovery_20d"] = asset_wide(lambda s: s / s.rolling(20).max() - 1.0)
cand["hilo_pos_20d"] = asset_wide(lambda s: (s - s.rolling(20).min()) / (s.rolling(20).max() - s.rolling(20).min()))
cand["corr_spx_20d"] = asset_wide(lambda s: s.pct_change().rolling(20).corr(clean["SPX"].pct_change()))
cand["downside_vol_ratio_60d"] = asset_wide(lambda s: -(s.pct_change().where(s.pct_change() < 0, 0.0).rolling(60).std() / s.pct_change().rolling(60).std()))
cand["eff_ratio_20d"] = asset_wide(lambda s: (s - s.shift(20)).abs() / s.pct_change().abs().rolling(20).sum())
cand["var_ratio_20d"] = asset_wide(lambda s: s.pct_change().rolling(20).var() / (20.0 * s.pct_change().var()))
cand["up_day_ratio_60d"] = asset_wide(lambda s: (s.pct_change() > 0).rolling(60).mean())
cand["vol_term_60_20"] = asset_wide(lambda s: s.pct_change().rolling(60).std() / s.pct_change().rolling(20).std())
cand["autocorr_20d"] = asset_wide(lambda s: s.pct_change().rolling(20).corr(s.pct_change().shift(1)))
cand["max_gain_20d"] = asset_wide(lambda s: s.pct_change().rolling(20).max())
cand["skew_term_20_60"] = asset_wide(lambda s: s.pct_change().rolling(20).skew() - s.pct_change().rolling(60).skew())

# high/low / volume / intraday candidates (per-asset series, union-calendar reindex)
for name, build in [
    ("range_ratio_20d", lambda a, h, l, c, v: -(((h - l) / c).rolling(20).mean())),
    ("parkinson_vol_inv_20d", lambda a, h, l, c, v: -np.sqrt((np.log(h / l) ** 2).rolling(20).mean())),
    ("intraday_pos_20d", lambda a, h, l, c, v: ((c - l) / (h - l)).rolling(20).mean()),
    ("vol_surprise_5d", lambda a, h, l, c, v: -((v / v.rolling(60).mean()).rolling(5).mean())),
]:
    out = {}
    for a in TRADABLE:
        if a not in clean:
            continue
        h = panels[a]["high"].astype(float).dropna()
        l = panels[a]["low"].astype(float).dropna()
        c = clean[a]
        v = panels[a]["volume"].astype(float).dropna() if "volume" in panels[a] else None
        idx = c.index
        for other in (h, l):
            idx = idx.intersection(other.index)
        if v is not None:
            idx = idx.intersection(v.index)
        try:
            out[a] = build(a, h.loc[idx], l.loc[idx], c.loc[idx], v.loc[idx] if v is not None else None)
        except Exception as e:
            print(f"  {name}/{a} build error: {e}")
    cand[name] = pd.DataFrame(out).reindex(closes_all.index)

# O: corr with US10Y yield change
us10y_r = panels["US10Y"]["close"].astype(float).dropna().pct_change()
cand["corr_us10y_20d"] = asset_wide(lambda s: pd.concat([s.pct_change().rename("a"), us10y_r.rename("u")], axis=1).dropna().pipe(
    lambda z: z["a"].rolling(20).corr(z["u"])))

# P: crypto spillover (same value for every asset -> mostly a check on common factor)
cr = ((clean["BTC"].pct_change() + clean["ETH"].pct_change()) / 2.0).reindex(closes_all.index)
cand["crypto_spillover_10d"] = pd.DataFrame({a: cr.rolling(10).mean() for a in TRADABLE})
print(f"candidates built {time.time()-t0:.1f}s | n={len(cand)}")

idx = (closes_all.index >= WINDOW[0]) & (closes_all.index <= WINDOW[1])
closes = closes_all.loc[idx]
cand = {k: v.loc[idx] for k, v in cand.items()}

# precompute forward returns for all horizons once
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
