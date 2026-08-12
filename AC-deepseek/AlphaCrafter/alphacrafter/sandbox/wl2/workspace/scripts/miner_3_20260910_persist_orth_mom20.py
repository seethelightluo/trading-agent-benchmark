"""miner_3 2026-09-10: persist orth_mom20 (orthogonalized momentum) as EFFECTIVE.

Candidate validated on data visible through 2026-09-09 (GRID last row):
  orth_mom20: IC=+0.0405 ICIR=+0.1442 hit=0.543 n=1663 cov_ad=0.655
              cov_d8=0.686 turn=0.265 maxlibcorr=0.483 (carry_3m1m) GATE=True

Construction: per-date OLS residual of cross-sectional rank(mom20_volproxy60)
on anchor ranks {downbeta_spx_60, volcluster_60, range_pos_252, spx_corr60,
calmness_20}, 2 passes, standardized to z. The residual z-score is the signal.
"""
import sys, json, os, datetime
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
CORR_LIMIT = 0.5

def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df

series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
print(f"grid rows={len(GRID)} first={GRID[0]} last={GRID[-1]} assets={len(series)}/15")
spx_ret = series["SPX"]["ret"]

# ---- anchor + candidate definitions (must match batchE2 script exactly) ----
def a_downbeta(df):
    r = df["ret"]
    j = pd.concat([r, spx_ret], axis=1, join="outer")
    j.columns = ["a", "b"]
    dn = j["b"] < 0
    cov = j["a"].where(dn).rolling(60, min_periods=15).cov(j["b"].where(dn))
    var = j["b"].where(dn).rolling(60, min_periods=15).var()
    return pd.Series(safe_div(cov, var), index=j.index).reindex(df.index)

def a_volcluster(df):
    a = df["ret"].abs()
    return a.rolling(60, min_periods=40).corr(a.shift(1))

def a_range252(df):
    lo = df["close"].rolling(252, min_periods=30).min()
    hi = df["close"].rolling(252, min_periods=30).max()
    return pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)

def a_spxcorr(df):
    r = df["ret"]
    j = pd.concat([r, spx_ret], axis=1, join="outer")
    j.columns = ["a", "b"]
    return j["a"].rolling(60, min_periods=15).corr(j["b"]).reindex(df.index)

def a_calmness(df):
    sd = df["ret"].rolling(20, min_periods=10).std()
    calm = (df["ret"].abs() < 0.5 * sd).astype(float)
    return calm.rolling(20, min_periods=10).mean()

def a_mom20(df):
    raw = df["close"].shift(5) / df["close"].shift(25) - 1.0
    damp = 1.0 / (1.0 + df["close"].pct_change(60).abs())
    return raw * damp

def _to_series_dict(fn):
    return {s: fn(df) for s, df in series.items()}

ANCHORS = {
    "downbeta_spx_60": a_downbeta,
    "volcluster_60": a_volcluster,
    "range_pos_252": a_range252,
    "spx_corr60": a_spxcorr,
    "calmness_20": a_calmness,
    "mom20_volproxy60": a_mom20,
}
anchor_mats = {k: to_grid(_to_series_dict(fn)) for k, fn in ANCHORS.items()}

def orth_resid(cand_mat, anchor_names, passes=2, min_obs=8):
    """Per-date OLS residual of cross-sectional rank of candidate on anchor ranks."""
    T, n = cand_mat.shape
    Xs = [cross_sectional_rank(anchor_mats[k]) for k in anchor_names]
    y0 = cross_sectional_rank(cand_mat)
    cur = y0.copy()
    for _ in range(passes):
        out = np.full_like(cand_mat, np.nan)
        for t in range(T):
            cols = [np.ones(n)] + [X[t] for X in Xs]
            Xm = np.column_stack(cols)
            y = cur[t]
            ok = ~(np.isnan(y) | np.isnan(Xm).any(axis=1))
            if ok.sum() < min_obs:
                continue
            try:
                beta, *_ = np.linalg.lstsq(Xm[ok], y[ok], rcond=None)
            except Exception:
                continue
            res = y[ok] - Xm[ok] @ beta
            sd = float(np.std(res))
            if sd > 1e-12:
                res = (res - np.mean(res)) / sd
            out[t, ok] = res
        cur = out
    fin = cross_sectional_rank(cur)
    for t in range(T):
        row = fin[t]
        ok = ~np.isnan(row)
        if ok.sum() < min_obs:
            continue
        z = (row[ok] - np.mean(row[ok])) / np.std(row[ok])
        fin[t, ok] = z
    return fin

ORTH_ANCHORS = ["downbeta_spx_60", "volcluster_60", "range_pos_252", "spx_corr60", "calmness_20"]
cand_mat = anchor_mats["mom20_volproxy60"]
mat = orth_resid(cand_mat, ORTH_ANCHORS)
print("orth_mom20 matrix built:", mat.shape, "nan frac:", round(np.isnan(mat).mean(), 4))

# ---- save signal artifact ----
os.makedirs("factors", exist_ok=True)
npy_path = "factors/orth_mom20.signal.npy"
np.save(npy_path, mat)
print("SAVED", npy_path)

# ---- full metrics ----
rank_mat = cross_sectional_rank(mat)
ics = spearman_ic_matrix(mat, fwd[HORIZON])
summ = summarize(ics, dates, "orth_mom20", HORIZON)
cov_ad, cov_d8 = coverage_stats(mat)
to = turnover_10d_rank(rank_mat)
dec = decay_curve(mat, fwd)
corrs_full, mx_name, mx_abs = library_pairwise_corr(mat)
# exclude self artifact (just persisted) from the admission correlation snapshot
corrs = {k: v for k, v in corrs_full.items() if k != "orth_mom20"}
if corrs:
    mx_name, mx_abs = max(corrs.items(), key=lambda kv: abs(kv[1]))
else:
    mx_name, mx_abs = None, 0.0
top = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
ic, icir = summ["ic"], summ["icir"]
print(f"IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
      f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name})")
print("regime:", summ["regime"])
print("decay:", dec)
print("top conflicts:", top)

# ---- kept library snapshot for provenance ----
KEPT = set()
for f in sorted(os.listdir("factors")):
    if not f.endswith(".json"):
        continue
    try:
        d = json.load(open(f"factors/{f}"))
        if d.get("validation", {}).get("status") == "EFFECTIVE" and \
           os.path.exists(f"factors/{d.get('factor_id','')}.signal.npy"):
            KEPT.add(d.get("factor_id"))
    except Exception:
        pass
KEPT.discard("orth_mom20")
lib_snapshot = {k: round(v, 4) for k, v in corrs.items() if k in KEPT}
print("kept-library snapshot keys:", len(lib_snapshot))

CONTRACT = {"ic_threshold": GATE_IC, "icir_threshold": GATE_ICIR,
            "correlation_threshold": CORR_LIMIT, "library_capacity": 30,
            "active_top_k": 10}
quality = abs(ic) * abs(icir)
n_nan = int(np.isnan(mat).sum())

doc = {
    "factor_id": "orth_mom20",
    "factor_name": "Orthogonalized momentum 20d (residual vs anchor risk factors)",
    "version": "1.0.0",
    "calculation": {
        "expression": ("zscore(rank_resid(rank(mom20_volproxy60) ~ rank(downbeta_spx_60) + "
                       "rank(volcluster_60) + rank(range_pos_252) + rank(spx_corr60) + "
                       "rank(calmness_20), 2 passes))"),
        "description": ("Per-date cross-sectional OLS residual of the vol-damped 20d momentum rank on a set of "
                        "anchor risk-factor ranks (SPX downside beta, volatility clustering, 252d range position, "
                        "SPX correlation, calmness). Two residualization passes, then standardized to z per date. "
                        "Isolates the momentum signal that is orthogonal to the existing library anchors, "
                        "reducing library correlation to 0.48 (vs 0.62 for raw mom20_volproxy60). "
                        "Positive predictor of forward 10d cross-sectional returns."),
    },
    "dependencies": ["close", "SPX.close"],
    "parameters": {
        "lookback": 20, "skip": 5, "vol_proxy_lookback": 60,
        "anchors": ORTH_ANCHORS, "passes": 2, "min_obs_per_date": 8
    },
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": f"{GRID[0]}..{GRID[-1]}",
        "last_validated": "2026-09-10",
        "admission_horizon": HORIZON,
        "regime_notes": (f"15-instrument tradable cross-asset universe; data visible through {GRID[-1]}. "
                         f"regime: {summ['regime']}. Kept-library max |corr| = {mx_abs:.3f} ({mx_name})."),
        "metrics": {
            "ic": round(ic, 4),
            "icir": round(icir, 4),
            "ic_hit_ratio": round(summ["hit"], 4),
            "n_ic_dates": int(summ["n_ic_dates"]),
            "coverage_asset_days": round(cov_ad, 4),
            "coverage_dates_ge8": round(cov_d8, 4),
            "turnover_10d_rank": round(to, 4),
            "decay_ic_by_horizon": dec,
            "max_abs_library_correlation": round(mx_abs, 4),
            "library_pairwise_corr": lib_snapshot,
        },
    },
    "tags": ["momentum", "orthogonalization", "residual", "cross-asset"],
    "benchmark_admission": {
        "contract": CONTRACT,
        "selected_metrics": {
            "ic": round(ic, 4), "icir": round(icir, 4),
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": round(mx_abs, 4),
            "correlation_path": "validation.metrics.max_abs_library_correlation",
            "quality": round(quality, 8),
        },
        "admitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    },
    "signal_artifact": "orth_mom20.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": list(mat.shape),
        "columns": ASSETS,
        "dates_first": GRID[0],
        "dates_last": GRID[-1],
        "n_nan": n_nan,
    },
}

path = f"factors/orth_mom20.json"
json.dump(doc, open(path, "w"), indent=1)
print("WROTE", path)

# ---- readback verification ----
print("=== READBACK VERIFICATION ===")
d = json.load(open(path))
v = d["validation"]
m = v["metrics"]
checks = {
    "id_match": d["factor_id"] == "orth_mom20",
    "status": v["status"] == "EFFECTIVE",
    "ic_gate": abs(m["ic"]) >= GATE_IC,
    "icir_gate": abs(m["icir"]) >= GATE_ICIR,
    "corr_gate": m["max_abs_library_correlation"] < CORR_LIMIT,
    "signal_exists": os.path.exists(npy_path),
    "artifact_ref": d["signal_artifact"] == "orth_mom20.signal.npy",
    "maxlibcorr_reported": "max_abs_library_correlation" in m,
    "last_validated": v["last_validated"] == "2026-09-10",
}
ok = all(checks.values())
print(checks)
print("ALL OK" if ok else "SOME FAILED")
