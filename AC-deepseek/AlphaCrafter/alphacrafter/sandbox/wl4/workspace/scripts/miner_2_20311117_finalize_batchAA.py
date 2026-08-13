"""miner_2 batch AA finalize - full validation + library correlation for the 7 IC/ICIR passers.

Computes, for each passing candidate from batchAA v4:
  - full metrics at h=10 (IC, ICIR, hit, n, coverage, turnover, decay)
  - max_abs_library_correlation vs the CURRENT live library (3 effective factors)
  - recent-window IC (63/126/252/504 trading days)
Persists to factors/<factor_id>.json ONLY if IC/ICIR gates pass AND
max_abs_library_correlation vs live < 0.5 (else they would be evicted by the
deterministic gate as lower-quality conflicts).
"""
import sys, time, json, zlib, base64, hashlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, max_library_corr)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
ASOF = str(closes.index.max().date())
print(f"[t0] closes {closes.shape} | {closes.index.min().date()}..{ASOF} | {time.time()-t0:.1f}s", flush=True)


def align(series, idx):
    return series.reindex(idx).ffill()


vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)

H = 10
fwd = forward_returns(closes, H)
GATES = {"abs_ic": 0.0070, "abs_icir": 0.0840, "min_valid": 8, "h": H}
RHO_CAP = 0.5


def rolling_beta(y, x, win=60, min_obs=40):
    xr = x.reindex(y.index)
    xy = y.mul(xr, axis=0)
    x2 = xr.pow(2)
    my = y.rolling(win).mean()
    mx = xr.rolling(win).mean()
    mxy = xy.rolling(win).mean()
    mx2 = x2.rolling(win).mean()
    cov = mxy - my.mul(mx, axis=0)
    var = mx2 - mx.pow(2)
    beta = cov.div(var.replace(0, np.nan), axis=0)
    cnt = xr.notna().rolling(win).sum()
    return beta.where(cnt >= min_obs)


vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

CAND = {}
CAND["hl_pos_20d"] = ((closes - closes.rolling(20).min()) /
                      (closes.rolling(20).max() - closes.rolling(20).min() + 1e-12))
CAND["downside_ratio_60d"] = pd.DataFrame(
    {a: rets[a].clip(upper=0).rolling(60).std() / (rets[a].rolling(60).std() + 1e-12)
     for a in rets.columns}, index=rets.index)
CAND["kurt_60d"] = rets.rolling(60).kurt()
CAND["corr_asset_mkt_20"] = rets.rolling(20).corr(mkt_ret)
CAND["corr_asset_mkt_60"] = rets.rolling(60).corr(mkt_ret)
CAND["max_dd_60d"] = (closes - closes.rolling(60).max()) / closes.rolling(60).max()
CAND["mom60_skip5_voladj"] = (closes.shift(5) / closes.shift(65) - 1) / vol60

# ---- live library (current effective factors, recomputed from stored defs) ----
LIVE = {}
LIVE["vol_adj_mom_accel_20x60"] = ((closes / closes.shift(20) - 1 - (closes / closes.shift(60) - 1))
                                   / rets.rolling(20).std())
LIVE["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
LIVE["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

summary = {}
to_persist = {}
for name, sig in CAND.items():
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=1)
    s.update(coverage_metrics(sig))
    s["turnover_10d_rank"] = turnover_rank(sig, 10)
    s["decay_ic_by_horizon"] = decay_profile(sig, closes, (1, 3, 5, 10, 20), 8, 1)
    rho, key = max_library_corr(sig, LIVE)
    s["max_abs_library_correlation"] = rho
    s["max_corr_factor"] = key
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics.iloc[-w:]
        if len(sub) > 2 and sub.std(ddof=1) > 0:
            recent[f"ic_r{w}"] = round(float(sub.mean()), 4)
            recent[f"icir_r{w}"] = round(float(sub.mean() / sub.std(ddof=1)), 3)
        else:
            recent[f"ic_r{w}"] = None
            recent[f"icir_r{w}"] = None
    s.update(recent)
    ic_pass = abs(s["ic"]) >= GATES["abs_ic"]
    icir_pass = abs(s["icir"]) >= GATES["abs_icir"]
    rho_pass = rho < RHO_CAP
    status = "PERSIST" if (ic_pass and icir_pass and rho_pass) else \
             ("RHO-CONFLICT" if (ic_pass and icir_pass and not rho_pass) else "FAIL")
    summary[name] = {"ic": s["ic"], "icir": s["icir"], "hit": s["ic_hit_ratio"],
                     "n": s["n_ic_dates"], "rho": rho, "rho_key": key,
                     "cov8": s["coverage_dates_ge8"], "to": s["turnover_10d_rank"],
                     "r63": recent["ic_r63"], "r252": recent["ic_r252"],
                     "decay10": s["decay_ic_by_horizon"]["10"], "status": status}
    print(f"[{time.time()-t0:5.0f}s] {name:22s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} "
          f"rho={rho:.3f}({key}) cov8={s['coverage_dates_ge8']:.2f} to={s['turnover_10d_rank']:.2f} "
          f"r63={recent['ic_r63']} r252={recent['ic_r252']} => {status}", flush=True)
    if status == "PERSIST":
        to_persist[name] = (s, ics, sig)

print("\n=== PERSISTING ===", list(to_persist.keys()), flush=True)

with open("scripts/_miner2_20311117_batchAA_final_summary.json", "w") as fh:
    json.dump({"asof": ASOF, "gates": GATES, "rho_cap": RHO_CAP, "summary": summary}, fh, indent=1, default=str)
print("summary saved", flush=True)


def build_artifact(sig: pd.DataFrame):
    csv_bytes = sig.round(8).to_csv().encode("utf-8")
    comp = zlib.compress(csv_bytes, 6)
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates, cols = assets.",
        "shape": [int(sig.shape[0]), int(sig.shape[1])],
        "columns": list(sig.columns),
        "n_valid_values": int(sig.notna().sum().sum()),
        "sha256": hashlib.sha256(comp).hexdigest()[:16],
        "data": base64.b64encode(comp).decode("ascii"),
    }


def make_factor_json(fid, fname, expr, desc, params, direction, s, sig):
    doc = {
        "factor_id": fid,
        "factor_name": fname,
        "version": "1.0.0",
        "calculation": {"expression": expr, "description": desc},
        "dependencies": ["close"],
        "parameters": params,
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": f"2020-01-01..{ASOF}",
            "last_validated": "2031-11-17",
            "admission_horizon": H,
            "regime_notes": (
                f"Validated 2020-01-01..{ASOF} on the 15-asset cross-asset universe "
                f"(min_valid=8, h=10) spanning COVID 2020, 2021-22 tightening, 2023-24 equity rally, "
                f"2024-26 crypto/commodity cycles, 2027-31 tape incl. risk-off windows. "
                f"n_ic_dates={s['n_ic_dates']}; direction sign: {'positive' if direction == 1 else 'negative'} IC."
            ),
            "metrics": {k: v for k, v in s.items()},
            "signal_artifact": build_artifact(sig),
        },
        "tags": ["cross_asset", "risk", "technical"],
        "benchmark_admission": {
            "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                         "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
            "selected_metrics": {"ic": s["ic"], "icir": s["icir"],
                                 "metric_path": "validation.metrics",
                                 "reported_max_abs_library_correlation": s["max_abs_library_correlation"],
                                 "correlation_path": "validation.metrics.max_abs_library_correlation",
                                 "quality": round(float(abs(s["ic"]) * abs(s["icir"])), 10)},
        },
    }
    return doc


META = {
    "hl_pos_20d": ("High-low position 20d", "(close - rolling_min(close,20)) / (rolling_max(close,20) - rolling_min(close,20))",
                   "Position of close within its 20d range; high position -> mean-reversion (negative IC): assets near range highs tend to underperform over h=10.",
                   {"window": 20}, -1),
    "downside_ratio_60d": ("Downside volatility ratio 60d", "rolling_std(min(ret,0),60) / rolling_std(ret,60)",
                           "Share of total 60d vol attributable to downside moves. Positive IC: higher downside-vol share -> higher forward return (risk premium / reversal).",
                           {"window": 60}, 1),
    "kurt_60d": ("Return excess kurtosis 60d", "rolling_kurtosis(ret,60)",
                 "Excess kurtosis of 60d daily returns (fat tails). Positive IC: fatter-tailed assets earn higher forward returns (tail-risk premium).",
                 {"window": 60}, 1),
    "corr_asset_mkt_20": ("Asset-market correlation 20d", "rolling_corr(ret, mean(ret_cross_section), 20)",
                          "20d correlation of each asset's daily returns with the equal-weight cross-asset market return. Positive IC: high-correlation assets outperform (market-timing tilt).",
                          {"window": 20}, 1),
    "corr_asset_mkt_60": ("Asset-market correlation 60d", "rolling_corr(ret, mean(ret_cross_section), 60)",
                          "60d correlation of each asset's daily returns with the equal-weight cross-asset market return. Positive IC; slower variant.",
                          {"window": 60}, 1),
    "max_dd_60d": ("Max drawdown 60d", "(close - rolling_max(close,60)) / rolling_max(close,60)",
                   "Distance below the 60d closing high (negative when in drawdown). Negative IC: assets furthest below their highs rebound (contrarian), consistent with mean reversion.",
                   {"window": 60}, -1),
    "mom60_skip5_voladj": ("Vol-adjusted 60d momentum (skip 5)", "(close.shift(5)/close.shift(65) - 1) / rolling_std(ret,60)",
                           "60d momentum measured with 5d skip, scaled by 60d vol. Strong negative IC: recent 60d winners (per unit risk) revert at h=10.",
                           {"windows": {"mom": 60, "skip": 5, "vol": 60}}, -1),
}

for fid, (s, ics, sig) in to_persist.items():
    fname, expr, desc, params, direction = META[fid]
    doc = make_factor_json(fid, fname, expr, desc, params, direction, s, sig)
    path = f"factors/{fid}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh)
    # read-back verification
    chk = json.load(open(path))
    ok = (chk["factor_id"] == fid and chk["validation"]["status"] == "EFFECTIVE"
          and chk["validation"]["metrics"]["ic"] == s["ic"]
          and chk["validation"]["metrics"]["icir"] == s["icir"]
          and "data" in chk["validation"]["signal_artifact"]
          and chk["validation"]["signal_artifact"]["format"] == "base64:zlib:csv")
    print(f"persisted {path} | readback_ok={ok} | size={len(open(path,'rb').read())}B", flush=True)

print(f"done {time.time()-t0:.1f}s", flush=True)
