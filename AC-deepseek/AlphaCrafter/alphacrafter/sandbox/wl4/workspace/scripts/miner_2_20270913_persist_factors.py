"""miner_2 2027-09-13: persist passing candidates (vol_ratio_20_60, volume_z_20)
as EFFECTIVE and deprecate eurusd_beta_60d (re-validation failed).
Validation data through 2027-09-10. Rank IC h=10, 15-asset universe.
Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
import json, base64, zlib, hashlib, time, os

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VISIBLE_THROUGH = "2027-09-10"
LAST_VALIDATED = "2027-09-13"
IC_TH, ICIR_TH = 0.0070, 0.0840
T0 = time.time()

def load_asset(a):
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].reset_index(drop=True)

def load_macro(m):
    df = pd.read_csv(f"../persistent/index_data/{m}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].reset_index(drop=True)

PX = {a: load_asset(a) for a in ASSETS}
CLOSE = {a: dict(zip(px["date"], px["close"])) for a, px in PX.items()}
VOL = {a: dict(zip(px["date"], px["volume"])) for a, px in PX.items()}

RETS = {}
for a in ASSETS:
    ds = sorted(CLOSE[a])
    r = {}
    for i in range(1, len(ds)):
        r[ds[i]] = CLOSE[a][ds[i]] / CLOSE[a][ds[i-1]] - 1.0
    RETS[a] = r

# asset trading-calendar index for forward-return alignment
CAL = {a: {d: i for i, d in enumerate(sorted(CLOSE[a]))} for a in ASSETS}

def f_vol_ratio_20_60(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(60, len(ds)):
        s20 = np.std([RETS[a].get(d, np.nan) for d in ds[i-20:i]])
        s60 = np.std([RETS[a].get(d, np.nan) for d in ds[i-60:i]])
        if np.isfinite(s60) and s60 > 0 and np.isfinite(s20):
            out[ds[i]] = s20 / s60
    return out

def f_volume_z_20(a):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(20, len(ds)):
        vs = np.array([VOL[a].get(d, np.nan) for d in ds[i-20:i]])
        if np.isfinite(vs).all() and np.std(vs) > 0 and np.mean(vs) > 0:
            out[ds[i]] = (VOL[a][ds[i]] - np.mean(vs)) / np.std(vs)
    return out

def f_rolling_beta(a, bench_ret, win=60):
    ds = sorted(CLOSE[a]); out = {}
    for i in range(win, len(ds)):
        seg = ds[i-win:i+1]
        rs = np.array([RETS[a].get(d, np.nan) for d in seg[1:]])
        bs = np.array([bench_ret.get(d, np.nan) for d in seg[1:]])
        m = np.isfinite(rs) & np.isfinite(bs)
        if m.sum() >= 30 and np.std(rs[m]) > 0 and np.std(bs[m]) > 0:
            out[ds[i]] = np.cov(rs[m], bs[m])[0, 1] / np.var(bs[m])
    return out

mkt_ret = {}
for d in sorted(set().union(*[set(RETS[a]) for a in ASSETS])):
    fv = [RETS[a].get(d, np.nan) for a in ASSETS]
    fv = [v for v in fv if np.isfinite(v)]
    if len(fv) >= 8:
        mkt_ret[d] = np.mean(fv)

eur_df = load_macro("EURUSD.csv") if False else load_macro("EURUSD")
eur_dates = sorted(eur_df["date"])
eur_ret = {eur_dates[i]: eur_df["close"].iloc[i] / eur_df["close"].iloc[i-1] - 1.0
           for i in range(1, len(eur_dates))}

factors = {
    "vol_ratio_20_60": {a: f_vol_ratio_20_60(a) for a in ASSETS},
    "volume_z_20": {a: f_volume_z_20(a) for a in ASSETS},
    "eurusd_beta_60d": {a: f_rolling_beta(a, eur_ret, 60) for a in ASSETS},
}
print(f"factors computed elapsed={time.time()-T0:.0f}s", flush=True)

# ---- vectorized IC machinery ----
def panel_frame(fvals):
    """DataFrame dates x assets with factor values."""
    all_dates = sorted(set().union(*[set(v) for v in fvals.values()]))
    return pd.DataFrame({a: pd.Series(fvals[a]) for a in ASSETS}).sort_index()

def ic_series_vec(fvals, h=10):
    """Rank IC per date using vectorized row-wise spearman."""
    rows = []
    base_dates = sorted(fvals[ASSETS[0]].keys()) if fvals[ASSETS[0]] else []
    for d in base_dates:
        xs, ys = [], []
        for a in ASSETS:
            fv = fvals[a].get(d)
            if fv is None or not np.isfinite(fv):
                continue
            j = CAL[a].get(d)
            if j is None:
                continue
            ds = sorted(CLOSE[a])
            k = j + h
            if k >= len(ds):
                continue
            xs.append(fv); ys.append(CLOSE[a][ds[k]] / CLOSE[a][d] - 1.0)
        if len(xs) >= 8:
            ic = spearmanr(xs, ys).correlation
            if np.isfinite(ic):
                rows.append((d, ic))
    return rows

def metrics(name, fvals, horizons=(1, 2, 3, 5, 10, 20)):
    decay = {}
    for h in horizons:
        rows = ic_series_vec(fvals, h)
        ics = [r[1] for r in rows]
        m = float(np.mean(ics)) if ics else 0.0
        s = float(np.std(ics)) if ics else 0.0
        decay[str(h)] = round(m, 4)
        if h == 10:
            full = dict(ic=m, icir=m / s if s > 0 else 0.0,
                        ic_hit_ratio=float(np.mean([1 if x > 0 else 0 for x in ics])),
                        n_ic_dates=len(ics), ic_std=s, rows=rows)
    # coverage: asset-day fraction over asset's own data window (>=2020-04)
    cov_cells = tot_cells = 0
    for a in ASSETS:
        ds = sorted(CLOSE[a])
        for d in ds:
            if d in fvals[a] and np.isfinite(fvals[a][d]):
                cov_cells += 1
            tot_cells += 1
    cov_ad = cov_cells / tot_cells if tot_cells else 0.0
    # coverage: fraction of dates with >=8 valid assets (over factor base dates)
    n_ge8 = 0
    base_dates = sorted(set().union(*[set(v) for v in fvals.values()]))
    for d in base_dates:
        nv = sum(1 for a in ASSETS if d in fvals[a] and np.isfinite(fvals[a][d]))
        if nv >= 8:
            n_ge8 += 1
    cov_d8 = n_ge8 / len(base_dates) if base_dates else 0.0
    # turnover: mean abs change of cross-sectional rank between dates ~10 td apart
    ds_sorted = sorted(set().union(*[set(v) for v in fvals.values()]))
    # use the asset-0 calendar to pick 10-td-apart pairs
    cal0 = [d for d in sorted(CLOSE[ASSETS[0]])]
    pos0 = {d: i for i, d in enumerate(cal0)}
    turns = []
    for i in range(0, len(cal0) - 10, 10):
        d1, d2 = cal0[i], cal0[i + 10]
        if d1 not in pos0 or d2 not in pos0:
            continue
        r1, r2 = [], []
        for a in ASSETS:
            v1, v2 = fvals[a].get(d1), fvals[a].get(d2)
            if v1 is not None and v2 is not None and np.isfinite(v1) and np.isfinite(v2):
                r1.append(v1); r2.append(v2)
        if len(r1) >= 8:
            rr1 = rankdata(r1); rr2 = rankdata(r2)
            turns.append(float(np.mean(np.abs(rr1 - rr2))))
    turnover = float(np.mean(turns)) if turns else np.nan
    full.update(dict(coverage_asset_days=round(cov_ad, 3),
                     coverage_dates_ge8=round(cov_d8, 3),
                     turnover_10d_rank=round(turnover, 3) if np.isfinite(turnover) else None,
                     decay_ic_by_horizon=decay))
    return full

def build_artifact(fvals):
    """base64:zlib:csv panel artifact (mirrors vol_price_corr_20 format)."""
    all_dates = sorted(set().union(*[set(v) for v in fvals.values()]))
    lines = ["," + ",".join(ASSETS)]
    n_valid = 0
    for d in all_dates:
        row = [str(d.strftime("%Y-%m-%d"))]
        for a in ASSETS:
            v = fvals[a].get(d)
            if v is not None and np.isfinite(v):
                row.append(f"{v:.8f}")
                n_valid += 1
            else:
                row.append("")
        lines.append(",".join(row))
    csvb = ("\n".join(lines)).encode()
    comp = zlib.compress(csvb)
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape ({len(all_dates)}, {len(ASSETS)})",
        "columns": ASSETS,
        "shape": [len(all_dates), len(ASSETS)],
        "n_valid_values": n_valid,
        "sha256": hashlib.sha256(comp).hexdigest()[:16],
        "data": base64.b64encode(comp).decode(),
    }

# ---- compute metrics ----
print("computing metrics...", flush=True)
M = {}
for name, fv in factors.items():
    m = metrics(name, fv)
    M[name] = m
    print(f"{name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} n={m['n_ic_dates']} "
          f"hit={m['ic_hit_ratio']:.2f} cov_ad={m['coverage_asset_days']} cov_d8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} decay={m['decay_ic_by_horizon']}", flush=True)
print(f"metrics done elapsed={time.time()-T0:.0f}s", flush=True)

CONTRACT = {"ic_threshold": IC_TH, "icir_threshold": ICIR_TH, "correlation_threshold": 0.5,
            "library_capacity": 30, "active_top_k": 10}

def write_factor(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {path}", flush=True)

def verify(path, expect_id, expect_status):
    d = json.load(open(path))
    assert d["factor_id"] == expect_id, f"id mismatch {d['factor_id']}"
    assert d["validation"]["status"] == expect_status, f"status {d['validation']['status']}"
    ic, icir = d["validation"]["metrics"]["ic"], d["validation"]["metrics"]["icir"]
    if expect_status == "EFFECTIVE":
        assert abs(ic) >= IC_TH and abs(icir) >= ICIR_TH, "gate not met in persisted file"
    sa = d["validation"]["signal_artifact"]
    comp = base64.b64decode(sa["data"])
    assert hashlib.sha256(comp).hexdigest()[:16] == sa["sha256"], "artifact sha mismatch"
    print(f"VERIFIED {path}: id={d['factor_id']} status={d['validation']['status']} "
          f"IC={ic:+.4f} ICIR={icir:+.4f} artifact={sa['shape']} valid={sa['n_valid_values']}", flush=True)
    return d

# ---- 1) vol_ratio_20_60 (EFFECTIVE) ----
m = M["vol_ratio_20_60"]
write_factor("factors/vol_ratio_20_60.json", {
    "factor_id": "vol_ratio_20_60",
    "factor_name": "20d/60d volatility ratio",
    "version": "1.0.0",
    "calculation": {
        "expression": "std(ret,20) / std(ret,60)",
        "description": "Ratio of 20-day return volatility to 60-day return volatility. Negative IC: assets whose short-term vol has compressed relative to their longer-term vol (calm after turbulence) earn higher forward 10d returns; recent 250d ICIR strongly negative confirms the low-vol-ratio signal."
    },
    "dependencies": ["close"],
    "parameters": {"short_win": 20, "long_win": 60, "min_obs": 60},
    "expected_direction": -1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2027-09-10",
        "last_validated": LAST_VALIDATED,
        "admission_horizon": 10,
        "regime_notes": "Validated 2020-01-01..2027-09-10 across regimes incl. 2020 COVID crash, 2021-22 tightening, 2023-24 AI rally, 2025-27 crypto/commodity cycles. Negative IC means low 20/60 vol ratio (recent calm) precedes outperformance. Recent 250 obs IC=-0.134 ICIR=-0.555: signal strengthened in the latest regime.",
        "metrics": {
            "ic": round(m["ic"], 4),
            "icir": round(m["icir"], 4),
            "ic_hit_ratio": round(m["ic_hit_ratio"], 3),
            "n_ic_dates": m["n_ic_dates"],
            "ic_std": round(m["ic_std"], 4),
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "turnover_10d_rank": m["turnover_10d_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": 0.140,
            "max_corr_factor": "vol_price_corr_20",
        },
        "signal_artifact": build_artifact(factors["vol_ratio_20_60"]),
    },
    "tags": ["volatility", "mean-reversion", "cross-asset"],
    "benchmark_admission": {
        "contract": CONTRACT,
        "selected_metrics": {
            "ic": round(m["ic"], 4),
            "icir": round(m["icir"], 4),
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": 0.140,
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
        "admitted_at": "2027-09-13T00:00:00",
    },
})

# ---- 2) volume_z_20 (EFFECTIVE) ----
m = M["volume_z_20"]
write_factor("factors/volume_z_20.json", {
    "factor_id": "volume_z_20",
    "factor_name": "20d volume z-score",
    "version": "1.0.0",
    "calculation": {
        "expression": "(volume - mean(volume,20)) / std(volume,20)",
        "description": "Z-score of current volume vs its trailing 20-day average. Positive IC: assets with abnormally high volume (relative to their own recent norm) earn higher forward 10d returns - volume-confirmed participation/interest. Very low library correlation (0.014)."
    },
    "dependencies": ["volume"],
    "parameters": {"win": 20, "min_obs": 20},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2027-09-10",
        "last_validated": LAST_VALIDATED,
        "admission_horizon": 10,
        "regime_notes": "Validated 2020-01-01..2027-09-10 across regimes (COVID crash, tightening bear, AI rally, crypto/commodity cycles). Consistent positive IC; recent 250 obs IC=+0.041 ICIR=+0.099 - stable. Coverage 1.00 over the 15-asset cross-section.",
        "metrics": {
            "ic": round(m["ic"], 4),
            "icir": round(m["icir"], 4),
            "ic_hit_ratio": round(m["ic_hit_ratio"], 3),
            "n_ic_dates": m["n_ic_dates"],
            "ic_std": round(m["ic_std"], 4),
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "turnover_10d_rank": m["turnover_10d_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": 0.014,
            "max_corr_factor": "rate_beta_cn10y_60d",
        },
        "signal_artifact": build_artifact(factors["volume_z_20"]),
    },
    "tags": ["volume", "liquidity", "cross-asset"],
    "benchmark_admission": {
        "contract": CONTRACT,
        "selected_metrics": {
            "ic": round(m["ic"], 4),
            "icir": round(m["icir"], 4),
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": 0.014,
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
        "admitted_at": "2027-09-13T00:00:00",
    },
})

# ---- 3) deprecate eurusd_beta_60d ----
m = M["eurusd_beta_60d"]
old = json.load(open("factors/eurusd_beta_60d.json"))
old["validation"]["status"] = "DEPRECATED"
old["validation"]["period"] = "2020-01-01..2027-09-10"
old["validation"]["last_validated"] = LAST_VALIDATED
old["validation"]["regime_notes"] = (
    "DEPRECATED 2027-09-13: re-validation through 2027-09-10 failed the admission gate - "
    f"full IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} (|ICIR| < 0.084), IC hit ratio {m['ic_hit_ratio']:.2f}. "
    "Predictive sign flipped positive in the recent 250 obs (IC=+0.022 ICIR=+0.085) vs the historical "
    "negative direction, i.e. the risk-appetite hedge no longer behaves as validated. "
    "Factor removed from the EFFECTIVE library; the ensemble still references it - Screener should re-tilt."
)
old["validation"]["metrics"].update({
    "ic": round(m["ic"], 4),
    "icir": round(m["icir"], 4),
    "ic_hit_ratio": round(m["ic_hit_ratio"], 3),
    "n_ic_dates": m["n_ic_dates"],
    "ic_std": round(m["ic_std"], 4),
    "coverage_asset_days": m["coverage_asset_days"],
    "coverage_dates_ge8": m["coverage_dates_ge8"],
    "turnover_10d_rank": m["turnover_10d_rank"],
    "decay_ic_by_horizon": m["decay_ic_by_horizon"],
})
old["validation"]["signal_artifact"] = build_artifact(factors["eurusd_beta_60d"])
old["deprecated_reason"] = (
    "re-validation failed: |ICIR| 0.038 < 0.084; recent 250-obs sign flipped positive (IC +0.022) "
    "against expected direction -1; drift since ~2027-08 validation."
)
write_factor("factors/eurusd_beta_60d_deprecated.json", old)
os.remove("factors/eurusd_beta_60d.json")
print("removed factors/eurusd_beta_60d.json (moved to _deprecated)", flush=True)

# ---- verify all ----
verify("factors/vol_ratio_20_60.json", "vol_ratio_20_60", "EFFECTIVE")
verify("factors/volume_z_20.json", "volume_z_20", "EFFECTIVE")
verify("factors/eurusd_beta_60d_deprecated.json", "eurusd_beta_60d", "DEPRECATED")
print(f"ALL PERSISTED AND VERIFIED elapsed={time.time()-T0:.0f}s", flush=True)
