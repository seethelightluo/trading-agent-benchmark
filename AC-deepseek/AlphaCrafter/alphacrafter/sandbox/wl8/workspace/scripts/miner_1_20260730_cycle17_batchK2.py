"""miner_1 cycle17 batchK2: fixed rolling implementations.
Candidates:
  - variance_ratio_5x60 / variance_ratio_10x60 : VR trend/mean-reversion stats
  - coskew_60     : rolling market coskewness (loading on squared market return)
  - updown_beta_60: rolling up-market beta - down-market beta
  - downside_beta_60: rolling beta on down-market days only
  - vol_beta_60   : rolling beta of asset 5d vol on market 5d vol
  - tw_mom_20     : linearly time-weighted momentum
  - disp_beta_60  : rolling beta of asset return on cross-sectional dispersion
"""
import sys, json, base64, zlib, io, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, validate_factor, max_library_corr,
                                   artifact_b64, print_result, ASSETS, IC_GATE,
                                   ICIR_GATE, CURRENT_DATE)

close, vol, open_, high, low = load_closes(CURRENT_DATE)
ret_panel = close.pct_change()
market = ret_panel.mean(axis=1, skipna=True)
disp = ret_panel.std(axis=1, skipna=True)
macro = {"market": market, "disp": disp}


def variance_ratio(close, vol, open_, high, low, macro, k=5, window=60):
    r = close.pct_change()
    var1 = r.rolling(window, min_periods=max(20, window // 2)).var()
    rk = r.rolling(k).sum()
    vark = rk.rolling(window, min_periods=max(20, window // 2)).var()
    vr = vark / (k * var1)
    return vr.replace([np.inf, -np.inf], np.nan)


def coskew_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["market"].reindex(r.index)
    m2 = m ** 2
    y = r
    var_x = m.rolling(window, min_periods=30).var()
    var_x2 = m2.rolling(window, min_periods=30).var()
    cov_yx = y.rolling(window, min_periods=30).cov(m)
    cov_yx2 = y.rolling(window, min_periods=30).cov(m2)
    cov_xx2 = m.rolling(window, min_periods=30).cov(m2)
    denom = var_x * var_x2 - cov_xx2 ** 2
    c = (cov_yx2 * var_x - cov_yx * cov_xx2) / denom
    return c.replace([np.inf, -np.inf], np.nan)


def _cond_beta(rr, m, mask, min_obs=5):
    if mask.sum() < min_obs:
        return np.nan
    mm = m[mask]
    rr2 = rr[mask]
    v = float(mm.var())
    if v < 1e-14:
        return np.nan
    return float(np.cov(rr2, mm)[0, 1] / v)


def updown_beta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["market"].reindex(r.index)
    df = pd.concat([r.rename("r"), m.rename("m")], axis=1)
    out = pd.Series(np.nan, index=df.index)
    arr = df.values
    for t in range(len(df)):
        lo = max(0, t - window + 1)
        a = arr[lo:t + 1]
        if len(a) < 30:
            continue
        mm = a[:, 1]
        rr = a[:, 0]
        bu = _cond_beta(rr, mm, mm > 0)
        bd = _cond_beta(rr, mm, mm < 0)
        if np.isfinite(bu) and np.isfinite(bd):
            out.iloc[t] = bu - bd
    return out


def downside_beta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["market"].reindex(r.index)
    df = pd.concat([r.rename("r"), m.rename("m")], axis=1)
    out = pd.Series(np.nan, index=df.index)
    arr = df.values
    for t in range(len(df)):
        lo = max(0, t - window + 1)
        a = arr[lo:t + 1]
        if len(a) < 30:
            continue
        mm = a[:, 1]
        rr = a[:, 0]
        out.iloc[t] = _cond_beta(rr, mm, mm < 0)
    return out


def vol_beta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["market"].reindex(r.index)
    av = r.rolling(5).std()
    mv = m.rolling(5).std()
    cov = av.rolling(window, min_periods=30).cov(mv)
    var = mv.rolling(window, min_periods=30).var()
    vb = cov / var
    return vb.replace([np.inf, -np.inf], np.nan)


def tw_mom_20(close, vol, open_, high, low, macro, window=20):
    r = close.pct_change()
    w = np.arange(1, window + 1, dtype=float)
    out = r.rolling(window, min_periods=10).apply(
        lambda x: float(np.dot(x, w) / w.sum()), raw=True)
    return out


def disp_beta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    d = macro["disp"].reindex(r.index)
    cov = r.rolling(window, min_periods=30).cov(d)
    var = d.rolling(window, min_periods=30).var()
    db = cov / var
    return db.replace([np.inf, -np.inf], np.nan)


CANDIDATES = {
    "variance_ratio_5x60": lambda *a, **k: variance_ratio(*a, **k, k=5, window=60),
    "variance_ratio_10x60": lambda *a, **k: variance_ratio(*a, **k, k=10, window=60),
    "coskew_60": coskew_60,
    "updown_beta_60": updown_beta_60,
    "downside_beta_60": downside_beta_60,
    "vol_beta_60": vol_beta_60,
    "tw_mom_20": tw_mom_20,
    "disp_beta_60": disp_beta_60,
}


def load_panel(path):
    d = json.load(open(path))
    art = d["validation"]["signal_artifact"]
    raw = base64.b64decode(art["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                    index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return p


lib_panels = {}
for path, fid in [("factors/usdcny_beta_60.json", "usdcny_beta_60")]:
    try:
        lib_panels[fid] = load_panel(path)
    except Exception as e:
        print(f"[warn] cannot load {path}: {e}")

info_panels = {}
for path, fid in [("factors/evicted/mom_10d_skip5.json", "mom_10d_skip5"),
                  ("factors/evicted/vix_beta_cond_60x20.json", "vix_beta_cond_60x20"),
                  ("factors/evicted/yield_beta_cond_60x20.json", "yield_beta_cond_60x20")]:
    try:
        info_panels[fid] = load_panel(path)
    except Exception as e:
        print(f"[warn] cannot load {path}: {e}")

results, panels = {}, {}
for name, fn in CANDIDATES.items():
    res = validate_factor(fn, close, vol, open_, high, low, macro,
                          horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10)
    results[name] = res
    panels[name] = res["panel"]
    print_result(name, res)

print("\n=== library correlation ===")
for name in CANDIDATES:
    rho_active = max_library_corr(panels[name], lib_panels)
    rho_info = max_library_corr(panels[name], info_panels)
    results[name]["rho_vs_active_lib"] = round(rho_active, 4)
    results[name]["rho_vs_evicted_info"] = round(rho_info, 4)
    print(f"  {name}: max_rho_active={rho_active:.4f}  max_rho_evicted_info={rho_info:.4f}")

print("\n=== candidate-candidate pairwise rho (pooled) ===")
names = list(CANDIDATES)
for i in range(len(names)):
    row = []
    for j in range(len(names)):
        a = panels[names[i]].values.ravel()
        b = panels[names[j]].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        row.append(round(float(np.corrcoef(a[m], b[m])[0, 1]), 3) if m.sum() > 200 else np.nan)
    print(f"  {names[i]:24s} {row}")

print("\n=== regime IC (10d horizon, union-panel approx) ===")
fr10 = close.pct_change(10).shift(-10)
for name in CANDIDATES:
    regs = {}
    for label, lo, hi in [("2020-2021", "2020-01-01", "2021-12-31"),
                          ("2022-2023", "2022-01-01", "2023-12-31"),
                          ("2024-2026-07", "2024-01-01", "2026-07-30")]:
        sub = panels[name].loc[lo:hi]
        frs = fr10.loc[lo:hi]
        ics = []
        for dt in sub.index:
            x, y = sub.loc[dt], frs.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() >= 8:
                ics.append(x[m].rank().corr(y[m].rank()))
        if ics:
            s = pd.Series(ics)
            regs[label] = [round(float(s.mean()), 4), round(float(s.mean() / s.std()), 4), len(s)]
    print(f"  {name}: {regs}")

print("\n=== PERSISTENCE ===")
for name in CANDIDATES:
    res = results[name]
    ic, icir = res["ic"], res["icir"]
    gate_ok = np.isfinite(ic) and np.isfinite(icir) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    rho_ok = res["rho_vs_active_lib"] < 0.5
    if gate_ok and rho_ok:
        fid = name
        doc = {
            "factor_id": fid,
            "factor_name": {
                "variance_ratio_5x60": "60d variance ratio (5d/1d) trend persistence",
                "variance_ratio_10x60": "60d variance ratio (10d/1d) trend persistence",
                "coskew_60": "60d market coskewness (squared-market loading)",
                "updown_beta_60": "60d asymmetric beta (up-market beta - down-market beta)",
                "downside_beta_60": "60d downside beta (beta on down-market days)",
                "vol_beta_60": "60d volatility beta vs market vol",
                "tw_mom_20": "20d linearly time-weighted momentum",
                "disp_beta_60": "60d beta of returns to cross-sectional dispersion",
            }[fid],
            "version": "1.0.0",
            "calculation": {
                "expression": {
                    "variance_ratio_5x60": "VR = Var(5d rolling sum of returns, 60d) / (5 * Var(1d returns, 60d))",
                    "variance_ratio_10x60": "VR = Var(10d rolling sum of returns, 60d) / (10 * Var(1d returns, 60d))",
                    "coskew_60": "rolling c in OLS r_t = a + b*mkt_t + c*mkt_t^2 over 60d window",
                    "updown_beta_60": "rolling [beta(r,mkt | mkt>0) - beta(r,mkt | mkt<0)] over 60d",
                    "downside_beta_60": "rolling beta(r,mkt | mkt<0) over 60d",
                    "vol_beta_60": "rolling slope of asset 5d realized vol on market 5d realized vol, 60d",
                    "tw_mom_20": "sum(w_i*r_i)/sum(w_i), w=1..20 linear weights over 20d",
                    "disp_beta_60": "rolling beta of asset returns on cross-sectional dispersion, 60d",
                }[fid],
                "description": "Cross-asset risk/trend structure factor on 15-asset tradable universe; market proxy = cross-sectional mean return, dispersion = cross-sectional std.",
            },
            "dependencies": ["close"],
            "parameters": {"window": 60, "horizon": 10},
            "tags": ["trend", "risk", "cross-asset"],
            "expected_direction": 1 if ic > 0 else -1,
            "validation": {
                "status": "EFFECTIVE",
                "period": f"2020-01-01..{CURRENT_DATE.date()}",
                "last_validated": datetime.datetime.now().isoformat(timespec="seconds"),
                "admission_horizon": 10,
                "regime_notes": "mixed/corrective; high dispersion, low cross-asset correlation; SPX bull, HSI V-recovery, CSI300 bear, tech-crypto correction, DXY weak",
                "metrics": {
                    "ic": round(ic, 4),
                    "icir": round(icir, 4),
                    "ic_hit_ratio": round(res["ic_hit_ratio"], 4),
                    "n_ic_dates": int(res["n_ic_dates"]),
                    "coverage_asset_days": res["coverage_asset_days"],
                    "coverage_dates_ge8": res["coverage_dates_ge8"],
                    "turnover_10d_rank": res["turnover_10d_rank"],
                    "decay_ic_by_horizon": res["decay_ic_by_horizon"],
                    "max_abs_library_correlation": res["rho_vs_active_lib"],
                    "library_correlation_detail": {"usdcny_beta_60": res["rho_vs_active_lib"]},
                },
                "signal_artifact": {
                    "format": "base64:zlib:csv",
                    "descrip": "factor value panel rows=date cols=asset (15-asset cross-asset universe)",
                    "data": artifact_b64(panels[name]),
                },
            },
            "benchmark_admission": {
                "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                             "correlation_threshold": 0.5, "library_capacity": 30,
                             "active_top_k": 10},
                "selected_metrics": {"ic": round(ic, 4), "icir": round(icir, 4),
                                     "metric_path": "validation.metrics",
                                     "reported_max_abs_library_correlation": round(res["rho_vs_active_lib"], 4),
                                     "quality": round(abs(ic) * abs(icir), 8)},
                "admitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        }
        path = f"factors/{fid}.json"
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1, allow_nan=True)
        print(f"  PERSISTED {path}")
    else:
        print(f"  skip {name}: gate_ok={gate_ok} rho_ok={rho_ok} (ic={ic:.4f} icir={icir:.4f} rho={res['rho_vs_active_lib']:.4f})")

out = {name: {k: v for k, v in results[name].items() if k != "panel"} for name in CANDIDATES}
json.dump(out, open("scripts/_miner1_cycle17_batchK2_results.json", "w"), indent=1, default=str)
print("\ndone.")
