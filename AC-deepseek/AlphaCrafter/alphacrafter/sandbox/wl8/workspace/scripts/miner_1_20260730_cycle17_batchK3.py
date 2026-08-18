"""
miner_1 cycle17 batchK3: follow-up on near-miss candidates from K2.
Fixes:
  - tw_mom_20: rolling apply now truncates weights to window length (previous bug -> NaN)
Candidates:
  - tw_mom_20 (fixed), tw_mom_10, tw_mom_40
  - vol_beta_40, vol_beta_90 (longer window -> lower turnover -> higher ICIR)
  - downside_beta_40, downside_beta_90
  - vol_beta_60_cond_disp: vol_beta_60 only when cross-sectional dispersion is elevated
Validation window: through 2026-07-30. No lookahead.
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


def tw_mom(close, vol, open_, high, low, macro, window=20):
    r = close.pct_change()
    w = np.arange(1, window + 1, dtype=float)
    out = r.rolling(window, min_periods=window).apply(
        lambda x: float(np.dot(x, w[:len(x)]) / w[:len(x)].sum()), raw=True)
    return out


def vol_beta(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["market"].reindex(r.index)
    av = r.rolling(5).std()
    mv = m.rolling(5).std()
    cov = av.rolling(window, min_periods=max(20, window // 2)).cov(mv)
    var = mv.rolling(window, min_periods=max(20, window // 2)).var()
    vb = cov / var
    return vb.replace([np.inf, -np.inf], np.nan)


def _cond_beta(rr, mm, mask, min_obs=5):
    if mask.sum() < min_obs:
        return np.nan
    mm2 = mm[mask]
    rr2 = rr[mask]
    v = float(mm2.var())
    if v < 1e-14:
        return np.nan
    return float(np.cov(rr2, mm2)[0, 1] / v)


def downside_beta(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["market"].reindex(r.index)
    df = pd.concat([r.rename("r"), m.rename("m")], axis=1)
    out = pd.Series(np.nan, index=df.index)
    arr = df.values
    for t in range(len(df)):
        lo = max(0, t - window + 1)
        a = arr[lo:t + 1]
        if len(a) < max(20, window // 2):
            continue
        mm = a[:, 1]
        rr = a[:, 0]
        out.iloc[t] = _cond_beta(rr, mm, mm < 0)
    return out


def vol_beta_cond_disp(close, vol, open_, high, low, macro, window=60):
    """vol_beta_60 but only defined when dispersion is above its trailing median
    (regime-conditional: systematic vol transmission matters during dispersion spikes)."""
    vb = vol_beta(close, vol, open_, high, low, macro, window=window)
    d = macro["disp"].reindex(vb.index)
    med = d.rolling(120, min_periods=60).median()
    cond = (d > med).reindex(vb.index)
    return vb.where(cond)


CANDIDATES = {
    "tw_mom_20": lambda *a, **k: tw_mom(*a, **k, window=20),
    "tw_mom_10": lambda *a, **k: tw_mom(*a, **k, window=10),
    "tw_mom_40": lambda *a, **k: tw_mom(*a, **k, window=40),
    "vol_beta_40": lambda *a, **k: vol_beta(*a, **k, window=40),
    "vol_beta_90": lambda *a, **k: vol_beta(*a, **k, window=90),
    "downside_beta_40": lambda *a, **k: downside_beta(*a, **k, window=40),
    "downside_beta_90": lambda *a, **k: downside_beta(*a, **k, window=90),
    "vol_beta_60_cond_disp": vol_beta_cond_disp,
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
                "tw_mom_20": "20d linearly time-weighted momentum",
                "tw_mom_10": "10d linearly time-weighted momentum",
                "tw_mom_40": "40d linearly time-weighted momentum",
                "vol_beta_40": "40d volatility beta vs market vol",
                "vol_beta_90": "90d volatility beta vs market vol",
                "downside_beta_40": "40d downside beta",
                "downside_beta_90": "90d downside beta",
                "vol_beta_60_cond_disp": "60d volatility beta conditioned on elevated dispersion",
            }[fid],
            "version": "1.0.0",
            "calculation": {
                "expression": {
                    "tw_mom_20": "sum(w_i*r_i)/sum(w_i), w=1..20 over 20d",
                    "tw_mom_10": "sum(w_i*r_i)/sum(w_i), w=1..10 over 10d",
                    "tw_mom_40": "sum(w_i*r_i)/sum(w_i), w=1..40 over 40d",
                    "vol_beta_40": "rolling slope of asset 5d realized vol on market 5d realized vol, 40d",
                    "vol_beta_90": "rolling slope of asset 5d realized vol on market 5d realized vol, 90d",
                    "downside_beta_40": "rolling beta(r,mkt | mkt<0) over 40d",
                    "downside_beta_90": "rolling beta(r,mkt | mkt<0) over 90d",
                    "vol_beta_60_cond_disp": "vol_beta_60 defined only when 120d rolling median of cross-sectional dispersion is exceeded",
                }[fid],
                "description": "Cross-asset risk/trend structure factor on 15-asset tradable universe; market proxy = cross-sectional mean return, dispersion = cross-sectional std.",
            },
            "dependencies": ["close"],
            "parameters": {"horizon": 10},
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
json.dump(out, open("scripts/_miner1_cycle17_batchK3_results.json", "w"), indent=1, default=str)
print("\ndone.")