"""miner_1 cycle17 batchK: novel factor exploration.
Candidates (none previously persisted/explored in library):
  - variance_ratio_5x60 : classic variance-ratio trend/mean-reversion statistic
  - coskew_60           : market coskewness (loading on squared market return)
  - updown_beta_60      : up-market beta minus down-market beta (asymmetric beta)
  - vol_beta_60         : beta of asset 5d realized vol on market 5d realized vol
  - tw_mom_20           : linearly time-weighted momentum (recent-weighted)

Validation window: through 2026-07-30 (current date). No lookahead.
"""
import sys, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, validate_factor, max_library_corr,
                                   artifact_b64, print_result, ASSETS, IC_GATE,
                                   ICIR_GATE, CURRENT_DATE)

# ----------------------------------------------------------------------------
# factor functions (per-asset dense calendar; market proxy passed via macro dict)
# ----------------------------------------------------------------------------

def _market(macro):
    return macro["market"]


def variance_ratio_5x60(close, vol, open_, high, low, macro, k=5, window=60):
    r = close.pct_change()
    var1 = r.rolling(window, min_periods=max(20, window // 2)).var()
    rk = r.rolling(k).sum()
    vark = rk.rolling(window, min_periods=max(20, window // 2)).var()
    vr = vark / (k * var1)
    return vr.replace([np.inf, -np.inf], np.nan)


def coskew_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    mkt = _market(macro).reindex(r.index)
    df = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna().tail(window)
    if len(df) < 30:
        return pd.Series(np.nan, index=close.index)
    X = np.column_stack([np.ones(len(df)), df["m"].values, (df["m"] ** 2).values])
    y = df["r"].values
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=close.index)
    s = pd.Series(np.nan, index=close.index)
    s.iloc[-1] = coef[2]
    return s


def updown_beta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    mkt = _market(macro).reindex(r.index)
    df = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna().tail(window)

    def beta(sub):
        if len(sub) < 5:
            return np.nan
        v = float(sub["m"].var())
        if v < 1e-14:
            return np.nan
        return float(sub["r"].cov(sub["m"]) / v)

    bu = beta(df[df["m"] > 0])
    bd = beta(df[df["m"] < 0])
    s = pd.Series(np.nan, index=close.index)
    if np.isfinite(bu) and np.isfinite(bd):
        s.iloc[-1] = bu - bd
    return s


def vol_beta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    mkt = _market(macro).reindex(r.index)
    av = r.rolling(5).std()
    mv = mkt.rolling(5).std()
    df = pd.concat([av.rename("av"), mv.rename("mv")], axis=1).dropna().tail(window)
    s = pd.Series(np.nan, index=close.index)
    if len(df) >= 20:
        v = float(df["mv"].var())
        if v > 1e-14:
            s.iloc[-1] = float(df["av"].cov(df["mv"]) / v)
    return s


def tw_mom_20(close, vol, open_, high, low, macro, window=20):
    r = close.pct_change().tail(window)
    w = np.arange(1, window + 1, dtype=float)
    s = pd.Series(np.nan, index=close.index)
    if len(r) >= 10:
        s.iloc[-1] = float((r.values * w[:len(r)]) / w[:len(r)].sum())
    return s


# ----------------------------------------------------------------------------
# market proxy: cross-sectional mean daily return on the union panel
# ----------------------------------------------------------------------------
close, vol, open_, high, low = load_closes(CURRENT_DATE)
ret_panel = close.pct_change()
market = ret_panel.mean(axis=1, skipna=True)
macro = {"market": market}

CANDIDATES = {
    "variance_ratio_5x60": variance_ratio_5x60,
    "coskew_60": coskew_60,
    "updown_beta_60": updown_beta_60,
    "vol_beta_60": vol_beta_60,
    "tw_mom_20": tw_mom_20,
}

# ----------------------------------------------------------------------------
# library panels (active + informational from evicted)
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# run validation
# ----------------------------------------------------------------------------
results = {}
panels = {}
for name, fn in CANDIDATES.items():
    res = validate_factor(fn, close, vol, open_, high, low, macro,
                          horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10)
    results[name] = res
    panels[name] = res["panel"]
    print_result(name, res)

print("\n=== library correlation (active: usdcny_beta_60) ===")
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

print("\n=== regime IC (10d horizon) ===")
for name in CANDIDATES:
    ic_ser = results[name].get("ic_series")
    # recompute quickly from panel
    fr = close.pct_change(10).shift(-10)  # approximate: union-panel forward returns
    fr10 = fr
    regs = {}
    for label, lo, hi in [("2020-2021 COVID/recovery", "2020-01-01", "2021-12-31"),
                          ("2022-2023 tightening/AI", "2022-01-01", "2023-12-31"),
                          ("2024-2026-07 crypto/commodity", "2024-01-01", "2026-07-30")]:
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

# ----------------------------------------------------------------------------
# persistence for gate-passing candidates
# ----------------------------------------------------------------------------
print("\n=== PERSISTENCE ===")
import datetime
for name in CANDIDATES:
    res = results[name]
    ic, icir = res["ic"], res["icir"]
    gate_ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    rho_ok = res["rho_vs_active_lib"] < 0.5
    if gate_ok and rho_ok:
        fid = name
        doc = {
            "factor_id": fid,
            "factor_name": {
                "variance_ratio_5x60": "60d variance ratio (5d/1d) trend persistence",
                "coskew_60": "60d market coskewness (squared-market loading)",
                "updown_beta_60": "60d asymmetric beta (up-market beta - down-market beta)",
                "vol_beta_60": "60d volatility beta vs market vol",
                "tw_mom_20": "20d linearly time-weighted momentum",
            }[fid],
            "version": "1.0.0",
            "calculation": {
                "expression": {
                    "variance_ratio_5x60": "VR = Var(5d rolling sum of returns, 60d) / (5 * Var(1d returns, 60d))",
                    "coskew_60": "loading c in OLS r_t = a + b*mkt_t + c*mkt_t^2 over 60d",
                    "updown_beta_60": "beta(r,mkt | mkt>0) - beta(r,mkt | mkt<0) over 60d",
                    "vol_beta_60": "slope of regressing asset 5d realized vol on market 5d realized vol, 60d",
                    "tw_mom_20": "sum(w_i * r_i) / sum(w_i), w = 1..20 linear weights over 20d",
                }[fid],
                "description": {
                    "variance_ratio_5x60": "Variance ratio > 1 indicates trending, < 1 mean reversion. Classic Lo-MacKinlay statistic computed on dense per-asset calendar with market proxy from cross-sectional mean.",
                    "coskew_60": "Market coskewness: positive loading on squared market return implies upside convexity (lottery-like), negative implies crash sensitivity.",
                    "updown_beta_60": "Asymmetric beta: positive value means asset is more sensitive in up markets (defensive in down markets); negative means downside-heavy tail risk.",
                    "vol_beta_60": "Volatility beta: how strongly the asset's own realized volatility comoves with aggregate market volatility (systematic vol transmission).",
                    "tw_mom_20": "Recent-weighted momentum emphasizing the most recent daily returns; smoother than raw 10d momentum.",
                }[fid],
            },
            "dependencies": ["close"],
            "parameters": {"window": 60, "horizon": 10},
            "tags": ["trend", "cross-asset", "risk"] if fid in ("variance_ratio_5x60", "tw_mom_20") else ["beta", "risk", "cross-asset"],
            "expected_direction": 1 if ic > 0 else -1,
            "validation": {
                "status": "EFFECTIVE",
                "period": f"2020-01-01..{CURRENT_DATE.date()}",
                "last_validated": datetime.datetime.now().isoformat(timespec="seconds"),
                "admission_horizon": 10,
                "regime_notes": "mixed/corrective regime; HIGH cross-sectional dispersion, LOW cross-asset correlation; SPX resilient bull, HSI V-recovery, CSI300 sharp bear leg, tech-crypto correction, XAU correction, DXY weak, VIX low",
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
                    "library_correlation_detail": {
                        "usdcny_beta_60": res["rho_vs_active_lib"],
                    },
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

# save results
out = {name: {k: v for k, v in results[name].items() if k not in ("panel",)} for name in CANDIDATES}
json.dump(out, open("scripts/_miner1_cycle17_batchK_results.json", "w"), indent=1, default=str)
print("\ndone. results saved to scripts/_miner1_cycle17_batchK_results.json")
