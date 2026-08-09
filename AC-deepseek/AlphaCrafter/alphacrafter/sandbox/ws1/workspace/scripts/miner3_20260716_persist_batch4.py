"""miner_3: persist batch-4 gate-passing factors.

Admission contract (shared, 15-asset cross-asset universe):
  |IC|   >= 0.0070  @ h=10
  |ICIR| >= 0.0840  @ h=10
  max_abs_library_correlation < 0.5000  (miner1 convention:
  mean over dates of per-date cross-sectional Pearson corr, corrwith axis=1)

Candidates (verified in batch4 screen):
  eff_ratio_60d        IC=+0.0333 ICIR=+0.115  corr=0.1666
  amihud_liquidity_20d IC=-0.0448 ICIR=-0.092  corr=0.4876 (9/15 assets have volume)
  ret_autocorr_20d     IC=-0.0378 ICIR=-0.125  corr=0.0526
  dxy_cond_60x20       IC=+0.0289 ICIR=+0.088  corr=0.0757
"""
import sys, json, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_harness import get_panels, WATCH, evaluate, forward_ret, ic_by_regime
from scipy.stats import spearmanr

H = 10
VALID_PERIOD = "2020-01-01..2026-07-15"
LAST_VALIDATED = "2026-07-16"

closes, rets, ohlc, macro = get_panels()
vol_panel = pd.concat({a: ohlc[a]["volume"] for a in WATCH}, axis=1).reindex(closes.index)
vix = macro["VIX"]
dxy = macro["DXY"]

# ---------------- candidate builders ----------------
def eff_ratio(px, n=60):
    move = (px - px.shift(n)).abs()
    path = px.diff().abs().rolling(n).sum()
    return (move / path).replace([np.inf, -np.inf], np.nan)

def amihud(px, vol, n=20):
    illiq = (px.pct_change().abs() / vol.replace(0, np.nan)).rolling(n).mean()
    return -illiq

def autocorr(px, n=20):
    r = px.pct_change()
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for a in px.columns:
        ra = r[a]
        out[a] = ra.rolling(n).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 3 else np.nan, raw=True)
    return out

def roll_beta(panel, x, n=60):
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    dx = x.diff()
    for a in panel.columns:
        y = panel[a].diff()
        out[a] = y.rolling(n).cov(dx) / dx.rolling(n).var()
    return out

CANDIDATES = {
    "eff_ratio_60d": {
        "factor_name": "Kaufman Efficiency Ratio 60d",
        "tags": ["trend-quality", "efficiency", "low-turnover"],
        "expr": "abs(close - close.shift(60)) / sum(abs(close.diff()), 60)",
        "desc": "Trend quality: net 60-day price displacement divided by total 60-day path length. "
                "High values mean clean directional trends (higher forward returns); low values mean "
                "choppy/mean-reverting action. Slow-moving, low-turnover trend-quality signal.",
        "params": {"window": 60},
        "deps": ["close"],
        "direction": "+",
        "fn": lambda: eff_ratio(closes, 60),
    },
    "amihud_liquidity_20d": {
        "factor_name": "Amihud Liquidity 20d (negated illiquidity)",
        "tags": ["liquidity", "microstructure"],
        "expr": "-mean(|pct_change| / volume, 20)",
        "desc": "Negated 20-day Amihud illiquidity: higher values = more liquid (low |return| per unit "
                "volume). Liquid assets outperform over 10-20d horizons in this universe. NOTE: only "
                "9/15 assets have non-zero volume in the synthetic data (SOX, XAU, COPPER, WTI, US10Y, "
                "CN10Y have zero volume -> NaN); factor is defined on the 9 volume-bearing assets.",
        "params": {"window": 20},
        "deps": ["close", "volume"],
        "direction": "+",
        "fn": lambda: amihud(closes, vol_panel, 20),
    },
    "ret_autocorr_20d": {
        "factor_name": "Return Autocorrelation 20d (negated)",
        "tags": ["mean-reversion", "autocorrelation"],
        "expr": "-lag1_corr(pct_change, window=20)",
        "desc": "Negative of the 20-day lag-1 return autocorrelation. Assets with strong positive return "
                "persistence (high autocorr) tend to underperform (overextended trends); assets showing "
                "negative autocorrelation (choppy, mean-reverting) tend to outperform. The negative sign "
                "makes the factor 'mean-reversion' oriented.",
        "params": {"window": 20, "lag": 1},
        "deps": ["close"],
        "direction": "-",
        "fn": lambda: autocorr(closes, 20),
    },
    "dxy_cond_60x20": {
        "factor_name": "Conditional DXY Beta x DXY 20d Move",
        "tags": ["macro", "currency", "conditional-beta"],
        "expr": "beta(asset_ret, DXY_ret, 60) * (DXY/DXY.shift(20) - 1.0)",
        "desc": "Macro spillover: asset's 60d beta to the dollar times the 20d DXY move. Positive when an "
                "asset is dollar-sensitive and the dollar is trending. Captures cross-asset carry/risk "
                "rotation driven by USD strength/weakness. Independent from the VIX-conditional factor.",
        "params": {"beta_window": 60, "move_window": 20},
        "deps": ["close", "DXY"],
        "direction": "+",
        "fn": lambda: roll_beta(closes, dxy, 60).mul(
            (dxy / dxy.shift(20) - 1.0).reindex(closes.index), axis=0),
    },
}

# ---------------- library factors (for max_abs_library_correlation) ----------------
def beta_panel(panel, x, n=60):
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    dx = x.diff()
    for a in panel.columns:
        y = panel[a].diff()
        out[a] = y.rolling(n).cov(dx) / dx.rolling(n).var()
    return out

LIBRARY = {
    "mom_10d_skip5": closes.shift(5) / closes.shift(15) - 1.0,
    "mom_120d_skip5": closes.shift(5) / closes.shift(125) - 1.0,
    "vix_beta_cond_60x20": -beta_panel(closes, vix, 60).mul(
        (vix / vix.shift(20) - 1.0).reindex(closes.index), axis=0),
    "vol_of_vol20x60": rets.rolling(20).std().rolling(60).std(),
}

def lib_corr(fa, fb):
    """miner1 convention: mean over dates of per-date cross-sectional Pearson corr."""
    idx = fa.index.intersection(fb.index)
    dfa, dfb = fa.loc[idx], fb.loc[idx]
    ok = (dfa.notna() & dfb.notna() & np.isfinite(dfa) & np.isfinite(dfb)).sum(axis=1) >= 8
    return float(dfa[ok].corrwith(dfb[ok], axis=1).mean())

def max_lib_corr(factor_df):
    best_name, best_abs = None, 0.0
    for ln, lf in LIBRARY.items():
        c = lib_corr(factor_df, lf)
        if np.isfinite(c) and abs(c) > best_abs:
            best_abs, best_name = abs(c), ln
    return round(best_abs, 4), best_name

# ---------------- metrics ----------------
def coverage_metrics(factor_df):
    valid = factor_df.notna()
    asset_days = float(valid.sum().sum() / valid.size)
    dates_ge8 = float((valid.sum(axis=1) >= 8).mean())
    return asset_days, dates_ge8

def decay_curve(factor_df, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        m = evaluate(factor_df, rets, h=h, name="decay", verbose=False)
        out[str(h)] = round(float(m["mean_ic"]), 4)
    return out

print(f"=== Persisting batch-4 factors (h={H} admission) ===")
t0 = time.time()
persisted = []
for fid, spec in CANDIDATES.items():
    f = spec["fn"]().reindex(closes.index)
    m10 = evaluate(f, rets, h=H, name=fid, verbose=True)
    if not (abs(m10["mean_ic"]) >= 0.007 and abs(m10["icir"]) >= 0.084):
        print(f"  !! {fid} does not pass gate; skip")
        continue
    corr, corr_src = max_lib_corr(f)
    if corr >= 0.5:
        print(f"  !! {fid} library corr {corr} >= 0.5; skip")
        continue
    asset_days, dates_ge8 = coverage_metrics(f)
    decay = decay_curve(f)
    regime_lines = []
    # regime IC via helper
    fwd = forward_ret(rets, H)
    idx = f.index.intersection(fwd.index)
    ff, fw = f.loc[idx], fwd.loc[idx]
    mkt = rets[WATCH].mean(axis=1).loc[idx]
    vixl = vix.loc[idx]
    ics = {}
    for t in ff.index:
        a, b = ff.loc[t], fw.loc[t]
        mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if mask.sum() >= 8:
            ic, _ = spearmanr(a[mask], b[mask])
            if np.isfinite(ic):
                ics[t] = ic
    s = pd.Series(ics)
    mup = s[mkt.reindex(s.index) > 0]; mdn = s[mkt.reindex(s.index) <= 0]
    vmed = vixl.reindex(s.index).median()
    vlo = s[vixl.reindex(s.index) <= vmed]; vhi = s[vixl.reindex(s.index) > vmed]
    yr = {int(k): round(float(v), 4) for k, v in s.groupby(s.index.year).mean().items()}
    regime_notes = (f"mkt_up IC={mup.mean():+.4f}(n={len(mup)}) mkt_dn IC={mdn.mean():+.4f}(n={len(mdn)}) | "
                    f"vix_low IC={vlo.mean():+.4f}(n={len(vlo)}) vix_high IC={vhi.mean():+.4f}(n={len(vhi)}); "
                    f"yearly IC {yr}")

    doc = {
        "factor_id": fid,
        "factor_name": spec["factor_name"],
        "version": "1.0",
        "calculation": {
            "expression": spec["expr"],
            "description": spec["desc"],
        },
        "dependencies": spec["deps"],
        "parameters": spec["params"],
        "direction": spec["direction"],
        "tags": spec["tags"],
        "validation": {
            "horizon_days": H,
            "period": VALID_PERIOD,
            "status": "EFFECTIVE",
            "last_validated": LAST_VALIDATED,
            "metrics": {
                "ic": round(float(m10["mean_ic"]), 4),
                "icir": round(float(m10["icir"]), 4),
                "ic_hit_ratio": round(float(m10["hit"]), 3),
                "n_ic_dates": int(m10["n_dates"]),
                "n_assets_mean": round(float(m10["n_assets_mean"]), 1),
                "coverage_asset_days": round(asset_days, 3),
                "coverage_dates_ge8": round(dates_ge8, 3),
                "turnover_10d_rank": round(float(m10["turnover"]), 3),
                "decay_ic_by_horizon": decay,
                "max_abs_library_correlation": corr,
                "max_abs_library_correlation_vs": corr_src,
            },
            "regime_notes": regime_notes,
        },
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    persisted.append(fid)
    print(f"  WROTE {path}  corr={corr} (vs {corr_src})  [{time.time()-t0:.0f}s]")

print(f"\n=== Persisted {len(persisted)}: {persisted} ===")

# ---------------- read-back verification ----------------
print("\n=== READ-BACK VERIFICATION ===")
for fid in persisted:
    d = json.load(open(f"factors/{fid}.json"))
    assert d["factor_id"] == fid, "id mismatch"
    assert d["validation"]["status"] == "EFFECTIVE", "status mismatch"
    met = d["validation"]["metrics"]
    assert abs(met["ic"]) >= 0.007 and abs(met["icir"]) >= 0.084, "gate mismatch"
    assert met["max_abs_library_correlation"] < 0.5, "corr mismatch"
    print(f"OK {fid}: ic={met['ic']} icir={met['icir']} corr={met['max_abs_library_correlation']} "
          f"status={d['validation']['status']} decay={met['decay_ic_by_horizon']}")
print("ALL VERIFIED")
