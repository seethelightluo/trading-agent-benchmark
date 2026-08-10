"""miner_3 cycle 35: explore NEW orthogonal axes after beta-family persistence.

Lesson from gate: vol_price_corr_60 evicted at stacked Spearman rho 0.573 vs
mom20_volproxy60 / 0.571 vs usdjpy_beta_cond_120x60 (my raw-stack estimate was
too low). To be safe, candidates must be far from momentum/trend and I report
BOTH per-date mean spearman AND two stacked variants (raw global rank,
per-date-ranked), requiring all < 0.45.

Candidate axes (avoid momentum/trend/volume-corr families):
  A. ratebeta_cond_60x20  : beta(asset, US10Y ret, 60) x US10Y 20d move
  B. cnybeta_cond_60x20   : beta(asset, USDCNY ret, 60) x USDCNY 20d move
  C. cryptobeta_cond_60x20: beta(asset, BTC ret, 60) x BTC 20d move (self NaN)
  D. goldbeta_cond_60x20  : beta(asset, XAU ret, 60) x XAU 20d move
  E. market_relative_20   : 20d ret - cross-sectional mean(20d ret)
  F. avg_corr_60          : mean pairwise rolling corr with other 14 assets
  G. mdd_60               : close/rolling_max(close,60) - 1 (drawdown depth)
  H. risk_prem_20_20      : 20d ret / 20d realized vol
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series,
                         forward_returns, compute_ic, validate_factor,
                         VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------- active library: every top-level EFFECTIVE factor with .npy ----
lib = {}
for npy in sorted(glob.glob("factors/*.signal.npy")):
    fid = Path(npy).stem.replace(".signal", "")
    jp = npy.replace(".signal.npy", ".json")
    try:
        d = json.load(open(jp))
        if d.get("validation", {}).get("status") != "EFFECTIVE":
            continue
    except Exception:
        continue
    a = np.load(npy)
    if a.shape == (len(panel), len(panel.columns)):
        lib[fid] = pd.DataFrame(a, index=panel.index, columns=panel.columns)
print(f"active library ({len(lib)}): {sorted(lib)}")


def stacked_spearman(a, b, per_date_rank=False):
    """Spearman on stacked (date x asset) values; optionally rank per date first."""
    aa, bb = a.copy(), b.copy()
    if per_date_rank:
        aa = aa.rank(axis=1)
        bb = bb.rank(axis=1)
    df = pd.concat([aa.stack().rename("x"), bb.stack().rename("y")], axis=1).dropna()
    if len(df) < 30:
        return 0.0
    return float(df["x"].corr(df["y"], method="spearman"))


def library_corr_metrics(cand):
    out = {}
    for fid, sig in lib.items():
        r1 = round(stacked_spearman(cand, sig, per_date_rank=False), 4)
        r2 = round(stacked_spearman(cand, sig, per_date_rank=True), 4)
        out[fid] = {"raw": r1, "ranked": r2}
    max_raw = max((abs(v["raw"]) for v in out.values()), default=0.0)
    max_ranked = max((abs(v["ranked"]) for v in out.values()), default=0.0)
    return out, round(max_raw, 4), round(max_ranked, 4)


def cond_beta_factor(macro_name, w=60, mw=20, mp=30, exclude=None):
    """beta_w(asset, macro) x macro mw-day momentum, per-asset own calendar."""
    m = macro_series(macro_name)
    m_ret = m.pct_change()
    m_mom = m / m.shift(mw) - 1.0
    parts = {}
    for a in TRADABLES:
        if exclude and a in exclude:
            parts[a] = pd.Series(np.nan, index=panel.index)
            continue
        s = panel[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), m_ret.rename("m")], axis=1).dropna()
        b = df["a"].rolling(w, min_periods=mp).cov(df["m"]) / df["m"].rolling(w, min_periods=mp).var()
        parts[a] = b.mul(m_mom.reindex(b.index), axis=0).reindex(panel.index)
    return pd.DataFrame(parts, index=panel.index)


def market_relative(w=20):
    mom = panel / panel.shift(w) - 1.0
    return mom.sub(mom.mean(axis=1), axis=0)


def avg_corr(w=60, mp=40):
    ret = panel.pct_change()
    out = {}
    for i, a in enumerate(TRADABLES):
        ra = ret[a]
        cs = [ra.rolling(w, min_periods=mp).corr(ret[b])
              for b in TRADABLES[i + 1:]]
        # include both (a,b) and (b,a) to get full mean over 14 partners
        cs2 = [ret[b].rolling(w, min_periods=mp).corr(ra)
               for b in TRADABLES[:i]]
        out[a] = pd.concat(cs + cs2, axis=1).mean(axis=1).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def mdd_depth(w=60, mp=40):
    peak = panel.rolling(w, min_periods=mp).max()
    return panel / peak - 1.0


def risk_prem(w=20):
    mom = panel / panel.shift(w) - 1.0
    vol = panel.pct_change().rolling(w, min_periods=10).std()
    return mom / (vol + 1e-12)


cands = {
    "ratebeta_cond_60x20": cond_beta_factor("US10Y"),
    "cnybeta_cond_60x20": cond_beta_factor("USDCNY"),
    "cryptobeta_cond_60x20": cond_beta_factor("BTC", exclude={"BTC", "ETH"}),
    "goldbeta_cond_60x20": cond_beta_factor("XAU"),
    "market_relative_20": market_relative(20),
    "avg_corr_60": avg_corr(),
    "mdd_60": mdd_depth(),
    "risk_prem_20_20": risk_prem(20),
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, F in cands.items():
    m = validate_factor(F, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    lc, max_raw, max_ranked = library_corr_metrics(F)
    m["max_abs_library_correlation"] = max_raw
    m["max_ranked_library_correlation"] = max_ranked
    m["library_pairwise_corr"] = lc
    m["turnover_10d_rank"] = m.pop("turnover_10_rank", None)
    ic_ser = compute_ic(F, fwd_cache[str(ADM_H)]).dropna()
    reg = {}
    for r0, r1, tag in [("2020-01-01", "2021-12-31", "2020-21"),
                        ("2022-01-01", "2022-12-31", "2022"),
                        ("2023-01-01", "2024-12-31", "2023-24"),
                        ("2025-01-01", "2026-07-29", "2025-26")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            reg[tag] = {"ic": round(float(sub.mean()), 4),
                        "icir": round(float(sub.mean() / sd) if sd > 0 else 0.0, 4),
                        "n": int(len(sub))}
    last = ic_ser.iloc[-250:]
    if len(last) >= 30:
        sd = last.std()
        reg["last250"] = {"ic": round(float(last.mean()), 4),
                          "icir": round(float(last.mean() / sd) if sd > 0 else 0.0, 4),
                          "n": int(len(last))}
    p_ic = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    p_corr = max_raw < 0.45 and max_ranked < 0.45
    p = p_ic and p_corr
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"cov_asset={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxrho_raw={max_raw} maxrho_ranked={max_ranked} "
          f"=> {'PASS' if p else 'FAIL'}")
    top = sorted(lc.items(), key=lambda kv: -abs(kv[1]["raw"]))[:3]
    print(f"    top-rho: {[(k, v) for k, v in top]}")
    print(f"    regime: {json.dumps(reg)}")
    results[name] = {"metrics": m, "pass": bool(p), "regime": reg}

json.dump(results, open("scripts/_miner3_cycle35_results.json", "w"), indent=1, default=float)
print("\nDONE cycle35")
