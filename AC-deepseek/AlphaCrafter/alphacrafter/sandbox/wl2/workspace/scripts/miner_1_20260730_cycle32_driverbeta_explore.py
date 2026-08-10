"""miner_1 cycle 32: explore macro-driver beta-conditional factors with NEW drivers.

Motivation: beta_cond(driver, w, m) = rolling_beta(asset_ret, driver_ret, w) * (driver/driver.shift(m)-1)
has produced the strongest library members (dxy_beta_cond_60x20, usdjpy_beta_cond_120x60).
The FX-risk dimension is now crowded (usdjpy vs dxy last60 rho 0.497 near gate). Test
orthogonal drivers: US10Y (rates), CN10Y (China rates), BTC (crypto risk), WTI (energy),
COPPER (global growth), XAU (safe haven), SPX (market beta), plus references
skew_60 and vol_ratio_5x60.

Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation<0.5, turnover vs 10d cadence.
Validation date 2026-07-30 (visible through 2026-07-29).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor, turnover_rank)

close_panel = load_panel()

# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------
def beta_cond(asset_close, driver_close, w=60, m=20, minp_frac=0.5):
    dcs = driver_close.reindex(asset_close.index).ffill()
    ar = asset_close.pct_change()
    dr = dcs.pct_change()
    df = pd.concat([ar.rename("a"), dr.rename("d")], axis=1).dropna()
    minp = max(int(w * minp_frac), 15)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["d"])
    var = df["d"].rolling(w, min_periods=minp).var()
    beta = cov / var
    mom = dcs / dcs.shift(m) - 1.0
    return beta * mom.reindex(beta.index)


def skew_60(s):
    return s.pct_change().rolling(60, min_periods=30).skew()


def vol_ratio_5x60(s):
    r = s.pct_change()
    v5 = r.rolling(5, min_periods=4).std()
    v60 = r.rolling(60, min_periods=30).std()
    return v5 / v60


cands = {}
cands["us10y_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("US10Y"), 60, 20)
cands["us10y_beta_cond_120x60"] = per_asset(close_panel, beta_cond, macro_series("US10Y"), 120, 60)
cands["cn10y_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("CN10Y"), 60, 20)
cands["btc_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("BTC"), 60, 20)
cands["wti_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("WTI"), 60, 20)
cands["wti_beta_cond_120x60"] = per_asset(close_panel, beta_cond, macro_series("WTI"), 120, 60)
cands["copper_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("COPPER"), 60, 20)
cands["copper_beta_cond_120x60"] = per_asset(close_panel, beta_cond, macro_series("COPPER"), 120, 60)
cands["xau_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("XAU"), 60, 20)
cands["spx_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("SPX"), 60, 20)
cands["eurusd_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("EURUSD"), 60, 20)
cands["skew_60"] = per_asset(close_panel, skew_60)
cands["vol_ratio_5x60"] = per_asset(close_panel, vol_ratio_5x60)

# ---------------------------------------------------------------------------
# Library signals (current ACTIVE set + recent references) for correlation gate
# ---------------------------------------------------------------------------
def mom20_volproxy60(s):
    mom = s.shift(5) / s.shift(25) - 1.0
    proxy = s.shift(5) / s.shift(65) - 1.0
    return mom / (1.0 + proxy.abs())

def calmness_20(s):
    r = s.pct_change()
    return r.abs().rolling(20, min_periods=10).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan,
        raw=True)

def downbeta_spx_60(s):
    spx = macro_series("SPX").reindex(s.index).ffill()
    ar, sr = s.pct_change(), spx.pct_change()
    df = pd.concat([ar.rename("a"), sr.rename("s")], axis=1).dropna()
    neg = df[df["s"] < 0]
    if len(neg) >= 15:
        b = neg["a"].rolling(60, min_periods=15).cov(neg["s"]) / neg["s"].rolling(60, min_periods=15).var()
        return b.reindex(df.index)
    return pd.Series(np.nan, index=df.index)

lib = {}
lib["mom20_volproxy60"] = per_asset(close_panel, mom20_volproxy60)
lib["dxy_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("DXY"), 60, 20)
lib["calmness_20"] = per_asset(close_panel, calmness_20)
lib["usdjpy_beta_cond_120x60"] = per_asset(close_panel, beta_cond, macro_series("USDJPY"), 120, 60)
lib["downbeta_spx_60"] = per_asset(close_panel, downbeta_spx_60)

fwd = {str(h): forward_returns(close_panel, h) for h in (1, 2, 3, 5, 10, 20)}

# ---------------------------------------------------------------------------
# Validate all candidates
# ---------------------------------------------------------------------------
results = {}
for name, sig in cands.items():
    m = validate_factor(sig, close_panel, library=lib, fwd_cache=fwd)
    # regime splits on admission-horizon (10d) IC series
    ic_ser = compute_ic(sig, fwd["10"]).dropna()
    reg = {}
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29"),
                   ("2026-03-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 20:
            sd = sub.std()
            reg[f"{r0[:4]}-{r1[:4]}"] = {"ic": round(sub.mean(), 4),
                                          "icir": round(sub.mean() / sd, 3) if sd > 0 else 0.0,
                                          "n": int(len(sub))}
    results[name] = {"metrics": m, "regime": reg}
    passed = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 and m["max_abs_library_correlation"] < 0.5
    print(f"[{name:28s}] IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} to={m['turnover_10_rank']} "
          f"maxlib={m['max_abs_library_correlation']:.4f} => {'PASS' if passed else 'fail'}")
    if passed:
        reg_s = " | ".join(f"{k}:{v['ic']:+.4f}/{v['icir']:+.3f}(n{v['n']})" for k, v in reg.items())
        print(f"      regime: {reg_s}")

json.dump(results, open("scripts/_miner1_cycle32_driverbeta_results.json", "w"), indent=1, default=str)
print("\nDONE cycle32 explore")
