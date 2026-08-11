"""miner_2: Explore batch A - macro-beta / rate-conditional factor family (2026-07-30).

Motivation: existing library already spans equity momentum, trend-quality, vol-of-vol,
crash-risk and VIX/DXY/WTI macro-betas. Rate and FX-conditional sensitivities are
under-represented and likely low-correlation with the live library.

Candidates (all asset-specific, cross-sectional by construction):
  us10y_beta_60, cn10y_beta_60            : raw rate betas (60d)
  us10y_cond_20, cn10y_cond_20            : -beta(asset, macro_ret) * macro_20d_ret (conditional)
  rate_spread_cond_20                     : -beta(asset, d(spread)) * spread_20d_chg, spread=US10Y-CN10Y
  bond_corr_60                            : rolling corr(asset_ret, US10Y_ret, 60)
  usdcny_cond_20, usdjpy_cond_20          : FX-conditional (same pattern)
  eurusd_cond_20                          : EURUSD-conditional
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
LIB_DIR = "factors"

close = closes_panel(VIS)
idx = close.index
fr = forward_returns(close, H)
ret = close.pct_change()
macro = macro_closes(VIS)
mret = macro.pct_change()


def rolling_beta(asset_ret, fac_ret, win):
    """rolling beta of asset_ret on fac_ret (aligned on union index)."""
    pair = pd.concat([asset_ret.rename("a"), fac_ret.rename("f")], axis=1).dropna()
    if len(pair) < win + 5:
        return pd.Series(np.nan, index=idx)
    cov = pair["a"].rolling(win).cov(pair["f"])
    var = pair["f"].rolling(win).var()
    b = (cov / var.replace(0, np.nan)).reindex(idx)
    return b


def rolling_corr(asset_ret, fac_ret, win):
    pair = pd.concat([asset_ret.rename("a"), fac_ret.rename("f")], axis=1).dropna()
    if len(pair) < win + 5:
        return pd.Series(np.nan, index=idx)
    return pair["a"].rolling(win).corr(pair["f"]).reindex(idx)


def cond_factor(asset_ret, fac_ret, macro_20d_ret, win=60):
    """-beta(asset, macro_ret, win) * macro_20d_ret (vix_beta_cond pattern)."""
    b = rolling_beta(asset_ret, fac_ret, win)
    return (-b * macro_20d_ret).reindex(idx)


# --- macro 20d moves (rate uses level change; FX uses pct change) ---
us10y_20 = macro["US10Y"].diff(20) if "US10Y" in macro.columns else None
cn10y_20 = macro["CN10Y"].diff(20) if "CN10Y" in macro.columns else None
us10y_ret = mret["US10Y"] if "US10Y" in macro.columns else None
cn10y_ret = mret["CN10Y"] if "CN10Y" in macro.columns else None

spread = None
if "US10Y" in macro.columns and "CN10Y" in macro.columns:
    spread = macro["US10Y"] - macro["CN10Y"]
    spread_20 = spread.diff(20)

cands = {}
# --- raw rate betas ---
if us10y_ret is not None:
    cands["us10y_beta_60"] = pd.DataFrame({s: rolling_beta(ret[s], us10y_ret, 60) for s in WATCH}, index=idx)
if cn10y_ret is not None:
    cands["cn10y_beta_60"] = pd.DataFrame({s: rolling_beta(ret[s], cn10y_ret, 60) for s in WATCH}, index=idx)
# --- conditional rate factors ---
if us10y_ret is not None and us10y_20 is not None:
    cands["us10y_cond_20"] = pd.DataFrame({s: cond_factor(ret[s], us10y_ret, us10y_20, 60) for s in WATCH}, index=idx)
if cn10y_ret is not None and cn10y_20 is not None:
    cands["cn10y_cond_20"] = pd.DataFrame({s: cond_factor(ret[s], cn10y_ret, cn10y_20, 60) for s in WATCH}, index=idx)
# --- rate spread conditional ---
if spread is not None:
    sret = spread.diff()
    cands["rate_spread_cond_20"] = pd.DataFrame({s: cond_factor(ret[s], sret, spread_20, 60) for s in WATCH}, index=idx)
# --- bond correlation ---
if us10y_ret is not None:
    cands["bond_corr_60"] = pd.DataFrame({s: rolling_corr(ret[s], us10y_ret, 60) for s in WATCH}, index=idx)
# --- FX conditionals ---
for mname in ("USDCNY", "USDJPY", "EURUSD"):
    if mname not in mret.columns:
        continue
    m20 = macro[mname].pct_change(20)
    cands[f"{mname.lower()}_cond_20"] = pd.DataFrame({s: cond_factor(ret[s], mret[mname], m20, 60) for s in WATCH}, index=idx)

# --- library IC map (decoded artifacts) for rho ---
def decode_artifact(meta):
    a = meta.get("validation", {}).get("signal_artifact")
    if not a:
        return None
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig = pd.read_csv(io.StringIO(dec), index_col=0, parse_dates=True)
    return sig.reindex(columns=close.columns).reindex(close.index)

import io, base64, zlib
lib_ics = {}
for fn in sorted(os.listdir(LIB_DIR)):
    if not fn.endswith(".json") or fn == "factor_ensemble.json":
        continue
    with open(os.path.join(LIB_DIR, fn), encoding="utf-8") as f:
        meta = json.load(f)
    sig = decode_artifact(meta)
    if sig is None:
        continue
    ic = ic_series(sig, fr, min_valid=8)
    if len(ic.dropna()) > 30:
        lib_ics[meta["factor_id"]] = ic
print("library IC series decoded:", len(lib_ics))


def rho_vs_lib(my_ic):
    best, best_id = 0.0, None
    for fid, s in lib_ics.items():
        pair = pd.concat([my_ic.rename("a"), s.rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        r = pair["a"].corr(pair["b"])
        if np.isfinite(r) and abs(float(r)) > best:
            best, best_id = abs(float(r)), fid
    return round(best, 4), best_id


print(f"\n--- BATCH A SCREEN (h={H}) ---")
results = {}
for name, sig in cands.items():
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{name:22s} insufficient ({len(ic.dropna())})")
        continue
    m["rho"], m["rho_id"] = rho_vs_lib(ic)
    m["regime"] = regime_split(ic)
    results[name] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    flag = "PASS" if gate else "fail"
    print(f"{name:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} turn={m.get('turnover_10d_rank')} "
          f"rho={m['rho']:.3f}({m['rho_id']}) [{flag}]")
    print("     regime:", json.dumps({k: v["ic"] for k, v in m["regime"].items()}))

with open("scripts/miner2_20260730_explore_batchA_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved -> scripts/miner2_20260730_explore_batchA_results.json")
