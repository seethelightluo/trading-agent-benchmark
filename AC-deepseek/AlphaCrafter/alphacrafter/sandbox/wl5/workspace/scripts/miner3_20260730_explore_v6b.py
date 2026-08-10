"""miner_3 2026-07-30: debug MOM_REL_EQ_20 n=0 + test more relative-momentum variants."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import closes_panel, forward_returns, ic_series, summary_metrics, regime_split

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)
WATCH = list(close.columns)

# ---- debug cross-sectional demean ----
mom20 = close / close.shift(20) - 1.0
demean = mom20.sub(mom20.mean(axis=1), axis=0)
print("demean NaN frac:", round(float(demean.isna().mean().mean()), 3))
print("sample row 2026-07-01:", demean.loc["2026-07-01"].round(4).to_dict() if "2026-07-01" in demean.index else "no row")
ics = ic_series(demean, fr, min_valid=8)
print("MOM_REL_EQ_20 n IC:", len(ics))
if len(ics):
    print("IC:", round(float(ics.mean()), 4), "ICIR:", round(float(ics.mean() / ics.std(ddof=1)), 4))
else:
    # inspect why: factor nunique per date vs fwd ret nunique
    d = "2026-07-01"
    if d in demean.index:
        f = demean.loc[d]; r = fr.loc[d]
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        print("pair len:", len(pair), "f nunique:", pair["f"].nunique(), "r nunique:", pair["r"].nunique())

# ---- more relative-momentum variants ----
cands = {}
cands["XAU_REL_MOM_10"] = (close / close.shift(10) - 1.0).sub(mom20["XAU"], axis=0)
cands["CN_REL_MOM_20"] = mom20.sub(mom20["000300.SH"], axis=0)
cands["WTI_REL_MOM_20"] = mom20.sub(mom20["WTI"], axis=0)
cands["BOND_REL_MOM_20"] = mom20.sub(mom20["US10Y"], axis=0)
cands["MOM_SKIP5_20"] = close.shift(5) / close.shift(25) - 1.0  # 20d mom skipping last 5d
cands["XAU_REL_MOM_SKIP5"] = (close.shift(5) / close.shift(25) - 1.0).sub(mom20["XAU"], axis=0)

print("\n=== variants ===")
out = {}
for fid, sig in cands.items():
    sig = sig.reindex(close.index)
    ics = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ics, sig, fr, close, h=H)
    if m is None:
        print(f"{fid:20s} INSUFFICIENT n={len(ics)}")
        continue
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    flag = "*** PASS ***" if gate else ""
    print(f"{fid:20s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} turn={m['turnover_10d_rank']:.3f} {flag}")
    out[fid] = {"ic": m["ic"], "icir": m["icir"], "n": m["n_ic_dates"],
                "regime": regime_split(ics), "pass": gate, "sig": sig}

with open("scripts/miner3_20260730_explore_v6b_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "sig"} for k, v in out.items()},
              f, indent=1, default=str)
print("saved")
