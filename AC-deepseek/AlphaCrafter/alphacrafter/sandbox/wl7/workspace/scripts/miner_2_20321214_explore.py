"""miner_2 candidate screening cycle 2032-12-14 (visible through 2032-12-13).
Novel macro-FX conditional beta factors (USDCNY, USDJPY) + revalidate skew_20d_skip5.
Also screen a couple of related conditional-betas for robustness.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (ASSETS, load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr as _mlc, rank_turnover,
                          ACTIVE_LIB)

END = "2032-12-13"
close = load_close(END); macro = load_macro(END)
ret = close.pct_change()
fwd10 = forward_ret(close, 10)
lib = library_panel(close, macro)


def max_lib_corr(cand, libp):
    flat = cand.stack(); best = 0.0; pairs = {}
    for name, p in libp.items():
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"])); pairs[name] = round(rho, 3)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs


def beta_cond(close, fx, beta_win=60, cond_win=20, min_periods=30, sign=1.0):
    r = close.pct_change()
    fxr = fx.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(fxr)
    var = fxr.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    fx_mom = fx / fx.shift(cond_win) - 1.0
    return sign * beta.multiply(fx_mom, axis=0)


def skew(close, window=20, skip=5, min_periods=12):
    r = close.pct_change()
    rr = r.shift(skip)
    return rr.rolling(window, min_periods=min_periods).skew()


rows = []
cands = {}
for fx_name in ["USDCNY", "USDJPY", "US10Y", "VIX"]:
    for sign, lab in [(1.0, "pos"), (-1.0, "neg")]:
        f = beta_cond(close, macro[fx_name], sign=sign)
        fid = f"{fx_name.lower()}_beta_cond_60x20_{lab}"
        cands[fid] = f

# skew candidate
cands["skew_20d_skip5"] = skew(close)

rows = []
for name, f in cands.items():
    ic = daily_ic(f, fwd10)
    st = ic_stats(ic, 10)
    s = ic.dropna()
    m6 = s[s.index >= "2032-06-01"]; m = m6.mean(); sd = m6.std(ddof=1)
    r6 = (float(m), float(m/sd) if sd > 0 else np.nan, float((m6 > 0).mean()) if len(m6) else 0.0)
    cov = coverage_stats(f, fwd10)
    turn = rank_turnover(f, 10)
    mrho, pairs = max_lib_corr(f, lib)
    gate_full = abs(st["ic"]) >= 0.007 and abs(st["icir"]) >= 0.084
    gate_recent = abs(r6[0]) >= 0.007 and abs(r6[1]) >= 0.084
    rows.append(dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                     hit=round(st["hit"],3), n=st["n"], ic_6m=round(r6[0],4),
                     icir_6m=(round(r6[1],3) if np.isfinite(r6[1]) else None),
                     hit_6m=round(r6[2],3), gate_full=bool(gate_full), gate_recent=bool(gate_recent),
                     covAD=round(cov["coverage_asset_days"],3), turn=round(turn,2),
                     maxlib=round(mrho,4), pairs=pairs))

out = sorted(rows, key=lambda r: abs(r["ic"]), reverse=True)
pd.set_option("display.width", 300)
print(pd.DataFrame(out)[["name","ic","icir","hit","n","ic_6m","icir_6m","gate_full","gate_recent","covAD","turn","maxlib"]].to_string(index=False))
json.dump(out, open("scripts/miner_2_20321214_explore.json","w"), indent=1)
print("\nsaved scripts/miner_2_20321214_explore.json")