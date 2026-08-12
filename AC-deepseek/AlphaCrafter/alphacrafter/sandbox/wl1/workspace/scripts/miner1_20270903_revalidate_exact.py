"""miner_1: re-validate library factors with EXACT persisted formulas through 2027-09-02.
Gate: abs IC >= 0.0070, abs ICIR >= 0.0840 at the factor's admission horizon.
Drift: 12m / 6m IC & ICIR.
"""
import numpy as np
import pandas as pd
import pickle, json

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]; O = panel["open"]; H = panel["high"]; L = panel["low"]
M = panel["macro"]
vix = M["VIX"]; dxy = M["DXY"]
ret = C.pct_change()
vix_ret = vix.pct_change(); dxy_ret = dxy.pct_change()

gate_ic, gate_icir = 0.0070, 0.0840

factors = {}
# --- exact library formulas ---
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0          # admission h=10
factors["mom_10d_skip5"] = C.shift(5) / C.shift(15) - 1.0            # h=5
for nd in (1, 2, 3, 5):
    factors[f"rev_{nd}d"] = -(np.log(C) - np.log(C.shift(nd)))       # h=1
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    factors[f"nclv_{nd}d"] = -(C - lo) / (hi - lo)                    # h=1
factors["id_rev_1d"] = -(C / O - 1.0)                                # h=1
rng = (H - L).replace(0, np.nan)
factors["nbody_1d"] = (C - L) / rng - 0.5                            # h=1
factors["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std() # h=10
cov60 = ret.rolling(60).cov(vix_ret); var60 = vix_ret.rolling(60).var()
factors["vix_beta_cond_60x20"] = -(cov60 / var60) * (vix / vix.shift(20) - 1.0)  # h=10

ADM = {"mom_120d_skip5": 10, "mom_10d_skip5": 5, "rev_1d": 1, "rev_2d": 1, "rev_3d": 1,
       "rev_5d": 1, "nclv_1d": 1, "nclv_2d": 1, "nclv_3d": 1, "nclv_5d": 1,
       "id_rev_1d": 1, "nbody_1d": 1, "vol_of_vol20x60": 10, "vix_beta_cond_60x20": 10}


def daily_ic_series(f, h):
    fwd = C.shift(-h) / C - 1.0
    out = []
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            out.append((dt, ic))
    return pd.Series({d: v for d, v in out})


def stats(s):
    if len(s) == 0:
        return None
    ic = s.mean(); sd = s.std(ddof=1)
    return ic, (ic / sd if sd > 0 else 0.0), float((s > 0).mean()), len(s)


print(f"{'factor':24s} {'h':>3s} {'IC':>8s} {'ICIR':>8s} {'hit':>5s} {'n':>5s} {'IC12m':>8s} {'ICIR12m':>8s} {'IC6m':>8s} {'ICIR6m':>8s}  gate")
res = {}
for name, f in factors.items():
    h = ADM[name]
    s = daily_ic_series(f, h)
    st = stats(s)
    if st is None:
        print(f"{name:24s} {h:>3d}  no data"); continue
    ic, icir, hit, n = st
    s12 = s[s.index >= s.index.max() - pd.Timedelta(days=365)]
    s6 = s[s.index >= s.index.max() - pd.Timedelta(days=183)]
    r12 = stats(s12); r6 = stats(s6)
    f12 = f"{r12[0]:8.4f} {r12[1]:8.4f}" if r12 else "        -"
    f6 = f"{r6[0]:8.4f} {r6[1]:8.4f}" if r6 else "        -"
    passed = abs(ic) >= gate_ic and abs(icir) >= gate_icir
    print(f"{name:24s} {h:>3d} {ic:8.4f} {icir:8.4f} {hit:5.2f} {n:5d} {f12:>16s} {f6:>16s}  {'PASS' if passed else 'FAIL'}")
    res[name] = {"h": h, "ic": round(float(ic), 4), "icir": round(float(icir), 4),
                 "hit": round(float(hit), 3), "n": int(n),
                 "ic12m": round(float(r12[0]), 4) if r12 else None,
                 "icir12m": round(float(r12[1]), 4) if r12 else None,
                 "ic6m": round(float(r6[0]), 4) if r6 else None,
                 "icir6m": round(float(r6[1]), 4) if r6 else None,
                 "passed": bool(passed)}

json.dump(res, open("scripts/miner1_20270903_revalidate_exact.json", "w"), indent=1)
print("\nsaved scripts/miner1_20270903_revalidate_exact.json")
