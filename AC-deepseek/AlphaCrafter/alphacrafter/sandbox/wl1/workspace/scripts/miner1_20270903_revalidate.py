"""miner_1: re-validate all currently effective library factors on data through 2027-09-02.
Admission gate: abs IC >= 0.0070, abs ICIR >= 0.0840 (10d horizon).
Also reports 12m (last 365d) and 6m (last 183d) IC for drift monitoring.
"""
import numpy as np
import pandas as pd
import pickle, json

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
M = panel["macro"]

gate_ic, gate_icir = 0.0070, 0.0840

factors = {}

# mom family
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
factors["mom_10d_skip5"] = C.shift(5) / C.shift(15) - 1.0

# reversal family (close-based): positive after declines
for nd in (1, 2, 3, 5):
    factors[f"rev_{nd}d"] = -(C.shift(nd) / C - 1.0)

# close location value (nclv)
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    factors[f"nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5

# intraday reversal: -(close-open)/open of day t
O = panel["open"]
factors["id_rev_1d"] = -(C / O - 1.0)

# nbody: body position within range
H = panel["high"]; L = panel["low"]
rng = (H - L).replace(0, np.nan)
factors["nbody_1d"] = (C - L) / rng - 0.5

# vol of vol
def vol_series(x, w):
    return x.pct_change().rolling(w).std()
factors["vol_of_vol20x60"] = vol_series(C.rolling(20).std(), 60)

# vix beta conditional (direction negative)
vix = M["VIX"]
vix_ret = vix.pct_change()
cov60 = C.pct_change().rolling(60).cov(vix_ret)
var60 = vix_ret.rolling(60).var()
beta_vix = cov60 / var60
cond = (vix > vix.rolling(20).mean()).astype(float)
factors["vix_beta_cond_60x20"] = beta_vix * cond


def daily_ic_series(f, h):
    fwd = C.shift(-h) / C - 1.0
    ics = []
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            ics.append((dt, ic))
    return pd.Series({d: v for d, v in ics})


def summarize(s):
    if len(s) == 0:
        return None
    ic = s.mean(); sd = s.std(ddof=1)
    icir = ic / sd if sd > 0 else 0
    hit = float((s > 0).mean()) if ic > 0 else float((s < 0).mean())
    return ic, icir, hit, len(s)


print(f"{'factor':24s} {'h':>3s} {'IC':>8s} {'ICIR':>8s} {'hit':>5s} {'n':>5s} {'IC12m':>8s} {'ICIR12m':>8s} {'IC6m':>8s} {'ICIR6m':>8s}  gate")
out = {}
for name, f in factors.items():
    best = None
    for h in (1, 2, 3, 5, 10):
        s = daily_ic_series(f, h)
        if len(s) == 0:
            continue
        ic, icir, hit, n = summarize(s)
        if best is None or abs(icir) > abs(best[2]):
            best = (h, ic, icir, hit, n, s)
    if best is None:
        print(f"{name:24s}  no data"); continue
    h, ic, icir, hit, n, s = best
    s12 = s[s.index >= s.index.max() - pd.Timedelta(days=365)]
    s6 = s[s.index >= s.index.max() - pd.Timedelta(days=183)]
    r12 = summarize(s12); r6 = summarize(s6)
    ic12 = f"{r12[0]:8.4f} {r12[1]:8.4f}" if r12 else "     -"
    ic6 = f"{r6[0]:8.4f} {r6[1]:8.4f}" if r6 else "     -"
    passed = abs(ic) >= gate_ic and abs(icir) >= gate_icir
    print(f"{name:24s} {h:>3d} {ic:8.4f} {icir:8.4f} {hit:5.2f} {n:5d} {ic12:>16s} {ic6:>16s}  {'PASS' if passed else 'FAIL'}")
    out[name] = {"h": h, "ic": round(float(ic), 4), "icir": round(float(icir), 4),
                 "hit": round(float(hit), 3), "n": int(n),
                 "ic12m": (round(float(r12[0]), 4) if r12 else None),
                 "icir12m": (round(float(r12[1]), 4) if r12 else None),
                 "ic6m": (round(float(r6[0]), 4) if r6 else None),
                 "icir6m": (round(float(r6[1]), 4) if r6 else None),
                 "passed": bool(passed)}

json.dump(out, open("scripts/miner1_20270903_revalidate_results.json", "w"), indent=1)
print("\nsaved scripts/miner1_20270903_revalidate_results.json")
