"""miner_3: re-validate all library factors on panel up to 2027-06-24.
Gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 (15-asset cross-asset universe).
Reports full-period and recent 12m for drift detection.
"""
import numpy as np
import pandas as pd
import pickle
import json

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
lr = C.pct_change()
lnC = np.log(C)
VIX = panel["macro"]["VIX"]
vix_ret = VIX.pct_change()

def nclv(win):
    return (C / C.shift(win) - 1).rank(axis=1)

factors = {}
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
for w in (1, 2, 3, 5):
    factors[f"nclv_{w}d"] = nclv(w)
factors["rev_1d"] = -(C / C.shift(1) - 1)
factors["rev_2d"] = -(C / C.shift(2) - 1)
factors["rev_3d"] = -(C / C.shift(3) - 1)
factors["rev_5d"] = -(C / C.shift(5) - 1)
factors["rev_1d_vs"] = -lnC.diff(1) / lr.rolling(20).std()
factors["id_rev_1d"] = -(C / panel["open"] - 1.0)
factors["nbody_1d"] = -((C - panel["open"]) / (panel["high"] - panel["low"]))
vol20 = lr.rolling(20).std()
factors["vol_of_vol20x60"] = vol20.rolling(60).std()
beta60 = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
for c in C.columns:
    beta60[c] = (C[c].pct_change().rolling(60, min_periods=30).cov(vix_ret)
                 / vix_ret.rolling(60, min_periods=30).var())
vix_move20 = VIX / VIX.shift(20) - 1.0
factors["vix_beta_cond_60x20"] = -beta60.mul(vix_move20, axis=0)

gate_ic, gate_icir = 0.0070, 0.0840

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

print(f"{'factor':26s} {'h':>3s} {'IC':>9s} {'ICIR':>9s} {'hit':>6s} {'n':>5s} {'IC12m':>9s} {'ICIR12m':>9s}  gate")
out = {}
for name, f in factors.items():
    best = None
    for h in (1, 2, 3, 5, 10):
        s = daily_ic_series(f, h)
        if len(s) == 0:
            continue
        ic = s.mean(); sd = s.std(ddof=1)
        icir = ic / sd if sd > 0 else 0
        if best is None or abs(icir) > abs(best[1]):
            hit = float((s > 0).mean()) if ic > 0 else float((s < 0).mean())
            best = (h, icir, ic, len(s), hit, s)
    if best is None:
        print(f"{name:26s}  no data")
        continue
    h, icir, ic, n, hit, s = best
    cut = s.index.max() - pd.Timedelta(days=365)
    s12 = s[s.index >= cut]
    ic12 = s12.mean(); icir12 = ic12 / s12.std(ddof=1) if s12.std(ddof=1) > 0 else 0
    ok = "PASS" if abs(ic) >= gate_ic and abs(icir) >= gate_icir else "fail"
    print(f"{name:26s} {h:3d} {ic:+9.5f} {icir:+9.5f} {hit:6.3f} {n:5d} {ic12:+9.5f} {icir12:+9.5f}  {ok}")
    out[name] = dict(h=h, ic=ic, icir=icir, hit=hit, n=n, ic12=ic12, icir12=icir12, ok=ok)

json.dump(out, open("scripts/miner3_20270625_revalidate_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/miner3_20270625_revalidate_results.json")
