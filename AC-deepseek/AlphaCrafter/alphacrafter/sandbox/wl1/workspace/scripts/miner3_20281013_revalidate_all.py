"""miner_3 2028-10-13: re-validate all library factors on panel through 2028-10-12.
Admission gate: abs IC >= 0.0070, abs ICIR >= 0.0840.
Reports full-sample + last-365d metrics for drift monitoring.
"""
import numpy as np
import pandas as pd
import pickle, json

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
O = panel["open"]; H = panel["high"]; L = panel["low"]; V = panel["vol"]
R = C.pct_change()

gate_ic, gate_icir = 0.0070, 0.0840

factors = {}

# --- effective library factors (re-validate exactly as defined) ---
# mom family
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
# reversal family (close-based): positive after declines
for nd in (1, 2, 3, 5):
    factors[f"rev_{nd}d"] = -(C.shift(nd) / C - 1.0)
factors["rev_1d_vs"] = -(C.shift(1) / C - 1.0) * np.sign(C - C.shift(1))
# close location value (nclv)
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    factors[f"nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5
# intraday reversal
factors["id_rev_1d"] = -(C / O - 1.0)
# nbody
rng = (H - L).replace(0, np.nan)
factors["nbody_1d"] = (C - L) / rng - 0.5
# vol of vol
factors["vol_of_vol20x60"] = R.rolling(20).std().rolling(60).std()
# vix beta conditional (direction negative)
M = panel["macro"]
vix = M["VIX"]
vix_ret = vix.pct_change()
cov60 = R.rolling(60).cov(vix_ret)
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


print(f"{'factor':22s} {'h':>3s} {'IC':>9s} {'ICIR':>9s} {'hit':>6s} {'n':>5s} {'IC12m':>9s} {'ICIR12m':>9s}  gate")
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
        print(f"{name:22s}  no data"); continue
    h, icir, ic, n, hit, s = best
    cut = s.index.max() - pd.Timedelta(days=365)
    s12 = s[s.index >= cut]
    ic12 = s12.mean(); icir12 = ic12 / s12.std(ddof=1) if len(s12) > 2 and s12.std(ddof=1) > 0 else 0
    ok = "PASS" if abs(ic) >= gate_ic and abs(icir) >= gate_icir else "fail"
    print(f"{name:22s} {h:3d} {ic:+9.5f} {icir:+9.5f} {hit:6.3f} {n:5d} {ic12:+9.5f} {icir12:+9.5f}  {ok}")
    out[name] = dict(h=h, ic=ic, icir=icir, hit=hit, n=n, ic12=ic12, icir12=icir12, ok=ok, last_date=str(s.index.max().date()))

json.dump(out, open("scripts/miner3_20281013_revalidate_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/miner3_20281013_revalidate_results.json")
