"""miner_3 2028-10-27: re-validate all library factors on panel through 2028-10-26.
Admission gate: abs IC >= 0.0070, abs ICIR >= 0.0840.
Reports full-sample + last-365d + last-120d metrics for drift monitoring.
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


out = {}
for name, f in factors.items():
    row = {}
    for h in (1, 2, 3, 5):
        ic_ser = daily_ic_series(f, h)
        row[h] = dict(n=len(ic_ser), ic=float(ic_ser.mean()), icir=float(ic_ser.mean() / ic_ser.std()),
                      hit=float((ic_ser > 0).mean()), last=ic_ser.index.max().date())
    # drift windows for h=1
    ic1 = daily_ic_series(f, 1)
    out[name] = row
    last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
    last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
    ok = (abs(row[1]['ic']) >= gate_ic) and (abs(row[1]['icir']) >= gate_icir)
    print(f"{name:22s} h1 ic={row[1]['ic']:+.5f} icir={row[1]['icir']:+.5f} hit={row[1]['hit']:.3f} n={row[1]['n']:4d} | "
          f"365d ic={last365.mean():+.5f} icir={last365.mean()/last365.std():+.5f} | 120d ic={last120.mean():+.5f} icir={last120.mean()/last120.std():+.5f} | gate={ok}")

json.dump(out, open("scripts/miner3_20281027_revalidate_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner3_20281027_revalidate_results.json")
