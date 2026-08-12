"""miner_3 2029-04-13: re-validate all library factors on panel through 2029-04-12.
Admission gate: abs IC >= 0.0070, abs ICIR >= 0.0840 (daily, h=1).
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
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
for nd in (1, 2, 3, 5):
    factors[f"rev_{nd}d"] = -(C.shift(nd) / C - 1.0)
factors["rev_1d_vs"] = -(C.shift(1) / C - 1.0) * np.sign(C - C.shift(1))
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    factors[f"nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5
factors["id_rev_1d"] = -(C / O - 1.0)
rng = (H - L).replace(0, np.nan)
factors["nbody_1d"] = (C - L) / rng - 0.5
factors["vol_of_vol20x60"] = R.rolling(20).std().rolling(60).std()
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
print(f"{'factor':22s} {'h1 ic':>9s} {'icir':>7s} {'hit':>6s} {'n':>5s} | {'365d ic':>9s} {'icir':>7s} {'n':>5s} | {'120d ic':>9s} {'icir':>7s} {'n':>5s} | gate")
for name, f in factors.items():
    row = {}
    for h in (1, 2, 3, 5):
        ic_ser = daily_ic_series(f, h)
        row[h] = dict(n=len(ic_ser), ic=float(ic_ser.mean()), icir=float(ic_ser.mean() / ic_ser.std()),
                      hit=float((ic_ser > 0).mean()) if len(ic_ser) else float('nan'), last=(ic_ser.index.max().date() if len(ic_ser) else None))
    ic1 = daily_ic_series(f, 1)
    out[name] = row
    last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
    last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
    ok = (abs(row[1]['ic']) >= gate_ic) and (abs(row[1]['icir']) >= gate_icir)
    n365 = len(last365); n120 = len(last120)
    ic365 = last365.mean() if n365 else np.nan
    icir365 = last365.mean() / last365.std() if n365 > 1 else np.nan
    ic120 = last120.mean() if n120 else np.nan
    icir120 = last120.mean() / last120.std() if n120 > 1 else np.nan
    print(f"{name:22s} {row[1]['ic']:+9.5f} {row[1]['icir']:+7.4f} {row[1]['hit']:6.3f} {row[1]['n']:5d} | "
          f"{ic365:+9.5f} {icir365:+7.4f} {n365:5d} | {ic120:+9.5f} {icir120:+7.4f} {n120:5d} | {ok}")

json.dump(out, open("scripts/miner3_20290413_revalidate_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner3_20290413_revalidate_results.json")
