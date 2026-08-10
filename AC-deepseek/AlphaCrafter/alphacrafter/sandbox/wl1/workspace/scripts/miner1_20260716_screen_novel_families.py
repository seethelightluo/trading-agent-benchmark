"""miner_1: broad screen of NOVEL factor families (2026-07-16 cycle).

Goal: find factors passing admission gate (|IC1|>=0.007, |ICIR1|>=0.084) that are
NOT correlated >0.5 with the already-quarantined short-horizon reversal panels
(rev_1d/2d/3d/5d, id_rev_1d, nbody_1d, nclv_1d..5d, clv_5d, rev_1d_vs).

Validation window: 2021-01-01 .. 2026-07-15 (2020 used for warm-up windows).
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

T0 = time.time()
VALID_DATE = "2026-07-15"
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= pd.Timestamp("2026-07-15"))]

OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
VAL = idx[idx >= pd.Timestamp("2021-01-01")]
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} "
      f"(val {VAL.min().date()}..{VAL.max().date()}) [{time.time()-T0:.1f}s]")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(VAL) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

# ---- quarantined library panels (real signal reconstruction, provenance only) ----
lib = {}
for nd in (1, 2, 3, 5):
    lib[f"rev_{nd}d"] = -np.log(CP / CP.shift(nd))
for nd in (1, 2, 3, 5):
    hmax = HP.rolling(nd).max(); lmin = LP.rolling(nd).min()
    lib[f"nclv_{nd}d"] = -(CP - lmin) / (hmax - lmin).replace(0, np.nan)
rng1 = (HP - LP).replace(0, np.nan)
lib["nbody_1d"] = -(CP - OP) / rng1
lib["id_rev_1d"] = -(CP / OP - 1.0)
lib["rev_1d_vs"] = -LOG / (RET.rolling(20).std() + 1e-12)
lib["clv_5d"] = -np.log(CP / CP.shift(5)) / (RET.rolling(5).std() + 1e-12)


def panel_corr(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    if int(m.sum()) < 50:
        return np.nan
    x = A[m]; y = B[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cs_z(panel):
    m = panel.mean(axis=1)
    s = panel.std(axis=1)
    return panel.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)


# ================= candidate construction (novel families) =================
cands = {}

# ---- A. Trend / momentum (risk-adjusted, non-reversal) ----
v5 = RET.rolling(5).std() * np.sqrt(252)
v10 = RET.rolling(10).std() * np.sqrt(252)
v20 = RET.rolling(20).std() * np.sqrt(252)
v60 = RET.rolling(60).std() * np.sqrt(252)
v120 = RET.rolling(120).std() * np.sqrt(252)

cands["mom10_vs"] = (CP / CP.shift(10) - 1.0) / (v20 + 1e-12)
cands["mom20_vs"] = (CP / CP.shift(20) - 1.0) / (v20 + 1e-12)
cands["mom60_vs"] = (CP / CP.shift(60) - 1.0) / (v60 + 1e-12)
cands["mom20_60"] = (CP / CP.shift(20) - 1.0) - (CP / CP.shift(60) - 1.0)
cands["ma20_dist"] = CP / CP.rolling(20).mean() - 1.0
cands["ma60_dist"] = CP / CP.rolling(60).mean() - 1.0
cands["macd_hist_n"] = (CP.ewm(span=12, adjust=False).mean() - CP.ewm(span=26, adjust=False).mean()) / (v20 + 1e-12)
cands["eff_ratio_60"] = (CP - CP.shift(60)).abs() / (LOG.abs().rolling(60).sum() + 1e-12)
cands["high20_prox"] = CP / HP.rolling(20).max() - 1.0
cands["high120_prox"] = CP / HP.rolling(120).max() - 1.0

# ---- B. Volatility family (level / term-structure / z-score) ----
cands["vol_ts_10_60"] = v10 / (v60 + 1e-12)
cands["vol_ts_20_60"] = v20 / (v60 + 1e-12)
cands["vol_z_20_120"] = (v20 - v20.rolling(120).mean()) / (v20.rolling(120).std() + 1e-12)
cands["vol_z_60_240"] = (v60 - v60.rolling(240).mean()) / (v60.rolling(240).std() + 1e-12)
cands["vov_20_120"] = (v20.rolling(120).std()) / (v20.rolling(120).mean() + 1e-12)

# ---- C. Cross-sectional relative factors (not pure reversal) ----
cands["rel_mom20"] = (CP / CP.shift(20) - 1.0) - (CP / CP.shift(20) - 1.0).mean(axis=1)
cands["cs_z_mom20"] = cs_z(CP / CP.shift(20) - 1.0)
cands["cs_z_vol20"] = cs_z(-v20)          # low relative vol -> high value

# ---- D. Macro-beta factors (observation-only macro -> per-asset sensitivities) ----
def load_macro(name):
    d = pd.read_csv(os.path.join("../persistent/index_data", f"{name}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= pd.Timestamp("2026-07-15")].sort_values("date").set_index("date")
    return d["close"].astype(float).reindex(idx)

DXY = load_macro("DXY")
VIX = load_macro("VIX")
JPY = load_macro("USDJPY")
CNY = load_macro("USDCNY")
EUR = load_macro("EURUSD")

def beta_to(sym_ret, macro_ret, nd=60):
    """Rolling beta of each asset's returns to macro returns."""
    c = RET.rolling(nd).corr(macro_ret)
    s_macro = macro_ret.rolling(nd).std()
    s_asset = RET.rolling(nd).std()
    return c * (s_asset / (s_macro + 1e-12))

dxy_r = np.log(DXY / DXY.shift(1))
vix_r = np.log(VIX / VIX.shift(1))
jpy_r = np.log(JPY / JPY.shift(1))
cny_r = np.log(CNY / CNY.shift(1))
eur_r = np.log(EUR / EUR.shift(1))

# factor = per-asset beta to macro x macro momentum (cross-sectional macro exposure)
cands["dxy_mom20_beta"] = beta_to(RET, dxy_r, 60) * (DXY / DXY.shift(20) - 1.0)
cands["vix_chg5_beta"] = -beta_to(RET, vix_r, 60) * (VIX / VIX.shift(5) - 1.0)
cands["jpy_mom20_beta"] = beta_to(RET, jpy_r, 60) * (JPY / JPY.shift(20) - 1.0)
cands["cny_mom20_beta"] = beta_to(RET, cny_r, 60) * (CNY / CNY.shift(20) - 1.0)
cands["eur_mom20_beta"] = beta_to(RET, eur_r, 60) * (EUR / EUR.shift(20) - 1.0)

# ---- E. Yield-curve / rates factors ----
us10 = CP["US10Y"]; cn10 = CP["CN10Y"]
spread = us10 - cn10
cands["us10y_mom20"] = (us10 / us10.shift(20) - 1.0)
cands["cn10y_mom20"] = (cn10 / cn10.shift(20) - 1.0)
cands["yld_spread_z"] = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-12)

# ---- F. Intraday structure (position in day range) ----
cands["range_pos_1d"] = (CP - LP) / (HP - LP).replace(0, np.nan) - 0.5
cands["range_pos_5d"] = (CP - LP.rolling(5).min()) / (HP.rolling(5).max() - LP.rolling(5).min()).replace(0, np.nan) - 0.5

# ---- G. Conditional reversal variants (gated, distinct from raw reversal) ----
cands["rev1_lowvol"] = -LOG * (v20 < v60).astype(float)          # reversal only in calm regime
cands["rev1_hi_eff"] = -LOG * (cands["eff_ratio_60"] > 0.3).astype(float)  # reversal in efficient trends
cands["rev5_ma20up"] = -(CP / CP.shift(5) - 1.0) * (CP > CP.rolling(20).mean()).astype(float)

# ================= evaluation =================
def run(name, panel):
    panel = panel.reindex(idx)
    cov = float(panel.reindex(VAL).notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ics = {h: F.fast_ic(panel, fwd[h]) for h in (1, 2, 3, 5, 10, 20, 30)}
    ic1 = ics[1]
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    corrs = [panel_corr(panel, lv) for lv in lib.values()]
    corrs = [c for c in corrs if c is not None and np.isfinite(c)]
    maxc = max(abs(c) for c in corrs) if corrs else np.nan
    div = passed and (maxc < 0.50 if np.isfinite(maxc) else False)
    dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
    print(f"{name:16s} cov={cov:.2f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | maxLibCorr={maxc:.2f} | {dec} | "
          f"{'PASS+DIV' if div else ('PASS' if passed else 'fail')}")
    return {"name": name, "panel": panel, "cov": cov, "to": to, "ic": ics,
            "passed": passed, "diverse": div, "max_lib_corr": maxc}


res = {}
for nm, p in cands.items():
    try:
        res[nm] = run(nm, p)
    except Exception as e:
        print(f"{nm}: ERROR {e}")

divs = {k: v for k, v in res.items() if v["diverse"]}
print(f"\nTotal: {len(cands)} | PASS gate: {sum(r['passed'] for r in res.values())} | "
      f"PASS and diverse (maxLibCorr<0.5): {len(divs)} -> {list(divs.keys())}")

# by-year IC1 for diverse passers
for nm in divs:
    p = divs[nm]["panel"]
    yr = {}
    for y in range(2021, 2027):
        lo = pd.Timestamp(f"{y}-01-01"); hi = pd.Timestamp(f"{y}-12-31")
        m = (idx >= lo) & (idx <= hi)
        r = F.fast_ic(p.reindex(idx[m]), fwd[1].reindex(idx[m]))
        yr[y] = {"ic": round(r["ic"], 4), "icir": round(r["icir"], 3), "n": r["n_dates"]}
    print(f"{nm:16s} by_year={yr}")

import pickle
with open("scripts/_miner1_screen_families.pkl", "wb") as fh:
    pickle.dump({k: v["panel"] for k, v in res.items()}, fh)
print(f"\ndone [{time.time()-T0:.1f}s]")
