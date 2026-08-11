"""Miner2 cycle 7: re-validate core reversal family + novel candidates.

Goal: (1) re-validate ensemble-referenced factors (rev_1d, nclv_*, nbody_1d,
id_rev_1d, rev_1d_vs, rev_2d/3d) on the full sample; (2) screen novel
candidates (vol-scaled reversal, gap/intraday decomposition, cross-asset
betas: US10Y/CN10Y diff, XAU beta, BTC beta, vol term-structure, drawdown).

Admission gates (15-name cross-asset universe, daily paper IC):
    |IC1| >= 0.0070  and  |ICIR1| >= 0.0840
Validation window: 2021-01-01 .. 2026-07-15.
Output: scripts/miner2_cycle7_results.json
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

T0 = time.time()
idx = None
closes = load_close()
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
N_CELLS = len(idx) * len(SYMBOLS)
print(f"panel: {len(idx)} dates {idx.min().date()}..{idx.max().date()} x {len(SYMBOLS)} syms "
      f"({time.time()-T0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}

def rolling_beta(ret_s, mkt_s, win, min_obs):
    out = np.full(len(ret_s), np.nan)
    r, m = ret_s.values, mkt_s.values
    for i in range(win, len(r)):
        a, b = r[i - win:i], m[i - win:i]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_obs:
            continue
        aa, bb = a[ok], b[ok]
        if aa.std() <= 1e-12 or bb.std() <= 1e-12:
            continue
        out[i] = np.cov(aa, bb)[0, 1] / bb.var()
    return out

def run(name, panel):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    return {"name": name, "panel": panel, "cov": cov, "to": to, "ic1": ic1,
            "ic5": ic5, "ic10": ic10, "passed": passed}

cands = {}

# ---- A. core reversal family (re-validation) ----
for nd in (1, 2, 3, 5):
    cands[f"rev_{nd}d"] = -np.log(CP / CP.shift(nd))
for nd in (1, 2, 3, 5):
    hmax = HP.rolling(nd).max()
    lmin = LP.rolling(nd).min()
    rng = (hmax - lmin).replace(0, np.nan)
    cands[f"nclv_{nd}d"] = -(CP - lmin) / rng
rng1 = (HP - LP).replace(0, np.nan)
cands["nbody_1d"] = -(CP - OP) / rng1
cands["id_rev_1d"] = -(CP / OP - 1.0)
vol20 = RET.rolling(20).std()
cands["rev_1d_vs"] = -np.log(CP / CP.shift(1)) / (vol20 + 1e-12)

# ---- B. novel candidates ----
cands["rev_vol_1d"] = -np.log(CP / CP.shift(1)) / (vol20 + 1e-12) * (vol20 / vol20.rolling(60).mean() + 1e-12)  # vol-scaled rev x vol-regime
cands["nbody_5d"] = -(CP - OP).rolling(5).mean() / (HP - LP).rolling(5).mean().replace(0, np.nan)
gap = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0
cands["ngap_5d"] = -gap.rolling(5, min_periods=3).mean()
cands["nintra_5d"] = -intra.rolling(5, min_periods=3).mean()
cands["wick_up_1d"] = (HP - CP) / rng1
cands["wick_dn_1d"] = (LP - CP) / rng1  # negative lower wick => (CP-LP)/range, contrarian lower-shadow
# cross-asset betas (on panel rets)
for mkt, win in [("XAU", 60), ("BTC", 60), ("US10Y", 60), ("CN10Y", 60)]:
    mr = RET[mkt]
    beta = pd.DataFrame({s: rolling_beta(RET[s], mr, win, max(5, win // 2)) for s in SYMBOLS}, index=idx)
    cands[f"beta_{mkt.lower()}_{win}d"] = beta
b60 = pd.DataFrame({s: rolling_beta(RET[s], RET["US10Y"], 60, 30) for s in SYMBOLS}, index=idx)
bc60 = pd.DataFrame({s: rolling_beta(RET[s], RET["CN10Y"], 60, 30) for s in SYMBOLS}, index=idx)
cands["bond_beta_diff_60"] = b60 - bc60
# vol term structure
v5 = RET.rolling(5, min_periods=3).std()
v20b = RET.rolling(20, min_periods=10).std()
cands["nvol_ratio_5_20"] = -(v5 / (v20b + 1e-12))
# drawdown contrarian
cands["ndist_60_high"] = 1.0 - CP / CP.rolling(60, min_periods=30).max()
# volume-confirmed reversal
volz = (VOL - VOL.rolling(20).mean()) / (VOL.rolling(60).std() + 1e-9)
cands["rev_volz_1d"] = -np.log(CP / CP.shift(1)) * (1.0 + 0.5 * volz)

res = {}
for nm, p in cands.items():
    res[nm] = run(nm, p)

print(f"\n{'name':16s} {'cov':>6s} {'to10':>6s} {'IC1':>8s} {'ICIR1':>7s} {'hit1':>5s} {'n1':>5s} "
      f"{'IC5':>8s} {'IC10':>8s}  gate")
for nm in sorted(res):
    r = res[nm]
    print(f"{nm:16s} {r['cov']:6.3f} {r['to']:6.2f} {r['ic1']['ic']:+8.4f} {r['ic1']['icir']:+7.3f} "
          f"{r['ic1']['hit']:5.2f} {r['ic1']['n_dates']:5d} {r['ic5']['ic']:+8.4f} "
          f"{r['ic10']['ic']:+8.4f}  {'PASS' if r['passed'] else 'fail'}")

passers = {n: r for n, r in res.items() if r["passed"]}
print(f"\n{len(passers)} passers: {sorted(passers)}")

# ---- deep validation for passers ----
extra = {}
for nm in passers:
    dec = F.fast_ic_all(passers[nm]["panel"].reindex(idx), closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    yr = {}
    for yr_i in range(2021, 2027):
        lo = pd.Timestamp(f"{yr_i}-01-01")
        hi = pd.Timestamp(f"{yr_i}-12-31") if yr_i < 2026 else pd.Timestamp("2026-12-31")
        m = (idx >= lo) & (idx <= hi)
        sub = passers[nm]["panel"].reindex(idx[m])
        r = F.fast_ic(sub, fwd[1].reindex(idx[m]))
        yr[yr_i] = {"ic": round(r["ic"], 4), "icir": round(r["icir"], 3), "n": r["n_dates"]}
    extra[nm] = {"decay": {k: round(v["ic"], 4) for k, v in dec.items()}, "by_year": yr}
    print(f"{nm:12s} decay1/2/3/5/10/20 = "
          f"{[extra[nm]['decay'].get(h) for h in (1,2,3,5,10,20)]}")

# ---- library correlation (recompute 4 current library signals) ----
lib = {}
lib["mom_10d_skip5"] = CP.shift(5) / CP.shift(15) - 1.0
lib["mom_120d_skip5"] = CP.shift(5) / CP.shift(125) - 1.0
lib["vol_of_vol20x60"] = RET.rolling(20).std().rolling(60).std()
import pandas as pd
vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.set_index("date")["close"].reindex(idx)
vix_ret = vix.pct_change()
vixz = vix / vix.shift(20) - 1.0
beta_vix = pd.DataFrame({s: rolling_beta(RET[s], vix_ret, 60, 30) for s in SYMBOLS}, index=idx)
lib["vix_beta_cond_60x20"] = -beta_vix * vixz

def stacked_rho(a, b):
    a = a.stack(); b = b.stack()
    common = a.index.intersection(b.index)
    if len(common) < 30:
        return np.nan
    return float(np.corrcoef(a[common].values, b[common].values)[0, 1])

lib_corr = {}
for nm in passers:
    rhos = {lf: round(stacked_rho(passers[nm]["panel"], libsig), 3) for lf, libsig in lib.items()}
    valid = [v for v in rhos.values() if np.isfinite(v)]
    lib_corr[nm] = {"per_factor": rhos, "max_abs": round(max(abs(v) for v in valid), 3) if valid else np.nan}
    print(f"{nm:12s} lib_rho(max)={lib_corr[nm]['max_abs']} per={rhos}")

out = {nm: {"cov": res[nm]["cov"], "to": res[nm]["to"],
            "ic1": res[nm]["ic1"], "ic5": res[nm]["ic5"], "ic10": res[nm]["ic10"],
            "passed": res[nm]["passed"], "extra": extra.get(nm), "lib_corr": lib_corr.get(nm)}
       for nm in res}
json.dump(out, open("scripts/miner2_cycle7_results.json", "w"), indent=1, default=str)
print(f"\nsaved scripts/miner2_cycle7_results.json | total {time.time()-T0:.1f}s")
