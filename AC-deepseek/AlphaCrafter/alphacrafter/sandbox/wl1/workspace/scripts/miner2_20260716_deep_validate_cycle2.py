"""Miner2 cycle2 follow-up: fix macro-beta via per-column loops, deep-validate passers
(decay, by-year), and compute pairwise signal correlations among passers to select a
diverse, non-redundant persistence set (library correlation gate ~0.5)."""
import sys, time, os
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner1_common import SYMBOLS, load_close, MACRO, IDX_DIR
import miner2_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))

macro = {}
for m in MACRO:
    d = pd.read_csv(os.path.join(IDX_DIR, f"{m}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    macro[m] = pd.to_numeric(d.set_index("date")["close"].reindex(idx), errors="coerce").astype(float)
DXY_R = macro["DXY"].pct_change()
VIX_C = macro["VIX"].diff()
JPY_R = macro["USDJPY"].pct_change()

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})


def rolling_beta(ret_df, ref_series, nd):
    """per-column rolling beta of each asset's ret on ref_series changes"""
    cols = {}
    for s in SYMBOLS:
        x = ref_series
        y = ret_df[s]
        df = pd.concat([x, y], axis=1).dropna()
        cov = df[y.name].rolling(nd).cov(df[x.name])
        var = df[x.name].rolling(nd).var()
        cols[s] = (cov / (var + 1e-12)).reindex(idx)
    return pd.DataFrame(cols, index=idx)


def run(name, panel):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    print(f"{name:24s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} | IC10={ic10['ic']:+.4f} | "
          f"{'PASS' if passed else 'fail'}")
    return {"name": name, "panel": panel, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5,
            "ic10": ic10, "passed": passed}


cands = {}
cands["beta_spx_60d"] = rolling_beta(RET, RET["SPX"], 60)
cands["nbeta_spx_60d"] = -cands["beta_spx_60d"]
cands["beta_dxy_60d"] = rolling_beta(RET, DXY_R, 60)
cands["nbeta_dxy_60d"] = -cands["beta_dxy_60d"]
cands["beta_vix_60d"] = rolling_beta(RET, VIX_C, 60)
cands["nbeta_vix_60d"] = -cands["beta_vix_60d"]
cands["beta_jpy_60d"] = rolling_beta(RET, JPY_R, 60)
cands["nbeta_jpy_60d"] = -cands["beta_jpy_60d"]

# existing passers from screen2 to deep-validate
r1 = -LOG
r3 = -np.log(CP / CP.shift(3))
r5 = -np.log(CP / CP.shift(5))
gap1 = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0
m5 = CP / CP.shift(5) - 1.0
m10 = CP / CP.shift(10) - 1.0
cands["rev_1d"] = r1
cands["rev_1_3_5"] = r1 + 0.5 * r3 + 0.25 * r5
cands["gap_adj_rev"] = -(LOG - np.log1p(gap1))
cands["oi_comp"] = -(0.5 * gap1 + 0.5 * intra)
cands["crev_5d"] = -(m5.sub(m5.median(axis=1), axis=0))
cands["ndist_5d_high"] = -(CP / CP.rolling(5).max() - 1.0)
cands["rev1_x_voltrend"] = r1 * (VOL.rolling(5).mean() / (VOL.rolling(20).mean() + 1e-9))
rng1 = (HP - LP).replace(0, np.nan)
cands["nclv_1d"] = -((CP - LP) / rng1)

res = {}
for name, panel in cands.items():
    try:
        res[name] = run(name, panel)
    except Exception as e:
        print(f"{name}: ERROR {e}")

passers = {n: r for n, r in res.items() if r["passed"]}
print(f"\npassers: {list(passers.keys())}")

# pairwise signal corr among all candidates (for redundancy analysis)
names = list(cands.keys())
print("\npairwise signal corr (signed) among passers:")
pn = list(passers.keys())
for i in range(len(pn)):
    for j in range(i + 1, len(pn)):
        a = passers[pn[i]]["panel"].stack()
        b = passers[pn[j]]["panel"].stack()
        common = a.index.intersection(b.index)
        if len(common) > 50:
            rho = np.corrcoef(a[common].values, b[common].values)[0, 1]
            print(f"  {pn[i]:18s} | {pn[j]:18s} : {rho:+.3f}")

# also corr vs rev_1d baseline for macro betas
print("\ncorr of macro betas vs rev_1d:")
for nm in ["beta_spx_60d", "nbeta_spx_60d", "beta_dxy_60d", "nbeta_dxy_60d",
           "beta_vix_60d", "nbeta_vix_60d", "beta_jpy_60d", "nbeta_jpy_60d"]:
    a = cands[nm].stack()
    b = cands["rev_1d"].stack()
    common = a.index.intersection(b.index)
    rho = np.corrcoef(a[common].values, b[common].values)[0, 1]
    print(f"  {nm:18s} vs rev_1d: {rho:+.3f}")

# deep validation for chosen set: decay + by-year
deep = ["rev_1d", "gap_adj_rev", "oi_comp", "crev_5d", "ndist_5d_high", "rev1_x_voltrend"]
print("\n=== deep validation ===")
extra = {}
for nm in deep:
    p = passers.get(nm)
    if p is None:
        continue
    dec = F.fast_ic_all(p["panel"].reindex(idx), closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    yr = {}
    for y in range(2021, 2027):
        lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        m = (idx >= lo) & (idx <= hi)
        r = F.fast_ic(p["panel"].reindex(idx[m]), fwd[1].reindex(idx[m]))
        yr[y] = {"ic": round(r["ic"], 4), "icir": round(r["icir"], 3), "n": r["n_dates"]}
    extra[nm] = {"decay": {k: round(v["ic"], 4) for k, v in dec.items()}, "by_year": yr}
    print(f"{nm:16s} decay={ {k: v for k, v in extra[nm]['decay'].items() if k in (1,2,3,5,10)} }")
    print(f"{'':16s} by_year={yr}")

print(f"\ndone {time.time()-t0:.1f}s")
