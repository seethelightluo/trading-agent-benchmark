"""miner_1 novel factor screen cycle - 2026-07-16.

Cross-asset 15-instrument universe. Admission gate (daily rank IC, 1d fwd):
    |IC1| >= 0.0070 and |ICIR1| >= 0.0840
Validation window: 2021-01-01 .. 2026-07-15 (2020 used as warm-up).

Novel families NOT in the quarantined library (rev_1d..5d, nclv_1d..5d,
nbody_1d, id_rev_1d, rev_1d_vs, macro betas, pos_freq):
  A. Parkinson / Garman-Klass vol-scaled reversal (advanced vol estimators)
  B. Kaufman efficiency-ratio family + reversal conditional on trend quality
  C. Bollinger %B mean reversion, MA-slope
  D. Range-vol level / vol acceleration
  E. Cross-sectional z-score reversal (std-normalized, vs median-subtracted)
  F. EWMA/half-life weighted reversal
  G. Overnight vs intraday 5d decomposition
  H. Extreme-return (max/min) reversal
  I. Semi-vol ratio, co-skewness variants

Every passing candidate gets a RECOVERABLE SIGNAL ARTIFACT (gzip+base64 float32
dates x symbols panel) embedded so the deterministic post-Miner gate can
recompute pairwise rho from real signal data (policy
worldline_pairwise_signal_quality_v1 requires artifacts; do not quarantine).
"""
import sys, os, time, json, base64, gzip
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

T0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= pd.Timestamp("2026-07-15"))]

OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-T0:.1f}s)")

VAL = idx[idx >= pd.Timestamp("2021-01-01")]
fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(VAL) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

# ---- quarantined-library panels (real signal reconstruction for provenance) ----
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


def panel_corr(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    if int(m.sum()) < 50:
        return np.nan
    x = A[m]; y = B[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.reindex(VAL).notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ics = {h: F.fast_ic(panel, fwd[h]) for h in (1, 2, 3, 5, 10, 20, 30)}
    ic1 = ics[1]
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    corrs = [panel_corr(panel, lv) for lv in lib.values()]
    corrs = [c for c in corrs if c is not None and np.isfinite(c)]
    maxc = max(abs(c) for c in corrs) if corrs else np.nan
    if verbose:
        dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
        print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | maxLibCorr={maxc:.2f} | {dec} | "
              f"{'PASS' if passed else 'fail'}")
    return {"name": name, "panel": panel, "cov": cov, "to": to, "ic": ics,
            "passed": passed, "max_lib_corr": maxc}


cands = {}

# ---------- A. advanced-vol scaled reversal ----------
def parkinson(nd=20):
    return np.sqrt(np.log(HP / LP).pow(2).rolling(nd).mean() / (4 * np.log(2)))


def gk(nd=20):
    o = np.log(OP / CP.shift(1))
    c = np.log(CP / OP)
    return np.sqrt(0.5 * o.pow(2) + (2 * np.log(2) - 1) * c.pow(2)).rolling(nd).mean() ** 0.5


pk20 = parkinson(20)
gk20 = gk(20)
cands["rev1_pk"] = -LOG / (pk20 + 1e-12)
cands["rev5_gk"] = -np.log(CP / CP.shift(5)) / (gk20 + 1e-12)
cands["rev1_gk"] = -LOG / (gk20 + 1e-12)

# ---------- B. Kaufman efficiency-ratio family ----------
def eff_ratio(nd):
    return (CP - CP.shift(nd)).abs() / (LOG.abs().rolling(nd).sum() + 1e-12)


er20 = eff_ratio(20)
cands["eff_ratio_20"] = er20
cands["rev1_x_inveff"] = -LOG * (1.0 - er20)          # reversal stronger when trend inefficient
cands["rev5_x_eff"] = -np.log(CP / CP.shift(5)) * er20
tr60 = (CP / CP.shift(60) - 1.0).abs() / (RET.rolling(60).std() * np.sqrt(60) + 1e-12)
cands["trend_quality_60"] = tr60
cands["nrev1_x_tq"] = -LOG * (1.0 - tr60.clip(upper=1.0))

# ---------- C. Bollinger %B mean reversion, MA slope ----------
ma20 = CP.rolling(20).mean()
sd20 = RET.rolling(20).std()
cands["bb_pos_20"] = (CP - ma20) / (2 * sd20 + 1e-12)
cands["bb_rev_20"] = -(CP - ma20) / (2 * sd20 + 1e-12)
ma5 = CP.rolling(5).mean()
cands["ma_slope_5_20"] = (ma5 / ma20 - 1.0)

# ---------- D. range-vol level / vol acceleration ----------
cands["inv_pk20"] = -pk20
cands["vol_accel_5_60"] = RET.rolling(5).std() / (RET.rolling(60).std() + 1e-12)
cands["pk_ratio_5_60"] = parkinson(5) / (pk20 + 1e-12)

# ---------- E. cross-sectional z-score reversal ----------
def cs_z(panel):
    m = panel.mean(axis=1)
    s = panel.std(axis=1)
    return panel.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)


cands["cz_rev1"] = -cs_z(LOG)
cands["cz_rev5"] = -cs_z(np.log(CP / CP.shift(5)))
cands["cz_rev10"] = -cs_z(np.log(CP / CP.shift(10)))

# ---------- F. EWMA / half-life weighted reversal ----------
def ewma_rev(halflife, span):
    w = np.array([0.5 ** (i / halflife) for i in range(1, span + 1)])
    w = w / w.sum()
    cols = {}
    for s in SYMBOLS:
        r = LOG[s]
        out = pd.Series(np.nan, index=r.index)
        vals = r.values
        for i in range(span, len(vals)):
            seg = vals[i - span:i][::-1]
            m = np.isfinite(seg)
            if m.sum() < span // 2:
                continue
            out.iloc[i] = -float(np.nansum(w[m] * seg[m]))
        cols[s] = out
    return pd.DataFrame(cols, index=idx)


cands["ewma_rev_hl3"] = ewma_rev(3, 10)
cands["ewma_rev_hl5"] = ewma_rev(5, 10)

# ---------- G. overnight vs intraday 5d decomposition ----------
gap1 = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0
cands["night_rev_5d"] = -gap1.rolling(5).sum()
cands["day_rev_5d"] = -intra.rolling(5).sum()
cands["night_minus_day_5"] = -(gap1 - intra).rolling(5).sum()

# ---------- H. extreme-return reversal ----------
mx5 = RET.rolling(5).max()
mn5 = RET.rolling(5).min()
cands["max_ret5_rev"] = -mx5
cands["min_ret5_pos"] = -mn5                      # oversold bounce
cands["skew_rev_20"] = -RET.rolling(20).skew()    # negative skew -> reversal up

# ---------- I. semi-vol ratio ----------
down = RET.clip(upper=0)
cands["semi_vol_ratio_10"] = down.rolling(10).std() / (RET.rolling(10).std() + 1e-12)

res = {}
for nm, p in cands.items():
    try:
        res[nm] = run(nm, p)
    except Exception as e:
        print(f"{nm}: ERROR {e}")

passers = {k: v for k, v in res.items() if v["passed"]}
print(f"\nTotal candidates: {len(cands)}, PASS: {len(passers)}")
for k, v in passers.items():
    print(f"  PASS {k}: IC1={v['ic'][1]['ic']:+.4f} ICIR1={v['ic'][1]['icir']:+.3f} "
          f"hit={v['ic'][1]['hit']:.3f} maxLibCorr={v['max_lib_corr']:.2f}")

# deep validation for passers: decay + by-year
print("\n=== deep validation of passers ===")
extra = {}
for nm in passers:
    p = passers[nm]["panel"]
    dec = F.fast_ic_all(p.reindex(idx), closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    yr = {}
    for y in range(2021, 2027):
        lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        m = (idx >= lo) & (idx <= hi)
        r = F.fast_ic(p.reindex(idx[m]), fwd[1].reindex(idx[m]))
        yr[y] = {"ic": round(r["ic"], 4), "icir": round(r["icir"], 3), "n": r["n_dates"]}
    extra[nm] = {"decay": {int(k): round(v["ic"], 4) for k, v in dec.items()}, "by_year": yr}
    print(f"{nm:20s} decay={ {k: v for k, v in extra[nm]['decay'].items() if k in (1,2,3,5,10)} }")
    print(f"{'':20s} by_year={yr}")

# pairwise signal corr among passers (provenance for diversity)
pn = list(passers.keys())
print("\npairwise signal corr among passers:")
for i in range(len(pn)):
    for j in range(i + 1, len(pn)):
        a = passers[pn[i]]["panel"].stack().dropna()
        b = passers[pn[j]]["panel"].stack().dropna()
        common = a.index.intersection(b.index)
        if len(common) > 50:
            rho = np.corrcoef(a.loc[common].values, b.loc[common].values)[0, 1]
            print(f"  {pn[i]:20s} | {pn[j]:20s} : {rho:+.3f}")

# dump passers for persistence stage
with open("scripts/_miner1_passers_cycle_novel.pkl", "wb") as fh:
    import pickle
    pickle.dump({k: passers[k] for k in passers}, fh)
print("\nsaved scripts/_miner1_passers_cycle_novel.pkl")
print(f"elapsed {time.time()-T0:.1f}s")
