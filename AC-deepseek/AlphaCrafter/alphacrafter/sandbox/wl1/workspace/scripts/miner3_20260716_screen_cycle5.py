"""miner_3 cycle-5 screening (2026-07-16).

Families NOT yet covered by the effective library (reversal-intraday, volz,
10d momentum): return autocorrelation (trend persistence), overnight gap,
Amihud illiquidity (volume-based), 52-week-high proximity, cross-sectional
global beta, and 20d max/min return (lottery). Goal: decorrelated, robust
cross-sectional signals on the 15-instrument cross-asset universe.

Admission gates (shared benchmark): abs(daily paper rank IC) >= 0.0070 and
abs(daily paper ICIR) >= 0.0840 over 2021-01-04..2026-07-15.
"""
import sys, os, json, io, gzip, base64
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close

EVAL_START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2026-07-15")
GATE_IC, GATE_ICIR = 0.0070, 0.0840
MIN_N = 8
HORIZONS = (1, 2, 3, 5, 10, 20)

closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= END)]
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HI = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LO = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
LRET = np.log(CP / CP.shift(1))
RET = CP.pct_change()
fwd1 = RET.shift(-1)
print("dates:", len(idx), "| symbols:", len(SYMBOLS),
      "| volume coverage:", float(VO.notna().mean().mean()))

# ---------------- factor definitions ----------------
def roll_autocorr(ret, win):
    """Rolling autocorrelation of daily returns over win days."""
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for s in ret.columns:
        r = ret[s]
        mu = r.rolling(win).mean()
        cov = (r * r.shift(1)).rolling(win).mean() - mu * mu.shift(1)
        var = r.rolling(win).var()
        out[s] = cov / var
    return out

FACS = {}
FACS["autocorr_10"] = roll_autocorr(LRET, 10)
FACS["autocorr_20"] = roll_autocorr(LRET, 20)
FACS["gap_1d"] = (OP - CP.shift(1)) / CP.shift(1)
FACS["gap_20_mean"] = ((OP - CP.shift(1)) / CP.shift(1)).rolling(20).mean()
amihud = (RET.abs() / VO.replace(0, np.nan))
FACS["amihud_20"] = amihud.rolling(20).mean()
FACS["hi52_252"] = CP / CP.rolling(252).max()
cs_mean = RET.mean(axis=1)
FACS["beta_cs_60"] = RET.apply(lambda col: col.rolling(60).cov(cs_mean) / cs_mean.rolling(60).var(), axis=0)
FACS["maxret_20"] = RET.rolling(20).max()
FACS["minret_20"] = RET.rolling(20).min()
# downside semi-dev ratio at 60d (20d variant failed ICIR gate in cycle 4)
neg = RET.where(RET < 0, 0.0)
downside = np.sqrt((neg ** 2).rolling(60).mean()) * np.sqrt(252)
FACS["downside_ratio_60"] = downside / (RET.rolling(60).std() * np.sqrt(252))

# ---------------- IC machinery ----------------
def row_ic(a, b):
    out = []
    for i in range(len(a)):
        x = a.iloc[i].to_numpy(dtype=float)
        y = b.iloc[i].to_numpy(dtype=float)
        m = (~np.isnan(x)) & (~np.isnan(y))
        if m.sum() < MIN_N:
            out.append(np.nan)
            continue
        out.append(spearmanr(x[m], y[m]).statistic)
    return np.array(out)

def fwd_ret(h):
    return RET.shift(-h)

# ---------------- library artifacts (for correlation) ----------------
def load_artifact_matrix(j):
    a = j.get("signal_artifact")
    if a is None:
        return None
    if isinstance(a, str):
        if a.endswith(".npy"):
            p = a if os.path.exists(a) else os.path.join("factors", a)
            return np.load(p, allow_pickle=True)
        return None
    if isinstance(a, dict):
        data = a.get("data") or a.get("matrix") or a.get("encoded")
        if data is None:
            return None
        raw = base64.b64decode(data)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return np.load(io.BytesIO(raw))
    return None

LIB_FILES = [
    "factors/miner2_20260716_mom_10d_skip5.json",
    "factors/miner3_20260716_rev_intraday_1d.json",
    "factors/miner3_20260716_volz_20.json",
    "factors/miner3_20260716_rev_intra_x_volrank.json",
]
LIB = {}
for f in LIB_FILES:
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    arr = load_artifact_matrix(j)
    if arr is None:
        continue
    a = j.get("signal_artifact")
    if isinstance(a, dict) and a.get("n_dates") == len(idx):
        LIB[j["factor_id"]] = pd.DataFrame(arr, index=idx, columns=SYMBOLS)
    elif isinstance(a, str) and a.endswith(".npy") and arr.shape[0] >= len(idx):
        rows = (idx - pd.Timestamp("2020-01-01")).days.to_numpy()
        LIB[j["factor_id"]] = pd.DataFrame(arr[rows], index=idx, columns=SYMBOLS)
    else:
        print("skip artifact alignment for", j["factor_id"], arr.shape)
print("library artifacts loaded:", list(LIB.keys()))

# ---------------- evaluate ----------------
results = {}
for name, fac in FACS.items():
    ev = fac.loc[fac.index >= EVAL_START]
    if ev.shape[0] < 100:
        print(name, "-> insufficient eval rows", ev.shape)
        continue
    fr = ev.rank(axis=1)
    horiz = {}
    for h in HORIZONS:
        fh = fwd_ret(h).loc[ev.index].rank(axis=1)
        s = row_ic(fr, fh)
        s = s[~np.isnan(s)]
        ic = float(s.mean())
        icir = ic / float(s.std(ddof=1)) if s.std(ddof=1) > 1e-12 else 0.0
        horiz[str(h)] = {"ic": ic, "icir": icir, "hit": float((s > 0).mean()), "n": int(len(s))}
        if h == 1:
            ic1s = s
    ic1 = horiz["1"]["ic"]
    icir1 = horiz["1"]["icir"]
    cov = float(ev.notna().mean().mean())
    rk = fac.rank(axis=1, pct=True)
    turn10 = float((rk.loc[ev.index] - rk.loc[ev.index].shift(10)).abs().mean().mean())
    # by year on raw daily IC series
    by_year = {}
    ic_series = pd.Series(ic1s, index=ev.index)
    for y, grp in ic_series.groupby(ic_series.index.year):
        g = grp.dropna()
        if len(g) < 20:
            continue
        by_year[str(y)] = {"ic": float(g.mean()),
                           "icir": float(g.mean() / g.std(ddof=1)) if g.std(ddof=1) > 1e-12 else 0.0,
                           "n": int(len(g))}
    # library correlation (pooled rank corr on eval window)
    lib_max = 0.0
    lib_corr = {}
    fv = ev.values.ravel()
    for fid, lib in LIB.items():
        lv = lib.reindex(ev.index).values.ravel()
        m = (~np.isnan(fv)) & (~np.isnan(lv))
        if m.sum() < 500:
            continue
        rho = spearmanr(fv[m], lv[m]).statistic
        lib_corr[fid] = float(rho)
        lib_max = max(lib_max, abs(float(rho)))
    results[name] = {
        "horizons": horiz, "coverage": cov, "turnover_10d": turn10,
        "by_year": by_year, "lib_corr": lib_corr, "max_abs_library_correlation": lib_max,
        "pass": abs(ic1) >= GATE_IC and abs(icir1) >= GATE_ICIR,
    }
    flag = "PASS" if results[name]["pass"] else "fail"
    print(f"=== {name} [{flag}] | IC1={ic1:+.4f} ICIR1={icir1:+.3f} | cov={cov:.3f} turn10={turn10:.3f} lib_max={lib_max:.3f}")
    print("   decay:", {h: round(v["ic"], 4) for h, v in horiz.items()})
    print("   by_year:", {k: (round(v["ic"], 4), round(v["icir"], 3)) for k, v in by_year.items()})
    print("   lib_corr:", {k: round(v, 3) for k, v in lib_corr.items()})

out = "scripts/miner3_screen_cycle5_results.json"
with open(out, "w") as fh:
    json.dump(results, fh, indent=1, default=float)
print("wrote", out)
