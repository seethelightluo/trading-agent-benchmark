"""miner_3 cycle-5 screening: non-reversal families (liquidity, range, beta, tail, gap).

Tests 12 candidates on the 15-name cross-asset panel, 2021-01-04..2026-07-15.
Admission gate: |daily rank IC1| >= 0.0070 and |ICIR1| >= 0.0840.
Diversity gate: pooled rank corr vs effective library < 0.5 (reported, not enforced here).
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
ev_idx = idx[idx >= EVAL_START]

# --- macro (observation-only) ---
vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix[vix["date"] <= END].sort_values("date").set_index("date")
VIX = vix["close"].reindex(idx).astype(float)

logvol = np.log1p(VO.replace(0, np.nan))
vol20 = LRET.rolling(20).std() * np.sqrt(252)

# --- candidate factors (positive predicts higher next-day return; sign chosen after inspection) ---
cands = {}

# 1 volume z-score (60d): liquidity pressure
cands["vol_z_60"] = (logvol - logvol.rolling(60).mean()) / logvol.rolling(60).std()

# 2 Parkinson efficiency: Park(20) / close-close vol(20)
park = np.sqrt((np.log(HI / LO) ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
cands["park_eff_20"] = park / vol20

# 3 range ratio 20d
cands["range_ratio_20"] = ((HI - LO) / CP).rolling(20).mean()

# 4 max daily return 20d (lottery)
cands["max_ret_20"] = RET.rolling(20).max()

# 5 min daily return 20d (crash sensitivity)
cands["min_ret_20"] = RET.rolling(20).min()

# 6 beta vs SPX 60d (per-column rolling to keep 15-col panel)
spx_ret = LRET["SPX"]
spx_var60 = spx_ret.rolling(60).var()
beta_spx = pd.DataFrame({s: LRET[s].rolling(60).cov(spx_ret) / spx_var60 for s in SYMBOLS})
cands["beta_spx_60"] = beta_spx

# 7 corr vs SPX 60d
corr_spx = pd.DataFrame({s: LRET[s].rolling(60).corr(spx_ret) for s in SYMBOLS})
cands["corr_spx_60"] = corr_spx

# 8 corr vs VIX daily change 60d
dVIX = VIX.pct_change()
vix_corr = pd.DataFrame({s: LRET[s].rolling(60).corr(dVIX) for s in SYMBOLS})
cands["vix_corr_60"] = vix_corr

# 9 overnight gap reversal: -(open/prev_close - 1)
gap = OP / CP.shift(1) - 1.0
cands["gap_rev_1d"] = -gap

# 10 vol trend: vol20/vol20.shift(20) - 1
cands["vol_trend_20"] = vol20 / vol20.shift(20) - 1.0

# 11 risk-adjusted momentum: mom(20,skip5)/vol20
mom20 = CP / CP.shift(5) - 1.0
mom20 = (CP / CP.shift(20) - 1.0)
cands["mom_risk_adj_20"] = (CP / CP.shift(20) - 1.0) / vol20

# 12 intraday reversal x VIX-high regime
vix_hi = (VIX > VIX.rolling(60).median()).astype(float)
cands["rev_intra_x_vixhi"] = (1.0 - CP / OP).mul(vix_hi, axis=0)

# --- eval helpers ---
def row_ic(a, b):
    out = []
    for i in range(len(a)):
        x = a.iloc[i].to_numpy(dtype=float)
        y = b.iloc[i].to_numpy(dtype=float)
        m = (~np.isnan(x)) & (~np.isnan(y))
        if m.sum() < 8:
            out.append(np.nan)
            continue
        out.append(spearmanr(x[m], y[m]).statistic)
    return np.array(out)

def ic_stats(fac):
    ev = fac.loc[ev_idx]
    frr = fwd1.loc[ev.index].rank(axis=1)
    fr = ev.rank(axis=1)
    s = row_ic(fr, frr)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return None
    ic = float(s.mean())
    icir = ic / float(s.std(ddof=1)) if s.std(ddof=1) > 1e-12 else 0.0
    return dict(ic=ic, icir=icir, hit=float((s > 0).mean()), n=int(len(s)),
                cov=float(ev.notna().mean().mean()))

# --- library artifacts for diversity ---
def load_artifact(j):
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

LIB_FILES = ["factors/miner2_20260716_mom_10d_skip5.json",
             "factors/miner3_20260716_rev_intraday_1d.json",
             "factors/miner3_20260716_rev_intra_x_volrank.json",
             "factors/miner3_20260716_volz_20.json"]
lib_arrs = []
for f in LIB_FILES:
    if not os.path.exists(f):
        print("  [warn] missing lib", f)
        continue
    j = json.load(open(f))
    arr = load_artifact(j)
    if arr is None:
        print("  [warn] no artifact", f)
        continue
    a = j.get("signal_artifact")
    if isinstance(a, dict) and a.get("n_dates") == len(ev_idx):
        lib = pd.DataFrame(arr, index=ev_idx, columns=SYMBOLS)
    elif isinstance(a, str) and a.endswith(".npy") and arr.shape[0] >= len(idx):
        rows = (ev_idx - idx[0]).days.to_numpy()  # fallback: day offsets from panel start
        lib = pd.DataFrame(arr[rows], index=ev_idx, columns=SYMBOLS)
    else:
        print("  [warn] cannot align", f, arr.shape)
        continue
    lib_arrs.append(lib)
print(f"loaded {len(lib_arrs)} library artifacts for diversity check")

def lib_max_corr(fac):
    ev = fac.loc[ev_idx]
    fv = ev.values.ravel()
    mx = 0.0
    for lib in lib_arrs:
        lv = lib.values.ravel()
        m = (~np.isnan(fv)) & (~np.isnan(lv))
        if m.sum() < 500:
            continue
        rho = spearmanr(fv[m], lv[m]).statistic
        mx = max(mx, abs(float(rho)))
    return mx

# --- run ---
rows = []
for name, fac in cands.items():
    print('RUNNING', name, fac.shape, file=sys.stderr)
    st = ic_stats(fac)
    if st is None:
        print(f"{name:20s} no data")
        continue
    rk = fac.rank(axis=1, pct=True)
    turn = float((rk.loc[ev_idx] - rk.loc[ev_idx].shift(10)).abs().mean().mean())
    lm = lib_max_corr(fac)
    rows.append((name, st["ic"], st["icir"], st["hit"], st["n"], st["cov"], turn, lm))
    flag = "PASS" if (abs(st["ic"]) >= GATE_IC and abs(st["icir"]) >= GATE_ICIR) else "    "
    print(f"{flag} {name:20s} IC1={st['ic']:+.4f} ICIR1={st['icir']:+.3f} hit={st['hit']:.3f} "
          f"n={st['n']} cov={st['cov']:.3f} turn={turn:.3f} lib_max={lm:.3f}")

print("\n--- summary (passing gates) ---")
for r in sorted(rows, key=lambda x: -abs(x[1])):
    name, ic, icir, hit, n, cov, turn, lm = r
    if abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR:
        print(f"{name:20s} IC1={ic:+.4f} ICIR1={icir:+.3f} hit={hit:.3f} n={n} cov={cov:.3f} "
              f"turn={turn:.3f} lib_max={lm:.3f}  {'DIVERSE' if lm < 0.5 else 'CORRELATED'}")

with open("scripts/miner3_screen_cycle5_results.json", "w") as fh:
    json.dump({r[0]: dict(ic=r[1], icir=r[2], hit=r[3], n=r[4], cov=r[5], turn=r[6], lib_max=r[7])
               for r in rows}, fh, indent=1)
print("saved scripts/miner3_screen_cycle5_results.json")
