"""miner_3 cycle-5b screening: price/return-shape families (breadth, asymmetry, tails, trend).

12 candidates, 15-name cross-asset panel, 2021-01-04..2026-07-15.
Gate: |daily rank IC1| >= 0.0070, |ICIR1| >= 0.0840; diversity reported vs effective library.
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
LRET = np.log(CP / CP.shift(1))
RET = CP.pct_change()
fwd1 = RET.shift(-1)
ev_idx = idx[idx >= EVAL_START]

SMA20 = CP.rolling(20).mean()
std20 = LRET.rolling(20).std()

cands = {}

# 1 win rate 20d: breadth of recent up-moves
cands["win_rate_20"] = (RET > 0).rolling(20).mean()

# 2 gain/loss ratio 20d
pos_m = RET.clip(lower=0).rolling(20).mean()
neg_m = (-RET.clip(upper=0)).rolling(20).mean()
cands["gain_loss_20"] = pos_m / (neg_m + 1e-12)

# 3 lag-1 autocorrelation of daily returns over 20d (continuation persistence)
cands["autocorr_1_20"] = RET.rolling(20).corr(RET.shift(1))

# 4 distance from 250d high (George-Hwang style)
cands["dist_high_250"] = CP / CP.rolling(250).max() - 1.0

# 5 SMA20 slope (5d)
cands["sma_slope_20"] = SMA20 / SMA20.shift(5) - 1.0

# 6 Bollinger %B (close vs 20d mean in units of 2*std)
cands["boll_pct_20"] = (CP - SMA20) / (2.0 * std20 + 1e-12)

# 7 RSI(14)
up = RET.clip(lower=0).rolling(14).mean()
dn = (-RET.clip(upper=0)).rolling(14).mean()
cands["rsi_14"] = 100.0 - 100.0 / (1.0 + up / (dn + 1e-12))

# 8 wick asymmetry: log(upper wick / lower wick), 20d mean
eps = 1e-9
cands["hl_asym_20"] = np.log((HI - CP + eps) / (CP - LO + eps)).rolling(20).mean()

# 9 overnight share of realized variance over 20d
ov = OP / CP.shift(1) - 1.0
idr = CP / OP - 1.0
v_ov = ov.rolling(20).var()
v_id = idr.rolling(20).var()
cands["overnight_share_20"] = v_ov / (v_ov + v_id + 1e-12)

# 10 tail ratio: 95th pct / |5th pct| of daily returns over 20d
q95 = RET.rolling(20).quantile(0.95)
q05 = RET.rolling(20).quantile(0.05)
cands["tail_ratio_20"] = q95 / (q05.abs() + 1e-12)

# 11 downside/upside vol ratio 20d
dv = RET.where(RET < 0).rolling(20).std()
uv = RET.where(RET > 0).rolling(20).std()
cands["down_up_vol_20"] = dv / (uv + 1e-12)

# 12 60d momentum skipping most recent 10d
cands["mom_60d_skip10"] = CP.shift(10) / CP.shift(60) - 1.0

# 13 short-term vol acceleration: vol5/vol20 - 1
vol5 = LRET.rolling(5).std()
vol20 = LRET.rolling(20).std()
cands["vol_accel_5_20"] = vol5 / (vol20 + 1e-12) - 1.0

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
        rho = spearmanr(x[m], y[m]).statistic
        out.append(rho)
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

# --- library artifacts ---
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
        continue
    j = json.load(open(f))
    arr = load_artifact(j)
    if arr is None:
        continue
    a = j.get("signal_artifact")
    if isinstance(a, dict) and a.get("n_dates") == len(ev_idx):
        lib = pd.DataFrame(arr, index=ev_idx, columns=SYMBOLS)
    elif isinstance(a, str) and a.endswith(".npy") and arr.shape[0] >= len(idx):
        rows = (ev_idx - idx[0]).days.to_numpy()
        lib = pd.DataFrame(arr[rows], index=ev_idx, columns=SYMBOLS)
    else:
        continue
    lib_arrs.append(lib)

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

rows = []
for name, fac in cands.items():
    if fac is None:
        continue
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

with open("scripts/miner3_screen_cycle5b_results.json", "w") as fh:
    json.dump({r[0]: dict(ic=r[1], icir=r[2], hit=r[3], n=r[4], cov=r[5], turn=r[6], lib_max=r[7])
               for r in rows}, fh, indent=1)
print("saved scripts/miner3_screen_cycle5b_results.json")
