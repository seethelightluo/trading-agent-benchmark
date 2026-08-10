"""miner_1 novel factor screen v2 - 2026-07-16.

Cross-asset 15-instrument universe. Admission gate (daily rank IC, 1d fwd):
    |IC1| >= 0.0070 and |ICIR1| >= 0.0840
Validation window: 2021-01-01 .. 2026-07-15 (2020 used as warm-up).

Focus: NOVEL families NOT already in the (quarantined) library which contains
rev_1d..5d, nclv_1d..5d, nbody_1d, id_rev_1d, rev_1d_vs.
New directions: trend/momentum, volatility dynamics, oscillator families,
range/ATR, downside/upside capture, autocorrelation, drawdown, gap/overnight.

Also computes max_abs_library_correlation vs the 13 quarantined library panels
(real signal panels reconstructed from OHLC) so passers are provably novel.
"""
import sys, os, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= pd.Timestamp("2026-07-15"))]

OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

VAL = idx[idx >= pd.Timestamp("2021-01-01")]
fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(VAL) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

# ---- library panels (quarantined factors, real signal reconstruction) ----
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
    # library correlation (max abs)
    corrs = [panel_corr(panel, lv) for lv in lib.values()]
    corrs = [c for c in corrs if c is not None]
    maxc = max(abs(c) for c in corrs) if corrs else np.nan
    if verbose:
        dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
        print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | maxLibCorr={maxc:.2f} | {dec} | "
              f"{'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic": ics, "passed": passed, "max_lib_corr": maxc}


cands = {}
vol20 = RET.rolling(20).std() * np.sqrt(252)
vol60 = RET.rolling(60).std() * np.sqrt(252)
vol10 = RET.rolling(10).std() * np.sqrt(252)

# ---------- 1. trend / momentum (long-horizon) ----------
for nd in (20, 60, 120, 252):
    cands[f"mom_{nd}d"] = CP / CP.shift(nd) - 1.0
cands["mom_12_1"] = (CP / CP.shift(252) - 1.0) - (CP / CP.shift(21) - 1.0)
cands["tsmom_60_vol"] = (CP / CP.shift(60) - 1.0) / vol60
cands["dist_52w_high"] = CP / CP.rolling(252).max() - 1.0
cands["dist_120d_high"] = CP / CP.rolling(120).max() - 1.0
cands["drawdown_60d"] = CP / CP.rolling(60).max() - 1.0
cands["close_vs_ma60"] = CP / CP.rolling(60).mean() - 1.0

# ---------- 2. volatility dynamics ----------
cands["inv_vol20"] = -vol20
cands["vol_chg_5_60"] = RET.rolling(5).std() / RET.rolling(60).std()
cands["vol_z_20_120"] = (vol20 - RET.rolling(120).std() * np.sqrt(252)) / (vol20.rolling(120).std() + 1e-12)
cands["vol_skew_20"] = RET.rolling(20).skew()
cands["vol_kurt_20"] = RET.rolling(20).kurt()
cands["autocorr1_20"] = RET.rolling(20).apply(lambda x: x.autocorr(1) if len(x) > 3 else np.nan, raw=False)

# ---------- 3. oscillators (RSI / CCI / stochastic) ----------
def rsi(close, nd=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(nd).mean()
    dn = (-d.clip(upper=0)).rolling(nd).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))
cands["rsi_14"] = rsi(CP, 14) - 50
cands["rsi_14_z"] = (rsi(CP, 14) - 50) / (rsi(CP, 14).rolling(120).std() + 1e-12)
tp = (HP + LP + CP) / 3
cands["cci_20"] = (tp - tp.rolling(20).mean()) / (1.5 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean() + 1e-12)
ll = LP.rolling(14).min(); hh = HP.rolling(14).max()
cands["stoch_k"] = (CP - ll) / (hh - ll).replace(0, np.nan) - 0.5
ema12 = CP.ewm(span=12, adjust=False).mean(); ema26 = CP.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
cands["macd_hist_norm"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (CP.rolling(20).std() + 1e-12)

# ---------- 4. range / ATR families ----------
atr14 = ((HP - LP) + (HP - CP.shift(1)).abs() + (LP - CP.shift(1)).abs()).rolling(14).mean()
cands["atr_z_60"] = (atr14 - atr14.rolling(60).mean()) / (atr14.rolling(60).std() + 1e-12)
cands["rng_expand_5"] = (HP - LP) / ((HP - LP).rolling(5).mean() + 1e-12)
cands["upper_wick_1d"] = (HP - CP) / rng1          # overreaction up (reversal long side)
cands["lower_wick_1d"] = (CP - LP) / rng1          # overreaction down (reversal short side)
cands["high_pos_1d"] = (CP - OP) / rng1            # signed body (already have nbody neg; keep pos sign to test)
cands["upper_shadow_20"] = ((HP - CP).rolling(20).mean()) / (atr14 + 1e-12)

# ---------- 5. gap / overnight ----------
cands["gap_1d"] = OP / CP.shift(1) - 1.0
cands["gap_rev_1d"] = -(OP / CP.shift(1) - 1.0)
cands["overnight_1d"] = OP / CP.shift(1) - 1.0
cands["intraday_1d"] = CP / OP - 1.0

# ---------- 6. downside / upside capture ----------
down = RET.clip(upper=0); up = RET.clip(lower=0)
cands["downside_vol_20"] = down.rolling(20).std() * np.sqrt(252)
cands["downside_ratio_20"] = down.rolling(20).std() / (RET.rolling(20).std() + 1e-12)
cands["updown_20"] = up.rolling(20).mean() / (down.abs().rolling(20).mean() + 1e-12)

# ---------- 7. conditional / interaction ----------
cands["rev1_x_vol20"] = -LOG / (RET.rolling(5).std() + 1e-12)   # faster vol scaling
cands["rev5_x_atr"] = -(CP / CP.shift(5) - 1.0) / (atr14 / CP + 1e-12)
cands["rev1_ma_gate"] = -LOG * (CP > CP.rolling(20).mean()).astype(float)  # reversal only in uptrend

res = {}
for nm, p in cands.items():
    res[nm] = run(nm, p)

passers = {k: v for k, v in res.items() if v["passed"]}
print(f"\nTotal candidates: {len(cands)}, PASS: {len(passers)}")
for k, v in passers.items():
    print(f"  PASS {k}: IC1={v['ic'][1]['ic']:+.4f} ICIR1={v['ic'][1]['icir']:+.3f} maxLibCorr={v['max_lib_corr']:.2f}")

# dump passers' panels for exact artifact persistence
if passers:
    with open("scripts/_miner1_passers_v2.pkl", "wb") as fh:
        pickle.dump({k: cands[k].reindex(idx) for k in passers}, fh)
    print("saved scripts/_miner1_passers_v2.pkl")
print(f"elapsed {time.time()-t0:.1f}s")
