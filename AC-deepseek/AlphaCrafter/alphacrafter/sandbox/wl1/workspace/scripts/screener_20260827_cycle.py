"""Screener cycle 2026-08-27: regime assessment + IC/ICIR recompute on visible data.

Visible horizon: last common trading date <= 2026-08-26 (previous completed day).
Methodology mirrors miner3_fast.fast_ic (daily cross-sectional rank IC, ICIR=mean/std).
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, MACRO, DATA_DIR, IDX_DIR

CUT = pd.Timestamp("2026-08-26")
START = pd.Timestamp("2021-01-01")  # validation window start (as miner pipeline)
MIN_NAMES = 8

# ---------- load ----------
def load_all():
    frames = {}
    for s in SYMBOLS:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv"))
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] <= CUT].sort_values("date").set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        frames[s] = d
    # common calendar
    idx = None
    for s, df in frames.items():
        idx = df.index if idx is None else idx.intersection(df.index)
    idx = idx[idx >= pd.Timestamp("2020-01-01")]
    CP = pd.DataFrame({s: frames[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
    OP = pd.DataFrame({s: frames[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
    HP = pd.DataFrame({s: frames[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
    LP = pd.DataFrame({s: frames[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
    # VIX (observation-only macro)
    vix = pd.read_csv(os.path.join(IDX_DIR, "VIX.csv"))
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["date"] <= CUT].sort_values("date").set_index("date")
    vixc = vix["close"].astype(float).reindex(idx)
    return idx, CP, OP, HP, LP, vixc

idx, CP, OP, HP, LP, VIXC = load_all()
RET = CP.pct_change()
print(f"common dates: {len(idx)}  {idx.min().date()}..{idx.max().date()}")
print(f"last date: {idx.max().date()}")

# ---------- factor panels ----------
def fwd_returns(fwd_days=1):
    return CP.shift(-fwd_days) / CP - 1.0

def fast_ic(factor_df, fwd_df, min_names=MIN_NAMES):
    F = factor_df.reindex(idx).astype(float)
    R = fwd_df.reindex(idx).astype(float)
    F = F.rank(axis=1); R = R.rank(axis=1)
    Fv, Rv = F.values, R.values
    mask = np.isfinite(Fv) & np.isfinite(Rv)
    n = mask.sum(axis=1)
    ok = n >= min_names
    if not ok.any():
        return dict(n_dates=0, n_obs=0, ic=np.nan, icir=np.nan, hit=np.nan)
    Fm = np.where(mask, Fv, 0.0); Rm = np.where(mask, Rv, 0.0)
    sx = Fm.sum(1); sy = Rm.sum(1)
    sxx = (Fm*Fm).sum(1); syy = (Rm*Rm).sum(1); sxy = (Fm*Rm).sum(1)
    with np.errstate(all="ignore"):
        num = n*sxy - sx*sy
        den = np.sqrt((n*sxx - sx*sx)*(n*syy - sy*sy))
        ic = num/den
    ic = ic[ok]; ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return dict(n_dates=0, n_obs=0, ic=np.nan, icir=np.nan, hit=np.nan)
    return dict(n_dates=int(len(ic)), n_obs=int(n[ok].sum()),
                ic=float(ic.mean()),
                icir=float(ic.mean()/ic.std()) if ic.std() > 0 else np.nan,
                hit=float((ic > 0).mean()))

def panel(name, fn):
    cols = {}
    for s in SYMBOLS:
        df = pd.DataFrame({"open": OP[s], "high": HP[s], "low": LP[s], "close": CP[s]})
        try:
            v = fn(df)
            if v is not None:
                cols[s] = v
        except Exception as e:
            print(f"  [warn] {name} {s}: {e}")
    return pd.DataFrame(cols).reindex(idx)

rng1 = (HP - LP).replace(0, np.nan)
vol20 = RET.rolling(20).std()
factors = {}
factors["nclv_1d"] = panel("nclv_1d", lambda d: -(d["close"] - d["low"]) / (d["high"] - d["low"]))
factors["nclv_2d"] = panel("nclv_2d", lambda d: -(d["close"] - d["low"].rolling(2).min()) / (d["high"].rolling(2).max() - d["low"].rolling(2).min()))
factors["nclv_3d"] = panel("nclv_3d", lambda d: -(d["close"] - d["low"].rolling(3).min()) / (d["high"].rolling(3).max() - d["low"].rolling(3).min()))
factors["nclv_5d"] = panel("nclv_5d", lambda d: -(d["close"] - d["low"].rolling(5).min()) / (d["high"].rolling(5).max() - d["low"].rolling(5).min()))
factors["rev_1d"] = panel("rev_1d", lambda d: -np.log(d["close"] / d["close"].shift(1)))
factors["rev_2d"] = panel("rev_2d", lambda d: -np.log(d["close"] / d["close"].shift(2)))
factors["rev_3d"] = panel("rev_3d", lambda d: -np.log(d["close"] / d["close"].shift(3)))
factors["rev_5d"] = panel("rev_5d", lambda d: -np.log(d["close"] / d["close"].shift(5)))
factors["rev_1d_vs"] = panel("rev_1d_vs", lambda d: -np.log(d["close"] / d["close"].shift(1)) / (vol20.reindex(d.index) + 1e-12))
factors["id_rev_1d"] = panel("id_rev_1d", lambda d: -(d["close"] / d["open"] - 1.0))
factors["nbody_1d"] = panel("nbody_1d", lambda d: -(d["close"] - d["open"]) / (d["high"] - d["low"]))
factors["mom_10d_skip5"] = panel("mom_10d_skip5", lambda d: np.log(d["close"].shift(5) / d["close"].shift(15)))
factors["mom_120d_skip5"] = panel("mom_120d_skip5", lambda d: np.log(d["close"].shift(5) / d["close"].shift(125)))
# vix_beta_cond_60x20: -beta(asset_ret, VIX_ret, 60) * (VIX/VIX.shift(20)-1)
vixr = VIXC.pct_change()
def vix_beta(d):
    ar = np.log(d["close"] / d["close"].shift(1))
    vr = vixr.reindex(d.index)
    cov = ar.rolling(60).cov(vr)
    var = vr.rolling(60).var().replace(0, np.nan)
    beta = cov / var
    vx = (VIXC / VIXC.shift(20) - 1.0).reindex(d.index)
    return -beta * vx
factors["vix_beta_cond_60x20"] = panel("vix_beta", vix_beta)
factors["vol_of_vol20x60"] = panel("vol_of_vol", lambda d: RET.reindex(d.index).rolling(20).std().rolling(60).std())

fwd1 = fwd_returns(1)
fwd5 = fwd_returns(5)

print("\n=== IC/ICIR full window 2021-01-01..2026-08-26 ===")
full_rows = []
for name, p in factors.items():
    r = fast_ic(p, fwd1)
    r5 = fast_ic(p, fwd5)
    full_rows.append((name, r, r5))
    print(f"{name:20s} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f} hit1={r['hit']:.3f} n={r['n_dates']} | IC5={r5['ic']:+.4f}")

print("\n=== IC/ICIR recent window (last 120 common dates) ===")
rec_idx = idx[-120:]
rec_rows = []
for name, p in factors.items():
    r = fast_ic(p.reindex(rec_idx), fwd1.reindex(rec_idx))
    r5 = fast_ic(p.reindex(rec_idx), fwd5.reindex(rec_idx))
    rec_rows.append((name, r, r5))
    print(f"{name:20s} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f} hit1={r['hit']:.3f} n={r['n_dates']} | IC5={r5['ic']:+.4f}")

print("\n=== IC/ICIR post-online window (2026-07-16..2026-08-26) ===")
po_idx = idx[idx >= pd.Timestamp("2026-07-16")]
po_rows = []
for name, p in factors.items():
    r = fast_ic(p.reindex(po_idx), fwd1.reindex(po_idx))
    po_rows.append((name, r))
    print(f"{name:20s} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f} hit1={r['hit']:.3f} n={r['n_dates']}")

# ---------- regime stats ----------
print("\n=== REGIME (data <= 2026-08-26) ===")
last = idx.max()
for s in SYMBOLS:
    c = CP[s].dropna()
    if len(c) < 70:
        continue
    c21 = c.iloc[-1] / c.iloc[-22] - 1
    c5 = c.iloc[-1] / c.iloc[-6] - 1
    lo60, hi60 = c.iloc[-61:-1].min(), c.iloc[-61:-1].max()
    pos60 = (c.iloc[-2] - lo60) / (hi60 - lo60) if hi60 > lo60 else np.nan
    annvol = c.pct_change().tail(21).std() * np.sqrt(252)
    print(f"{s:10s} 21d={c21:+.2%} 5d={c5:+.2%} pos60={pos60:.2f} annvol21={annvol:.1%}")

# pairwise corr of daily returns over last 21 days
rc = RET.tail(21)
cm = rc.corr().abs()
np.fill_diagonal(cm.values, np.nan)
print(f"\nmean |pairwise corr| (21d): {np.nanmean(cm.values):.3f}")
print(f"VIX last: {VIXC.dropna().iloc[-1]:.2f}  21d change: {(VIXC.dropna().iloc[-1]/VIXC.dropna().iloc[-22]-1):+.2%}")

# 20d mean return across assets (regime proxy)
m20 = RET.tail(20).mean().mean()
print(f"cross-asset mean 20d daily ret: {m20:+.4%}  (annualized ~{m20*252:+.1%})")
