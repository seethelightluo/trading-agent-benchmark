"""miner_3: exploration screen of candidate factor ideas (vectorized IC).
Current date 2028-02-04; panel visible through 2028-02-03.
Gate: |IC|>=0.0070 and |ICIR|>=0.0840 (same-horizon admission).
"""
import pandas as pd
import numpy as np
import pickle, json, time

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, O, H, L, V, M = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"], panel["macro"]
ret = C.pct_change()

f = {}
# --- short/medium anti-momentum (regime flipped: momentum now contrarian) ---
f["anti_mom_20d_skip5"] = -(C.shift(5) / C.shift(25) - 1.0)
f["anti_mom_60d_skip5"] = -(C.shift(5) / C.shift(65) - 1.0)
f["anti_mom_120d_skip5"] = -(C.shift(5) / C.shift(125) - 1.0)
# --- risk-adjusted momentum ---
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
f["anti_mom_adjvol_20d"] = -(C.shift(5) / C.shift(25) - 1.0) / vol20.shift(5)
f["mom_adjvol_60d"] = (C.shift(5) / C.shift(65) - 1.0) / vol60.shift(5)
# --- drawdown / distance from highs ---
f["dd_60d"] = C / C.rolling(60).max() - 1.0
f["dd_120d"] = C / C.rolling(120).max() - 1.0
f["dd_250d"] = C / C.rolling(250).max() - 1.0
# --- skewness / shape ---
f["skew_20d"] = ret.rolling(20).skew()
f["skew_60d"] = ret.rolling(60).skew()
# --- volatility expansion ---
f["vol_ratio_5_60"] = ret.rolling(5).std() / ret.rolling(60).std()
f["vol_ratio_20_60"] = vol20 / vol60
f["range_20d"] = (H.rolling(20).max() - L.rolling(20).min()) / C
# --- volume / participation ---
f["volm_ratio_5_20"] = V.rolling(5).mean() / V.rolling(20).mean() - 1.0
f["volm_ratio_20_60"] = V.rolling(20).mean() / V.rolling(60).mean() - 1.0
# --- candle-shape factors ---
rng = (H - L).replace(0, np.nan)
f["upper_wick_1d"] = (H - np.maximum(O, C)) / rng
f["lower_wick_1d"] = (np.minimum(O, C) - L) / rng
f["gap_1d"] = O / C.shift(1) - 1.0
f["intraday_1d"] = C / O - 1.0
# --- conditional reversal (gated on vol / VIX regime) ---
rev1 = -(C.shift(1) / C - 1.0)
vol20z = (vol20 - vol20.rolling(120).mean()) / vol20.rolling(120).std()
f["cond_rev_highvol"] = rev1 * (vol20z > 0).astype(float)
vix = M["VIX"]
vix_hi = (vix > vix.rolling(20).mean()).astype(float)
f["cond_rev_vixhi"] = rev1 * vix_hi
# --- trend strength ---
f["ma_slope_20_60"] = C.rolling(20).mean() / C.rolling(60).mean() - 1.0
f["ma_slope_50_200"] = C.rolling(50).mean() / C.rolling(200).mean() - 1.0
# --- up/down day breadth ---
up = (ret > 0).astype(float)
f["updown_ratio_20d"] = (up.rolling(20).sum() - 20) / 20.0
f["updown_ratio_60d"] = (up.rolling(60).sum() - 30) / 60.0

GATE_IC, GATE_ICIR = 0.0070, 0.0840
HORIZONS = (1, 2, 3, 5, 10)

fwd = {h: (C.shift(-h) / C - 1.0) for h in HORIZONS}

def ic_series_fast(fac, fh):
    fr = fac.rank(axis=1)
    rr = fh.rank(axis=1)
    frc = fr.sub(fr.mean(axis=1), axis=0)
    rrc = rr.sub(rr.mean(axis=1), axis=0)
    num = (frc * rrc).sum(axis=1)
    den = np.sqrt((frc ** 2).sum(axis=1) * (rrc ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    n_valid = (fac.notna() & fh.notna()).sum(axis=1)
    return ic.where(n_valid >= 8)

t0 = time.time()
print(f"{'factor':24s} {'h':>3s} {'IC':>9s} {'ICIR':>9s} {'hit':>6s} {'n':>5s} {'IC12':>9s} {'ICIR12':>9s}  gate")
out = {}
for name, fac in f.items():
    best = None
    for h in HORIZONS:
        s = ic_series_fast(fac, fwd[h]).dropna()
        if len(s) == 0:
            continue
        ic = float(s.mean()); sd = float(s.std(ddof=1))
        icir = ic / sd if sd > 0 else 0.0
        if best is None or abs(icir) > abs(best[1]):
            hit = float((s > 0).mean()) if ic > 0 else float((s < 0).mean())
            best = (h, icir, ic, len(s), hit, s)
    if best is None:
        print(f"{name:24s}  no data"); continue
    h, icir, ic, n, hit, s = best
    cut = s.index.max() - pd.Timedelta(days=365)
    s12 = s[s.index >= cut]
    ic12 = float(s12.mean()); icir12 = float(ic12 / s12.std(ddof=1)) if s12.std(ddof=1) > 0 else 0.0
    ok = "PASS" if abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR else "fail"
    print(f"{name:24s} {h:3d} {ic:+9.5f} {icir:+9.5f} {hit:6.3f} {n:5d} {ic12:+9.5f} {icir12:+9.5f}  {ok}")
    out[name] = dict(h=h, ic=ic, icir=icir, hit=hit, n=n, ic12=ic12, icir12=icir12, ok=ok)

json.dump(out, open("scripts/miner3_20280204_screen_results.json", "w"), indent=1, default=float)
print(f"\nsaved scripts/miner3_20280204_screen_results.json  elapsed={time.time()-t0:.1f}s")
