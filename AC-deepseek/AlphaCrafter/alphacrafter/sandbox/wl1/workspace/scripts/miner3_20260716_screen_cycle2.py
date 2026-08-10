"""miner_3 cycle: screen NOVEL factor families on the 15-instrument cross-asset
panel, including macro-regime-conditional reversal, rate/yield beta, downside
risk, efficiency, and lead-lag spillover factors. Known reversal family is
recomputed as a control.

Admission gates (15-name cross-asset universe):
    |IC1| >= 0.0070  and  |ICIR1| >= 0.0840
Validation window: 2021-01-01 .. 2026-07-15 (1y warm-up for rolling windows).
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, MACRO, load_close
import miner3_fast as F

T0 = time.time()
closes = load_close()
# macro observation-only signals
macro = {}
for m in MACRO:
    d = pd.read_csv(os.path.join("../persistent/index_data", f"{m}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= pd.Timestamp("2026-07-15")].sort_values("date").set_index("date")
    macro[m] = pd.to_numeric(d["close"], errors="coerce")

idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
vol5 = RET.rolling(5).std() * np.sqrt(252)
vol20 = RET.rolling(20).std() * np.sqrt(252)
vol60 = RET.rolling(60).std() * np.sqrt(252)
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-T0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20)}
N_CELLS = len(idx) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840

# macro series aligned to idx
DXY = macro["DXY"].reindex(idx).ffill()
USDCNY = macro["USDCNY"].reindex(idx).ffill()
USDJPY = macro["USDJPY"].reindex(idx).ffill()
EURUSD = macro["EURUSD"].reindex(idx).ffill()
VIX = macro["VIX"].reindex(idx).ffill()
DXY_R = DXY.pct_change()
VIX_C = VIX.diff()
USDCNY_R = USDCNY.pct_change()
EURUSD_R = EURUSD.pct_change()


def roll_beta(y_panel, x_series, win):
    """Rolling regression slope of each y column on x (aligned index)."""
    out = {}
    xs = x_series.rename("x")
    for s in y_panel.columns:
        df = pd.concat([y_panel[s].rename("y"), xs], axis=1).dropna()
        if len(df) < win + 5:
            out[s] = pd.Series(np.nan, index=y_panel.index)
            continue
        cov = df["y"].rolling(win).cov(df["x"])
        var = df["x"].rolling(win).var()
        out[s] = (cov / (var + 1e-12)).reindex(y_panel.index)
    return pd.DataFrame(out)


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ics = {h: F.fast_ic(panel, fwd[h]) for h in (1, 2, 3, 5, 10, 20)}
    ic1 = ics[1]
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    if verbose:
        dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
        print(f"{name:26s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | {dec} | {'PASS' if passed else 'fail'}")
    return {"name": name, "panel": panel, "cov": cov, "to": to, "ics": ics, "passed": passed}


cands = {}

# ---------- control: known reversal family ----------
cands["rev_1d"] = -LOG.diff(1)
cands["nclv_1d"] = -((CP - LP) / (HP - LP).replace(0, np.nan))
cands["rev_1d_vs"] = -LOG.diff(1) / (vol20 + 1e-12)

# ---------- Family A: macro-regime-conditional reversal ----------
# Reversal should be stronger in high-vol / risk-off regimes (VIX high, DXY up)
vix_z = (VIX - VIX.rolling(60).mean()) / (VIX.rolling(60).std() + 1e-9)
dxy_trend = DXY / DXY.rolling(60).mean() - 1.0
cands["rev1d_vix_cond"] = -LOG.diff(1) * (1.0 + vix_z.clip(-2, 2))
cands["rev1d_dxy_cond"] = -LOG.diff(1) * (1.0 + 2.0 * dxy_trend.clip(-1, 1))
cands["rev1d_riskoff"] = -LOG.diff(1) * (1.0 + (vix_z - dxy_trend).clip(-2, 2))
cands["rev1d_us10y_cond"] = -LOG.diff(1) * (1.0 + 2.0 * (CP["US10Y"] / CP["US10Y"].shift(20) - 1.0).clip(-1, 1))

# ---------- Family B: rate / yield beta ----------
b_us10y = roll_beta(RET, CP["US10Y"].diff(), 60)
b_cn10y = roll_beta(RET, CP["CN10Y"].diff(), 60)
cands["beta_us10y_60"] = b_us10y
cands["negbeta_us10y_60"] = -b_us10y
cands["beta_cn10y_60"] = b_cn10y
cands["negbeta_cn10y_60"] = -b_cn10y
# yield spread interaction: assets with positive US10Y beta when yields rise (sign flip)
yld_up = (CP["US10Y"] / CP["US10Y"].shift(20) - 1.0)
cands["negbeta_us10y_x_yld"] = -b_us10y * (1.0 + 2.0 * yld_up.clip(-1, 1))

# ---------- Family C: downside risk ----------
down = RET.where(RET < 0, 0.0)
down_vol20 = np.sqrt((down ** 2).rolling(20).mean()) * np.sqrt(252)
cands["neg_downvol_20"] = -down_vol20
cands["downvol_ratio"] = down_vol20 / (vol20 + 1e-12)
cands["neg_negret_freq_20"] = -(RET < 0).rolling(20).mean()
# downside beta to SPX
b_spx = roll_beta(RET, RET["SPX"], 60)
spx_down = RET["SPX"].where(RET["SPX"] < 0, 0.0)
b_spx_down = roll_beta(RET, spx_down, 60)
cands["neg_downbeta_spx_60"] = -b_spx_down

# ---------- Family D: efficiency / trend quality ----------
efficiency_20 = (CP / CP.shift(20) - 1.0).abs() / ((RET.abs()).rolling(20).sum() + 1e-12)
cands["efficiency_20"] = efficiency_20
# close location in 10d range * 10d momentum direction (trend confirmation)
cands["clv5_x_mom10"] = ((CP - LP.rolling(5).min()) / (HP.rolling(5).max() - LP.rolling(5).min() + 1e-12) - 0.5) * \
                        (CP / CP.shift(10) - 1.0)

# ---------- Family E: lead-lag spillover timing ----------
btc_r = RET["BTC"]
spx_r = RET["SPX"]
xau_r = RET["XAU"]
cands["btc_lead_beta60"] = roll_beta(RET, btc_r, 60).shift(1).mul(
    btc_r.shift(1).to_numpy().reshape(-1, 1), axis=0)
cands["spx_lead_beta60"] = roll_beta(RET, spx_r, 60).shift(1).mul(
    spx_r.shift(1).to_numpy().reshape(-1, 1), axis=0)
# gold as risk-off hedge: beta to XAU * XAU return
cands["xau_lead_beta60"] = roll_beta(RET, xau_r, 60).shift(1).mul(
    xau_r.shift(1).to_numpy().reshape(-1, 1), axis=0)
# VIX lead: beta to VIX change * VIX change (negative for risky assets)
cands["vix_lead_beta60"] = -roll_beta(RET, VIX_C, 60).shift(1).mul(
    VIX_C.shift(1).to_numpy().reshape(-1, 1), axis=0)

# ---------- Family F: volume-price efficiency ----------
# return per unit of dollar volume (12-month), negated -> high = liquid
dv = (CP * VO).rolling(252).mean()
cands["neg_inv_dv12"] = -(1.0 / (dv + 1e-9))
# volume trend x reversal (contrarian after volume surge)
vma20 = VO.rolling(20).mean()
cands["rev1d_volsurge"] = -LOG.diff(1) * (VO / vma20)
# OBV momentum
obv = (np.sign(LOG.diff(1)) * VO).cumsum()
cands["obv_mom_10"] = obv / obv.rolling(20).mean() - 1.0

res = [run(n, p) for n, p in cands.items()]
print(f"\nscreen done {time.time()-T0:.1f}s | {len(res)} candidates | {sum(r['passed'] for r in res)} PASSED gate")
print("\n=== sorted by |IC1| ===")
for r in sorted(res, key=lambda r: -abs(r["ics"][1]["ic"])):
    mark = "PASS" if r["passed"] else "   "
    print(f"[{mark}] {r['name']:26s} IC1={r['ics'][1]['ic']:+.4f} ICIR1={r['ics'][1]['icir']:+.3f} "
          f"hit={r['ics'][1]['hit']:.3f} to={r['to']:.3f} cov={r['cov']:.3f}")

# save results for the persist stage
import json
with open("scripts/_miner3_cycle2_screen.json", "w") as fh:
    json.dump({r["name"]: {"passed": r["passed"], "cov": r["cov"], "to": r["to"],
                            "ic1": r["ics"][1]["ic"], "icir1": r["ics"][1]["icir"],
                            "hit1": r["ics"][1]["hit"], "n_dates": r["ics"][1]["n_dates"],
                            "decay": {h: r["ics"][h]["ic"] for h in (2, 3, 5, 10, 20)}}
               for r in res}, fh, indent=1)
