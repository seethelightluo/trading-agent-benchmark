"""miner_3 broad factor screen - 2026-07-16.
Cross-asset 15-instrument universe, data window 2020-01-01..2026-07-15 (warm-up).
Admission gate (daily paper): |IC1| >= 0.0070 and |ICIR1| >= 0.0840, rank IC on 1d fwd returns.
Families: trend/momentum, short-term reversal, volatility, volume, OHLC-range, macro-beta conditional.
"""
import sys, os, time
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
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in SYMBOLS})
HP = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in SYMBOLS})
LP = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in SYMBOLS})
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
VO = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
LOG = np.log(CP / CP.shift(1))
ATR = ((HP - LP) + (HP - CP.shift(1)).abs() + (LP - CP.shift(1)).abs()).rolling(14).mean()
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20)}
N_CELLS = len(idx) * len(SYMBOLS)
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ics = {h: F.fast_ic(panel, fwd[h]) for h in (1, 2, 3, 5, 10, 20)}
    ic1 = ics[1]
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    if verbose:
        dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
        print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | {dec} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic": ics, "passed": passed}


cands = {}
# ---------- trend / momentum ----------
for nd in (10, 20, 60, 120, 252):
    cands[f"mom_{nd}d"] = CP / CP.shift(nd) - 1.0
cands["mom_12_1"] = (CP / CP.shift(252) - 1.0) - (CP / CP.shift(21) - 1.0)
vol20 = RET.rolling(20).std() * np.sqrt(252)
vol60 = RET.rolling(60).std() * np.sqrt(252)
cands["tsmom_20_vol20"] = (CP / CP.shift(20) - 1.0) / vol20
cands["tsmom_60_vol20"] = (CP / CP.shift(60) - 1.0) / vol20
cands["tsmom_120_vol60"] = (CP / CP.shift(120) - 1.0) / vol60
cands["tsmom_12_1_vol"] = cands["mom_12_1"] / vol60
ma20 = CP.rolling(20).mean(); ma60 = CP.rolling(60).mean(); ma120 = CP.rolling(120).mean()
cands["ma_dist_20_60"] = (ma20 - ma60) / (CP.rolling(20).std() + 1e-12)
cands["ma_dist_60_120"] = (ma60 - ma120) / (CP.rolling(60).std() + 1e-12)
cands["close_vs_ma60"] = CP / ma60 - 1.0
# RSI (simple)
def rsi(close, nd=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(nd).mean()
    dn = (-d.clip(upper=0)).rolling(nd).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))
cands["rsi_14"] = rsi(CP, 14)
cands["rsi_14_z"] = (rsi(CP, 14) - 50) / (rsi(CP, 14).rolling(120).std() + 1e-12)
# CCI
tp = (HP + LP + CP) / 3
cands["cci_20"] = (tp - tp.rolling(20).mean()) / (1.5 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean() + 1e-12)
# MACD histogram
ema12 = CP.ewm(span=12, adjust=False).mean(); ema26 = CP.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
cands["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
cands["macd_hist_norm"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (CP.rolling(20).std() + 1e-12)
# ---------- short-term reversal ----------
for nd in (1, 2, 3, 5, 10):
    cands[f"rev_{nd}d"] = -(CP / CP.shift(nd) - 1.0)
cands["rev_5d_vol"] = -(CP / CP.shift(5) - 1.0) / vol20
cands["rev_3d_vol"] = -(CP / CP.shift(3) - 1.0) / vol20
# ---------- volatility ----------
cands["inv_vol_20d"] = -vol20
cands["inv_vol_60d"] = -vol60
cands["vol_chg_5_60"] = RET.rolling(5).std() / RET.rolling(60).std()
cands["vol_chg_10_60"] = RET.rolling(10).std() / RET.rolling(60).std()
cands["vol_z_20_120"] = (vol20 - RET.rolling(120).std() * np.sqrt(252)) / (vol20.rolling(120).std() + 1e-12)
# ---------- volume ----------
vret = VO.pct_change()
cands["vol_vol_5_60"] = VO.rolling(5).mean() / VO.rolling(60).mean()
cands["vol_vol_20_60"] = VO.rolling(20).mean() / VO.rolling(60).mean()
cands["vol_z_60"] = (VO.rolling(5).mean() - VO.rolling(60).mean()) / (VO.rolling(60).std() + 1e-12)
cands["pv_corr_20"] = RET.rolling(20).corr(vret)
obv = (np.sign(RET) * VO).cumsum()
cands["obv_slope_20"] = (obv / obv.rolling(20).mean() - 1.0)
# ---------- OHLC / range ----------
clv = (CP - LP) / (HP - LP + 1e-12)
cands["clv_1d"] = clv
cands["clv_5d"] = (CP - LP.rolling(5).min()) / (HP.rolling(5).max() - LP.rolling(5).min() + 1e-12)
cands["clv_20d"] = (CP - LP.rolling(20).min()) / (HP.rolling(20).max() - LP.rolling(20).min() + 1e-12)
cands["body_5d"] = (CP - OP).rolling(5).mean() / (ATR + 1e-12)
cands["gap_5d"] = (OP - CP.shift(1)).rolling(5).mean() / (ATR + 1e-12)
cands["upper_shadow"] = (HP - np.maximum(OP, CP)).rolling(5).mean() / (ATR + 1e-12)
cands["lower_shadow"] = (np.minimum(OP, CP) - LP).rolling(5).mean() / (ATR + 1e-12)
cands["range_pos_20"] = (CP - LP.rolling(20).min()) / (HP.rolling(20).max() - LP.rolling(20).min() + 1e-12) - 0.5
# ---------- macro-beta conditional ----------
macro_dir = "../persistent/index_data"
def load_macro(name):
    d = pd.read_csv(os.path.join(macro_dir, f"{name}.csv"))
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= pd.Timestamp("2026-07-15")].set_index("date")["close"].astype(float)
    return d.reindex(idx).ffill()
DXY = load_macro("DXY"); VIX = load_macro("VIX"); USDCNY = load_macro("USDCNY")
dxy_r = DXY.pct_change(); vix_c = VIX.diff(); usdcny_r = USDCNY.pct_change()

def roll_beta(y_panel, x, win):
    out = {}
    for s in y_panel.columns:
        y = y_panel[s]
        df = pd.concat([y, x], axis=1).dropna()
        cov = df.iloc[:, 0].rolling(win).cov(df.iloc[:, 1])
        var = df.iloc[:, 1].rolling(win).var()
        out[s] = (cov / var).reindex(idx)
    return pd.DataFrame(out)

for win in (30, 60, 120):
    b_dxy = roll_beta(RET, dxy_r, win)
    b_vix = roll_beta(RET, vix_c, win)
    b_cny = roll_beta(RET, usdcny_r, win)
    cands[f"beta_dxy_{win}"] = b_dxy
    cands[f"negbeta_dxy_{win}"] = -b_dxy
    cands[f"beta_vix_{win}"] = b_vix
    cands[f"negbeta_vix_{win}"] = -b_vix
    cands[f"beta_cny_{win}"] = b_cny
    # USD trend tilt: when USD trends down, favor assets with negative DXY beta
    dxy_trend = np.sign(dxy_r.rolling(win).mean())
    cands[f"usd_tilt_{win}"] = -b_dxy * dxy_trend
    # VIX level tilt: risk-on in low VIX regime
    vix_hi = (VIX > VIX.rolling(252).median()).astype(float)
    cands[f"risk_off_tilt_{win}"] = -b_vix * vix_hi

res = [run(n, p) for n, p in cands.items()]
print(f"\nscreen done {time.time()-t0:.1f}s | {len(res)} candidates | {sum(r['passed'] for r in res)} PASSED gate")
print("\n=== sorted by |IC1| ===")
for r in sorted(res, key=lambda r: -abs(r["ic"][1]["ic"])):
    mark = "PASS" if r["passed"] else "   "
    print(f"[{mark}] {r['name']:22s} IC1={r['ic'][1]['ic']:+.4f} ICIR1={r['ic'][1]['icir']:+.3f} hit={r['ic'][1]['hit']:.3f}")
