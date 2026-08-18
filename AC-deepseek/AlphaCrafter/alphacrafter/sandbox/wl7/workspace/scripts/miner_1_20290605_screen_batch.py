"""miner_1: screen novel candidate factors as of 2029-06-04.

Universe: 15 cross-asset instruments. Admission gates (benchmark-wide):
abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 at h=10; max_abs_library_correlation < 0.5.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, max_lib_corr)

END = "2029-06-04"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
lib_panels = library_panel(close, macro)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")

cands = {}

# 1. Parkinson vol ratio 20x60 (range-based vol regime)
def parkinson_vol(close, win):
    hl = np.log(close["high"] if False else None) if False else None
    # use high/low if we had them; here we approximate with close-only realized vol
    return None

# Use OHLCV from raw CSVs for high/low based factors
DATA_DIR = "../persistent/stock_data"
ASSETS = list(close.columns)
def load_ohlc(end):
    cal = close.index
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").reindex(cal).ffill()
        out[a] = df[["open", "high", "low", "close", "volume"]]
    return out

ohlc = load_ohlc(END)
H = pd.DataFrame({a: ohlc[a]["high"] for a in ASSETS})
L = pd.DataFrame({a: ohlc[a]["low"] for a in ASSETS})
V = pd.DataFrame({a: ohlc[a]["volume"] for a in ASSETS})

# 2. Parkinson vol ratio (20d/60d)
pk = (np.log(H / L) ** 2)
pk20 = pk.rolling(20).mean().apply(np.sqrt)
pk60 = pk.rolling(60).mean().apply(np.sqrt)
cands["parkinson_vol_ratio_20x60"] = -(pk20 / pk60)  # low short-term vol vs long-term -> prefer? try sign later

# 3. Drawdown depth 60d (distance below running max)
cands["drawdown_60d"] = close / close.rolling(60, min_periods=30).max() - 1.0

# 4. Stochastic position 20d (close within 20d high-low range)
hi20 = H.rolling(20).max()
lo20 = L.rolling(20).min()
cands["stoch_pos_20d"] = (close - lo20) / (hi20 - lo20).replace(0, np.nan)

# 5. Return autocorrelation 10x60 (mean-reversion vs trend)
def ret_autocorr(close, lag=5, win=60, minp=30):
    r = ret
    rl = r.shift(lag)
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for a in r.columns:
        out[a] = r[a].rolling(win, min_periods=minp).corr(rl[a])
    return out
cands["ret_autocorr_5x60"] = ret_autocorr(close, lag=5, win=60)

# 6. Volume trend 20d (z-score of volume vs 60d)
vz = (V - V.rolling(60).mean()) / V.rolling(60).std()
cands["volume_z_20d"] = vz.rolling(20).mean()

# 7. Semi-deviation ratio 20d (downside / upside vol)
def semi_ratio(close, win=20):
    r = ret
    neg = r.where(r < 0, 0.0)
    pos = r.where(r > 0, 0.0)
    ds = (neg ** 2).rolling(win).mean().apply(np.sqrt)
    us = (pos ** 2).rolling(win).mean().apply(np.sqrt)
    return -(ds / us)
cands["semi_ratio_20d"] = semi_ratio(close, 20)

# 8. VIX-change conditional beta 60x20 (asset beta to dVIX, conditioned on VIX momentum)
vix = macro["VIX"]
vix_r = vix.pct_change()
ret2 = ret
cov = ret2.rolling(60, min_periods=30).cov(vix_r)
var = vix_r.rolling(60, min_periods=30).var()
beta_vix = cov.divide(var, axis=0)
vix_mom = vix / vix.shift(20) - 1.0
cands["vix_beta_cond_60x20"] = -beta_vix.multiply(vix_mom, axis=0)  # defensive when VIX rising

# 9. USDJPY conditional beta 60x20
usdjpy = macro["USDJPY"]
uj_r = usdjpy.pct_change()
cov2 = ret2.rolling(60, min_periods=30).cov(uj_r)
var2 = uj_r.rolling(60, min_periods=30).var()
beta_uj = cov2.divide(var2, axis=0)
uj_mom = usdjpy / usdjpy.shift(20) - 1.0
cands["usdjpy_beta_cond_60x20"] = beta_uj.multiply(uj_mom, axis=0)  # carry-trade proxy: high beta to JPY weakness when JPY weakening

# 10. XAU lead 20d (XAU return as risk-sentiment predictor for all assets)
xau_r = close["XAU"].pct_change().rolling(20).mean()
cands["xau_lead_20d"] = pd.DataFrame({a: xau_r for a in ASSETS}, index=close.index)

# 11. BTC lead 20d
btc_r = close["BTC"].pct_change().rolling(20).mean()
cands["btc_lead_20d"] = pd.DataFrame({a: btc_r for a in ASSETS}, index=close.index)

# 12. WTI lead 20d (oil as macro risk proxy)
wti_r = close["WTI"].pct_change().rolling(20).mean()
cands["wti_lead_20d"] = pd.DataFrame({a: wti_r for a in ASSETS}, index=close.index)

# 13. Yield spread momentum effect (US10Y - CN10Y momentum on all assets)
us10_r = close["US10Y"].pct_change().rolling(20).mean()
cn10_r = close["CN10Y"].pct_change().rolling(20).mean()
spread_mom = us10_r - cn10_r
cands["yield_spread_mom_20d"] = pd.DataFrame({a: spread_mom for a in ASSETS}, index=close.index)

# 14. Risk-adjusted momentum (20d return / 20d vol), skip 5
mom20 = close / close.shift(25) - 1.0
vol20 = ret.rolling(20).std()
cands["risk_adj_mom_20x20_skip5"] = mom20 / vol20

# 15. Vol-of-vol 10x60 (stability of vol)
vol10 = ret.rolling(10).std()
vol60 = ret.rolling(60).std()
cands["vol_of_vol_10x60"] = vol10 / vol60

# 16. Max drawdown depth over 60d (tail risk)
def max_dd_60(close):
    roll_max = close.rolling(60, min_periods=30).max()
    return close / roll_max - 1.0
cands["tail_dd_60d"] = max_dd_60(close)

# 17. Equity-bond correlation regime (rolling corr of asset with US10Y, 60d)
us10 = close["US10Y"]
out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
for a in ret.columns:
    out[a] = ret[a].rolling(60, min_periods=30).corr(us10.pct_change())
cands["bond_beta_60d"] = -out  # negative bond correlation (equity-like) -> lower weight in risk-off?

# ---- evaluate ----
fwd = forward_ret(close, 10)
print(f"\n{'factor':26s} {'IC10':>8s} {'ICIR10':>8s} {'hit':>5s} {'n':>5s} | {'IC_r':>7s} {'ICIR_r':>7s} | {'covAD':>6s} {'covD8':>5s} {'turn':>6s} {'maxCorr':>7s}")
for name, f in cands.items():
    ic = daily_ic(f, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd)
    turn = rank_turnover(f, 10)
    mc, pairs = max_lib_corr(f, lib_panels)
    f_r = f.tail(500)
    ic_r = daily_ic(f_r, forward_ret(close, 10).reindex(f_r.index))
    st_r = ic_stats(ic_r, 10)
    flag = ""
    if abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840:
        flag = " <<< PASS"
    print(f"{name:26s} {st['ic']:8.4f} {st['icir']:8.3f} {st['hit']:5.2f} {st['n']:5d} | "
          f"{st_r['ic']:7.4f} {st_r['icir']:7.3f} | {cov['coverage_asset_days']:6.2f} {cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mc:7.3f}{flag}")
    if abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840:
        print("   corr pairs:", {k: v for k, v in pairs.items() if abs(v) > 0.3})
