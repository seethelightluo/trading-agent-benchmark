"""miner_3 fast screen v5 (2026-07-30 cycle): vectorized rank-IC screening.

Motivation: previous v4 screen timed out because per-date scipy spearman was
too slow over ~35 candidates x 6 horizons. This version vectorizes the daily
cross-sectional Spearman IC (Pearson on cross-sectional ranks) with numpy
masked arrays, so a 35-candidate screen runs in well under a minute.

Data visible through 2026-07-29 (previous completed trading day before the
2026-07-30 runtime date). Cross-section = 15 tradable assets; >=8 valid
instruments per date required for an IC observation.

Admission gate (benchmark-wide, 15-instrument universe):
  |IC| >= 0.0070  and  |ICIR| >= 0.0840  at horizon 10.
"""
import sys, json, math, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_utils import load_panel, load_close, CURRENT_DATE
from factor_validation_lib import TRADABLE, MIN_INSTR, load_macro

VISIBLE = "2026-07-29"
H_ADMIT = 10
MIN_IC_DATES = 250

t0 = time.time()
px, vol = load_panel()
px = px[px.index <= pd.Timestamp(VISIBLE)]
vol = vol[vol.index <= pd.Timestamp(VISIBLE)]
ret = px.pct_change()
print(f"panel: {px.shape} dates={len(px)} range {px.index.min().date()}..{px.index.max().date()} "
      f"(load {time.time()-t0:.1f}s)", flush=True)

# OHLC from CSV
from pathlib import Path
SD = Path("../persistent/stock_data")
def _ohlc(sym):
    df = pd.read_csv(SD / f"{sym}.csv", parse_dates=["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE)].set_index("date").sort_index()
ohlc = {s: _ohlc(s) for s in TRADABLE}
hi = pd.DataFrame({s: ohlc[s]["high"] for s in TRADABLE}).reindex(px.index)
lo = pd.DataFrame({s: ohlc[s]["low"] for s in TRADABLE}).reindex(px.index)
op = pd.DataFrame({s: ohlc[s]["open"] for s in TRADABLE}).reindex(px.index)
vl = pd.DataFrame({s: ohlc[s]["volume"] if "volume" in ohlc[s] else np.nan for s in TRADABLE}).reindex(px.index)

rng = (hi - lo).replace(0, np.nan)
clv = (cl := px)  # alias
clv_loc = (px - lo) / rng
body = (px - op) / rng

def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)

def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()
def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()
def rsum(x, w):
    return x.rolling(w, min_periods=mp(w)).sum()

# ---------------- macro ----------------
vix = load_macro("VIX", max_date=VISIBLE)
vixr = vix.pct_change()
dxy = load_macro("DXY", max_date=VISIBLE)
dxy_r = dxy.pct_change()
us10y_r = px["US10Y"].pct_change()
cn10y_r = px["CN10Y"].pct_change()
spread = px["US10Y"] - px["CN10Y"]
spread_r = spread.pct_change()

ew = px.mean(axis=1)
ewr = ew.pct_change()

# ---------------- candidate families ----------------
C = {}

# --- momentum / trend ---
C["mom_20d_skip5"] = px.shift(5) / px.shift(25) - 1.0
C["mom_60d_skip5"] = px.shift(5) / px.shift(65) - 1.0
C["ma_slope_20d"] = px / rm(px, 20) - 1.0
C["ma_slope_60d"] = px / rm(px, 60) - 1.0
C["macd_hist_12x26"] = (rm(px, 12) - rm(px, 26)) - rm(rm(px, 12) - rm(px, 26), 9)
C["rel_strength_60d"] = px / rm(px, 60) - 1.0

# --- vol / risk ---
C["vol_z_20x120"] = (rs(ret, 20) - rs(ret, 20).rolling(120, min_periods=mp(120)).mean()) / \
    rs(ret, 20).rolling(120, min_periods=mp(120)).std().replace(0, np.nan)
park = np.sqrt(np.log(hi / lo).replace(0, np.nan) ** 2 / (4 * np.log(2)))
C["parkinson_vol_20d"] = rm(park, 20)
C["semi_down_20d"] = (-ret.clip(upper=0)).rolling(20, min_periods=mp(20)).std()
C["skew_20d"] = ret.rolling(20, min_periods=mp(20)).skew()
C["skew_60d"] = ret.rolling(60, min_periods=mp(60)).skew()
C["kurt_60d"] = ret.rolling(60, min_periods=mp(60)).kurt()
C["idio_vol_ratio_20d"] = rs(ret, 20) / rs(ewr, 20).iloc[:, 0] if False else \
    rs(ret, 20).div(rs(ewr, 20), axis=0)
C["vol_ratio_5x60"] = rs(ret, 5) / rs(ret, 60)
C["atr_ratio_20d"] = rm((hi - lo), 20) / rm(px, 20)

# --- candle / oscillator ---
C["clv_20d"] = clv_loc.rolling(20, min_periods=mp(20)).mean()
C["body_ratio_20d"] = body.rolling(20, min_periods=mp(20)).mean()
def _rsi(n):
    up = ret.clip(lower=0).rolling(n, min_periods=mp(n)).mean()
    dn = (-ret.clip(upper=0)).rolling(n, min_periods=mp(n)).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
C["rsi_14"] = _rsi(14)
hi_n = hi.rolling(14, min_periods=mp(14)).max()
lo_n = lo.rolling(14, min_periods=mp(14)).min()
C["stoch_k_14"] = ((px - lo_n) / (hi_n - lo_n).replace(0, np.nan)) * 100

# --- autocorr / persistence ---
def ac1(w):
    mu = ret.rolling(w, min_periods=mp(w)).mean()
    num = ((ret - mu) * (ret.shift(1) - mu.shift(1))).rolling(w, min_periods=mp(w)).sum()
    den = (ret ** 2).rolling(w, min_periods=mp(w)).sum()
    return num / den.replace(0, np.nan)
C["autocorr_1_20d"] = ac1(20)
C["autocorr_1_60d"] = ac1(60)

# --- drawdown / recovery ---
C["dd_120d"] = px / px.rolling(120, min_periods=mp(120)).max() - 1.0
C["dd_ratio_20x120"] = (px / px.rolling(20, min_periods=mp(20)).max() - 1.0) - \
    (px / px.rolling(120, min_periods=mp(120)).max() - 1.0)
C["days_since_hi_120d"] = (px.rolling(120, min_periods=mp(120)).apply(
    lambda s: (s == s.max()).argmax(), raw=True)) / 120.0

# --- volume / flow ---
upday = (ret > 0).astype(float)
up_vol = (vl * upday).rolling(20, min_periods=mp(20)).sum()
dn_vol = (vl * (1 - upday)).rolling(20, min_periods=mp(20)).sum()
C["vol_imbalance_20d"] = (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)
C["win_rate_20d"] = rsum(upday, 20) / 20.0
upm = ret.where(ret > 0, 0.0)
dnm = ret.where(ret < 0, 0.0)
C["updown_asym_20d"] = (rsum(upm, 20) / rsum(upday, 20)) / \
    ((-rsum(dnm, 20)) / rsum((ret < 0).astype(float), 20))

# --- overnight / intraday ---
gap = op / px.shift(1) - 1.0
intra = px / op - 1.0
C["overnight_ret_20d"] = gap.rolling(20, min_periods=mp(20)).mean()
C["overnight_share_20d"] = gap.rolling(20, min_periods=mp(20)).mean() / \
    (gap.abs() + intra.abs()).rolling(20, min_periods=mp(20)).mean().replace(0, np.nan)

# --- cross-asset linkage ---
def beta_of(a, m, w):
    return a.rolling(w, min_periods=mp(w, 2)).cov(m) / m.rolling(w, min_periods=mp(w, 2)).var()
C["beta_us10y_60d"] = beta_of(ret, us10y_r, 60)
C["beta_cn10y_60d"] = beta_of(ret, cn10y_r, 60)
C["corr_ew_20d"] = ret.rolling(20, min_periods=mp(20)).corr(ewr)
C["beta_ew_60d"] = beta_of(ret, ewr, 60)
C["beta_dxy_120d"] = beta_of(ret, dxy_r, 120)
C["corr_wti_60d"] = ret.rolling(60, min_periods=mp(60)).corr(ret["WTI"])
C["corr_xau_60d"] = ret.rolling(60, min_periods=mp(60)).corr(ret["XAU"])

# --- macro risk / regime ---
beta_vix = beta_of(ret, vixr, 60)
corr_vix = ret.rolling(60, min_periods=mp(60)).corr(vixr)
C["vix_beta_60d"] = beta_vix
C["vix_cond_60x20"] = -beta_vix * (vix / vix.shift(20) - 1.0)
C["vix_corr_60d"] = corr_vix

# --- risk-adjusted reversal / acceleration ---
C["zrev_5d_20"] = -(px / px.shift(5) - 1.0) / rs(ret, 5)
C["mom_accel_20x60"] = px.pct_change(20) - px.pct_change(60)
C["rel_mom_20d"] = px.pct_change(20) - px.pct_change(20).mean(axis=1)
C["rel_mom_60d"] = px.pct_change(60) - px.pct_change(60).mean(axis=1)

print(f"candidates: {len(C)} (built in {time.time()-t0:.1f}s)", flush=True)


# ---------------- vectorized rank IC ----------------
def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    dates = factor.index.intersection(fwd.index)
    fr = factor.loc[dates].rank(axis=1, pct=True)
    rr = fwd.loc[dates].rank(axis=1, pct=True)
    mask = fr.isna().values | rr.isna().values
    nvalid = (~mask).sum(axis=1)
    F = np.ma.array(fr.values, mask=mask)
    R = np.ma.array(rr.values, mask=mask)
    Fm = F - F.mean(axis=1, keepdims=True)
    Rm = R - R.mean(axis=1, keepdims=True)
    num = (Fm * Rm).sum(axis=1)
    den = np.sqrt((Fm ** 2).sum(axis=1) * (Rm ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic = np.ma.filled(ic, np.nan)
    ic[nvalid < min_valid] = np.nan
    return pd.Series(ic, index=dates)


def ic_summary(ic):
    ic = ic.dropna()
    m = float(ic.mean())
    s = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = m / s if s and math.isfinite(s) and s > 0 else 0.0
    hit = float((ic > 0).mean()) if len(ic) else np.nan
    return m, icir, hit, int(len(ic))


fwd = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
FR10 = fwd[H_ADMIT]

print(f"\n{'factor':<24}{'ic':>8}{'icir':>8}{'hit':>7}{'n':>6}  gate", flush=True)
results = {}
for name, f in C.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, FR10)
    m, icir, hit, n = ic_summary(ic)
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "signal": f}
    ok = abs(m) >= 0.0070 and abs(icir) >= 0.0840 and n >= MIN_IC_DATES
    print(f"{name:<24}{m:>8.4f}{icir:>8.4f}{hit:>7.3f}{n:>6d}  {'PASS' if ok else ''}", flush=True)

print(f"\nscreen done in {time.time()-t0:.1f}s", flush=True)
