"""
miner_2 cycle 2026-07-16: novel factor family screen (batch 5).

Motivation: the persisted library is dominated by 1-5d mean-reversion signals
(rev_1d/2d/3d/5d, rev_1d_vs, nclv_1d, nbody_1d, id_rev_1d, mom_10d_skip5) plus
two vol/macro factors. This screen tests structurally different, interpretable
families that are NOT simple variants of the existing signals:
  - downside volatility concentration (downside_vol_ratio_5_20)
  - return autocorrelation / trend persistence (autocorr_20d)
  - volatility squeeze / range contraction (vol_squeeze_5_20)
  - overnight gap reversal (overnight_rev_5d, ovn_rev_x_intra)
  - volume-confirmed trend (vol_trend_x_price_20)
  - cross-sectional dispersion z-score (cs_disp_z_5d)
  - candle wick pressure (wick_dn_1d, wick_up_1d)
  - 20d close location value (clv_20d)
  - risk-adjusted momentum (kelly_mom_60d)
  - liquidity-scaled reversal (rev_x_amihud_1d, rev_x_amihud_5d)
  - drawdown depth (dist_high_20d)

Admission gate (shared, 15-instrument universe):
    |daily paper IC|  >= 0.0070
    |daily paper ICIR| >= 0.0840
Validation: 2021-01-01 .. 2026-07-15, >=8 valid names per date.
"""
import pickle, time
import numpy as np
import pandas as pd

T0 = time.time()
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VALID_LO, VALID_HI = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-15")
IC_MIN, ICIR_MIN = 0.0070, 0.0840
MIN_NAMES = 8

cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]
open_ = cache["open"][SYMBOLS]
high = cache["high"][SYMBOLS]
low = cache["low"][SYMBOLS]
vol = cache["vol"][SYMBOLS]
ret = cache["ret"][SYMBOLS]

# restrict to validation window
mask = (close.index >= VALID_LO) & (close.index <= VALID_HI)
close, open_, high, low, vol, ret = (x.loc[mask] for x in (close, open_, high, low, vol, ret))
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} symbols  ({close.index[0].date()}..{close.index[-1].date()})")


def fwd_log(closes, h):
    return np.log(closes.shift(-h)) - np.log(closes)


def fast_ic(factor_df, fwd, min_names=MIN_NAMES):
    F = factor_df.values.astype(float); R = fwd.values.astype(float)
    n = np.isfinite(F) & np.isfinite(R)
    ok = n.sum(axis=1) >= min_names
    if not ok.any():
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    Fm = np.where(n, F, 0.0); Rm = np.where(n, R, 0.0)
    cnt = n.sum(axis=1)[ok]
    sx = Fm[ok].sum(axis=1); sy = Rm[ok].sum(axis=1)
    sxx = (Fm[ok] ** 2).sum(axis=1); syy = (Rm[ok] ** 2).sum(axis=1)
    sxy = (Fm[ok] * Rm[ok]).sum(axis=1)
    with np.errstate(all="ignore"):
        num = cnt * sxy - sx * sy
        den = np.sqrt((cnt * sxx - sx * sx) * (cnt * syy - sy * sy))
        ic = num / den
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    return {"n_dates": int(len(ic)), "n_obs": int(cnt.sum()),
            "ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hit": float((ic > 0).mean())}


def turnover10(factor_df, rebal=10):
    ranks = factor_df.rank(axis=1)
    chg = []
    for i in range(rebal, len(ranks)):
        prev = ranks.iloc[i - rebal].dropna(); cur = ranks.iloc[i].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 2:
            continue
        chg.append((cur[common] - prev[common]).abs().mean() / (len(common) - 1))
    return float(np.mean(chg)) if chg else np.nan


def coverage_panel(factor_df):
    return float(factor_df.notna().sum().sum()) / factor_df.size


# ---------------------------------------------------------------- factors
rng = (high - low).replace(0, np.nan)
body = (close - open_).abs()
w_up = (high - np.maximum(open_, close)) / rng          # upper wick (selling pressure)
w_dn = (np.minimum(open_, close) - low) / rng           # lower wick (buying pressure)

lret = np.log(close).diff()
rv5 = lret.rolling(5).std()
rv20 = lret.rolling(20).std()
rv60 = lret.rolling(60).std()

dn_ret = lret.where(lret < 0, 0.0)
dn_rv5 = dn_ret.rolling(5).std()
dn_rv20 = dn_ret.rolling(20).std()

amihud20 = (ret.abs() / vol).rolling(20).mean()

factors = {
    "downside_vol_ratio_5_20": dn_rv5 / rv20,
    "autocorr_20d": (np.sign(lret) * np.sign(lret.shift(1))).rolling(20).mean(),
    "vol_squeeze_5_20": ((high - low) / close).rolling(5).mean() / ((high - low) / close).rolling(20).mean(),
    "overnight_rev_5d": -(open_ / close.shift(1) - 1.0).rolling(5).sum(),
    "ovn_rev_x_intra_1d": -(open_ / close.shift(1) - 1.0) * ((close - open_) / open_).clip(-0.05, 0.05),
    "vol_trend_x_price_20": np.sign(lret.rolling(20).mean()) * (vol.rolling(5).mean() / vol.rolling(20).mean() - 1.0),
    "cs_disp_z_5d": lret.rolling(5).sum().sub(lret.rolling(5).sum().mean(axis=1), axis=0).div(lret.rolling(5).sum().std(axis=1), axis=0),
    "wick_dn_1d": w_dn,
    "wick_up_1d": -w_up,
    "clv_20d": -(close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min()),
    "kelly_mom_60d": (close / close.shift(60) - 1.0) / rv60,
    "rev_x_amihud_1d": -lret * (1.0 / (1.0 + amihud20)),
    "rev_x_amihud_5d": -lret.rolling(5).sum() * (1.0 / (1.0 + amihud20)),
    "dist_high_20d": -(close / high.rolling(20).max() - 1.0),
    "rev_5d_x_volsq": -lret.rolling(5).sum() * vol_squeeze_5_20,
}

fwd1, fwd5, fwd10 = fwd_log(close, 1), fwd_log(close, 5), fwd_log(close, 10)
n_cells = close.shape[0] * close.shape[1]

results = []
for name, panel in factors.items():
    panel = panel.replace([np.inf, -np.inf], np.nan)
    cov = coverage_panel(panel)
    to = turnover10(panel)
    ic1 = fast_ic(panel, fwd1)
    ic5 = fast_ic(panel, fwd5)
    ic10 = fast_ic(panel, fwd10)
    passed = (abs(ic1["ic"]) >= IC_MIN) and (abs(ic1["icir"]) >= ICIR_MIN)
    results.append({"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed})
    print(f"{name:26s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
          f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")

print("\n--- PASSED (gate |IC|>=%.4f & |ICIR|>=%.4f) ---" % (IC_MIN, ICIR_MIN))
for r in results:
    if r["passed"]:
        print(r["name"])
print(f"\nelapsed {time.time()-T0:.1f}s")
