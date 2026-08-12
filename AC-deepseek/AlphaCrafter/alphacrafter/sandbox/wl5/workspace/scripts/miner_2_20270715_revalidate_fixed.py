# -*- coding: utf-8 -*-
"""miner_2 2027-07-15: corrected re-validation of all library factors.

Fix for the 2027-07-01 rebuild artifact: the union-calendar panel contains
weekend rows (BTC/ETH trade 7d) which poison rolling windows for stock-like
assets (rolling(20).mean() needs 20 valid rows -> NaN for stocks on the union
calendar, coverage collapsed to ~0.19).

Correct approach: validate on the weekday-only common calendar where all 15
tradable assets have rows; rolling windows and shift(h) horizons are then
clean h-trading-day quantities for every asset. Gates: |IC| >= 0.0070,
|ICIR| >= 0.0840 at h=10 on the 15-asset universe.
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
import factor_validate as fv

VISIBLE = "2027-06-30"   # previous completed trading day before current date 2027-07-15
H = 10

# ---- weekday-only common calendar panel (artifact fix) ----
close = fv.closes_panel(visible_through=VISIBLE)
close = close[close.index.dayofweek < 5].copy()
close = close.dropna(how="all", axis=0)
ret = close.pct_change()
macro = fv.macro_closes(visible_through=VISIBLE)
macro = macro[macro.index.dayofweek < 5].copy()
fwd = fv.forward_returns(close, H)

print(f"WEEKDAY panel: {close.shape[0]} dates x {close.shape[1]} assets, visible through {VISIBLE}")
print(f"  per-date valid: min {close.notna().sum(axis=1).min()}, "
      f"dates with >=8 valid: {(close.notna().sum(axis=1) >= 8).sum()}/{len(close)}")
print(f"Macro cols: {list(macro.columns)}")

# ---- signal builders on the weekday panel ----
def build_signal(fid, close, ret, macro):
    if fid == "trend_r2_30_signed":
        logc = np.log(close)
        t = np.arange(len(close))
        tdf = pd.DataFrame(np.tile(t, (close.shape[1], 1)).T, index=close.index, columns=close.columns)
        cov = logc.rolling(30, min_periods=18).cov(tdf)
        vart = tdf.rolling(30, min_periods=18).var()
        varl = logc.rolling(30, min_periods=18).var()
        r2 = (cov ** 2) / (vart * varl)
        sign = np.sign(cov)
        return sign * r2
    if fid == "semi_down_ratio_20":
        down = ret.clip(upper=0.0)
        up = ret.clip(lower=0.0)
        sd = (down ** 2).rolling(20, min_periods=10).mean().apply(np.sqrt)
        su = (up ** 2).rolling(20, min_periods=10).mean().apply(np.sqrt)
        return sd / su - 1.0
    if fid == "mom_120d_skip5":
        return close.shift(5) / close.shift(125) - 1.0
    if fid == "mom_10d_skip5":
        return close.shift(5) / close.shift(15) - 1.0
    if fid == "vol_of_vol20x60":
        return ret.rolling(20, min_periods=10).std().rolling(60, min_periods=30).std()
    if fid == "time_under_water_120":
        rollmax = close.rolling(120, min_periods=30).max()
        underwater = close < rollmax
        out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for c in close.columns:
            cnt = 0
            vals = []
            for v in underwater[c].fillna(False):
                cnt = cnt + 1 if v else 0
                vals.append(cnt)
            out[c] = vals
        return out
    if fid == "tail_ratio_20":
        q95 = ret.rolling(20, min_periods=10).quantile(0.95)
        q05 = ret.rolling(20, min_periods=10).quantile(0.05)
        return q95 / q05.abs()
    if fid == "dxy_beta_60":
        dxy_ret = macro["DXY"].pct_change()
        beta = {}
        for a in close.columns:
            pair = pd.concat([ret[a].rename("a"), dxy_ret.rename("m")], axis=1).dropna()
            b = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
            beta[a] = b
        return pd.DataFrame(beta).reindex(close.index)
    if fid == "WTI_BETA_60":
        wti_ret = close["WTI"].pct_change()
        beta = {}
        for a in close.columns:
            pair = pd.concat([ret[a].rename("a"), wti_ret.rename("m")], axis=1).dropna()
            b = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
            beta[a] = b
        return pd.DataFrame(beta).reindex(close.index)
    if fid == "kurt_20":
        def kurt(x):
            m2 = (x ** 2).mean()
            m4 = (x ** 4).mean()
            return m4 / (m2 ** 2) - 3.0 if m2 > 0 else np.nan
        out = ret.rolling(20, min_periods=8).apply(kurt, raw=True)
        return out
    if fid == "vix_beta_cond_60x20":
        vix = macro["VIX"]
        vix_ret = vix.pct_change()
        beta = {}
        for a in close.columns:
            pair = pd.concat([ret[a].rename("a"), vix_ret.rename("m")], axis=1).dropna()
            b = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
            beta[a] = b
        bdf = pd.DataFrame(beta).reindex(close.index)
        return -bdf * (vix / vix.shift(20) - 1.0)
    return None

EFFECTIVE = ["trend_r2_30_signed", "semi_down_ratio_20", "mom_120d_skip5", "dxy_beta_60",
             "mom_10d_skip5", "vol_of_vol20x60", "time_under_water_120", "tail_ratio_20",
             "vix_beta_cond_60x20", "kurt_20", "WTI_BETA_60"]

results = {}
for fid in EFFECTIVE:
    sig = build_signal(fid, close, ret, macro)
    if sig is None:
        print(f"{fid}: BUILD FAILED")
        continue
    ic = fv.ic_series(sig, fwd, min_valid=8)
    m = fv.summary_metrics(ic, sig, fwd, close, h=H, step=10)
    if m is None:
        print(f"{fid}: no metrics (n<30, n={len(ic.dropna())})")
        continue
    results[fid] = m
    reg = fv.regime_split(ic)
    m["regime_split"] = reg
    gate_ic = abs(m["ic"]) >= 0.0070
    gate_icir = (m["icir"] is not None) and (abs(m["icir"]) >= 0.0840)
    print(f"{fid:22s} IC={m['ic']:+.4f} ICIR={m['icir'] if m['icir'] is None else round(m['icir'],4)} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} "
          f"cov_ge8={m['coverage_dates_ge8']:.2f} turn={m['turnover_10d_rank']} | GATE={'PASS' if (gate_ic and gate_icir) else 'FAIL'}")
    if reg:
        print(f"     regimes: " + "; ".join(f"{k}: IC={v['ic']:+.3f}/ICIR={v['icir']}" for k, v in reg.items()))

with open("scripts/miner_2_20270715_revalidate_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved scripts/miner_2_20270715_revalidate_results.json")
