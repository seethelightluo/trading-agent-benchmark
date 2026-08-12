# -*- coding: utf-8 -*-
"""miner_2 2027-12-16: re-validation of currently effective library factors.

Visible window ends 2027-12-15 (previous completed trading day). Rebuilds each
persisted library signal from raw price/macro data, computes 10d cross-sectional
rank IC/ICIR on the 15-asset weekday panel, and checks the admission gate
(|IC| >= 0.0070, |ICIR| >= 0.0840). Also reports recent-window drift (last 250
trading days) to flag timeliness.
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
import factor_validate as fv

VISIBLE = "2027-12-15"
H = 10

close = fv.closes_panel(visible_through=VISIBLE)
close = close[close.index.dayofweek < 5].copy()
close = close.dropna(how="all", axis=0)
ret = close.pct_change()
macro = fv.macro_closes(visible_through=VISIBLE)
macro = macro[macro.index.dayofweek < 5].copy()
fwd = fv.forward_returns(close, H)

print(f"WEEKDAY panel: {close.shape[0]} dates x {close.shape[1]} assets, visible through {VISIBLE}")
print(f"  dates with >=8 valid: {(close.notna().sum(axis=1) >= 8).sum()}/{len(close)}")


def rolling_beta(ret_a, ret_m, win=60, minp=30):
    pair = pd.concat([ret_a.rename("a"), ret_m.rename("m")], axis=1)
    cov = pair["a"].rolling(win, min_periods=minp).cov(pair["m"])
    var = pair["m"].rolling(win, min_periods=minp).var()
    return cov / var


def build_signal(fid, close, ret, macro):
    if fid == "trend_r2_30_signed":
        logc = np.log(close)
        t = np.arange(len(close))
        tdf = pd.DataFrame(np.tile(t, (close.shape[1], 1)).T, index=close.index, columns=close.columns)
        cov = logc.rolling(30, min_periods=18).cov(tdf)
        vart = tdf.rolling(30, min_periods=18).var()
        varl = logc.rolling(30, min_periods=18).var()
        r2 = (cov ** 2) / (vart * varl)
        return np.sign(cov) * r2
    if fid == "semi_down_ratio_20":
        down = ret.clip(upper=0.0); up = ret.clip(lower=0.0)
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
            cnt = 0; vals = []
            for v in underwater[c].fillna(False):
                cnt = cnt + 1 if v else 0
                vals.append(cnt)
            out[c] = vals
        return out
    if fid == "tail_ratio_20":
        return (ret.rolling(20, min_periods=10).quantile(0.95)
                / ret.rolling(20, min_periods=10).quantile(0.05).abs())
    if fid == "dxy_beta_60":
        dxy_ret = macro["DXY"].pct_change()
        return pd.DataFrame({a: rolling_beta(ret[a], dxy_ret) for a in close.columns}).reindex(close.index)
    if fid == "WTI_BETA_60":
        wti_ret = close["WTI"].pct_change()
        return pd.DataFrame({a: rolling_beta(ret[a], wti_ret) for a in close.columns}).reindex(close.index)
    if fid == "kurt_20":
        def kurt(x):
            m2 = (x ** 2).mean(); m4 = (x ** 4).mean()
            return m4 / (m2 ** 2) - 3.0 if m2 > 0 else np.nan
        return ret.rolling(20, min_periods=8).apply(kurt, raw=True)
    if fid == "vix_beta_cond_60x20":
        vix = macro["VIX"]; vix_ret = vix.pct_change()
        bdf = pd.DataFrame({a: rolling_beta(ret[a], vix_ret) for a in close.columns}).reindex(close.index)
        return -bdf * (vix / vix.shift(20) - 1.0)
    return None


LIBRARY = ["trend_r2_30_signed", "semi_down_ratio_20", "mom_120d_skip5", "dxy_beta_60",
           "mom_10d_skip5", "vol_of_vol20x60", "time_under_water_120", "tail_ratio_20",
           "vix_beta_cond_60x20", "kurt_20", "WTI_BETA_60"]

print(f"\n{'factor':24s} {'IC':>8s} {'ICIR':>8s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>5s}  | {'IC_250d':>8s} {'ICIR_250d':>9s} {'n250':>5s}  GATE")
results = {}
for fid in LIBRARY:
    sig = build_signal(fid, close, ret, macro)
    if sig is None:
        print(f"{fid:24s} NOT BUILT")
        continue
    sig = sig.reindex(close.index)
    ic = fv.ic_series(sig, fwd, min_valid=8)
    m = fv.summary_metrics(ic, sig, fwd, close, h=H, step=10)
    if m is None:
        print(f"{fid:24s} NO METRICS (n={len(ic.dropna())})")
        continue
    gate_ic = abs(m["ic"]) >= 0.0070
    gate_icir = (m["icir"] is not None) and (abs(m["icir"]) >= 0.0840)
    gate = "PASS" if (gate_ic and gate_icir) else "FAIL"
    # recent 250d window drift
    ic250 = ic.dropna().tail(250)
    ic250_mean = float(ic250.mean()) if len(ic250) >= 30 else None
    ic250_std = float(ic250.std(ddof=1)) if len(ic250) > 1 else None
    ic250_icir = (ic250_mean / ic250_std) if (ic250_mean is not None and ic250_std and ic250_std > 0) else None
    print(f"{fid:24s} {m['ic']:+8.4f} {str(m['icir']):>8s} {m['ic_hit_ratio']:5.3f} {m['n_ic_dates']:5d} "
          f"{m['coverage_asset_days']:5.2f} {str(m['turnover_10d_rank']):>5s}  | "
          f"{str(round(ic250_mean,4) if ic250_mean is not None else None):>8s} {str(round(ic250_icir,3) if ic250_icir is not None else None):>9s} {len(ic250):5d}  {gate}")
    results[fid] = {"ic": m["ic"], "icir": m["icir"], "ic_hit_ratio": m["ic_hit_ratio"],
                    "n_ic_dates": m["n_ic_dates"], "coverage_asset_days": m["coverage_asset_days"],
                    "turnover_10d_rank": m["turnover_10d_rank"], "gate": gate,
                    "ic_250d": ic250_mean, "icir_250d": ic250_icir, "n_250d": int(len(ic250)),
                    "decay": m["decay_ic_by_horizon"]}

with open("scripts/miner_2_20271216_revalidate_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_2_20271216_revalidate_results.json")
