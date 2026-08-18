# -*- coding: utf-8 -*-
"""miner_2 2028-03-23: re-validation of library factors.

Visible window ends 2028-03-22 (previous completed trading day; today is
2028-03-23). Rebuilds each persisted library signal from raw price/macro data,
computes 10d cross-sectional rank IC/ICIR on the 15-asset weekday panel, and
checks the admission gate (|IC| >= 0.0070, |ICIR| >= 0.0840). Reports recent
250d drift for timeliness.
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
import factor_validate as fv

VISIBLE = "2028-03-22"
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
    if fid == "rsi_14":
        delta = ret
        up = delta.clip(lower=0.0).rolling(14, min_periods=8).mean()
        dn = (-delta.clip(upper=0.0)).rolling(14, min_periods=8).mean()
        rs = up / dn
        return 100.0 - 100.0 / (1.0 + rs)
    if fid == "ret_vol_ratio_20":
        mu = ret.rolling(20, min_periods=10).mean()
        sd = ret.rolling(20, min_periods=10).std()
        return mu / sd
    if fid == "skew_20_raw":
        def skew(x):
            m2 = (x ** 2).mean(); m3 = (x ** 3).mean()
            sd = np.sqrt(m2)
            return m3 / (sd ** 3) if sd > 0 else np.nan
        return ret.rolling(20, min_periods=8).apply(skew, raw=True)
    if fid == "mom_vol_scaled_20x10":
        mom = close.shift(5) / close.shift(15) - 1.0
        vol = ret.rolling(10, min_periods=5).std()
        return mom / vol
    if fid == "skew_vol_comp_20":
        def skew(x):
            m2 = (x ** 2).mean(); m3 = (x ** 3).mean()
            sd = np.sqrt(m2)
            return m3 / (sd ** 3) if sd > 0 else np.nan
        sk = ret.rolling(20, min_periods=8).apply(skew, raw=True)
        vol = ret.rolling(20, min_periods=10).std()
        return sk * vol
    if fid == "dd_depth_20":
        rollmax = close.rolling(20, min_periods=10).max()
        return close / rollmax - 1.0
    if fid == "dd_pos_60":
        rollmax = close.rolling(60, min_periods=30).max()
        return (close / rollmax - 1.0) * (rollmax > close.rolling(5).mean())
    if fid == "breadth_cond_mom_20":
        mom = close.shift(5) / close.shift(15) - 1.0
        up_count = (ret > 0).rolling(20, min_periods=10).sum()
        return mom * up_count
    if fid == "ETH_BETA_60":
        eth_ret = close["ETH"].pct_change()
        return pd.DataFrame({a: rolling_beta(ret[a], eth_ret) for a in close.columns}).reindex(close.index)
    if fid == "btc_beta_60":
        btc_ret = close["BTC"].pct_change()
        return pd.DataFrame({a: rolling_beta(ret[a], btc_ret) for a in close.columns}).reindex(close.index)
    if fid == "MOM_REL_EQ_20":
        eq = close[["SPX", "NDX", "SOX", "HSI", "N225", "SX5E", "000300.SH", "000688.SH"]].mean(axis=1)
        return close.pct_change(20) - eq.pct_change(20)
    return None


LIBRARY = ["trend_r2_30_signed", "semi_down_ratio_20", "mom_120d_skip5", "dxy_beta_60",
           "mom_10d_skip5", "vol_of_vol20x60", "time_under_water_120", "tail_ratio_20",
           "vix_beta_cond_60x20", "kurt_20", "WTI_BETA_60", "rsi_14", "ret_vol_ratio_20",
           "skew_20_raw", "mom_vol_scaled_20x10", "skew_vol_comp_20", "dd_depth_20",
           "dd_pos_60", "breadth_cond_mom_20", "ETH_BETA_60", "btc_beta_60", "MOM_REL_EQ_20"]

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

with open("scripts/miner_2_20280323_revalidate_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_2_20280323_revalidate_results.json")
