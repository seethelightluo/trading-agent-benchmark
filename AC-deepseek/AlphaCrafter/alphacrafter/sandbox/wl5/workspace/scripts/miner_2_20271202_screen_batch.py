# -*- coding: utf-8 -*-
"""miner_2 2027-12-02: exploration batch of novel factor candidates.

Validated on the weekday-only common calendar panel of the 15 tradable assets,
visible through 2027-12-01 (previous completed trading day). Horizon h=10.
Gate: |IC| >= 0.0070, |ICIR| >= 0.0840.
Reports coverage, turnover, decay, regime split, and max-abs IC correlation
vs the persisted library signal set (library factors rebuilt).
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
import factor_validate as fv

VISIBLE = "2027-12-01"
H = 10

# ---------------- panel ----------------
close = fv.closes_panel(visible_through=VISIBLE)
close = close[close.index.dayofweek < 5].copy()
close = close.dropna(how="all", axis=0)
ret = close.pct_change()
macro = fv.macro_closes(visible_through=VISIBLE)
macro = macro[macro.index.dayofweek < 5].copy()
fwd = fv.forward_returns(close, H)

# full OHLCV panel for range/intraday/VWAP factors
bars = {}
for s in fv.WATCH:
    fp = f"../persistent/stock_data/{s}.csv"
    df = pd.read_csv(fp, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].copy()
    df = df[df["date"].dt.dayofweek < 5].set_index("date")
    bars[s] = df[["open", "high", "low", "close", "volume"]]
openp = pd.DataFrame({s: b["open"] for s, b in bars.items()}).reindex(close.index)
highp = pd.DataFrame({s: b["high"] for s, b in bars.items()}).reindex(close.index)
lowp = pd.DataFrame({s: b["low"] for s, b in bars.items()}).reindex(close.index)
volm = pd.DataFrame({s: b["volume"] for s, b in bars.items()}).reindex(close.index)

print(f"WEEKDAY panel: {close.shape[0]} dates x {close.shape[1]} assets, visible through {VISIBLE}")
print(f"  dates with >=8 valid: {(close.notna().sum(axis=1) >= 8).sum()}/{len(close)}")
print(f"Macro cols: {list(macro.columns)}")

# ---------------- library signal builders (for correlation provenance) ----------------
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
        beta = {}
        for a in close.columns:
            pair = pd.concat([ret[a].rename("a"), dxy_ret.rename("m")], axis=1).dropna()
            beta[a] = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
        return pd.DataFrame(beta).reindex(close.index)
    if fid == "WTI_BETA_60":
        wti_ret = close["WTI"].pct_change()
        beta = {}
        for a in close.columns:
            pair = pd.concat([ret[a].rename("a"), wti_ret.rename("m")], axis=1).dropna()
            beta[a] = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
        return pd.DataFrame(beta).reindex(close.index)
    if fid == "kurt_20":
        def kurt(x):
            m2 = (x ** 2).mean(); m4 = (x ** 4).mean()
            return m4 / (m2 ** 2) - 3.0 if m2 > 0 else np.nan
        return ret.rolling(20, min_periods=8).apply(kurt, raw=True)
    if fid == "vix_beta_cond_60x20":
        vix = macro["VIX"]; vix_ret = vix.pct_change()
        beta = {}
        for a in close.columns:
            pair = pd.concat([ret[a].rename("a"), vix_ret.rename("m")], axis=1).dropna()
            beta[a] = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
        bdf = pd.DataFrame(beta).reindex(close.index)
        return -bdf * (vix / vix.shift(20) - 1.0)
    return None

LIBRARY = ["trend_r2_30_signed", "semi_down_ratio_20", "mom_120d_skip5", "dxy_beta_60",
           "mom_10d_skip5", "vol_of_vol20x60", "time_under_water_120", "tail_ratio_20",
           "vix_beta_cond_60x20", "kurt_20", "WTI_BETA_60"]
lib_ics = {}
for fid in LIBRARY:
    sig = build_signal(fid, close, ret, macro)
    if sig is None:
        continue
    s = fv.ic_series(sig, fwd, min_valid=8)
    if len(s.dropna()) > 30:
        lib_ics[fid] = s

# ---------------- candidate builders ----------------
def rolling_beta(ret_a, ret_m, win=60, minp=30):
    pair = pd.concat([ret_a.rename("a"), ret_m.rename("m")], axis=1)
    cov = pair["a"].rolling(win, min_periods=minp).cov(pair["m"])
    var = pair["m"].rolling(win, min_periods=minp).var()
    return cov / var

def candidates():
    C = {}
    # 1. 60d beta vs USDJPY (global risk-on/off FX regime)
    jpy_ret = macro["USDJPY"].pct_change()
    C["usdjpy_beta_60"] = pd.DataFrame(
        {a: rolling_beta(ret[a], jpy_ret) for a in close.columns}).reindex(close.index)
    # 2. 60d beta vs EURUSD (dollar bloc / carry regime)
    eur_ret = macro["EURUSD"].pct_change()
    C["eurusd_beta_60"] = pd.DataFrame(
        {a: rolling_beta(ret[a], eur_ret) for a in close.columns}).reindex(close.index)
    # 3. 20d momentum conditioned on USDCNY 20d change (CN-sensitive regime)
    cny_chg = macro["USDCNY"].pct_change(20)
    mom20 = close / close.shift(20) - 1.0
    C["usdcny_cond_mom_20"] = np.sign(cny_chg).to_frame("s")["s"] * mom20
    # 4. 20d momentum conditioned on yield-curve slope change (US10Y-CN10Y)
    slope = close["US10Y"] - close["CN10Y"]
    slope_chg = slope.diff(20)
    C["yield_slope_cond_mom_20"] = np.sign(slope_chg).to_frame("s")["s"] * mom20
    # 5. 20d momentum conditioned on VIX percentile (250d rank) - fear regime
    vix_perc = macro["VIX"].rolling(250, min_periods=60).rank(pct=True)
    C["vix_cond_mom_20"] = (vix_perc - 0.5).to_frame("s")["s"] * mom20
    # 6. equity-relative alpha momentum: 60d asset ret minus 60d SPX ret
    spx_mom60 = close["SPX"] / close["SPX"].shift(60) - 1.0
    C["alpha_mom_60_spx"] = (close / close.shift(60) - 1.0).subtract(spx_mom60, axis=0)
    # 7. risk-adjusted momentum 20d: ret20 / std20
    C["risk_adj_mom_20x20"] = mom20 / ret.rolling(20, min_periods=10).std()
    # 8. Sharpe ratio 60d (quality-momentum)
    C["sharpe_60"] = ret.rolling(60, min_periods=30).mean() / ret.rolling(60, min_periods=30).std()
    # 9. vol clustering persistence: lag-1 autocorr of |ret| over 60d
    absr = ret.abs()
    C["vol_autocorr_60"] = ret.rolling(60, min_periods=30).apply(
        lambda x: pd.Series(x).autocorr(1) if len(x) >= 8 else np.nan, raw=False)
    # 10. VWAP deviation 20d (close vs rolling VWAP using volume)
    tp = (highp + lowp + close) / 3.0
    pv = (tp * volm).rolling(20, min_periods=10).sum()
    vv = volm.rolling(20, min_periods=10).sum()
    C["vwap_dev_20"] = close / (pv / vv) - 1.0
    # 11. crypto regime conditioner: 20d mom * sign(BTC/ETH ratio 20d change)
    btceth = close["BTC"] / close["ETH"]
    be_chg = np.sign(btceth.pct_change(20)).to_frame("s")["s"]
    C["btceth_cond_mom_20"] = be_chg * mom20
    # 12. downside beta vs SPX (60d, only SPX-down days)
    spx_ret = ret["SPX"]
    db = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), spx_ret.rename("m")], axis=1).dropna()
        mask = pair["m"] < 0
        sub = pair[mask]
        if len(sub) >= 15:
            db[a] = sub["a"].cov(sub["m"]) / sub["m"].var()
        else:
            db[a] = np.nan
    C["downside_beta_spx_60"] = pd.DataFrame(db).reindex(close.index)
    return C

cands = candidates()
print(f"\n{'factor':24s} {'IC':>8s} {'ICIR':>8s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'cov8':>5s} {'turn':>5s} {'rhoLib':>7s}  GATE")
results = {}
for fid, sig in cands.items():
    sig = sig.reindex(close.index)
    ic = fv.ic_series(sig, fwd, min_valid=8)
    m = fv.summary_metrics(ic, sig, fwd, close, h=H, step=10)
    if m is None:
        print(f"{fid:24s} NO METRICS (n={len(ic.dropna())})")
        continue
    m["regime_split"] = fv.regime_split(ic)
    rho = fv.max_abs_library_corr(ic, lib_ics)
    m["max_abs_library_correlation"] = rho
    gate_ic = abs(m["ic"]) >= 0.0070
    gate_icir = (m["icir"] is not None) and (abs(m["icir"]) >= 0.0840)
    gate = "PASS" if (gate_ic and gate_icir) else "fail"
    print(f"{fid:24s} {m['ic']:+8.4f} {str(m['icir']):>8s} {m['ic_hit_ratio']:5.3f} {m['n_ic_dates']:5d} "
          f"{m['coverage_asset_days']:5.2f} {m['coverage_dates_ge8']:5.2f} {str(m['turnover_10d_rank']):>5s} {rho:7.3f}  {gate}")
    reg = m["regime_split"]
    if reg:
        print("     regimes: " + "; ".join(f"{k}: IC={v['ic']:+.3f}/ICIR={v['icir']}" for k, v in reg.items()))
    results[fid] = m

with open("scripts/miner_2_20271202_screen_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_2_20271202_screen_results.json")
