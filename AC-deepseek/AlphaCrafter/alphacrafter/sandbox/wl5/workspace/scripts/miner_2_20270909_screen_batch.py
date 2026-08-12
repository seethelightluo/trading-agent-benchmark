# -*- coding: utf-8 -*-
"""miner_2 2027-09-09: exploration batch of novel factor candidates.

Validated on the weekday-only common calendar panel of the 15 tradable assets,
visible through 2027-09-08. Horizon h=10. Gate: |IC| >= 0.0070, |ICIR| >= 0.0840.
Reports coverage, turnover, decay, regime split, and max-abs IC correlation
vs the persisted library signal set (all 11 library factors rebuilt).
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
import factor_validate as fv

VISIBLE = "2027-09-08"
H = 10

# ---------------- panel ----------------
close = fv.closes_panel(visible_through=VISIBLE)
close = close[close.index.dayofweek < 5].copy()
close = close.dropna(how="all", axis=0)
ret = close.pct_change()
macro = fv.macro_closes(visible_through=VISIBLE)
macro = macro[macro.index.dayofweek < 5].copy()
fwd = fv.forward_returns(close, H)

# full OHLCV panel for range/intraday/overnight factors
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
def candidates():
    C = {}
    # 1. Kaufman efficiency ratio 20d: trend smoothness / noise
    C["eff_ratio_20"] = (close - close.shift(20)).abs() / ret.abs().rolling(20, min_periods=10).sum()
    # 2. lag-1 autocorrelation of returns over 20d (trend persistence vs mean reversion)
    C["autocorr_1_20"] = ret.rolling(20, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(1) if len(x) >= 6 else np.nan, raw=False)
    # 3. vol term-structure ratio 10/60
    C["vol_ratio_10_60"] = ret.rolling(10, min_periods=6).std() / ret.rolling(60, min_periods=30).std()
    # 4. intraday position: mean((close-low)/(high-low)) 20d
    rng = (highp - lowp).replace(0, np.nan)
    C["intraday_pos_20"] = ((close - lowp) / rng).rolling(20, min_periods=10).mean()
    # 5. up-day frequency 20d
    C["upday_ratio_20"] = (ret > 0).rolling(20, min_periods=10).mean()
    # 6. 60d beta vs SPX (equity-market beta, distinct from DXY/WTI/VIX betas)
    spx_ret = ret["SPX"]
    beta = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), spx_ret.rename("m")], axis=1).dropna()
        beta[a] = pair["a"].rolling(60, min_periods=30).cov(pair["m"]) / pair["m"].rolling(60, min_periods=30).var()
    C["spx_beta_60"] = pd.DataFrame(beta).reindex(close.index)
    # 7. crypto regime * asset 20d momentum (conditional trend)
    btc_mom = close["BTC"] / close["BTC"].shift(20) - 1.0
    C["crypto_regime_mom_20"] = np.sign(btc_mom).to_frame("s")["s"] * (close / close.shift(20) - 1.0)
    # 8. normalized daily range 20d
    C["range_norm_20"] = ((highp - lowp) / close).rolling(20, min_periods=10).mean()
    # 9. downside semi-dev / total std 20d (return asymmetry)
    C["downside_vol_ratio_20"] = (ret.clip(upper=0.0) ** 2).rolling(20, min_periods=10).mean().apply(np.sqrt) \
        / ret.rolling(20, min_periods=10).std()
    # 10. overnight return momentum 20d (open vs prev close)
    overn = openp / openp.shift(1) - 1.0
    C["overnight_mom_20"] = overn.rolling(20, min_periods=10).mean()
    # 11. intraday return momentum 20d (close vs open)
    intra = close / openp - 1.0
    C["intraday_mom_20"] = intra.rolling(20, min_periods=10).mean()
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

with open("scripts/miner_2_20270909_screen_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_2_20270909_screen_results.json")
